# Porting rules (production TT/TTI → HAL)

Source of truth: `tt_llk_blackhole/llk_lib/**` and the helpers they call in `tt_llk_blackhole/common/inc/`.
HAL: `tt_llk_blackhole/llk_lib/hal/**`; usage examples in `tools/tests/unit/lit/hal/blackhole/codegen/**`.
ISA (hardware behaviour, latencies, auto-waits, MOP/replay): Tenstorrent `tt-isa-documentation`, BlackholeA0
pages; `tt_llk_blackhole/instructions/assembly.yaml` for encodings.

## What a port is

A function-by-function, instruction-by-instruction translation. For every function a test reaches (and every helper
it calls):
- same name, template parameters, arguments and defaults; same behaviour on every branch (formats, broadcast types,
  transpose, partial faces, tile shapes, dest modes, fidelity, reuse);
- the same Tensix instruction sequence in the same order: same STALLWAIT targets/conditions, semaphore operations,
  mutexes, counter updates, MOP/replay programming and config registers written;
- the same access kind for every register access: an MMIO store stays an MMIO store, RMWCIB stays RMWCIB, SETC16 stays
  SETC16, WRCFG stays WRCFG, a Tensix SEMGET stays SEMGET, an MMIO semaphore post stays MMIO. Never replace a stall
  with a poll or a semaphore, never add or drop a stall, poll, NOP or mutex;
- every `LLK_ASSERT` kept.

Allowed improvements, and only these, during porting:
1. grouping writes to the same config word into one HAL write (same final contents, same access kind);
2. runtime → compile time: where production computes at runtime a value that is really a template/constexpr value,
   use the compile-time HAL form (immediate instruction) instead of the instruction-buffer form.

Keep RISC code at least as lean as production on per-tile paths (see the edge-case reference).
If the HAL has no API for an instruction or access, keep the production line with `// HAL-GAP: <what is missing>`.
Do not edit HAL headers, production headers or tests to make a port work; report the need instead.

## Mapping

**STALLWAIT** `TTI_STALLWAIT(targets, conditions)` → `hal::sync::wait::stall<StallTarget…, StallCondition…>()`.
Targets: STALL_TDMA=Tdma, STALL_SYNC=Sync, STALL_PACK=Pack, STALL_UNPACK=Unpack, STALL_XMOV=Mover, STALL_THCON=Thcon
(=Scalar), STALL_MATH=Math, STALL_CFG=Config, STALL_SFPU=Sfpu. Conditions: THCON(0x1)=ScalarIdle,
UNPACK0=Unpacker0Idle, UNPACK1=Unpacker1Idle, PACK=PackerIdle, MATH=MathIdle, SRCA_CLR=SrcACleared,
SRCB_CLR=SrcBCleared, SRCA_VLD=SrcAValid, SRCB_VLD=SrcBValid, XMOV=MoverIdle, TRISC_CFG(0x400)=RiscvAccessProcessed,
WAIT_SFPU=SfpuIdle, CFGEXU=ConfigUnitIdle.
**SEMWAIT** → `hal::sync::wait::semaphore<Targets, SemaphoreCondition::WhileZero|WhileMaximum, Semaphore::Sn>()`.
**Semaphores** `ckernel::semaphore::X` (index n, see `common/inc/ckernel_structs.h`) → `hal::sync::Semaphore::Sn`;
`t6_semaphore_post/get` → `semaphore::post<S>()/get<S>()`; `semaphore_post/semaphore_read` (MMIO) →
`post<Access::MMIO, S>()/read<S>()`; `t6_semaphore_init` → `semaphore::init<…>()`.
**Mutex** `t6_mutex_acquire/release(mutex::X)` → `hal::sync::mutex::acquire/release<Mutex::Mn>()` (REG_RMW=M0).
**CFG** (`hal/cfg.h`):
- `cfg[ADDR32] = v` → a whole-word MMIO store: compose the word from field descriptors
  (`(v << F.shamt(S)) & F.mask(S)`) and write it with `cfg::write<Access::MMIO, Anchor, S, Count>(std::array{…})`.
  `cfg::write<Access::MMIO>(set<…>…)` on a partially covered word becomes an MMIO *load* + store — never what
  production does.
- `cfg_reg_rmw_tensix<…>(v)` → `cfg::write<Access::TensixCfgUnit, Field, Sec>(v)` (RMWCIB per touched byte).
- `TTI_SETC16(addr, v)` → `cfg::write<Access::TensixCfgUnit>(set<thread fields>…)`, supplying every field of the word;
  for an already packed runtime word use `cfg::write_thread_word<Access::TensixCfgUnit, F, S>(word)`.
- `TTI_WRCFG(gpr, mode, addr)` → `cfg::write<Access::TensixCfgUnit, Anchor, S, GprTransferSize::Bits32|Bits128,
  WrcfgCompletion::Deferred>(hal::gpr<g>())`; Deferred emits no NOP, keep production's DMANOPs as
  `hal::nop::scalar()`. `TTI_RDCFG` → `cfg::read<Access::TensixCfgUnit, F, S>(hal::gpr<i>())`.
**GPR** `TT_SETDMAREG` → `hal::scalar::set_low/set_high/set`; other DMAREG ops in `hal/scalar.h`. RISC
`regfile[x] = v` / `sync_regfile_write`: HAL-GAP (keep production lines).
**Address counters** SETADCXY/ZW/XX, INCADC*, ADDRCR*, SETADC → `hal::counters::adc::set<AddrCounter{…}
.for_clients<Client…>()>()` (or the runtime overload). `p_setadc::UNP_A/UNP_B/PAC` → `Client::Unpacker0/Unpacker1/
Packers`. The planner chooses the encoding: check the instruction kind and count in the disassembly.
**RWC** SETRWC/INCRWC → `hal::counters::rwc::set<MathCounter{…}>()`; `flip` is SETRWC's clear-source-valid field.
**ADDR_MOD** `addr_mod_t{…}.set(n)` / `addr_mod_pack_t` → `cfg::write<Access::TensixCfgUnit>(set<cfg::AddrMod[…]…>…)`
with `Sec::Sn`, supplying all fields of each word `set()` writes.
**MOP** `ckernel_template(outer, inner, op[, op1])` + start/end ops + `set_last_inner_loop_instr` /
`set_last_outer_loop_instr` + `program()` → `hal::mop::program(hal::mop::loop(outer, [start,] loop(inner, op[, op1]),
[end[, end2]]).patch(last_outer, …).patch(last_inner, …))`; `m_loop0_last_instr` = `last_outer`, `m_loop1_last_instr`
= `last_inner`; `patch(last_inner, …)` needs outer > 1. `run()` → `hal::mop::Runner<MopTemplate::Template1>::run()`.
Template 0 → `foreach_mask_bit`/`if_clear`/`otherwise` and `Runner<MopTemplate::Template0>`.
**Replay** `load_replay_buf`, `lltt::replay*`, TTI_REPLAY → `hal/replay.h`.
**Instructions** UNPACR/UNPACR_NOP → `hal/unpack.h`; PACR → `hal::pack::DataTransfer`; ELWADD/ELWSUB/ELWMUL →
`hal::fpu::Elementwise` (other FPU ops in `hal/fpu.h`); ZEROACC → `hal::dst::Zero`; MOVD2A/MOVD2B/MOVA2D/MOVB2D →
`hal::move::Transfer<Src, Dst>`; ZEROSRC/CLEARDVALID/SETDVALID → `hal/src.h`; NOP/DMANOP/SFPNOP →
`hal::nop::thread/scalar/vector`; atomics `hal/atomic.h`.

## Verifying

- Codegen: `scripts/compare_codegen.sh` (see SKILL.md). Note that instructions with runtime operands appear as stores
  to the instruction buffer (`sw … __instrn_buffer`) in both builds.
- Functional: every test that includes the header passes with `--impl hal`, same counts as `--impl tt`, through the
  chip queue.
- Builds land in a private tree per queue job (`RUNNER_TEMP`), so parallel agents never delete each other's ELFs.
