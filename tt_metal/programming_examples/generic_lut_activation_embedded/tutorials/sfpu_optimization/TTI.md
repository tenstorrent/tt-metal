# TTI Dispatch vs sfpi — what it is and why it's faster

This note explains the two layers you can write SFPU kernels in — the **sfpi**
C++ DSL and **raw TTI** instruction macros — why the native special-function
ops (exp, recip, ...) ship a `_tti_` variant alongside the sfpi one, and what
the ~0.3–0.5 µs gap between them actually buys. All citations are `file:line`
against the Blackhole LLK tree unless noted.

---

## 1. The two layers: sfpi DSL vs raw TTI

**sfpi** is a C++ DSL for the SFPU. You write with vector types `vFloat`,
`vInt`, `vUInt` (32-lane SIMD, `runtime/sfpi/include/sfpi.h:14-34`) and normal
operator overloading; the GCC RISC-V backend lowers each operation to a single
`__builtin_rvtt_sfp*` builtin, which maps 1:1 to one Tensix SFPU instruction
(`runtime/sfpi/include/sfpi_classes.h:89-157`). Arithmetic operators
(`operator+`, `operator*`) each expand to exactly one `flt_add`/`flt_mul` TTI
builtin with **no fusion and no cross-statement optimization**
(`runtime/sfpi/include/sfpi_funcs.h:482-495`). Constants are reloaded on every
use via `sfploadi`/`sfpxloadi` — no constant pooling
(`runtime/sfpi/include/sfpi_funcs.h:457-463`). Reading `dst_reg[i]` emits an
implicit `sfpload`, writing emits an implicit `sfpstore`
(`runtime/sfpi/include/sfpi_classes.h:352-399`,
`sfpi_funcs.h:142-156,194-208`). Predication (`v_if`/`v_else`/`v_endif`)
expands to `sfpxvif`/`sfpcompc`/`sfppushc`/`sfppopc`
(`runtime/sfpi/include/sfpi.h:56-71`); because predicates can extend a
variable's lifetime, assignment is forced through `sfpassign_lv`
(`runtime/sfpi/include/sfpi_classes.h:95-98`).

**TTI** is the layer underneath. A `TTI_*` macro (e.g. `TTI_SFPMAD`,
`TTI_SFPLOADI`, `TTI_SFPSTORE`) emits **one raw Tensix instruction directly into
the instruction stream** at compile time. It wraps a `TT_OP_*` encoding
expression (opcode in bits 31:24, operand fields packed into bits 23:0 by
shifts) inside `INSTRUCTION_WORD`, which is
`__asm__ __volatile__(".ttinsn %0" :: "n"((x)))`
(`tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel_ops.h:11-12`,
encodings at `:709-717` for SFPMAD). Contrast the three forms:
- `TT_OP_*` — computes the 32-bit encoding only.
- `TT_*` — writes that encoding to `instrn_buffer[0]` for deferred dispatch.
- `TTI_*` — injects the instruction inline with zero indirection (immediate
  dispatch).

So: **sfpi is what you write; TTI is what runs.** The sfpi compiler emits TTI
for you. The cost is that it emits TTI *naively* — per-statement, per-use,
per-iteration — with none of the scheduling, caching, or addressing tricks a
hand-written TTI body uses.

---

## 2. How a tile is processed, and the dispatch floor

An SFPU op does not see "a tile." Every elementwise unary op is invoked through
the parametric wrapper `_llk_math_eltwise_unary_sfpu_params_`
(`tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_math_eltwise_unary_sfpu_params.h:13-20`),
which runs three phases per call:

1. **start** (`llk_math_eltwise_sfpu_common.h:17-21`): `TT_SETC16` to set the
   dest write address, then `TTI_STALLWAIT(STALL_SFPU, MATH)` — one config write
   + one sync barrier.
2. **apply_vector_mode** (`llk_math_eltwise_sfpu_common.h:54-94`): in RC mode,
   loops **4 times, one per face**, calling the SFPU function per face and
   advancing the dest address between faces via
   `_llk_math_eltwise_sfpu_inc_dst_face_addr_` = **2× `TTI_SETRWC(CR_D, 8)` per
   face** (`:36-39`).
3. **done** (`llk_math_eltwise_sfpu_common.h:23-25`): one final `TTI_SETRWC` to
   clear the dest offset.

Inside each face, the SFPU function itself loops `ITERATIONS=8` over the 32
lanes (a tile face is processed in 8 passes of 32 lanes; see `calculate_abs`,
`ckernel_sfpu_abs.h:15-23`, and `calculate_identity`,
`ckernel_sfpu_identity.h:19-26`). That is the **32 SFPU passes** per tile (4
faces × 8 iters).

Fixed per-call overhead, independent of op complexity: **1 SETC16 + 1 STALLWAIT
+ 8 SETRWC (2×4 faces) + 1 SETRWC cleanup ≈ 11 TTI instructions of pure
scaffolding.** For a trivial op (`abs`, `identity`) the scaffolding dwarfs the
compute. This is the source of the **~1.6 µs fixed dispatch floor**: it is paid
on every op call regardless of what the op does.

The `_tti_` path attacks this floor by pre-configuring address modifiers **once
at init** (`eltwise_unary_sfpu_configure_addrmod`,
`llk_math_eltwise_unary_sfpu.h:22-59`) and replaying canned microcode, so the
per-call SETRWC/recompute machinery falls away.

---

## 3. The three TTI mechanisms that cut dispatch

### (a) Replay buffer — record once, replay N−1 times

The Tensix replay buffer is a hardware instruction cache. A sequence is recorded
on the first pass and re-executed without re-fetch/re-decode. The primitive is
`TTI_REPLAY(start, len, execute, record)`
(`runtime/sfpi/include/lltt.h:15-32`): `execute=1, record=1` records the body
the first time; subsequent `execute=0, record=0` calls replay the cached body.
The C++ wrappers are `load_replay_buf(slot, len, lambda)` to record and
`lltt::replay(start, len)` (a `__builtin_rvtt_ttreplay`) to replay
(`lltt.h:31-32`). This collapses an `ITERATIONS × BODY_LEN` unroll down to
`BODY_LEN + (ITERATIONS−1)` lightweight replay words, cutting instruction
fetch/decode by ~(N−1)× while keeping single-iteration throughput. sfpi has no
analog — it inline-unrolls the loop body, so binary size and decode cost grow
linearly with `ITERATIONS`.

### (b) ADDR_MOD dest auto-increment — addressing in hardware, not in the loop

`ADDR_MOD` registers hold per-mode src/dest increment values that `SFPLOAD`/
`SFPSTORE` apply automatically to the hardware Dest row counter. The `addr_mod_t`
struct defines 8 modes (indices 0–7) with `srca`/`srcb`/`dest` increment fields
(`ckernel_addrmod.h:30-57`); `addr_mod_t::set(idx)` writes them to hardware via
`TTI_SETC16` (`ckernel_addrmod.h:138-145`). `TTI_SFPSTORE` encodes a 3-bit
addr_mod selector in bits 15:13 (`TT_OP_SFPSTORE`, opcode 0x72). With
`dest.incr=2` preloaded, each store auto-advances the Dest row counter — no
per-iteration address math in the loop body.

sfpi does the same stepping, but **explicitly in C++**: `dst_reg++` compiles to
`__builtin_rvtt_ttincrwc(0, SFP_DESTREG_STRIDE*ix, 0, 0)`
(`sfpi_classes.h:389-398`), one increment instruction per pass. So sfpi pays an
extra `ttincrwc` per element where TTI pays nothing — the increment is folded
into the store's addr_mod field.

### (c) LREG constant preload — load once, not per-use

Hand-written TTI loads a constant into an LREG once and reuses it across the
whole body. sfpi cannot: every constant use re-emits `sfploadi` (16-bit imm) or
`sfpxloadi` (32-bit) with no constant pooling or CSE
(`sfpi_funcs.h:457-463,519-526`), and a `FIXME` confirms the constant-load is
not yet fused into comparison instructions
(`sfpi_funcs.h:505-511`). In a tight per-element body these redundant reloads
are pure dispatch waste; a TTI body slots one `SFPLOADI` into a latency window
(see §4) and keeps the constant resident in an LREG for the rest of the loop.

---

## 4. Case study: exp — sfpi vs TTI, instruction-level

Both variants live in `ckernel_sfpu_exp.h`:
`_sfpu_exp_21f_bf16_` (sfpi, `:100-152`) vs `_sfpu_exp_21f_bf16_tti_`
(TTI, `:164-297`). The TTI variant is ~0.3–0.5 µs faster per call. Three
distinct sources:

**1. Replay-buffer compression.** The TTI body is **15 fixed instructions**
(`:195-198`): `SFPLOAD, SFPMAD, SFPLOADI, SFPSWAP, SFPEXEXP, SFPEXMAN8, SFPSHFT,
SFPEXMAN9, SFPCAST, SFPMAD(poly1), SFPGT, SFPMAD(poly2), SFPAND, SFPSETEXP,
SFPSTORE` (+ conditional scale/fp32 ops). `BODY_LEN` is computed at `:198`,
recorded once via `TTI_REPLAY(0, BODY_LEN, 1, 1)` at `:214`, then replayed
`ITERATIONS−1` times in the loop at `:291-296`. The sfpi variant inline-unrolls
the exexp/exman/shft/convert/poly chain per iteration via `PolynomialEvaluator`,
which expands to sequential Horner `c0 + frac*(c1 + frac*c2)`
(`ckernel_sfpu_polyval.h:66-71`, sfpi eval at `:138`) — full reissue every pass.

**2. ADDR_MOD_6 auto-increment.** `exp_init` configures
`addr_mod_t{.dest={.incr=2}}.set(ADDR_MOD_6)` (`:1043-1062`); the `TTI_SFPSTORE`
at `:288` uses ADDR_MOD_6 to advance dest by 2 per store. The sfpi variant has
no addr_mod and instead does `sfpi::dst_reg++` per element (`:502`) — one extra
`ttincrwc` each pass.

**3. Latency-window interleaving.** The TTI body is hand-scheduled to hide
multi-cycle latencies: `SFPLOADI(255)` at `:228` fills the 2-cycle post-write
window of the `SFPMAD` at `:225`; the `SFPGT` mask at `:259` hides in the
latency of the `SFPMAD` at `:251`; the polynomial MADs at `:251,261` chain with
`SFPGT`+`SFPAND` (`:259,268`) sandwiched between them. Negative-input handling
is restructured too: sfpi uses two sequential `vec_min_max` clamps
(`:125-126`), while TTI replaces the lower clamp with an `SFPGT` mask
(`SFPGT_MOD1_SET_VD`) + `SFPAND` that fit in existing MAD windows
(`:259,268`). sfpi's `PolynomialEvaluator::eval` (`:138`) is strictly
sequential — no interleaving.

Net: the ~0.3–0.5 µs comes from (1) eliminating `ITERATIONS−1` body reissues,
(2) removing one `dst_reg++` per element, and (3) hiding 2–3 cycles of latency
per iteration via scheduling.

---

## 5. Implications for the generated-kernel (fitter/codegen) project

Our generated piecewise polynomial/rational kernels are written in **sfpi**.
That choice already gets us most of the way:

- **Iso-accuracy** with native special functions — the fit + Horner/parity
  evaluation matches reference within target ULP (this is independent of the
  dispatch layer).
- **Near-parity throughput** for compute-heavy bodies. When the per-element body
  is large (high-degree poly, rational with reciprocal), the fixed dispatch
  overhead is amortized and the sfpi/TTI gap shrinks.

**EMPIRICAL CORRECTION (disassembly, exp deg-2, `tools/disasm_exp_body_vs_native.sh`):**
the earlier "recoverable only at the TTI level" framing is WRONG. Disassembling our
*compiled* exp kernel shows it is **already saturated with `ttreplay`** (record+replay
throughout) — so the replay-buffer benefit (§3a) is **already obtained**, via `lltt`
+ the LLK math wrapper, from a plain sfpi kernel. The measured residual to native
(2.30µs vs 1.80µs) is **NOT replay and NOT ADDR_MOD** — it is **constant
rematerialization**: our recorded body contains **9 `sfploadi`** (re-loading
1/ln2, +127, the 3 poly coeffs, clamp bounds *inside* the body), whereas native
preloads those into LREGs / `vConstFloatPrgm` once at init. The polynomial math is
identical (both 2× `sfpmad`, degree-2). So the gap is fixable **entirely in sfpi**:
program the constants into `vConstFloatPrgm0/1/2` + persistent LREGs in an `_init`
and reference them in the body (native's *own* sfpi variant does exactly this —
`vConstFloatPrgm1 = c2`). That removes ~8 instrs/pass and closes most of the 0.5µs.
A full raw-`TTI_*` rewrite is therefore NOT required to tie native exp — the
remaining sliver (latency-window scheduling, §4.3) is the only TTI-only part.

**What emitting TTI from the fitter would take:**
- Generate the per-segment evaluator body as a flat sequence of `TTI_*` macros
  (SFPLOAD / SFPMAD chain / SFPSTORE) with explicit LREG allocation for
  coefficients (preload once) and for `x`/`x²`.
- Wrap the body in `load_replay_buf` / `lltt::replay` so the 32 passes share one
  recorded sequence (§3a; `lltt.h:15-32`).
- Add an `..._init` that programs an `ADDR_MOD` slot with `dest.incr` and an
  `eltwise_unary_sfpu_configure_addrmod` call, and emit `TTI_SFPSTORE` against
  that slot (§3b; `llk_math_eltwise_unary_sfpu.h:22-59`,
  `ckernel_addrmod.h:138-145`).
- Optionally hand-schedule constant loads into MAD latency windows (§4) — the
  highest-effort, lowest-portability part.

**Risk / effort.** High. TTI bodies are architecture-specific (Quasar /
Wormhole_b0 / Blackhole carry separate `ckernel_ops.h` with the same encodings
but distinct files), so the codegen must branch per arch. Manual LREG allocation
and replay-buffer slot management remove the safety the sfpi compiler provides
(liveness via `sfpassign_lv`, register allocation). Per-segment bodies of
varying degree (our adaptive-degree path) and parity variants multiply the
number of TTI templates to maintain and validate. Latency-window scheduling is
brittle and must be re-tuned per arch. The accuracy is unchanged by going to
TTI — the entire payoff is the dispatch delta, which is bounded by the ~1.6 µs
floor minus what we already amortize.

---

## 6. Bottom line: when TTI is worth it

**Worth it** when the goal is to **tie a tuned native op** on a *short* body
where dispatch dominates — exactly the regime where the native ops ship a
`_tti_` variant (`ckernel_sfpu_exp.h`, `ckernel_sfpu_recip.h`). If a generated
kernel must compete head-to-head with `exp`/`recip` at small body sizes, the
~0.3–0.5 µs is only reachable through replay + ADDR_MOD + LREG preload at the
TTI level, and the engineering cost is justified.

**Not worth it** for the **special-function tail** the fitter targets, where
sfpi already wins: the per-element body is large enough that the fixed dispatch
floor is amortized, sfpi delivers iso-accuracy and near-parity throughput, and
the residual TTI delta does not pay for the per-arch, per-degree maintenance and
brittleness of hand-emitted TTI. Stay in sfpi; reserve TTI for the specific
short-body ops where we are explicitly trying to match a hand-tuned native
kernel.
