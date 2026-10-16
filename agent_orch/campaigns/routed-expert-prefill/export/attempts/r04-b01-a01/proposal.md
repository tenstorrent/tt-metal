# r04-b01-a01: op-local matmul unpack MOP that advances the streamed SrcA tile address with one CFGSHIFTMASK per tile instead of the stock RDCFG / ADDDMAREG / STALLWAIT / WRCFG chain

## Motivation
- r03-b02-a03 (this node's parent) measured the compute-bound windows with hardware counters:
  - ~31 cycles per tile-mult on mistral / minimax-m3 / dsv3 / glm GU and on every DN window.
  - The FPU is busy only 16.1 cycles of those.
  - MATH_SRC_DATA_READY tracks FPU busy, so the FPU waits on source data.
  - Unpacker 0 (the streamed bf4 weight tile into SrcA) is busy only ~17.5 cycles per mult.
- So the resource the FPU waits on, the SrcA unpacker, is itself idle ~13 cycles per tile. Something in the
  UNPACK thread's instruction stream keeps it from being fed back to back.
- r03-b03-a04 and r03-b02-a04 swapped the streamed operand to SrcB (bf8 x). They halved the unpack work per
  mult, and cyc/mult stayed at ~31. So the gap is per streamed tile, whichever Src it goes to, and it is not
  unpack bandwidth.
- The only per-streamed-tile work common to both modes is the stock MOP body in `llk_unpack_AB_matmul.h`:
  `UNPACR -> RDCFG -> ADDDMAREG -> STALLWAIT(STALL_CFG, THCON) -> WRCFG -> NOP`. It reads the context's L1 base
  address into a GPR, adds the tile size, waits for THCON, and writes it back, once per tile.
- BH has a one-instruction replacement, and the stock LLK already uses it in hot loops:
  - `CFGSHIFTMASK(1, ADD, 31, 0, scratch, Base_address)` adds a scratch config register to the base address
    (2 cycles + NOP, no GPR, no THCON stall).
  - BH `llk_unpack_tilize.h` uses it in its per-row replay, with exactly this two-context layout.
  - `llk_pack_untilize.h` uses it ("replaces ADDDMAREG + STALLWAIT + WRCFG + NOP").
  - The DeepSeek decode `custom_mm` unpack streams SrcA with `UNPACR; CFGSHIFTMASK; NOP` per tile.
- r03-b02-a04 next #2 and r03-b03-a04 next #2 both pointed at the unpack MOP's per-tile chain as the next thing
  to test with an op-local copy.

## Mechanism
- Kernel-only. Files:
  - new `device/kernels/compute/re_mm_unpack.h`;
  - `device/kernels/compute/fused_swiglu.cpp`.
- `re_mm_unpack.h` (UNPACK thread only) adds `re_mm::shiftmask_mop(in1_cb, ct_dim, rt_dim)`:
  - When the call streams SrcA (`ct_dim >= rt_dim`, i.e. every matmul in this op, rt = 1) and in1 is not
    partial-face, it reloads the matmul unpack replay buffer with, per config context:
    `UNPACR(SrcA, dvalid); CFGSHIFTMASK(Base_address[ctx] += SCRATCH_SEC0); NOP`.
    It reprograms the same two-context unpack template the stock init uses, so the stock
    `_llk_unpack_AB_matmul_` execute (per-call address setup, SrcB unpack, `TT_MOP(.., ct-1, ctx mask)`) runs
    unchanged.
  - It preloads SCRATCH_SEC0 with in1's tile size (CB page size, unpacker 16 B units). It uses the in-order
    Tensix sequence the tilize init uses (`SETDMAREG x2; STALLWAIT(CFG, THCON); WRCFG; NOP`), so it cannot
    overtake the owner tilize's CFGSHIFTMASKs, which read the same scratch register.
  - Otherwise (reuse-B or partial faces) it keeps the stock MOP.
- `fused_swiglu.cpp`: every `matmul_block_init` goes through `re_mm_block_init`, which is the stock init plus
  `UNPACK(re_mm::shiftmask_mop(...))`. The 5 sites are:
  - the chunk start (gate/up);
  - after the owner tilize;
  - leave_act in the last-block pipeline;
  - the narrowed-LW last block;
  - the down phase.

  Every switch back to matmul already passes through one of these, so the replay buffer and the scratch are
  always rewritten after the tilize / copy / bcast inits that change them.
- Numerics: the same tiles are unpacked from the same L1 addresses in the same order, so the result is
  bit-identical. Math, packer and formats are untouched.

## Why this is not a repeat
- r01-b02-a04 (subblock width), r03-b01-a04 / r03-b02-a04 / r03-b03-a04 (operand swap / reuse mode):
  - All of them changed WHICH operand is streamed or HOW MANY tiles go per call.
  - All of them kept the stock per-tile address chain, and all of them measured cyc/mult unchanged.
  - This node changes nothing about the subblock geometry or operands. It only replaces the per-tile address
    increment instruction sequence in the unpack MOP.
- No earlier node touched LLK instruction sequences. They were all schedule, DM, planner, SFPU or LLK-API-level
  changes.

## Expected effect and risk
- If the chain is the gap: per streamed tile ~31 -> ~20-24 cycles on the compute-bound windows.
  - mistral / minimax-m3 / gpt-oss GU and DN, and dsv3/glm phase A, -10..-25% in those windows.
  - The op time on gpt-oss / minimax-m3 / mistral should move -5..-15%.
  - The DRAM-bound shapes (kimi-k2.7, dsv4-pro, kimi-k3) are flat, maybe -1..-3% from their compute-bound
    sub-phases.
  - The parent's RE-HW counters will show it directly: cyc/mult, UNP0 busy %, FPU util.
- If the chain is hidden behind the unpack: neutral, and cyc/mult unchanged. That also settles the question.
- Risks:
  - Wrong scratch value: wrong L1 addresses, garbage or a PCC fail.
  - A replay-buffer or template mismatch: a hang (the eval tool handles timeouts).
  - The scratch race with the owner tilize: avoided by the in-order WRCFG.
  - Kernel size: the UNPACK binary grows by ~100-200 B per init site (5 sites), well inside the ~12 KB
    headroom left on h7168.
