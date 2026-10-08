# r01-b01-a04: accumulate sum(x^2) in fp32 DST (ELWMUL dest-MAC) under one acquire, so PRE packs once per row instead of an L1-acc fp32 pack per tile

## Motivation
Parent r01-b01-a03 (1.1868) profile, h7168, µs from kernel start (its reflection): R_INPUT ends 5.0-6.2 but W_PUSH
(PRE + stick) ends 7.1-9.9, so PRE trails the input by ~2-3.7 µs at ~125 ns/tile, and the AG starts on the SLOWEST
worker's stick. r01-b02-a02, r01-b02-a03 and r01-b04-a03 measured the same ~1.3-2.4 µs PRE tail after the deep read.
Every one of those reflections lists "cheaper PRE: accumulate x^2 in DST, pack once" as a next step. No node has tried it.

Current PRE (resident path) per 4-tile block: acquire, 4x mul_tiles(x,x) into DST 0..3, commit, then 4 fp32
pack_tile<true> with packer L1 accumulation into one pre_intermediate tile (a 4 KB L1 read-modify-write per input
tile), release. The per-tile fp32 L1-acc pack is the likely limiter (POST eltwise passes without L1 acc run ~90-100 ns/tile).

## Mechanism
Compute kernel only (`kernels/compute/dit_rmsnorm_fused_compute.cpp`), resident (non-streaming) PRE branch:
On BH, ELWMUL is a dest MAC (dst += srcA*srcB). Its dest_accum_en field is 0 and the HiFi fidelity phases already rely
on that (see tt-llk experimental/llk_math_eltwise_binary_custom.h, which documents the MAC). Dest is zeroed on release.
So one tile_regs_acquire per row (per group), and mul_tiles(input, input, k, k, /*dst*/0) for every tile k, with the same
cumulative per-block cb_input waits. DST slot 0 then holds sum_k x_k^2 in fp32. Then one commit/wait,
one pack_tile(0, pre_intermediate_cb) (no L1 acc) and release. The reduce<SUM,REDUCE_ROW>, transpose and stick push are unchanged.
No host change (kernel JIT only), no CB change.

## Why this is not a repeat
Every previous node changed dataflow (trid reads, gamma placement, bank rotation), POST order (x*gamma pre-pass) or
decomposition (column split). None touched PRE math/pack. The kernel's own PERF NOTE says no cheaper PRE path exists
"with current LLKs". It overlooked the ELWMUL dest MAC.

## Expected effect and risk
PRE goes from ~125 ns/tile toward the unpack/math floor (~50-70 ns/tile), so W_PUSH ends ~0.3-0.8 µs after the input
instead of 2-3.7 µs. The AG start moves earlier by ~1.5-3 µs on wide shapes and ~1 µs on narrow ones. Part of that is
absorbed because x*gamma (~5 µs at h7168) then spills past the shorter AG wait. Net expectation: -1 to -1.5 µs per
shape, score ~1.24-1.28.
Risks: accuracy, if the FPU MAC accumulate into fp32 dest is lower precision than the packer's fp32 L1 add (check
pcc/max_abs vs 0.9999985 / 0.022-0.024). Wrong sums if dest is not zero at acquire (PCC fail). Judge the gain with
W_PUSH end minus R_INPUT end in the profile.
