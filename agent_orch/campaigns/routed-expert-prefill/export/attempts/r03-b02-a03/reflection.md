# r03-b02-a03 result: 1.6103 (ok). Measurement node: code functionally identical to the parent (1.5952). The +0.9% is inside the ±1% noise band, not a win.

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the parent and the baseline.
- The JIT compiled on the first run. There was no hang and no host build.
- The instrumentation cost nothing measurable: every shape is at or below the parent's time.
  - Largest moves: mistral -2.9%, kimi-k2.7 -0.5%, gpt-oss -1.6%, dsv3 -1.1%.
  - These are the usual run-to-run spread.
- All 16 `RE-HW` records arrived from every TRISC_1, in all 117 launches.
  - So TS_DATA from TRISC_1 records fine.
  - Programming the perf-counter banks from TRISC_1 works without the stock perf-counter build.

## Why: the measurement
Parser: /tmp/r03b02a03/hw.py on reports/r03-b02-a03/.logs/profile_log_device.csv. Values are summed over each core's per-expert windows, mean over the 10 measured calls.
- "cyc/mult" = window reference cycles / tile-mults issued.
- Percentages are of window cycles.

| shape | phase | cyc/mult | FPU cyc/mult | FPU util | UNP0 (SrcA) busy | UNP1 (SrcB) busy | SrcA wr cyc/mult | throttled K-blocks |
|---|---|---|---|---|---|---|---|---|
| mistral | GU / DN | 31.0 / 29.1 | 16.1 | 52 / 56% | 57 / 62% | 27 / 25% | 17.5 | 0 / 16720 |
| minimax-m3 | GU / DN | 30.3 / 30.2 | 16.1 | 53 / 53% | 59 / 60% | 56 / 26% | 17.7 | 0 |
| dsv3 | GU / DN | 32.6 / 30.9 | 16.1 | 49 / 52% | 54 / 61% | 24 / 18% | 17.5 | 0 |
| glm | GU / DN | 33.3 / 30.1 | 16.1 | 48 / 54% | 53 / 63% | 24 / 28% | 17.6 | 0 |
| gpt-oss | GU / DN | 41.5 / 30.9 | 16.1 | 39 / 52% | 44 / 60% | 41 / 55% | 17.9 | 0 |
| dsv4-flash | GU / DN | 39.7 / 46.4 | 16.1 | 41 / 35% | 45 / 39% | 21 / 15% | 17.6 | 0 |
| kimi-k2.7 | GU / DN | 45.4 / 44.3 | 16.1 | 35 / 36% | 39 / 42% | 17 / 12% | 17.5 | 0 |
| dsv4-pro | GU / DN | 47.5 / 49.3 | 16.1 | 34 / 33% | 38 / 37% | 35 / 10% | 17.7 | 0 |
| kimi-k3 | GU / DN | 44.6 / 60.9 | 16.1 | 36 / 26% | 41 / 29% | 37 / 13% | 17.8 | 0 |

1. **The FW matmul throttle never engaged.**
   - The ARC scratch word was even on every one of ~500k K-block samples, across all shapes and phases.
   - Hypothesis 1 is ruled out. matmul_block always runs the unthrottled MOP.
2. **The FPU runs at exactly the LoFi ideal when it runs**: 16.1 FPU-busy cycles per tile-mult on every shape and phase. The FPU itself is not slow.
3. **On the compute-bound shapes a tile-mult costs ≈ FPU (16) + SrcA unpack (17.5), serialized.**
   - Shapes: mistral, minimax-m3, dsv3/glm, and gpt-oss DN.
   - The SrcA unpacker (unpacker 0, the streamed bf4 weight tile) is busy 17.5-18 cycles per mult.
   - Total 29-33 cycles/mult ≈ 16 + 17.5. FPU busy and unpacker-0 busy are each ~50-60% of the window and add to ~105-115%, so they barely overlap.
   - MATH_SRC_DATA_READY tracks FPU busy almost exactly (49.1 vs 49.3% on dsv3 GU). Sources are ready only while the FPU computes, so the math is waiting for the next SrcA tile.
   - With rt = 1 subblocks (reuse_a, `ct >= rt`), the weight tile is unpacked into SrcA once per mult. The unpack of tile i+1 does not proceed while tile i is in the FPU.
   - That accounts for the whole "~2x per tile-mult" every earlier node saw. It also explains why subblock width (r01-b02-a04), ring depth, read-ahead and x placement never moved it: none of them changes which operand is streamed.
4. **The DRAM-bound shapes add data starvation on top**: 44-61 cycles/mult at the same 16 FPU / 17.5 unpack per mult.
   - Shapes: kimi-k2.7, dsv4-pro, kimi-k3, dsv4-flash.
   - The extra ~12-28 cycles/mult is the share-of-DRAM wait the zones and bw.py already showed (~400-430 GB/s aggregate).
   - kimi-k3 DN is the worst at 61 cycles/mult. That matches its slow DN per byte: 57 KB down blocks, so per-block handshakes are a larger share.
5. **Side observations:**
   - SFPU busy is 14% of kimi-k3 GU (SiTU-GLU on PACK), 12.6% gpt-oss GU, 10% mistral GU (SiLU), 3-6% elsewhere.
   - Thread stalls: UNPACK 12-20%, MATH 1-14% (kimi-k3 GU 14%, gpt-oss GU 10%: the PACK-SFPU hand-off), PACK 1-4%.
   - Selectors 29/30 (waiting-for-SrcA/B-valid) read 0 everywhere. On BH these sels do not seem to count this condition. Ignore them; use MATH_SRC_DATA_READY vs FPU instead.
   - SrcB writes per mult are 2.9 on the per_core_N = 6 shapes (one x tile per 1x6 call) and 8.6-9 on the N = 9 shapes (1x3 calls, plus ragged).

## Classification
Neutral (within noise): a measurement node. It finds the compute bottleneck and rules out two hypotheses, FW throttling and a slow FPU. The instrumentation is profiler-only, ~1.5 KB of TRISC1 code under the profiler, 0 B otherwise, and harmless to keep.

## What a child of this node should try next
1. **Stream x instead of the weights in the non-last gate/up K-blocks.** This is the direct lever on the measured bottleneck: up to ~40% off compute-bound GU (mistral, minimax-m3, gpt-oss, dsv3/glm phase A).
   - Call matmul_block with rt = per_core_M (2-4) and ct = 1-2 (`ct < rt`, so reuse_b).
   - The bf4 weight tile then stays in SrcA for rt mults, and x (bf8) is streamed into SrcB.
   - This is the mode BH big GEMMs use (rt 4 x ct 2), and they reach ~96% of 16 cycles/tile. So SrcB streaming evidently overlaps with the FPU.
   - Mechanics:
     - Re-init (matmul_block_init(x, gate, 0, ct, rt, in0_block_w)) per expert. Re-init again after the owner's tilize, and before the last block, whose PACK pipeline needs rt = 1.
     - Pack rt x ct tiles with dst index m*ct + j to partials slot m*per_core_N + n0 + j.
     - Keep the current path when per_core_M == 1.
   - Numerics stay bit-identical: same per-tile K order within a block, same L1_ACC across blocks.
   - Judge it with this node's counters: cyc/mult should fall toward ~17-20 and FPU util rise on the compute-bound shapes.
   - Apply the same to the down matmul_phase (rt = per_core_M, ct = 1-2) once GU shows a gain.
2. If (1) does not overlap either, look at the unpack MOP for the streamed operand. Check whether the SrcA bank is double-buffered in this config (sel 27/28, waiting for SrcA/SrcB clear, which this node did not read).
3. The DRAM-bound shapes are unchanged by (1). Their ceiling is aggregate DRAM (~430 GB/s), already established.
4. Don't retry: FW throttle theories (measured 0), subblock width with rt = 1, ring depth, read-ahead, VCs, x placement.
5. Tooling:
   - /tmp/r03b02a03/hw.py parses the RE-HW records.
   - /tmp/r03b02a03/sz/kc.sh compiles the compute kernel locally per TRISC with a dsv3-like CTA (pass -DPROFILE_KERNEL=1 -DPROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=65536). Use it to check code size before an eval.
