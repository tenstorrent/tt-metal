# r03-b03-a04 result: 1.5554 (ok). Regression: -3.4% vs the best node r03-b02-a03 (1.6103), whose code this node builds on

## What happened vs expected
- All 9 shapes are valid. PCC, max_abs and mean_abs are bit-identical to the baseline on every printed
  digit. So the reuse-B path checks out: the 3x2 / 4x2 blocks, the dst layout, the absolute partials
  slots, the TILE_SIZE_B * KT_DIM in0 stride, the re-inits around the owner tilize and the last block,
  and the out-of-line helpers. The parent's SiLU-GLU accuracy failure is reverted.
- No JIT error, no hang, no kernel-config-buffer overflow on the first and only run. The local compile
  showed net -728 B of compute text vs the best node.
- I expected -5..-15% on the compute-bound shapes. They got slower instead:

| shape | r03-b02-a03 µs | this µs | Δ | mean-core end (MATH zones) |
|---|---|---|---|---|
| mistral | 185.9 | 204.2 | **+9.8%** | 166 -> 176 |
| minimax-m3 | 365.5 | 391.5 | **+7.1%** | 344 -> 359 |
| gpt-oss | 261.8 | 276.1 | **+5.5%** | 238 -> 246 |
| dsv3 | 514.0 | 532.3 | +3.6% | 493 -> 511 |
| glm | 451.2 | 461.6 | +2.3% | 429 -> 438 |
| kimi-k2.7 | 729.5 | 747.0 | +2.4% | 719 -> 740 |
| dsv4-flash | 317.0 | 323.4 | +2.0% | 298 -> 301 |
| kimi-k3 | 1365.8 | 1365.9 | 0 | 1273 -> 1259 |
| dsv4-pro | 1070.9 | 1064.7 | -0.6% | 1059 -> 1054 |

The mean core moved too, not just the tail lottery. The loss is real on the per_core_M >= 3 shapes.

## Why (profiler evidence: the same RE-HW counters as r03-b02-a03, /tmp/r03b02a03/hw.py)
Per-window counters, best node -> this node:

| shape / phase | cyc/mult | SrcA wr cyc/mult | SrcB wr cyc/mult | UNP0 busy | UNP1 busy | FPU util |
|---|---|---|---|---|---|---|
| mistral GU | 31.0 -> 34.1 | 17.5 -> **3.6** | 3.2 -> 7.6 | 57% -> **28%** | 27% -> 48% | 52% -> 47% |
| mistral DN | 29.1 -> 29.4 | 17.5 -> 3.0 | 2.4 -> 8.4 | 62% -> 30% | 25% -> 61% | 56% -> 55% |
| mm3 GU | 30.3 -> 31.5 | 17.7 -> 3.3 | 8.9 -> 7.8 | 59% -> 29% | 56% -> 53% | 53% -> 51% |
| mm3 DN | 30.2 -> 32.0 | 17.8 -> 3.1 | 2.6 -> 8.3 | 60% -> 27% | 26% -> 56% | 53% -> 50% |
| dsv3 GU | 32.6 -> 34.1 | 17.5 -> 8.7 | 2.9 -> 5.8 | 54% -> 37% | 24% -> 38% | 49% -> 47% |
| gpt-oss GU | 41.5 -> 43.2 | 17.9 -> 6.3 | 9.0 -> 10.0 | 44% -> 28% | 41% -> 37% | 39% -> 37% |

1. **The mechanism did exactly what it was built to do.** The bf4 weight tile now stays in SrcA:
   - SrcA write cycles per mult fell 17.5 -> 3-4 on the all-pcM-3 shapes (mistral, mm3).
   - Unpacker-0 busy halved.
   - Total unpack write work per mult fell from ~20.7 to ~11 cycles (SrcA + SrcB).
2. **cyc/mult did not fall. It rose 1-3 cycles.** FPU busy is still exactly 16.1 per mult, and
   MATH_SRC_DATA_READY still tracks FPU busy. So the FPU still idles ~15-18 cycles per mult, even with
   half the unpack work and both unpackers mostly idle (UNP0 28%, UNP1 48-61%).
3. **This refutes r03-b02-a03's reading.** "tile-mult = FPU 16 + serialized SrcA unpack 17.5" was a
   coincidence of two numbers. The ~15-cycle non-FPU gap per mult is not unpack bandwidth, and it is
   not specific to SrcA or to bf4. Whichever operand streams, the cost per mult stays ~31.
4. **Other things that look ruled out:**
   - Per-call overhead: gpt-oss DN at 1x3 (3 mults per call) was already 30.9, the same as mistral
     DN at 1x6. Here 3x2 / 4x2 put 6-8 mults on each call and nothing improved.
   - FW throttle: still 0 everywhere.
5. **Where the extra +1-3 cycles/mult and the op-level loss come from.** Likely sources:
   - the out-of-line `mm_accumulate` with runtime ct/rt (no unrolled LLK loops, both reuse branches,
     and the BH dynamic-throttle check per call);
   - reuse_b's two unpack contexts per call (one per ct column);
   - the last block also going through the generic helper.

   These are small per call, but every pcM >= 3 window pays them. kimi-k2.7 / dsv4-flash (+2%) mostly
   run pcM <= 2, which is the old shape now going through the out-of-line helper, so the helper itself
   probably costs ~1-2% there. kimi-k3 / dsv4-pro are DRAM-bound and flat.

## Classification
flawed idea (premise refuted by measurement). The implementation is correct and bit-identical, but
moving the weight tile to the held operand does not shorten a tile-mult on BH here. Don't keep it:
start children from r03-b02-a03 (best).

## What a child of this node should try next
1. **Do not retry operand-reuse / subblock-shape changes** (this node, r01-b02-a04). Unpack volume
   per mult is not the limiter: halving it changed nothing.
2. **Find the real ~15 idle FPU cycles per mult before more compute work.** Candidates, cheapest
   first:
   - (a) **MATH-thread issue rate.** Per tile, `_llk_math_matmul_` does `set_dst_write_addr`
     (several TT ops) plus a MOP run, and every matmul_block call reads the throttle word and
     branches. The MATH thread is barely stalled (ST1 1-6%), i.e. it is busy issuing. Probe: count
     MATH RISC cycles per tile with a cycle counter around one matmul_block call (TRISC_1 TS_DATA
     records work, see re_hwprof.hpp). Or compare against a variant that calls the LLK math matmul
     directly, without the dynamic-throttle wrapper.
   - (b) **Src bank handshake latency** (dvalid -> FPU start) per tile. Read sel 27/28 (waiting for
     SrcA/SrcB clear), which r03-b02-a03 did not read.
   - (c) **L1 read-port contention** of the unpackers against NoC writes landing at ~20-40 GB/s per
     core. Probe it on a shape with no DM in flight.
3. If MATH issue is the limit, a longer MOP per call is the lever. Bigger ct in reuse_a (fewer
   set_dst_write_addr / MOP runs per FPU cycle) is only possible up to the dst half: 1x8 on N=8
   shapes. Only a measured probe should pick this.
4. The out-of-line helper pattern (one `mm_accumulate` / `mm_init` copy) saves ~0.7 KB of compute
   text net. Use it only if a future node needs kernel-config-buffer headroom, and expect ~1-2% cost
   on compute-bound shapes from losing the inlined constant-shape LLK.
5. Tooling: /tmp/w_r03b03a04/sz (kc.sh, kc2.sh with CTA=..., base / new / gpt / tile JIT dirs) does
   local 3-TRISC compiles plus size checks for SiLU-dsv3, gpt-oss SwiGLU-OAI+bias and TILE-x
   variants.
