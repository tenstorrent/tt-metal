# r03-b02-a03: measurement node: profiler-only Tensix hardware perf counters (FPU, unpacker, SrcA/SrcB-valid stalls, thread stalls) and FW matmul-throttle samples on the MATH thread, split into gate/up and down windows

## Motivation
Every lever aimed at the compute-bound shapes has come back null, so the per-tile-mult cost there is
still unexplained. Those shapes are mistral, minimax-m3, gpt-oss, and dsv3/glm's phase A.

My reading of the parent's (r03-b02-a02) profile (/tmp/r03b02a03/*.py on reports/r03-b02-a02):
- **mistral, minimax-m3 and gpt-oss sit at their compute floor, but only if a tile-mult costs ~30 cycles.**
  - mistral's GU time scales with per_core_M: pcM 2 rows take 92 µs, pcM 3 rows 110 µs. Per pcM unit
    per block that is ~2.3 µs ≈ 3.1k cycles for 192 tile-mults.
  - With ~30 cycles/mult the whole pcM-3 expert models to ~154 µs; the mean core ends at 167.
  - minimax-m3: GU ~227 µs for 12 blocks of 864 mults is ~30 cycles/mult.
  - gpt-oss: GU pcM 3 → ~41 cycles/mult, pcM 4 → ~36 cycles/mult.
  - The LoFi ideal is 16 cycles/tile (tech_reports/GEMM_FLOPS; BH big GEMMs reach 96% of it).
- **Ruled out already for this ~2x:**
  - subblock shape (r01-b02-a04, 1x3 → 1x5+1x4: < 1%);
  - deeper rings (r01-b04-a03);
  - read-ahead (r01-b03-a01, r01-b04-a01);
  - x-owner placement (r03-b02-a02);
  - x read timing (r03-b04-a02);
  - per-core tilize (r01-b01-a02: worth ~5% on gpt-oss, no more).
- **The DRAM-bound shapes are at ~400-430 GB/s** in their full phases (bw.py timeline: kimi-k3 steady
  400-433, early 360-385). kimi-k3, kimi-k2.7, dsv4-pro and dsv3 are within ~3-8% of a 430 GB/s floor.
  So the remaining compute-side prize is the per-mult cost on the 4-5 compute-bound shapes.
- **Candidate causes, not yet distinguished by any node:**
  1. **FW dynamic matmul throttling.**
     - On Blackhole, `matmul_block` reads an ARC-FW-written word in every core's L1
       (MEM_L1_ARC_FW_SCRATCH). When it is odd, the matmul runs the MM_THROTTLE_MAX = 5 MOP
       (`api/compute/matmul.h`), which `llk_math_matmul.h` caps at 33% of peak.
     - Partial throttling would give exactly 30-40 cycles/mult. It would also explain why no
       data-movement or subblock change has moved these shapes.
  2. **MATH starved by unpack.** The bf4 weight tile is streamed into SrcA once per tile-mult.
     High unpacker busy plus SrcA-valid waits would show this.
  3. **MATH starved by data delivery.** UNPACK would be waiting on CBs: low unpacker busy, MATH
     waiting on SrcA/SrcB valid.
  4. **Non-FPU MATH-thread overhead per call.** Thread stalls with neither of the above.

## Mechanism
Profiler-only instrumentation. Nothing is compiled in without PROFILE_KERNEL, and numerics are
untouched.
- New op-local header `kernels/compute/re_hwprof.hpp`, MATH thread only (`TRISC_MATH && PROFILE_KERNEL`).
  It programs three Tensix perf-counter banks directly, the same registers `tools/profiler/perf_counters.hpp`
  uses. Those are idle here: the eval does not set TT_METAL_PROFILE_PERF_COUNTERS.
  - FPU: FPU busy, SFPU busy.
  - TDMA_UNPACK: unpacker 0/1 busy for thread 0, SrcA/SrcB write requests, math-src-data-ready.
  - INSTRN_THREAD: per-thread stall cycles for T0/T1/T2, waiting-for-SrcA-valid, waiting-for-SrcB-valid.
- In `fused_swiglu.cpp`:
  - Each expert's RE-GU and RE-DN window starts and stops the counters. The values are accumulated
    per phase.
  - The FW throttle word is sampled once per K-block in both K-loops. A sample counts as throttled
    when the word is odd.
  - The tile-mults issued are counted per phase.
  - At kernel end MATH emits ~16 DeviceTimestampedData records (`RE-HW-*`), two 32-bit values each.
- Cost: ~50 MMIO accesses per window on MATH, ~0.5 µs per expert window. That is expected to be
  inside noise. The extra binary size is small (TRISC1 only).

## Why this is not a repeat
- No node has read hardware counters or the FW throttle flag. Earlier measurement nodes used:
  - wall-clock zones and counters on the DM RISCs (r03-b03-a02, r03-b04-a02);
  - MATH phase zones (r02-b01-a02).
- Those tell how long MATH spent, not why it was slow.
- This node answers the question in the parent's next #2 and r03-b04-a01's next #3 directly.
  That question is what costs ~2x per tile-mult on the compute-bound shapes.

## Expected effect and risk
- **Speed:** neutral, inside noise, since the code paths are unchanged. A slowdown above ~0.5% would
  mean the MMIO reads cost more than estimated.
- **Accuracy:** bit-identical.
- **Risks:**
  - JIT compile error (repairable before the device run).
  - Profiler buffer overflow if too many markers: 16 more records ≈ 32 markers per TRISC1, within
    the 250-marker budget.
  - Reading perf-counter registers from TRISC1 is what the stock profiler already does.
- **How to read the eval:** per shape and phase:
  - FPU util = FPU / ref;
  - unpacker busy;
  - SrcA/SrcB-valid wait share;
  - throttled-sample share;
  - ref cycles per tile-mult.
