# r01-b02-a01: row-group expert parallelism (runtime G-row groups run disjoint experts concurrently)

## Motivation
Baseline runs every local expert on the whole 11x8 grid, one after another: the 8 M-rows split one
expert's tokens and the weights are multicast down each full column from one sender per column.
At the production prefill shapes each chip holds 4-28 experts with only 70-272 tokens each
(3-9 tile-rows, see the baseline run log `counts_min/max`), so:

* per_core_M is 1-2 and 3/8 to 5/8 of the M-rows compute phantom rows (DeepSeek-V3: 5 tiles on
  8 rows -> per_core_M 1; Kimi-K3: 3 tiles on 8 rows; gpt-oss: 9 tiles -> per_core_M 2 = 16 rows);
* only 11 cores (one per column) issue every weight DRAM read, and the whole grid waits on them
  per K-block. Measured weight throughput is a flat ~200-265 GB/s on every shape (baseline log
  `weight_gbps_per_chip`), about half of Blackhole DRAM bandwidth, consistent with the 11 issuing
  cores being the limit rather than DRAM.

## Mechanism
All three kernels already read every local expert's count at start. Use it to build one
deterministic schedule, identical on every RISC of every core:

* G = smallest power of two in {1,2,4,8} such that the largest local expert fits in ONE chunk of
  G * per_core_M_max tile-rows; grown further while there would be more groups than active
  experts. Huge counts give G = 8, which is exactly the baseline schedule (same senders,
  same rectangles, same chunking), so the large-M path is unchanged.
* The 8 M-rows split into 8/G groups of G rows; active experts are dealt round-robin to groups.
  Each group runs its experts independently: per_core_M = ceil(count_tiles / G), the weight
  multicast spans only the group's G rows (sender = group row gx % G, same stagger as baseline),
  G = 1 means every core reads its own weight slice with no multicast at all.
* When G < 8 the x (in0) DRAM read rotates across the row's 11 cores per K-block (one row now
  owns a whole expert's x, which would otherwise pile onto one core); G = 8 keeps the fixed
  baseline x sender.
* Host: pass each core its column's NoC coordinates (reader + writer) and my_mt (compute).

Files: `unified_routed_expert_ffn/device/kernels/adaptive_chunk.hpp` (schedule helper),
reader/writer/compute kernels, `unified_routed_expert_ffn_program_factory.cpp` (runtime args).

## Why this is not a repeat
First round, no prior nodes. The baseline's own history (comments in the factory) tuned sender
placement, read splitting between RISCs and L1 widths, all within the one-expert-at-a-time
schedule; this changes the schedule itself.

## Expected effect and risk
Fewer phantom M-rows (compute) and 4-8x more weight-reading cores (DRAM issue). Expect the
biggest gain on Kimi-K3 (28 experts x 3 tiles, G=1) and DeepSeek-V4-pro (12 x 3 tiles, G=1),
~1.2-1.5x on the G=2 shapes (DeepSeek-V3, Kimi-K2.7, GLM, Mistral, V4-flash, MiniMax), gpt-oss
G=4. Total DRAM bytes are unchanged (each expert still read once). Risks: semaphore/rectangle
mistakes -> hang; row-mapping mismatch between reader/compute/writer -> accuracy fail; load
imbalance across groups (expert counts not divisible by groups) caps the gain.
