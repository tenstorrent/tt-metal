# unified_routed_expert_moe: prefill optimization

This branch makes `ttnn.experimental.deepseek_prefill.unified_routed_expert_moe` **1.78× faster** (geomean over 9 production prefill shapes, one Blackhole chip). Accuracy is unchanged on every shape.

It contains only the op changes plus a small fix to the benchmark test. The search that produced the changes (an agent-driven optimization loop) is described at the end; its tooling isn't included.

## Results

Device kernel time of `UnifiedRoutedExpertFfnDeviceOperation`, mean of 10 measured calls after 3 warmup calls, from the tracy ops CSV. Measured on a 2× p300c Blackhole host (one chip used), firmware 19.7. "This branch" is the code on this branch, re-measured twice after the final reformatting commit.

| case | baseline µs | this branch µs | speedup | PCC (baseline = branch) |
|---|---|---|---|---|
| deepseek-v3 h7168 i2048 | 801.3 | 478.3 | 1.68× | 0.98031 |
| kimi-k2.7 h7168 i2048 | 1162.0 | 677.5 | 1.72× | 0.98029 |
| kimi-k3 latent h3584 i3072 | 2570.4 | 1306.6 | 1.97× | 0.98044 |
| glm-5.3 h6144 i2048 | 704.1 | 422.4 | 1.67× | 0.98029 |
| mistral-small-4 h4096 i2048 | 261.8 | 160.3 | 1.63× | 0.98026 |
| deepseek-v4-flash h4096 i2048 | 525.2 | 287.3 | 1.83× | 0.98049 |
| deepseek-v4-pro h7168 i3072 | 1604.1 | 958.4 | 1.67× | 0.98048 |
| minimax-m3 h6144 i3072 | 477.8 | 317.2 | 1.51× | 0.98244 |
| gpt-oss-120b h2880 i2880 | 572.0 | 228.8 | **2.50×** | 0.98350 |
| **geomean** | | | **1.78×** | |

- **Accuracy:** PCC, max abs error and mean abs error are identical to the baseline on all 9 shapes; for example mean_abs 0.273027 on deepseek-v3, the same as before. Nothing lowers precision: math fidelity, data formats, fp32 accumulation and approximation modes are all unchanged.
- **Noise:** run-to-run spread is about ±1%. Two runs of this branch gave geomean 1.7777 and 1.7777. The search recorded 1.7812 for the same code before formatting.

## What's on this branch

On top of `opgen_hackathon/agent_orch` (Danilo's routed-expert benchmark):

1. **Test fix** (`tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py`). On some hosts a single host→device write above about 512 MiB never completes. The test wrote each chip's whole dispatch buffer (0.5–1 GB) in one go, so it hung before the op ran. Large ROW_MAJOR inputs are now uploaded in ≤256 MiB row slices and joined on device; the tensor the op receives is byte-identical. In production the dispatch buffer is produced on device, so this only affects the test.
2. **Op changes**: 8 files in `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/`.
3. **This note** and `routed_expert_optimization_attempts.csv`: every attempt the search made, with per-shape times.

## How to reproduce

```bash
./build_metal.sh --release
python -m tracy -r -p -v -m pytest tests/ttnn/nightly/unit_tests/operations/fused/test_routed_expert_prefill.py
# per-shape device time: mean "DEVICE KERNEL DURATION [ns]" of UnifiedRoutedExpertFfnDeviceOperation,
# dropping the first 3 of each shape's 13 calls (rows are in test order: 9 shapes x 13 calls)
```

The test runs in about 45 s.

## What the changes do

The final code is the end of one lineage. Each row is one change, in the order it entered that lineage. Some ideas were first found on a sibling branch and ported later; the attempt id is where the change landed. Score is the geomean speedup measured for that attempt, and every id is in the CSV.

| attempt | score | change |
|---|---|---|
| r01-b01-a01 | 1.44 | **Row-group expert parallelism.** Split the 8 matmul M-rows into 1/2/4 groups that run different experts at the same time, instead of every core row working on one expert. This was the big structural win. |
| r01-b01-a02 | 1.47 | **Row-shared x tilize.** Each gate/up K-block of the row-major input is tilized once per M-row by a rotating owner core and multicast as bf8 tiles, instead of every core tilizing it. |
| r01-b04-a03 | 1.46 | Ported both of the above, plus **deeper input rings** (3 slots for x, gate, up, down and activated, when L1 has room). |
| r01-b04-a04 | 1.50 | **All 10 worker rows**: an 11×10 grid with row groups of 10/5/2, instead of 11×8 with 8/4/2. |
| r02-b03-a01 | 1.50 | **Two-phase expert schedule**: full waves of experts on the small row groups, then the leftover experts on larger groups. |
| r02-b03-a03 | 1.54 | **Cheaper SFPU activations**: SwiGLU-OAI, SiTU-GLU and clamped SiLU-GLU with the same math in fewer instructions (constants in LRegs, SFPSWAP clamps, unrolled bodies). |
| r02-b02-a04 | 1.57 | **Mixed-height layouts for the leftover phase** (4+3+3 etc., one leftover expert per group, largest first), so the last few experts don't leave rows idle. The idea came from r02-b02-a01's mixed-height row groups. |
| r03-b02-a01 | 1.59 | **PACK-thread pipelined activation** (from r02-b01-a04), plus a cheaper planner prologue (tile counts cached once per RISC, uniform layouts scheduled once). |
| r03-b02-a02 | 1.60 | **x owner off the weight-sender cores**: the row's non-sender columns read, tilize and multicast x, so the weight senders only fetch weights. |
| r03-b02-a03 | 1.61 | Profiler-only hardware perf counters (`re_hwprof.hpp`). Measurement only, used to find the compute bound; the score difference from its parent is within noise. |
| r04-b02-a01 | 1.63 | **Weight read-ahead**: weight senders start the next block's DRAM read as soon as their own ring slot is free, in both the gate/up and down phases. |
| r05-b02-a01 | 1.67 | **NoC rebalancing**: weight DRAM reads split about 60/40 toward NoC 0, so the hot NoC 1 response links in the DRAM columns carry about 40% instead of 50%. |
| r06-b04-a01 | 1.69 | **Wrap-free down-weight multicasts**: senders sit on their group's top or bottom row, and a bottom-row sender multicasts on NoC 1 (−y), so no column multicast wraps the torus. |
| r07-b02-a01 | 1.74 | **Wrap-free gate/up multicasts**: the same fix for the gate/up half, with the DRAM read issued before the receivers' acks. |
| r08-b04-a01 | 1.78 | **Cross-group weight pacing**: a row group's weight senders hold their next block's DRAM read while the group is more than one gate/up block ahead of any unfinished group. A per-group progress board on core (0,0) costs one atomic and one 32 B read per block. |

## Things that didn't help

These are worth knowing before trying them again. The CSV has every attempt.

- **Issuing the next weight block's DRAM read before the current multicast** (as a first attempt): about −7%. The read traffic contends with the critical multicast on the sender. It only paid off later, once the read was gated on the sender's own ring slot (step 11).
- **Doubling the gate/up K-block** (`in0_block_w_gu` 16→32): this is a precision knob, because it doubles the K-span accumulated in bf16 dest. Mean error went up 2–3.5%, so it was rejected.
- **Fusing plain SiLU into a SiLU-GLU SFPU op**: removes an FPU pass, but mean error rose about 1.4–1.5%. Rejected.
- **Streaming x through SrcB / reuse-B matmul** (hold the bf4 weight tile in SrcA): SrcA writes fell about 5×, but cycles per tile-mult didn't move; the SrcB unpack became the bottleneck. −2 to −3%.
- **Per-sender NoC request VCs** and a **deeper down-weight look-ahead** (r02-b02-a02/a03): neutral, and dropped when the lineage moved onto r02-b03-a03's code. The contended resources are read responses and DRAM banks, not request arbitration.
- **Grid-wide group pacing through a multicast progress board**: the spread between groups collapsed, but every group got 29–55% slower. The cheap per-group board in step 15 is the version that works.
- **Plan-aware start-up stagger** and **tilting the NoC split by grid row**: neutral or a loss.
- **Moving gate/up weight reads to the reader only**: put all responses on NoC 0 and cut weight bandwidth 8–10%.

## Caveats

- Only the ROW_MAJOR dispatch-buffer path is benchmarked. The search required every other path to stay correct: TILE inputs, biases, every activation, any token count including 0, and multiple experts per chip. Not all of them are covered by this test, though, so run the op's other tests before relying on this.
- All numbers come from one Blackhole machine. Most of the gains are data-movement and scheduling changes, so they may move on other boards or in a full mesh where other ops share the NoC.
- `re_hwprof.hpp` (step 10) is measurement scaffolding. It can be dropped if you don't want it; I haven't re-measured without it, but it is profiler-gated.

## How it was found

An agent-driven search loop based on Dream-RSI ("Recursive Self-Improvement through Evolving Worlds") ran 73 attempts over 8 rounds, about 10 hours, all on the test above. In each step up to 4 Claude Code agents each proposed, implemented and benchmarked one change in its own git branch. A policy, tuned between rounds by replaying earlier rounds, decided which branches to continue. Each new round started from the best attempt so far. Every attempt had to pass a per-shape accuracy gate relative to the baseline (mean_abs within +1%, PCC within −1e-4), and no edits outside the op directories were allowed.

| round | attempts | best after round |
|---|---|---|
| 1 | 16 | 1.501 |
| 2 | 16 | 1.570 |
| 3 | 16 | 1.610 |
| 4 | 9 | 1.628 |
| 5 | 4 | 1.667 |
| 6 | 4 | 1.693 |
| 7 | 4 | 1.739 |
| 8 | 4 | 1.781 |

69 of the 73 attempts were valid; 4 failed (2 accuracy, 1 runtime error, 1 hang).
