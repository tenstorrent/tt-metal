# Reading tt-perf-report output

## Headline metric

`DEVICE FW DURATION [ns]` — time from first RISC firmware start to last RISC
firmware end for the op. This is the on-device cost and the canonical number
to minimize.

Rank ops by this column; the top 1-3 dominate the device-time budget and
are the default optimization targets.

## Key CSV columns

| Column | What it tells you |
|---|---|
| `OP CODE` | C++ class name of the op |
| `OP TYPE` | `tt_dnn_device` = ran on device; `python_fallback` / `tt_dnn_cpu` = host code (not profilable at kernel level) |
| `CORE COUNT` | Number of Tensix cores the op dispatched to. Compare against the grid available on the part (64 on Wormhole B0). Low count under a heavy op = under-parallelized. |
| `PARALLELIZATION STRATEGY` | How the op shards work across cores. Names differ per op; a missing or fallback strategy often means sub-optimal placement. |
| `MATH FIDELITY` | LoFi / HiFi2 / HiFi3 / HiFi4. Lower fidelity = fewer cycles per tile in the FPU. Over-specified fidelity is a common silent cost. |
| `DEVICE FW DURATION [ns]` | Headline. |
| `DEVICE KERNEL DURATION [ns]` | Compute-kernel-only window. `FW - KERNEL` ≈ setup + teardown + CB flush. A large gap on a short op means setup dominates. |
| `DEVICE BRISC KERNEL DURATION [ns]` | Reader (DM0) kernel time — usually dominated by `noc_async_read` traffic and barriers. |
| `DEVICE NCRISC KERNEL DURATION [ns]` | Writer (DM1) kernel time — `noc_async_write` and output draining. |
| `DEVICE TRISC0/1/2 KERNEL DURATION [ns]` | Unpack / Math / Pack compute stages. |
| `HOST DURATION [ns]` | Queue time + host dispatch overhead. Inflated on first iter (see First-run caveat below). |

## Bottleneck tags

After reading the per-RISC columns for a top op, tag the bottleneck. Rule of
thumb: the RISC with the **largest kernel duration relative to its peers** is
the bound.

| Tag | Signal |
|---|---|
| `reader-bound` | BRISC duration ≳ TRISC1 duration. Compute waits on tiles from DRAM/L1. Often fixable by batching `noc_async_read` before the barrier, increasing CB depth, or switching to a sharded input layout. |
| `compute-bound` | TRISC1 (math) dominates. Either a legitimate compute ceiling, or math fidelity is higher than the model requires, or the op could use a lower-cost variant. |
| `writer-bound` | NCRISC dominates. Compute waits to push output. Uncommon in isolation — usually means downstream pressure or under-sized output CB. |
| `under-parallelized` | Low `CORE COUNT` relative to device grid, combined with high per-core duration. Sharding or a larger dispatch grid can help. |
| `NOC-stall` | BRISC and NCRISC both high but TRISC low. Data movement is saturating NOC bandwidth — batching, routing, or layout changes help. |
| `host-dominated` | `FW` is short but `HOST DURATION` is large on runs past the first. Rare for device ops; points to Python or dispatch overhead, not kernel code. |

Record the tag only. Fix selection belongs to the caller (developer or
`tt:optimizer`).

## Matmul bound classification (op-level, from tt-perf-report)

`tt-perf-report` emits `DRAM %` and `FLOPs %` utilization columns for matmul
ops and tags suspect rows `SLOW`. These classify the op-level bound — a
different question from the processor-level tags above (which classify which
RISC is the blocker within a single kernel).

| DRAM % | FLOPs % | Tag | Bound |
|---|---|---|---|
| low (< 40) | low (< 40) | SLOW | **overhead / sync-bound** — dispatch, barriers, under-sized blocks. Default progcfgs commonly land here. |
| high (≥ 60) | low (< 40) | — | **bandwidth-bound** — memory feed is the ceiling |
| low (< 40) | high (≥ 60) | — | **compute-bound** — math engine is the ceiling |
| — | high (≥ 70) | — | **near peak** — little headroom |
| mid / mid | — | — | mixed; inspect per-RISC |

Lever families per class (for the *caller*, not in the note):
- **Overhead-bound**: raise `in0_block_w`, reshape M to enable a 2D progcfg
  with larger `per_core_M`, enlarge `out_subblock_h × w`, lower math
  fidelity, drop `fp32_dest_acc_en` (frees DST for bigger subblocks).
- **Bandwidth-bound**: prefer L1-sharded activations matching the matmul
  grid over DRAM-interleaved, `packer_l1_acc=True`, DRAM-sharded matmul
  variant, reduced-precision weights (BFP8).
- **Compute-bound**: fidelity is already the ceiling — remaining levers are
  a different kernel family (1D vs 2D vs DRAM-sharded) or reducing
  arithmetic intensity via fusion.
- **Near peak**: stop tuning this op; pick the next bottleneck.

Confirm the bound class from DRAM% and FLOPs% before proposing levers.
Overhead-bound matmuls tuned with compute-bound levers waste iterations.

## Peak reference figures

For computing FLOPs% and DRAM% against ceilings, and for sanity-checking
absolute numbers:

| Part | Tensix cores | Peak @ HiFi2 BF16 | Peak @ HiFi4 BF16 | DRAM bandwidth |
|---|---|---|---|---|
| Wormhole B0 (per N150 / per N300 device) | 64 | ~128 TFLOPs | ~64 TFLOPs | ~288 GB/s |
| Blackhole | 140 | ~280 TFLOPs | ~140 TFLOPs | ~512 GB/s |

HiFi2 peak is 2× HiFi4 because math cycles per tile halve; LoFi is another 2×.
`tt-perf-report`'s FLOPs% is against the *current* fidelity's peak, not the
max-fidelity peak. Swapping HiFi4 → HiFi2 can show FLOPs% dropping even as
absolute TFLOPs rise (the denominator doubled). Compare `Abs TFLOPs`, not
`FLOPs%`, across fidelity settings.

For reshaped-batch matmul (seq folded into batch), the `M, K, N` used by
`Abs TFLOPs` are the *effective* per-device shape after the reshape — not
the pre-reshape activation shape. Record the effective shape in the note
when the reshape is in play.

## Overhead confirmation — FW vs KERNEL gap

`overhead_ratio = (FW - KERNEL) / FW` directly measures dispatch + barriers
+ CB-flush overhead.

| overhead_ratio | Reading |
|---|---|
| < 15% | compute/bandwidth dominates — use DRAM%/FLOPs% table |
| 15-40% | mixed; decide from per-RISC tag |
| > 40% | **dispatch/sync dominates** — overhead-bound even if FLOPs% is ambiguous |

overhead_ratio > 50% with low FLOPs% signals a **reshaped-batch matmul**
(long sequences folded into batch to shrink `per_core_M`, producing many
tiny dispatches — typical in prefill MLPs at seq ≥ 4K). Levers: larger
`in0_block_w`, larger `out_subblock_h × w`, or switch to a DRAM-sharded
or 1D matmul variant.

## Layer-scope profiling: contribution column

When profiling an enclosing layer (MLP, attention block) rather than a
single op, the top-op ranking is more useful with a contribution column:

| Rank | Op Code | Device FW [ns] | % of scope total | Cumulative % |
|---|---|---|---|---|
| 1 | ... | 5,273,000 | 36.4% | 36.4% |
| 2 | ... | 4,326,000 | 29.9% | 66.3% |

`% of scope total` makes "is op X THE bottleneck" quantifiable: an op is THE
bottleneck iff its contribution > sum(others). Use this when the caller's
goal is utilization-typed (see `skills/optimizer/convergence.md`).

## First-run caveat

First iteration of any test populates the program cache — host times are
inflated. Device-side columns remain valid.

## When numbers look wrong

| Symptom | Likely cause |
|---|---|
| All device durations = 0 | Env conflict (`TT_METAL_DPRINT_CORES`, `TT_METAL_WATCHER`, `TTNN_CONFIG_PATH` still set) |
| Only a handful of ops captured | Buffer overflow — add periodic `ReadDeviceProfilerResults(device)` calls |
| Wildly different numbers across runs | Program-cache miss on measured iter, or concurrent agent workload (MCP queue serializes device, not caches) |
