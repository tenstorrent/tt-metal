# Reading tt-perf-report output

## Headline metric

`DEVICE FW DURATION [ns]` — time from first RISC firmware start to last RISC
firmware end for the op. This is the on-device cost and the canonical number
to minimize.

Rank ops by this column. The top 1-3 dominate the device time budget — any
optimization push targets one of these unless there is a specific reason
otherwise.

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
| `HOST DURATION [ns]` | Includes queue time + host dispatch overhead. Inflated on first iteration (cache-miss populates the program cache). |

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

Use the tag only — do not prescribe the fix in a profile note. Fix selection
belongs to the caller (developer or `tt:optimizer`).

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

Misclassifying an overhead-bound matmul as compute-bound and tuning fidelity
or subblocks in isolation wastes iterations — confirm the class from DRAM%
and FLOPs% before proposing levers.

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
absolute TFLOPs rise (the denominator doubled). Compare absolute TFLOPs, not
FLOPs%, across fidelity settings.

### Absolute-TFLOPs formula (for fidelity-spanning comparisons)

When the only column available is FLOPs% (fallback-to-pandas path) or when
comparing across fidelities:

```
absolute_TFLOPs = 2 * M * K * N / (DEVICE_FW_DURATION_ns * 1e-9) / 1e12
```

where `M, K, N` are the *effective* matmul shape after any batch reshape
(per-device, post-M-reshape if seq was folded into a batch dim). Record both
`FLOPs%` and `absolute TFLOPs` per iteration when the optimization spans a
fidelity change — the percentage alone is misleading.

## Overhead confirmation — FW vs KERNEL gap

`DEVICE FW DURATION` spans fw_start → fw_end. `DEVICE KERNEL DURATION` is
the compute-kernel-only window. The ratio

```
overhead_ratio = (FW - KERNEL) / FW
```

is a direct measure of dispatch + barriers + CB-flush overhead.

| overhead_ratio | Reading |
|---|---|
| < 15% | compute/bandwidth dominates — lever selection by DRAM%/FLOPs% table |
| 15-40% | mixed; blocking decisions by per-RISC tag |
| > 40% | **dispatch/sync dominates** — confirms overhead-bound class even if FLOPs% is ambiguous |

An overhead_ratio > 50% combined with low FLOPs% is the signature of a
**reshaped-batch matmul**: long sequences folded into a batch dim to keep
`per_core_M` small, creating many tiny per-chunk dispatches. Typical symptom
in prefill-path vision or LLM MLPs at seq ≥ 4K. Levers: larger `in0_block_w`,
larger `out_subblock_h × w`, or (structural) switch to a DRAM-sharded or 1D
matmul variant that amortizes dispatch differently.

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

The first iteration of any test populates the program cache. Host times on
that iteration are inflated. If the test runs the layer only once,
`HOST DURATION` is unreliable. The device-side columns are still valid.

## When numbers look wrong

- All device durations = 0 → profiler did not run. Check for env conflicts
  (`TT_METAL_DPRINT_CORES`, `TT_METAL_WATCHER`, `TTNN_CONFIG_PATH` still set).
- Only a handful of ops captured but the test ran many → buffer overflow;
  test needs periodic `ReadDeviceProfilerResults(device)` calls.
- Wildly different numbers between two otherwise-identical runs → program
  cache miss on the measured iteration, or another agent's workload
  interleaving. Re-run; the MCP queue serializes device access but
  build/compile caches do not.
