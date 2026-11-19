# Historical B1 profiler findings — global dispatch, GU block 16

CPU analysis of the earlier screening source's S32/B1 captures in `tracy/layer_0` and
`tracy/layer_3`. Both `correctness.json` files pass the summarizer's PCC, state,
determinism and runtime-fallback checks. The default CPU summary was regenerated
with `python models/autoports/qwen_qwen3_6_35b_a3b/tests/summarize_multichip_perf.py`.
This is historical profiler interpretation. The subsequent geometry/local-dispatch
screenings addressed the opportunities below; final-source profiles live in the parent
directory. Original embedded paths identify the capture location before preservation.
To reproduce this historical summary, pass `--doc` pointing to this directory.

## Accounting and provenance

Each capture contains four physical-rank rows per logical operation. Decode
contains16 replay sessions; operation signatures match across all16 replays
within each rank and all durations are present and positive. Linear decode has
177 operations per rank per replay; full decode has79. Prefill has1545 and72
operations per rank, respectively, for one measured pass.

| Layer / phase | Rank0 sum, ms | Rank1 sum, ms | Rank2 sum, ms | Rank3 sum, ms | Critical-rank sum bound, ms |
|---|---:|---:|---:|---:|---:|
| Linear prefill | 34.114148 | 34.182578 | 34.126285 | 34.086466 | 34.182578 |
| Linear decode | 0.882343 | 0.879801 | 0.879529 | 0.878268 | 0.882343 |
| Full prefill | 1.757827 | 1.641347 | 1.758601 | 1.764924 | 1.764924 |
| Full decode | 0.365605 | 0.364151 | 0.363867 | 0.361402 | 0.365605 |

Decode entries average16 per-rank replay sums. The maximum rank mean is a
serialized-work bound/proxy, **not measured mesh latency**: kernel intervals can
overlap and omit dispatch gaps. Four ranks are never summed as latency. Clean
warmed timing, speedup and efficiency must come from separate timing evidence.

The external `tt-perf-report` merges ranks. Its installed
`tt_perf_report/perf_report.py:3190` selects the maximum-duration rank for
noncollective operations, but averages rank durations for AllGather,
ReduceScatter and AllReduce, labeling collective rows with the first rank.
Consequently its `Device` column is a representative label, not a per-rank
workload. For example, linear decode's2832 report rows have representative
counts742/672/828/590, whereas raw CSV has2832 rows on **each** rank.
The summarizer now records report advice/bounds by merged operation and requires
all four ranks only in raw evidence, not in representative report labels.

Human report `Total %` and op-to-op gaps include roughly one-second profiler
dumps between decode replays. They cannot serve as clean latency shares.
No report totals or representative-rank row counts are used for per-rank time.
Human-readable reports remain unchanged.

## Category totals and precision verification

Numbers below are microseconds on the critical rank, per pass/replay. DRAM
matmul means the weight operand resides in DRAM; it does not establish that an
operation saturates bandwidth. Generic inline kernels remain compute because
the CSV alone cannot subdivide their internal work.

| Layer / phase | Communication | DRAM matmul | Other compute | Data movement/conversion |
|---|---:|---:|---:|---:|
| Linear prefill | 63.174 | 80.289 | 30413.414 | 3625.701 |
| Linear decode | 25.644 | 72.162 | 627.632 | 156.905 |
| Full prefill | 69.364 | 118.826 | 702.241 | 874.493 |
| Full decode | 20.406 | 120.728 | 139.801 | 84.670 |

Exact Dispatch/Combine operations are local data movement in this selected
size-one dispatch-axis path. Typecast is data conversion. PostCombineReduce
remains compute. These classifications avoid hiding the large local combine
under the compute total.

Raw rows confirm BFP8/LoFi indexed expert GU/down and grouped expert SwiGLU,
BFP8/HiFi2 shared projections and full-attention QKVG/output, and FP32-weight
HiFi4 router matmuls. Linear prefill output retains BF16/HiFi4. Linear decode
includes separate FP32 and BF16 all-reduces:17.274 and8.370us on rank0.
Its ordered projection/recurrence helpers appear as GenericOp; ordinary
matmul advice does not justify changing those sensitive arithmetic paths.

## Advice priorities

1. **Full prefill routing movement dominates.** Local combine is744.827us,
   dispatch39.455us, and grouped experts495.439us on rank3. Study the padded
   dispatch/combine representation and its real B32 profile before prioritizing
   a small dense projection change. Combine is not an inter-chip collective in
   this path.
2. **Indexed GU is the largest ordinary decode matmul.** It costs49.370us
   linear /49.311us full, versus down11.579/11.630us. Reports recommend L1
   input and larger subblocks. Current GU has one output tile per core, so
   larger subblocks require a legal smaller core grid; see the
   [local geometry audit](../local_geometry_audit.md). Those local candidates are
   **unmeasured**, not rejected by single-chip geometry results.
3. **Router and movement warrant attention.** TopK costs43.524/43.454us;
   the full router matmul adds24.981us. Full decode has84.670us of movement and
   conversion, versus20.406us of collective kernels. Replacement router paths
   need exact routing/PCC evidence; the report's fidelity advice is not an
   automatic precision change.
4. **Dense geometry experiments are bounded opportunities.** Full decode
   QKVG/output cost15.951/7.627us; shared packed/down8.585/2.643us.
   Linear shared packed/down cost8.531/2.681us. Reader/K-block/storage choices
   in the local audit are source-admissible, **unmeasured** candidates pending
   the planned precision-locked experiments. No optimality claim follows from
   the report's generic `in0_block_w=4 looks good` advice.
5. **Linear prefill is dominated by inherited sensitive inline work.**
   GenericOp groups total27997.531us of34182.578us. Ordinary DRAM matmuls total
   only80.289us. Geometry changes there cannot remove the principal cost.
   Linear decode's three ordered projection groups cost57.916,55.614 and
   53.662us; all GenericOp groups total283.649us. Any replacement requires an
   arithmetic-preserving design and new correctness evidence.

The selected dtype/packing policy has baseline evidence. That evidence does
not reject new local geometry, and these profiles do not prove a candidate
faster. B32 uses grouped EP and batched shared projections, so it needs its own
profile interpretation rather than extrapolation from B1 indexed sparse work.

Exact artifacts: `perf_summary.json`,
`tracy/layer_{0,3}/{ops.csv,ops_source.txt,correctness.json,correctness.json.provenance.json}`,
`tracy/layer_{0,3}/{prefill,decode}_{op_groups.csv,perf_report.csv,perf_report.txt}`.
