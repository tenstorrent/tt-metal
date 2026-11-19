# Remaining functional-floor diagnosis

CPU-only source and historical-profile audit, 2026-11-14. No hardware runs or
implementation changes were made for this report. The independent installed
AutoDebug investigation is preserved in `AUTODEBUG_functional_floor.md`;
launcher provenance and limitations are described below.

## Failed gate and accounting boundary

`functional_floor_audit_v1.json` is authoritative for this comparison:
113.881454 ms/token (8.781061 tokens/s/user), versus a prepared-major-weight
plus short-window KV floor of 6.414528 ms. Achieved fraction is 5.632636%, below
10%. Required latency is at most 64.145280 ms, leaving **49.736174 ms/token**
to remove. This is a failed gate, not a pass based on a different denominator.

The 11.599040/12.975296 ms implemented-graph byte subtotals in
`performance_accounting_plan.md` explain avoidable traffic. They cannot be
used to pass this gate. FP32 product writes/reads, padded scorer outputs and
unary passes are optimization opportunities, not additional useful model work.
The strict inventory itself is a subtotal; any refinement must identify real
required weights/state and avoid counting an implementation's inefficiency.

The selected native causal scorer, LM head 20 cores/3 readers/block2 and split
sampler synchronous CCL are already included in the 113.881454 ms result. The
old capacity-wide QSA scorer and old slow GR injection are **not** remaining
optimization candidates. This diagnosis changes no dtype/fidelity, four-stream
residual ownership, KV representation, sampler contract, or precision rejection.

## Measured evidence and limits

Historical evidence is
`../full_model/terminal_final_perf_v1/model_perf_report.csv`, joined by
`Global Call Count` to `ops_perf_results.csv` for shapes, attributes and kernels.
The merged report's chosen global call may come from any rank: do not restrict
the raw join to device0. Source baseline is `de432e04`; current implementations
of GR/GDN discussed here are unchanged. CSV regions below use zero-based
half-open **data-row** indexes, excluding the header.

| Complete historical region | Rows | Device time | Full-stack multiplicity |
|---|---:|---:|---:|
| Token entry | 0:5 | 39.06975 us | 1 |
| Plain GDN layer0 | 5:298 | 2022.53800 us | 35 |
| PLE/GDN layer1 | 298:795 | 3043.04075 us | 1 |
| QSA layer3, including obsolete scorer | 795:1204 | 5066.12250 us | 12 |
| Final mixer/head/layout | 1204:1253 | 1086.03650 us | 1 |

These are historical per-operation device sums, not new clean trace latencies.
Do not subtract their weighted sum from the selected clean end-to-end time and
call the difference host overhead. Operation timing, rank reduction, trace
boundaries and changed QSA code must first be reconciled with a fresh matched
profile. `run_terminal_final_v1.sh` provides the matching reduced full-path
profile recipe at capacity262144. Profiler output is for attribution; clean
resident model+sample replay remains the acceptance timing boundary.

Layer0 partitions expose the main remaining problem:

| Layer0 region | Rows | Time, us | Finding |
|---|---:|---:|---|
| First GR read | 5:58 | 407.49450 | 53 operations |
| GDN body | 58:161 | 670.82325 | Includes exact QKV and output projections |
| First GR write | 161:169 | 31.50800 | 8 operations |
| Second GR read | 169:222 | 400.73400 | 53 operations |
| MoE | 222:290 | 482.30025 | Includes routing, expert compute, shared branch and rank reduction |
| Second GR write | 290:298 | 29.67800 | 8 operations |

Two GR reads plus writes consume 869.4145 us, about43% of this layer's device
sum. There are97 GR reads including the terminal mixer. GR is substantial
model mathematics, but its current product materializations, padded packing
and many small formatting kernels are avoidable implementation work.

## Prioritized exact-math changes

### 1. Fuse precise projection products with the selected SFPU reduction

Target `tt/fused_decoder.py::FusedPreciseProjection.__call__`, used by
`tt/multichip_decoder.py::MeshExactProjection`. Current B1 code expands the
input, creates an FP32 `[1,1,N,K]` product in DRAM, then runs accurate SFPU W
sum, reshapes, concatenates and rounds to BF16. The strongest first targets are
GR down/up and GDN output; router projection can use the same implementation.

| B1 local matrix | Existing critical operations | Historical cost |
|---|---|---:|
| GR down, logical80/padded96 outputs, K10240, FP32 weights | FP32 product; accurate W sum on only3 cores; one-core compact concat after gather | 36.573 +71.393 +30.354 us per GR read |
| GR up, N2560 K320, FP32 weights | Product; accurate W sum on80 cores | 33.587 +12.733 us per GR read |
| GDN output, N640 K6144, BF16 weights | Product; accurate W sum on20 cores | 95.021 +68.527 us per GDN layer |

A native implementation can read weights/input tiles, multiply in FP32 DST,
fold products in exactly the current W-tile sequence, apply the same SFPU
within-tile row reduction, and write only compact reduced output. FP32 products
must retain the existing separate multiply rounding; do not silently use FMA.
BF16 weights/input widen exactly, GR weights stay FP32, and the final BF16
rounding stays at the same boundary. Keep scalar scaling/nonlinearity rounding
separate until bitwise equivalence proves an epilogue preserves it.

The authoritative reduction is
`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl`: the accurate REDUCE_ROW
path copies the first W tile, sequentially folds later tiles with SFPU add,
then calls the SFPU row reducer. It is **not** a free reassociation into K
partial sums. A K-split parallel reduction or ordinary HiFi4 matmul is not an
exact replacement. Reuse the existing helper's within-tile algorithm and
Blackhole FP32 settings rather than importing the QKV/indexer's different tree.

GR down exposes only3 output tiles to the generic reducer, so simply replacing
two kernels with one tile-per-core kernel leaves most of the mesh idle. Partition
output rows into smaller groups and pack their weights/accumulators so the same
per-output K order is preserved. Avoid loading a whole32-row weight tile on32
cores for one useful row each: that trades idle cores for32x input requests.
Candidate implementations are row-group4/8 packed weights or shared tile loads
distributed to row workers. Setup may replace a weight representation; it must
not retain duplicate FP32 matrices in the capacity262144 full model.

Removing product traffic alone is not a latency guarantee: small-core serial
SFPU work and per-tile CB handshakes remain. First compare a conservative fused
tile implementation, then row packing. The GR down/up product+reduce and
compact-concat ceiling is17.848851 ms across97 reads (10.367852 ms down,
4.536640 ms up,2.944359 ms compaction); GDN output product+reduce ceiling is
5.892206 ms across36 layers. These use each layer-kind's own measured costs,
weighted35/1/12, plus the terminal mixer once. They are impossible zero-cost upper
bounds, not expected savings, and exclude required new fused-kernel runtime.

### 2. Compact GR output packing and fuse local ordered rank reduction

`GatheredProjection(valid_width=80)` currently gathers four padded96-wide
outputs, slices80 from each, and concatenates to320. The local concat alone
costs30.354 us in the representative first GR (about2.94 ms across97 reads,
already included in item1's ceiling). A native gather-output compactor can
read rank segments in fixed order into the compact tiled result. Alternatively
choose a private projection output layout that the compactor can consume while
retaining the public320-wide output and original transport precision.

`Collectives.sum` keeps a synchronous BF16 gather, widens to FP32, and reduces
rank dimension1. The historical local implementation adds clone, fill-padding,
HC transpose, accurate H reduction and transpose-back. One representative
sequence costs7.316 +9.796 +9.004 +8.475 +2.797 +2.668 =40.056 us after gather.
A custom local reduction can read rank-major gathered data directly, assemble
the four logical H rows with zeros in padded rows, and apply the **same** SFPU
H reduction. Preserve BF16 gather and resulting FP32/BF16 round boundaries.
Do not assume ordinary serial four-value addition matches that reduction tree.

There are approximately158 ordered sums per full token:96 GR,48 MoE,12 QSA
output,1 PLE and1 final mixer. A40 us local-cleanup ceiling would be6.3 ms;
the96 GR contributions overlap the whole-GR budget and must not be added again.
Actual savings exclude the mandatory sum and gather. This candidate is a local
formatting/reduction fusion, not adoption of previously rejected async/native
all-reduce or a replicated residual stream.

### 3. Fuse GR pointwise groups with explicit rounding

`MeshResidual.write` currently launches casts, multiply, a BF16 boundary, casts,
add and another BF16 boundary. A local native write can retain both roundings
in a single kernel. The older single-device `_RESIDUAL_WRITE_*` kernels in
`fused_decoder.py` are useful implementation evidence, but their flattened
four-stream indexing is not the one-whole-stream-per-rank mesh contract; do not
wire them directly. Historical layer0 ceiling is61.186 us/layer; weighting each
layer-kind's own write regions35/1/12 gives3.004752 ms/full token.

The read path's scale/cast/SiLU/cast and up-projection/cast/sigmoid/cast groups
are similarly fusible epilogues, preserving every selected BF16 boundary.
Fuse only after the projection is correct; evaluating a nonlinear in a
different dtype or removing an intermediate BF16 rounding is a policy change.
Normalization must retain all2560 local-stream features and the selected RMS
reduction. A fast hidden640 norm is a different model.

### 4. Remaining native kernels and routing: smaller, necessary follow-ups

* Exact GDN QKV costs132.536 us on80 cores in layer0, at most4.771 ms across36
  layers. Its selected balanced tree differs from the projection reducer above.
  `_QKV_TREE_COMPUTE` repeatedly packs full FP32 partial tiles through CBs to
  implement the16-way and5-group tree. Compact partial rows/retain more tree
  nodes in DST while preserving every pair and odd tail. Avoid accidental FMA
  or a native matmul replacement. The complete old kernel cannot be saved twice
  by separately counting its reader and compute subparts.
* The ordered QSA indexer costs438.237 us on100 cores, at most5.259 ms over12
  layers. `parallel_ordered_projection` preserves five512-feature CPU block
  accumulations and ordered high/low BF16 products. It performs per-feature
  CB/DST transfers. Pack several sequential input/weight steps per CB transaction
  or use direct SFPU loads without altering the ordered accumulation. Merely
  sharding outputs four ways reduces work but also worker count; it does not
  establish4x latency improvement. This kernel is distinct from the now-fixed
  causal scorer.
* GDN recurrent update is71.224 us on12 cores, at most2.564 ms over36 layers.
  Parallelizing value rows can improve occupancy only if each state row's
  reduction/order remains intact. It is stateful: replay and mixed-slot
  recurrence comparisons are mandatory. This is lower priority than17.9 ms of
  repeated GR projection/packing work.
* MoE routing/metadata before fused SwiGLU occupies175.356 us/layer (8.417 ms
  at48 layers). It contains required router projection, exact ordered top-k,
  metadata construction and dispatch. Native metadata construction and routing
  fusion can remove formatting launches; do not replace `aten_router_topk`
  tie order or expert weighting. Fused SwiGLU itself costs96.881 us/layer
  (4.650 ms total), with required weight bandwidth and EP imbalance; it is not
  zero-cost potential savings. B1 indexed expert kernels need a compatible
  weight format without a second resident expert copy. Such a layout change
  needs full-capacity memory evidence before being selected.

## Can these changes honestly close49.7 ms?

Not yet established. The strongest nonoverlapping visible pieces—GR
product/reduce/compact packing17.85 ms, GDN output5.89 ms, GR writes3.00 ms,
QKV4.77 ms and ordered indexer5.26 ms—sum to only36.77 ms even if they became
free. Their actual savings must be smaller. Additional GR normalization,
pointwise work, local rank reduction, MoE routing and dispatch gaps matter.
The whole-GR read/write budget is about42 ms at layer0 costs, but contains
required normalization, two weight projections, three gathers and model math;
it cannot be claimed as removable. Network gathers alone are approximately
6.26 ms in the historical weighted layer report and cannot explain50 ms.

Therefore projection fusion is a compelling first experiment, not proof of a
64.145 ms endpoint. A fresh selected per-operation profile must quantify what
remains after that experiment. If measured improvement leaves no plausible
policy-preserving route to the strict gate, record concrete tested failures and
continue through AutoFix's failure contract; do not inflate the denominator,
silently accept a rejected policy, or call an unimplemented idea a hard limit.

## Focused validation and decision sequence

1. Capture current clean layer0/1/3 B1 trace at capacity262144 and selected
   full-token resident trace; profile reduced full-path shapes separately with
   Watcher off, profiler capacity8192 and tracking off. Use complete layer
   boundaries and all-rank/max-rank timing. Do not profile all48 with overflowing
   buffers or use an eager per-op sum as the trace result.
2. For native projection, compare logical FP32 reductions and final BF16 outputs
   bitwise to the selected operation graph with real checkpoint matrices and
   real activations. Cover K320/2560/6144/10240, N80→96/128/640/2560,
   Q1/2/3/17/31/32, negative/cancelling values, tile tails and all four ranks.
   Include consecutive calls with changed tensor addresses and cached programs.
   Bitwise disagreement is a repro to investigate, not permission to loosen
   the selected policy. Non-aligned prefill must continue to use its logical mask.
3. Compare whole GDN/QSA/PLE layer hidden outputs, router IDs/weights, QSA IDs,
   recurrent/KV state and position advancement; include mixed slots/inactive
   rows, decode after non-aligned prefill, repeated traces and cache reset.
   Validate native CCL local sum against the selected gathered tensor including
   rank-distinct signed values and padded physical rows.
4. Keep only measured complete-layer wins with identical inputs/configs;
   record candidate compile/L1/accuracy failures. After a material kept change,
   refresh installed full48 AIME prefill/teacher-forcing98% top5/100% top100,
   autoregression and serving-state checks, then remeasure the clean resident
   model+sampler boundary with no interior host synchronization/readback.
5. Recompute lower bound `35*t_L0 + t_L1 + 12*t_L3`, entry, terminal and split
   sampling separately. Compare it with selected resident trace, explain any
   >10–15% gap, and separately recompute the strict useful-weight/KV efficiency.
   Kernel profiler gaps and host gaps need their own evidence; they are not
   interchangeable. Apply the optimize Stop Rule without claiming failed gates
   complete; preserve untried candidates and exact reproduction commands.

The inherited rejection artifacts and user precision authority are enumerated
in `operation_topology_audit.md::Inherited rejection ledger and precision
authority`. In particular `async_l0.json`, `async-persistent_l0.json`,
`async-bfp8_l0.json`, `hidden_qsa_rs.json`, `hidden_qsa_agmm.json`,
`real_bfp8_kv.json` and `real_qsa_bfp4.json` remain binding evidence. No broad
datatype sweep or vLLM work is proposed here.

## Isolated investigation provenance

The installed AutoDebug launcher was used, preserving fresh-process isolation.
The first launch selected `/home/ttuser/.local/bin/codex`, whose adjacent
`codex-code-mode-host` was missing. All child filesystem calls failed before
source inspection; `autodebug_functional_floor_launch.log` records this failure.
The retry places the complete installed CLI directory
`/home/ttuser/qwen38-flash/codex-venv/lib/python3.12/site-packages/codex_cli_bin/bin`
first in PATH. It retains the explicitly authorized
`AUTODEBUG_SKIP_CHILD_SANDBOX=1` because the earlier sandbox preflight failed
and this parent workspace is already unrestricted. No skill/source file or
system installation was modified to repair the launcher. Retry output is
`autodebug_functional_floor_retry.log`. The final isolated report is copied to
`AUTODEBUG_functional_floor.md`; root `AUTODEBUG.md` is restored to HEAD after
the launcher completes.

## New focused component timing recipe

`tests/measure_full_model_components.py` now provides a CPU-checked, hardware-
unverified alternative to generating six single-chip/mesh reference jobs.
It builds one reduced0/1/3 generator at capacity262144, performs native eager
natural128-token prefill, and retains the inputs and outputs of one real decode
pass. It warms and captures token entry, embedding, each complete layer,
terminal mixer/head, row assembly, canonical split greedy and a combined
reduced model. Every first replay must exactly match its eager output.

```bash
source models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/environment.sh
unset TT_METAL_WATCHER TT_METAL_DEVICE_PROFILER TT_METAL_TRACE_PROFILER TT_METAL_STREAMING_PROFILER
export TT_METAL_PROFILE_PERF_COUNTERS=0
export TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_TRACE_ALLOC_TRACEBACKS=0
timeout --signal=INT --kill-after=60 3600 python_env/bin/python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.measure_full_model_components \
  --prompt-length 128 --iterations 127 --trials 3 \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/optimized_full_model/component_trace_v1.json
```

State snapshots and all program warmups precede the first capture; restores
and readback occur outside timing. Each timed window enqueues127 nonblocking
replays and synchronizes only once at the end. Layer3/combined-model positions
advance128→255; component inputs remain frozen. GDN state also advances.
The output reports35/1/12 weighting, separate terminal/sampler costs, and
reduced-component sum versus combined reduced trace. This is a representative
stack estimate using reduced-model reachable activations, not all48 residency,
full autoregression, a new quality gate, or proof of an absolute lower bound.
Syntax, formatting and `--help` were checked without importing device code.
