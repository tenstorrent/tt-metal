# Optional lane-partitioned router projection

The optional `LanePartitionRouter` is implemented in `tt/optimized_decoder.py`.
Defaults remain unchanged (`router_lanes=0`). No hardware test was run by this
investigator, and no latency gain or routing equivalence is claimed yet.

The measured target is **93.376 us** of projection operations in the initial
[expert-candidate decode table](tracy/expert_candidate/decode_perf_report.csv):

| Row ID | Projection operation | Device time (us) |
| --- | --- | ---: |
| 3304 | FP32 broadcast multiply | 49.258 |
| 3305 | FP32 reduction | 42.044 |
| 3306 | Transpose | 2.074 |

This sum excludes inter-operation gaps. The separately recorded TopK operation
at row 3308 costs 24.379 us and is unchanged. The approximately 252 us reduction
in the same initial profile belongs to QKV, not the router. These old profile
rows identify the optimization target; a fresh paired run must establish any
gain in the current native-SDPA decoder.

## Preserved arithmetic and new projection

The actual wrapper chain is `BroadcastRouter -> routing_precision.Router ->
Gemma4 router`. The new wrapper keeps the original `BroadcastRouter` for
prefill and retains its normalization callback. Decode accepts the already
computed `normalized` argument without repeating normalization, or computes
normalization through that same callback when the argument is absent.

The learned scale is `routing_precision.Router.scale`, already represented in
FP32. The fused multiply and `scalar_root_size` unary activation are copied
exactly from `BroadcastRouter`. Only the projection of this **already scaled**
FP32 activation changes. Selection still uses the same top eight logits,
selected-logit softmax, BF16 values and scatter, and original learned
`per_expert_scale`. No learned scale is replaced, folded, or omitted.

The projection reuses `LanePartitionQKV` with a small immutable-by-convention
source adapter containing the original BF16 `router.proj_weight`, the precision
router compute configuration, and L1 output placement. The adapter has no tied
K/V attributes. Prefill never calls this adapter: it delegates to the original
router, including any supplied normalized tensor.

For decode, 16 disjoint K lanes and two BF16 activation components produce
M=32 rows. One BF16-by-BF16 matmul produces FP32 row outputs; an FP32 row sum
returns the 128 scores. Three components are also supported and produce
M=48/padded64. This preserves the lane mechanism used for QKV; its accuracy
must still be checked on the router's much tighter rank-eight/rank-nine gaps.
All mask construction/upload and compute/program configuration happen during
setup. Forward performs only device operations.

## Configurations and checks

Factory controls are:

```python
router_lanes=0
router_terms=2
router_grid=(4, 1)
router_block_w=11
router_fidelity=ttnn.MathFidelity.HiFi4
```

Enable the initial candidate using existing JSON overrides:

```json
{"router_lanes": 16, "router_terms": 2, "router_grid": [4, 1], "router_block_w": 11, "router_fidelity": "HiFi4"}
```

The runtime resolves string fidelity values for this option, so the CLI needs
no edit. Enabled policy metadata records the actual original weight shape and
dtype, lane/component counts, grid, K block, fidelity, FP32 accumulation/output,
disabled approximate math and L1 packer accumulation, and L1 output placement.

K=2816 is 88 tiles; N=128 is four tiles. Grid 4x1 yields `per_core_N=1`, with
`per_core_M=1` for two terms or 2 for three terms. Subblock 1x1 satisfies FP32
destination limits. K blocks 11, 22, 44 and 88 divide K and are accepted by the
1D input-multicast configuration. Larger blocks still need an actual L1
allocation check; shape legality alone is not a memory proof.

An AST check confirms the router decode normalization, scaling and selection
body is identical to `BroadcastRouter` after replacing its projection branch.
It also confirms every existing factory default, unrelated class, and other
decoder method is unchanged. The entire candidate was formatted and compiled
before one atomic replacement of the runtime file; a hash check prevented
overwriting concurrent edits. Black, compilation and `git diff --check` pass.

The device-owner probe should compare score errors and selected expert IDs on
the same activation, then run the unchanged actual-text layer gate and warmed
latency measurement. Retain the option only if correctness passes and measured
latency improves; mask/decomposition overhead can consume the small projection
savings. Three terms or a different legal K block are discriminating controls
if two-term reconstruction crosses a routing boundary.
