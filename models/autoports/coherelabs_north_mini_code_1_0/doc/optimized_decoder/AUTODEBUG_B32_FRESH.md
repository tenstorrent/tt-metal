# Fresh AutoDebug: batch-32 routed MoE

## Verdict

**No current-HEAD model-local fix exists that meets the contract.**

At current HEAD `acadb63da7d687e86e293b3a5d97f3315d8244ef`, the only fast
active-expert implementation, `ttnn.experimental.moe_compute`, has two
usable modes:

- `compute_only=True` is fabric-free but exposes an incomplete rolling
  two-slot buffer.
- `compute_only=False` produces a complete routed output only through the
  fabric-backed selective-reduce-combine path.

Neither a public option nor a Python-reachable internal option selects a
single-device local combine. That capability first appears in upstream commit
`50c56281566` (`Feature: Add single-device fused moe_compute support
(#49886)`), which is present on `origin/main` but is not contained in this
checkout's HEAD. Advancing shared TTNN to that commit or newer is the concrete
external dependency. It is necessary, though exact North-Mini Blackhole
correctness, traceability, and performance still require focused validation.

No implementation file was edited and no TT hardware was used.

## Direct observations

1. `OptimizedDecoder._sparse_moe()` sends batch-32 decode and other workloads
   with at least 32 tokens through `_dense_expert_moe_chunk`
   (`optimized_decoder.py:1431-1436`). That function repeats every token across
   all 128 experts and evaluates all expert projections
   (`optimized_decoder.py:1285-1383`).
2. The model-local active path uses `ttnn.sparse_matmul`, but reshapes its
   outputs over the full `[token, expert, ...]` surface
   (`optimized_decoder.py:1070-1161`).
3. Sparse-matmul output shape is derived like dense batched matmul
   (`sparse_matmul_device_operation.cpp:20-62`). Both newly allocated and
   caller-provided outputs are zeroed before every invocation
   (`sparse_matmul_device_operation.cpp:213-252`).
4. Current `moe_compute` defines only `Full` and `ComputeOnly`
   (`moe_compute_device_operation_types.hpp:21-25`).
5. Current compute-only output is an L1 tensor shaped
   `[worker_cores, 2, 32, hidden]`; its row-major view aliases the same buffer
   (`moe_compute_device_operation.cpp:274-327`). The `2` is the producer's
   double buffer, not a persistent expert dimension.
6. Current full mode requires combine parameters, positive `num_links`, and
   axis 0 or 1 (`moe_compute_device_operation.cpp:121-132`). Its invocation
   dereferences `cluster_axis` to resolve links and constructs the
   selective-reduce-combine parameters
   (`moe_compute_device_operation.cpp:451-491`).
7. Current standalone selective-reduce-combine likewise requires
   `num_links > 0`; its program/writer path constructs fabric mux workers and
   opens fabric connections. There is no current local-combine flag.

These observations independently reproduce the prior output-contract
diagnosis. The new finding is that upstream has now implemented the exact
missing capability.

## Hypothesis adjudication

### H1 — A hidden current-HEAD local mode can be selected model-locally

**Verdict: refuted.**

`cluster_axis=None` is accepted only for `compute_only=True`. With
`compute_only=False`, current code dereferences the missing axis while building
the fabric path. The current enum has no `FullLocal` value,
`SelectiveReduceCombineParams` has no `local_combine` field, and neither is
bound to Python. Passing `num_links=1`, an empty mux set, or a 1x1 topology
does not change the selected writer kernel; it remains the fabric writer.

**Focused check:** call current `moe_compute(compute_only=False,
cluster_axis=None)` with otherwise valid exact-shape inputs.

**Predicted outcome:** host-side failure while resolving the cluster axis, not
a six-tensor local result. Supplying an axis instead selects fabric and
reproduces the missing-peer/handshake behavior already observed.

### H2 — Ordinary TTNN ops can consume the compute-only buffer before reuse

**Verdict: refuted for a single fused call.**

The matmul output view aliases one fixed double buffer. Expert production and
buffer reuse occur inside the device program; an ordinary downstream TTNN op
is enqueued only after `moe_compute` completes. At that boundary, outputs for
earlier experts have already been overwritten. Standalone
`selective_reduce_combine` cannot restore them, and compute-only disables the
producer/consumer semaphore protocol that the fused consumer needs.

**Focused check:** route sentinel-valued tokens to at least four experts, run
compute-only once, and compare slot 4 against per-expert sentinels.

**Predicted outcome:** only the final two rolling slots are recoverable;
reducing or scattering slot 4 cannot reconstruct all routes.

### H3 — Repeated compact calls over static one-/two-expert groups can escape

**Verdict: not a viable no-regression solution; semantic details remain
experimentally checkable.**

Restricting each invocation to at most two resident experts could avoid
in-call overwrite, but complete 128-expert coverage needs up to 64 fixed
calls, 64 packed weight/mapping families, and a device-side reconstruction of
the compact expert-token order from the returned metadata. Current output is
not laid out as `[token, top_k, hidden]`; the token map/count tensors are
kernel metadata, not a ready dense index tensor. This is no longer competitive
with the 3.330-ms selected whole-layer path and is not the intended fix now
that an upstream local consumer exists.

**Focused experiment:** implement a test-only 2-expert sentinel call, prove
whether its slot-4 rows plus returned token maps can be converted to
`[32, hidden]` using only trace-safe TTNN ops, then measure 1, 2, 4, and 8
static groups and extrapolate 64.

**Predicted outcome:** a two-expert case may be reconstructible, but launch,
metadata, and packing costs scale with the group count and exceed the
batch-32 no-regression budget well before 64 groups. Any failure to express
the metadata scatter refutes the construction earlier.

### H4 — Alternate sparse-matmul packing or caller-provided output is compact

**Verdict: refuted.**

Changing rank/batch interpretation can move the expert dimension but does not
create dynamic route compaction. A `[route, ...]` formulation would require
gathering/duplicating selected full expert matrices or statically issuing
per-expert calls. Caller-provided sparse outputs are still zeroed every
invocation, and the kernel writes dense batch-derived offsets. Supplying a
smaller logical output would violate the writer's address contract rather
than provide supported compaction.

**Focused check:** inspect the allocated/output buffer bytes and zero-fill
row in a fixed-`nnz=256` call while varying only the optional output tensor.

**Predicted outcome:** unchanged full-surface allocation/zeroing, or an
unsupported shape/address failure; no compact `[32, 8, N]` result.

### H5 — Upstream `50c56281566` supplies the missing primitive

**Verdict: verified at source level; exact North runtime validation pending.**

Compared with this HEAD, the upstream commit:

- adds `MoEComputePath::FullLocal` alongside `FullCcl` and `ComputeOnly`;
- interprets `compute_only=False, cluster_axis=None` on a 1x1 mesh as
  `FullLocal`;
- adds `SelectiveReduceCombineParams.local_combine`;
- compiles the combine writer with `LOCAL_COMBINE`, using local NOC writes
  while excluding fabric headers, connection setup, mux workers, and
  cross-device barriers;
- extends the single-card test to run both compute-only and fused-local modes,
  validate the sixth combine output, and exercise the cached-program path.

This is exactly the capability prior reports proposed. It cannot be recreated
in `optimized_decoder.py`: the decisive change is a shared kernel that drains
each rolling slot while it is live.

The upstream single-card coverage uses DeepSeek/GPT-OSS shapes, not North's
`E=128, T=32, top_k=8, I=768, H=2048` matrix. Its test also carries a
Blackhole matmul-PCC caveat associated with issue `#50038`. Therefore the
commit is the correct dependency boundary, not by itself proof that this
stage can pass.

## Strongest actionable experiments

Run these only after the checkout and built TTNN are advanced to
`50c56281566` or newer.

1. **Exact component correctness**

   Use the already-supported North dimensions and packed weights, but call:

   ```python
   outputs = ttnn.experimental.moe_compute(
       inputs,
       top_indices,
       top_scores,
       expert_mapping,
       packed_w0_w1,
       packed_w2,
       layer_id=0,
       output_height_shard_dim=4,
       intermediate_size=768,
       cluster_axis=None,
       compute_only=False,
       activation_type=MoEActivationFunction.SILU,
   )
   assert len(outputs) == 6
   routed = ttnn.sum(outputs[5], dim=0)
   ```

   Compare `routed` against the authentic router+expert reference. Predicted
   outcome: slot 5 has complete score-weighted `[top_k, 32, 2048]` output and
   its top-k reduction matches the reference. A PCC failure would most likely
   expose the upstream Blackhole math issue, not missing routes.

2. **Trace/cache stability**

   Preallocate the optional combine output, capture the full decoder trace,
   replay it ten times with changed activations/routes, and compare every
   replay to an uncaptured call. Include a program-cache-hit call with a fresh
   output address, mirroring the upstream regression.

   Predicted outcome: stable six-output full-local execution with no stale
   address or semaphore state. Any trace-capture rejection identifies the next
   shared dependency before model integration is retained.

3. **Performance decision**

   A/B the same packed inputs under compute-only and full-local, then profile
   the complete layer including router, local combine, top-k reduction, and
   residual. Compare with the selected 3.330-ms batch-32 layer.

   Acceptance: complete routed output, authentic PCC at the current bar,
   whole-layer batch-32 latency no worse than 3.330 ms, and no dense-expert,
   host-readback, or fabric rows. The prior 1.642-ms compute-only measurement
   leaves plausible headroom, but this must be measured.

4. **Watcher and branch guard**

   Run the exact authentic layer-1/layer-4 batch-32 decode and representative
   prefill checks under watcher, then add a static/runtime test that fails if
   batch 32 calls `_dense_expert_moe_chunk`.

## Recommended intervention boundary

Do not add another current-HEAD model-local workaround. Advance or backport
the complete shared-TTNN single-device fused-MoE change represented by
`50c56281566` (prefer a revision at or after that commit rather than copying
only the visible flag). Rebuild TTNN, run the four experiments above, and only
then integrate the full-local call and top-k reduction in
`optimized_decoder.py`.

Until that dependency is available and passes exact North validation, the
honest status remains:

`blocked — shared TTNN FullLocal moe_compute required; no current-HEAD
model-local fix satisfies completeness, trace safety, fabric-free execution,
and batch-32 no-regression.`
