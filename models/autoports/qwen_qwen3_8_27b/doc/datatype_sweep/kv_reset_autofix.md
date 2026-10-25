# AutoFix — BFP4 KV reset host packing

Status: repaired and verified through primitive, real-layer and full64 native-context checks.

The full BFP4-KV candidate progressed normally but spent tens of minutes preparing
native-cache prompt signatures. Source inspection found that `Generator.reset`
uses `copy(zeros_like(value),value)` for every state. TTNN creation.cpp
`full_like_impl` selects device fill for BF8/BF16/F32, but BFP4 falls through to
host `full_impl`, repacking and uploading the entire native cache on every reset.
The normal generate timer starts before reset, so this also affects request TTFT.
This is not evidence of a hang or device failure.

Preserved `failures/kv_bfp4_host_reset.json` and `.log`. Sent SIGINT to the owned
candidate PID1622994 after preservation; traceback confirms generator.reset ->
zeros_like. Mesh closed normally. It reached bucket1536 without a model accuracy
or timing result, so this diagnostic attempt cannot enter the Pareto frontier.
The serialized matrix exited; no later candidate started. CPU-only HF control
continues independently.

Independent source review: `kv_reset_review.md`. Hypothesis: only for BF4 cache
fields, use `ttnn.fill(value,0.0,output_tensor=value)` to overwrite the existing
packed device allocation. Same-dtype unary fill has no BF4 validation blocker;
the interleaved reader/packer preserves tile ownership. `moreh_full_like` is not
an alternative: its device op rejects BF4. No global TTNN change is proposed.
Default BF8 behavior remains unchanged.

First discriminating experiment: `tests/probe_bfp4_reset.py`. Verify nonzero to
exact-zero on all four ranks, repeated values and cache hits, unchanged allocation
addresses, both one page and the native8192-page key/value shape. Diagnostic host
wall time is not model performance and may overlap the CPU HF control. Next,
verify populated-cache reset followed by existing trace replay using the reduced
real four-layer/full-head generator, then retry full64 candidate accuracy/timing.

Primitive proof passed on all four ranks for the initial equal-volume flattened shape, with three distinct
nonzero fills before exact-zero reset and unchanged addresses. The initial primitive used1024x256 token rows, equal in volume to native KV but
not its exact page shape. It was subsequently rerun as8192x32 to match production
exactly, with an explicit four-rank count assertion; bfp4_reset_probe.json now
contains this stronger proof. Primitive timing is not full-model performance.

Applied BF4-only `fill(value,0.0,output_tensor=value)` in Generator.reset. Real
layers0/1/2/3 with full head and shared request owner then passed all preparation
and lengths1,33,4095,4096,4097. Each populated-cache reset checks exact zero on
all four ranks and stable addresses; subsequent repeated generation has identical
tokens and the same three trace IDs/capture count. Evidence:
`kv_bfp4_reset_replay.json`, with model-source hashes. Tracking includes program
cache allocations. Full64 native-context retry has started; final closure waits
for that result. No hardware reset was needed.

Full64 retry completes native262144 preparation and prefill/three traced teacher
forcing repeats with top1=.97,top5=1,top100=1. Median35.5352713t/s/u and
TTFT122.411798ms. Evidence results/kv_bfp4.json includes the repaired source hash,
actual runtime dtype/fidelity inventory and the completed CPU-control receipt.
This closes the repeated-reset host-packing defect. Selection remains a separate
quality/performance/memory decision; initial interrupted attempt is not ranked.

The exact production-shape primitive rerun also passes all four ranks, three
nonzero-to-zero cycles and stable addresses. Real-generator/full64 evidence
already used the actual production page shape throughout.
