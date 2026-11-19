# P2 top-k boundary resume audit

Source-only audit, 2026-11-16 23:22 UTC. No hardware, benchmark, server request, production edit, or waiver was performed. Inspected tt-metal HEAD `47c0bbd606a691e6111b46c5ee1eb9e01fd423d8`; the shared sampling sources below have no working-tree diff. This short audit is not a two-hour repair attempt.

## Finding

There is no already-implemented wider draw available by changing this model's capability or constructor. The real shared `ttnn.sampling` draw retains exactly 32 candidates per user. Retain `max_device_top_k=32` and the explicit host compatibility route for the unchanged reasoning profile. P2's critical performance ratio remains failed; its request-acceptance check passes.

`doc/eval_readiness/harness_requests.json` and its raw `harness_requests/runs.json` record the exact reasoning request: temperature 1.0, top-p 0.95, top-k 40, seed 42, n=1. Each timing request generates 128 tokens. At 32 users, median per-request inter-token time is 655.373249 ms versus the plain baseline's 432.692670 ms, ratio **1.514639128**, exceeding 1.15. At one user it is 72.205207 versus 53.956283 ms, ratio **1.338216859**. Neither run reports request errors. Predicted routing names top-k40 above the declared 32; the recorded reasoning intervals have host sampling. Checkpoint defaults remain top-k20, temperature1, top-p0.95, within the device boundary.

## Source proof of the boundary

- `tt/serving_capabilities.py:33` declares 32. `tt/generator.py:105` constructs the canonical `SamplingGenerator` with `max_top_k=32`; lines 234–235 reject stochastic k>32 without explicit host compatibility; line 250 builds the capped device parameter tensor. Removing these guards would not add the missing kernel capability.
- `models/common/sampling/tt_sampling.py:304` accepts a configurable candidate-gather width, and its local top-k/gather pipeline can produce more than 32 input candidates. That is not the final draw width. Its final call at approximately line 1470 still invokes the same `ttnn.sampling` operation into `tt_out_tok`.
- `ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp:469` fixes `nearest32_K=32`, with `logk=5`; the ensuing `top_k` produces `Kt=1` candidate tile before masking, temperature, and softmax. Wider gathered input does not change these constants.
- `.../kernels/dataflow/writer_interleaved.cpp:140` waits for one local-value and one local-index tile. Lines 152–162 address each user's first 16-candidate face and second 16-candidate face. At k=40 the second scan extends 24 entries, past that user's 16 entries into unrelated tile positions; it does not address a second candidate tile. Subsequent probability accumulation and index selection reuse this layout. This is a correctness hazard, not merely slower sampling.
- `models/common/sampling/tt_sampling.py:222` explicitly documents the two-face bound and out-of-row reads. Its normalization clamps only to the configured `max_top_k`; changing that number to 64 would remove this guard while leaving the native 32-candidate draw unchanged.
- `.../sampling_device_operation.cpp` validates candidate input width and its power-of-two tile count, but no runtime tensor-value validation establishes a wider k. A successful construction or larger input tensor is therefore not evidence that k40 works.

The reduction sampling implementation is the only categorical token-sampling operation located in the operation tree. Wider `ttnn.topk` and the Blackhole large-index top-k composite are selection primitives, not a complete compatible stochastic draw.

## Smallest plausible implementation, untried hand-off

A real widening belongs in the shared native sampling implementation and the full-model canonical generator. It would need a supported draw-width contract, correct global top-64 retention, multi-tile mask/softmax and writer indexing, matched buffer/production/consumption counts, and deterministic seed behavior. The existing native top-k merge code itself would require review for K>32; changing its constant alone is not a justified implementation. Only after a wider support test demonstrates valid token IDs, nonzero reachability of ranks 33–40, exclusion of ranks above k, top-p/temperature behavior, and traced in-place feedback could the generator and capability move to 64. This would invalidate sampling evidence and require affected gates again. No such implementation or probe is already present in this model's saved stage evidence.

No adapter-only sampler, host argmax, field clamp, forced-host seeded policy, altered P2 profile, or claimed capability64 is recommended. Existing host acceptance preserves the exact harness request but does not meet the performance ratio.

## Repair-time and waiver audit

The existing work log does **not** substantiate an honest two-hour repair attempt for this P2 top-k boundary. Its 19:35:11 entry starts P2 collection; its 19:53:47 entry reports the failure and explicitly says: “No2hourP2repairboxwasclaimed orspent aftercap, no waiveradded; widercanonicaldevicecandidate support is hand-offwork.” Prior AutoFix entries investigate thinking budgets, host-sampler equivalence, and constrained qualitative divergence. Those are different failures and cannot be counted as time spent repairing this P2 boundary. The roughly 19-minute P2 collection is measurement, not evidence of a two-hour fix effort. Elapsed time while another long-context job runs is also not repair effort.

The selected skill's `references/eval-readiness-requests.md` says “Spend at most 2 h fixing one probe,” then record a limitation. Its actual `scripts/check_eval_readiness.py:161–180` only waives a matching failure with existing evidence and either `original_model == "same"` or `fix_hours >= 2.0`. This TT device/host routing overhead is not established as the original HF model's behavior. Thus neither `original_model: same` nor `fix_hours: 2` is supported by the existing record, and no automatic waiver is currently justified. The owner's cap does not turn that missing evidence into a completed repair box.

Verdict: exact capability blocker established; no safe configuration-only widening; P2 acceptance passes, P2 32-user performance failure remains work or an explicitly authorized unresolved hand-off. Preserve its failed result rather than relabeling it.
