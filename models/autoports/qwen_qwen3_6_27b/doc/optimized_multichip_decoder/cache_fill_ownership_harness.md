# Executable full-layer ownership coverage

`tests/probe_cache_fill_ownership.py` is prepared for the reviewed integration, but has not opened devices. CPU syntax compilation and `--help` pass. It imports the new ownership helper at execution time and will fail explicitly until integration is installed.

Example after integration (use a fresh tag/output):

```
TT_METAL_WATCHER=10 MULTICHIP_JOB_TIMEOUT=1200 models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/run_job.sh cache_owner_b2 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python -m models.autoports.qwen_qwen3_6_27b.tests.probe_cache_fill_ownership --batch 2 --length 33 --layer 3 --reference models/autoports/qwen_qwen3_6_27b/doc/multichip_decoder/correctness_v3_ref_l3_b2_s33.pt --policy-file models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/tiled_conv7_l3_policy.json --output models/autoports/qwen_qwen3_6_27b/doc/optimized_multichip_decoder/cache_owner_b2.json
```

Select the current policy file; the harness enables `batched_prefill_cache`. The reference must match synthetic seed-17 `run_multichip_decoder --baseline` metadata exactly, including frozen optimized source. If the listed existing reference does not match, generate a fresh stage-local oracle with that baseline runner; do not weaken metadata checks. B32 can use the matching delivery reference. No performance claims come from this instrumented harness.

Coverage is actual TP4 full attention with initial bulk dispatch required and unchanged 0.995 whole-output/contribution/per-user/KV gates for prefill and following decode. Exact comparisons to the same layer's original ordered path cover private bulk, exposed disjoint table, shallow/deep aliases mutating an aliased table, external replacement, fresh private disjoint recertification, fresh aliased uncertified replacement, cumulative resumes at 1/31/32/33, and captured prefill replay after public table exposure plus input and disjoint/aliased table changes. Complete physical caches are compared, including unused pages. Physical allocations remain equal even if an aliased table stops referring to the highest physical page. Deepcopy supplies a memo for cache tensors (retaining cache storage intentionally), isolating the table-lease lifetime contract without requiring unsupported copies of device buffers.

All trace buffers, bank inputs, metadata copies and ordered reference executions are prepared before capture; capture must record ordered calls while the original state lease is still private. The state is exposed only after capture, then all three mutable-table cases replay twice. This test does not claim bulk prefill trace support or correctness after replacing a captured buffer address. Focused skip-page/reversed-user watcher probes remain complementary required coverage.
