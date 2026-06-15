---
name: kimi-prefill-env-vars
description: Env vars required to run Kimi K2.6 chunked prefill block/transformer tests + runner
metadata: 
  node_type: memory
  type: project
  originSessionId: 66f1de8d-b7ca-4fa2-84c2-bbb3793dd8fa
---

Running Kimi K2.6 chunked prefill (block + transformer tests, and the prefill runner) requires these env vars:

```bash
export KIMI_K2_6_HF_MODEL=/data/nbabin/Kimi-K2_6-dequantized
export TT_KIMI_PREFILL_TTNN_CACHE=/mnt/models/Kimi-K2_6-Cache/Kimi-K2_6-Cache-prefill
export TT_KIMI_PREFILL_HOST_REF_CACHE=/mnt/models/kimi-prefill-cache/golden
# bug-fix path also needs the DS vars pointed at the Kimi caches:
export TT_DS_PREFILL_TTNN_CACHE="$TT_KIMI_PREFILL_TTNN_CACHE"
export TT_DS_PREFILL_HOST_REF_CACHE="$TT_KIMI_PREFILL_HOST_REF_CACHE"
```

Plus a Kimi golden trace dir via `KIMI_PREFILL_TRACE_DIR` (default in model_variants.py is
`/mnt/models/kimi-prefill-cache/golden/kimi_k2_6_chunked_trace`, which did not exist as of 2026-06-11;
user is supplying the real path). Tests `pytest.skip` if the trace dir is absent.

Related: [[kimi-chunked-prefill-work-state]]. Kimi uses GateComputeMode.HOST_ALL; block test uses MoE
layer 1 (Kimi has NUM_DENSE_LAYERS=1). Branch: ppopovic/chunked_prefill_runner_integration_rebased.

IMPORTANT (2026-06-11): the above exports are for the TESTS. The RUNNER (prefill_runner.py) needs only
`PREFILL_MODEL_VARIANT=kimi_k2_6` — all cache/HF-config paths have correct variant defaults for the 8x4
BH box (verified). See [[kimi-chunked-prefill-work-state]] "RUNNER RUN MODES + MINIMAL ENV" for run modes
(standalone / standalone-chunked-PCC / request-loop) and the producer connect-timeout caveat.
