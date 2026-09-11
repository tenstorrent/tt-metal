# Native GDN batch-capacity repair

The genuine full-batch allocation32 prefill failed with `num_heads 384 exceeds compute cores 110` in `/tmp/qwen_native_final_slot_lifecycle_b32_s128.log`. Earlier allocation32 checks with `QWEN36_PREFILL_PER_REQUEST=1` ran one active request at a time and did not cover this native batch-capacity boundary.

The native phased scan's `distribute_scan` explicitly requires `B * HV <= grid.x * grid.y` (`ttnn/cpp/ttnn/operations/transformer/chunk_gated_delta_rule/device/chunk_gdn_phased_program_factory.cpp`). Qwen TP4 has 12 local value heads, so this mesh's 110 cores support at most 9 batch rows per native invocation. The native prep stage and fused gated RMSNorm distribute work items across cores with per-core loops; neither requires a further model-side batch split.

`tt/prefill_recurrence.py` now applies the same capacity-aware helper to `flat_forward` and the rank4 adapter's `__call__`. It queries the device grid once during construction. A batch within capacity follows the unchanged native operation call, including the B1 path. Larger batches slice Q/K/V, gates, beta, and initial state along the batch axis; B32 executes groups9+9+9+5. Head-major outputs concatenate on axis0, preserving `[B*HV,T,V]` ordering; final states concatenate on axis0, preserving `[B,HV,K,V]`. All data remains on device. No eager recurrence fallback, host readback, or environment mutation is involved.

## Verification

Both hardware runs used the final default graph (native recurrence, flat QKV where aligned, fused gated norm, prefill block limit8), real Qwen3.8 weights, TP4, **four layers and all32 active batch rows**. They are full-batch lifecycle checks, not full64-layer quality benchmarks. `QWEN36_PREFILL_PER_REQUEST` was explicitly removed from the environment.

- Physical sequence128 exercises the flat native call. Result: `SLOT_LIFECYCLE_EXACT`.
- Physical sequence65 is not divisible by32 and exercises the rank4 native adapter. Result: `SLOT_LIFECYCLE_EXACT`.

Each test starts from distinct nonzero per-slot state and checks exact reset of slots3/17, bit-identical preservation of peers, exact remap, and a subsequent narrowed prefill of slot9 that changes its own convolution/recurrent state while preserving every peer. All three linear layers used fused KDA decode state layout. Both runs had zero reset/remap/narrowing errors and closed the mesh normally.

```bash
timeout 600 env -u QWEN36_PREFILL_PER_REQUEST PYTHONPATH=. HF_HUB_OFFLINE=1 \
  QWEN_AUTOPORT_MODEL_ID=Qwen/Qwen3.8-27B \
  QWEN_AUTOPORT_MODEL_REVISION=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  python_env/bin/python models/autoports/qwen_qwen3_6_27b/tests/slot_lifecycle_b32.py \
  --num-layers 4 --prompt-tokens 128 \
  --output models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/artifacts/native_batched_b32_s128.json
```

Repeat with `--prompt-tokens 65` and output `native_batched_b32_s65.json`. Full logs: `/tmp/qwen_native_batched_b32_s128.log` and `/tmp/qwen_native_batched_b32_s65.log`.

Host tests in `tests/test_prefill_recurrence_batching.py` cover B1, the exact capacity9, capacity+1, B32, batch/head ordering, input/state batch alignment, and preserving original B1 tensor objects. No C++ source changed, so no build is required. This repair does not alter B1 arithmetic; the parent's final B1 accuracy/performance evidence is independent of these larger-batch checks.

Host command: `python_env/bin/python -m pytest models/autoports/qwen_qwen3_6_27b/tests/test_prefill_recurrence_batching.py -q` → **5 passed** in0.89seconds. Black and `git diff --check` passed. No watcher/profiler was enabled in the two lifecycle runs.
