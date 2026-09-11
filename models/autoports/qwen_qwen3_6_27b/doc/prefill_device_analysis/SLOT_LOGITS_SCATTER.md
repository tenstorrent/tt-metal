# Slot-logits scatter hypothesis and focused probe

**Verified in the standalone TP4 probe.** Widening one BF8 vocabulary row to 32
fixed slots constructs and packs the 31 zero rows on the host. Replacing that
with a device zero fill and repeated-row concat reduced median scatter time
from 176–185 ms to 0.5–1.0 ms, with exact logical outputs on every device.
Actual full-model/HTTP improvement remains for parent integration and validation.
A separate 20 ms empty narrow-view measurement does not explain the full
difference between B1 generator and B32 adapter prefill.

The measured original `tt/generator.py:_scatter_slot_logits` creates one or two `ttnn.zeros` blocks,
concatenates them around the real row, and deallocates the blocks and consumed
input. The batch-1 path returns the exact input object immediately.
`_sampling_logits` is currently an identity function, so this boundary uses the
model's local vocabulary width before any sampler-internal padding.

The cached Qwen3.8 checkpoint at revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` has vocabulary **248,320**. The model's
TP4 padding rule gives **62,080 columns per rank**, not 65,536. The probe derives
this value from the actual cached config and records that config's hash.

Source chain:

- `ttnn/cpp/ttnn/operations/creation/creation.cpp:203–210`: BF8 `full_impl`
  creates a logical-volume FP32 vector, calls `Tensor::from_vector`, and uploads
  the result to the device. `zeros` delegates to this path.
- `ttnn/core/tensor/tensor.cpp:193–199` delegates host construction to
  `host_tensor_from_vector_with_pad_value`.
- `tt_metal/impl/tensor/host_tensor_factory.cpp:92–111` encodes tiled physical
  data and then converts it to the requested BF8 dtype.
  `tt_metal/impl/tensor/tensor_impl.cpp:146` allocates the physical padded vector.
- For shape `[1,31,1,62080]`, the logical FP32 zero data is **7,697,920 bytes**;
  the physical FP32 representation is **246,333,440 bytes** across those rows
  before BF8 packing. Both counts are recorded in the artifact.
  Splitting into head and tail changes allocation count but retains 31 rows.
- In contrast, `creation.cpp:232–244` sends `zeros_like` on a tiled BF8 device
  tensor with matching output dtype to the device `fill` kernel. The existing
  multi-request assembly in `tt/generator.py:238–248` already repeats one zero
  tensor in its concat list.

The isolated candidate therefore creates one `zeros_like(logits)` row and
concatenates 32 references, placing the real row at the active slot. It retains
the baseline's consumed-input behavior and batch-1 identity return. No
production method or global operation is patched by the probe. After the
successful measurement, the coordinating agent integrated this exact candidate
into the generator and began separate serving validation.

[probe_slot_logits_scatter.py](probe_slot_logits_scatter.py) compiles the exact
baseline method from its AST and binds both methods to private owner
instances. It loads no weights. A shared random finite BF8 input has distinct
rank-local patterns; each invocation receives a private device clone. For slots
0, 7 and 31, it compares every logical host row on every rank against both the
quantized canonical input/zero oracle and the original method. It verifies
canonical buffer ownership and values stay intact, while the transferred clone
is consumed. Batch-1 checks cover identity, stable owners and unchanged program
count; the host-only check forbids any TTNN attribute access on that path.

Timing uses two discarded warmups and three alternating A/B samples per slot.
Each sample includes the scatter method and device completion. Input cloning,
host readback, owner checks and final output deallocation are excluded equally.
Warmed samples must not grow the program cache. There is no model trace,
allocation-tracker debug overhead or device profiler. CPU Torch uses 8 threads.

Host preparation completed without importing Torch or TTNN. After integration,
use the preserved baseline from the original measurement:

```bash
python3 models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/probe_slot_logits_scatter.py \
  --inspect-source \
  --baseline-artifact models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/artifacts/slot_logits_scatter.json \
  --output models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/artifacts/slot_logits_scatter_replay_prepared.json
```

The coordinating agent granted exclusive hardware after the preceding server
closed. The original bounded command completed successfully; its invocation is
preserved in the result artifact. To reproduce the same comparison after the
production edit, use this command in a newly granted hardware window:

```bash
env TT_METAL_HOME=/home/mvasiljevic/tt-metal \
  PYTHONPATH=/home/mvasiljevic/tt-metal \
  TT_METAL_TRACE_ALLOC_TRACKING=0 TT_METAL_DEVICE_PROFILER=0 \
  timeout 120 python_env/bin/python \
  models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/probe_slot_logits_scatter.py \
  --baseline-artifact models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/artifacts/slot_logits_scatter.json \
  --output models/autoports/qwen_qwen3_6_27b/doc/prefill_device_analysis/artifacts/slot_logits_scatter_repeat.json
```

## Measured result, 2026-09-11

| Active slot | Original median | Device-fill median | Static scatter speedup |
| --- | ---: | ---: | ---: |
| 0 | 175.957 ms | 0.506 ms | 347.5× |
| 7 | 184.647 ms | 0.997 ms | 185.2× |
| 31 | 175.891 ms | 0.521 ms | 337.8× |

Each median uses three warmed samples with alternating arm order. Program-cache
entries were stable at 4, 5 and 6 for slots 0, 7 and 31 respectively. No trace
capture, weights, profiler or allocation-tracker debug checks were involved.

Every logical row matches the quantized input/zero oracle on all four ranks;
the candidate also matches the baseline exactly for each slot. All 31 inactive
rows are zero. The four input ranks contain distinct values and their canonical
owners and bytes remain unchanged. Both methods consume only their private
input clone. Batch 1 returns the exact input object with no TTNN access in the
host contract check, and device ownership/program count remain unchanged.

The run exited 0 and cleanly closed every device at **15:29:24.363 UTC**.
Hardware was returned immediately; this agent ran no full model or server and
made no production edit. These results verify the isolated host-zero cost and
support integrating the exact candidate; the large static speedup is not a
full-model or HTTP speedup claim.

- [Device results, baseline source and source hashes](artifacts/slot_logits_scatter.json)
- [Preserved complete run log](artifacts/qwen_slot_logits_scatter.log)
- [Host preparation and batch-1 contracts](artifacts/slot_logits_scatter_prepared.json)
- [Preserved-baseline preparation and provenance check](artifacts/slot_logits_scatter_replay_prepared.json)

By default the harness reads the current production method at invocation.
`--baseline-artifact` instead compiles the preserved `baseline_method` from a
prior measured or prepared JSON. It records that artifact's path and SHA256,
the historical generator hash, and the method hash. It rejects using that input
artifact as the output path, preserving the original measurement. The saved
baseline was host-validated after the option was added; no additional device
run was needed or performed.
