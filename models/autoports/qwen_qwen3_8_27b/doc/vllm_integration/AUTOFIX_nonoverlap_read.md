# AutoFix: nonoverlap raw token readback

## Starting evidence

The parent's live nonoverlap vLLM run failed on 2026-09-14 at 00:43:19 UTC,
EngineCore PID 643904. The investigator read the traceback from
`readiness_vllm/server.log` before its planned archival. The relevant chain was:

```text
model_runner._forward_with_model_input
  -> async_decode.finalize_decode
  -> Qwen38ForCausalLM.process_decode_output_host
  -> ttnn.to_torch
  -> pytensor.cpp:300: buffers.size() == 1
Can't convert a tensor distributed on MeshShape([1, 4]) mesh ...
```

The runner checkout is based on vLLM
`03fa3af2e15b5f8dc07cbaa67d92f979aa00be11`, with the parent's integration changes.
This investigator performed source inspection and host tests only. The parent
owns the adapter implementation change and all server/device reruns.

## Diagnosis

In `plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_runner.py`,
`_forward_with_model_input` submits decode with
`read_from_device=False, async_read=False` and immediately calls
`finalize_decode`. Consequently, `async_decode.py:632–637` does not call the
adapter's `read_decode_output`, and `finalize_decode` calls
`process_decode_output_host` with the raw device tensor at lines 672–677.

Before the fix, the adapter passed that tensor directly to `ttnn.to_torch`.
`ttnn/ttnn/operations/core.py` first transfers device storage to host without
choosing one replica; the subsequent conversion in
`ttnn/cpp/ttnn-nanobind/pytensor.cpp:295–305` requires exactly one host buffer
unless a mesh composer is supplied. The TP4 token feedback tensor has multiple
replicas, explaining the specific assertion.

The async path supplies different input to the same processor: its prior
`read_decode_output` has selected `get_device_tensors(tt_out)[0]` and submitted
that replica's CPU transfer. The prefill adapter similarly performs its own
read before processing. Their successful conversion does not prove the raw
nonoverlap path works.

This is an output-interface contract bug, separate from page-table growth or
model numerical correctness.

## Smallest repair and focused verification

The parent added the device-storage check to `process_decode_output_host`:
device input is passed through the existing synchronous `read_decode_output`
before conversion. Already-host output proceeds directly to `to_torch`. This
reuses the existing first-replica token read and its readback counter. It does
not require gathering or composing the whole replicated tensor.

Two focused host regressions were added to `tests/test_vllm_adapter_host.py`:

- A raw device sentinel must select only replica zero, call `cpu(blocking=True)`
  exactly once, convert only the resulting host sentinel, and increment
  `token_readbacks` once. It must not read the other replicas or record an async
  event. The result must be the expected unpadded INT64 token column.
- An already-host sentinel must be converted without calling
  `read_decode_output` or incrementing the readback counter again.

Both invoke the actual adapter methods while replacing TTNN effects with mocks;
they do not create a device or device tensor. The expanded suite passed:

```bash
source ../run-env.sh
PYTHONPATH="$VLLM_ROOT:$PYTHONPATH" python_env/bin/python -m models.autoports.qwen_qwen3_8_27b.tests.test_vllm_adapter_host
```

Result: **12 tests passed in 0.004 seconds**. A temporary in-memory copy of
`process_decode_output_host` with the new device-read branch removed produced
exactly one assertion failure in the raw-device regression and no test errors.
The missing boundary is therefore observable by the new test; no implementation
file was modified for this negative control.

## Corrected device control

The initial `tests/check_vllm_adapter.py` called `read_decode_output` before
processing output in both its control and steady paths. That design tested
fresh versus stale decode inputs but bypassed the exact nonoverlap failure.

The control now follows the plugin boundary:

```text
decode_forward(read_from_device=False)
  -> process_decode_output_host(raw_device_output, is_tokens=True)
  -> update the next step's host token state
```

There is no caller-side token read or async event in this control. It still
requires exactly one token readback per step through the processor. The steady
path retains the explicit asynchronous first-replica read, event wait, and host
processing split. Only that path retains host snapshots for later ownership
checks; the control must never retain and later reprocess the mutable raw device
feedback buffer. Output JSON records which boundary each path exercised.

Syntax compilation and Black formatting checks passed for both test files.
The revised device runner has **not been executed by this investigator**.
No build is required for these Python tests and this markdown report.

## Final status

The source diagnosis matches the live traceback, and host tests verify the
parent's minimal repair, including a negative control. Hardware verification is
pending: rerun the revised device control and the original nonoverlap vLLM
request, then verify the async path still performs exactly one read per output.
This report is not a serving or device-correctness pass.
