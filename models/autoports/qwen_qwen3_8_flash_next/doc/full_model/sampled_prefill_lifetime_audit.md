# Sampled-prefill allocation lifetime repair

`serving_probe_v3.json/log` fails in the first heterogeneous sampled-prefill
control, before sampling replay. The tracker reports one live unsafe device
buffer,346470, with operation context `ttnn.allocate_tensor_on_device`.
The caller supplied the three-token prompt; the stack ends at
`QwenGenerator.prefill_forward -> _sample -> execute_trace`. Earlier real-model
feedback, inactivity, page-table and trace-retention checks in that artifact
passed. The caller then closed the mesh normally.

Source inspection finds a concrete ownership defect: the terminal prefill
branch constructs `device_rows`, holding each last-logit slice, then copies the
concatenated rows into persistent `sample_logits`. It calls `_sample` while
`device_rows` still owns those temporary tensors. A nonaligned tiled slice
(here rows2:3) can allocate storage through the native slice layout conversion.
The temporary data is unnecessary after the copy, yet its live allocation can
overlap a previously captured sampler trace's reserved address range.

The narrow repair deletes `device_rows` immediately after the copy and before
sampling. It preserves persistent input/output addresses and command order;
it adds no host readback, host sampling, synchronization, tracker exemption or
postcapture persistent allocation. Python scope already releases the temporary
concat result after the copy call. The explicit deletion releases the retained
row slices as well. The same command queue orders the copy before replay.

The original log did not enable allocation tracebacks, so buffer346470's exact
allocation stack/referrer is not available. The source defect and repair are
established; assigning that buffer ID to the row slice remains a diagnosis to
confirm with the serialized serving rerun. The synthetic sampler-only pass
does not refute an allocator-overlap defect in a different model/trace memory
layout. Do not weaken allocation tracking if the rerun exposes another owner.

CPU validation, without importing TTNN:

```bash
python_env/bin/python -m unittest \
models.autoports.qwen_qwen3_8_flash_next.tests.test_prefill_sampling_lifetime_cpu
```

All4 tests pass. They extract the actual terminal branch from `generator.py`
and verify temporary lifetime at sampling entry for one nonaligned prompt and
reordered mixed rows, plus the persistent logits-only return. A negative
control removes the deletion and must detect the original live owner. This
proves Python ownership and unchanged return routing, not hardware allocator
safety. Real-device confirmation passes in `serving_probe_v4.json/log` (all16 checks,
including sampled prefill and later replay/lifecycle) under
`TT_METAL_TRACE_ALLOC_TRACKING=1` and
`TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0`.
