# Opt-in BFP8 real-weight epilogue integration

Implemented `single_step_shared_qk_epilogue` with unchanged BFP8 weights/KV
and FP32 recurrent state. It consumes raw recurrence output, fuses tilization,
gated RMSNorm and z multiply at B16/B32, and uses persistent caller-owned
output buffers. B1/B8 retain the existing epilogue; prefill remains native.
The separate config is `precision_single_step_shared_qk_epilogue_bfp8_all.json`.
No default or frozen active performance/eval snapshot was changed.

Preflight: 470 CPU tests and 40 subtests pass, one unrelated test skipped;
the real hardware test collects successfully. The queued physical test covers
B32/B16/B8/B1, native/fused/native epilogue controls, 64 recurrent updates,
per-rank FP32 reference checks and bit-identical state/raw/projected outputs.
Timing uses five trace samples with 30 replays each and a 3% control-drift gate.
This is one real-weight GDN block including projection/convolution/epilogue,
not full-model accuracy, throughput or serving qualification.

Persistent unit `qwen38-gdn-epilogue-layer-v1-20261009.service`, invocation
`f5344bf767584289b32a55ca2c335271`, PID2993870 was active waiting for the shared
`/tmp/tt-device.lock`. Two-hour service bound, 90-minute capture including
lock wait, 20-minute pytest bound, 48GiB/8CPU. Survives disconnect, not reboot.
Source, launch, preflight and queue snapshots are retained. Hardware outcomes
must be collected separately; this launch is not a pass receipt.
