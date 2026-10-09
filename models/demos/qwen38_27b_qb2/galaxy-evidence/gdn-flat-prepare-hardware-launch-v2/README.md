# Direct GDN preparation hardware experiment

Queued persistently under `qwen38-gdn-flat-prepare-hardware-v2-20261009.service`,
invocation `c6713def66f8497c9e695154edb2ddbe`. It waits for the shared
`/tmp/tt-device.lock`; it does not interrupt the running full-model sweep.

CPU preflight: 472 passed, 40 subtests passed, one skipped. Physical test
collection passed. Nine geometries cover B16/B32, one/32 token rows, L1/DRAM
inputs, plus B1. The test checks all four ranks against the existing FP32
normalization and native packing, independent allocations and changed-input
trace replay. Timing uses native/fused/native brackets. Exp stays native and
outside both timed paths. No model or precision policy selects this prototype.

The first staging attempt passed CPU tests but failed physical test collection
because its copied source lacked `flat_prepare.py`. It accessed no hardware.
v2 includes both prototype files explicitly; the original failed preflight is
retained. This follows the earlier simulator's unsupported SETDVALID failure,
which yielded no completed correctness comparison.

Bounds: 2-hour service, 90-minute capture including lock waiting, 20-minute
pytest deadline after lock acquisition, 48 GiB host RAM and eight CPU cores.
The process survives disconnect, not reboot. Hardware correctness, speedup and
full-model integration remain pending at launch capture.
