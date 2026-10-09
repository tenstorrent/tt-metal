# Refreshed native BFP8 baseline

Completed October 9 at 23:08 UTC; one hardware test passed and the device closed
cleanly. One TP4 replica, BFP8 weights/KV, FP32 recurrent state, native recurrence,
128 output tokens, one warmup plus three measured fresh-prefill repetitions.
HTTP overhead is excluded. This is the first arm of the live
native/shared-QK/native comparison, not its final comparison result.

| Context | Batch | Output tok/s/user | Output tok/s per TP4 |
| --- | --- | --- | --- |
| 32K | 32 | 7.343 | 234.975 |
| 32K | 16 | 11.754 | 188.059 |
| 16K | 32 | 8.040 | 257.282 |
| 16K | 16 | 12.646 | 202.328 |

Raw samples, output hashes, capacity, prefill timing and source hashes are in
`sweep.json`; its original relative host path is preserved below this directory.
Do not describe eight-times scaling as measured Galaxy throughput.
