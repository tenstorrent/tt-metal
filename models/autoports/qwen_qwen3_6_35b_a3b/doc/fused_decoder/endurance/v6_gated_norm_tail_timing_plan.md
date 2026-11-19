# Complete normalization adapter timing handoff

The parent completed [v6_gated_norm_tail_fusion_device.json](v6_gated_norm_tail_fusion_device.json): all14 cases are bit-exact against the explicit native path at normalized FP32, weighted FP32 carrier, inverse, and final BF16 output, with a clean device close. The frozen helper SHA remains `47d50c370b8038bc1b10f24f86c5e58f61222e21b6cc7a1bb22d6318055961bd`.

The new [timing probe](../../../tests/v6_gated_norm_tail_timing_probe.py) compares complete adapters using `GatedNormTail(debug=False)`. Its frozen SHA is `c0af8685423e6688a3362aa04a99c6db4d66999bebe1a5f9a890dc000f75994c`. [v6_gated_norm_tail_timing_cpu_final.json](v6_gated_norm_tail_timing_cpu_final.json) passed CPU setup on all14 fixtures without importing TTNN. The source snapshot and exact inputs/gamma are saved alongside it. No device was accessed while preparing this harness; no timing benefit is claimed.

Both adapters take a persistent FP32 tiled recurrent core `[1,1,32,128]` in interleaved L1 and a persistent BF16 tiled Z `[1,1,4096]` in interleaved DRAM. Both return the same merged BF16 tensor `[1,1,4096]`, padded `[1,32,4096]`, as the existing decoder immediately before its output projection.

Timing includes:

- Core BF16 cast, reshape and widening; FP32 squares and the unchanged native sum.
- The complete existing SiLU-table lookup, including its bitcast/layout/gather operations.
- Either every explicit normalization-tail operation or the fused tail, including all three candidate placement copies.
- The existing final head reshape/sharding and `nlp_concat_heads_decode` conversion to the shared consumer boundary.

Setup uploads, reference readbacks, output readbacks and the subsequent output projection are excluded equally. The full decoder still needs its own traced comparison after integration; this microbenchmark alone does not satisfy the decoder performance gate.

Before timing, the harness checks the six saved positions128/129/130/137/181/3271 and eight seeded random cases. It uses the captured pre-BF16 recurrent core where available, preserving the real first rounding boundary. Fused and explicit outputs must be bit-exact; CPU-reference comparisons are reported separately. The timing input is position3271 by default.

Each variant receives five uncaptured dispatch warmups, one capture, five warmed trace replays, then five samples of100 nonblocking replays each. Synchronization occurs before and after each100-replay group; reported per-replay latency is elapsed wall time divided by100. Outputs are read before and after the timed groups, with exact reference and deterministic-output assertions. Only one trace is live at a time. The replay-owned output is acknowledged as intentionally overwritten; persistent inputs/setup constants remain live, and the trace is released in a `finally` block. The JSON reports every sample, median/min/max, output shape/memory config, speedup/delta, correctness and device-close state.

```bash
# CPU-only setup; choose a new output name for reruns.
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_gated_norm_tail_timing_probe --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_gated_norm_tail_timing_cpu_final.json
# Parent-owned, watcher/profiler disabled, after serial device ownership is clear.
python_env/bin/python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_gated_norm_tail_timing_probe --run-device --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_gated_norm_tail_timing_device.json
```

Defaults are `--replays 100 --samples 5 --warmup 5 --position 3271`. `--order fused-first` permits a reverse-order follow-up if variation or thermal drift warrants one. The harness rejects fewer than 100 replays, refuses to overwrite evidence, and requires watcher/profiler off for timing. Integration and any keep/reject decision remain contingent on these device results and the unchanged decoder gates.
