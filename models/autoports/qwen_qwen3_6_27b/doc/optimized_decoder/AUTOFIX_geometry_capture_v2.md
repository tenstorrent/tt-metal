# Geometry capture lifetime repair

`geometry106.log` failed before any geometry row during the captured decode's MLP down projection. The call stack is `decode_forward -> _finish -> _linear(mlp.down_proj.weight) -> ttnn.linear`. `dataflow_buffer.cpp:3088` reports a live L1 allocation beginning at 1,189,696 overlapping a static CB region ending at 1,312,448 on cores `[0-0 - 6-9]`. The device subsequently closed cleanly. This is evidence about the diagnostic harness's allocation lifetime, not a failed attention geometry.

The original capture hook saved references to both attention activations and all four FP32 partial-output tensors. Those are L1 allocations whose normal lifetimes end when the corrected attention projections return. Holding them through the remainder of the decoder raises the live L1 footprint during the MLP down program. The original probe also retained the prior decode result until assignment of the next result, and retained the last result throughout the sweep.

The immutable replacement is `tests/run_attention_geometry_probe_v2.py`. Inside the untimed `ttnn.linear` interception, it immediately snapshots the actual activation and FP32 result to host. It keeps only host snapshots, the existing DRAM weight object, and non-tensor program/config metadata. It returns the original runtime output so the decoder executes unchanged. Each decode result/input is released before the next step. After capture, setup uploads the exact captured BF16 activation; this upload remains outside timing. It preserves the original native C32/R4/B1 program/config and all prequantized weight equality checks.

The geometry plan, measured projection function, trace measurement function, and metric/error recorder are AST-identical to v1. The original measured source remains unchanged. The host-only `candidates/check_geometry_capture_v2_cpu.py` extracts and executes the new hook against tensor sentinels, verifying that activation/output objects are absent from saved data and the returned runtime tensor is unchanged. It also checks identical geometry plans and records both source hashes in `geometry_capture_v2_cpu.json`. Syntax and the hardware-free plan pass. No hardware run was performed for this repair.

Retry with the same environment/policy and a fresh output path:

```sh
QWEN_DECODER=optimized QWEN_OPT_CANDIDATE=grouped96 \
python_env/bin/python -m models.autoports.qwen_qwen3_6_27b.tests.run_attention_geometry_probe_v2 \
  --key packed_gdn --key linear_attn.out_proj.weight \
  --output models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/geometry_v2_results.json \
  > models/autoports/qwen_qwen3_6_27b/doc/optimized_decoder/geometry_v2_results.log 2>&1
```

The first successful native row and exact same-input metrics will validate the repaired lifetime on device. Any later large-block L1 error belongs to its recorded geometry row; the failed v1 capture does not reject those candidates.
