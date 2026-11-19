# Corrected prefill matmul transpose-address fusion

Source-only candidate; device correctness and speed are not yet established. No production or frozen helper was edited.

## Exact boundary

The fixed baseline is `v6_transport_conv_norm_terminal_scan_integration.py.snapshot`, SHA256 `803685e105d7ef1d46a6f9a8ceafb1758d9b1131e1c8c35aed1ab05cbf1475a5`. Its three applicable producer transposes are the Gram input key (line2057), state-update weighted key (2077), and output-attention key (2334). This probe absorbs only these last-two-axis transposes into raw FP32 tiled reader addresses. It supports all four combinations of transpose flags and arbitrary broadcast leading axes.

The baseline compute and writer bytes, corrected FMA arithmetic, increasing-K loop, per-K FP32 feedback pack/unpack, positive-zero initial accumulator, tile geometry, 64-core ceiling, output DRAM placement, and 40960 bytes/core circular buffers remain identical. No device allocations occur in construction. Operand transposes disappear; neither output shape nor persistent state/cache allocation changes. CPU evidence records the exact six reader substitutions and four class substitutions, with exact inverse-reader restoration and whole-class inverse-AST restoration to the frozen baseline. The runtime candidate has one `generic_op` and no tensor host conversion or transpose call. Source extraction/AST execution exists only in this isolated test's setup, never its measured runtime; a future integrated candidate must embed static definitions.

Physical matrix shapes supply TensorAccessor descriptors. Effective shapes supply the existing broadcast planner. Transposed A page is `(batch*Kt+k/32)*Mt+mr`, lane `(k%32,r)`; transposed B page is `(batch*Nt+nc)*Kt+k/32`, lane `(c,k%32)`. The unchanged reader visits increasing K and copies uint32 payloads. Scratch tile reads remain once per32 K. Existing invalid output-row/column masking and logical K termination remain exact; partial logical sizes have no divisibility restriction.

## Fixtures and gates

`v6_corrected_matmul_transpose_cpu_setup.json` passed 4,576,480 independent raw-address checks, including rectangular64x128/128x64, broadcast2x3, singleton, partial33x65x17, and higher leading-rank cases, across all four flags. Face layout is independently checked as a bijection over1024 positions. This source proof does not substitute for hardware arithmetic checks.

The22 device fixtures include six actual saved fullgraph operand pairs (two chunks, Gram/state-update/output attention, all32 heads),12 seeded random rectangular/broadcast/partial/singleton cases, and four explicit signed-zero/subnormal controls. Actual operands come from pinned CPU normalized keys plus saved device k_beta/weighted_k/v_new/query_scaled of the previously exact fullgraph run. CPU and device tensor file hashes are in the setup JSON. Torch matmul is not an arithmetic oracle here: the unchanged corrected native-transpose adapter is the comparison baseline.

Every case strictly gates FP32 upload roundtrip, native transpose logical bits, final FP32 codes, and BF16 codes. Native transpose itself must retain the finite special controls; if it flushes a subnormal or changes signed zero, the probe stops and records failure rather than silently accepting a new transport policy. Timings include all removed transposes in the baseline and the complete new adapter. They cover chunk0 actual Gram, state update, and output attention with3 warmups,5 eager samples, and5 batches of20 trace replays (100 exact repeated executions for each adapter). Host upload/readback remain outside measured intervals. Timing gains, regressions, and wholegraph behavior remain unmeasured.

## Reproduction

Frozen new test: `tests/v6_corrected_matmul_transpose_probe.py`, SHA256 `e48c3ba24c24be0099c7b45eb2fc0ffeb98a0c90d4181b015e110c8236b9e767`. Pinned Black23.10.1/isort5.13.2/autoflake2.3.1 were applied before this setup evidence. Kernel, baseline extraction, candidate class and executed source snapshots are adjacent to the JSON. No hardware was opened during setup.

```bash
PYTHONPATH=. python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_corrected_matmul_transpose_probe.py \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_corrected_matmul_transpose_cpu_setup_fresh.json

env -u TT_VISIBLE_DEVICES -u TT_METAL_WATCHER -u TT_METAL_DEVICE_PROFILER \
  -u TT_METAL_PROFILER_ENABLE -u TT_METAL_TRACE_ALLOC_TRACKING \
  -u TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE PYTHONPATH=. \
  python_env/bin/python models/autoports/qwen_qwen3_6_35b_a3b/tests/v6_corrected_matmul_transpose_probe.py \
  --run-device --timing --samples 5 --replays 20 \
  --output models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_corrected_matmul_transpose_device.json \
  > models/autoports/qwen_qwen3_6_35b_a3b/doc/fused_decoder/endurance/v6_corrected_matmul_transpose_device.log 2>&1
```

Parent owns serialized hardware access and watcher/profiler separation. Physical chip0 is selected explicitly; `TT_VISIBLE_DEVICES` must be unset. Output artifacts must be fresh. If the isolated adapter passes, the next separate gate is the full chunk output/state wrapper with nonzero initial state at logical lengths127,65,1, preserving padding and complete graph arithmetic. Only that evidence can authorize a standalone model integration; fewer operators alone do not establish a speed improvement.
