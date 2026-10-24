# qwen38_27b bring-up breadcrumbs

## C.linear.attn_norm (layer 0)

- `tt/norm.py:TtRmsNorm`: one fused `ttnn.rms_norm` on the local [1, 1, 640, 5120] bf16 rows (SP 8 on mesh axis 0,
  replicated on axis 1), no collective. The `1 + w` gamma is added in fp32 at load and replicated as a
  [1, 1, 1, 5120] fp32 TILE tensor. HiFi4 + fp32 dest (`NORM_FIDELITY`, `NORM_FP32_ACC`, `NORM_WEIGHT_DTYPE` in
  `tt/settings.py`).
- A ROW_MAJOR [1, 1, 160, 32] gamma overflowed L1 (1.85 MB of CBs): a row-major affine turns off the op's streaming
  kernel for a TILE input. A TILE gamma fixes it.
- `bringup/hooks.py`: `device_component` builds the module from the checkpoint's `input_layernorm.weight`. The host
  boundary (`_rows_to_device` / `_rows_to_host`) splits rows over mesh axis 0 and reads column 0 of each mesh row.
  `_ops_visible` runs the call with `ttnn.CONFIG.enable_fast_runtime_mode = False` for that call only, because
  FastOperation never calls ttnn's pre-operation hooks. Without it the gate's device proof counts 0 ops.
- Gate: PCC 0.999998, rel 0.0020 vs CPU (limit 0.0064), auto_margin_min 3.21, 1 device op, determinism
  bit-identical. Re-run: `bfw job check --task C.linear.attn_norm --gate-only`.
