
## C.sliding_moe.attn_norm (layer 0)

- `tt/norm.py:TtRmsNorm`: one local `ttnn.rms_norm` per chip (the full 2880 hidden is on every chip, so no
  collective), gamma bf16 ROW_MAJOR [1, 1, 90, 32] replicated, DRAM, bf16 out. Compute config from settings
  NORM_FIDELITY (HiFi4) and NORM_FP32_ACC (True), both in MAX_PRECISION. The same class serves ffn_norm (hooks map
  `ffn_norm` to post_attention_layernorm.weight) and can serve final_norm.
- `tt/layout.py`: harness-boundary helpers only. `rows_to_device` splits [S, H] by sequence over axis 0 (S/8 rows
  per chip row) and replicates it over axis 1; `rows_to_host` reads column 0 of each row back.
- `bringup/hooks.py:device_component` sets `ttnn.CONFIG.enable_fast_runtime_mode = False`: in fast mode ttnn's
  FastOperation skips PRE_OPERATION_HOOKS, and the device proof counted 0 ops (device_ops_attn_norm = 0).
- Gate: pcc 0.999998, auto_margin_min 3.21 (closest: vs_cpu rel 0.0019 of limit 0.0061), device_ops 1, host
  FLOPs 0, deterministic.
- Re-run: `bfw job check --task C.sliding_moe.attn_norm`.
