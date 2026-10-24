# mimo_v26_flash_rl bring-up breadcrumbs

## C.attn_dense.attn_norm (layer 0)

- `tt/rms_norm.py` `RMSNorm`: stock `ttnn.rms_norm`, weight `[1, 1, 1, 4096]` bf16 TILE replicated, eps from config
  (1e-6), plain `w` (not `1 + w`). Compute config from `tt/settings.py` `NORM_FIDELITY` (HiFi4) / `NORM_FP32_ACC`
  (True); both are in MAX_PRECISION.
- Runs on each chip's local residual rows `[1, 1, 160, 4096]` (plan: SP row r = tokens r*640.., TP column c = rows
  c*160.. of those). The all_gather over TP before qkv is not part of the norm: the attention module owns it.
- `tt/mesh_io.py`: host <-> mesh layout of the residual (`ShardTensor2dMesh` dims (0, 1) over [8, 4] of
  x.reshape(8, 4, 160, 4096)); used only by the component adapters in `bringup/hooks.py`.
- `bringup/hooks.py` runs the component adapter with `ttnn.CONFIG.enable_fast_runtime_mode = False` (`_op_hooks`):
  in fast runtime mode (the C++ default on this tt-metal) ttnn skips its pre-operation hooks, so the framework's
  device proof counted 0 ops (`device_ops_attn_norm = 0`) although the op ran. Reuse `_op_hooks` for every component.
- Gate: PCC 0.999998, rel vs CPU 0.00198 (limit 0.0061), auto_margin_min 2.83, 1 ttnn op, host FLOPs 0, determinism ok.
- Re-run: `bfw job check --task C.attn_dense.attn_norm --gate-only`.
