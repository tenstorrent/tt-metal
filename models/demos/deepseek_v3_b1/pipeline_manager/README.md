# DeepSeek V3 B1 Pipeline Manager

This folder contains a bare-bones C++ `PipelineManager` class and a loopback harness
for exported DeepSeek V3 B1 H2D/D2H sockets.

## Build

```bash
bash models/demos/deepseek_v3_b1/pipeline_manager/build.sh
```

- `build-standalone/libdeepseek_v3_b1_pipeline_manager.a`
- `build-standalone/deepseek_pipeline_loopback_harness`

## Test

Example single-GLX launch:

```bash
TT_METAL_SLOW_DISPATCH_MODE=1 tt-run --rank-binding bh_4x2_multi_mesh_rank_binding.yaml \
  python -m models.demos.deepseek_v3_b1.demo.cli \
  --cache-path /mnt/models/deepseek-ai/cache-2026-03-09/ \
  --max-new-tokens 1024 \
  --weights synthetic \
  --dense-layer-id-override 0 \
  --moe-layer-id-override 3 \
  --launch-only
```

Then run:

```bash
models/demos/deepseek_v3_b1/pipeline_manager/build-standalone/deepseek_pipeline_loopback_harness \
  --h2d-socket-id deepseek_v3_b1_h2d \
  --d2h-socket-id deepseek_v3_b1_d2h \
  --page-size-bytes 64 \
  --iterations 8 \
  --initial-token 1 \
  --connect-timeout-ms 30000
```
