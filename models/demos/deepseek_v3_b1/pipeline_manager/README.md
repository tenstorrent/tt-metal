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

Launch the Python pipeline in `--launch-only` mode with prefix `deepseek_v3_b1`, then run:

```bash
models/demos/deepseek_v3_b1/pipeline_manager/build-standalone/deepseek_pipeline_loopback_harness \
  --h2d-socket-id deepseek_v3_b1_h2d \
  --d2h-socket-id deepseek_v3_b1_d2h \
  --page-size-bytes 64 \
  --iterations 8 \
  --initial-token 1 \
  --connect-timeout-ms 30000
```
