# PatchTST (TTNN)

## Overview
This is a public PatchTST demo for TTNN on Wormhole.

It covers:
- forecasting on real benchmark data,
- classification and regression with real labeled datasets,
- optional channel-attention checkpoints,
- traced batch inference,
- shared-encoder forecast + classification inference,
- online forecasting with a cached rolling context,
- long-context and high-channel runs with task-matched local checkpoints.

## Environment
Build and use the in-repo runtime:

```bash
./create_venv.sh --force --env-dir ./python_env
./build_metal.sh
source ./python_env/bin/activate
export PYTHONPATH=.
```

## Assets
```bash
python -m models.demos.wormhole.patchtst.scripts.download_assets \
  --dataset-root data/patchtst \
  --checkpoint-root models/demos/wormhole/patchtst/.hf_checkpoints
```

## Generate Local Task Checkpoints
Some public runs use local task-matched checkpoints.

```bash
./python_env/bin/python -m models.demos.wormhole.patchtst.demo.demo finetune \
  --task forecast \
  --channel-mode attention \
  --dataset etth1 \
  --dataset-root data/patchtst \
  --steps 0 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --seed 1337 \
  --output-dir models/demos/wormhole/patchtst/artifacts/finetune/forecast_attention_etth1_ckpt

./python_env/bin/python -m models.demos.wormhole.patchtst.demo.demo finetune \
  --task classification \
  --dataset heartbeat_cls \
  --dataset-root data/patchtst \
  --steps 20 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --seed 1337 \
  --output-dir models/demos/wormhole/patchtst/artifacts/finetune/classification_heartbeat_ckpt

./python_env/bin/python -m models.demos.wormhole.patchtst.demo.demo finetune \
  --task regression \
  --dataset flood_modeling1_reg \
  --dataset-root data/patchtst \
  --steps 20 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --seed 1337 \
  --output-dir models/demos/wormhole/patchtst/artifacts/finetune/regression_flood_modeling1_ckpt

./python_env/bin/python -m models.demos.wormhole.patchtst.demo.demo finetune \
  --task forecast \
  --dataset etth1 \
  --dataset-root data/patchtst \
  --context-length 4096 \
  --prediction-length 96 \
  --steps 200 \
  --batch-size 4 \
  --learning-rate 1e-4 \
  --seed 1337 \
  --output-dir models/demos/wormhole/patchtst/artifacts/finetune/forecast_long_context_etth1_4096_ckpt

./python_env/bin/python -m models.demos.wormhole.patchtst.demo.demo finetune \
  --task forecast \
  --dataset traffic \
  --dataset-root data/patchtst \
  --context-length 512 \
  --prediction-length 96 \
  --steps 5 \
  --batch-size 2 \
  --learning-rate 1e-4 \
  --seed 1337 \
  --output-dir models/demos/wormhole/patchtst/artifacts/finetune/forecast_high_channel_traffic_ckpt

./python_env/bin/python -m models.demos.wormhole.patchtst.demo.demo finetune \
  --task forecast \
  --dataset heartbeat_cls \
  --dataset-root data/patchtst \
  --context-length 309 \
  --prediction-length 96 \
  --steps 40 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --seed 1337 \
  --output-dir models/demos/wormhole/patchtst/artifacts/finetune/forecast_heartbeat_multitask_ckpt

./python_env/bin/python -m models.demos.wormhole.patchtst.demo.demo finetune \
  --task classification \
  --dataset heartbeat_cls \
  --dataset-root data/patchtst \
  --context-length 309 \
  --steps 80 \
  --batch-size 8 \
  --learning-rate 1e-4 \
  --seed 1337 \
  --checkpoint-id-override models/demos/wormhole/patchtst/artifacts/finetune/forecast_heartbeat_multitask_ckpt \
  --checkpoint-revision-override local-generated \
  --output-dir models/demos/wormhole/patchtst/artifacts/finetune/classification_heartbeat_multitask_ckpt
```

## Public Demo Commands
Single forecast run:

```bash
./python_env/bin/python -m models.demos.wormhole.patchtst.demo.demo run \
  --task forecast \
  --dataset etth1 \
  --dataset-root data/patchtst
```

Cached online forecast:

```bash
./python_env/bin/python -m models.demos.wormhole.patchtst.demo.demo streaming \
  --dataset etth1 \
  --dataset-root data/patchtst \
  --trace \
  --streaming-mode cached \
  --stream-steps 4
```

Shared-encoder forecast + classification run:

```bash
./python_env/bin/python -m models.demos.wormhole.patchtst.demo.demo run \
  --task multi_task \
  --dataset heartbeat_cls \
  --dataset-root data/patchtst \
  --context-length 309
```

## Verification
Unit tests:

```bash
PYTHONPATH=. ./python_env/bin/python -m pytest models/demos/wormhole/patchtst/tests/unit -s --timeout=600
```

Integration tests:

```bash
PYTHONPATH=. ./python_env/bin/python -m pytest models/demos/wormhole/patchtst/tests/integration -s --timeout=600
```

Performance and runtime checks:

```bash
PYTHONPATH=. ./python_env/bin/python -m pytest models/demos/wormhole/patchtst/tests/perf -s --timeout=1200
```

## Notes
- The primary forecast workload uses `etth1`, `context_length=512`, `prediction_length=96`.
- The default runtime is strict no-fallback.
- The traced batch-1 and batch-16 rows are the main latency and throughput measurements.
- The cached streaming path keeps rolling context state and exact sliding-window normalization statistics, but it does not claim decoder-style autoregressive KV-cache semantics.
- The shared-encoder multi-head path is forecast + classification only.
- The long-context run uses the `etth1` training split because the public test split is shorter than `4096 + 96`.
- Generated outputs belong under `generated/patchtst/`.

Device recovery:

```bash
tt-smi -r 0
```
