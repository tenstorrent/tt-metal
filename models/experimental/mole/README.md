# MoLE -- Mixture of Linear Experts on Tenstorrent (TTNN)

Time-series forecasting with Mixture-of-Linear-Experts, accelerated on Tenstorrent Wormhole hardware using TTNN.

MoLE Paper: https://arxiv.org/abs/2312.06786
Reference implementation: https://github.com/RogerNi/MoLE/ - this is also the implementation used to train the models used in this demo.

### 1. Run full demo

Runs every demo checkpoint on TTNN and the PyTorch reference, printing latency, throughput, MSE, and MAE.

```bash
python3 -m models.experimental.mole.demo.long_demo
```

Use `--batch-size N` to override the default batch size of 1.

### 2. Benchmark inference performance on a single model

Measures TTNN inference latency (ms) and throughput (sequences/sec) for the selected checkpoint.

```bash
python3 -m models.experimental.mole.demo.benchmark \
  --dataset ETTh1 \
  --base-model-type dlinear \
  --num-experts 4 \
  --batch-size 8 \
  --warmup-iterations 100 \
  --measure-iterations 1000
```

### 3. Visualize router specialization

Generates a PNG showing how each expert is activated across time steps.

```bash
python3 -m models.experimental.mole.demo.specialization \
  --dataset ETTh1 \
  --base-model-type dlinear \
  --num-experts 4 \
  --image-path /tmp/mole_router_weights.png
```

### Available demo models
In total there are 63 example models available in this demo.
For each dataset (ETTh1, ETTh2, ETTm1, ETTm2, ECL, traffic, weather), there are 3 base model types (dlinear, rlinear, rmlp), and 3 expert configurations (2, 4, 8).
