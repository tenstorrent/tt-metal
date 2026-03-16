# MoLE bring up using TTNN

Reference:
https://arxiv.org/abs/2312.06786
https://github.com/RogerNi/MoLE/

## Demo Scripts

### benchmark.py - latency and throughput benchmarking

python -m models.experimental.mole.demo.benchmark \
  --base-model-type dlinear \
  --dataset-name etth1 \
  --num-experts 4 \
  --batch-size 8 \
  --warmup-iterations 2 \
  --measure-iterations 10 \
  --skip-expert-overhead \
  --e2e

### compare.py — linear model vs MoLE comparison

python -m models.experimental.mole.demo.compare \
  --base-model-type dlinear \
  --dataset-name etth1 \
  --num-experts 4 \

### specialization.py — router weights visualization

python -m models.experimental.mole.demo.specialization \
  --base-model-type dlinear \
  --dataset-name etth1 \
  --num-experts 2 \
  --image-path /tmp/mole_router_weights.png
