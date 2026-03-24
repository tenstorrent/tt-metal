# MoLE bring up using TTNN

Reference:
https://arxiv.org/abs/2312.06786
https://github.com/RogerNi/MoLE/

## Demo scripts - example usage

### benchmark.py - latency and throughput benchmarking

python -m models.experimental.mole.demo.benchmark \
  --base-model-type dlinear \
  --dataset-name etth1 \
  --num-experts 4 \
  --batch-size 1 \
  --warmup-iterations 2 \
  --measure-iterations 100 \
  --e2e \
  --skip-expert-overhead

### compare.py — linear model vs MoLE prediction accuracy comparison

python -m models.experimental.mole.demo.compare \
  --base-model-type dlinear \
  --dataset-name etth1 \
  --num-experts 4

### specialization.py — router weights visualization

python -m models.experimental.mole.demo.specialization \
  --base-model-type dlinear \
  --dataset-name etth1 \
  --num-experts 2 \
  --image-path /tmp/mole_router_weights.png

## Scope

Implemented here:
- TTNN forward path for MoLE inference
- Supported base models: dlinear, rlinear, and rmlp
- Configurable expert counts
- TSLib dataset-backed benchmark, comparison, and router-visualization demos

## Tests
Device PCC against the PyTorch reference:

python -m pytest models/experimental/mole/tests/pcc/test_e2e_mole.py -q