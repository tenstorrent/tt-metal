# MoLE bring up using TTNN

Reference:
https://arxiv.org/abs/2312.06786
https://github.com/RogerNi/MoLE/

## Demo Scripts

### benchmark.py

```
python -m models.experimental.mole.demo.benchmark \
  --base-model-type dlinear \
  --dataset-name etth1 \
  --num-experts 4 \
  --batch-size 8 \
  --warmup-iterations 2 \
  --measure-iterations 5 \
  --e2e
```

### compare.py — baseline vs MoLE comparison

```
python -m models.experimental.mole.demo.compare \
  --base-model-type dlinear \
  --dataset-name etth1 \
  --num-experts 4 \
  --steps 80 \
  --batch-size 16 \
  --eval-batch-size 32
```

### specialization.py — router weights visualization

```
python -m models.experimental.mole.demo.specialization \
  --base-model-type dlinear \
  --dataset-name etth1 \
  --num-experts 2 \
  --image-path /tmp/mole_router_weights.png
```

## Tests

```
pytest models/experimental/mole/tests/pcc/ -x -q
```
