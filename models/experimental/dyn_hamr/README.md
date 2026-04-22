# Dyn-HaMR on Blackhole p150

Port of the Dyn-HaMR (CVPR 2025 Highlight) hand mesh recovery model to a single
Tenstorrent p150 Blackhole chip.  The NPU-resident component is the HaMeR
per-frame neural regressor, which dominates per-frame latency in Dyn-HaMR; the
temporal optimization pass is CPU-bound numerics and stays on the host.

Target neural network:

- Input: `(B, 3, 256, 192)` RGB hand crop.
- Backbone: ViT-H/16 (32 blocks, 1280 embed dim, 16 heads, MLP ratio 4,
  `qkv_bias=True`) producing `(B, 1280, 16, 12)` feature map.
- Head: cross-attention `TransformerDecoder` (1-token query over 192 context
  tokens) followed by three linear readouts for 16×6-D pose, 10-D shape and
  3-D weak-perspective camera.
- Post-NN: 6D → rotation matrix conversion, MANO forward kinematics
  (778 vertices / 21 joints).

## Layout

```
dyn_hamr/
  reference/    # torch-only reference (strips lightning/yacs/timm deps)
  tt/           # tt-nn port
  tests/        # pytest benchmark harness — emits inference_speed + accuracy
```

Reference source lives at `/home/ttuser/experiments/dyn-hamr/reference/Dyn-HaMR`
(upstream `ZhengdiYu/Dyn-HaMR` + `geopavlakos/hamer`); those trees are *not*
vendored into `tt-metal`.

## Benchmark

`pytest -s models/experimental/dyn_hamr/tests/test_dyn_hamr.py`
prints two key lines that the outer loop grep-parses:

- `inference_speed: <float> frames/sec`
- `accuracy: <float>` (Pearson correlation × 100, vs. torch reference on CPU)

The harness compares the tt-nn forward pass against the torch reference with
matching random weights; both are seeded to keep PCC deterministic. Real
checkpoint weights are swapped in once the port is end-to-end.
