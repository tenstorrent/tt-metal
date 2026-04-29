# MASt3R / DUSt3R on Blackhole

End-to-end DUSt3R (and MASt3R-backbone) inference on a single Tenstorrent
**Blackhole p150a** chip via tt-nn. Patch embed, ViT-L encoder, dual-branch
decoder, RoPE100, and the DPT pointmap head all run on device.

## Platforms

Blackhole (p150, p300c)

## Introduction

[**DUSt3R**](https://github.com/naver/dust3r) (Naver, 2024) reformulates
multi-view 3D reconstruction as a direct dense regression problem: given two
RGB views, the network produces a dense 3D pointmap for each view in a shared
camera-1 frame, plus per-pixel confidence. Pose, depth, and reconstruction
follow as downstream optimisations. The architecture is ViT-L encoder + a
dual-branch transformer decoder with cross-attention + DPT regression heads.

[**MASt3R**](https://github.com/naver/mast3r) extends DUSt3R with a learned
matching head; the DUSt3R backbone is the heavy compute. This port
implements the full DUSt3R backbone (encoder, decoder, DPT) on Blackhole.

Reference checkpoint: `naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt`.

## Image-size support

`dust3r_forward(img1, img2, state, device)` accepts any input shape
`(B, 3, H, W)` where:

- `H` and `W` are divisible by `2 × patch_size` (32 for ViT-L/16). The
  factor 2× comes from the DPT chain: `ap3_down` is a stride-2 conv whose
  output must later be upsampled 2× to match an un-resampled tap.
- `img1.shape == img2.shape`.
- Square or non-square; `B ≥ 1`.

Invalid sizes raise `ValueError` at the entry point with a clear message.
Encoder, decoder, and DPT pipelines derive all intermediate dimensions from
weights and input shape — there are no hard-wired token counts or channel
counts.

Verified shapes: 256×256, 384×256, 384×384, 512×320, 512×512, 256×512.

## Files

| File | Role |
|---|---|
| `tt/ttnn_dust3r.py` | TT-NN port: `patch_embed_device`, `encoder_block_device_pre`, `decoder_block_device_pre`, `dpt_head_device`, `full_encoder`, `full_decoder`, `dust3r_forward` |
| `reference/torch_dust3r.py` | Pure-torch reference implementation used for PCC validation |
| `demo.py` | CO3Dv2 demo — runs port + reference on a real image pair, saves a side-by-side visualisation to `media/` |
| `tests/` | Per-layer pytest harness (currently empty; entry point is `test_mast3r.py` at workspace root) |
| `media/` | Sample demo output PNGs |

## How to Run

### Per-layer + end-to-end PCC test

```bash
cd /home/ttuser/experiments/mast3r
source ~/.tenstorrent-venv/bin/activate
python3 test_mast3r.py --layer end_to_end --runs 3 --device-id 0
```

`--layer` accepts `end_to_end`, `full_encoder`, `full_decoder`,
`patch_embed`, `dpt_head`, `dpt_head_2`, `encoder_block_{0..23}`,
`decoder_block_{0..11}`, `decoder2_block_{0..11}`. Outputs grep-friendly
fields: `inference_speed`, `accuracy`, `pcc`, `latency_ms`, `peak_dram`,
`status`.

### Multi-size verification

```bash
python3 test_mast3r_sizes.py
```

Runs `dust3r_forward` at 256×256, 384×256, 512×320 against the torch
reference using the baseline-equivalent combined-PCC criterion (≥0.99).

### CO3Dv2 accuracy + absolute-pose evaluation

```bash
python3 eval_mast3r.py \
    --co3d-root /home/ttuser/experiments/vggt/co3d_data \
    --category apple --pairs 4 --device-id 0
```

Reports per-pair port-vs-reference PCC (full / xyz / conf) on real images,
median depth-relative error, normalised Chamfer, and absolute pose error
vs CO3D ground truth (rot_med, tr_med, RRA, RTA, AUC@30°) via DUSt3R's
2-view PairViewer global aligner. New flags:

- `--img-h H --img-w W` — non-square aspect (each divisible by 32).
  Defaults to `--img-size 512` (square).

### Demo

```bash
python3 models/experimental/mast3r/demo.py \
    --seq 110_13051_23361 --frame-i 0 --frame-j 40 \
    --img-h 512 --img-w 512 --device-id 0
```

Writes `media/demo_<seq>_<i>_<j>_<H>x<W>.png` — a 2×5 panel comparing
input images, reference depth, port depth, reference confidence, and port
confidence — plus a `.txt` caption with PCC + latency.

## Headline numbers

### End-to-end (`test_mast3r.py --layer end_to_end`, 512×512, single p150a)

| Metric | Value |
|---|---:|
| Inference speed | ≈4.4 frames/sec (post warm-up) |
| Combined PCC vs torch reference | 0.9962 |
| Per-head PCC (head1 / head2) | 0.97 / 0.99 |

### CO3Dv2 evaluation (3 scenes × 4 pairs, 512×512)

Real-image port-vs-reference (mean across 12 pairs):

| Metric | mean | min |
|---|---:|---:|
| `pcc_head1_xyz` | 0.9934 | 0.9777 |
| `pcc_head2_xyz` | 0.9962 | 0.9901 |
| depth rel err mean (head1) | 4.1 % | — |
| chamfer normalised (head1) | 0.037 | — |

PairViewer absolute-pose vs CO3D GT (estimated focal):

| Model | rot_med | tr_med | AUC@30 |
|---|---:|---:|---:|
| reference | 24.6° | 32.1° | 35.0 |
| **port (this code)** | **21.7°** | **29.2°** | **38.1** |

Full per-pair table + ablations in `co3d_eval_results.md` (workspace root).

### Multi-size verification (random inputs vs torch reference)

| Size (HxW) | Combined PCC |
|---|---:|
| 512×512 | 0.9962 |
| 384×256 (non-square) | 0.9944 |
| 256×256 | 0.9930 |
| 512×320 (non-square) | 0.9921 |

### Real-data non-default sizes (CO3D apple, scene `110_13051_23361`, 3 pairs)

| Size (HxW) | mean head1_xyz | mean head2_xyz |
|---|---:|---:|
| 384×384 (square, non-default) | 0.9676 | 0.9757 |
| 256×512 (non-square 2:1) | 0.9866 | 0.9895 |

The pretrained checkpoint is `_512_dpt` so absolute accuracy peaks at 512×512;
port-vs-reference correlation stays high at any valid size.

## Known constraints

- **Patch size and DPT chain require `H, W` divisible by 32.** Sizes that
  are divisible by 16 only (e.g. 16×16) are rejected because the DPT's
  stride-2 down-conv would produce an odd output that does not match its
  upsampled tap.
- **PairViewer focal-vote median is unstable for pure-orbit viewpoint
  sequences.** This is a property of the `PairViewer` global aligner (the
  paper's full `PointCloudOptimizer` regularises across pairs); not a port
  bug. Affects scene `189_20393_38136` of CO3D apple in the same way for
  both reference and port.
- **MASt3R matcher head is not yet ported.** This repo implements the
  DUSt3R backbone — sufficient for pointmap, depth, confidence, and
  PairViewer pose. The MASt3R matcher would add learned correspondences
  on top.
