# BEVFormer Encoder — PyTorch Reference

This directory holds the PyTorch reference implementation the TTNN port in
[../tt/](../tt/) is validated against. The module hierarchy and op inventory below were
captured with `torch.inspect` on the reference encoder and are the authoritative
description of what the TTNN side has to reproduce.

## Files

| File | Contents |
| --- | --- |
| [encoder.py](encoder.py) | `BEVFormerEncoder`, `BEVFormerLayer` |
| [spatial_cross_attention.py](spatial_cross_attention.py) | `SpatialCrossAttention` |
| [temporal_self_attention.py](temporal_self_attention.py) | `TemporalSelfAttention` |
| [ms_deformable_attention.py](ms_deformable_attention.py) | `MSDeformableAttention` |
| [point_sampling_3d_2d.py](point_sampling_3d_2d.py) | 3D→2D reference-point projection and per-camera visibility masks |

## Inspected configuration

The dump below is for the nuScenes-shaped default: 1600×900 input images, 6 cameras,
4 FPN levels at strides 8/16/32/64, a 50×50 BEV grid, `embed_dims=256`, 6 layers.

That fixes the shapes that appear throughout:

- **2500** — BEV queries, `50 × 50`.
- **30125** — flattened multi-level camera feature tokens per camera:
  `200·113 + 100·57 + 50·29 + 25·15`.
- **447** — `max_len` for this input. Spatial cross-attention rebatches BEV queries
  per camera into a padded bucket sized by the largest per-camera hit count, so the
  deformable attention inside it runs on `[6, 447, 256]` rather than on all 2500.
  This value is data-dependent: it changes with the camera rig and the projection
  matrices, so treat it as illustrative, not as a fixed shape.
- **6** — call counts equal to the layer count; the encoder shares no weights across
  layers.

## Module structure

```
BEVFormerEncoder(
  num_layers=6, embed_dims=256, num_heads=8, num_levels=4, num_points=4, num_cams=6, pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], num_points_in_pillar=4, z_cfg={'num_points': 4, 'start': -5.0, 'end': 3.0}
  (layers): ModuleList(
    (0-5): 6 x BEVFormerLayer(
      (temporal_self_attention): TemporalSelfAttention(
        (deformable_attention): MSDeformableAttention(
          embed_dims=256, num_heads=8, num_levels=1, num_points=4, im2col_step=64, batch_first=True
          (value_proj): Linear(in_features=256, out_features=256, bias=True)
          (sampling_offsets): Linear(in_features=256, out_features=64, bias=True)
          (attention_weights): Linear(in_features=256, out_features=32, bias=True)
          (output_proj): Linear(in_features=256, out_features=256, bias=True)
        )
      )
      (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      (spatial_cross_attention): SpatialCrossAttention(
        (deformable_attention): MSDeformableAttention(
          embed_dims=256, num_heads=8, num_levels=4, num_points=4, im2col_step=64, batch_first=True
          (value_proj): Linear(in_features=256, out_features=256, bias=True)
          (sampling_offsets): Linear(in_features=256, out_features=256, bias=True)
          (attention_weights): Linear(in_features=256, out_features=128, bias=True)
          (output_proj): Linear(in_features=256, out_features=256, bias=True)
        )
        (output_proj): Linear(in_features=256, out_features=256, bias=True)
      )
      (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
      (ffn): Sequential(
        (0): Linear(in_features=256, out_features=1024, bias=True)
        (1): ReLU(inplace=True)
        (2): Linear(in_features=1024, out_features=256, bias=True)
      )
      (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
    )
  )
)
```

## Op inventory

Columns: module/op, input shapes, output shapes, number of calls per encoder forward.

| Op | Inputs | Outputs | Calls |
| --- | --- | --- | --- |
| `BEVFormerEncoder` | `[1, 2500, 256]`, `[6, 30125, 1, 256]`, `[6, 30125, 1, 256]`, `[1, 2500, 256]`, `[4, 2]`, `[4]` | `[2500, 256]` | 1 |
| `BEVFormerLayer` | — | `[2500, 256]` | 6 |
| `TemporalSelfAttention` | — | `[2500, 256]` | 6 |
| `SpatialCrossAttention` | — | `[2500, 256]` | 6 |
| `MSDeformableAttention` (temporal) | — | `[2500, 256]` | 6 |
| `MSDeformableAttention` (spatial) | — | 6 × `[447, 256]` | 6 |
| `LayerNorm` | `[1, 2500, 256]` | `[2500, 256]` | 18 |
| `Linear` (`attention_weights`, temporal) | `[1, 2500, 256]` | `[2500, 32]` | 6 |
| `Linear` (`sampling_offsets`, temporal) | `[1, 2500, 256]` | `[2500, 64]` | 6 |
| `Linear` (`value_proj` / `output_proj`, BEV-side) | `[1, 2500, 256]` | `[2500, 256]` | 18 |
| `Linear` (FFN up) | `[1, 2500, 256]` | `[2500, 1024]` | 6 |
| `Linear` (FFN down) | `[1, 2500, 1024]` | `[2500, 256]` | 6 |
| `Linear` (`attention_weights`, spatial) | 6 × `[6, 447, 256]` | 6 × `[447, 128]` | 6 |
| `Linear` (`sampling_offsets` + `output_proj`, spatial) | 6 × `[6, 447, 256]` | 6 × `[447, 256]` | 12 |
| `Linear` (`value_proj`, spatial) | 6 × `[6, 30125, 256]` | 6 × `[30125, 256]` | 6 |
| `ReLU` | `[1, 2500, 1024]` | `[2500, 1024]` | 6 |
| `Sequential` (FFN) | `[1, 2500, 256]` | `[2500, 256]` | 6 |

The encoder inputs, in order: BEV queries, camera features, camera features for the
previous frame, previous BEV, `spatial_shapes` `[4, 2]`, `level_start_index` `[4]`.

The only op that touches the full camera feature volume is the spatial `value_proj`:
`[6, 30125, 256] → [6, 30125, 256]`, i.e. 180 750 rows against the 2500 rows every
BEV-side `Linear` sees. It is by far the largest matmul in a layer.
