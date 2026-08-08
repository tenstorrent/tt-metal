# DINO-5scale Swin-L in TTNN — Implementation Summary

Quick reference for bringing up this model in TTNN.

## Model

**DINO-5scale Swin-L** = Swin-L backbone → 5-scale neck (256-d) → 6-layer deformable encoder (5 levels) → query selection → 6-layer decoder (self + deformable cross-attn) → cls + bbox heads. Input 1333×800; 80 COCO classes; 900 queries.

## Reuse from tt-metal

| Component | Source | Change for DINO |
|-----------|--------|-----------------|
| **Backbone** | `swin_s` / `swin_v2` | Swin-S: return 4 stage maps (no classifier). Swin-L: extend to embed_dim=192, num_heads=[6,12,24,48], window_size=12. |
| **MS deformable attn (5 levels)** | BEVFormer `tt_ms_deformable_attention.py` or UniAD encoder/decoder | num_levels=5, embed_dims=256, num_heads=8. |
| **Encoder** | UniAD `ttnn_detr_transformer_encoder.py` | num_levels=5, feedforward_channels=2048; map 36e checkpoint keys. |
| **Decoder** | UniAD `ttnn_detr_transformer_decoder.py` | Same config; map decoder.layers.*. |
| **MHA, FFN** | UniAD `ttnn_mha.py`, `ttnn_ffn.py` | Use as-is; map ffn.layers.0.0 / ffn.layers.1. |

## New in TTNN

- **5-scale neck:** ChannelMapper — 4× 1×1 conv+GN (192/384/768/1536→256) + P6 conv.
- **Query selection:** Top-K from encoder → ref points + content queries.
- **DINO heads:** 7× cls (256→80), 6× reg with box refine.

## Bring-up order

1. PyTorch reference (mmdet) + save intermediates for PCC.
2. Backbone (Swin-S 4-stage or Swin-L).
3. Neck → encoder → decoder → heads + query select.
4. Full pipeline; then optimization.

## Effort

- **Low-res + Swin-S:** ~1–2 weeks.
- **Full Swin-L + 1333×800:** ~3–4 weeks.
