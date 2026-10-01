# EXAONE-4.5 Vision Tower (on-device)

TT port of the EXAONE-4.5 vision encoder (Qwen2.5-VL-style ViT: 28 blocks,
hidden 2048, **GQA 32 Q / 8 KV heads**, head_dim 64, window attention 112 with
full-attention blocks [6, 13, 20, 27], 2x2 spatial merger -> 5120).

The attention, block, MLP, RMSNorm and host-side preprocessing are imported
from `models/demos/qwen25_vl` (the HF `Exaone4_5_VisionModel` is a modular copy
of the Qwen2.5-VL tower; both TT implementations read every size and head count
from `VisionModelArgs`). Only the EXAONE-specific pieces live here:

- `tt/model_config.py` — GQA: `vision_n_kv_heads` from
  `vision_config.num_key_value_heads`, plus grids sized for hidden 2048 / out 5120.
- `tt/model.py` — GQA-aware fused-qkv split (an equal three-way split would
  silently corrupt the weights; guarded by an assertion in
  `DropInVisionTransformer`) and the `Exaone4_5_VisionModel` reference wiring
  (its `forward` returns `BaseModelOutputWithPooling`; `pooler_output` holds
  per-image merged embeds).
- `tt/patch_merger.py` — loads and applies the HF merger MLP **biases**, which
  the qwen25_vl merger drops.

head_dim 64 is tile-aligned, so Qwen's 80->96 head padding paths are inert here.
The tower is replicated per device (no TP); patch_embed and window/rope index
computation stay on host, window structure reaches the device SDPA via
`cu_window_seqlens` (block-diagonal mask synthesized in-kernel).

Validated on P150x8 (2026-08-25, real weights): attention PCC 0.9997, merger
0.9995, all 28 blocks >= 0.995, full tower (DropInVisionTransformer) 0.991-0.9993,
e2e tower-vs-HF 0.9984 inside the text demo.

Run tests:
```bash
export HF_MODEL=LGAI-EXAONE/EXAONE-4.5-33B MESH_DEVICE=P150x8
pytest models/experimental/exaone45_vl/tests/
```

End-to-end image chat (vision on device + text TP=8):
```bash
python models/tt_transformers/demo/exaone_45_vision_hybrid.py --vision-device tt
```
