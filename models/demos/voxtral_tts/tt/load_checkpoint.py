"""
Load and preprocess weights from consolidated.safetensors for Voxtral-4B-TTS-2603.

Weight key namespaces in consolidated.safetensors:
  layers.{0-25}.*                            — text decoder backbone (26 layers)
  norm.weight                                — final RMSNorm of text backbone
  mm_audio_embeddings.tok_embeddings.weight  — text token embeddings (tied lm_head)
  mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight
                                             — audio token embeddings [9088, 3072]
  acoustic_transformer.*                     — flow-matching module (semantic head + ODE transformer)
  audio_tokenizer.decoder_blocks.*           — Voxtral Codec decoder (4-stage upsampler)
  audio_tokenizer.output_proj.*              — codec output conv
  audio_tokenizer.quantizer.*               — VQ codebook (for semantic dequantization)
"""

from pathlib import Path

import torch
from safetensors import safe_open


def _fuse_weight_norm(g: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Fuse PyTorch weight_norm parametrization: w = g * v / ||v||."""
    out_ch = v.shape[0]
    v_norm = v.view(out_ch, -1).norm(dim=1).view(out_ch, *([1] * (v.dim() - 1)))
    return g * v / v_norm


def _fuse_codec_conv_weights(raw: dict) -> dict:
    """Replace weight_norm parametrizations with fused weights in codec dict."""
    fused = {}
    skip = set()

    for k in raw:
        if k.endswith(".parametrizations.weight.original0"):
            base = k[: -len(".parametrizations.weight.original0")]
            g = raw[k]  # magnitude
            v = raw[base + ".parametrizations.weight.original1"]  # direction
            fused[base + ".weight"] = _fuse_weight_norm(g, v)
            skip.add(base + ".parametrizations.weight.original0")
            skip.add(base + ".parametrizations.weight.original1")
        elif k in skip:
            continue
        else:
            fused[k] = raw[k]

    return fused


def load_state_dict(checkpoint_path: str | Path) -> dict[str, torch.Tensor]:
    """
    Load consolidated.safetensors and return a clean state dict.

    Weight_norm parametrizations in the codec are fused into plain weights.
    The semantic VQ codebook embeddings are materialized from running stats.
    """
    checkpoint_path = Path(checkpoint_path)
    raw = {}
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            raw[key] = f.get_tensor(key)

    fused = _fuse_codec_conv_weights(raw)

    return fused


def load_voice_embeddings(model_dir: str | Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load preset voice embeddings from voice_embedding/*.pt."""
    model_dir = Path(model_dir)
    voices = {}
    voice_dir = model_dir / "voice_embedding"
    for pt_file in sorted(voice_dir.glob("*.pt")):
        name = pt_file.stem  # e.g. "casual_male"
        voices[name] = torch.load(pt_file, map_location=device, weights_only=True)
    return voices


def get_text_decoder_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract text backbone weights (layers.* + norm + embeddings)."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("layers.") or k == "norm.weight":
            out[k] = v
        elif k.startswith("mm_audio_embeddings."):
            out[k] = v
    return out


def get_acoustic_transformer_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract acoustic flow-matching transformer weights."""
    return {
        k.removeprefix("acoustic_transformer."): v
        for k, v in state_dict.items()
        if k.startswith("acoustic_transformer.")
    }


def get_codec_decoder_state(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract codec decoder weights (fused weight_norm already applied)."""
    out = {}
    for k, v in state_dict.items():
        if k.startswith("audio_tokenizer."):
            out[k.removeprefix("audio_tokenizer.")] = v
    return out


def get_semantic_codebook(state_dict: dict[str, torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """
    Materialize the VQ semantic codebook from EMA running stats.

    The VQ codebook stores:
      embedding_sum  [8192, 256]  — sum of encoder outputs assigned to each entry
      cluster_usage  [8192]       — count of assignments

    Codebook vectors = embedding_sum / cluster_usage (safe division).
    """
    prefix = "audio_tokenizer.quantizer.semantic_codebook."
    emb_sum = state_dict[prefix + "embedding_sum"]  # [8192, 256]
    usage = state_dict[prefix + "cluster_usage"]  # [8192]
    usage_safe = usage.clamp(min=eps).unsqueeze(-1)
    return emb_sum / usage_safe  # [8192, 256]
