"""Weight conversion for CosyVoice2-0.5B → TTNN state dicts.

Converts `llm.pt` (Qwen2LM full state dict) into:
  1. Meta-format backbone weights compatible with tt_transformers `Transformer`
  2. CosyVoice-specific speech heads (speech_embedding, llm_embedding, llm_decoder)

Key mapping:
  llm.pt key                              → Meta-format key
  ─────────────────────────────────────────────────────────────
  llm.model.model.embed_tokens.weight     → tok_embeddings.weight
  llm.model.model.norm.weight             → norm.weight
  llm.model.lm_head.weight                → (discarded — replaced by llm_decoder)
  llm.model.model.layers.{i}.self_attn.*  → layers.{i}.attention.* (QKV permuted)
  llm.model.model.layers.{i}.mlp.*        → layers.{i}.feed_forward.*
  llm.model.model.layers.{i}.*_layernorm  → layers.{i}.*_norm
  speech_embedding.weight                 → (carried separately)
  llm_embedding.weight                    → (carried separately)
  llm_decoder.weight / .bias              → (carried separately; used as output head)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import torch

TT_TRANSFORMERS_ROOT = Path(__file__).resolve().parents[3] / "tt_transformers"
if str(TT_TRANSFORMERS_ROOT) not in sys.path:
    sys.path.insert(0, str(TT_TRANSFORMERS_ROOT.parent.parent))

from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format, map_hf_to_meta_keys, split_hf_keys


def load_llm_pt(llm_pt_path: str | Path) -> Dict[str, torch.Tensor]:
    return torch.load(str(llm_pt_path), map_location="cpu", weights_only=True)


def strip_cosyvoice_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip `llm.model.` prefix to get standard HF Qwen2.5 keys.

    Also separates the CosyVoice-specific speech heads (which have no prefix).
    """
    hf_dict = {}
    speech_dict = {}

    for key, tensor in state_dict.items():
        if key.startswith("llm.model."):
            new_key = key[len("llm.model.") :]
            hf_dict[new_key] = tensor
        elif key in ("speech_embedding.weight", "llm_embedding.weight", "llm_decoder.weight", "llm_decoder.bias"):
            speech_dict[key] = tensor
        else:
            raise ValueError(f"Unexpected key in llm.pt: {key}")

    return hf_dict, speech_dict


def convert_qwen_to_meta(
    hf_dict: Dict[str, torch.Tensor],
    head_dim: int = 64,
    n_heads: int = 14,
    n_kv_heads: int = 2,
) -> Dict[str, torch.Tensor]:
    """Convert standard HF Qwen2.5 keys to Meta-format keys with QKV permutation."""
    state_dict = split_hf_keys(hf_dict, n_heads, n_kv_heads)
    state_dict = convert_hf_qkv_to_meta_format(state_dict, head_dim)
    state_dict = map_hf_to_meta_keys(state_dict)
    return state_dict


def load_cosyvoice_llm_weights(
    llm_pt_path: str | Path,
    head_dim: int = 64,
    n_heads: int = 14,
    n_kv_heads: int = 2,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Load llm.pt and return (meta_backbone_dict, speech_heads_dict).

    meta_backbone_dict: Meta-format keys ready for tt_transformers Transformer.
        Includes tok_embeddings.weight, norm.weight, output.weight (Qwen lm_head),
        and all 24 layers.
    speech_heads_dict: CosyVoice-specific tensors:
        speech_embedding.weight  (6564, 896)
        llm_embedding.weight     (2, 896)
        llm_decoder.weight       (6564, 896)
        llm_decoder.bias         (6564,)
    """
    raw = load_llm_pt(llm_pt_path)
    hf_dict, speech_dict = strip_cosyvoice_prefix(raw)
    meta_dict = convert_qwen_to_meta(hf_dict, head_dim, n_heads, n_kv_heads)
    return meta_dict, speech_dict


def build_cosyvoice_state_dict(
    llm_pt_path: str | Path,
    head_dim: int = 64,
    n_heads: int = 14,
    n_kv_heads: int = 2,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """Build the full state dict for CosyVoiceLLM.

    Returns (backbone_state_dict, speech_heads):
      backbone_state_dict: Meta-format keys with `output.weight` replaced by
          llm_decoder.weight (6564×896) so the LMHead produces speech-token logits.
          Also includes llm_decoder.bias under a custom key.
      speech_heads: {speech_embedding.weight, llm_embedding.weight}
    """
    meta_dict, speech_dict = load_cosyvoice_llm_weights(llm_pt_path, head_dim, n_heads, n_kv_heads)

    meta_dict["output.weight"] = speech_dict["llm_decoder.weight"]
    meta_dict["output.bias"] = speech_dict["llm_decoder.bias"]

    speech_heads = {
        "speech_embedding.weight": speech_dict["speech_embedding.weight"],
        "llm_embedding.weight": speech_dict["llm_embedding.weight"],
    }

    return meta_dict, speech_heads


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify CosyVoice LLM weight conversion")
    parser.add_argument("--llm-pt", type=str, default="model_data/cosyvoice2-0.5B/llm.pt")
    args = parser.parse_args()

    meta_dict, speech_heads = build_cosyvoice_state_dict(args.llm_pt)

    print(f"Backbone keys: {len(meta_dict)}")
    print(f"Speech head keys: {list(speech_heads.keys())}")

    expected_backbone = 1 + 1 + 1 + 24 * 12  # tok_emb + norm + output + 24*(q,k,v,o,qb,kb,vb,ln1,ln2,w1,w2,w3)
    layer0_keys = sorted(k for k in meta_dict if "layers.0." in k)
    print(f"\nLayer 0 keys ({len(layer0_keys)}):")
    for k in layer0_keys:
        print(f"  {k}: {list(meta_dict[k].shape)}")

    print(f"\nNon-layer keys:")
    for k in sorted(meta_dict):
        if "layers." not in k:
            print(f"  {k}: {list(meta_dict[k].shape)}")

    print(f"\nSpeech heads:")
    for k, v in speech_heads.items():
        print(f"  {k}: {list(v.shape)}")

    assert meta_dict["output.weight"].shape == (6564, 896), "llm_decoder weight shape mismatch"
    assert meta_dict["output.bias"].shape == (6564,), "llm_decoder bias shape mismatch"
    assert meta_dict["tok_embeddings.weight"].shape == (151936, 896), "embed_tokens shape mismatch"
    assert speech_heads["speech_embedding.weight"].shape == (6564, 896)
    assert speech_heads["llm_embedding.weight"].shape == (2, 896)
    print("\nOK — all shapes verified.")
