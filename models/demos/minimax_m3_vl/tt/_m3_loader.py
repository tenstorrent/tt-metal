# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Main-env-safe weight loader for the MiniMax-M3-VL vision tower + projector.

Reads only the two vision-bearing safetensors shards (`vision_tower.*`,
`multi_modal_projector.*`, `patch_merge_mlp.*`) directly via `safetensors`
— NO transformers dependency, so it works in the repo's transformers-4.53
env. Applies the checkpoint(4.52.4)→v5.12 key remap that was validated
against the real model (see memory `minimax-m3-vl-port`), and builds a
small torch reference-module tree with M3-native submodule names that the
ttnn `from_torch` builders and PCC tests consume.

Note: this builds torch modules purely to hold weights for `from_torch`;
the *reference activations* come from goldens generated in the
transformers-5.12 venv (`tests/gen_goldens.py`), not from running these.
"""
from __future__ import annotations

import os
from typing import Dict

import torch
from safetensors import safe_open

VISION_SHARDS = ("model-00026-of-00059.safetensors", "model-00059-of-00059.safetensors")
_WANT = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")


def _remap(key: str) -> str:
    """Checkpoint (transformers 4.52.4) key -> v5.12 model key layout."""
    if key.startswith("patch_merge_mlp."):
        return "multi_modal_projector.merge_" + key[len("patch_merge_mlp.") :]
    if key.startswith("vision_tower.vision_model."):
        rest = key[len("vision_tower.vision_model.") :]
        rest = rest.replace("embeddings.patch_embedding", "embeddings.proj")
        rest = rest.replace("encoder.layers.", "layers.")
        return "vision_tower." + rest
    return key  # multi_modal_projector.linear_* unchanged


def load_vision_state_dict(snapshot_dir: str, dtype: torch.dtype = torch.float32) -> Dict[str, torch.Tensor]:
    """Return {v5.12_key: tensor} for vision_tower + projector, remapped from the checkpoint.

    The Conv3d patch-embed weight (1280,3,2,14,14) is flattened to the
    per-patch linear form (1280, 1176).
    """
    sd: Dict[str, torch.Tensor] = {}
    for shard in VISION_SHARDS:
        p = os.path.join(snapshot_dir, shard)
        if not os.path.exists(p):
            raise FileNotFoundError(f"vision shard not found: {p}")
        with safe_open(p, framework="pt") as f:
            for k in f.keys():
                if k.startswith(_WANT):
                    sd[_remap(k)] = f.get_tensor(k).to(dtype)
    # Flatten the Conv3d patch embed to a Linear weight.
    pe = "vision_tower.embeddings.proj.weight"
    if pe in sd and sd[pe].dim() == 5:
        out_ch = sd[pe].shape[0]
        sd[pe] = sd[pe].reshape(out_ch, -1).contiguous()  # (1280, 3*2*14*14=1176)
    return sd


def _linear(sd, prefix, bias=True) -> torch.nn.Linear:
    w = sd[f"{prefix}.weight"]
    lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=bias)
    with torch.no_grad():
        lin.weight.copy_(w)
        if bias:
            lin.bias.copy_(sd[f"{prefix}.bias"])
    return lin.eval()


def _layernorm(sd, prefix, eps) -> torch.nn.LayerNorm:
    w = sd[f"{prefix}.weight"]
    ln = torch.nn.LayerNorm(w.shape[0], eps=eps)
    with torch.no_grad():
        ln.weight.copy_(w)
        ln.bias.copy_(sd[f"{prefix}.bias"])
    return ln.eval()


class _Attn(torch.nn.Module):
    def __init__(self, sd, p):
        super().__init__()
        self.q_proj = _linear(sd, f"{p}.q_proj")
        self.k_proj = _linear(sd, f"{p}.k_proj")
        self.v_proj = _linear(sd, f"{p}.v_proj")
        self.out_proj = _linear(sd, f"{p}.out_proj")


class _MLP(torch.nn.Module):
    def __init__(self, sd, p):
        super().__init__()
        self.fc1 = _linear(sd, f"{p}.fc1")
        self.fc2 = _linear(sd, f"{p}.fc2")


class _Layer(torch.nn.Module):
    def __init__(self, sd, p, eps):
        super().__init__()
        self.layer_norm1 = _layernorm(sd, f"{p}.layer_norm1", eps)
        self.layer_norm2 = _layernorm(sd, f"{p}.layer_norm2", eps)
        self.self_attn = _Attn(sd, f"{p}.self_attn")
        self.mlp = _MLP(sd, f"{p}.mlp")


class _Projector(torch.nn.Module):
    """multi_modal_projector: per-patch linear_1/2, then merge_linear_1/2 (after 2x2 concat)."""

    def __init__(self, sd):
        super().__init__()
        self.linear_1 = _linear(sd, "multi_modal_projector.linear_1")
        self.linear_2 = _linear(sd, "multi_modal_projector.linear_2")
        self.merge_linear_1 = _linear(sd, "multi_modal_projector.merge_linear_1")
        self.merge_linear_2 = _linear(sd, "multi_modal_projector.merge_linear_2")


class VisionReference(torch.nn.Module):
    """Torch module tree holding M3-VL vision weights (for ttnn from_torch / shape reference)."""

    def __init__(self, args):
        super().__init__()
        sd = load_vision_state_dict(args.snapshot_dir)
        eps = args.layer_norm_eps
        # patch embed as a Linear (1176 -> 1280, no bias)
        self.patch_embed = _linear(sd, "vision_tower.embeddings.proj", bias=("vision_tower.embeddings.proj.bias" in sd))
        self.pre_layrnorm = _layernorm(sd, "vision_tower.pre_layrnorm", eps)
        self.layers = torch.nn.ModuleList(
            _Layer(sd, f"vision_tower.layers.{i}", eps) for i in range(args.num_hidden_layers)
        )
        self.projector = _Projector(sd)
        self.eval()


def build_reference(args) -> VisionReference:
    return VisionReference(args)
