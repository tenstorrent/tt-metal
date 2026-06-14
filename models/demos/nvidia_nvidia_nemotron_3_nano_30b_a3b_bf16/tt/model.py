# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""NemotronH-30B full model forward pass using TTNN components.

Layer pattern (52 layers):
  M = Mamba2 block   (SSM only — norm + Mamba2 mixer + residual)
  E = MoE MLP block  (norm + gate + 128 routed experts + shared expert + residual)
  * = Dense attention (norm + GQA attention + residual, no RoPE in HF source)

Weight loading is lazy: weights are fetched from safetensors shards on first
access and cached in a dict so each shard is opened at most once.
"""

import json
import os

import torch

from .dense_attention import dense_attention_forward
from .embedding import embedding_forward
from .layer_norm import layer_norm_forward
from .lm_head import lm_head_forward
from .mamba2_layer import mamba2_layer_forward
from .moe_experts import moe_experts_forward
from .moe_gate import moe_gate_forward
from .shared_expert import shared_expert_forward

SNAP = (
    "/home/ttuser/.cache/huggingface/hub/"
    "models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16/snapshots/"
    "cbd3fa9f933d55ef16a84236559f4ee2a0526848"
)
PATTERN = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
N_LAYERS = 52


class WeightCache:
    """Lazy-loading weight cache backed by safetensors shards."""

    def __init__(self, snap: str = SNAP):
        idx_path = os.path.join(snap, "model.safetensors.index.json")
        with open(idx_path) as f:
            self._idx = json.load(f)["weight_map"]
        self._snap = snap
        self._shards: dict = {}

    def _shard(self, filename: str):
        if filename not in self._shards:
            from safetensors.torch import load_file

            self._shards[filename] = load_file(os.path.join(self._snap, filename))
        return self._shards[filename]

    def __getitem__(self, key: str) -> torch.Tensor:
        filename = self._idx[key]
        return self._shard(filename)[key]

    def __contains__(self, key: str) -> bool:
        return key in self._idx


def _moe_layer_forward(
    mesh_device,
    hidden_states: torch.Tensor,
    layer_idx: int,
    wc: WeightCache,
) -> torch.Tensor:
    """E-type block: pre-norm → gate + experts + shared_expert → residual."""
    residual = hidden_states
    p = f"backbone.layers.{layer_idx}"

    norm_w = wc[f"{p}.norm.weight"]
    normed = layer_norm_forward(mesh_device, hidden_states, norm_w)  # [B, S, 2688]

    B, S, H = normed.shape
    flat = normed.reshape(B * S, H)  # [tokens, 2688]

    gate_w = wc[f"{p}.mixer.gate.weight"]
    gate_b = wc[f"{p}.mixer.gate.e_score_correction_bias"]
    topk_idx, topk_wts = moe_gate_forward(mesh_device, flat, gate_w, gate_b)

    experts_up = [wc[f"{p}.mixer.experts.{e}.up_proj.weight"] for e in range(128)]
    experts_down = [wc[f"{p}.mixer.experts.{e}.down_proj.weight"] for e in range(128)]
    expert_out = moe_experts_forward(mesh_device, flat, topk_idx, topk_wts, experts_up, experts_down)
    # expert_out: [tokens, 2688]

    shared_out = shared_expert_forward(
        mesh_device,
        normed,
        w_up=wc[f"{p}.mixer.shared_experts.up_proj.weight"],
        w_down=wc[f"{p}.mixer.shared_experts.down_proj.weight"],
    )
    # shared_out: [B, S, 2688]

    moe_out = expert_out.reshape(B, S, H) + shared_out
    return (residual + moe_out).bfloat16()


def nemotron_h_forward(
    mesh_device,
    input_ids: torch.Tensor,  # [B, S] int64
    wc: WeightCache | None = None,
    num_layers: int = N_LAYERS,
) -> torch.Tensor:
    """Full NemotronH forward returning logits [B, S, vocab_size].

    Args:
        mesh_device:  Open TTNN MeshDevice.
        input_ids:    Token ids [B, S].
        wc:           WeightCache (created internally if None).
        num_layers:   Run only the first N layers (default: all 52).

    Returns:
        Logits [B, S, 131072] bfloat16 on CPU.
    """
    if wc is None:
        wc = WeightCache()

    B, S = input_ids.shape
    position_ids = torch.arange(S, dtype=torch.long).unsqueeze(0).expand(B, -1)

    # 1. Embedding
    emb_w = wc["backbone.embeddings.weight"]
    hidden_states = embedding_forward(mesh_device, input_ids, emb_w)  # [B, S, 2688]

    # 2. Layer stack
    for li in range(min(num_layers, N_LAYERS)):
        layer_type = PATTERN[li]
        p = f"backbone.layers.{li}"

        if layer_type == "M":
            hidden_states = mamba2_layer_forward(
                mesh_device,
                hidden_states,
                norm_weight=wc[f"{p}.norm.weight"],
                in_proj_weight=wc[f"{p}.mixer.in_proj.weight"],
                conv1d_weight=wc[f"{p}.mixer.conv1d.weight"],
                conv1d_bias=wc[f"{p}.mixer.conv1d.bias"],
                dt_bias=wc[f"{p}.mixer.dt_bias"],
                A_log=wc[f"{p}.mixer.A_log"],
                norm_mixer_weight=wc[f"{p}.mixer.norm.weight"],
                D=wc[f"{p}.mixer.D"],
                out_proj_weight=wc[f"{p}.mixer.out_proj.weight"],
            )

        elif layer_type == "E":
            hidden_states = _moe_layer_forward(mesh_device, hidden_states, li, wc)

        else:  # '*' — dense attention (no RoPE in HF source)
            hidden_states = dense_attention_forward(
                mesh_device,
                hidden_states,
                norm_weight=wc[f"{p}.norm.weight"],
                wq=wc[f"{p}.mixer.q_proj.weight"],
                wk=wc[f"{p}.mixer.k_proj.weight"],
                wv=wc[f"{p}.mixer.v_proj.weight"],
                wo=wc[f"{p}.mixer.o_proj.weight"],
            )

    # 3. LM head
    logits = lm_head_forward(
        mesh_device,
        hidden_states,
        norm_f_weight=wc["backbone.norm_f.weight"],
        lm_head_weight=wc["lm_head.weight"],
    )
    return logits
