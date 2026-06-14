# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""NemotronH-30B full model forward pass using TTNN components.

Layer pattern (52 layers):
  M = Mamba2 block   (SSM only — norm + Mamba2 mixer + residual)
  E = MoE MLP block  (norm + gate + 128 routed experts + shared expert + residual)
  * = Dense attention (norm + GQA attention + residual, no RoPE in HF source)

hidden_states is a ttnn.Tensor on device throughout (embedding → lm_head).
Only lm_head.forward brings the final result to CPU (logits).

Weight loading is lazy: weights are fetched from safetensors shards on first
access and cached in a dict so each shard is opened at most once.
"""

import json
import os

import torch

import ttnn

from .dense_attention import dense_attention_forward
from .embedding import embedding_forward, embedding_forward_tt
from .layer_norm import layer_norm_forward
from .lm_head import lm_head_forward, lm_head_forward_device
from .mamba2_layer import mamba2_layer_forward
from .moe_experts import moe_experts_forward
from .moe_gate import moe_gate_forward
from .shared_expert import shared_expert_forward
from .tp import _upload

# Per-layer cache for pre-stacked expert weight tensors on device.
# Weights stored as bfloat4_b (4-bit) replicated on all TP devices:
# [1, 128, 2688, 1856] bfloat4_b = ~305 MiB per matrix, 46 matrices ≈ 14 GiB
# total — fits in 32 GiB device DRAM.  Replicated (not sharded) because
# 4-way expert-dim sharding deadlocks sparse_matmul when a device receives
# 0 active experts (noc_semaphore_wait hang, tt-metal#45943).
_EXPERT_DEVICE_CACHE: dict = {}  # layer_idx -> (up_tt, down_tt)

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


def _get_stacked_expert_weights(mesh_device, layer_idx: int, wc: "WeightCache"):
    """Return (up_tt, down_tt) stacked expert-weight tensors for layer_idx.

    up_tt:   [1, 128, 2688, 1856] bfloat4_b, replicated on all TP devices
    down_tt: [1, 128, 1856, 2688] bfloat4_b, replicated on all TP devices

    bfloat4_b (4-bit) reduces per-matrix footprint from ~1218 MiB (bf16) to
    ~305 MiB.  All 23 E-layers' tensors together consume ~14 GiB — fits in
    32 GiB device DRAM.  The cache ensures no H2D uploads during trace capture.
    """
    if layer_idx in _EXPERT_DEVICE_CACHE:
        return _EXPERT_DEVICE_CACHE[layer_idx]

    p = f"backbone.layers.{layer_idx}"
    up_cpu = (
        torch.stack([wc[f"{p}.mixer.experts.{e}.up_proj.weight"] for e in range(128)])
        .transpose(-1, -2)  # [128, 1856, 2688] → [128, 2688, 1856]
        .unsqueeze(0)  # [1, 128, 2688, 1856]
        .bfloat16()
        .contiguous()
    )
    down_cpu = (
        torch.stack([wc[f"{p}.mixer.experts.{e}.down_proj.weight"] for e in range(128)])
        .transpose(-1, -2)  # [128, 2688, 1856] → [128, 1856, 2688]
        .unsqueeze(0)  # [1, 128, 1856, 2688]
        .bfloat16()
        .contiguous()
    )
    # Replicate weights on all TP devices; use bfloat4_b (4× smaller than bf16) so all
    # 23 E-layers × 2 matrices × 305 MiB ≈ 14 GiB fits in 32 GiB device DRAM.
    # 4-way expert-dim sharding deadlocks sparse_matmul when a device receives 0 active
    # experts (sender sends 0 tiles, receiver loops waiting → noc_semaphore_wait hang).
    up_tt = _upload(up_cpu, mesh_device, shard_dim=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b)
    down_tt = _upload(down_cpu, mesh_device, shard_dim=None, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b)
    _EXPERT_DEVICE_CACHE[layer_idx] = (up_tt, down_tt)
    return up_tt, down_tt


def _moe_layer_forward(
    mesh_device,
    hidden_states: ttnn.Tensor,  # [B, S, 2688] bf16 on device
    layer_idx: int,
    wc: "WeightCache",
) -> ttnn.Tensor:
    """E-type block: pre-norm → gate + experts + shared_expert → residual."""
    residual = hidden_states
    p = f"backbone.layers.{layer_idx}"

    normed_tt = layer_norm_forward(mesh_device, hidden_states, wc[f"{p}.norm.weight"])

    B = normed_tt.shape[0]
    S = normed_tt.shape[1]
    H = normed_tt.shape[2]

    # Flatten for gate/experts: [B, S, H] → [B*S, H]
    flat_tt = ttnn.reshape(normed_tt, [B * S, H])

    # Gate: returns dense [B*S, 128] routing-weight tensor on device.
    routing_weights_tt = moe_gate_forward(
        mesh_device,
        flat_tt,
        wc[f"{p}.mixer.gate.weight"],
        wc[f"{p}.mixer.gate.e_score_correction_bias"],
    )

    # Lazily build and cache pre-stacked expert weight tensors on device.
    up_tt, down_tt = _get_stacked_expert_weights(mesh_device, layer_idx, wc)

    # Routed experts via sparse_matmul (returns [B*S, 2688] on device).
    expert_out_tt = moe_experts_forward(mesh_device, flat_tt, routing_weights_tt, up_tt, down_tt)

    # Shared expert
    shared_out_tt = shared_expert_forward(
        mesh_device,
        normed_tt,
        w_up=wc[f"{p}.mixer.shared_experts.up_proj.weight"],
        w_down=wc[f"{p}.mixer.shared_experts.down_proj.weight"],
    )

    expert_out_reshaped = ttnn.reshape(expert_out_tt, [B, S, H])
    moe_out_tt = ttnn.add(expert_out_reshaped, shared_out_tt)
    return ttnn.add(residual, moe_out_tt)


def _layer_stack_forward(mesh_device, hidden_states: ttnn.Tensor, wc: WeightCache, num_layers: int) -> ttnn.Tensor:
    """Shared layer loop used by both forward variants."""
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
        else:
            hidden_states = dense_attention_forward(
                mesh_device,
                hidden_states,
                norm_weight=wc[f"{p}.norm.weight"],
                wq=wc[f"{p}.mixer.q_proj.weight"],
                wk=wc[f"{p}.mixer.k_proj.weight"],
                wv=wc[f"{p}.mixer.v_proj.weight"],
                wo=wc[f"{p}.mixer.o_proj.weight"],
            )
    return hidden_states


def nemotron_h_forward(
    mesh_device,
    input_ids: torch.Tensor,  # [B, S] int64
    wc: WeightCache | None = None,
    num_layers: int = N_LAYERS,
) -> torch.Tensor:
    """Full NemotronH forward returning logits [B, S, vocab_size] on CPU.

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

    hidden_states = embedding_forward(mesh_device, input_ids, wc["backbone.embeddings.weight"])
    hidden_states = _layer_stack_forward(mesh_device, hidden_states, wc, num_layers)

    return lm_head_forward(
        mesh_device,
        hidden_states,
        norm_f_weight=wc["backbone.norm_f.weight"],
        lm_head_weight=wc["lm_head.weight"],
    )


def nemotron_h_forward_device(
    mesh_device,
    ids_tt: ttnn.Tensor,  # pre-allocated uint32 device tensor [B, S]
    wc: WeightCache | None = None,
    num_layers: int = N_LAYERS,
) -> ttnn.Tensor:
    """Full NemotronH forward returning logits as ttnn.Tensor (no D2H).

    Accepts a pre-allocated device tensor for the token IDs so that the
    caller can update it via ttnn.copy_host_to_device_tensor and replay
    the captured trace without re-tracing.

    Returns:
        Logits [B, S, 131072] bfloat16 on device.
    """
    if wc is None:
        wc = WeightCache()

    hidden_states = embedding_forward_tt(mesh_device, ids_tt, wc["backbone.embeddings.weight"])
    hidden_states = _layer_stack_forward(mesh_device, hidden_states, wc, num_layers)

    return lm_head_forward_device(
        mesh_device,
        hidden_states,
        norm_f_weight=wc["backbone.norm_f.weight"],
        lm_head_weight=wc["lm_head.weight"],
    )
