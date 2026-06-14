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
from .tp import _rep

# Two persistent template tensors reused for all 23 E-layers.
# On first E-layer: allocated via _rep (1218 MiB each on device).
# On subsequent E-layers: updated in-place via copy_host_to_device_tensor.
# Total device DRAM for expert weights: ~2.4 GiB regardless of layer count.
_EXPERT_TEMPLATE: dict = {}  # mesh_id -> (up_tt, down_tt)

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
    """Load expert weights for layer_idx into the two shared template device tensors.

    up_tt:   [1, 128, 2688, 1856]  — hidden × intermediate, for up projection
    down_tt: [1, 128, 1856, 2688]  — intermediate × hidden, for down projection

    On the first E-layer call the templates are allocated via _rep (one-time DRAM cost
    ~2.4 GiB for both).  Every subsequent call tilizes the new layer's weights on the
    HOST and copies them into the templates in-place via copy_host_to_device_tensor.
    This keeps peak device DRAM for expert weights constant at ~2.4 GiB regardless of
    the number of E-layers (vs ~56 GiB if all 23 layers were kept live simultaneously).

    copy_host_to_device_tensor issues a DMA command that is captured in TTNN traces,
    so trace replay correctly streams each layer's weights into the template in order.
    """
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

    mesh_id = id(mesh_device)
    if mesh_id not in _EXPERT_TEMPLATE:
        # First call: allocate the two persistent template tensors on device.
        up_tt = _rep(up_cpu, mesh_device)
        down_tt = _rep(down_cpu, mesh_device)
        _EXPERT_TEMPLATE[mesh_id] = (up_tt, down_tt)
    else:
        up_tt, down_tt = _EXPERT_TEMPLATE[mesh_id]
        # Tilize on host (avoids device-side tilize DRAM overhead), then DMA into templates.
        up_host = ttnn.from_torch(up_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        down_host = ttnn.from_torch(down_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(up_host, up_tt)
        ttnn.copy_host_to_device_tensor(down_host, down_tt)

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
