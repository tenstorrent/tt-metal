# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid decoder layer (vLLM + TP) for Qwen3.6-27B: DeltaNet or GQA Attention +
DENSE gated MLP.

Identical structure to the single-device decoder.py but:
  * token-mixer and MLP are the TP-aware `_vllm` versions,
  * the attention forward receives the vLLM paged/contiguous hooks
    (page_table, cur_pos, batch_idx, cache_batch) and a device `position` tensor
    for trace decode,
  * RMSNorm weights are replicated across the mesh (ReplicateTensorToMesh).
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen36_27b.tt.deltanet_vllm import TtGatedDeltaNet
from models.demos.qwen36_27b.tt.attention_vllm import TtGatedAttention
from models.demos.qwen36_27b.tt.mlp_vllm import TtMLP


TILE = 32


class SimpleRMSNorm(LightweightModule):
    """Device-side RMSNorm using ttnn.rms_norm. Weight is replicated across the mesh."""

    def __init__(self, device, dim, state_dict, key, eps=1e-6, dtype=ttnn.bfloat16, replicate=False):
        super().__init__()
        self.device = device
        self.eps = eps
        w = state_dict[f"{key}.weight"]
        torch_weight = (w + 1.0).unsqueeze(0).view(1, 1, dim).reshape(1, 1, dim // TILE, TILE)
        mapper = ttnn.ReplicateTensorToMesh(device) if replicate else None
        self.weight = ttnn.from_torch(
            torch_weight, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=mapper
        )

    def forward(self, x):
        return ttnn.rms_norm(x, epsilon=self.eps, weight=self.weight)


class TtHybridDecoderLayer(LightweightModule):
    def __init__(self, device, state_dict, layer_idx, config, dtype=ttnn.bfloat16):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        replicate = getattr(config, "dense_tp", False)

        if self.layer_type == "linear_attention":
            self.token_mixer = TtGatedDeltaNet(device, state_dict, layer_idx, config, dtype=dtype)
        else:
            self.token_mixer = TtGatedAttention(device, state_dict, layer_idx, config, dtype=dtype)

        self.mlp = TtMLP(device, state_dict, layer_idx, config, dtype=dtype)

        prefix = f"model.layers.{layer_idx}"
        self.input_layernorm = SimpleRMSNorm(
            device, config.hidden_size, state_dict, f"{prefix}.input_layernorm",
            eps=config.rms_norm_eps, dtype=dtype, replicate=replicate,
        )
        self.post_attention_layernorm = SimpleRMSNorm(
            device, config.hidden_size, state_dict, f"{prefix}.post_attention_layernorm",
            eps=config.rms_norm_eps, dtype=dtype, replicate=replicate,
        )

    def forward(self, hidden_states, deltanet_state=None, cos=None, sin=None, kv_cache=None,
                mode="decode", position=0, page_table=None, cur_pos=None, batch_idx=0, cache_batch=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        new_kv_cache = None
        if self.layer_type == "linear_attention":
            hidden_states = self.token_mixer(hidden_states, deltanet_state, mode=mode)
            import os as _os
            if _os.environ.get("QWEN36_DUMP_DNOUT") and self.layer_idx in (0, 1, 2, 4, 5, 6):
                import torch as _torch
                from models.demos.qwen36_27b.tt.mesh_utils import to_torch as _m2t
                S = int(hidden_states.shape[-2])
                _torch.save(_m2t(hidden_states).float().reshape(1, S, -1).clone(),
                            f"/home/yito/work/tt_dnout_{self.layer_idx}.pt")
        else:
            hidden_states, new_kv_cache = self.token_mixer(
                hidden_states, cos, sin, kv_cache, mode=mode, position=position,
                page_table=page_table, cur_pos=cur_pos, batch_idx=batch_idx, cache_batch=cache_batch,
            )

        hidden_states = ttnn.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = ttnn.add(residual, hidden_states)

        return hidden_states, new_kv_cache
