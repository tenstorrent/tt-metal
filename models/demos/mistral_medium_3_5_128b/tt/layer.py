# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One Ministral3 decoder layer composed from the named blocks (outline after gpt_oss_d_p/tt/layer.py):

    h   = x + Attention(RMSNorm_in(x))
    out = h + MLP(RMSNorm_post(h))

The residual stream stays SP-sharded on sequence and TP-sharded on hidden (``[1, 1, s_local, hidden/tp]``
per chip): each norm gathers to full width for its column-parallel consumer, and attention / MLP close
with a reduce-scatter back into the residual's layout.
"""

import ttnn

from .attention import Attention
from .common import cache_name
from .mlp import MLP
from .precision import precision
from .rms_norm import RMSNorm


def _sub(state_dict, prefix):
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)} if state_dict else {}


class DecoderLayer:
    def __init__(
        self,
        mesh_device,
        mesh_config,
        ccl_manager,
        cfg,
        state_dict,
        *,
        layer_idx: int,
        dtypes: dict,
        tensor_cache_path=None,
    ):
        """``state_dict``: this layer's HF weights without the ``layers.{i}.`` prefix. ``dtypes``: ttnn
        dtypes keyed ``attention`` / ``mlp_gate`` / ``mlp_up`` / ``mlp_down`` (resolved from the spec)."""

        def cache(name):
            return cache_name(tensor_cache_path, name)

        self.layer_idx = layer_idx
        self.residual_dtype = precision().residual_dtype
        self.input_layernorm = RMSNorm(
            mesh_device,
            mesh_config,
            ccl_manager,
            (state_dict or {}).get("input_layernorm.weight"),
            cfg.rms_norm_eps,
            tensor_cache_path=cache("input_layernorm"),
        )
        self.self_attn = Attention(
            mesh_device,
            mesh_config,
            ccl_manager,
            cfg,
            _sub(state_dict, "self_attn."),
            layer_idx=layer_idx,
            weight_dtype=dtypes["attention"],
            tensor_cache_path=cache("self_attn"),
        )
        self.post_attention_layernorm = RMSNorm(
            mesh_device,
            mesh_config,
            ccl_manager,
            (state_dict or {}).get("post_attention_layernorm.weight"),
            cfg.rms_norm_eps,
            tensor_cache_path=cache("post_attention_layernorm"),
        )
        self.mlp = MLP(
            mesh_device,
            mesh_config,
            ccl_manager,
            _sub(state_dict, "mlp."),
            gate_dtype=dtypes["mlp_gate"],
            up_dtype=dtypes["mlp_up"],
            down_dtype=dtypes["mlp_down"],
            tensor_cache_path=cache("mlp"),
        )

    def __call__(self, x, rope, *, kv_cache=None, user_id: int = 0, cached_len: int = 0):
        """``x`` sharded residual ``[1, 1, s_local, hidden/tp]`` -> same layout (``x`` is not consumed)."""
        normed = self.input_layernorm(x)
        attn = self.self_attn(normed, rope, kv_cache=kv_cache, user_id=user_id, cached_len=cached_len)
        normed.deallocate(True)
        h = ttnn.add(x, attn, dtype=self.residual_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn.deallocate(True)
        normed = self.post_attention_layernorm(h)
        mlp = self.mlp(normed)
        normed.deallocate(True)
        out = ttnn.add(h, mlp, dtype=self.residual_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        h.deallocate(True)
        mlp.deallocate(True)
        return out
