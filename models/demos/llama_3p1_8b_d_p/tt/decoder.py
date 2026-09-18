# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B prefill decoder layer (tt-blaze#4146).

    x = x + attention(attn_norm(x))
    x = x + mlp(ffn_norm(x))

Pre-norm, as HF. Structurally ``deepseek_v3_d_p/tt/tt_prefill_block.py`` — same sharded residual,
same norm -> submodule -> add shape — with the MLA/MoE branches replaced by dense GQA and SwiGLU.

**The residual stream is TP-sharded on hidden**: ``[1, 1, seq, emb_dim / tp]``, and it stays that
way across the whole layer boundary. Every collective in this layer therefore sits *inside* a
submodule, not between them:

    residual x            [1, 1, seq, emb/tp]   TP-sharded
      attn_norm(x)        [1, 1, seq, emb]      all-gather, inside the norm
      attention(...)      [1, 1, seq, emb/tp]   reduce-scatter, inside attention
      x = x + attn_out                          local elementwise add, no collective
      ffn_norm(x)         [1, 1, seq, emb]      all-gather, inside the norm
      mlp(...)            [1, 1, seq, emb/tp]   reduce-scatter, inside the MLP
      x = x + ffn_out                           local elementwise add, no collective

Four collectives per layer, which is the floor for TP: each of attention and the MLP needs its
input replicated and produces a partial sum. The alternative — a replicated residual, as
``gpt_oss_d_p/tt/layer.py`` uses — would carry the full 4096-wide residual on every chip and
all-gather after each submodule instead, moving the same bytes while holding ``tp`` times as much
residual live. See the layout notes in ``tt/mlp.py`` and ``tt/attention.py``; both were written to
this contract, which is what lets the adds here be plain ``ttnn.add``.

Kept free of reference-model and safetensors imports; the import-light contract is asserted by
``tests/unit/test_scaffold.py``.
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig
from models.demos.llama_3p1_8b_d_p.tt.attention import TtLlamaAttention
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import Llama31KVCache
from models.demos.llama_3p1_8b_d_p.tt.mlp import TtLlamaMLP
from models.demos.llama_3p1_8b_d_p.tt.rms_norm import TtLlamaRMSNorm

# HuggingFace's per-layer parameter names, which is what this module's ``torch_weights`` is keyed by
# so a checkpoint slice can be handed over untouched.
ATTENTION_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_KEYS = ("gate_proj", "up_proj", "down_proj")
NORM_KEYS = ("input_layernorm", "post_attention_layernorm")


def _substate(weights: dict, prefix: str) -> dict:
    """The sub-dict of ``weights`` under ``prefix``, with the prefix stripped."""
    dot = prefix + "."
    return {key[len(dot) :]: value for key, value in weights.items() if key.startswith(dot)}


class TtLlamaDecoderLayer(LightweightModule):
    """One prefill decoder layer. TP-sharded residual in, TP-sharded residual out."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        mesh_config,
        torch_weights: Optional[dict] = None,
        layer_idx: int = 0,
        cache_layer_idx: Optional[int] = None,
        emb_dim: int = Llama31_8BConfig.EMB_SIZE,
        hidden_dim: int = Llama31_8BConfig.INTERMEDIATE_SIZE,
        n_heads: int = Llama31_8BConfig.NUM_ATTENTION_HEADS,
        n_kv_heads: int = Llama31_8BConfig.NUM_KEY_VALUE_HEADS,
        head_dim: int = Llama31_8BConfig.HEAD_DIM,
        rms_norm_eps: float = Llama31_8BConfig.RMS_NORM_EPS,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        activations_dtype: ttnn.DataType = ttnn.bfloat16,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
    ):
        """
        Args:
            torch_weights: keyed by HuggingFace's per-layer names, i.e.
                ``self_attn.{q,k,v,o}_proj.weight``, ``mlp.{gate,up,down}_proj.weight``,
                ``input_layernorm.weight`` and ``post_attention_layernorm.weight`` — the keys
                ``reference/model.py:hf_key_map`` produces once the ``model.layers.N.`` prefix is
                stripped, so a checkpoint slice needs no renaming. Random weights when omitted,
                which is shape bring-up only.
            layer_idx: this layer's GLOBAL index in the 32-layer stack. Identity only — weight
                names, logging, and the per-layer completion sink, which keys on it across ranks.
            cache_layer_idx: this layer's slot index *within this rank's* KV cache. Defaults to
                ``layer_idx``, which is right whenever a rank holds the whole stack.

                Separate from ``layer_idx`` so a pipeline rank allocates only the layers it fills.
                The cache packs slots as ``user_id * num_layers + cache_layer_idx``; addressing it
                by the global index instead would make a rank holding layers 24..31 index slots
                24..31 of its own 8-slot cache, so every rank would have to allocate all 32 layers'
                worth of KV to use 8 of it.
        """
        super().__init__()

        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.layer_idx = layer_idx
        self.cache_layer_idx = layer_idx if cache_layer_idx is None else cache_layer_idx
        self.emb_dim = emb_dim
        self.emb_dim_per_chip = mesh_config.shard_size(emb_dim)

        if torch_weights is not None:
            missing = [
                key
                for key in (
                    *(f"self_attn.{name}.weight" for name in ATTENTION_KEYS),
                    *(f"mlp.{name}.weight" for name in MLP_KEYS),
                    *(f"{name}.weight" for name in NORM_KEYS),
                )
                if key not in torch_weights
            ]
            if missing:
                raise ValueError(f"layer {layer_idx} torch_weights is missing {missing}")
            attn_weights = {name: torch_weights[f"self_attn.{name}.weight"] for name in ATTENTION_KEYS}
            mlp_weights = {name: torch_weights[f"mlp.{name}.weight"] for name in MLP_KEYS}
            attn_norm_weight = torch_weights["input_layernorm.weight"]
            ffn_norm_weight = torch_weights["post_attention_layernorm.weight"]
        else:
            logger.warning(f"TtLlamaDecoderLayer[{layer_idx}] built with random weights — bring-up only")
            attn_weights = mlp_weights = None
            attn_norm_weight = ffn_norm_weight = None

        def norm(weight, name):
            return TtLlamaRMSNorm(
                mesh_device=mesh_device,
                mesh_config=mesh_config,
                torch_weight=weight,
                emb_dim=emb_dim,
                eps=rms_norm_eps,
                num_links=num_links,
                topology=topology,
                weights_dtype=weights_dtype,
                weight_cache_path=weight_cache_path,
                cache_name_prefix=f"{cache_name_prefix}.{name}" if cache_name_prefix else None,
            )

        self.attn_norm = norm(attn_norm_weight, "attn_norm")
        self.attention = TtLlamaAttention(
            mesh_device=mesh_device,
            mesh_config=mesh_config,
            torch_weights=attn_weights,
            emb_dim=emb_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            num_links=num_links,
            topology=topology,
            activations_dtype=activations_dtype,
            weights_dtype=weights_dtype,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"{cache_name_prefix}.attention" if cache_name_prefix else None,
        )
        self.ffn_norm = norm(ffn_norm_weight, "ffn_norm")
        self.mlp = TtLlamaMLP(
            mesh_device=mesh_device,
            mesh_config=mesh_config,
            torch_weights=mlp_weights,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_links=num_links,
            topology=topology,
            activations_dtype=activations_dtype,
            weights_dtype=weights_dtype,
            weight_cache_path=weight_cache_path,
            cache_name_prefix=f"{cache_name_prefix}.mlp" if cache_name_prefix else None,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        rope_mats,
        transformation_mat,
        *,
        kv_cache: Optional[Llama31KVCache] = None,
        ccl_manager=None,
        user_id: int = 0,
        cached_len: int = 0,
        indexed_rope: bool = False,
    ) -> ttnn.Tensor:
        """``x``: TP-sharded ``[1, 1, seq, emb_dim / tp]`` -> the same layout.

        Arguments other than ``x`` are forwarded to attention unchanged; see
        ``tt/attention.py:TtLlamaAttention.forward``. The cache slot index is not an argument
        because the layer owns it — a caller passing the wrong one would write another layer's KV.
        """
        tp = self.mesh_device.shape[self.mesh_config.tp_axis]
        expected = self.emb_dim if tp == 1 else self.emb_dim_per_chip
        if x.shape[-1] != expected:
            raise ValueError(
                f"layer {self.layer_idx}: residual last dim {x.shape[-1]} != {expected}; this layer "
                f"consumes and returns the TP-sharded residual stream"
            )

        normed = self.attn_norm(x, ccl_manager)
        attn_out = self.attention(
            normed,
            rope_mats,
            transformation_mat,
            kv_cache=kv_cache,
            ccl_manager=ccl_manager,
            cache_layer_idx=self.cache_layer_idx,
            user_id=user_id,
            cached_len=cached_len,
            indexed_rope=indexed_rope,
        )
        ttnn.deallocate(normed)
        # In-place into attn_out rather than allocating a third tensor: x is the incoming residual
        # and is still needed as the addend, attn_out is dead after this.
        x = ttnn.add(x, attn_out, output_tensor=attn_out)

        normed = self.ffn_norm(x, ccl_manager)
        ffn_out = self.mlp(normed)
        ttnn.deallocate(normed)
        return ttnn.add(x, ffn_out, output_tensor=ffn_out)

    @staticmethod
    def weights_from_layer_state_dict(state_dict: dict, layer_idx: int) -> dict:
        """Slice one layer's weights out of a full ``model.layers.N.`` state dict.

        Convenience for callers holding a whole checkpoint: the keys this returns are exactly what
        ``torch_weights`` wants.
        """
        prefix = f"model.layers.{layer_idx}"
        sliced = _substate(state_dict, prefix)
        if not sliced:
            # An empty slice would construct a layer with random weights and no complaint, which is
            # indistinguishable from a working layer until the logits are wrong.
            raise KeyError(f"no keys under {prefix}. in the given state dict")
        return sliced


def random_layer_weights(
    emb_dim: int = Llama31_8BConfig.EMB_SIZE,
    hidden_dim: int = Llama31_8BConfig.INTERMEDIATE_SIZE,
    n_heads: int = Llama31_8BConfig.NUM_ATTENTION_HEADS,
    n_kv_heads: int = Llama31_8BConfig.NUM_KEY_VALUE_HEADS,
    head_dim: int = Llama31_8BConfig.HEAD_DIM,
    scale: float = 0.02,
) -> dict:
    """A full set of HF-named layer weights, for tests and bring-up.

    Norm gammas are centred on 1.0, not 0.0: RMSNorm multiplies by gamma, so a zero-mean gamma
    makes the layer output ~0 and any PCC comparison meaningless.
    """
    return {
        "self_attn.q_proj.weight": torch.randn(n_heads * head_dim, emb_dim) * scale,
        "self_attn.k_proj.weight": torch.randn(n_kv_heads * head_dim, emb_dim) * scale,
        "self_attn.v_proj.weight": torch.randn(n_kv_heads * head_dim, emb_dim) * scale,
        "self_attn.o_proj.weight": torch.randn(emb_dim, n_heads * head_dim) * scale,
        "mlp.gate_proj.weight": torch.randn(hidden_dim, emb_dim) * scale,
        "mlp.up_proj.weight": torch.randn(hidden_dim, emb_dim) * scale,
        "mlp.down_proj.weight": torch.randn(emb_dim, hidden_dim) * scale,
        "input_layernorm.weight": 1.0 + torch.randn(emb_dim) * scale,
        "post_attention_layernorm.weight": 1.0 + torch.randn(emb_dim) * scale,
    }
