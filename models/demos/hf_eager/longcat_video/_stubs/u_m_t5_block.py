# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `u_m_t5_block`
(meituan-longcat/LongCat-Video's `text_encoder.encoder.block.0`, a real
`transformers.models.umt5.modeling_umt5.UMT5Block` -- `text_encoder` is a
genuine registered transformers architecture, loaded directly via
`transformers.UMT5EncoderModel`, see `tests/pcc/_reference_loader.py`).

Rather than re-deriving a tensor-parallel T5/UMT5 encoder from scratch, this
adapts the already-validated, TP-aware `T5EncoderLayer` (+ `T5Attention`,
`T5FF`, `T5DenseGatedActDense`, `T5RMSNorm`) in
`models/tt_dit/encoders/t5/model_t5.py` -- same pattern as the graduated
`autoencoder_k_l_wan` adapting `models/tt_dit/models/vae/vae_wan2_1.py`.
That library's parameter names (`_prepare_torch_state` renames `layer.0` ->
`self_attn`, `layer.1` -> `ff`, `SelfAttention.q/k/v/o`,
`DenseReluDense.wi_0/wi_1/wo`) match real HF UMT5 state-dict keys exactly
(verified against `transformers.models.umt5.modeling_umt5`), so
`load_torch_state_dict` works directly on real UMT5 weights with no manual
key mapping. Its TP scheme (already implemented, not re-derived): q/k/v/o
COLUMN-parallel split by head, all-gathered back to replicated inside
`T5Attention.forward`; FFN `wi0`/`wi1` COLUMN-parallel, `wo` ROW-parallel
with an all_gather (not all_reduce -- see `T5DenseGatedActDense.forward`).

UMT5 (unlike classic T5) gives every layer its own relative-position-bias
embedding (`has_relative_attention_bias=True` on every block, see
`UMT5Attention.__init__`), matching `UMT5Config.use_relative_position_bias =
[True] * num_layers` in `model_umt5.py` -- so a standalone block, in
isolation, computes its own bias correctly via
`layer.self_attn.relative_attention_bias(seq_length)`.

`attention_mask`, when given, is applied as a real additive key-position bias
(0 at valid tokens, a large finite negative at padded ones) added directly to
`position_bias` before the single `T5EncoderLayer` call below -- the reference
`UMT5Attention.forward` does the same `position_bias = position_bias +
attention_mask`. This block deliberately does NOT reuse `T5Stack.forward`'s
own mask-conversion (`(attention_mask - 1.0) * float("inf")`): for a real
mask, valid positions are 1, so that formula computes `(1 - 1) * inf == 0 *
inf == nan`, corrupting EVERY valid position's bias, not just the padded
ones -- which is exactly why this bring-up originally left masking
unapplied rather than reuse that path. Using a large finite multiplier
(`1e9`) instead of `inf` avoids the NaN entirely while still driving the
softmax weight on padded keys to ~0. `attention_mask=None` (the PCC
harness's synthetic-input default, an implicit all-valid mask) skips this
block entirely, byte-identical to the pre-fix behavior.
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.encoders.t5.model_t5 import T5EncoderLayer
from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager


def umt5_config_from_layer(layer, num_hidden_layers: int = 1) -> UMT5Config:
    """Introspect a real HF `UMT5Block`/`UMT5LayerSelfAttention` for the
    dimensions `UMT5Config` needs, rather than hardcoding them (robust to
    this checkpoint's actual config without reading `text_encoder/config.json`
    at runtime). `vocab_size` is unused by anything below the token-embedding
    table, so any placeholder is fine for a standalone (non-`u_m_t5_encoder_model`)
    component."""
    attn = layer.layer[0].SelfAttention
    ff = layer.layer[-1].DenseReluDense
    return UMT5Config(
        vocab_size=1,
        embed_dim=attn.d_model,
        ff_dim=ff.wi_0.out_features,
        kv_dim=attn.key_value_proj_dim,
        num_heads=attn.n_heads,
        num_hidden_layers=num_hidden_layers,
        layer_norm_eps=layer.layer[0].layer_norm.variance_epsilon,
        relative_attention_num_buckets=attn.relative_attention_num_buckets,
        relative_attention_max_distance=attn.relative_attention_max_distance,
    )


def tp_parallel_config(mesh_device: ttnn.MeshDevice) -> EncoderParallelConfig:
    """A 1xN (or Nx1) TP mesh has exactly one axis with factor>1."""
    shape = tuple(mesh_device.shape)
    axis = 0 if shape[0] > shape[1] else 1
    return EncoderParallelConfig(tensor_parallel=ParallelFactor(factor=shape[axis], mesh_axis=axis))


class TtUMT5Block:
    def __init__(self, mesh_device: ttnn.MeshDevice, torch_module) -> None:
        self.mesh_device = mesh_device
        config = umt5_config_from_layer(torch_module)
        ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        parallel_config = tp_parallel_config(mesh_device)

        self.layer = T5EncoderLayer(config, mesh_device, ccl_manager, parallel_config, use_relative_position_bias=True)
        self.layer.load_torch_state_dict(torch_module.state_dict())

    def __call__(self, hidden_states: ttnn.Tensor, attention_mask=None) -> ttnn.Tensor:
        seq_length = hidden_states.shape[-2]
        position_bias = self.layer.self_attn.relative_attention_bias(seq_length)
        if attention_mask is not None:
            mask_bias = (attention_mask.reshape(1, 1, 1, -1).to(torch.float32) - 1.0) * 1e9
            mask_bias_tt = ttnn.from_torch(
                mask_bias.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            position_bias = position_bias + mask_bias_tt
        return self.layer(hidden_states, position_bias=position_bias)


def build(mesh_device: ttnn.MeshDevice, torch_module) -> TtUMT5Block:
    return TtUMT5Block(mesh_device, torch_module)
