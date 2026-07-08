# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `u_m_t5_layer_f_f`
(meituan-longcat/LongCat-Video's `text_encoder.encoder.block.0.layer.1`, a
real `transformers.models.umt5.modeling_umt5.UMT5LayerFF`):

    forward(hidden_states):
        forwarded = layer_norm(hidden_states)              # UMT5LayerNorm (RMSNorm)
        forwarded = DenseReluDense(forwarded)               # gated-GELU FFN
        return hidden_states + forwarded

Adapts the already-validated `T5FF` in `models/tt_dit/encoders/t5/model_t5.py`
-- the same class `u_m_t5_block` uses internally as `T5EncoderLayer.ff` --
same rationale/precedent as `u_m_t5_block`/`u_m_t5_dense_gated_act_dense`.
`T5FF._prepare_torch_state` renames `DenseReluDense` -> `dense_gated_dense`;
`layer_norm` matches HF's key directly, so `load_torch_state_dict` works
directly on real UMT5 weights with no manual key mapping. TP scheme (already
implemented, not re-derived): `wi0`/`wi1` COLUMN-parallel, `wo` ROW-parallel
with an all_gather (not all_reduce) -- see `T5DenseGatedActDense.forward`.
"""

from __future__ import annotations

import ttnn
from models.demos.hf_eager.longcat_video._stubs.u_m_t5_block import tp_parallel_config
from models.tt_dit.encoders.t5.model_t5 import T5FF
from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config
from models.tt_dit.parallel.manager import CCLManager


class TtUMT5LayerFF:
    def __init__(self, mesh_device: ttnn.MeshDevice, torch_module) -> None:
        self.mesh_device = mesh_device
        dense = torch_module.DenseReluDense
        config = UMT5Config(
            vocab_size=1,
            embed_dim=dense.wi_0.in_features,
            ff_dim=dense.wi_0.out_features,
            kv_dim=1,
            num_heads=1,
            num_hidden_layers=1,
            layer_norm_eps=torch_module.layer_norm.variance_epsilon,
        )
        ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        parallel_config = tp_parallel_config(mesh_device)

        self.module = T5FF(config, mesh_device, ccl_manager, parallel_config)
        self.module.load_torch_state_dict(torch_module.state_dict())

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        return self.module(hidden_states)


def build(mesh_device: ttnn.MeshDevice, torch_module) -> TtUMT5LayerFF:
    return TtUMT5LayerFF(mesh_device, torch_module)
