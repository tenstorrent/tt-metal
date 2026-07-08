# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `u_m_t5_dense_gated_act_dense`
(meituan-longcat/LongCat-Video's
`text_encoder.encoder.block.0.layer.1.DenseReluDense`, a real
`transformers.models.umt5.modeling_umt5.UMT5DenseGatedActDense`):

    wi_0, wi_1 = Linear(d_model, d_ff, bias=False) x2
    wo = Linear(d_ff, d_model, bias=False)
    forward(x): return wo(act(wi_0(x)) * wi_1(x))     # act = gelu_new (tanh-approx GELU)

Adapts the already-validated `T5DenseGatedActDense` in
`models/tt_dit/encoders/t5/model_t5.py` (same rationale/precedent as
`u_m_t5_block`) -- its parameter names (`wi_0`/`wi_1`/`wo`) match real HF
UMT5 state-dict keys exactly, so `load_torch_state_dict` works directly.
TP scheme (already implemented): `wi0`/`wi1` COLUMN-parallel (SwiGLU-style
gate/up, `gelu_tanh` fused into `wi0`), `wo` ROW-parallel with an all_gather
(not all_reduce) to reassemble the replicated output.
"""

from __future__ import annotations

import ttnn
from models.demos.hf_eager.longcat_video._stubs.u_m_t5_block import tp_parallel_config
from models.tt_dit.encoders.t5.model_t5 import T5DenseGatedActDense
from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config
from models.tt_dit.parallel.manager import CCLManager


class TtUMT5DenseGatedActDense:
    def __init__(self, mesh_device: ttnn.MeshDevice, torch_module) -> None:
        self.mesh_device = mesh_device
        config = UMT5Config(
            vocab_size=1,
            embed_dim=torch_module.wi_0.in_features,
            ff_dim=torch_module.wi_0.out_features,
            kv_dim=1,
            num_heads=1,
            num_hidden_layers=1,
        )
        ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        parallel_config = tp_parallel_config(mesh_device)

        self.module = T5DenseGatedActDense(config, mesh_device, ccl_manager, parallel_config)
        self.module.load_torch_state_dict(torch_module.state_dict())

    def __call__(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        return self.module(hidden_states)


def build(mesh_device: ttnn.MeshDevice, torch_module) -> TtUMT5DenseGatedActDense:
    return TtUMT5DenseGatedActDense(mesh_device, torch_module)
