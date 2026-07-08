# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn, tensor-parallel port of `u_m_t5_stack`
(meituan-longcat/LongCat-Video's `text_encoder.encoder`, a real
`transformers.models.umt5.modeling_umt5.UMT5Stack`).

`UMT5EncoderModel.forward` (the `u_m_t5_encoder_model` component) does
nothing but call `self.encoder(input_ids=..., ...)` and return that
directly (checked against `transformers.models.umt5.modeling_umt5`) --
`u_m_t5_stack` computes byte-identical math to `u_m_t5_encoder_model` on the
same weights, so this reuses the exact same `T5Encoder` port (token
embedding + 24-layer `T5Stack` + final `T5RMSNorm`, fp32 residual stream to
clear the layer-20+ activation-outlier precision wall -- see
`u_m_t5_encoder_model.py`'s docstring for why).

The one real difference: `torch_module` here IS the `UMT5Stack` submodule
itself (`text_encoder.encoder`), not the outer `UMT5EncoderModel`
(`text_encoder`) -- so its `state_dict()` keys have no `encoder.` prefix
(`embed_tokens.weight`, not `encoder.embed_tokens.weight`) and no tied
`shared.weight` key at all. `T5Encoder._prepare_torch_state` expects the
`encoder.`-prefixed form (it strips that prefix to get `token_embeddings`);
re-adding the prefix here before `load_torch_state_dict` keeps that shared
renaming logic untouched instead of forking it.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.hf_eager.longcat_video._stubs.u_m_t5_block import tp_parallel_config
from models.demos.hf_eager.longcat_video._stubs.u_m_t5_encoder_model import _replicated_ttnn_to_torch
from models.tt_dit.encoders.t5.model_t5 import T5Encoder
from models.tt_dit.encoders.umt5.model_umt5 import UMT5Config
from models.tt_dit.parallel.manager import CCLManager


class TtUMT5Stack:
    def __init__(self, mesh_device: ttnn.MeshDevice, torch_module) -> None:
        self.mesh_device = mesh_device
        cfg = torch_module.config
        config = UMT5Config(
            vocab_size=cfg.vocab_size,
            embed_dim=cfg.d_model,
            ff_dim=cfg.d_ff,
            kv_dim=cfg.d_kv,
            num_heads=cfg.num_heads,
            num_hidden_layers=cfg.num_layers,
            layer_norm_eps=cfg.layer_norm_epsilon,
            relative_attention_num_buckets=cfg.relative_attention_num_buckets,
            relative_attention_max_distance=cfg.relative_attention_max_distance,
            dtype=ttnn.float32,
        )
        ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
        parallel_config = tp_parallel_config(mesh_device)

        self.model = T5Encoder(config, mesh_device, ccl_manager, parallel_config)
        encoder_prefixed_state = {f"encoder.{k}": v for k, v in torch_module.state_dict().items()}
        self.model.load_torch_state_dict(encoder_prefixed_state)

    def __call__(self, input_ids: ttnn.Tensor, attention_mask=None) -> ttnn.Tensor:
        # See u_m_t5_encoder_model.py -- skip the host round-trip when already true int ids.
        if input_ids.dtype == ttnn.uint32:
            ids_tt = input_ids
        else:
            ids_torch = _replicated_ttnn_to_torch(input_ids, self.mesh_device).round().to(torch.int32)
            ids_tt = ttnn.from_torch(
                ids_torch,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        hidden_states = self.model(ids_tt)
        return hidden_states[-1]


def build(mesh_device: ttnn.MeshDevice, torch_module) -> TtUMT5Stack:
    return TtUMT5Stack(mesh_device, torch_module)
