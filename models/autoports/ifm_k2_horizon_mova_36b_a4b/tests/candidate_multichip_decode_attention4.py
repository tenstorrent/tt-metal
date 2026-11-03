# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""BFP4 decode attention with the accepted BFP8 prefill projection preserved."""

import torch

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.multichip_decoder import MultichipDecoder as Base


class MultichipDecoder(Base):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        assert not self.sparse
        prefix = f"model.layers.{self.layer_idx}."
        sd = {k.removeprefix(prefix): v for k, v in state_dict.items()}
        packed = []
        for rank in range(4):
            packed.append(
                torch.cat(
                    [sd[f"self_attn.{name}_proj.weight"].T.chunk(4, dim=-1)[rank] for name in ("q", "k", "v", "gate")],
                    dim=-1,
                )
            )
        original = self.weights["attention_packed.weight"]
        self.weights["attention_packed.weight"] = ttnn.from_torch(
            torch.cat(packed, dim=0),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._prepare_tp_decode_weights()
        self.weights["attention_packed.weight"] = original
        self.options["attention_decode_dtype"] = "bfloat4_b"
        return self
