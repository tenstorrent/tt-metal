# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate shared gate/up/router packed weight precision on the final layout."""

import torch

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tt.optimized_decoder import OptimizedDecoder as Base


class OptimizedDecoder(Base):
    @classmethod
    def from_state_dict(cls, *args, **kwargs):
        self = super().from_state_dict(*args, **kwargs)
        assert self.sparse
        prefix = f"model.layers.{self.layer_idx}."
        sd = {k.removeprefix(prefix): v for k, v in args[0].items()}
        router = torch.zeros(2560, 128, dtype=torch.bfloat16)
        router[:, :100] = sd["mlp.gate.weight"].T
        weight = torch.cat(
            [sd["mlp.shared_experts.gate_proj.weight"].T, sd["mlp.shared_experts.up_proj.weight"].T, router], dim=-1
        )
        name = "mlp.shared_router.weight"
        dtype = getattr(ttnn, self.options["shared_router_dtype"])
        self.weights[name] = ttnn.from_torch(
            weight.contiguous(), device=self.mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT
        )
        previous = self.decode_weights[name]
        padded = torch.nn.functional.pad(weight, (0, previous.shape[-1] - weight.shape[-1]))
        prepared = ttnn.from_torch(padded.contiguous(), device=self.mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.decode_weights[name] = ttnn.to_memory_config(prepared, previous.memory_config())
        return self
