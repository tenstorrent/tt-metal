# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Separate-projection controls on the cumulative TP4 topology and precision."""

import torch

import ttnn
from models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.candidate_stage5_multicore_combined import (
    MultichipDecoder as Base,
)


class MultichipDecoder(Base):
    @classmethod
    def from_state_dict(cls, state_dict, **kwargs):
        self = super().from_state_dict(state_dict, **kwargs)
        decode_attention = self.decode_weights.get("attention_decode.weight")
        prefix = f"model.layers.{self.layer_idx}."
        sd = {k.removeprefix(prefix): v for k, v in state_dict.items()}

        def put(name, parts, dtype):
            self.weights[name] = ttnn.from_torch(
                torch.cat(parts, dim=0).contiguous(),
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.weight_manifest[name] = dict(local_shape=list(parts[0].shape), dtype=str(dtype), mapping="TP4")

        if self.options.get("split_attention"):
            self.attention_names = ["self_attn.q_proj", "self_attn.k_proj"]
            if not self.sparse:
                self.attention_names += ["self_attn.v_proj"]
            self.attention_names += ["self_attn.gate_proj"]
            if self.sparse:
                self.attention_names += ["self_attn.v_router"]
            for name in self.attention_names:
                w = sd[name + ".weight"].T
                parts = [w] * 4 if name.endswith("v_router") else list(w.chunk(4, dim=-1))
                put(name + ".weight", parts, getattr(ttnn, self.options["attention_dtype"]))
        if self.options.get("split_dense"):
            self.options["packed_dense"] = False
            for part in ("gate_proj", "up_proj"):
                put(
                    "mlp." + part + ".weight",
                    list(sd["mlp." + part + ".weight"].T.chunk(4, dim=-1)),
                    getattr(ttnn, self.options["mlp_dtype"]),
                )
        self._prepare_tp_decode_weights()
        if decode_attention is not None:
            self.decode_weights["attention_decode.weight"] = decode_attention
        return self

    def _mlp(self, x, *, valid_tokens=None):
        if not self.options.get("split_dense") or x.shape[-2] != 32:
            return super()._mlp(x, valid_tokens=valid_tokens)
        gate = self._linear(
            x, "mlp.gate_proj", activation="silu" if self.options.get("split_silu_fused", False) else None
        )
        up = self._linear(x, "mlp.up_proj")
        return self._reduce(
            self._linear(
                ttnn.mul(
                    gate,
                    up,
                    input_tensor_a_activations=[]
                    if self.options.get("split_silu_fused", False)
                    else [ttnn.UnaryOpType.SILU],
                ),
                "mlp.down_proj",
            ),
            slot=1,
        )
