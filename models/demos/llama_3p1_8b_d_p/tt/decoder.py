# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""One eager Llama-3.1 decoder layer for fixed SP4/TP8 chunked prefill."""

from collections.abc import Mapping

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig as Model
from models.demos.llama_3p1_8b_d_p.tt.attention import (
    AttentionOutputProjection,
    FullCausalAttention,
    _validate_device_tensor,
)
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import write_kv_chunk
from models.demos.llama_3p1_8b_d_p.tt.mlp import MLP
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PREFILL_LAYOUT as layout
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import validate_mesh
from models.demos.llama_3p1_8b_d_p.tt.qkv import QKVProjection
from models.demos.llama_3p1_8b_d_p.tt.rms_norm import RMSNorm
from models.demos.llama_3p1_8b_d_p.tt.rope import apply_indexed_rope


class DecoderLayer:
    """Own one layer's weights and borrow sequential attention, RoPE and per-call KV storage.

    Inputs/outputs are local [1,1,256,4096] BF16 TILE interleaved DRAM tensors. The caller supplies
    encounter-ordered SP rows and TP replicas. Padded output queries have no defined value.
    One attention instance is shared across layers; calls through it must remain sequential.
    """

    def __init__(self, mesh_device, mesh_config, state_dict, *, layer_idx, attention, rope_tables, transformation_mat):
        if type(layer_idx) is not int:
            raise TypeError("layer_idx must be an eager Python int")
        if not 0 <= layer_idx < Model.NUM_LAYERS:
            raise ValueError(f"layer_idx {layer_idx} out of range [0, 32)")
        validate_mesh(mesh_device, mesh_config, "DecoderLayer")
        if not isinstance(attention, FullCausalAttention) or attention.mesh_device != mesh_device:
            raise ValueError("attention must be a FullCausalAttention on the constructor mesh")
        if len(rope_tables) != 2:
            raise ValueError("rope_tables must contain cos and sin")
        for name, table in zip(("cos", "sin"), rope_tables):
            _validate_device_tensor(
                table,
                mesh_device,
                name=f"decoder RoPE {name}",
                shape=(1, 1, attention.geometry.rope_local_sequence, Model.HEAD_DIM),
                dtype=ttnn.bfloat16,
            )
        _validate_device_tensor(
            transformation_mat,
            mesh_device,
            name="decoder RoPE transformation",
            shape=(1, 1, 32, 32),
            dtype=ttnn.bfloat16,
        )
        if not isinstance(state_dict, Mapping):
            raise ValueError("decoder state_dict must be a mapping of raw HF layer-relative weights")
        required = (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            *(f"self_attn.{name}_proj.weight" for name in ("q", "k", "v", "o")),
            *(f"mlp.{name}_proj.weight" for name in ("gate", "up", "down")),
        )
        missing = [name for name in required if name not in state_dict]
        if missing:
            raise ValueError(f"decoder state_dict is missing required weights: {', '.join(missing)}")

        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self._layer_idx = layer_idx
        self.attention = attention
        self.rope_tables = tuple(rope_tables)
        self.transformation_mat = transformation_mat
        attention_weights = {
            name.removeprefix("self_attn."): value
            for name, value in state_dict.items()
            if name.startswith("self_attn.")
        }
        mlp_weights = {
            name.removeprefix("mlp."): value for name, value in state_dict.items() if name.startswith("mlp.")
        }
        self.input_norm = RMSNorm(mesh_device, state_dict["input_layernorm.weight"])
        self.qkv = QKVProjection(mesh_device, mesh_config, attention_weights)
        self.output_projection = AttentionOutputProjection(mesh_device, mesh_config, attention_weights)
        self.post_attention_norm = RMSNorm(mesh_device, state_dict["post_attention_layernorm.weight"])
        self.mlp = MLP(mesh_device, mesh_config, mlp_weights)

    @property
    def layer_idx(self):
        return self._layer_idx

    def __call__(self, x, kv_cache, *, slot_idx, actual_start, actual_end):
        # Reject invalid input/cache/metadata before any K or V write. This is argument atomicity;
        # device execution failures are not transactional and must be handled by the caller.
        _validate_device_tensor(
            x,
            self.mesh_device,
            name="decoder input",
            shape=(1, 1, layout.local_sequence, Model.EMB_SIZE),
            dtype=ttnn.bfloat16,
        )
        self.attention.validate_request(
            kv_cache,
            slot_idx=slot_idx,
            layer_idx=self.layer_idx,
            actual_start=actual_start,
            actual_end=actual_end,
        )
        topology = ttnn.get_usable_topology(x, topology=ttnn.Topology.Ring, cluster_axis=layout.tp_axis)
        if topology != ttnn.Topology.Ring:
            raise RuntimeError(f"decoder requires a live TP ring, but TTNN selected {topology}")

        # Track only owned activations. Inputs, weights, tables, attention buffers and the cache
        # never enter this collection. A queued last use permits the usual TTNN explicit release.
        owned = {}

        def keep(tensor):
            owned[id(tensor)] = tensor
            return tensor

        def release(tensor):
            owned.pop(id(tensor)).deallocate(True)

        try:
            normalized = keep(self.input_norm(x))
            q, k, v = self.qkv(normalized)
            for tensor in (q, k, v):
                keep(tensor)
            release(normalized)
            q_rotated = keep(
                apply_indexed_rope(
                    q,
                    self.rope_tables,
                    self.transformation_mat,
                    kv_actual_global=actual_start,
                    sp_axis=layout.sp_axis,
                )
            )
            k_rotated = keep(
                apply_indexed_rope(
                    k,
                    self.rope_tables,
                    self.transformation_mat,
                    kv_actual_global=actual_start,
                    sp_axis=layout.sp_axis,
                )
            )
            release(q)
            release(k)
            write_kv_chunk(
                kv_cache,
                k_rotated,
                v,
                slot_idx=slot_idx,
                layer_idx=self.layer_idx,
                actual_start=actual_start,
                actual_end=actual_end,
            )
            release(k_rotated)
            release(v)
            attended = keep(
                self.attention(
                    q_rotated,
                    kv_cache,
                    slot_idx=slot_idx,
                    layer_idx=self.layer_idx,
                    actual_start=actual_start,
                    actual_end=actual_end,
                )
            )
            release(q_rotated)
            projected = keep(self.output_projection(attended))
            release(attended)
            residual = keep(
                ttnn.add(
                    x,
                    projected,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    fast_and_approximate_mode=False,
                )
            )
            release(projected)
            normalized = keep(self.post_attention_norm(residual))
            mlp_output = keep(self.mlp(normalized))
            release(normalized)
            return ttnn.add(
                residual,
                mlp_output,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                fast_and_approximate_mode=False,
            )
        finally:
            for tensor in owned.values():
                tensor.deallocate(True)
