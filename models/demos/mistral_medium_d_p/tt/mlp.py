# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Mistral-Medium-3.5 dense SwiGLU MLP.

``down(silu(gate(x)) * up(x))`` at ``intermediate_size = 28672``, all three projections bias-free —
verified in ``transformers/models/mistral/modeling_mistral.py::MistralMLP`` (``hidden_act = "silu"``,
``bias=False``). This is a **plain** SwiGLU: no clamp, no alpha, no ``swigluoai`` — unlike gpt-oss,
MiniMax-M3 and Kimi, which is why none of their MoE/expert kernels are reusable here.

Parallelism (TP=8 on the mesh cols):
  * ``gate_proj`` / ``up_proj`` are **column-parallel**: the intermediate dim shards, 28672/8 = 3584
    per chip (112 tiles — exactly Llama-3.3-70B's per-chip FF width, so this side of the sharding is
    the one number carried over unchanged from the mechanism donor).
  * ``down_proj`` is **row-parallel**: it contracts the sharded intermediate, producing a partial sum
    over the full hidden dim, which a TP **reduce-scatter** both completes and lands as ``emb/tp`` —
    the sharded residual's layout. No trailing all-gather: that belongs in front of the next norm.

gate and up are stored FUSED as one ``[hidden, 2*I_local]`` weight so the two column-parallel
matmuls become one; the halves are split on device before the activation.
"""

import torch

import ttnn
from models.demos.mistral_medium_d_p.config import MeshConfig
from models.demos.mistral_medium_d_p.utils.general_utils import get_cache_file_name
from models.demos.mistral_medium_d_p.utils.substate import substate


class MLP:
    """Dense SwiGLU FFN (column-parallel gate/up, row-parallel down + TP all-reduce)."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        tensor_cache_path=None,
        mesh_config=None,
        weight_dtype=ttnn.bfloat8_b,
    ):
        act = getattr(hf_config, "hidden_act", "silu")
        if act != "silu":
            raise NotImplementedError(
                f"mistral_medium_d_p MLP implements plain SwiGLU (silu) only, got hidden_act={act!r}. "
                "A clamped 'swigluoai' variant needs the gpt-oss/M3 activation, not this one."
            )
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.ccl = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        if state_dict:
            for name in ("gate_proj", "up_proj", "down_proj"):
                sub = substate(state_dict, name)
                assert "weight" in sub, f"MLP state_dict is missing {name}.weight"
                assert "bias" not in sub, f"{name} carries a bias, but MistralMLP is bias-free"
            # HF stores nn.Linear as [out, in]; ttnn.linear wants [in, out].
            gate = substate(state_dict, "gate_proj")["weight"].transpose(-1, -2)  # [H, I]
            up = substate(state_dict, "up_proj")["weight"].transpose(-1, -2)  # [H, I]
            down = substate(state_dict, "down_proj")["weight"].transpose(-1, -2)  # [I, H]

            # Interleave per TP shard so device i holds [gate_i | up_i] contiguously; a plain
            # cat([gate, up], -1) would give device i the wrong halves once column_parallel splits it.
            tp = self.mesh_config.tp
            w13 = (
                torch.cat(
                    [
                        torch.cat([torch.chunk(gate, tp, dim=-1)[i], torch.chunk(up, tp, dim=-1)[i]], dim=-1)
                        for i in range(tp)
                    ],
                    dim=-1,
                )
                .unsqueeze(0)
                .unsqueeze(0)
            )
            down = down.unsqueeze(0).unsqueeze(0)
        else:
            w13 = None
            down = None

        self.w13 = ttnn.as_tensor(
            w13,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=self.mesh_config.column_parallel(mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, "w_gate_up_fused"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.w2 = ttnn.as_tensor(
            down,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=self.mesh_config.row_parallel(mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, "w_down"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.local_intermediate = self.intermediate_size // self.mesh_config.tp

    def __call__(self, x):
        """Forward.

        Args:
            x: ``[1, 1, S, hidden]`` — FULL hidden, replicated across the TP cols (a post-norm
               activation); the sequence is SP-sharded on the rows.
        Returns:
            ``[1, 1, S, hidden // tp]`` — reduce-scattered across TP, ready to add into the sharded
            residual. At ``tp == 1`` this is ``[1, 1, S, hidden]`` and no collective runs, so the
            single-chip tests exercise the same path.
        """
        activation_dtype = ttnn.bfloat8_b if x.shape[-2] > 32 * 1024 else ttnn.bfloat16
        i_local = self.local_intermediate

        w13_out = ttnn.linear(x, self.w13, dtype=activation_dtype)  # [1, 1, S, 2*I_local]
        b, c, s = w13_out.shape[0], w13_out.shape[1], w13_out.shape[2]
        gate = ttnn.slice(w13_out, [0, 0, 0, 0], [b, c, s, i_local], [1, 1, 1, 1])
        up = ttnn.slice(w13_out, [0, 0, 0, i_local], [b, c, s, 2 * i_local], [1, 1, 1, 1])
        w13_out.deallocate(True)

        # silu(gate) * up, with the activation fused into the multiply (the repo's idiom, e.g.
        # llama3_70b_galaxy/tt/llama_mlp.py) so this is one device op rather than two.
        gated = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], dtype=activation_dtype)
        gate.deallocate(True)
        up.deallocate(True)

        out = ttnn.linear(gated, self.w2, dtype=activation_dtype)  # partial sum over hidden
        gated.deallocate(True)

        if self.mesh_config.tp > 1:
            rs_out = self.mesh_config.reduce_scatter(out, self.ccl, dim=3, axis=self.mesh_config.tp_axis)
            out.deallocate(True)
            out = rs_out
        return out
