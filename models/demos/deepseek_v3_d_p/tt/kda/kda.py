# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Composed TTNN Kimi Delta Attention layer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.kda.config import KDA_SOFTPLUS_BETA, KDA_SOFTPLUS_THRESHOLD, KDAConfig
from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_BETA_DTYPE, KDA_OUTPUT_MEMORY_CONFIG, KDAProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.linear_attention import (
    KdaState,
    LinearAttentionGeometry,
    LinearAttentionSkeleton,
    mesh_axis_size,
    slice_width,
)
from models.demos.deepseek_v3_d_p.tt.kda.weights import KDAWeights, load_kda_weights
from models.tt_transformers.tt.ccl import TT_CCL

__all__ = ["KdaState", "ttKDA"]


@dataclass(frozen=True)
class _ProjectedInputs:
    qkv: ttnn.Tensor
    decay_rank: ttnn.Tensor
    output_gate: ttnn.Tensor
    beta: ttnn.Tensor


class ttKDA:
    """Prefill KDA for one fixed physical geometry and caller-owned logical state.

    ``active_seq_len`` is the global physical token count, matching MLA's
    construction contract. Each call supplies exactly ``active_seq_len / SP``
    local rows. Construct another instance for a different physical length;
    weights may be shared. Runtime ``actual_start`` changes chronology within
    the constructed graph without changing grouping or reading device values.
    The shared linear-attention graph is ``tt/kda/linear_attention.py``; this
    class owns the KDA projection split, gates and weights.
    """

    def __init__(
        self,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        config: KDAConfig,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        layer_idx: int = 0,
        weight_cache_path: Path | None = None,
        tt_ccl: TT_CCL | None = None,
        sp_axis: int = 0,
        tp_axis: int = 1,
        program_config: KDAProgramConfig | None = None,
        weights: KDAWeights | None = None,
        *,
        active_seq_len: int,
    ) -> None:
        program_config = program_config or KDAProgramConfig()
        tensor_parallel_size = mesh_axis_size(mesh_device, tp_axis) if tp_axis in (0, 1) else 1
        if config.num_heads % tensor_parallel_size:
            raise ValueError(
                f"num_heads {config.num_heads} must be divisible by tensor parallel size {tensor_parallel_size}"
            )
        local_heads = config.num_heads // tensor_parallel_size
        self._skeleton = LinearAttentionSkeleton(
            mesh_device,
            LinearAttentionGeometry(
                hidden_size=config.hidden_size,
                key_heads=local_heads,
                value_heads=local_heads,
                key_dim=config.head_k_dim,
                value_dim=config.head_v_dim,
                conv_kernel_size=config.conv_kernel_size,
                norm_eps=config.norm_eps,
            ),
            program_config,
            layer_name="KDA",
            sequence_parallel_axis=sp_axis,
            tensor_parallel_axis=tp_axis,
            active_seq_len=active_seq_len,
            tt_ccl=tt_ccl,
            scalar_decay=False,
        )
        self.device = mesh_device
        self.tt_ccl = tt_ccl
        self._activate_decay = self._softplus_decay if config.gate_lower_bound is None else self._bounded_decay
        if weights is not None and state_dict is not None:
            raise ValueError("pass either constructed KDAWeights or host state_dict, not both")
        if weights is None:
            weights = load_kda_weights(
                mesh_device,
                config,
                state_dict,
                weight_cache_path,
                cache_name_prefix=f"layer_{layer_idx}.kda",
                tensor_parallel_axis=tp_axis,
            )
        if weights.tensor_parallel_size != tensor_parallel_size or weights.tensor_parallel_axis != tp_axis:
            raise ValueError(
                "KDAWeights placement does not match the layer mesh: "
                f"weights TP={weights.tensor_parallel_size} axis={weights.tensor_parallel_axis}, "
                f"layer TP={tensor_parallel_size} axis={tp_axis}"
            )
        self.weights = weights
        self.config = replace(config, num_heads=local_heads)

    @property
    def qkv_convolution_program_config(self) -> ttnn.QkvCausalConv1dSiluProgramConfig:
        return self._skeleton.qkv_convolution_program_config

    @property
    def tp_ccl_topology(self) -> ttnn.Topology:
        return self._skeleton.tp_ccl_topology

    def allocate_state(self, batch_size: int = 1) -> KdaState:
        """Allocate the canonical device-resident KDA carries for one prefill stream."""
        return self._skeleton.allocate_state(batch_size)

    def _project_inputs(
        self,
        hidden_states: ttnn.Tensor,
    ) -> _ProjectedInputs:
        """Run the fused input projection and split its semantic outputs."""
        config = self.config
        projected = self._skeleton.project_input(hidden_states, self.weights.input_projection)
        auxiliary_start = self._skeleton.geometry.convolution_width
        return _ProjectedInputs(
            qkv=slice_width(projected, 0, auxiliary_start),
            decay_rank=slice_width(projected, auxiliary_start, auxiliary_start + config.head_k_dim),
            output_gate=slice_width(
                projected,
                auxiliary_start + config.head_k_dim,
                auxiliary_start + config.head_k_dim + config.v_dim,
            ),
            beta=slice_width(
                projected,
                auxiliary_start + config.head_k_dim + config.v_dim,
                auxiliary_start + config.head_k_dim + config.v_dim + config.num_heads,
            ),
        )

    def _compute_gates(
        self,
        *,
        beta: ttnn.Tensor,
        decay_rank: ttnn.Tensor,
    ) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Evaluate the decay and write gates consumed by the recurrence."""
        weights = self.weights
        # Preserve the sigmoid result at the FP32 precision required by chunk preparation.
        beta_for_recurrence = ttnn.sigmoid(
            ttnn.typecast(
                beta,
                KDA_BETA_DTYPE,
                memory_config=KDA_OUTPUT_MEMORY_CONFIG,
            ),
            memory_config=KDA_OUTPUT_MEMORY_CONFIG,
        )
        gate = ttnn.linear(
            decay_rank,
            weights.decay_output_projection,
            bias=weights.decay_bias_flat,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._skeleton.matmul_compute_config,
        )
        return self._activate_decay(gate), beta_for_recurrence

    def _softplus_decay(self, gate: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.multiply(
            self.weights.decay_scale_flat,
            gate,
            input_tensor_b_activations=[
                ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, KDA_SOFTPLUS_BETA, KDA_SOFTPLUS_THRESHOLD)
            ],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _bounded_decay(self, gate: ttnn.Tensor) -> ttnn.Tensor:
        gate = ttnn.multiply(
            self.weights.decay_scale_flat, gate, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        gate = ttnn.sigmoid(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.multiply(gate, self.config.gate_lower_bound, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        state: KdaState,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, KdaState]:
        """Run prefill KDA and return replacement logical carries.

        ``actual_start`` is the absolute global position of this chunk's first
        token. It selects the chronological SP segment order for MLA's
        block-cyclic layout. The caller owns a replicated UINT32 row-major scalar
        and must keep it alive at the captured address throughout trace use.
        Update its contents before replay; every SP rank must observe the same
        nonnegative, 32-aligned position. No device-to-host value validation is
        performed. Pass an explicit zero-valued tensor for a zero-start call.

        Optional ``actual_end`` is a replicated device scalar defining the
        exclusive global valid end. The interval is nonempty, 32-aligned, and
        no larger than the constructed capacity. Omission means full capacity.
        Bounds may change during trace replay; their addresses must stay alive.
        Padded output rows are unspecified; returned carries stop at the valid end.

        The input state is only read. No tensor reachable from it is used as a
        ``ttnn.copy`` destination or retained on this layer. The returned output
        is sequence-partitioned along SP and, when TP > 1, reduce-scattered on
        the hidden dimension; TP == 1 returns the full hidden dimension.
        """
        skeleton = self._skeleton
        skeleton.validate_forward(hidden_states, state, actual_start, actual_end)
        selections = skeleton.selections(actual_start, actual_end)
        projected = self._project_inputs(hidden_states)
        q, k, v, new_convolution = skeleton.convolve(
            projected.qkv, state.convolution, selections, actual_start, self.weights.convolution_taps
        )
        gate, beta = self._compute_gates(
            beta=projected.beta,
            decay_rank=projected.decay_rank,
        )
        result = skeleton.recur(
            q=q,
            k=k,
            v=v,
            gate=gate,
            beta=beta,
            initial_state=state.recurrent,
            selections=selections,
            actual_start=actual_start,
            actual_end=actual_end,
        )
        output = skeleton.gated_rms_norm(result.output, projected.output_gate, self.weights.norm)
        output = skeleton.project_output(output, self.weights.output_projection)
        return output, KdaState(recurrent=result.final_state, convolution=new_convolution)
