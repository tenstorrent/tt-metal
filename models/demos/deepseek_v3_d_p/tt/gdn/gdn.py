# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTNN Qwen Gated DeltaNet (GDN) prefill layer on the KDA path (design gdn-on-kda §3-4, D3).

GDN composes the linear-attention skeleton it shares with ``ttKDA`` (``tt/kda/linear_attention.py``: state,
chronological selections, q/k/v convolution with its SP carry exchange, the chunked recurrence, gated RMSNorm,
output projection with the TP reduce-scatter). GDN owns:

* the fused per-rank input projection ``[q | k | v | z | a, padded to a tile | b]`` (``tt/gdn/weights.py``);
* the scalar decay ``g = -exp(A_log) * softplus(a + dt_bias)`` and ``beta = sigmoid(b)``, both formed in FP32 from
  the projected ``a`` and ``b`` columns, one value per (V head, token): the recurrence's scalar-decay mode;
* q/k with ``num_key_heads`` heads, V head ``h`` reading K head ``h // (HV / Hk)`` (the recurrence's key-head
  mapping); the L2 norm of q and k (eps 1e-6) and the ``K^-0.5`` query scale are those of the KDA chunk preparation;
* the output gate: ``sigmoid`` uses the gated RMSNorm as is, ``silu`` multiplies its result by ``z``
  (``x * sigmoid(z) * z = x * silu(z)``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch

import ttnn
from models.demos.deepseek_v3_d_p.reference.gdn.config import GDNConfig
from models.demos.deepseek_v3_d_p.tt.gdn.weights import GDNWeights, gdn_input_projection_widths, load_gdn_weights
from models.demos.deepseek_v3_d_p.tt.kda.config import KDAProgramConfig
from models.demos.deepseek_v3_d_p.tt.kda.linear_attention import (
    KdaState,
    LinearAttentionGeometry,
    LinearAttentionSkeleton,
    mesh_axis_size,
    slice_width,
)
from models.tt_transformers.tt.ccl import TT_CCL

# transformers / torch ``F.softplus`` defaults of the Qwen GDN decay.
_SOFTPLUS_BETA = 1.0
_SOFTPLUS_THRESHOLD = 20.0
# The decay and the write strength are formed and consumed in FP32 (the scalar-decay preparation contract).
_GATE_DTYPE = ttnn.float32


class ttGDN:
    """Prefill GDN for one fixed physical geometry and caller-owned logical state.

    The construction and forward contract is ``ttKDA``'s: ``active_seq_len`` is the global physical token count and
    each call supplies ``active_seq_len / SP`` local rows; ``program_config`` fixes the execution policy
    (``gdn_program_config`` gives the production table); state is a ``KdaState`` with ``value_heads`` recurrent heads
    and ``[q | k | v]`` convolution channels per TP rank. A TP rank holds whole K-head groups, so
    ``num_key_heads`` must be divisible by the TP size.
    """

    def __init__(
        self,
        mesh_device: ttnn.Device | ttnn.MeshDevice,
        config: GDNConfig,
        state_dict: Mapping[str, torch.Tensor] | None = None,
        *,
        program_config: KDAProgramConfig,
        active_seq_len: int,
        layer_idx: int = 0,
        weight_cache_path: Path | None = None,
        tt_ccl: TT_CCL | None = None,
        sp_axis: int = 0,
        tp_axis: int = 1,
        weights: GDNWeights | None = None,
    ) -> None:
        if config.head_k_dim % ttnn.TILE_SIZE or config.head_v_dim % ttnn.TILE_SIZE:
            raise ValueError(f"GDN requires tile-aligned head dims, got K={config.head_k_dim}, V={config.head_v_dim}")
        tensor_parallel_size = mesh_axis_size(mesh_device, tp_axis) if tp_axis in (0, 1) else 1
        if config.num_key_heads % tensor_parallel_size:
            raise ValueError(
                f"num_key_heads {config.num_key_heads} must be divisible by tensor parallel size "
                f"{tensor_parallel_size} (a TP rank holds whole K-head groups)"
            )
        self._skeleton = LinearAttentionSkeleton(
            mesh_device,
            LinearAttentionGeometry(
                hidden_size=config.hidden_size,
                key_heads=config.num_key_heads // tensor_parallel_size,
                value_heads=config.num_value_heads // tensor_parallel_size,
                key_dim=config.head_k_dim,
                value_dim=config.head_v_dim,
                conv_kernel_size=config.conv_kernel_size,
                norm_eps=config.norm_eps,
            ),
            program_config,
            layer_name="GDN",
            sequence_parallel_axis=sp_axis,
            tensor_parallel_axis=tp_axis,
            active_seq_len=active_seq_len,
            tt_ccl=tt_ccl,
            scalar_decay=True,
        )
        if weights is not None and state_dict is not None:
            raise ValueError("pass either constructed GDNWeights or host state_dict, not both")
        if weights is None:
            weights = load_gdn_weights(
                mesh_device,
                config,
                state_dict,
                weight_cache_path,
                cache_name_prefix=f"layer_{layer_idx}.gdn",
                tensor_parallel_axis=tp_axis,
            )
        if weights.tensor_parallel_size != tensor_parallel_size or weights.tensor_parallel_axis != tp_axis:
            raise ValueError(
                "GDNWeights placement does not match the layer mesh: "
                f"weights TP={weights.tensor_parallel_size} axis={weights.tensor_parallel_axis}, "
                f"layer TP={tensor_parallel_size} axis={tp_axis}"
            )
        self.device = mesh_device
        self.config = config
        self.weights = weights
        self.tt_ccl = tt_ccl
        widths = gdn_input_projection_widths(config, tensor_parallel_size)
        offsets = {}
        start = 0
        for name, width in widths.items():
            offsets[name] = start
            start += width
        # Column ranges of one rank's projection block; ``a`` excludes its tile padding.
        local_value_heads = self._skeleton.geometry.value_heads
        self._columns = {
            "qkv": (0, offsets["z"]),
            "z": (offsets["z"], offsets["z"] + widths["z"]),
            "a": (offsets["a"], offsets["a"] + local_value_heads),
            "b": (offsets["b"], offsets["b"] + widths["b"]),
        }

    def allocate_state(self, batch_size: int = 1) -> KdaState:
        """Allocate the canonical device-resident GDN carries for one prefill stream."""
        return self._skeleton.allocate_state(batch_size)

    def _column(self, projected: ttnn.Tensor, name: str) -> ttnn.Tensor:
        return slice_width(projected, *self._columns[name])

    def _compute_gates(self, projected: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """FP32 ``g = decay_scale * softplus(a + dt_bias)`` and ``beta = sigmoid(b)``, ``[1, T, HV_local]`` each."""
        weights = self.weights
        a = ttnn.typecast(self._column(projected, "a"), _GATE_DTYPE, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        biased = ttnn.add(a, weights.decay_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate = ttnn.multiply(
            weights.decay_scale,
            biased,
            input_tensor_b_activations=[
                ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, _SOFTPLUS_BETA, _SOFTPLUS_THRESHOLD)
            ],
            dtype=_GATE_DTYPE,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = ttnn.typecast(self._column(projected, "b"), _GATE_DTYPE, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        beta = ttnn.sigmoid(b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return gate, beta

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        state: KdaState,
        actual_start: ttnn.Tensor,
        actual_end: ttnn.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, KdaState]:
        """Run prefill GDN and return replacement logical carries.

        Same contract as ``ttKDA.forward``: ``actual_start`` / optional ``actual_end`` are caller-owned replicated
        UINT32 row-major device scalars (32-aligned global positions, kept alive at their captured addresses during
        trace use); the input state is only read; padded output rows are unspecified and the returned carries stop
        at the valid end. The output is sequence-partitioned along SP and reduce-scattered on the hidden dimension
        when TP > 1.
        """
        skeleton = self._skeleton
        skeleton.validate_forward(hidden_states, state, actual_start, actual_end)
        selections = skeleton.selections(actual_start, actual_end)
        projected = skeleton.project_input(hidden_states, self.weights.input_projection)
        q, k, v, new_convolution = skeleton.convolve(
            self._column(projected, "qkv"), state.convolution, selections, actual_start, self.weights.convolution_taps
        )
        gate, beta = self._compute_gates(projected)
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
        output_gate = self._column(projected, "z")
        output = skeleton.gated_rms_norm(result.output, output_gate, self.weights.norm)
        if self.config.output_gate_activation == "silu":
            output = ttnn.multiply(output, output_gate, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = skeleton.project_output(output, self.weights.output_projection)
        return output, KdaState(recurrent=result.final_state, convolution=new_convolution)
