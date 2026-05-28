# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""SwiGLU MLP for Devstral-2 / Ministral3 with simple TP=mesh-axis-N.

HF ``Ministral3MLP``::

    down_proj(silu(gate_proj(x)) * up_proj(x))

We follow ``Ministral3Config.base_model_tp_plan``:
  - ``gate_proj``, ``up_proj``: **column-parallel** (split ``intermediate_size`` across devices).
  - ``down_proj``: **row-parallel** (split ``intermediate_size`` across devices), then **all-reduce**
    along the TP axis.

Activations entering / leaving this module are **replicated** along the TP axis (full ``hidden_size``
on every device). That keeps the boundary contract symmetric with HF and avoids the need to
all-gather inside the layer. Inside the layer, intermediate activations of width
``intermediate_size / TP`` live on each device.
"""

from __future__ import annotations

from typing import Optional

import ttnn

from models.experimental.devstral2_123B_instruct.tt.ccl_helpers import all_reduce_replicate
from models.experimental.devstral2_123B_instruct.tt.mem_config import (
    get_compute_kernel_config,
    get_compute_kernel_config_hifi4,
    get_decode_width_sharded_matmul_output_mem_config,
    get_decode_width_sharded_matmul_program_config,
    get_linear_program_config,
    get_prefill_width_sharded_matmul_output_mem_config,
    get_prefill_width_sharded_matmul_program_config,
    use_width_sharded_decode_norm_matmul,
    use_width_sharded_prefill_norm_matmul,
)
from models.experimental.devstral2_123B_instruct.tt.model_args import Devstral2Args
from models.experimental.devstral2_123B_instruct.tt.weight_loading import (
    resolve_weight_cache_path,
    upload_matmul_weight,
)

__all__ = ["TtMLP"]


class TtMLP:
    """Two-stage SwiGLU MLP with column-then-row TP and a single all-reduce.

    ``hidden_size``-wide activations come in replicated across TP; ``hidden_size``-wide activations
    leave replicated across TP (post all-reduce).
    """

    def __init__(
        self,
        args: Devstral2Args,
        mesh_device,
        state_dict: dict,
        layer_idx: int,
        tt_ccl,
        *,
        dtype: Optional[ttnn.DataType] = None,
        weight_cache_path: Optional[str] = None,
    ) -> None:
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.dtype = dtype or args.weight_dtype

        prefix = args.state_dict_prefix("mlp", layer_idx)
        gate_w = state_dict[prefix + "gate_proj.weight"]
        up_w = state_dict[prefix + "up_proj.weight"]
        down_w = state_dict[prefix + "down_proj.weight"]

        wp = resolve_weight_cache_path(weight_cache_path, args)
        # Quantize weights to bfloat8_b for DRAM bandwidth; matmul outputs stay bfloat16 with
        # HiFi4 fidelity. Quantizing outputs to bf8_b compounds across 88 layers and drops PCC.
        self.gate_proj_weight_dtype = ttnn.bfloat8_b
        self.gate_proj_output_dtype = ttnn.bfloat16
        self.gate_proj = upload_matmul_weight(
            gate_w,
            mesh_device,
            args,
            dtype=self.gate_proj_weight_dtype,
            shard_dim=-1,
            weight_cache_path=wp,
            cache_key=f"{prefix}gate_proj_bfp8",
        )
        self.up_proj_weight_dtype = ttnn.bfloat8_b
        self.up_proj_output_dtype = ttnn.bfloat16
        self.up_proj = upload_matmul_weight(
            up_w,
            mesh_device,
            args,
            dtype=self.up_proj_weight_dtype,
            shard_dim=-1,
            weight_cache_path=wp,
            cache_key=f"{prefix}up_proj_bfp8",
        )
        self.down_proj_weight_dtype = ttnn.bfloat8_b
        self.down_proj_output_dtype = ttnn.bfloat16
        self.down_proj = upload_matmul_weight(
            down_w,
            mesh_device,
            args,
            dtype=self.down_proj_weight_dtype,
            shard_dim=-2,
            weight_cache_path=wp,
            cache_key=f"{prefix}down_proj_bfp8",
        )

        self._compute_kernel_config = get_compute_kernel_config(mesh_device)
        self._compute_kernel_config_hifi4 = get_compute_kernel_config_hifi4(mesh_device)

    def __call__(self, x: ttnn.Tensor, *, mode: str = "decode") -> ttnn.Tensor:
        act_mem = self.args.get_activation_mem_config(mode, self.mesh_device)
        seq_len = max(1, int(x.shape[-2]))

        def _linear(
            inp,
            weight,
            *,
            kind: str,
            activation: Optional[str] = None,
            fused_activation: Optional[str] = None,
            output_dtype: Optional[ttnn.DataType] = None,
            compute_kernel_config=None,
            memory_config: Optional[ttnn.MemoryConfig] = None,
            program_config: Optional[ttnn.ProgramConfig] = None,
        ) -> ttnn.Tensor:
            n = int(weight.shape[-1])
            k = int(weight.shape[-2])
            if program_config is None:
                program_config = get_linear_program_config(
                    self.args,
                    self.mesh_device,
                    mode=mode,
                    kind=kind,
                    seq_len=seq_len,
                    k=k,
                    n=n,
                    fused_activation=fused_activation,
                )
            return ttnn.linear(
                inp,
                weight,
                dtype=output_dtype if output_dtype is not None else self.args.activation_dtype,
                memory_config=memory_config if memory_config is not None else act_mem,
                activation=activation,
                program_config=program_config,
                compute_kernel_config=compute_kernel_config
                if compute_kernel_config is not None
                else self._compute_kernel_config,
            )

        def _decode_width_sharded_linear(
            inp,
            weight,
            *,
            kind: str,
            fused_activation: Optional[str] = None,
            output_dtype: Optional[ttnn.DataType] = None,
            compute_kernel_config=None,
            keep_sharded: bool = False,
        ) -> ttnn.Tensor:
            """Width-sharded matmul on the decode norm 8×8 grid."""
            n = int(weight.shape[-1])
            out = _linear(
                inp,
                weight,
                kind=kind,
                activation=None,
                output_dtype=output_dtype,
                compute_kernel_config=compute_kernel_config,
                memory_config=get_decode_width_sharded_matmul_output_mem_config(),
                program_config=get_decode_width_sharded_matmul_program_config(
                    self.args,
                    self.mesh_device,
                    n=n,
                    fused_activation=fused_activation,
                ),
            )
            if not keep_sharded and out.memory_config().is_sharded():
                out = ttnn.sharded_to_interleaved(out, act_mem)
            return out

        def _prefill_width_sharded_linear(
            inp,
            weight,
            *,
            kind: str,
            fused_activation: Optional[str] = None,
            output_dtype: Optional[ttnn.DataType] = None,
            compute_kernel_config=None,
            keep_sharded: bool = False,
        ) -> ttnn.Tensor:
            """Width-sharded matmul on the prefill norm grid.

            Fused unary ops (e.g. SiLU on gate) must live in ``program_config.fused_activation`` only;
            ``ttnn.linear(activation=...)`` is rejected for sharded matmul.
            """
            n = int(weight.shape[-1])
            out = _linear(
                inp,
                weight,
                kind=kind,
                activation=None,
                output_dtype=output_dtype,
                compute_kernel_config=compute_kernel_config,
                memory_config=get_prefill_width_sharded_matmul_output_mem_config(),
                program_config=get_prefill_width_sharded_matmul_program_config(
                    self.args,
                    self.mesh_device,
                    seq_len=seq_len,
                    n=n,
                    fused_activation=fused_activation,
                ),
            )
            if not keep_sharded and out.memory_config().is_sharded():
                out = ttnn.sharded_to_interleaved(out, act_mem)
            return out

        def _swiglu_mul(gate: ttnn.Tensor, up: ttnn.Tensor) -> ttnn.Tensor:
            """``silu(gate) * up``; keep width-sharded when both matmul outputs share a layout."""
            if gate.memory_config().is_sharded() and gate.memory_config() == up.memory_config():
                inner = ttnn.mul(gate, up, memory_config=gate.memory_config())
                if inner.memory_config().is_sharded():
                    inner = ttnn.sharded_to_interleaved(inner, act_mem)
                return inner
            return ttnn.mul(gate, up, memory_config=act_mem)

        use_ws = use_width_sharded_prefill_norm_matmul(self.args, mode, seq_len)
        use_decode_ws = use_width_sharded_decode_norm_matmul(self.args, mode)
        if use_ws:
            gate = _prefill_width_sharded_linear(
                x,
                self.gate_proj,
                kind="gate",
                fused_activation="silu",
                output_dtype=self.gate_proj_output_dtype,
                compute_kernel_config=self._compute_kernel_config_hifi4,
                keep_sharded=True,
            )
            up = _prefill_width_sharded_linear(
                x,
                self.up_proj,
                kind="up",
                output_dtype=self.up_proj_output_dtype,
                compute_kernel_config=self._compute_kernel_config_hifi4,
                keep_sharded=True,
            )
        elif use_decode_ws:
            gate = _decode_width_sharded_linear(
                x,
                self.gate_proj,
                kind="gate",
                fused_activation="silu",
                output_dtype=self.gate_proj_output_dtype,
                compute_kernel_config=self._compute_kernel_config_hifi4,
                keep_sharded=True,
            )
            up = _decode_width_sharded_linear(
                x,
                self.up_proj,
                kind="up",
                output_dtype=self.up_proj_output_dtype,
                compute_kernel_config=self._compute_kernel_config_hifi4,
                keep_sharded=True,
            )
        else:
            gate = _linear(
                x,
                self.gate_proj,
                kind="gate",
                fused_activation="silu",
                output_dtype=self.gate_proj_output_dtype,
                compute_kernel_config=self._compute_kernel_config_hifi4,
            )
            up = _linear(
                x,
                self.up_proj,
                kind="up",
                output_dtype=self.up_proj_output_dtype,
                compute_kernel_config=self._compute_kernel_config_hifi4,
            )
        inner = _swiglu_mul(gate, up) if (use_ws or use_decode_ws) else ttnn.mul(gate, up, memory_config=act_mem)
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        down = _linear(
            inner,
            self.down_proj,
            kind="down",
            output_dtype=self.down_proj_output_dtype,
            compute_kernel_config=self._compute_kernel_config_hifi4,
        )
        ttnn.deallocate(inner)

        return all_reduce_replicate(
            down,
            mesh_device=self.mesh_device,
            tt_ccl=self.tt_ccl,
            dim=3,
            cluster_axis=self.args.cluster_axis,
            topology=self.args.ccl_topology,
            memory_config=self.args.get_ccl_output_mem_config(mode, self.mesh_device),
        )

    def forward(self, x: ttnn.Tensor, *, mode: str = "decode") -> ttnn.Tensor:
        return self(x, mode=mode)
