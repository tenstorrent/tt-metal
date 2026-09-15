# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B dense SwiGLU MLP for the TTNN prefill model (tt-blaze#4140).

    gate_out  = x @ gate_proj                     column-parallel, 4096 -> 1792/chip
    up_out    = x @ up_proj                       column-parallel, 4096 -> 1792/chip
    activated = silu(gate_out) * up_out           fused SiLU on the ttnn.mul
    out_full  = activated @ down_proj             row-parallel,   1792/chip -> 4096 partial
    out       = reduce_scatter(out_full, dim=-1)  -> 4096/tp per chip

**Layout contract.** ``forward`` consumes a *replicated* ``[..., emb_dim]`` activation (what the
distributed ``ffn_norm`` emits) and returns a ``reduce_scatter``-ed ``[..., emb_dim / tp]``, which is
the layout the residual stream is already in. That output layout is a contract, not an
implementation detail: the residual add downstream is TP-sharded on hidden, so returning a
convenient full-width DRAM-interleaved tensor here would force an unplanned collective at every
layer boundary. See ``deepseek_v3_d_p/tt/tt_prefill_block.py``, whose post-attention residual
carries the same "TP-sharded on hidden" comment.

**Reuse note.** tt-blaze#4140 points at ``gpt_oss_d_p/tt/mlp.py`` as "the same op with biases and
the SwiGLU-OAI activation". That file is actually a thin MoE wrapper (router + expert-parallel
experts, ``NotImplementedError`` on a single device); the "same op" remark describes the expert
*kernel*, not a reusable dense MLP. The dense SwiGLU in this family is
``deepseek_v3_d_p/tt/tt_ffn.py``, and this module follows its ``forward()``. It deliberately does
*not* inherit ``TtFfn``/``TtSharedExpert``: that base exists for MoE shared-expert weight
construction and sub-device-aware tuning, carries DeepSeek's 7168/18432 dims and a SiTU-GLU branch
Llama never takes, and inheriting it would couple Llama's prefill to the MoE substrate. The weight
handling below is self-contained instead.

Kept free of reference-model and safetensors imports so this module stays cheap to import; the
adapter's import-light contract is asserted by ``tests/unit/test_scaffold.py``.
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig

# HiFi2 with packer L1 accumulation — the fidelity the rest of the ``_d_p`` prefill family runs its
# FFN matmuls at (``tt_shared_expert.COMPUTE_KERNEL_CONFIG_HIFI2``). Named ``Wormhole...`` upstream
# but it is the generic compute-kernel config and is what Blackhole models use too.
COMPUTE_KERNEL_CONFIG_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


class TtLlamaMLP(LightweightModule):
    """Dense SwiGLU MLP, TP-sharded over ``mesh_config.tp_axis``."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        mesh_config,
        torch_weights: Optional[dict] = None,
        emb_dim: int = Llama31_8BConfig.EMB_SIZE,
        hidden_dim: int = Llama31_8BConfig.INTERMEDIATE_SIZE,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        activations_dtype: ttnn.DataType = ttnn.bfloat16,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_HIFI2,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
    ):
        """
        Args:
            mesh_config: ``tt/config.py`` ``MeshConfig``. Supplies the TP degree, the mesh axis TP
                lives on, and the column/row-parallel mesh mappers. Passed in rather than rebuilt
                here so a caller cannot end up with an MLP sharded on a different axis than the
                attention block next to it.
            torch_weights: ``gate_proj`` / ``up_proj`` / ``down_proj`` in HF ``(out, in)``
                orientation. Random weights are used when omitted, which is only useful for shape
                and memory bring-up — never for PCC.
            weights_dtype: ``bfloat16`` by default. #4140 specifies bf16 compute and a PCC floor of
                0.99; ``bfloat8_b`` (the DeepSeek FFN default, graded at 0.97) does not reliably
                clear 0.99 on this shape, so the tighter default is the one that matches the
                acceptance criterion. Callers trading accuracy for DRAM can still pass bfloat8_b.
        """
        super().__init__()

        tp = mesh_config.tp
        if mesh_device.shape[mesh_config.tp_axis] != tp:
            raise ValueError(
                f"mesh_config.tp({tp}) != mesh_device.shape[{mesh_config.tp_axis}]"
                f"({mesh_device.shape[mesh_config.tp_axis]}); the weight mappers shard the whole "
                f"TP axis, so a mismatch places data on devices the config does not know about."
            )
        for name, total in (("emb_dim", emb_dim), ("hidden_dim", hidden_dim)):
            if total % tp:
                raise ValueError(f"{name}({total}) is not divisible by tp({tp})")
            shard = total // tp
            # Llama-3.1-8B at TP=8 gives 512 and 1792, i.e. 16 and 56 whole tiles. Asserted rather
            # than assumed: an off-tile shard width does not fail here, it silently pads, and the
            # producer's padded shard width is baked into the consumer's matmul program config.
            if shard % ttnn.TILE_SIZE:
                raise ValueError(
                    f"{name}({total}) / tp({tp}) = {shard} is not a multiple of the "
                    f"{ttnn.TILE_SIZE}-wide tile; this shape would be silently padded"
                )

        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_per_chip = mesh_config.shard_size(hidden_dim)
        self.emb_dim_per_chip = mesh_config.shard_size(emb_dim)
        self.num_links = num_links
        self.topology = topology
        self.activations_dtype = activations_dtype
        self.weights_dtype = weights_dtype
        self.compute_kernel_config = compute_kernel_config
        self.weight_cache_path = weight_cache_path
        self.cache_name_prefix = cache_name_prefix

        if torch_weights is not None:
            missing = [p for p in PROJECTIONS if p not in torch_weights]
            if missing:
                raise ValueError(f"torch_weights is missing {missing}; expected all of {list(PROJECTIONS)}")
        else:
            logger.warning("TtLlamaMLP built with random weights — shape bring-up only, not valid for PCC")

        def weight(name: str, shape: tuple[int, int], column_parallel: bool) -> ttnn.Tensor:
            if torch_weights is not None:
                # HF (out, in) -> ttnn.matmul's (in, out).
                w = torch_weights[name].T.contiguous()
                if tuple(w.shape) != shape:
                    raise ValueError(
                        f"{name} has HF shape {tuple(torch_weights[name].shape)}, which transposes to "
                        f"{tuple(w.shape)}; expected {shape} for emb_dim={emb_dim}, hidden_dim={hidden_dim}"
                    )
            else:
                w = torch.randn(*shape, dtype=torch.float32) * 0.02
            return self._to_sharded_ttnn(w, name, column_parallel)

        # gate/up are column-parallel (shard the output/feature dim); down is row-parallel (shard
        # the input dim) so its partial sums reduce over the TP axis.
        self.gate_proj = weight("gate_proj", (emb_dim, hidden_dim), column_parallel=True)
        self.up_proj = weight("up_proj", (emb_dim, hidden_dim), column_parallel=True)
        self.down_proj = weight("down_proj", (hidden_dim, emb_dim), column_parallel=False)

        logger.debug(
            f"TtLlamaMLP: emb_dim={emb_dim} hidden_dim={hidden_dim} tp={tp} "
            f"-> {self.hidden_dim_per_chip}/chip intermediate, {self.emb_dim_per_chip}/chip out"
        )

    def _to_sharded_ttnn(self, torch_weight: torch.Tensor, name: str, column_parallel: bool) -> ttnn.Tensor:
        """Shard a ``(in, out)`` weight across the TP axis and move it to DRAM."""
        mesh_mapper = (
            self.mesh_config.column_parallel(self.mesh_device)
            if column_parallel
            else self.mesh_config.row_parallel(self.mesh_device)
        )
        cache_file_name = (
            str(self.weight_cache_path / f"{self.cache_name_prefix}.{name}")
            if self.weight_cache_path is not None and self.cache_name_prefix is not None
            else None
        )
        return ttnn.as_tensor(
            torch_weight,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            dtype=self.weights_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``x``: replicated ``[..., emb_dim]`` -> ``[..., emb_dim / tp]`` (reduce-scattered).

        With TP=1 there is nothing to reduce and the output is already full width; that path
        exercises the matmuls and the fused activation but *not* the collective.
        """
        if x.shape[-1] != self.emb_dim:
            raise ValueError(
                f"input last dim {x.shape[-1]} != emb_dim {self.emb_dim}: this module expects the "
                f"replicated full-width activation that ffn_norm emits, not a TP-sharded one"
            )
        if x.dtype != self.activations_dtype:
            logger.warning(f"TtLlamaMLP: typecasting input {x.dtype} -> {self.activations_dtype}")
            x = ttnn.typecast(x, self.activations_dtype)

        gate_out = ttnn.matmul(x, self.gate_proj, compute_kernel_config=self.compute_kernel_config)
        up_out = ttnn.matmul(x, self.up_proj, compute_kernel_config=self.compute_kernel_config)

        # SiLU is fused onto the gate input of the multiply, so the activation never round-trips
        # through its own op (and its output tensor).
        activated = ttnn.mul(gate_out, up_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate_out)
        ttnn.deallocate(up_out)

        out_full = ttnn.matmul(activated, self.down_proj, compute_kernel_config=self.compute_kernel_config)
        ttnn.deallocate(activated)

        if self.mesh_device.shape[self.mesh_config.tp_axis] == 1:
            return out_full

        # Row-parallel down_proj leaves each chip holding a partial sum over the full emb_dim.
        # reduce_scatter both completes the sum and lands the result TP-sharded, which is the
        # layout the residual stream is in — an all_reduce would cost an extra full-width tensor
        # and then need scattering anyway.
        out = ttnn.reduce_scatter(
            out_full,
            dim=-1,
            cluster_axis=self.mesh_config.tp_axis,
            num_links=self.num_links,
            topology=self.topology,
        )
        ttnn.deallocate(out_full)
        return out
