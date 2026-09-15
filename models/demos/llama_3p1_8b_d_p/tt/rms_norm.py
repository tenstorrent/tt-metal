# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Llama-3.1-8B RMSNorm for the TP-sharded prefill residual stream (tt-blaze#4146).

Consumes the TP-sharded residual ``[1, 1, seq, emb_dim / tp]`` and returns the **replicated
full-width** ``[1, 1, seq, emb_dim]`` that ``tt/attention.py`` and ``tt/mlp.py`` both take as input.
That pairing is the whole residual-stream contract for this model: norms widen, attention and the
MLP narrow (they end in a ``reduce_scatter``), and the residual add is always a local elementwise op
on the narrow layout.

**Why this gathers the activations instead of running the 3-op distributed norm.** The rest of this
fleet (``deepseek_v3_d_p/tt/tt_distributed_rms_norm.py``, ``models/common/rmsnorm.py``,
``minimax_m3/tt/rms_norm.py``) computes ``rms_norm_pre_all_gather`` -> all-gather the *statistics* ->
``rms_norm_post_all_gather``, which keeps the output ``emb_dim / tp`` wide. That is the right shape
when the consumer accepts a shard — DeepSeek's MLA does, and its dense FFN path then all-gathers
separately (``tt_prefill_block.py:_dense_ffn_path``). Llama's consumers need full width, so the
activation all-gather is unavoidable, and once it is paid the distributed-stats path is strictly
worse: it moves the same activation bytes *plus* a statistics all-gather, and computes the norm in
two passes instead of one. Gathering first and running a single full-width ``ttnn.rms_norm`` is one
collective, one op, and numerically exact rather than reassembled from partial sums of squares.
``minimax_m3/tt/residual.py`` reaches the same conclusion — ``gather_first`` is its default.

The float32 upcast inside the reduction is not cosmetic: HF's ``LlamaRMSNorm`` normalises in float32
(see ``reference/model.py:Llama31RMSNorm``), and a bf16 reduction over 4096 elements moves
end-of-sequence logits enough to read as a real gap. ``ttnn.rms_norm`` is given an fp32-accumulate
compute config for the same reason.
"""

from pathlib import Path
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig

# fp32 destination accumulation for the mean-of-squares reduction — the device-side counterpart of
# HF's float32 upcast. HiFi4 because the norm is cheap next to the matmuls it feeds.
COMPUTE_KERNEL_CONFIG_NORM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=False,
)


class TtLlamaRMSNorm(LightweightModule):
    """RMSNorm: TP-sharded in, replicated full-width out."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        mesh_config,
        torch_weight: Optional[torch.Tensor] = None,
        emb_dim: int = Llama31_8BConfig.EMB_SIZE,
        eps: float = Llama31_8BConfig.RMS_NORM_EPS,
        num_links: int = 1,
        topology: ttnn.Topology = ttnn.Topology.Linear,
        weights_dtype: ttnn.DataType = ttnn.bfloat16,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_NORM,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
    ):
        """
        Args:
            torch_weight: the ``[emb_dim]`` gamma. Ones when omitted, which makes the module a pure
                normaliser — useful for bring-up, never for PCC.
        """
        super().__init__()

        if mesh_device.shape[mesh_config.tp_axis] != mesh_config.tp:
            raise ValueError(
                f"mesh_config.tp({mesh_config.tp}) != mesh_device.shape[{mesh_config.tp_axis}]"
                f"({mesh_device.shape[mesh_config.tp_axis]})"
            )
        if emb_dim % mesh_config.tp:
            raise ValueError(f"emb_dim({emb_dim}) is not divisible by tp({mesh_config.tp})")

        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.emb_dim = emb_dim
        self.emb_dim_per_chip = mesh_config.shard_size(emb_dim)
        self.eps = eps
        self.num_links = num_links
        self.topology = topology
        self.compute_kernel_config = compute_kernel_config

        if torch_weight is None:
            logger.warning("TtLlamaRMSNorm built with unit gamma — bring-up only, not valid for PCC")
            torch_weight = torch.ones(emb_dim, dtype=torch.float32)
        if tuple(torch_weight.shape) != (emb_dim,):
            raise ValueError(f"expected gamma of shape {(emb_dim,)}, got {tuple(torch_weight.shape)}")

        cache_file_name = (
            str(weight_cache_path / f"{cache_name_prefix}.norm_weight")
            if weight_cache_path is not None and cache_name_prefix is not None
            else None
        )
        # Replicated, full width: the norm runs after the activation gather, so every chip needs the
        # whole gamma. (A TP-sharded gamma is what the distributed-stats norm wants — see the module
        # docstring for why this model does not take that path.)
        self.weight = ttnn.as_tensor(
            torch_weight.reshape(1, 1, 1, emb_dim),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=weights_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
        )

    def forward(self, x: ttnn.Tensor, ccl_manager=None) -> ttnn.Tensor:
        """``x``: TP-sharded ``[1, 1, seq, emb_dim / tp]`` -> replicated ``[1, 1, seq, emb_dim]``.

        At TP=1 there is nothing to gather and the input is already full width; that path exercises
        the norm but not the collective.
        """
        tp = self.mesh_device.shape[self.mesh_config.tp_axis]

        if tp == 1:
            if x.shape[-1] != self.emb_dim:
                raise ValueError(f"at tp=1 the input must already be full width: {x.shape[-1]} != {self.emb_dim}")
            gathered = x
        else:
            if x.shape[-1] != self.emb_dim_per_chip:
                raise ValueError(
                    f"input last dim {x.shape[-1]} != emb_dim/tp {self.emb_dim_per_chip}: this module "
                    f"expects the TP-sharded residual, not a replicated full-width activation"
                )
            if ccl_manager is not None:
                gathered = self.mesh_config.allgather(x, ccl_manager, dim=-1)
            else:
                gathered = ttnn.all_gather(
                    x,
                    dim=-1,
                    cluster_axis=self.mesh_config.tp_axis,
                    num_links=self.num_links,
                    topology=self.topology,
                )

        out = ttnn.rms_norm(
            gathered,
            epsilon=self.eps,
            weight=self.weight,
            compute_kernel_config=self.compute_kernel_config,
        )
        if gathered is not x:
            ttnn.deallocate(gathered)
        return out
