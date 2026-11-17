# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""2D tensor-parallel SwiGLU MLP: the shared expert and the dense FFN with every weight sharded once over the mesh.

``TtSharedExpert`` / ``TtFfn`` shard the intermediate dim over ONE mesh axis (TP) and replicate the weights over the
other (SP), which on the 4x8 Galaxy stores each shared expert / dense FFN 4x (8x on the dev 8x4 SP layout). The SC1
placement of #65884 (Hy4-preview, #65890) shards every weight once over all 32 chips. This module does that:

  mesh axis ``hidden_axis`` (size nh) splits the model hidden dim H     (the axis the block's residual shards H over)
  mesh axis ``inter_axis``  (size ni) splits the MLP intermediate dim I (the other axis; prefill puts tokens on it)

  gate_up  [H, 2I] -> per chip [H/nh, 2I/ni]: rows over hidden_axis, columns over inter_axis. Gate and up are fused
                      so each chip's column block is ``[gate_i | up_i]`` for its intermediate slice i (one matmul and
                      one all-reduce instead of two; ``fuse_gate_up``).
  down     [I, H]  -> per chip [I/ni, H/nh]: rows over inter_axis, columns over hidden_axis.

Forward, per chip, x ``[..., T, H/nh]`` (hidden sharded over hidden_axis):

  1. tokens_sharded (prefill: the block's sequence is split over inter_axis):
         all_gather(x, dim=-2, inter_axis)                  -> [..., T*ni, H/nh]
  2. gu = x @ gate_up                                       -> [..., S, 2I/ni]   partial over hidden_axis
  3. all_reduce(gu, hidden_axis)                            -> [..., S, 2I/ni]
  4. act = silu(g) * u  (or clamped: silu(min(g, L)) * clamp(u, +-L))   [..., S, I/ni]
  5. y = act @ down                                         -> [..., S, H/nh]    partial over inter_axis
  6. tokens_sharded: reduce_scatter(y, dim=-2, inter_axis)  -> [..., T, H/nh]   (back to the input layout)
     else (decode: tokens replicated over inter_axis): all_reduce(y, inter_axis) -> [..., S, H/nh]

The output has the input's layout, so it adds straight onto the residual (and onto the routed experts' output,
which ``TtMoe`` reduce-scatters to the same ``[T, H/nh]``). Steps 1/6 are the prefill form; with
``tokens_sharded=False`` the same weights serve decode (a few tokens replicated over the mesh), where steps 3 and 6
are the only collectives.

Which mesh axis gets H is the RESIDUAL's choice, not this module's: H goes wherever the block keeps the hidden
shards, so the input needs no redistribution. In the prefill block the residual is ``[S / sp, H / tp]``, so
``hidden_axis = tp_axis`` and ``inter_axis = sp_axis``; on the SC1 4x8 layout (SP 4, TP 8) that is H over the 8-axis
and I over the 4-axis. If the residual moves H onto the 4-axis (as #65884's attention table suggests), pass that
axis; nothing else changes.

Cache files: ``{prefix}.gate_up_proj_tp2d`` and ``{prefix}.down_proj_tp2d``, distinct from the 1D modules' so a
1D cache never loads as a 2D one (or back).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.utils.weight_cache_dump import weight_cache_dump_mode

ACTIVATION_SILU = "silu"
ACTIVATION_CLAMPED_SILU_GLU = "clamped_silu_glu"
SUPPORTED_ACTIVATIONS = (ACTIVATION_SILU, ACTIVATION_CLAMPED_SILU_GLU)

GATE_UP_CACHE_NAME = "gate_up_proj_tp2d"
DOWN_CACHE_NAME = "down_proj_tp2d"

# fp32 dest accumulation: the dense layer-0 down projection is K = 18432 / ni deep (4608 on SC1) and #65923 asks
# PCC >= 0.99 of it; bf16 accumulation over that depth costs more than the fp32 dest does.
COMPUTE_KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

PerAxis = Union[ttnn.Topology, Sequence[ttnn.Topology]]


def ffn_2d_tp(model_cfg) -> bool:
    """Whether a model config runs its shared expert and dense FFN as ``TtMlp2D`` (``FFN_2D_TP``; default off)."""
    return bool(getattr(model_cfg, "FFN_2D_TP", False))


def fuse_gate_up(gate_proj: torch.Tensor, up_proj: torch.Tensor, n_inter: int) -> torch.Tensor:
    """``[H, 2I]`` from HF ``gate_proj`` / ``up_proj`` ``[I, H]``: column block i of width ``2I / n_inter`` is
    ``[gate_i | up_i]``, the gate and up columns of intermediate slice i. Sharding the result's columns ``n_inter``
    ways hands each chip its own gate and up slice side by side (``split_gate_up`` undoes one block)."""
    inter, hidden = gate_proj.shape
    if up_proj.shape != gate_proj.shape:
        raise ValueError(f"gate_proj {tuple(gate_proj.shape)} and up_proj {tuple(up_proj.shape)} differ")
    if inter % n_inter:
        raise ValueError(f"intermediate {inter} does not split {n_inter} ways")
    g = gate_proj.T.reshape(hidden, n_inter, inter // n_inter)
    u = up_proj.T.reshape(hidden, n_inter, inter // n_inter)
    return torch.cat([g, u], dim=-1).reshape(hidden, 2 * inter).contiguous()


def shard_dims(hidden_axis: int, row_dim_axis: str) -> tuple:
    """ShardTensor2dMesh dims for a 2D weight: ``row_dim_axis`` names which logical axis the ROWS (-2) go over
    ("hidden" or "inter"); the columns (-1) go over the other mesh axis."""
    inter_axis = 1 - hidden_axis
    rows_axis = hidden_axis if row_dim_axis == "hidden" else inter_axis
    dims = [None, None]
    dims[rows_axis] = -2
    dims[1 - rows_axis] = -1
    return tuple(dims)


def _per_axis(value, axis: int):
    """``value`` if scalar, else its entry for mesh ``axis`` (the block passes per-axis tuples)."""
    if isinstance(value, (tuple, list)):
        return value[axis]
    return value


class TtMlp2D(LightweightModule):
    """SwiGLU MLP with gate / up / down sharded over both mesh axes (module docstring)."""

    @staticmethod
    def check_cache_complete(cache_path: Path, cache_name_prefix: str) -> bool:
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        for name in (GATE_UP_CACHE_NAME, DOWN_CACHE_NAME):
            if not pattern_exists(f"{cache_name_prefix}.{name}*.tensorbin", "Mlp2D"):
                logger.debug(f"TTNN cache missing: {cache_name_prefix}.{name}")
                return False
        return True

    @staticmethod
    def _convert(
        torch_weights: Optional[dict],
        *,
        emb_dim: int,
        hidden_dim: int,
        mesh_device,
        hidden_axis: int,
        weights_dtype,
        cache_path: Optional[Path],
        cache_name_prefix: Optional[str],
        device,
    ):
        n_inter = mesh_device.shape[1 - hidden_axis]
        n_hidden = mesh_device.shape[hidden_axis]
        if hidden_dim % (n_inter * ttnn.TILE_SIZE) or emb_dim % (n_hidden * ttnn.TILE_SIZE):
            raise ValueError(
                f"TtMlp2D: intermediate {hidden_dim} over {n_inter} and hidden {emb_dim} over {n_hidden} chips must "
                f"leave whole tiles per chip"
            )
        if torch_weights is not None:
            gate_up = fuse_gate_up(torch_weights["gate_proj"], torch_weights["up_proj"], n_inter)
            down = torch_weights["down_proj"].T.contiguous()
        else:
            gate_up = torch.empty(emb_dim, 2 * hidden_dim)
            down = torch.empty(hidden_dim, emb_dim)

        def _to_ttnn(t, rows_over, name):
            cache = (
                None
                if cache_path is None or cache_name_prefix is None
                else str(cache_path / f"{cache_name_prefix}.{name}")
            )
            return ttnn.as_tensor(
                t,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims(hidden_axis, rows_over)
                ),
                layout=ttnn.TILE_LAYOUT,
                device=device,
                dtype=weights_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if device is not None else None,
                cache_file_name=cache,
                cache_dump_mode=weight_cache_dump_mode(),
            )

        gate_up_tt = _to_ttnn(gate_up, "hidden", GATE_UP_CACHE_NAME)
        down_tt = _to_ttnn(down, "inter", DOWN_CACHE_NAME)
        if device is None:
            return None
        return gate_up_tt, down_tt

    @staticmethod
    def build_ttnn_cache(
        torch_weights: dict,
        *,
        emb_dim: int,
        hidden_dim: int,
        mesh_device,
        hidden_axis: int,
        weights_dtype,
        cache_path: Path,
        cache_name_prefix: str,
    ):
        """Write the weight cache without a device copy."""
        TtMlp2D._convert(
            torch_weights,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            mesh_device=mesh_device,
            hidden_axis=hidden_axis,
            weights_dtype=weights_dtype,
            cache_path=cache_path,
            cache_name_prefix=cache_name_prefix,
            device=None,
        )

    def __init__(
        self,
        mesh_device,
        *,
        emb_dim: int,
        hidden_dim: int,
        hidden_axis: int,
        torch_weights: Optional[dict] = None,
        weights_dtype=ttnn.bfloat8_b,
        activations_dtype=ttnn.bfloat16,
        activation: str = ACTIVATION_SILU,
        clamped_silu_glu_limit: Optional[float] = None,
        num_links: Union[int, Sequence[int]] = 1,
        topology: PerAxis = ttnn.Topology.Linear,
        weight_cache_path: Optional[Path] = None,
        cache_name_prefix: Optional[str] = None,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG,
    ):
        """``num_links`` / ``topology``: scalars, or per-mesh-axis pairs (indexed by mesh axis, as the prefill block
        and ``TtMoe`` carry them). Weights come from ``torch_weights`` (HF ``gate_proj`` / ``up_proj`` /
        ``down_proj``, ``[out, in]``) or, when None, from the cache at ``weight_cache_path``."""
        super().__init__()
        if hidden_axis not in (0, 1):
            raise ValueError(f"hidden_axis must be 0 or 1, got {hidden_axis}")
        if activation not in SUPPORTED_ACTIVATIONS:
            raise ValueError(f"TtMlp2D: unknown activation {activation!r}; expected one of {SUPPORTED_ACTIVATIONS}")
        if activation == ACTIVATION_CLAMPED_SILU_GLU and not (clamped_silu_glu_limit and clamped_silu_glu_limit > 0):
            raise ValueError(f"activation {activation!r} needs a positive clamped_silu_glu_limit")
        if torch_weights is None and weight_cache_path is None:
            raise ValueError("TtMlp2D needs torch_weights or a weight_cache_path to load from")
        self.mesh_device = mesh_device
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.hidden_axis = hidden_axis
        self.inter_axis = 1 - hidden_axis
        self.n_hidden = mesh_device.shape[self.hidden_axis]
        self.n_inter = mesh_device.shape[self.inter_axis]
        self.inter_per_chip = hidden_dim // self.n_inter
        self.activations_dtype = activations_dtype
        self.weights_dtype = weights_dtype
        self.activation = activation
        self.clamped_silu_glu_limit = clamped_silu_glu_limit
        self.compute_kernel_config = compute_kernel_config
        self.num_links = num_links
        self.topology = topology
        # TtMoe reads these on its (sub-device) overlap path; TtMlp2D never runs confined to a sub-device.
        self.subdevice_id = None
        self.subdevice_cores = None
        self.gate_up_proj, self.down_proj = self._convert(
            torch_weights,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            mesh_device=mesh_device,
            hidden_axis=hidden_axis,
            weights_dtype=weights_dtype,
            cache_path=weight_cache_path,
            cache_name_prefix=cache_name_prefix,
            device=mesh_device,
        )
        logger.debug(
            f"TtMlp2D {cache_name_prefix}: H={emb_dim} over mesh axis {hidden_axis} ({self.n_hidden}), I={hidden_dim} "
            f"over mesh axis {self.inter_axis} ({self.n_inter}); per chip gate_up {list(self.gate_up_proj.shape)}, "
            f"down {list(self.down_proj.shape)}, {activation}"
        )

    def _ccl_kwargs(self, axis: int) -> dict:
        return dict(
            cluster_axis=axis,
            num_links=_per_axis(self.num_links, axis),
            topology=_per_axis(self.topology, axis),
        )

    def forward(self, x: ttnn.Tensor, *, tokens_sharded: bool = True) -> ttnn.Tensor:
        """``x`` ``[..., T, H / nh]`` -> same shape and layout (module docstring). ``tokens_sharded``: the tokens are
        split over ``inter_axis`` (prefill); False when every chip of that axis holds the same tokens (decode)."""
        assert x.shape[-1] * self.n_hidden == self.emb_dim, (
            f"TtMlp2D expects the hidden dim sharded {self.n_hidden} ways over mesh axis {self.hidden_axis}: "
            f"got {x.shape[-1]} per chip for H={self.emb_dim}"
        )
        if x.dtype != self.activations_dtype:
            x = ttnn.typecast(x, self.activations_dtype)
        gathered = tokens_sharded and self.n_inter > 1
        if gathered:
            x = ttnn.all_gather(x, dim=-2, **self._ccl_kwargs(self.inter_axis))

        gu = ttnn.matmul(
            x,
            self.gate_up_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        if gathered:
            ttnn.deallocate(x)
        if self.n_hidden > 1:
            reduced = ttnn.all_reduce(gu, memory_config=ttnn.DRAM_MEMORY_CONFIG, **self._ccl_kwargs(self.hidden_axis))
            ttnn.deallocate(gu)
            gu = reduced

        rank = len(gu.shape)
        starts = [0] * rank
        mid = list(gu.shape)
        mid[-1] = self.inter_per_chip
        gate = ttnn.slice(gu, starts, mid)
        up_start = [0] * rank
        up_start[-1] = self.inter_per_chip
        up = ttnn.slice(gu, up_start, list(gu.shape))
        ttnn.deallocate(gu)
        if self.activation == ACTIVATION_CLAMPED_SILU_GLU:
            act = ttnn.clamped_silu_glu(gate, up, self.clamped_silu_glu_limit)
        else:
            act = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)

        y = ttnn.matmul(
            act,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(act)
        if self.n_inter > 1:
            if tokens_sharded:
                out = ttnn.reduce_scatter(
                    y, dim=-2, memory_config=ttnn.DRAM_MEMORY_CONFIG, **self._ccl_kwargs(self.inter_axis)
                )
            else:
                out = ttnn.all_reduce(y, memory_config=ttnn.DRAM_MEMORY_CONFIG, **self._ccl_kwargs(self.inter_axis))
            ttnn.deallocate(y)
            y = out
        return y
