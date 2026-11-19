# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""2D tensor-parallel SwiGLU MLP: the shared expert and the dense FFN with every weight sharded once over the mesh.

``TtSharedExpert`` / ``TtFfn`` shard the intermediate dim over ONE mesh axis (TP) and replicate the weights over the
other (SP), which on the 8x4 Galaxy (SP 8 x TP 4) stores each shared expert / dense FFN 8x. The SC1
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

Prefill runs steps 1-6 on the prefill's persistent collectives (``prefill_ccl="persistent"``, the default):
step 1 is a ``high_bw_all_gather`` into a TT_CCL buffer with the shared gather semaphores, step 3 is a fused
``matmul_reduce_scatter_prefill`` over hidden_axis (fp32 accumulation) followed by the same gather, and steps 5-6 are
one fused ``matmul_reduce_scatter_prefill`` over inter_axis. ``ttnn.all_gather`` / ``all_reduce`` /
``reduce_scatter`` allocate global semaphores per program in L1_SMALL; the prefill opens the mesh with 1536 B of it,
which sparse MLA's collectives fill, so next to ttMLA2DTP they fail ("Not enough space to allocate 1920 B L1_SMALL",
``tests/test_hy4_attn_mlp_2d_coexist.py``). ``prefill_ccl="generic"`` keeps those ops; decode always uses them.

The output has the input's layout, so it adds straight onto the residual (and onto the routed experts' output,
which ``TtMoe`` reduce-scatters to the same ``[T, H/nh]``). Steps 1/6 are the prefill form; with
``tokens_sharded=False`` the same weights serve decode (a few tokens replicated over the mesh), where steps 3 and 6
are the only collectives.

Which mesh axis gets H is the RESIDUAL's choice, not this module's: H goes wherever the block keeps the hidden
shards, so the input needs no redistribution. In the prefill block the residual is ``[S / sp, H / tp]``, so
``hidden_axis = tp_axis`` and ``inter_axis = sp_axis``. On SC1 (#65884: hidden over its 4 rows, the other dimension
over its 8 columns) that is the physical 8x4 mesh with TP 4 on axis 1 and SP 8 on axis 0: H over the 4-chip axis 1
(1536 per chip) and I over the 8-chip axis 0 (shared expert 2048 / 8 = 256, dense layer 0 18432 / 8 = 2304 per chip),
residual ``[S / 8, 6144 / 4]``, the same as 2D-TP attention (#65885) and iHC (#65891).

Cache files: ``{prefix}.gate_up_proj_{layout}`` and ``{prefix}.down_proj_{layout}`` with the layout derived from the
mesh (``Mlp2DLayout.tag``, ``utils/sharded_cache.layout_tag``: mesh shape, and the size and mesh axis of H and of I;
``gate_up_proj_8x4_h4ax1_i8ax0`` on SC1), distinct from the 1D modules' names and from any other 2D split's. Next to
them ``{prefix}.mlp2d_{layout}.layout.stamp`` holds the layout descriptor (module, ``CACHE_FORMAT``, mesh, axes,
weights dtype); shards without it, or with a stamp that disagrees, refuse to load. The first SC1 split (H on the
8-chip axis 0) cached as ``gate_up_proj_tp2d`` / ``down_proj_tp2d`` and never loads. Model configs name the layout
their caches use (``FFN_2D_MESH`` / ``FFN_2D_HIDDEN_AXIS``, ``ffn_2d_layout``) so the cache checks look for the right
files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.utils.sharded_cache import LayoutCacheGuard, layout_descriptor, layout_tag
from models.demos.deepseek_v3_d_p.utils.weight_cache_dump import weight_cache_dump_mode

# Prefill collectives: "persistent" (default) runs on the prefill's persistent TT_CCL buffers / semaphores and fp32
# matmul+reduce-scatter; "generic" is ttnn.all_gather / all_reduce / reduce_scatter (per-program L1_SMALL semaphores).
PREFILL_CCL_PERSISTENT = "persistent"
PREFILL_CCL_GENERIC = "generic"

ACTIVATION_SILU = "silu"
ACTIVATION_CLAMPED_SILU_GLU = "clamped_silu_glu"
SUPPORTED_ACTIVATIONS = (ACTIVATION_SILU, ACTIVATION_CLAMPED_SILU_GLU)

GATE_UP_CACHE_NAME = "gate_up_proj"
DOWN_CACHE_NAME = "down_proj"
CACHE_FORMAT = 1  # bump when the shards change without an axis moving


@dataclass(frozen=True)
class Mlp2DLayout:
    """Where ``TtMlp2D``'s dims go: H over mesh axis ``hidden_axis``, I over the other, on a ``mesh_shape`` mesh."""

    mesh_shape: tuple
    hidden_axis: int

    def __post_init__(self):
        if self.hidden_axis not in (0, 1):
            raise ValueError(f"hidden_axis must be 0 or 1, got {self.hidden_axis}")
        object.__setattr__(self, "mesh_shape", tuple(int(n) for n in self.mesh_shape))

    @classmethod
    def of(cls, mesh_device, hidden_axis: int) -> "Mlp2DLayout":
        return cls(tuple(mesh_device.shape), hidden_axis)

    @property
    def axes(self) -> dict:
        return {"h": self.hidden_axis, "i": 1 - self.hidden_axis}

    @property
    def tag(self) -> str:
        """``{rows}x{cols}_h{n}ax{a}_i{n}ax{a}``, e.g. ``8x4_h4ax1_i8ax0`` on SC1."""
        return layout_tag(self.mesh_shape, self.axes)


def cache_names(layout: Mlp2DLayout) -> tuple[str, str]:
    """(gate_up, down) cache basenames for this layout."""
    return f"{GATE_UP_CACHE_NAME}_{layout.tag}", f"{DOWN_CACHE_NAME}_{layout.tag}"


def cache_guard(cache_path, cache_name_prefix: str, layout: Mlp2DLayout, weights_dtype=ttnn.bfloat8_b):
    """The layout stamp of one TtMlp2D's cache files (``utils/sharded_cache.py``)."""
    descriptor = layout_descriptor(
        "TtMlp2D",
        CACHE_FORMAT,
        layout.mesh_shape,
        layout.axes,
        {"gate_up_proj": weights_dtype, "down_proj": weights_dtype},
    )
    return LayoutCacheGuard(
        cache_path,
        f"{cache_name_prefix}.mlp2d_{layout.tag}",
        f"{cache_name_prefix}.*_proj_{layout.tag}*.tensorbin",
        descriptor,
    )


# fp32 dest accumulation: the dense layer-0 down projection is K = 18432 / ni deep (2304 on SC1) and #65923 asks
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


def ffn_2d_layout(model_cfg) -> Mlp2DLayout:
    """The mesh and H axis a ``FFN_2D_TP`` model runs TtMlp2D on (``FFN_2D_MESH``, ``FFN_2D_HIDDEN_AXIS``: its prefill
    mesh and TP axis), which its cache names carry."""
    if not ffn_2d_tp(model_cfg):
        raise ValueError(f"{getattr(model_cfg, '__name__', model_cfg)} does not set FFN_2D_TP")
    return Mlp2DLayout(tuple(model_cfg.FFN_2D_MESH), int(model_cfg.FFN_2D_HIDDEN_AXIS))


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
    def check_cache_complete(
        cache_path: Path, cache_name_prefix: str, layout: Mlp2DLayout, weights_dtype=ttnn.bfloat8_b
    ) -> bool:
        """Complete only for ``layout``'s files with a matching layout stamp: a cache cut for another split, or
        unstamped, does not count."""
        from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import pattern_exists

        for name in cache_names(layout):
            if not pattern_exists(f"{cache_name_prefix}.{name}*.tensorbin", "Mlp2D"):
                logger.debug(f"TTNN cache missing: {cache_name_prefix}.{name}")
                return False
        return cache_guard(cache_path, cache_name_prefix, layout, weights_dtype).stamped()

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

        layout = Mlp2DLayout.of(mesh_device, hidden_axis)
        cached = cache_path is not None and cache_name_prefix is not None
        guard = cache_guard(cache_path if cached else None, cache_name_prefix or "", layout, weights_dtype)
        guard.check()  # shards from another layout (or unstamped) refuse to load

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

        gate_up_name, down_name = cache_names(layout)
        gate_up_tt = _to_ttnn(gate_up, "hidden", gate_up_name)
        down_tt = _to_ttnn(down, "inter", down_name)
        guard.write()
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
        prefill_ccl: str = PREFILL_CCL_PERSISTENT,
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
        if prefill_ccl not in (PREFILL_CCL_PERSISTENT, PREFILL_CCL_GENERIC):
            raise ValueError(f"prefill_ccl must be {PREFILL_CCL_PERSISTENT!r} or {PREFILL_CCL_GENERIC!r}")
        self.prefill_ccl = prefill_ccl
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

    def _act(self, gu: ttnn.Tensor, free_input: bool = True) -> ttnn.Tensor:
        """[..., 2I/ni] = [gate_i | up_i] -> SwiGLU (optionally clamped): [..., I/ni]. Frees ``gu`` unless it is a
        persistent buffer (``free_input=False``)."""
        rank = len(gu.shape)
        mid = list(gu.shape)
        mid[-1] = self.inter_per_chip
        gate = ttnn.slice(gu, [0] * rank, mid)
        up_start = [0] * rank
        up_start[-1] = self.inter_per_chip
        up = ttnn.slice(gu, up_start, list(gu.shape))
        if free_input:
            ttnn.deallocate(gu)
        if self.activation == ACTIVATION_CLAMPED_SILU_GLU:
            act = ttnn.clamped_silu_glu(gate, up, self.clamped_silu_glu_limit)
        else:
            act = ttnn.mul(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        ttnn.deallocate(gate)
        ttnn.deallocate(up)
        return act

    def _gather(self, x: ttnn.Tensor, axis: int, dim: int, name: str) -> ttnn.Tensor:
        """All-gather ``dim`` over mesh ``axis`` into a persistent TT_CCL buffer with the prefill's shared gather
        semaphores (no per-program L1_SMALL semaphores). The result aliases the buffer: do not free it."""
        from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

        tt_ccl = get_tt_ccl(self.mesh_device)
        shape = [1, 1, x.shape[-2], x.shape[-1]]
        shape[dim] *= self.mesh_device.shape[axis]
        out = tt_ccl.get_mla_high_bw_all_gather_buffer(
            name=f"mlp2d_{name}", shape=shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        ready, data_valid = tt_ccl.get_mla_high_bw_gather_semaphores()
        return ttnn.experimental.high_bw_all_gather(
            x,
            dim=dim,
            output_tensor=out,
            cluster_axis=axis,
            num_links=_per_axis(self.num_links, axis),
            ready_semaphore=ready,
            data_valid_semaphore=data_valid,
        )

    def _mm_rs(self, x: ttnn.Tensor, w: ttnn.Tensor, axis: int, scatter_dim: int) -> ttnn.Tensor:
        """``x @ w`` with the partials summed in fp32 over mesh ``axis`` and scattered on ``scatter_dim``."""
        return ttnn.experimental.matmul_reduce_scatter_prefill(
            x,
            w,
            cluster_axis=axis,
            scatter_dim=scatter_dim,
            topology=_per_axis(self.topology, axis),
            num_links=_per_axis(self.num_links, axis),
        )

    def _prefill_forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Prefill (tokens split over ``inter_axis``) on the prefill's persistent CCLs only. Same math as the generic
        path, with the hidden all-reduce as fp32 matmul+reduce-scatter (N) then a gather, and the output
        reduce-scatter fused into the down matmul. ``ttnn.all_gather`` / ``all_reduce`` / ``reduce_scatter``
        allocate global semaphores per program in L1_SMALL, which the prefill (1536 B) cannot spare next to sparse
        MLA's collectives."""
        shape = list(x.shape)
        x4 = ttnn.reshape(x, [1, 1, shape[-2], shape[-1]])
        xg = self._gather(x4, self.inter_axis, 2, "x") if self.n_inter > 1 else x4  # [S, H/nh]
        if self.n_hidden > 1:
            gu = self._mm_rs(xg, self.gate_up_proj, self.hidden_axis, -1)  # [S, 2I/(ni*nh)], fp32-reduced
            gu_full = self._gather(gu, self.hidden_axis, 3, "gu")  # [S, 2I/ni] = [gate_i | up_i], persistent
            ttnn.deallocate(gu)
            act = self._act(gu_full, free_input=False)
        else:
            gu = ttnn.matmul(
                xg,
                self.gate_up_proj,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            act = self._act(gu)
        if self.n_inter > 1:
            y = self._mm_rs(act, self.down_proj, self.inter_axis, -2)  # [T, H/nh], fp32-reduced
        else:
            y = ttnn.matmul(
                act,
                self.down_proj,
                compute_kernel_config=self.compute_kernel_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
        ttnn.deallocate(act)
        return ttnn.reshape(y, shape[:-2] + list(y.shape)[-2:])

    def forward(self, x: ttnn.Tensor, *, tokens_sharded: bool = True) -> ttnn.Tensor:
        """``x`` ``[..., T, H / nh]`` -> same shape and layout (module docstring). ``tokens_sharded``: the tokens are
        split over ``inter_axis`` (prefill); False when every chip of that axis holds the same tokens (decode)."""
        assert x.shape[-1] * self.n_hidden == self.emb_dim, (
            f"TtMlp2D expects the hidden dim sharded {self.n_hidden} ways over mesh axis {self.hidden_axis}: "
            f"got {x.shape[-1]} per chip for H={self.emb_dim}"
        )
        if x.dtype != self.activations_dtype:
            x = ttnn.typecast(x, self.activations_dtype)
        if tokens_sharded and self.prefill_ccl == PREFILL_CCL_PERSISTENT:
            return self._prefill_forward(x)
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

        act = self._act(gu)

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
