# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""LTX-2.5 DiffVAE video decoder: deterministic stages.

Every block is 3D neighborhood attention over a local window. Stages 1-4 deterministically
upsample the latent into a context volume; the diffusion stage that turns noise plus that
context into pixels lives in :mod:`diffvae_ltx_stage5`.

Submodules are named to mirror the checkpoint's own keys (``attn.qkv``, ``mlp.w_gate``, ...).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import torch
from safetensors import safe_open

import ttnn

from ...layers.feedforward import CHUNK_BYTES, SwiGLU
from ...layers.linear import Linear
from ...layers.module import Module, ModuleList
from ...layers.neighborhood_attention import NAKernel, neighborhood_attention_3d, resolve_na_kernel
from ...layers.neighborhood_attention_plan import NA3DDevicePlan, build_device_plan, plan_na3d
from ...layers.normalization import RMSNorm
from ...utils import timing_tree
from ...utils.ltx import read_vae_per_channel_stats
from ...utils.memory_log import log_ccl_cache, log_dram
from ...utils.tensor import depth_to_space_channels_last, prepare_depth_to_space_channels
from ...utils.tracing import traced_function
from .diffvae_ltx_stage5 import DiffVAEStage5, DiffVAEStage5Config, Grid
from .diffvae_ops import (
    TILE,
    consume,
    consume_all,
    device_major_qkv,
    mesh_axis_size,
    retile,
    split_qkv,
    to_row_major,
    wshard,
)
from .diffvae_rope import ROPE_BASE, axis_angles, default_rope_dim_split, rope_permutation


@dataclass(frozen=True)
class DetBlockOptions:
    """Which forms the deterministic NA blocks build.

    These change the parameter SET (one fused ``qkv`` or three projections, one packed ``gate_up``
    or two), so they are part of the weight-cache key; see :meth:`DiffVAEDecoder.parameter_layout`.
    """

    #: One fused qkv GEMM, split by ``nlp_create_qkv_heads``.
    fused_qkv: bool = False
    #: The fused qkv weight sharded on its output axis over ``tp_axis``. Implies ``fused_qkv``.
    colpar_qkv: bool = False
    #: ``rotary_embedding_hf`` while still TILE. Needs ``fused_qkv``.
    fused_rope: bool = False
    #: One ``[up | gate]`` GEMM whose epilogue emits ``silu(gate) * up``.
    fused_swiglu: bool = False
    #: gate/up column-parallel and w_down row-parallel over ``tp_axis``. Implies ``fused_swiglu``.
    tp_mlp: bool = False

    def resolve(self, tp_axis: int | None) -> DetBlockOptions:
        """The forms a block with this ``tp_axis`` actually builds.

        Stage 1 runs replicated with no TP axis, so the column-parallel forms fall away there and
        the plain fused ones stand; stages 2-4 get fused qkv from ``colpar_qkv`` alone.
        """
        colpar = self.colpar_qkv and tp_axis is not None
        tp_mlp = self.tp_mlp and tp_axis is not None
        fused_qkv = self.fused_qkv or colpar
        return DetBlockOptions(
            fused_qkv=fused_qkv,
            colpar_qkv=colpar,
            fused_rope=self.fused_rope and fused_qkv,
            fused_swiglu=self.fused_swiglu or tp_mlp,
            tp_mlp=tp_mlp,
        )


@dataclass(frozen=True)
class DiffVAEOptions:
    """Everything about HOW :class:`DiffVAEDecoder` runs that is not in the checkpoint's architecture.

    Resolved once by whoever builds the decoder and handed down; nothing below the decoder reads
    the environment. :meth:`production` is the configuration the runner scripts ship.
    """

    #: Stage-5 executor: ``"linear_order"`` (replicated), ``"bricked"`` (replicated, bricked order)
    #: or ``"bricked_sp_w_sharded"`` (this chip's W-band over ``stage5_sp_axis``).
    stage5_backend: str = "linear_order"
    stage5_sp_axis: int | None = None
    #: TP-over-heads on the orthogonal mesh axis, only under a W-sharded backend.
    stage5_tp_axis: int | None = None
    #: Deterministic stages 1-3 executor; the same names, minus ``"bricked"``. Stage 1 always runs
    #: replicated on ``"linear_order"`` (its W does not divide the mesh axis).
    stages_backend: str = "linear_order"
    stages_sp_axis: int | None = None
    stages_tp_axis: int | None = None
    #: Deterministic block forms.
    det: DetBlockOptions = DetBlockOptions()
    #: One fused stage-5 qkv GEMM, split by slicing the packed output.
    stage5_fused_qkv: bool = False
    #: Column-parallel stage-5 qkv over ``stage5_tp_axis``; each chip projects only its heads.
    stage5_tp_proj: bool = True
    #: Generalized Neighborhood Attention query-group stride, physical (t, h, w). (1, 1, 1) is the
    #: shipped architecture.
    gna_stride: tuple[int, int, int] = (1, 1, 1)
    #: Frames per stage-5 band, or None to run the volume whole.
    slab_frames: int | None = None
    #: The decode's two host boundaries run on device: ghost pad and flatten on the way in; x_t
    #: noise, tile-pad trim and depth-to-space on the way out. The traced decode needs it, and the
    #: output side needs a W-sharded stage 5.
    device_boundaries: bool = False
    #: Whether the pipeline evicts a resident DiT before decoding. None: exclusive unless stage 5
    #: is sharded, which is when it holds a fraction of the volume per chip and fits beside the DiT.
    exclusive_residency: bool | None = None

    @classmethod
    def production(cls, *, slab_frames: int | None = 78, tp_heads: bool = True) -> DiffVAEOptions:
        """The 4x8 1080p configuration the runner scripts ship: both halves W-sharded on the bricked
        executor over the columns axis, TP-over-heads on the rows axis, every fusion on."""
        tp_axis = 0 if tp_heads else None
        return cls(
            stage5_backend="bricked_sp_w_sharded",
            stage5_sp_axis=1,
            stage5_tp_axis=tp_axis,
            stages_backend="bricked_sp_w_sharded",
            stages_sp_axis=1,
            stages_tp_axis=tp_axis,
            det=DetBlockOptions(fused_qkv=True, colpar_qkv=True, fused_rope=True, fused_swiglu=True),
            slab_frames=slab_frames,
            device_boundaries=True,
        )


def decoder_config(path) -> dict:
    """The decoder's architecture block, read from the checkpoint's safetensors metadata."""
    with safe_open(str(path), "pt") as handle:
        vae = json.loads(handle.metadata()["config"])["vae"]
    config = dict(vae["decoder"])
    for key in ("in_channels", "out_channels", "patch_size", "head_dim", "model_output_type"):
        if key in vae:
            config[key] = vae[key]
    for key in ("stage_kernels", "upsamples", "stage5_kernel"):
        if key in config:
            config[key] = _tuplify(config[key])
    for key in ("stage_channels", "stage_depths"):
        if key in config:
            config[key] = tuple(config[key])
    return config


def _tuplify(value):
    return tuple(_tuplify(v) if isinstance(v, list) else v for v in value)


def read_decoder_tensors(path, prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    """The ``decoder.*`` tensors of an LTX-2.5 video-VAE safetensors file whose name (without the
    ``decoder.`` prefix) starts with one of ``prefixes``, as float32 under that name."""
    state: dict[str, torch.Tensor] = {}
    with safe_open(str(path), "pt") as handle:
        for key in handle.keys():
            if not key.startswith("decoder."):
                continue
            name = key[len("decoder.") :]
            if name.startswith(prefixes):
                state[name] = handle.get_tensor(key).float()
    return state


def rope_tables(
    dims: tuple[int, int, int],
    rope_dim_split: tuple[int, int, int],
    *,
    mesh_device,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """``(cos, sin)`` of shape ``(1, T, H, W, 1, head_dim // 2)`` for the permuted layout.

    Each axis contributes ``width // 2`` columns in T, H, W order, matching
    :func:`rope_permutation`. Positions are local to the volume: a global phase cancels inside
    the window's softmax, so local and absolute positions give the same attention.
    """
    t, h, w = dims
    cos_columns, sin_columns = [], []
    for axis, (length, width) in enumerate(zip(dims, rope_dim_split)):
        angle = axis_angles(torch.arange(length), width, ROPE_BASE)
        shape = [1, 1, 1, angle.shape[-1]]
        shape[axis] = length
        cos_columns.append(angle.cos().reshape(shape).expand(t, h, w, angle.shape[-1]))
        sin_columns.append(angle.sin().reshape(shape).expand(t, h, w, angle.shape[-1]))

    def upload(columns):
        table = torch.cat(columns, dim=-1).reshape(1, t, h, w, 1, -1)
        return ttnn.from_torch(table, device=mesh_device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

    return upload(cos_columns), upload(sin_columns)


def apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    """Rotate a permuted ``(1, T, H, W, heads, head_dim)`` tensor using contiguous halves. **Consumes** ``x``.

    The two output halves are built one after the other so at most two products are live at once.
    """
    shape = list(x.shape)
    half = shape[-1] // 2
    low = ttnn.slice(x, [0] * len(shape), shape[:-1] + [half])
    high = ttnn.slice(x, [0] * (len(shape) - 1) + [half], shape[:-1] + [2 * half])
    ttnn.deallocate(x)

    first = consume_all(ttnn.subtract, ttnn.multiply(low, cos), ttnn.multiply(high, sin))
    second = consume_all(ttnn.add, ttnn.multiply(low, sin), ttnn.multiply(high, cos))
    ttnn.deallocate(low)
    ttnn.deallocate(high)

    rotated = ttnn.concat([first, second], dim=-1)
    ttnn.deallocate(first)
    ttnn.deallocate(second)
    return rotated


class NeighborhoodAttention(Module):
    """3D neighborhood attention with absolute RoPE, matching upstream's parameter shell."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        *,
        head_dim: int = 64,
        mesh_device=None,
        na3d_backend: str | NAKernel = "linear_order",
        ccl_manager=None,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
        options: DetBlockOptions = DetBlockOptions(),
    ):
        super().__init__()
        assert dim % head_dim == 0, f"dim={dim} not divisible by head_dim={head_dim}"
        self.dim = dim
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5
        self.mesh_device = mesh_device
        # "linear_order": grouped gather + dense masked attention with the passed-in device_plan.
        # "bricked_sp_w_sharded": this chip's W-shard through the bricked executor (halo exchange
        # over sp_axis via ccl_manager).
        self.kernel = resolve_na_kernel(na3d_backend)
        self.na3d_backend = self.kernel.name
        self.ccl_manager = ccl_manager
        self.sp_axis = sp_axis
        # TP-over-heads on the orthogonal mesh axis, only under the W-sharded backend.
        self.tp_axis = tp_axis
        self.rope_dim_split = default_rope_dim_split(head_dim)

        opts = options.resolve(tp_axis)
        # colpar_qkv: the fused qkv weight sharded on its output axis, so each chip's matmul computes
        # only its own heads.
        self.colpar_qkv = opts.colpar_qkv
        # fused_qkv: one qkv matmul split by nlp_create_qkv_heads, partitioning the heads over
        # tp_axis first so the norms, scale and RoPE run on heads/tp. At tp=1 the partition is
        # skipped and the TILE (B, NH, S, HD) layout stands alone.
        self.fused_qkv = opts.fused_qkv
        # fused_rope: one rotary_embedding_hf per lane, applied while still TILE. The weight fold
        # already puts q/k in HF's rotate_half convention; it needs full-width cos/sin.
        self.fused_rope = opts.fused_rope
        self._fused_rope_cache: dict = {}
        self.tp = mesh_axis_size(mesh_device, tp_axis) if tp_axis is not None else 1
        if self.fused_qkv:
            assert self.num_heads % self.tp == 0, f"num_heads={self.num_heads} not divisible by tp={self.tp}"
        # The W-sharded executor never slices heads itself under TP, so on that backend this block
        # partitions the heads in every projection form.
        self.bricked = self.kernel.w_sharded
        self.heads_local = self.num_heads // self.tp if (self.fused_qkv or self.bricked) else self.num_heads

        if self.fused_qkv:
            qkv_linear = {"bias": True, "mesh_device": mesh_device}
            if self.colpar_qkv:
                qkv_linear["weight_mesh_axes"] = [None, tp_axis]
                qkv_linear["bias_mesh_axes"] = [None, tp_axis]
            self.qkv = Linear(dim, 3 * dim, **qkv_linear)
        else:
            self.to_q = Linear(dim, dim, bias=True, mesh_device=mesh_device)
            self.to_k = Linear(dim, dim, bias=True, mesh_device=mesh_device)
            self.to_v = Linear(dim, dim, bias=True, mesh_device=mesh_device)
        self.proj = Linear(dim, dim, bias=True, mesh_device=mesh_device)
        self.q_norm = RMSNorm(head_dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)
        self.k_norm = RMSNorm(head_dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Split (or regroup) the shipped fused ``qkv`` and fold the RoPE reordering into q/k."""
        perm = rope_permutation(self.rope_dim_split)

        def reorder_head_dim(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(self.num_heads, self.head_dim, *tensor.shape[1:])[:, perm].reshape(tensor.shape)

        for leaf in ("weight", "bias"):
            key = f"qkv.{leaf}"
            if key in state:
                q, k, v = split_qkv(state.pop(key))
                q, k = reorder_head_dim(q), reorder_head_dim(k)
                if self.fused_qkv:
                    state[key] = device_major_qkv(torch.cat([q, k, v], dim=0), self.tp)
                else:
                    state[f"to_q.{leaf}"] = q
                    state[f"to_k.{leaf}"] = k
                    state[f"to_v.{leaf}"] = v
        for key in ("q_norm.weight", "k_norm.weight"):
            if key in state:
                state[key] = state[key][perm]

    def _fused_rope_tables(self, cos: ttnn.Tensor, sin: ttnn.Tensor, tokens: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``(cos, sin)`` as ``rotary_embedding_hf`` wants them: full head_dim (halves doubled), flat, TILE."""
        key = (tokens, int(cos.shape[-1]))
        cached = self._fused_rope_cache.get(key)
        if cached is None:

            def prepare(table: ttnn.Tensor) -> ttnn.Tensor:
                doubled = ttnn.concat([table, table], dim=-1)
                flat = ttnn.reshape(doubled, (1, 1, tokens, self.head_dim))
                out = ttnn.to_layout(flat, ttnn.TILE_LAYOUT)
                ttnn.deallocate(doubled)
                return out

            cached = (prepare(cos), prepare(sin))
            self._fused_rope_cache[key] = cached
        return cached

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        dims: tuple[int, int, int],
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        device_plan: NA3DDevicePlan,
    ) -> ttnn.Tensor:
        t, h, w = dims
        tokens = t * h * w
        heads = self.heads_local
        q, k, v = self._project_qkv(x, tokens)
        q, k = self._norm_and_scale(q, k)

        if self.fused_rope:
            with timing_tree.span(self.mesh_device, "qkv-rope (fused op)", category=timing_tree.NORM_ROPE, deep=True):
                cos_full, sin_full = self._fused_rope_tables(cos, sin, tokens)
                q = consume(q, ttnn.experimental.rotary_embedding_hf, cos_full, sin_full)
                k = consume(k, ttnn.experimental.rotary_embedding_hf, cos_full, sin_full)

        # Untilize before splitting out the head axis: TILE pads both of the last two dims to 32,
        # so a trailing (heads, head_dim) in TILE costs many times its own size.
        shape = (1, t, h, w, heads, self.head_dim)

        def to_volume(part: ttnn.Tensor) -> ttnn.Tensor:
            part = consume(part, ttnn.to_layout, ttnn.ROW_MAJOR_LAYOUT)
            if self.fused_qkv:
                part = consume(part, ttnn.permute, (0, 2, 1, 3))
            return ttnn.reshape(part, shape)

        with timing_tree.span(self.mesh_device, "qkv-to-volume", category=timing_tree.RESHAPE, deep=True):
            q, k, v = (to_volume(part) for part in (q, k, v))
        if not self.fused_rope:
            with timing_tree.span(self.mesh_device, "qkv-rope (unfused)", category=timing_tree.NORM_ROPE, deep=True):
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)

        attended = self._attend(q, k, v, dims=dims, device_plan=device_plan)
        return self._out_proj(attended, tokens)

    @timing_tree.span("mesh_device", "qkv-proj", category=timing_tree.PROJ, deep=True)
    def _project_qkv(self, x: ttnn.Tensor, tokens: int) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Project ``x`` to per-head q, k, v, this chip's heads only. **Consumes** ``x``."""
        heads = self.heads_local
        if self.fused_qkv:
            flat = self.qkv(x)
            ttnn.deallocate(x)
            qkv = ttnn.reshape(flat, (1, 1, tokens, int(flat.shape[-1])))
            if not self.colpar_qkv and self.tp > 1:
                partitioned = ttnn.mesh_partition(qkv, dim=3, cluster_axis=self.tp_axis)
                ttnn.deallocate(qkv)
                qkv = partitioned
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv, num_heads=heads, num_kv_heads=heads, transpose_k_heads=False
            )
            ttnn.deallocate(qkv)
            return q, k, v

        heads_shape = (tokens * heads, self.head_dim)

        def own_heads(part: ttnn.Tensor) -> ttnn.Tensor:
            """This chip's contiguous head block of a ``(tokens, dim)`` projection under TP."""
            if self.tp == 1 or not self.bricked:
                return part
            rows = ttnn.reshape(part, (1, 1, tokens, self.dim))
            partitioned = ttnn.mesh_partition(rows, dim=3, cluster_axis=self.tp_axis)
            ttnn.deallocate(part)
            return partitioned

        q, k, v = (ttnn.reshape(own_heads(project(x)), heads_shape) for project in (self.to_q, self.to_k, self.to_v))
        ttnn.deallocate(x)
        return q, k, v

    @timing_tree.span("mesh_device", "qkv-norm", category=timing_tree.NORM_ROPE, deep=True)
    def _norm_and_scale(self, q: ttnn.Tensor, k: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        return ttnn.multiply(self.q_norm(q), self.scale), self.k_norm(k)

    @timing_tree.span(
        "mesh_device", lambda self, *a, **k: f"attention {self.na3d_backend}", category=timing_tree.SDPA, deep=True
    )
    def _attend(self, q, k, v, *, dims: tuple[int, int, int], device_plan: NA3DDevicePlan) -> ttnn.Tensor:
        t, h, w = dims
        if self.kernel.w_sharded:
            # q/k/v are this chip's W-slice, so `dims` here is local; the executor is told the full W.
            dims = (t, h, w * mesh_axis_size(q.device(), self.sp_axis))
        return neighborhood_attention_3d(
            q,
            k,
            v,
            kernel=self.kernel,
            kernel_size=self.kernel_size,
            scale=1.0,
            dims=dims,
            ccl_manager=self.ccl_manager,
            sp_axis=self.sp_axis,
            tp_axis=self.tp_axis,
            heads_presharded=True,
            device_plan=device_plan,
        )

    @timing_tree.span("mesh_device", "out-proj", category=timing_tree.PROJ, deep=True)
    def _out_proj(self, attended: ttnn.Tensor, tokens: int) -> ttnn.Tensor:
        flat = consume(attended, retile, (tokens, self.dim))
        out = self.proj(flat)
        ttnn.deallocate(flat)
        return out


class NABlock(Module):
    """Pre-norm block: neighborhood attention then SwiGLU, both with residual adds."""

    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        *,
        head_dim: int = 64,
        mesh_device=None,
        na3d_backend: str | NAKernel = "linear_order",
        ccl_manager=None,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
        options: DetBlockOptions = DetBlockOptions(),
    ):
        super().__init__()
        opts = options.resolve(tp_axis)
        # Upstream rounds the 4x MLP ratio up to a multiple of 16.
        hidden = (int(dim * 4.0) + 15) // 16 * 16
        self.norm1 = RMSNorm(dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)
        self.attn = NeighborhoodAttention(
            dim,
            kernel_size,
            head_dim=head_dim,
            mesh_device=mesh_device,
            na3d_backend=na3d_backend,
            ccl_manager=ccl_manager,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            options=opts,
        )
        self.norm2 = RMSNorm(dim, norm_eps=1e-6, bias=False, mesh_device=mesh_device)
        self.mlp = SwiGLU(
            dim,
            hidden,
            mesh_device=mesh_device,
            tp_axis=tp_axis,
            ccl_manager=ccl_manager,
            fused=opts.fused_swiglu,
            tp_mlp=opts.tp_mlp,
        )
        self.mesh_device = mesh_device

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        dims: tuple[int, int, int],
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        device_plan: NA3DDevicePlan,
    ) -> ttnn.Tensor:
        attended = self._attention(x, dims=dims, cos=cos, sin=sin, device_plan=device_plan)
        x = ttnn.add(x, attended)
        ttnn.deallocate(attended)
        projected = self._mlp(x)
        x = ttnn.add(x, projected)
        ttnn.deallocate(projected)
        return x

    @timing_tree.span("mesh_device", "attention", category=timing_tree.ATTENTION, deep=True)
    def _attention(self, x, *, dims, cos, sin, device_plan) -> ttnn.Tensor:
        return self.attn(self.norm1(x), dims=dims, cos=cos, sin=sin, device_plan=device_plan)

    @timing_tree.span("mesh_device", "mlp", category=timing_tree.MLP, deep=True)
    def _mlp(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.mlp(self.norm2(x))


class LinearPixelShuffleUpsample(Module):
    """Channel-expanding Linear then a channels-last 3D pixel shuffle.

    The checkpoint packs the projection's output channels as ``(c p1 p2 p3)``. The rows are
    reordered at load to ``(p1 p2 p3 c)`` (``prepare_depth_to_space_channels``, as the conv VAE's
    upsampler does) so the shuffle keeps the channel axis innermost.
    """

    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        out_channels_reduction_factor: int = 1,
        *,
        mesh_device=None,
    ):
        super().__init__()
        self.stride = tuple(stride)
        span = self.stride[0] * self.stride[1] * self.stride[2]
        self.proj_out_channels = span * in_channels // out_channels_reduction_factor
        self.out_channels = self.proj_out_channels // span
        self.proj = Linear(in_channels, self.proj_out_channels, bias=True, mesh_device=mesh_device)

    def _shuffle(self, projected: ttnn.Tensor, t: int, h: int, w: int, drop_leading_frame: bool) -> ttnn.Tensor:
        """Pixel-shuffle a ROW_MAJOR ``(t*h*w, proj_out_channels)`` projection into ``(t'*h*p2*w*p3, c)``
        in TILE. **Consumes** ``projected``; returns the tile tensor and the output frame count."""
        p1, p2, p3 = self.stride
        c = self.out_channels
        projected = consume(
            projected,
            lambda v: depth_to_space_channels_last(ttnn.reshape(v, (1, t, h, w, self.proj_out_channels)), self.stride),
        )
        out_t = t * p1
        rows = (h * p2) * (w * p3)
        if p1 == 2 and drop_leading_frame:
            # The temporal shuffle emits a duplicate first frame; dropping it preserves the causal
            # 1:2 mapping. Only the slab holding the true t=0 has one.
            projected = consume(
                projected, lambda v: ttnn.slice(ttnn.reshape(v, (out_t, rows, c)), [1, 0, 0], [out_t, rows, c])
            )
            out_t -= 1
        projected = consume(projected, retile, (out_t * rows, c))
        return projected, out_t

    def forward(
        self, x: ttnn.Tensor, *, dims: tuple[int, int, int], drop_leading_frame: bool = True
    ) -> tuple[ttnn.Tensor, tuple[int, int, int]]:
        """``x`` is ``(tokens, in_channels)``; returns ``(tokens', out_channels)`` and new dims.

        The shuffle maps each source frame to its own output frames, so when the widened projection
        would exceed :data:`CHUNK_BYTES` the volume is processed in source-frame slabs. A slab
        boundary must be tile-aligned (``h*w`` a multiple of TILE); otherwise it runs whole.
        """
        t, h, w = dims
        p1, p2, p3 = self.stride
        hw = h * w
        slab = max(1, CHUNK_BYTES // (hw * self.proj_out_channels * 2))

        if slab >= t or hw % TILE != 0:
            projected = consume(self.proj(x), ttnn.to_layout, ttnn.ROW_MAJOR_LAYOUT)
            out, out_t = self._shuffle(projected, t, h, w, drop_leading_frame)
            return out, (out_t, h * p2, w * p3)

        in_channels = int(x.shape[-1])
        parts: list[ttnn.Tensor] = []
        out_t_total = 0
        for start in range(0, t, slab):
            st = min(start + slab, t) - start
            x_slab = ttnn.slice(x, [start * hw, 0], [(start + st) * hw, in_channels])
            projected = consume(self.proj(x_slab), ttnn.to_layout, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x_slab)
            part, part_t = self._shuffle(projected, st, h, w, drop_leading_frame and start == 0)
            parts.append(part)
            out_t_total += part_t
        joined = ttnn.concat(parts, dim=-2)
        for part in parts:
            ttnn.deallocate(part)
        return joined, (out_t_total, h * p2, w * p3)


class DeterministicStages(Module):
    """Stages 1-4: NA blocks and upsamples that turn the latent into the stage-5 context.

    ``conv_in`` lives here so the latent's per-channel denormalization can be folded into it:
    ``conv_in(x * std + mean)`` is a Linear with ``std`` scaled into the weight columns and
    ``W @ mean`` added to the bias.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        stage_channels: tuple[int, ...],
        stage_depths: tuple[int, ...],
        stage_kernels: tuple[tuple[int, int, int], ...],
        upsamples: tuple[tuple[tuple[int, int, int], int], ...],
        head_dim: int = 64,
        mesh_device=None,
        ccl_manager=None,
        na3d_backend: str | NAKernel | None = None,
        sp_axis: int | None = None,
        tp_axis: int | None = None,
        block_options: DetBlockOptions = DetBlockOptions(),
    ):
        super().__init__()
        assert len(upsamples) == len(stage_channels) - 1, "one upsample between consecutive stages"
        self.stage_kernels = stage_kernels
        self.head_dim = head_dim
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        # Under a W-sharded backend the activation is W-sharded from stage 1 on (stage 0's W is not
        # divisible by the mesh axis, so it stays replicated). This backend reaches stages 1-4 only;
        # stage 5 has its own.
        self.kernel = resolve_na_kernel(na3d_backend or "linear_order")
        self.na3d_backend = self.kernel.name
        # The replicated bricked kernel is stage 5's; here a block is either replicated linear-order
        # or W-sharded.
        assert self.kernel.name == "linear_order" or self.kernel.w_sharded, (
            f"NA3D backend {self.kernel.name!r} is not a deterministic-stage backend; "
            "expected 'linear_order' or a W-sharded kernel"
        )
        self.sp_axis = sp_axis
        self.tp_axis = tp_axis
        self._w_sharded = self.kernel.w_sharded
        self.sp = mesh_axis_size(mesh_device, sp_axis) if self._w_sharded else 1
        if self._w_sharded:
            assert sp_axis is not None and ccl_manager is not None, f"{self.na3d_backend} needs sp_axis + ccl_manager"
        self.conv_in = Linear(in_channels, stage_channels[0], bias=True, mesh_device=mesh_device)

        def block_backend(stage: int) -> NAKernel:
            if self._w_sharded and stage == 0:
                return resolve_na_kernel("linear_order")
            return self.kernel

        self.block_backend = block_backend

        self.det_stages = ModuleList(
            [
                ModuleList(
                    [
                        NABlock(
                            stage_channels[stage],
                            stage_kernels[stage],
                            head_dim=head_dim,
                            mesh_device=mesh_device,
                            na3d_backend=block_backend(stage),
                            ccl_manager=ccl_manager if self._w_sharded and stage > 0 else None,
                            sp_axis=sp_axis if self._w_sharded and stage > 0 else None,
                            tp_axis=tp_axis if self._w_sharded and stage > 0 else None,
                            options=block_options,
                        )
                        for _ in range(stage_depths[stage])
                    ]
                )
                for stage in range(len(upsamples))
            ]
        )
        self.upsamples = ModuleList(
            [
                LinearPixelShuffleUpsample(
                    stage_channels[stage], upsamples[stage][0], upsamples[stage][1], mesh_device=mesh_device
                )
                for stage in range(len(upsamples))
            ]
        )
        self._plan_cache: dict[tuple, NA3DDevicePlan] = {}
        self._rope_cache: dict[tuple, tuple[ttnn.Tensor, ttnn.Tensor]] = {}

    def _plan(self, dims: tuple[int, int, int], kernel: tuple[int, int, int]) -> NA3DDevicePlan:
        key = (dims, kernel)
        if key not in self._plan_cache:
            self._plan_cache[key] = build_device_plan(
                plan_na3d(dims, kernel), mesh_device=self.mesh_device, ccl_manager=self.ccl_manager
            )
        return self._plan_cache[key]

    def _rope(self, dims: tuple[int, int, int]) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if dims not in self._rope_cache:
            self._rope_cache[dims] = rope_tables(
                dims, default_rope_dim_split(self.head_dim), mesh_device=self.mesh_device
            )
        return self._rope_cache[dims]

    def state_from_checkpoint(self, path, *, statistics: bool = True) -> dict[str, torch.Tensor]:
        """Load ``decoder.*`` tensors from an LTX-2.5 video-VAE safetensors file, folding the file's
        ``per_channel_statistics`` into ``conv_in`` unless ``statistics`` is off."""
        depth = len(self.det_stages)
        state = {
            name: tensor
            for name, tensor in read_decoder_tensors(path, ("conv_in.", "det_stages.", "upsamples.")).items()
            # A decoder built with fewer stages than the file ships ignores the rest.
            if not (name.startswith("det_stages.") and int(name.split(".")[1]) >= depth)
        }

        if statistics:
            mean, std = (stat.flatten() for stat in read_vae_per_channel_stats(path))
            weight, bias = state["conv_in.weight"], state["conv_in.bias"]
            state["conv_in.weight"] = weight * std[None, :]
            state["conv_in.bias"] = bias + weight @ mean

        for index, upsample in enumerate(self.upsamples):
            for leaf in ("weight", "bias"):
                key = f"upsamples.{index}.proj.{leaf}"
                state[key] = prepare_depth_to_space_channels(state[key], upsample.stride)

        return state

    def load_checkpoint(self, path, *, statistics: bool = True) -> None:
        self.load_state_dict(self.state_from_checkpoint(path, statistics=statistics))

    @timing_tree.span("mesh_device", "reshard: replicated -> W-sharded", category=timing_tree.RESHAPE)
    def _wshard(self, x: ttnn.Tensor, dims: tuple[int, int, int]) -> ttnn.Tensor:
        """Reshard a replicated ``(T*H*W, ch)`` volume into this chip's W-band ``(T*H*(W/sp), ch)``.
        **Consumes** ``x``."""
        return wshard(x, dims, sp_axis=self.sp_axis)

    @timing_tree.span("mesh_device", "det -> replicated context gather", category=timing_tree.ALLGATHER)
    def _wgather(self, x: ttnn.Tensor, dims: tuple[int, int, int]) -> ttnn.Tensor:
        """Gather a W-sharded ``(T*H*(W/sp), ch)`` band back to the replicated ``(T*H*W, ch)`` volume.
        **Consumes** ``x``."""
        t, h, w = dims
        ch = int(x.shape[-1])
        vol = consume(x, to_row_major, (t, h, w // self.sp, ch))
        full = self.ccl_manager.all_gather(vol, dim=2, mesh_axis=self.sp_axis, use_hyperparams=False)
        return retile(full, (t * h * w, ch))

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        dims: tuple[int, int, int],
        drop_leading_frame: bool = True,
        stages: int | None = None,
        gather_output: bool = True,
    ) -> tuple[ttnn.Tensor, tuple[int, int, int]]:
        """``x`` is ``(tokens, in_channels)`` channels-last in TILE layout, normalized latent.

        ``dims`` is always the FULL ``(T, H, W)``. ``gather_output=False`` skips the final
        all-gather and returns this chip's W-band ``(T*H*(W/sp), ch)`` for stage 5 to consume directly.
        """
        x = self._conv_in(x)
        count = len(self.upsamples) if stages is None else stages
        sharded = False
        for stage in range(count):
            x, dims, sharded = self._run_stage(x, stage, dims, sharded=sharded, drop_leading_frame=drop_leading_frame)
        if sharded and gather_output:
            x = self._wgather(x, dims)
            log_dram(self.mesh_device, f"det gathered to replicated {dims}")
        return x, dims

    @timing_tree.span("mesh_device", "conv_in (denorm folded)", category=timing_tree.MLP)
    def _conv_in(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.conv_in(x)

    @timing_tree.span(
        "mesh_device", lambda self, x, stage, dims, **k: f"det stage {stage} (in {dims[0]},{dims[1]},{dims[2]})"
    )
    def _run_stage(
        self, x: ttnn.Tensor, stage: int, dims: tuple[int, int, int], *, sharded: bool, drop_leading_frame: bool
    ) -> tuple[ttnn.Tensor, tuple[int, int, int], bool]:
        """One deterministic stage. Returns the activation, its FULL out-dims, and whether it is W-sharded."""
        t, h, w = dims
        stage_sharded = self._w_sharded and stage > 0
        if stage_sharded:
            assert w % self.sp == 0, f"stage {stage} W={w} not divisible by sp={self.sp}"
            if not sharded:
                x = self._wshard(x, dims)
                sharded = True
        local_dims = (t, h, w // self.sp) if stage_sharded else dims
        cos, sin, plan = self._stage_setup(stage, dims, stage_sharded)
        x = self._run_blocks(x, stage, local_dims, cos, sin, plan, stage_sharded)
        x, out_dims = self._upsample(x, stage, local_dims, drop_leading_frame)
        if stage_sharded:
            out_dims = (out_dims[0], out_dims[1], out_dims[2] * self.sp)
        log_dram(self.mesh_device, f"det stage {stage} upsampled to {out_dims} sharded={stage_sharded}")
        return x, out_dims, sharded

    @timing_tree.span(
        "mesh_device",
        lambda self, stage, *a: f"stage {stage + 1} setup: rope tables + plan",
        category=timing_tree.SETUP,
    )
    def _stage_setup(self, stage: int, dims: tuple[int, int, int], stage_sharded: bool):
        cos, sin = self._rope(dims)
        if stage_sharded:
            cos = ttnn.mesh_partition(cos, dim=3, cluster_axis=self.sp_axis)
            sin = ttnn.mesh_partition(sin, dim=3, cluster_axis=self.sp_axis)
        plan = None if stage_sharded else self._plan(dims, self.stage_kernels[stage])
        return cos, sin, plan

    @timing_tree.span(
        "mesh_device",
        lambda self, x, stage, *a: f"STAGE {stage + 1}: {len(self.det_stages[stage])}x NABlock dim {x.shape[-1]}",
    )
    def _run_blocks(self, x: ttnn.Tensor, stage: int, local_dims, cos, sin, plan, stage_sharded: bool) -> ttnn.Tensor:
        for index, block in enumerate(self.det_stages[stage]):
            x = block(x, dims=local_dims, cos=cos, sin=sin, device_plan=plan)
            log_dram(self.mesh_device, f"det stage {stage} block {index} dims={local_dims} sharded={stage_sharded}")
        return x

    @timing_tree.span("mesh_device", lambda self, x, stage, *a: f"upsample {stage + 1}", category=timing_tree.UPSAMPLE)
    def _upsample(self, x: ttnn.Tensor, stage: int, local_dims, drop_leading_frame: bool):
        return self.upsamples[stage](x, dims=local_dims, drop_leading_frame=drop_leading_frame)


class DiffVAEDecoder(Module):
    """The whole LTX-2.5 diffusion video decoder: deterministic stages then the diffusion stage.

    Takes the same normalized latent the conv decoder takes and returns pixels.

    Two temporal adjustments frame the deterministic stages: before stage 1 the last latent frame
    is replicated :attr:`ghost_latent_frames` times to give the attention a trailing border, and
    after stage 4 that appendix is cropped back off. Apply the pad without the crop and the video
    grows spurious frames.
    """

    supports_yuv = True

    #: Whether the pipeline evicts a resident DiT before decode; see DiffVAEOptions.exclusive_residency.
    requires_exclusive_residency = True

    def __init__(
        self,
        config: dict,
        *,
        mesh_device,
        dtype: ttnn.DataType = ttnn.bfloat16,
        ccl_manager=None,
        options: DiffVAEOptions = DiffVAEOptions(),
    ):
        super().__init__()
        self.options = options
        # A replicated stage 5's activations are too large to share the mesh with a resident DiT;
        # a sharded one holds a fraction of the volume per chip and can sit beside it, which also
        # keeps the DiT's captured trace valid.
        self.requires_exclusive_residency = (
            options.exclusive_residency if options.exclusive_residency is not None else options.stage5_sp_axis is None
        )

        self.config = config
        self.mesh_device = mesh_device
        self._timestep = None
        # Set True by the pipeline after warm-up: forward() then captures the device half of the
        # decode as a ttnn trace on its first call and replays it after.
        self._vae_traced = False
        self.ccl_manager = ccl_manager
        self.patch_size = config["patch_size"]
        self.out_channels = config["out_channels"]
        self.in_channels = config["in_channels"]
        self.stage5_kernel = config["stage5_kernel"]
        # Upstream: (stage_kernels[0][0] // 2) * 2 latent frames of trailing replication.
        self.ghost_latent_frames = (config["stage_kernels"][0][0] // 2) * 2
        # Composed temporal upscale of the four upsamples.
        self.time_scale = math.prod(stride[0] for stride, _ in config["upsamples"])

        self.stages = DeterministicStages(
            in_channels=config["in_channels"],
            stage_channels=config["stage_channels"],
            stage_depths=config["stage_depths"],
            stage_kernels=config["stage_kernels"],
            upsamples=config["upsamples"],
            head_dim=config["head_dim"],
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            na3d_backend=options.stages_backend,
            sp_axis=options.stages_sp_axis,
            tp_axis=options.stages_tp_axis,
            block_options=options.det,
        )
        self.stage5 = DiffVAEStage5(
            DiffVAEStage5Config(
                dim=config["stage_channels"][-1],
                head_dim=config["head_dim"],
                kernel_size=config["stage5_kernel"],
                gna_stride=options.gna_stride,
                context_channels=config["stage_channels"][-1],
                mlp_hidden=4 * config["stage_channels"][-1],
                num_blocks=config["stage_depths"][-1],
                patch_size=config["patch_size"],
                out_channels=config["out_channels"],
                timestep_scale_multiplier=config["timestep_scale_multiplier"],
            ),
            mesh_device=mesh_device,
            dtype=dtype,
            ccl_manager=ccl_manager,
            na3d_backend=options.stage5_backend,
            sp_axis=options.stage5_sp_axis,
            tp_axis=options.stage5_tp_axis,
            fused_qkv=options.stage5_fused_qkv,
            tp_proj=options.stage5_tp_proj,
            slab_frames=options.slab_frames,
            device_unpatchify=options.device_boundaries,
        )
        # When both halves W-shard on the same axis, the context is handed over W-sharded instead
        # of gathered and re-sharded.
        self._wsharded_handoff = (
            self.stages._w_sharded and self.stage5._w_sharded and self.stages.sp_axis == self.stage5.sp_axis
        )
        self.dtype = dtype

    def parameter_layout(self) -> str:
        """Which parameters this decoder has, as a token for the weight-cache path.

        The fusion flags change the parameter SET (one fused ``qkv`` or three projections, one
        packed ``gate_up`` or two), so a cache written under one flag set does not load under
        another. The readable prefix names the forms, read off the built modules rather than the
        options; the suffix hashes every parameter's name and shape, so any change to the set, a
        rename included, lands in a fresh folder instead of failing against a stale one.
        """

        def stage_token(blocks) -> str:
            block = blocks[0]
            return f"q{1 if block.attn.fused_qkv else 3}m{1 if block.mlp.fused else 2}"

        def parameters(module: Module, prefix: str = ""):
            for name, parameter in module.named_parameters():
                yield f"{prefix}{name}:{tuple(parameter.total_shape)}"
            for name, child in module.named_children():
                yield from parameters(child, f"{prefix}{name}.")

        det = "-".join(stage_token(self.stages.det_stages[i]) for i in range(len(self.stages.det_stages)))
        block = self.stage5.diff_blocks[0]
        stage5 = f"q{1 if block.attn.fused_qkv else 3}m{1 if block.mlp.fused else 2}"
        digest = hashlib.sha1("\n".join(sorted(parameters(self))).encode()).hexdigest()[:8]
        return f"det-{det}_s5-{stage5}-{digest}"

    def torch_state_from_checkpoint(self, path, *, statistics: bool = True) -> dict[str, torch.Tensor]:
        """One state dict for the whole decoder, keyed for :meth:`load_torch_state_dict`.

        Kept separate from applying it so the weights can go through the tt_dit disk cache.
        """
        state = {f"stages.{k}": v for k, v in self.stages.state_from_checkpoint(path, statistics=statistics).items()}
        prefixes = ("diff_blocks.", "shared_adaln.", "t_embedder.", "conv_in_x_t.", "conv_out.", "norm_out.")
        state |= {f"stage5.{k}": v for k, v in read_decoder_tensors(path, prefixes).items()}
        return state

    def load_checkpoint(self, path, *, statistics: bool = True) -> None:
        """Load both halves from one LTX-2.5 video-VAE safetensors file, bypassing the cache."""
        self.load_torch_state_dict(self.torch_state_from_checkpoint(path, statistics=statistics))

    def context_frames(self, latent_frames: int) -> int:
        """Stage-5 temporal extent for a latent of ``latent_frames``, after pad and crop."""
        padded = latent_frames + self.ghost_latent_frames
        # Each temporal upsample doubles then drops its duplicate leading frame, so the
        # composed map is causal: n -> time_scale * (n - 1) + 1.
        grown = self.time_scale * (padded - 1) + 1
        return max(grown - self.ghost_latent_frames * self.time_scale, self.stage5_kernel[0])

    @timing_tree.span("mesh_device", "det stages TOTAL (forward_context)")
    def forward_context(
        self, latent: torch.Tensor, *, gather_output: bool = True, latent_tt: ttnn.Tensor | None = None
    ) -> tuple[ttnn.Tensor, tuple[int, int, int]]:
        """Deterministic stages on a ``(B, C, T, H, W)`` normalized latent, ghost cropped.

        ``gather_output=False`` returns the context W-sharded (this chip's band). ``dims`` is
        always the FULL ``(T, H, W)``.

        ``latent_tt`` supplies the raw latent already on device, in the ROW_MAJOR ``(1, C, T, H*W)``
        form the device-preproc path uploads. A trace refuses host-to-device writes during capture,
        so the upload has to happen outside the captured region. ``latent`` is then read for its
        shape only, and the caller keeps ownership of the buffer.
        """
        batch, channels, t, h, w = latent.shape
        assert batch == 1, f"batched decode is not implemented; got batch={batch}"
        assert channels == self.in_channels, f"latent has {channels} channels, expected {self.in_channels}"

        ghost = self.ghost_latent_frames
        if self.options.device_boundaries:
            # Upload the latent as-is; the ghost pad and channels-last flatten run on device.
            with timing_tree.span(self.mesh_device, "host->mesh: upload latent (raw)", category=timing_tree.HOST_XFER):
                raw = latent_tt
                if raw is None:
                    raw = ttnn.from_torch(
                        latent.reshape(1, channels, t, h * w).contiguous(),
                        device=self.mesh_device,
                        dtype=self.dtype,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )

            with timing_tree.span(self.mesh_device, "device: ghost pad + flatten", category=timing_tree.RESHAPE):
                last = ttnn.slice(raw, [0, 0, t - 1, 0], [1, channels, t, h * w])
                parts = [raw] + [last] * ghost
                padded_tt = ttnn.concat(parts, dim=2)
                if latent_tt is None:
                    ttnn.deallocate(raw)
                ttnn.deallocate(last)
                moved = ttnn.permute(padded_tt, (0, 2, 3, 1))
                ttnn.deallocate(padded_tt)
                x = ttnn.to_layout(ttnn.reshape(moved, ((t + ghost) * h * w, channels)), ttnn.TILE_LAYOUT)
                ttnn.deallocate(moved)
        else:
            with timing_tree.span(
                self.mesh_device, "host: ghost pad + permute/flatten", category=timing_tree.HOST_COMPUTE
            ):
                padded = torch.cat([latent, latent[:, :, -1:].expand(-1, -1, ghost, -1, -1)], dim=2)
                tokens = padded.permute(0, 2, 3, 4, 1).reshape(-1, channels).contiguous()
            with timing_tree.span(self.mesh_device, "host->mesh: upload TILE", category=timing_tree.HOST_XFER):
                x = ttnn.from_torch(tokens, device=self.mesh_device, dtype=self.dtype, layout=ttnn.TILE_LAYOUT)

        x, dims = self.stages(x, dims=(t + ghost, h, w), gather_output=gather_output)
        sharded_out = self.stages._w_sharded and not gather_output
        w_eff = dims[2] // self.stages.sp if sharded_out else dims[2]
        keep = self.context_frames(t)
        with timing_tree.span(self.mesh_device, "ghost crop on T", category=timing_tree.RESHAPE):
            if keep < dims[0]:
                channels_out = self.config["stage_channels"][-1]
                # The crop is on T, which the W-shard leaves untouched.
                frames = consume(x, to_row_major, (dims[0], dims[1] * w_eff, channels_out))
                cropped = consume(frames, ttnn.slice, [0, 0, 0], [keep, dims[1] * w_eff, channels_out])
                x = consume(cropped, retile, (keep * dims[1] * w_eff, channels_out))
                dims = (keep, dims[1], dims[2])
        return x, dims

    @timing_tree.span("mesh_device", "decode TOTAL", root=True)
    def decode(
        self,
        latent: torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        seed: int = 0,
        latent_tt: ttnn.Tensor | None = None,
        device_out: bool = False,
        output_type: str = "float",
    ) -> torch.Tensor | ttnn.Tensor:
        """Normalized ``(B, C, T, H, W)`` latent to ``(B, 3, T', H', W')`` pixels.

        ``noise`` is an input: stage 5 predicts x0 from it in a single step. Pass it to compare
        against a reference that drew its own.

        ``latent_tt`` and ``device_out`` move the two host boundaries out of the way so the whole
        decode can be captured as one trace. See :meth:`forward_context` and
        :meth:`DiffVAEStage5.forward`.
        """
        out, _ = self._decode(
            latent, noise=noise, seed=seed, latent_tt=latent_tt, device_out=device_out, output_type=output_type
        )
        return out

    def _decode(self, latent, *, noise, seed, latent_tt, device_out, output_type):
        """:meth:`decode`, also returning the stage-5 grid the pixel pull needs."""
        context, dims = self.forward_context(latent, gather_output=not self._wsharded_handoff, latent_tt=latent_tt)
        grid = Grid(batch=1, t=dims[0], h=dims[1], w=dims[2])
        channels_out = self.config["stage_channels"][-1]
        with timing_tree.span(self.mesh_device, "context reshape for stage 5", category=timing_tree.RESHAPE):
            if self._wsharded_handoff:
                w_local = grid.w // self.stages.sp
                context = ttnn.reshape(context, (1, 1, grid.t * grid.h * w_local, channels_out))
            else:
                context = ttnn.reshape(context, (1, 1, grid.sites, channels_out))

        # With the boundaries on device, noise stays None and stage 5 draws it there.
        if noise is None and not self.options.device_boundaries:
            shape = (1, self.out_channels, grid.t, grid.h * self.patch_size, grid.w * self.patch_size)
            with timing_tree.span(
                self.mesh_device, f"host: noise randn {tuple(shape)}", category=timing_tree.HOST_COMPUTE
            ):
                noise = torch.randn(shape, generator=torch.Generator().manual_seed(seed))

        # default_num_inference_steps is 1 on this checkpoint, so the timestep is [1.0]. Uploaded
        # once and kept: a trace refuses host-to-device writes during capture.
        if self._timestep is None:
            self._timestep = ttnn.from_torch(
                torch.tensor([[[[1.0]]]]), device=self.mesh_device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
            )
        timestep = self._timestep
        out = self.stage5.forward(
            context,
            noise,
            timestep,
            grid,
            context_sharded=self._wsharded_handoff,
            seed=seed,
            device_out=device_out,
            output_type=output_type,
        )
        log_dram(self.mesh_device, "decode done")
        log_ccl_cache(self.stage5.ccl_manager, "decode done")
        return out, grid

    @traced_function(device=lambda self: self.mesh_device, prep_run=False, clone_prep_inputs=False)
    def decode_device(self, raw: ttnn.Tensor, t: int, h: int, w: int, seed: int):
        """The device half of the decode: raw latent buffer in, this chip's pixel volume out.

        Everything between the two host boundaries, so that it captures as one ttnn trace. ``raw``
        is the ``(1, C, T, H*W)`` ROW_MAJOR upload the device-preproc path reads. Returns the volume
        and the stage-5 grid ``(T', H', W')`` as plain ints, which the tracer passes through.
        """
        shape_only = torch.empty(1, self.in_channels, t, h, w)
        vol, grid = self._decode(shape_only, noise=None, seed=seed, latent_tt=raw, device_out=True, output_type="float")
        return vol, grid.t, grid.h, grid.w

    def _pixels_traced(self, latent: torch.Tensor, *, seed: int, output_type: str):
        """One decode through the captured trace (captured on the first call), pixels on the host.

        The latent is uploaded to a fresh buffer each call; the tracer copies it into the trace's
        own input buffer, so the fresh one is freed afterwards, except on the capture call, where
        the fresh buffer IS the trace's input and has to stay.
        """
        if not self.options.device_boundaries:
            msg = (
                "a traced DiffVAE decode needs DiffVAEOptions(device_boundaries=True): "
                "the host work it replaces cannot sit inside a trace"
            )
            raise ValueError(msg)
        if timing_tree.ENABLED:
            msg = "TT_DIT_STAGE_TIMING=1 synchronises the mesh inside every span, which a trace capture cannot hold"
            raise RuntimeError(msg)

        b, c, t, h, w = latent.shape
        key = (c, t, h, w, seed)
        tracer = type(self).decode_device._tracers_keyed.get(self, {}).get(key)
        capturing = tracer is None or not tracer.trace_captured
        raw = ttnn.from_torch(
            latent.reshape(1, c, t, h * w).contiguous(),
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        vol, gt, gh, gw = self.decode_device(raw, t, h, w, seed, traced=True, tracer_trace_key=key)
        if not capturing:
            ttnn.deallocate(raw)
        return self.stage5.pull_pixels(vol, Grid(batch=1, t=gt, h=gh, w=gw), output_type, release=False)

    def forward(
        self,
        latent: torch.Tensor,
        *,
        output_type: str = "float",
        noise: torch.Tensor | None = None,
        seed: int = 0,
    ) -> torch.Tensor:
        """``decode`` behind the pipeline's decoder signature.

        ``output_type`` matches ``LTXVideoDecoder.forward``: ``float`` keeps ``[-1, 1]``, ``rgb``
        maps it to planar uint8, ``yuv`` converts and gathers YUV 4:2:0 on device (needs
        ``DiffVAEOptions.device_boundaries``).
        """
        if self._vae_traced:
            if noise is not None:
                msg = "a traced decode draws its noise on device; a caller-supplied noise cannot enter the trace"
                raise ValueError(msg)
            pixels = self._pixels_traced(latent, seed=seed, output_type="yuv" if output_type == "yuv" else "float")
            if output_type == "yuv":
                return pixels
        elif output_type == "yuv":
            return self.decode(latent, noise=noise, seed=seed, output_type="yuv")
        else:
            pixels = self.decode(latent, noise=noise, seed=seed)
        if output_type == "float":
            return pixels
        if output_type == "rgb":
            return pixels.add(1.0).mul(0.5 * 255.0).clamp(0.0, 255.0).to(torch.uint8)
        raise ValueError(f"unknown output_type {output_type!r}")

    def release_trace(self) -> None:
        """Free every captured decode trace (on shutdown, or before re-warming)."""
        for tracer in type(self).decode_device._tracers_keyed.get(self, {}).values():
            tracer.release_trace()
