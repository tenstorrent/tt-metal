# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TTNN port of diffusers `AutoencoderKLHunyuanVideo15` decoder (HunyuanVideo15Decoder3D).

Built bottom-up, each block PCC-validated against the CPU reference:
  CausalConv3d -> RMS_norm -> ResnetBlock -> AttnBlock -> Upsample(DCAE) -> Decoder.

Tensors are carried in ttnn BTHWC (B, T, H, W, C) ROW_MAJOR layout to match
`ttnn.experimental.conv3d`. Torch reference uses NCTHW (B, C, T, H, W).
"""
from __future__ import annotations

import os

import torch

import ttnn
from models.demos.hf_eager.hunyuanvideo_1_5.tt.vae_spatial import (
    TILE_SIZE,
    SpatialShardPlan,
    attention_chunk_tokens_from_env,
    attention_distributed_from_env,
    attention_sdpa_from_env,
    block_causal_chunk_plan,
    canonicalize_replicated_shard_edges,
    largest_sdpa_k_chunk,
    replicate_pad_to_plan,
    stitch_tiles_ttnn,
)
from models.tt_dit.models.vae.vae_wan2_1 import get_neighbor_pad_num_links
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import get_conv3d_config, register_conv3d_configs
from models.tt_dit.utils.tensor import fast_device_to_host, typed_tensor_2dshard

# Conservative conv3d blockings for the decoder's channel combos. The util's
# fallback uses C_in_block = in_channels, which overflows L1 for the wide
# channel-expand upsample convs (e.g. 1024->8192). Cap C_in_block small; these
# are correctness-first (untuned) and can be re-swept for speed later.
_K = (3, 3, 3)
# Tuned blockings (hw_product ~32 for core-grid parallelism + wider C_out_block)
# over the conservative (C,32,1,1,1) default, following swept wan/ltx entries in
# conv3d.py::_BLOCKINGS. C_in_block capped at 128 so the wide channel-expand
# upsample convs (e.g. 1024->8192) fit L1.
register_conv3d_configs(
    {
        (32, 1024, _K): (32, 64, 1, 8, 4),
        (1024, 1024, _K): (128, 64, 1, 8, 4),
        (1024, 8192, _K): (128, 32, 1, 8, 4),
        (1024, 4096, _K): (128, 32, 1, 8, 4),
        (512, 512, _K): (128, 64, 1, 8, 4),
        (512, 1024, _K): (128, 64, 1, 8, 4),
        (512, 2048, _K): (128, 32, 1, 8, 4),
        (256, 256, _K): (128, 64, 1, 8, 4),
        (256, 512, _K): (128, 64, 1, 8, 4),
        (256, 1024, _K): (128, 64, 1, 8, 4),
        (128, 128, _K): (128, 64, 1, 8, 4),
        (128, 512, _K): (128, 64, 1, 8, 4),
        (128, 3, _K): (128, 32, 1, 8, 4),
    }
)


# Shared holder for the on-device-VAE submesh (set by the test's mesh_device
# fixture, read by the gen path). Lives here because pytest imports conftest.py
# under a private module name, so a conftest-level global isn't visible to the
# test via the package path -- but this module IS imported identically by both.
HY_VAE_SUBMESH = None


def replicate_pad_bthwc(x, t_front, hpad, wpad):
    """Replicate-pad a (B,T,H,W,C) ROW_MAJOR tensor: T front-only, H/W both sides.

    Matches diffusers HunyuanVideo15CausalConv3d's F.pad(mode="replicate", (W,W,H,H,T,0)).
    """
    B, T, H, W, C = x.shape
    if wpad > 0:
        left = ttnn.slice(x, [0, 0, 0, 0, 0], [B, T, H, 1, C])
        right = ttnn.slice(x, [0, 0, 0, W - 1, 0], [B, T, H, W, C])
        x = ttnn.concat([left] * wpad + [x] + [right] * wpad, dim=3)
        B, T, H, W, C = x.shape
    if hpad > 0:
        top = ttnn.slice(x, [0, 0, 0, 0, 0], [B, T, 1, W, C])
        bot = ttnn.slice(x, [0, 0, H - 1, 0, 0], [B, T, H, W, C])
        x = ttnn.concat([top] * hpad + [x] + [bot] * hpad, dim=2)
        B, T, H, W, C = x.shape
    if t_front > 0:
        f0 = ttnn.slice(x, [0, 0, 0, 0, 0], [B, 1, H, W, C])
        x = ttnn.concat([f0] * t_front + [x], dim=1)
    return x


def replicate_pad_bthwc_to_shape(x, target_h, target_w):
    """Replicate-pad gathered logical H/W back to equal mesh storage dimensions."""
    B, T, H, W, C = x.shape
    if target_h < H or target_w < W:
        raise ValueError(f"target {(target_h, target_w)} is smaller than input {(H, W)}")
    if target_h > H:
        bottom = ttnn.slice(x, [0, 0, H - 1, 0, 0], [B, T, H, W, C])
        x = ttnn.concat([x] + [bottom] * (target_h - H), dim=2)
        H = target_h
    if target_w > W:
        right = ttnn.slice(x, [0, 0, 0, W - 1, 0], [B, T, H, W, C])
        x = ttnn.concat([x] + [right] * (target_w - W), dim=3)
    return x


_VAE_WCACHE = {"dir": None, "n": 0}


def vae_weight_cache_begin(device, dtype, *, hw_sharded):
    """Arm the prepared-conv-weight cache for one adapter construction.

    Returns the directory in use, or None when the cache is off. The counter is
    reset here so indices are per-adapter and reproducible: convs are built in a
    fixed order, so index N always refers to the same conv for a given tag."""
    if os.environ.get("HY_VAE_WEIGHT_CACHE", "0") != "1":
        _VAE_WCACHE.update(dir=None, n=0)
        return None
    root = (
        os.environ.get("HY_VAE_WEIGHT_CACHE_DIR")
        or os.environ.get("HY_DIT_WEIGHT_CACHE_DIR")
        or os.environ.get("TT_DIT_CACHE_DIR")
        or "~/.cache/tt-dit"
    )
    grid = device.compute_with_storage_grid_size()
    mesh = "x".join(str(d) for d in tuple(device.shape)) if device.get_num_devices() > 1 else "1"
    tag = f"mesh{mesh}_grid{grid.x}x{grid.y}_{dtype}_hw{int(bool(hw_sharded))}"
    path = os.path.join(os.path.expanduser(root), "hunyuanvideo15_vae", tag)
    os.makedirs(path, exist_ok=True)
    _VAE_WCACHE.update(dir=path, n=0)
    return path


def _vae_wcache_path():
    """Next cache path for a prepared conv weight, or None when the cache is off."""
    if _VAE_WCACHE["dir"] is None:
        return None
    p = os.path.join(_VAE_WCACHE["dir"], f"conv{_VAE_WCACHE['n']}.tensorbin")
    _VAE_WCACHE["n"] += 1
    return p


class CausalConv3d:
    """ttnn HunyuanVideo15CausalConv3d: replicate-pad (on device) + conv3d(pad=0)."""

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        *,
        device,
        dtype=ttnn.bfloat16,
        parallel_config: VaeHWParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ):
        # weight: torch (Cout, Cin, kt, kh, kw)
        cout, cin, kt, kh, kw = weight.shape
        assert kt == kh == kw, f"only cubic kernels supported, got {(kt, kh, kw)}"
        self.k = kt
        self.cout = cout
        self.device = device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.t_front = kt - 1
        self.pad_hw = kt // 2
        if (parallel_config is None) != (ccl_manager is None):
            raise ValueError("parallel_config and ccl_manager must be provided together")

        self.cfg = get_conv3d_config(cin, cout, (kt, kh, kw), dtype, device.compute_with_storage_grid_size())
        # Prepared-conv-weight cache (HY_VAE_WEIGHT_CACHE=1). `prepare_conv3d_weights`
        # reformats every causal-conv weight for the conv3d kernel on the host at
        # adapter construction; the phase breakdown puts that whole stage at ~12.5s.
        # Same mechanism as the DiT cache: `DumpTensorMode.LOCAL` persists each
        # device's own shard and restores placement, so the reload is exact. Convs
        # are constructed in deterministic order, so a sequential index is a
        # sufficient key -- everything that changes the prepared layout (mesh, dtype,
        # core grid, sharding) is in the directory tag.
        _path = _vae_wcache_path()
        if _path is not None and os.path.exists(_path):
            w = ttnn.load_tensor(_path, device=device)
        else:
            w = ttnn.from_torch(weight, dtype=dtype)
            w = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=w, C_in_block=self.cfg.C_in_block, device=device
            )
            if not ttnn.is_tensor_storage_on_device(w):
                w = ttnn.to_device(w, device)
            if _path is not None:
                ttnn.dump_tensor(_path, w, mode=ttnn.DumpTensorMode.LOCAL)
        if not ttnn.is_tensor_storage_on_device(w):
            w = ttnn.to_device(w, device)
        self.w = w
        if bias is None:
            bias = torch.zeros(cout)
        self.b = ttnn.from_torch(bias.reshape(1, -1), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.ckc = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, x_bthwc, logical_h=0, logical_w=0):
        spatially_sharded = self.parallel_config is not None and (
            self.parallel_config.height_parallel.factor > 1 or self.parallel_config.width_parallel.factor > 1
        )
        if spatially_sharded:
            local_hpad = self.pad_hw if self.parallel_config.height_parallel.factor == 1 else 0
            local_wpad = self.pad_hw if self.parallel_config.width_parallel.factor == 1 else 0
            if self.t_front or local_hpad or local_wpad:
                x_bthwc = replicate_pad_bthwc(x_bthwc, self.t_front, local_hpad, local_wpad)
            dims, pad_left, pad_right, axes, neighbor_sems, links = [], [], [], [], [], []
            if self.pad_hw and self.parallel_config.height_parallel.factor > 1:
                dims.append(2)
                pad_left.append(self.pad_hw)
                pad_right.append(self.pad_hw)
                axis = self.parallel_config.height_parallel.mesh_axis
                axes.append(axis)
                neighbor_sems.append(self.ccl_manager.get_np_ping_pong_semaphore(axis))
                links.append(get_neighbor_pad_num_links(self.ccl_manager, x_bthwc, 2))
            if self.pad_hw and self.parallel_config.width_parallel.factor > 1:
                dims.append(3)
                pad_left.append(self.pad_hw)
                pad_right.append(self.pad_hw)
                axis = self.parallel_config.width_parallel.mesh_axis
                axes.append(axis)
                neighbor_sems.append(self.ccl_manager.get_np_ping_pong_semaphore(axis))
                links.append(get_neighbor_pad_num_links(self.ccl_manager, x_bthwc, 3))
            if dims:
                x_bthwc = self.ccl_manager.neighbor_pad_persistent_buffer(
                    x_bthwc,
                    dims=dims,
                    pad_left=pad_left,
                    pad_right=pad_right,
                    padding_mode="replicate",
                    axes=axes,
                    neighbor_sems=neighbor_sems,
                    num_links=links,
                )
        elif self.t_front or self.pad_hw:
            x_bthwc = replicate_pad_bthwc(x_bthwc, self.t_front, self.pad_hw, self.pad_hw)
        out = ttnn.experimental.conv3d(
            input_tensor=x_bthwc,
            weight_tensor=self.w,
            bias_tensor=self.b,
            device=self.device,
            config=self.cfg,
            output_channels=self.cout,
            kernel_size=(self.k, self.k, self.k),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.ckc,
        )
        if spatially_sharded and self.pad_hw and logical_h and logical_w:
            out = canonicalize_replicated_shard_edges(out, logical_h, logical_w, self.parallel_config, self.ccl_manager)
        return out


class RMSNorm:
    """ttnn HunyuanVideo15RMS_norm (channel_first, images=False, bias=False).

    Reference does F.normalize(x, dim=channel) * sqrt(dim) * gamma, i.e. RMS-norm
    across the channel dim (mean-square over C), scaled by per-channel gamma.
    In BTHWC the channel dim is last, so this is an RMS over the last axis.
    """

    def __init__(self, gamma: torch.Tensor, *, device, dtype=ttnn.bfloat16, eps=1e-12):
        g = gamma.reshape(-1).float()  # (C,)
        self.C = g.numel()
        self.eps = eps
        # broadcastable over (B,T,H,W,C): shape (1,1,1,1,C), TILE layout for elementwise on last dim
        self.gamma = ttnn.from_torch(g.reshape(1, 1, 1, 1, self.C), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.dtype = dtype

    def __call__(self, x_bthwc):
        xt = ttnn.to_layout(x_bthwc, ttnn.TILE_LAYOUT)
        ms = ttnn.mean(ttnn.mul(xt, xt), dim=-1, keepdim=True)  # mean-square over channel
        inv = ttnn.rsqrt(ttnn.add(ms, self.eps))
        out = ttnn.mul(ttnn.mul(xt, inv), self.gamma)
        return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)


def silu(x_bthwc):
    xt = ttnn.to_layout(x_bthwc, ttnn.TILE_LAYOUT)
    xt = ttnn.silu(xt)
    return ttnn.to_layout(xt, ttnn.ROW_MAJOR_LAYOUT)


def _block_causal_mask(n_frame, n_hw, dtype=torch.float32):
    """Additive (seq,seq) mask: token i (in frame f) attends to all tokens in frames 0..f."""
    seq = n_frame * n_hw
    mask = torch.full((seq, seq), float("-inf"), dtype=dtype)
    for i in range(seq):
        f = i // n_hw
        mask[i, : (f + 1) * n_hw] = 0.0
    return mask


class AttnBlock:
    """ttnn HunyuanVideo15AttnBlock: RMSNorm -> 1x1 q/k/v -> block-causal attention -> 1x1 proj + identity.

    The default path materializes the full ``seq x seq`` mask and score matrix,
    where ``seq = T * H * W``.  ``HY_VAE_ATTN_CHUNK=<tokens>`` (or the
    ``attention_chunk_tokens`` override) instead walks the queries in blocks
    that read only their causal key prefix, which is the same arithmetic
    without either quadratic tensor.  Blocks never cross a latent frame, so a
    request of at least ``H * W`` tokens gives one block per frame; 0 keeps the
    monolithic path.

    ``HY_VAE_ATTN_DIST=1`` additionally keeps the queries H/W-fractured.  Norm
    and the 1x1 ``to_q``/``proj_out`` convolutions are pointwise in H/W, so a
    rank can compute exactly the output rows it already stores; only keys and
    values need the all-gather.  That removes the post-attention
    ``mesh_partition`` and divides the attention, projection, and residual work
    by the number of ranks.

    ``HY_VAE_ATTN_SDPA=1`` replaces each block's explicit matmul/softmax/matmul
    with ``ttnn.transformer.scaled_dot_product_attention``, which never
    materializes the block's score tile at all.
    """

    def __init__(
        self,
        torch_attn,
        *,
        device,
        dtype=ttnn.bfloat16,
        parallel_config: VaeHWParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
        attention_chunk_tokens: int | None = None,
        attention_distributed: bool | None = None,
        attention_sdpa: bool | None = None,
    ):
        sd = torch_attn.state_dict()
        self.device = device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.norm = RMSNorm(sd["norm.gamma"], device=device, dtype=dtype)
        conv_kwargs = dict(device=device, dtype=dtype, parallel_config=parallel_config, ccl_manager=ccl_manager)
        self.to_q = CausalConv3d(sd["to_q.weight"], sd["to_q.bias"], **conv_kwargs)  # k=1
        self.to_k = CausalConv3d(sd["to_k.weight"], sd["to_k.bias"], **conv_kwargs)
        self.to_v = CausalConv3d(sd["to_v.weight"], sd["to_v.bias"], **conv_kwargs)
        self.proj_out = CausalConv3d(sd["proj_out.weight"], sd["proj_out.bias"], **conv_kwargs)
        self.C = int(torch_attn.in_channels)
        self.scale = self.C**-0.5
        self._mask_cache = {}
        if attention_chunk_tokens is None:
            attention_chunk_tokens = attention_chunk_tokens_from_env()
        elif attention_chunk_tokens < 0:
            raise ValueError(f"attention_chunk_tokens must be non-negative, got {attention_chunk_tokens}")
        self.attention_chunk_tokens = int(attention_chunk_tokens)
        self.attention_distributed = bool(
            attention_distributed_from_env() if attention_distributed is None else attention_distributed
        )
        self.attention_sdpa = bool(attention_sdpa_from_env() if attention_sdpa is None else attention_sdpa)
        self._sdpa_program_config = None
        self._sdpa_ckc = None
        if self.attention_sdpa:
            self._configure_sdpa()

    def _configure_sdpa(self):
        """Pick a key chunk whose circular buffers fit L1 at this head dim.

        The mid-block is one head of width ``C``, so the flash kernel's
        ``k_tiles``/``v_tiles`` circular buffers each cost
        ``(k_chunk / 32) * (C / 32) * 2`` tiles.  At ``C = 1024`` that is 32
        tiles per key row-of-tiles, so Wan's ``k_chunk=256`` does not fit and
        the chunk must be chosen from the head dim rather than copied.
        """
        k_chunk = largest_sdpa_k_chunk(self.C, q_chunk=TILE_SIZE)
        if k_chunk == 0:
            raise ValueError(f"flash SDPA cannot fit a single key tile at head_dim={self.C}; " "unset HY_VAE_ATTN_SDPA")
        self._sdpa_k_chunk = k_chunk
        self._sdpa_q_chunk = TILE_SIZE
        self._sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=self._sdpa_q_chunk,
            k_chunk_size=self._sdpa_k_chunk,
            exp_approx_mode=False,
        )
        self._sdpa_ckc = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def _attend_block_sdpa(self, q_block, keys, values, kv_stop, full_prefix):
        """One mask-free block through the flash kernel.

        ``keys``/``values`` arrive as ``(B, 1, kv_seq, C)``; the prefix slice is
        non-causal and mask-free, and SDPA's own padded-K handling covers a
        ``kv_stop`` that is not a multiple of the key chunk.
        """
        B = q_block.shape[0]
        C = q_block.shape[-1]
        k_prefix = keys if full_prefix else ttnn.slice(keys, [0, 0, 0, 0], [B, 1, kv_stop, C])
        v_prefix = values if full_prefix else ttnn.slice(values, [0, 0, 0, 0], [B, 1, kv_stop, C])
        out = ttnn.transformer.scaled_dot_product_attention(
            ttnn.reshape(q_block, (B, 1, q_block.shape[1], C)),
            k_prefix,
            v_prefix,
            is_causal=False,
            scale=self.scale,
            program_config=self._sdpa_program_config,
            compute_kernel_config=self._sdpa_ckc,
        )
        if not full_prefix:
            ttnn.deallocate(k_prefix)
            ttnn.deallocate(v_prefix)
        return ttnn.reshape(out, (B, out.shape[2], C))

    def _attend_blocks(self, q_flat, k_flat, v_flat, plan):
        """Walk a chunk plan over ``(B, seq, C)`` operands and concatenate.

        ``q_flat`` may be shorter than ``k_flat``/``v_flat``: under the
        distributed formulation the caller holds only its own query rows while
        the keys and values span the whole logical grid.
        """
        B, q_seq, C = q_flat.shape
        kv_seq = k_flat.shape[1]
        values = ttnn.to_layout(v_flat, ttnn.TILE_LAYOUT)
        k_tile = ttnn.to_layout(k_flat, ttnn.TILE_LAYOUT)
        if self.attention_sdpa:
            keys = ttnn.reshape(k_tile, (B, 1, kv_seq, C))
            values = ttnn.reshape(values, (B, 1, kv_seq, C))
        else:
            keys = ttnn.transpose(k_tile, -2, -1)  # (B, C, seq)
            ttnn.deallocate(k_tile)

        blocks = []
        for chunk in plan:
            q_block = ttnn.to_layout(ttnn.slice(q_flat, [0, chunk.q_start, 0], [B, chunk.q_stop, C]), ttnn.TILE_LAYOUT)
            full_prefix = chunk.kv_stop == kv_seq
            if self.attention_sdpa:
                block = self._attend_block_sdpa(q_block, keys, values, chunk.kv_stop, full_prefix)
                ttnn.deallocate(q_block)
            else:
                keys_t_prefix = keys if full_prefix else ttnn.slice(keys, [0, 0, 0], [B, C, chunk.kv_stop])
                values_prefix = values if full_prefix else ttnn.slice(values, [0, 0, 0], [B, chunk.kv_stop, C])
                scores = ttnn.matmul(q_block, keys_t_prefix)
                ttnn.deallocate(q_block)
                scaled = ttnn.mul(scores, self.scale)
                ttnn.deallocate(scores)
                # softmax masks the tail-tile padding of the unaligned prefix, so
                # the block reduces over exactly `kv_stop` logical keys.
                weights = ttnn.softmax(scaled, dim=-1)
                ttnn.deallocate(scaled)
                block = ttnn.matmul(weights, values_prefix)
                ttnn.deallocate(weights)
                if not full_prefix:
                    ttnn.deallocate(keys_t_prefix)
                    ttnn.deallocate(values_prefix)
            blocks.append(ttnn.to_layout(block, ttnn.ROW_MAJOR_LAYOUT))
            ttnn.deallocate(block)

        ttnn.deallocate(keys)
        ttnn.deallocate(values)
        out = blocks[0] if len(blocks) == 1 else ttnn.concat(blocks, dim=1)
        assert out.shape[1] == q_seq
        return out

    def _chunked_attention(self, q, k, v):
        """Block-causal attention with peak memory set by one query block.

        Exactly equivalent to the masked full-sequence form: a query in frame
        ``f`` may only attend to ``[0, (f + 1) * H * W)``, so restricting the
        key/value operands to that prefix removes precisely the entries the
        additive ``-inf`` mask would have zeroed after softmax.
        """
        B, T, H, W, C = q.shape
        n_hw = H * W
        seq = T * n_hw
        plan = block_causal_chunk_plan(T, n_hw, self.attention_chunk_tokens)
        out = self._attend_blocks(
            ttnn.reshape(q, (B, seq, C)),
            ttnn.reshape(k, (B, seq, C)),
            ttnn.reshape(v, (B, seq, C)),
            plan,
        )
        return ttnn.reshape(out, (B, T, H, W, C))

    def _distributed_attention(self, q_local, k_full, v_full):
        """Attention over rank-local queries and globally gathered keys/values.

        The rank holds ``local_h * local_w`` query rows per latent frame and
        they are contiguous in its own storage, so one block per frame (further
        split by ``attention_chunk_tokens``) covers them.  Each block still
        reduces over the full ``logical_h * logical_w`` key prefix of frames
        ``0..f``, which is what makes the result identical to the replicated
        form row by row.
        """
        B, T, local_h, local_w, C = q_local.shape
        local_hw = local_h * local_w
        kv_hw = k_full.shape[2] * k_full.shape[3]
        kv_seq = T * kv_hw
        plan = block_causal_chunk_plan(T, local_hw, self.attention_chunk_tokens, kv_hw=kv_hw)
        out = self._attend_blocks(
            ttnn.reshape(q_local, (B, T * local_hw, C)),
            ttnn.reshape(k_full, (B, kv_seq, C)),
            ttnn.reshape(v_full, (B, kv_seq, C)),
            plan,
        )
        return ttnn.reshape(out, (B, T, local_h, local_w, C))

    def _full_attention(self, q, k, v):
        """Monolithic form: one cached `seq x seq` mask and one `seq x seq` score matrix."""
        B, T, H, W, C = q.shape
        seq = T * H * W
        mask_key = (T, H, W)
        mask = self._mask_cache.get(mask_key)
        if mask is None:
            mask = ttnn.from_torch(
                _block_causal_mask(T, H * W), dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )
            self._mask_cache[mask_key] = mask

        def flat(t):
            return ttnn.to_layout(ttnn.reshape(t, (B, seq, C)), ttnn.TILE_LAYOUT)

        q2, k2, v2 = flat(q), flat(k), flat(v)
        scores = ttnn.matmul(q2, ttnn.transpose(k2, -2, -1))  # (B, seq, seq)
        scores = ttnn.add(ttnn.mul(scores, self.scale), mask)
        attn = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(attn, v2)  # (B, seq, C)
        return ttnn.to_layout(ttnn.reshape(out, (B, T, H, W, C)), ttnn.ROW_MAJOR_LAYOUT)

    def _gather_hw_and_crop(self, x_bthwc, logical_h, logical_w):
        """All-gather a rank-local H/W shard and crop the replicate padding off."""
        padded_h = x_bthwc.shape[2] * self.parallel_config.height_parallel.factor
        padded_w = x_bthwc.shape[3] * self.parallel_config.width_parallel.factor
        out = ttnn.to_layout(x_bthwc, ttnn.TILE_LAYOUT)
        if self.parallel_config.height_parallel.factor > 1:
            out = self.ccl_manager.all_gather_persistent_buffer(
                out, dim=2, mesh_axis=self.parallel_config.height_parallel.mesh_axis
            )
        if self.parallel_config.width_parallel.factor > 1:
            out = self.ccl_manager.all_gather_persistent_buffer(
                out, dim=3, mesh_axis=self.parallel_config.width_parallel.mesh_axis
            )
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        if logical_h and padded_h > logical_h:
            out = ttnn.slice(out, [0, 0, 0, 0, 0], [out.shape[0], out.shape[1], logical_h, out.shape[3], out.shape[4]])
        if logical_w and padded_w > logical_w:
            out = ttnn.slice(out, [0, 0, 0, 0, 0], [out.shape[0], out.shape[1], out.shape[2], logical_w, out.shape[4]])
        return out

    def _distributed_call(self, x_bthwc, logical_h, logical_w):
        """H/W-fractured mid-block attention: only K and V are gathered.

        The residual, ``norm``, the 1x1 ``to_q``/``proj_out`` convolutions, and
        every attention row are functions of a single spatial position, so this
        rank computes exactly the rows it stores and returns them already
        fractured.  No ``mesh_partition`` is needed on the way out.

        Storage-only rows beyond the logical grid are handled by making the
        input canonical first: because the whole chain from the input row to
        the output row is a per-position map given the shared keys and values,
        a padded row that holds a copy of the last logical row produces a copy
        of that row's output, which is exactly the replicate semantics
        ``canonicalize_replicated_shard_edges`` and ``replicate_pad_to_plan``
        maintain everywhere else in the decoder.
        """
        x_bthwc = canonicalize_replicated_shard_edges(
            x_bthwc, logical_h, logical_w, self.parallel_config, self.ccl_manager
        )
        h = self.norm(x_bthwc)
        q_local = self.to_q(h)
        k_full = self._gather_hw_and_crop(self.to_k(h), logical_h, logical_w)
        v_full = self._gather_hw_and_crop(self.to_v(h), logical_h, logical_w)
        out = self._distributed_attention(q_local, k_full, v_full)
        ttnn.deallocate(k_full)
        ttnn.deallocate(v_full)
        return ttnn.add(self.proj_out(out), x_bthwc)

    def __call__(self, x_bthwc, logical_h=0, logical_w=0):
        identity = x_bthwc
        spatially_sharded = self.parallel_config is not None and (
            self.parallel_config.height_parallel.factor > 1 or self.parallel_config.width_parallel.factor > 1
        )
        if spatially_sharded and self.attention_distributed:
            padded_h = identity.shape[2] * self.parallel_config.height_parallel.factor
            padded_w = identity.shape[3] * self.parallel_config.width_parallel.factor
            return self._distributed_call(identity, logical_h or padded_h, logical_w or padded_w)
        if spatially_sharded:
            padded_h = identity.shape[2] * self.parallel_config.height_parallel.factor
            padded_w = identity.shape[3] * self.parallel_config.width_parallel.factor
            identity = self._gather_hw_and_crop(identity, logical_h, logical_w)
        h = self.norm(identity)
        q = self.to_q(h)
        k = self.to_k(h)
        v = self.to_v(h)
        # The flash kernel replaces a block's score tile, so it only means
        # anything on the blocked path; requesting it implies frame blocking.
        if self.attention_chunk_tokens or self.attention_sdpa:
            out = self._chunked_attention(q, k, v)
        else:
            out = self._full_attention(q, k, v)
        out = self.proj_out(out)
        if spatially_sharded:
            out = ttnn.add(out, identity)
            out = replicate_pad_bthwc_to_shape(out, padded_h, padded_w)
            if self.parallel_config.height_parallel.factor > 1:
                out = ttnn.mesh_partition(out, dim=2, cluster_axis=self.parallel_config.height_parallel.mesh_axis)
            if self.parallel_config.width_parallel.factor > 1:
                out = ttnn.mesh_partition(out, dim=3, cluster_axis=self.parallel_config.width_parallel.mesh_axis)
            return out
        return ttnn.add(out, identity)


def _dcae_upsample_rearrange(x_bthwc, r1, r2, r3):
    """(b, f, h, w, r1*r2*r3*c) -> (b, r1*f, r2*h, r3*w, c). BTHWC analogue of the
    reference NCTHW view/permute (channel-packed r1->T, r2->H, r3->W)."""
    b, f, h, w, pc = x_bthwc.shape
    c = pc // (r1 * r2 * r3)
    x = ttnn.reshape(x_bthwc, (b, f, h, w, r1, r2, r3, c))
    x = ttnn.permute(x, (0, 1, 4, 2, 5, 3, 6, 7))  # (b, f, r1, h, r2, w, r3, c)
    return ttnn.reshape(x, (b, f * r1, h * r2, w * r3, c))


class Upsample:
    """ttnn HunyuanVideo15Upsample (DCAE): channel-expand CausalConv3d + rearrange + residual.

    Temporal variant (add_temporal_upsample) doubles frames [1:] but leaves frame 0
    spatial-only, to preserve temporal causality (the reference's asymmetric first frame).
    """

    def __init__(
        self,
        torch_up,
        *,
        device,
        dtype=ttnn.bfloat16,
        parallel_config: VaeHWParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ):
        w = torch_up.conv.conv.weight.detach()
        b = torch_up.conv.conv.bias.detach() if torch_up.conv.conv.bias is not None else None
        self.conv = CausalConv3d(
            w,
            b,
            device=device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.add_temporal = bool(torch_up.add_temporal_upsample)
        self.repeats = int(torch_up.repeats)

    def __call__(self, x_bthwc, logical_h=0, logical_w=0):
        h = self.conv(x_bthwc, logical_h, logical_w)
        if self.add_temporal:
            B, T, H, W, PC = h.shape
            h_first = ttnn.slice(h, [0, 0, 0, 0, 0], [B, 1, H, W, PC])
            h_first = _dcae_upsample_rearrange(h_first, 1, 2, 2)
            c_r = h_first.shape[-1]
            h_first = ttnn.slice(
                h_first,
                [0, 0, 0, 0, 0],
                [h_first.shape[0], h_first.shape[1], h_first.shape[2], h_first.shape[3], c_r // 2],
            )
            h_next = ttnn.slice(h, [0, 1, 0, 0, 0], [B, T, H, W, PC])
            h_next = _dcae_upsample_rearrange(h_next, 2, 2, 2)
            h = ttnn.concat([h_first, h_next], dim=1)

            Bx, Tx, Hx, Wx, Cx = x_bthwc.shape
            x_first = ttnn.slice(x_bthwc, [0, 0, 0, 0, 0], [Bx, 1, Hx, Wx, Cx])
            x_first = _dcae_upsample_rearrange(x_first, 1, 2, 2)
            x_first = ttnn.repeat_interleave(x_first, self.repeats // 2, dim=4)
            x_next = ttnn.slice(x_bthwc, [0, 1, 0, 0, 0], [Bx, Tx, Hx, Wx, Cx])
            x_next = _dcae_upsample_rearrange(x_next, 2, 2, 2)
            x_next = ttnn.repeat_interleave(x_next, self.repeats, dim=4)
            shortcut = ttnn.concat([x_first, x_next], dim=1)
        else:
            h = _dcae_upsample_rearrange(h, 1, 2, 2)
            shortcut = ttnn.repeat_interleave(x_bthwc, self.repeats, dim=4)
            shortcut = _dcae_upsample_rearrange(shortcut, 1, 2, 2)
        out = ttnn.add(h, shortcut)
        if logical_h:
            logical_h *= 2
            logical_w *= 2
            if self.parallel_config is not None:
                out = canonicalize_replicated_shard_edges(
                    out, logical_h, logical_w, self.parallel_config, self.ccl_manager
                )
            return out, logical_h, logical_w
        return out


class ResnetBlock:
    """ttnn HunyuanVideo15ResnetBlock: norm-silu-conv x2 + (1x1 conv shortcut if C changes)."""

    def __init__(
        self,
        torch_block,
        *,
        device,
        dtype=ttnn.bfloat16,
        parallel_config: VaeHWParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ):
        sd = torch_block.state_dict()
        conv_kwargs = dict(device=device, dtype=dtype, parallel_config=parallel_config, ccl_manager=ccl_manager)
        self.norm1 = RMSNorm(sd["norm1.gamma"], device=device, dtype=dtype)
        self.conv1 = CausalConv3d(sd["conv1.conv.weight"], sd["conv1.conv.bias"], **conv_kwargs)
        self.norm2 = RMSNorm(sd["norm2.gamma"], device=device, dtype=dtype)
        self.conv2 = CausalConv3d(sd["conv2.conv.weight"], sd["conv2.conv.bias"], **conv_kwargs)
        self.conv_shortcut = None
        if "conv_shortcut.weight" in sd:
            self.conv_shortcut = CausalConv3d(sd["conv_shortcut.weight"], sd.get("conv_shortcut.bias"), **conv_kwargs)

    def __call__(self, x_bthwc, logical_h=0, logical_w=0):
        residual = x_bthwc
        h = self.norm1(x_bthwc)
        h = silu(h)
        h = self.conv1(h, logical_h, logical_w)
        h = self.norm2(h)
        h = silu(h)
        h = self.conv2(h, logical_h, logical_w)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual, logical_h, logical_w)
        return ttnn.add(h, residual)


class MidBlock:
    """ttnn HunyuanVideo15MidBlock: resnet -> (attn -> resnet)*."""

    def __init__(
        self,
        torch_mid,
        *,
        device,
        dtype=ttnn.bfloat16,
        parallel_config: VaeHWParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ):
        kwargs = dict(device=device, dtype=dtype, parallel_config=parallel_config, ccl_manager=ccl_manager)
        self.resnets = [ResnetBlock(r, **kwargs) for r in torch_mid.resnets]
        self.attentions = [AttnBlock(a, **kwargs) if a is not None else None for a in torch_mid.attentions]

    def __call__(self, h, logical_h=0, logical_w=0):
        h = self.resnets[0](h, logical_h, logical_w)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                h = attn(h, logical_h, logical_w)
            h = resnet(h, logical_h, logical_w)
        return h


class UpBlock3D:
    """ttnn HunyuanVideo15UpBlock3D: resnets then optional upsampler."""

    def __init__(
        self,
        torch_up,
        *,
        device,
        dtype=ttnn.bfloat16,
        parallel_config: VaeHWParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ):
        kwargs = dict(device=device, dtype=dtype, parallel_config=parallel_config, ccl_manager=ccl_manager)
        self.resnets = [ResnetBlock(r, **kwargs) for r in torch_up.resnets]
        self.upsamplers = (
            [Upsample(u, **kwargs) for u in torch_up.upsamplers] if torch_up.upsamplers is not None else []
        )

    def __call__(self, h, logical_h=0, logical_w=0):
        for r in self.resnets:
            h = r(h, logical_h, logical_w)
        for u in self.upsamplers:
            result = u(h, logical_h, logical_w)
            if logical_h:
                h, logical_h, logical_w = result
            else:
                h = result
        if logical_h:
            return h, logical_h, logical_w
        return h


class HunyuanVideo15Decoder:
    """ttnn port of HunyuanVideo15Decoder3D. Input/output in BTHWC.

    Build from the torch decoder module:  dec_tt = HunyuanVideo15Decoder(vae.decoder, device=mesh)
    Call with a BTHWC latent (B, T, H, W, 32); returns BTHWC video (B, T', H', W', 3).
    """

    def __init__(
        self,
        torch_dec,
        *,
        device,
        dtype=ttnn.bfloat16,
        parallel_config: VaeHWParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ):
        self.device = device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.repeat = int(torch_dec.repeat)
        kwargs = dict(device=device, dtype=dtype, parallel_config=parallel_config, ccl_manager=ccl_manager)
        self.conv_in = CausalConv3d(
            torch_dec.conv_in.conv.weight.detach(), torch_dec.conv_in.conv.bias.detach(), **kwargs
        )
        self.mid_block = MidBlock(torch_dec.mid_block, **kwargs)
        self.up_blocks = [UpBlock3D(ub, **kwargs) for ub in torch_dec.up_blocks]
        self.norm_out = RMSNorm(torch_dec.norm_out.gamma.detach(), device=device, dtype=dtype)
        self.conv_out = CausalConv3d(
            torch_dec.conv_out.conv.weight.detach(), torch_dec.conv_out.conv.bias.detach(), **kwargs
        )

    def __call__(self, x_bthwc, logical_h=0, logical_w=0):
        h = self.conv_in(x_bthwc, logical_h, logical_w)
        h = ttnn.add(h, ttnn.repeat_interleave(x_bthwc, self.repeat, dim=4))  # conv_in residual
        h = self.mid_block(h, logical_h, logical_w)
        for ub in self.up_blocks:
            result = ub(h, logical_h, logical_w)
            if logical_h:
                h, logical_h, logical_w = result
            else:
                h = result
        h = self.norm_out(h)
        h = silu(h)
        h = self.conv_out(h, logical_h, logical_w)
        if logical_h:
            return h, logical_h, logical_w
        return h


class TTVAEDecodeAdapter:
    """Drop-in replacement for a diffusers VAE whose .decode() runs on device.

    Everything except decode() delegates to the real vae (config, scaling_factor,
    compression ratios, dtype). Swap into a pipeline like the TTTransformer wrapper:
        pipe.vae = TTVAEDecodeAdapter(pipe.vae, mesh_device)
    Uniform spatial tiles are batch-sharded across a multi-device mesh. Decoder
    rounds concatenate on device and use one final host readback before stitching.
    """

    def __init__(
        self,
        real_vae,
        device,
        *,
        dtype=ttnn.bfloat16,
        parallel_config: VaeHWParallelConfig | None = None,
        ccl_manager: CCLManager | None = None,
    ):
        self.__dict__["_real"] = real_vae
        self.__dict__["_device"] = device
        self.__dict__["_dtype_tt"] = dtype
        hw_sharded = os.environ.get("HY_VAE_HW_SHARD", "0") == "1"
        if hw_sharded:
            if device.get_num_devices() <= 1:
                raise ValueError("HY_VAE_HW_SHARD requires a multi-device mesh")
            if parallel_config is None:
                mesh_shape = tuple(device.shape)
                if len(mesh_shape) != 2:
                    raise ValueError(f"HY_VAE_HW_SHARD requires a 2D mesh, got {mesh_shape}")
                parallel_config = VaeHWParallelConfig(
                    height_parallel=ParallelFactor(factor=mesh_shape[0], mesh_axis=0),
                    width_parallel=ParallelFactor(factor=mesh_shape[1], mesh_axis=1),
                )
            if ccl_manager is None:
                ccl_manager = CCLManager(device, num_links=1, topology=ttnn.Topology.Linear)
        self.__dict__["_hw_sharded"] = hw_sharded
        self.__dict__["_parallel_config"] = parallel_config
        self.__dict__["_ccl_manager"] = ccl_manager
        # Arm the prepared-conv-weight cache before the decoder tree is built, so
        # every CausalConv3d below picks up a sequential index from the same run.
        self.__dict__["_wcache_dir"] = vae_weight_cache_begin(device, dtype, hw_sharded=hw_sharded)
        self.__dict__["_dec"] = HunyuanVideo15Decoder(
            real_vae.decoder,
            device=device,
            dtype=dtype,
            parallel_config=parallel_config if hw_sharded else None,
            ccl_manager=ccl_manager if hw_sharded else None,
        )

    def __getattr__(self, k):
        return getattr(self.__dict__["_real"], k)

    def _decode_tile(self, z_tile):
        """Decode one latent tile (torch NCTHW) -> output (torch NCTHW) via the ttnn decoder."""
        dev = self.__dict__["_device"]
        mm = ttnn.ReplicateTensorToMesh(dev) if dev.get_num_devices() > 1 else None
        z_bthwc = ttnn.from_torch(
            z_tile.permute(0, 2, 3, 4, 1).contiguous().float(),
            dtype=self.__dict__["_dtype_tt"],
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            mesh_mapper=mm,
        )
        out = self.__dict__["_dec"](z_bthwc)
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float().permute(0, 4, 1, 2, 3)

    def _decode_hw_sharded(self, z):
        """Decode one full latent with H/W fractured across the mesh and one final D2H."""
        dev = self.__dict__["_device"]
        parallel_config = self.__dict__["_parallel_config"]
        plan = SpatialShardPlan(
            logical_height=int(z.shape[-2]),
            logical_width=int(z.shape[-1]),
            height_factor=parallel_config.height_parallel.factor,
            width_factor=parallel_config.width_parallel.factor,
        )
        z_bthwc = z.permute(0, 2, 3, 4, 1).contiguous().float()
        z_bthwc = replicate_pad_to_plan(z_bthwc, plan, h_dim=2, w_dim=3)
        z_tt = typed_tensor_2dshard(
            z_bthwc,
            dev,
            shard_mapping={
                parallel_config.height_parallel.mesh_axis: 2,
                parallel_config.width_parallel.mesh_axis: 3,
            },
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.__dict__["_dtype_tt"],
        )
        out_tt, logical_h, logical_w = self.__dict__["_dec"](z_tt, plan.logical_height, plan.logical_width)
        concat_dims = [None, None]
        concat_dims[parallel_config.height_parallel.mesh_axis] = 2
        concat_dims[parallel_config.width_parallel.mesh_axis] = 3
        out = fast_device_to_host(
            out_tt,
            dev,
            concat_dims,
            ccl_manager=self.__dict__["_ccl_manager"],
        )
        return out[:, :, :logical_h, :logical_w, :].float().permute(0, 4, 1, 2, 3)

    @staticmethod
    def _blend_v(a, b, blend_extent):
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    @staticmethod
    def _blend_h(a, b, blend_extent):
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    @staticmethod
    def _blend_v_vectorized(a, b, blend_extent):
        """Vectorized equivalent of diffusers' per-row vertical blend loop."""
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        if blend_extent == 0:
            return b
        prior_weight = torch.tensor(
            [1 - y / blend_extent for y in range(blend_extent)], device=b.device, dtype=b.dtype
        ).view(1, 1, 1, -1, 1)
        current_weight = torch.tensor(
            [y / blend_extent for y in range(blend_extent)], device=b.device, dtype=b.dtype
        ).view(1, 1, 1, -1, 1)
        blended = a[:, :, :, -blend_extent:, :] * prior_weight + b[:, :, :, :blend_extent, :] * current_weight
        return torch.cat([blended, b[:, :, :, blend_extent:, :]], dim=-2)

    @staticmethod
    def _blend_h_vectorized(a, b, blend_extent):
        """Vectorized equivalent of diffusers' per-column horizontal blend loop."""
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        if blend_extent == 0:
            return b
        prior_weight = torch.tensor(
            [1 - x / blend_extent for x in range(blend_extent)], device=b.device, dtype=b.dtype
        ).view(1, 1, 1, 1, -1)
        current_weight = torch.tensor(
            [x / blend_extent for x in range(blend_extent)], device=b.device, dtype=b.dtype
        ).view(1, 1, 1, 1, -1)
        blended = a[:, :, :, :, -blend_extent:] * prior_weight + b[:, :, :, :, :blend_extent] * current_weight
        return torch.cat([blended, b[:, :, :, :, blend_extent:]], dim=-1)

    @classmethod
    def _stitch_tiles(cls, decoded, coords, ncol, blend_h, blend_w, row_limit_h, row_limit_w, *, legacy=False):
        """Blend and stitch decoded NCTHW tiles while preserving diffusers boundary semantics."""
        rows = [decoded[r * ncol : (r + 1) * ncol] for r in range(len(coords) // ncol)]
        blend_v = cls._blend_v if legacy else cls._blend_v_vectorized
        blend_h_fn = cls._blend_h if legacy else cls._blend_h_vectorized
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = blend_v(rows[i - 1][j], tile, blend_h)
                    rows[i][j] = tile
                if j > 0:
                    tile = blend_h_fn(row[j - 1], tile, blend_w)
                    rows[i][j] = tile
                result_row.append(tile[:, :, :, :row_limit_h, :row_limit_w])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _decode_batch_sharded_legacy(self, batch):
        """Prior path: synchronize and read back once after every device round."""
        dev = self.__dict__["_device"]
        ndev = dev.get_num_devices()
        n_total = batch.shape[0]
        out_chunks = []
        for r in range(0, n_total, ndev):
            chunk = batch[r : r + ndev]
            n = chunk.shape[0]
            if n < ndev:
                chunk = torch.cat([chunk, chunk[-1:].expand(ndev - n, *chunk.shape[1:])], dim=0)
            mm = ttnn.ShardTensorToMesh(dev, dim=0) if ndev > 1 else None
            zt = ttnn.from_torch(
                chunk.permute(0, 2, 3, 4, 1).contiguous().float(),
                dtype=self.__dict__["_dtype_tt"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                mesh_mapper=mm,
            )
            out = self.__dict__["_dec"](zt)
            comp = ttnn.ConcatMeshToTensor(dev, dim=0) if ndev > 1 else None
            ot = ttnn.to_torch(out, mesh_composer=comp).float().permute(0, 4, 1, 2, 3)
            out_chunks.append(ot[:n])
        return torch.cat(out_chunks, dim=0)

    def _decode_batch_sharded_device(self, batch):
        """Decode all rounds and concatenate their outputs on device.

        The returned mesh tensor is ordered device-major, then round-major.
        ``rounds`` and ``n_total`` are returned so the one host readback can
        restore the original round-major tile order.
        """
        dev = self.__dict__["_device"]
        ndev = dev.get_num_devices()
        n_total = batch.shape[0]
        out_rounds = []
        for r in range(0, n_total, ndev):
            chunk = batch[r : r + ndev]
            n = chunk.shape[0]
            if n < ndev:
                chunk = torch.cat([chunk, chunk[-1:].expand(ndev - n, *chunk.shape[1:])], dim=0)
            mm = ttnn.ShardTensorToMesh(dev, dim=0) if ndev > 1 else None
            zt = ttnn.from_torch(
                chunk.permute(0, 2, 3, 4, 1).contiguous().float(),
                dtype=self.__dict__["_dtype_tt"],
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=dev,
                mesh_mapper=mm,
            )
            out_rounds.append(self.__dict__["_dec"](zt))
        out = out_rounds[0] if len(out_rounds) == 1 else ttnn.concat(out_rounds, dim=0)
        return out, len(out_rounds), n_total

    @staticmethod
    def _restore_round_major(ot, ndev, rounds, n_total):
        """Convert composer device-major output back to input tile order."""
        if ndev > 1 and rounds > 1:
            ot = ot.reshape(ndev, rounds, *ot.shape[1:]).transpose(0, 1).reshape(ndev * rounds, *ot.shape[1:])
        return ot[:n_total]

    def _decode_batch_sharded(self, batch):
        """Decode a batch of uniform latent tiles (torch NCTHW, N=num tiles) by
        SHARDING the batch across the mesh -- each device decodes one tile per round
        (rounds of `num_devices`), so N tiles cost ceil(N/num_devices) batched decoder
        passes instead of N sequential ones. Per-device peak stays at ONE tile (so it
        fits the same DRAM the replicated single-tile path did), but the wall-clock is
        ~num_devices smaller. Decoder outputs remain device-resident across rounds and
        are concatenated before one final readback. Returns torch NCTHW.

        Set HY_VAE_LEGACY_TILE_READBACK=1 to retain the prior per-round readback path.
        """
        if os.environ.get("HY_VAE_LEGACY_TILE_READBACK", "0") == "1":
            return self._decode_batch_sharded_legacy(batch)

        dev = self.__dict__["_device"]
        ndev = dev.get_num_devices()
        out, rounds, n_total = self._decode_batch_sharded_device(batch)
        comp = ttnn.ConcatMeshToTensor(dev, dim=0) if ndev > 1 else None
        ot = ttnn.to_torch(out, mesh_composer=comp).float()
        # ConcatMeshToTensor groups each device's local rounds together.
        # Restore input order: (device, round, ...) -> (round, device, ...).
        ot = self._restore_round_major(ot, ndev, rounds, n_total)
        return ot.permute(0, 4, 1, 2, 3)

    def _tiled_decode(self, z):
        """Spatial (H/W) tiled decode mirroring diffusers `tiled_decode`. Tiling the
        latent H/W shrinks the mid-block attention (seq = T*Hl*Wl) so high frame counts
        fit. Tiles are decoded batch-SHARDED across the mesh (see _decode_batch_sharded)
        so a replicated 32-chip VAE no longer redundantly decodes every tile on every
        chip -- the tiles are split across chips. Edge tiles are padded to a uniform
        size for batching and cropped back to their real output extent; blend/crop
        stay on host torch, matching the reference."""
        rv = self.__dict__["_real"]
        tlh, tlw = rv.tile_latent_min_height, rv.tile_latent_min_width
        tsh, tsw = rv.tile_sample_min_height, rv.tile_sample_min_width
        ov = rv.tile_overlap_factor
        _, _, _, height, width = z.shape
        overlap_h = int(tlh * (1 - ov))
        overlap_w = int(tlw * (1 - ov))
        blend_h = int(tsh * ov)
        blend_w = int(tsw * ov)
        row_limit_h = tsh - blend_h
        row_limit_w = tsw - blend_w
        up = tsh // tlh  # spatial upscale (latent -> sample), == spatial_compression_ratio

        # Enumerate tiles at the reference positions; record each tile's real (h,w).
        coords = []
        for i in range(0, height, overlap_h):
            for j in range(0, width, overlap_w):
                t = z[:, :, :, i : i + tlh, j : j + tlw]
                coords.append((i, j, int(t.shape[-2]), int(t.shape[-1])))
        ncol = len(range(0, width, overlap_w))

        # Uniform-pad each tile to (tlh, tlw) so they can be batched together.
        padded = []
        for i, j, rh, rw in coords:
            t = z[:, :, :, i : i + tlh, j : j + tlw]
            if rh < tlh or rw < tlw:
                t = torch.nn.functional.pad(t, (0, tlw - rw, 0, tlh - rh))
            padded.append(t)
        padded_batch = torch.cat(padded, dim=0)
        # A replicated tile batch can be blended/cropped/stitched entirely in
        # TTNN.  Today _decode_batch_sharded_device distributes different tiles
        # to different mesh ranks, so this path is safe only on one device.  The
        # multi-device path remains the validated single-readback + host stitch
        # until the decoder adopts the per-layer Wan-style H/W halo contract.
        device_stitch = os.environ.get("HY_VAE_DEVICE_STITCH", "0") == "1"
        if device_stitch and self.__dict__["_device"].get_num_devices() == 1:
            decoded_tt, _, _ = self._decode_batch_sharded_device(padded_batch)
            stitched_tt = stitch_tiles_ttnn(
                decoded_tt,
                coords,
                ncol,
                blend_h,
                blend_w,
                row_limit_h,
                row_limit_w,
                spatial_scale=up,
                device=self.__dict__["_device"],
                dtype=self.__dict__["_dtype_tt"],
            )
            return ttnn.to_torch(ttnn.get_device_tensors(stitched_tt)[0]).float().permute(0, 4, 1, 2, 3)

        decoded_batch = self._decode_batch_sharded(padded_batch)
        # Crop each decoded tile back to its real output extent (real_latent * up).
        decoded = [decoded_batch[k : k + 1, :, :, : coords[k][2] * up, : coords[k][3] * up] for k in range(len(coords))]

        legacy_blend = os.environ.get("HY_VAE_LEGACY_TILE_BLEND", "0") == "1"
        return self._stitch_tiles(
            decoded,
            coords,
            ncol,
            blend_h,
            blend_w,
            row_limit_h,
            row_limit_w,
            legacy=legacy_blend,
        )

    def decode(self, z, return_dict=True):
        from diffusers.models.autoencoders.vae import DecoderOutput

        rv = self.__dict__["_real"]
        if self.__dict__.get("_hw_sharded", False):
            v = self._decode_hw_sharded(z).to(z.dtype)
            return DecoderOutput(sample=v) if return_dict else (v,)
        # Optional smaller tiles (HY_VAE_TILE_PX): shrink the per-tile spatial extent
        # so the upsample intermediate fits when the VAE time-shares chips with a
        # resident DiT (e.g. sp=4 on all 32 chips leaves little free DRAM). Smaller
        # tiles => lower peak but more tiles (slower). Default 256px = full speed.
        _px = int(os.environ.get("HY_VAE_TILE_PX", "0"))
        if _px:
            sc = int(rv.config.spatial_compression_ratio)
            rv.tile_sample_min_height = rv.tile_sample_min_width = _px
            rv.tile_latent_min_height = rv.tile_latent_min_width = max(1, _px // sc)
        # Spatial tiling avoids the full-res OOM at high frame counts. Trigger it
        # like the reference (latent H or W beyond a tile) but only when opted in
        # (HY_VAE_TILE=1 or vae.enable_tiling()), so the small-frame path stays fast.
        tile = (os.environ.get("HY_VAE_TILE", "0") == "1" or getattr(rv, "use_tiling", False)) and (
            z.shape[-1] > rv.tile_latent_min_width or z.shape[-2] > rv.tile_latent_min_height
        )
        if tile:
            v = self._tiled_decode(z).to(z.dtype)
        else:
            v = self._decode_tile(z).to(z.dtype)
        return DecoderOutput(sample=v) if return_dict else (v,)
