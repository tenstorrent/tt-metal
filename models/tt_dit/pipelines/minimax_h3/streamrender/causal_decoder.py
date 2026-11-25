# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""StreamRender's chunk-causal ViT-24 video decoder on TT.

The reference is Sana's `CausalViTDecoder.decode_chunk`: 24 of the H3 VAE's 36 ViT layers, two latents per call,
each layer's K/V of the previous four latents cached. Within a layer the chunk's tokens attend in 16x16 windows
(odd layers shifted by 8) to the same window of the cached and current latents, plus a global 5-token suffix
(4 register tokens and a zero cls token) that is rebuilt every chunk and itself attends to every token.

The layers are the H3 VAE blocks `MiniMaxH3TransformerBlock` already runs; this module only routes attention.

Layout: the residual stream is window-major, `[windows, 1, chunk * 256, dim]` with rows `(latent, y, x)` inside a
window, exactly the reference's `_partition` order. Every per-token op is order-blind, and in this layout the QKV
head split already yields per-window `[windows, heads, chunk * 256, head_dim]`, so attention needs no gather or
relayout. Windows cover the shifted, padded grid; rows outside the image are carried as pad rows, masked as keys and
dropped when the stream moves to the next layer's window grid, which is the only row gather per layer. Each layer's
K/V cache keeps its own layer's window layout, since a layer's shift never changes. RoPE ids are window-local, so one
`[chunk * 256, head_dim]` table serves every window.

Every device buffer that outlives a chunk -- the per-layer K/V cache, the masks and the gather indices -- is allocated
at construction and the cache is refreshed in place, so a trace captured later (the DiT's, on the same devices) can
never land on it. The cache always holds `cache_latents` latents; the ones a young stream has not produced yet are
masked, which also keeps every shape constant from the first chunk.

The suffix queries attend to every token of every window. Rather than rebuild token-order K/V, their softmax runs
jointly over the per-window score blocks: a shared max, summed exponents and summed `e @ V`.

On a mesh the windows are sharded over one axis (`sp_axis`) and replicated over the other: window attention, the
K/V cache and every per-token op stay on the device that owns the window. The two cross-device steps are the move
between window grids (all-gather the residual, then each device gathers its own target windows) and the suffix
softmax (each device reduces its windows to a running max, denominator and numerator; one all-gather combines them).
"""

from __future__ import annotations

import math

import torch

import ttnn

from ....layers.linear import Linear
from ....layers.module import Module, ModuleList
from ....layers.normalization import LayerNorm
from ....models.vae.minimax_h3.decoder_minimax_h3 import MiniMaxH3TransformerBlock
from ....utils.tensor import from_torch, local_device_to_torch
from ....utils.tracing import traced_function

_TILE = 32
# The suffix block of every key set is padded to this many rows, which keeps every key length a multiple of the
# K chunk.
_SUFFIX_BLOCK = 128
_MASKED = -1e9


def _student_block_state(state: dict[str, torch.Tensor], heads: int, head_dim: int) -> dict[str, torch.Tensor]:
    """One student block in the diffusers names `MiniMaxH3TransformerBlock` loads.

    The student fuses QKV per head (`(head, {q, k, v}, lane)` rows) and orders its SwiGLU `w1` as `[gate; value]`;
    the TT block wants separate `to_q/k/v` and `[value; gate]`.
    """
    qkv_w = state["attn.to_qkv.weight"].view(heads, 3, head_dim, -1)
    qkv_b = state["attn.to_qkv.bias"].view(heads, 3, head_dim)
    gate_w, value_w = state["ff.w1.weight"].chunk(2, dim=0)
    gate_b, value_b = state["ff.w1.bias"].chunk(2, dim=0)
    out = {
        "norm1.weight": state["norm1.weight"],
        "norm2.weight": state["norm2.weight"],
        "scale1": state["scale1"],
        "scale2": state["scale2"],
        "attn.to_out.0.weight": state["attn.to_out.weight"],
        "attn.to_out.0.bias": state["attn.to_out.bias"],
        "ff.net.0.proj.weight": torch.cat([value_w, gate_w]),
        "ff.net.0.proj.bias": torch.cat([value_b, gate_b]),
        "ff.net.2.weight": state["ff.w2.weight"],
        "ff.net.2.bias": state["ff.w2.bias"],
    }
    for index, name in enumerate(("to_q", "to_k", "to_v")):
        out[f"attn.{name}.weight"] = qkv_w[:, index].reshape(heads * head_dim, -1)
        out[f"attn.{name}.bias"] = qkv_b[:, index].reshape(heads * head_dim)
    return out


class CausalViTDecoderTT(Module):
    def __init__(
        self,
        reference: torch.nn.Module,
        *,
        height: int,
        width: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager=None,
        sp_axis: int | None = None,
        trace: bool = False,
        rgb: tuple[torch.Tensor, torch.Tensor] | None = None,
        eps: float = 1e-5,
    ) -> None:
        """`reference` is Sana's loaded `CausalViTDecoder`; it supplies the geometry, the RoPE ids and the weights.

        With `sp_axis`, windows are sharded over that mesh axis and `ccl_manager` runs the gathers. With `trace`, the
        steady state (full cache) replays a captured chunk; the decoder is device-bound, so that buys nothing yet, and
        it forbids long-lived allocations on the mesh afterwards (the DiT adapter's input cache among them). With `rgb`, the
        per-channel `(scale, bias)` of the codec's pixel revert, chunks come out as uint8 RGB -- reverted, clamped
        and truncated as `clamp(0, 1) * 255` then `byte()` would -- instead of decode_base floats; a quarter of the
        readback and no host-side revert.
        """
        super().__init__()
        self.ref = reference
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.sp_axis = sp_axis
        self.trace = trace
        self.emits_rgb = rgb is not None
        self.sp = 1 if sp_axis is None else tuple(mesh_device.shape)[sp_axis]
        if self.sp > 1 and ccl_manager is None:
            raise ValueError("sharding windows needs a ccl_manager")
        self.H, self.W = height, width
        self.heads, self.head_dim = reference.heads, reference.dh
        self.dim = self.heads * self.head_dim
        self.window, self.chunk, self.cache_latents = reference.window, reference.chunk, reference.cache_latents
        self.num_suffix = reference.register_tokens.shape[1] + 1
        self.rows = self.chunk * self.window * self.window
        self.shifts = [
            (self.window // 2) if (reference.shift and index % 2 == 1) else 0 for index in range(len(reference.blocks))
        ]

        self.embed = Linear(reference.latents_mean.numel(), self.dim, bias=True, mesh_device=mesh_device)
        self.blocks = ModuleList(
            MiniMaxH3TransformerBlock(
                self.dim, num_heads=self.heads, head_dim=self.head_dim, eps=eps, mesh_device=mesh_device
            )
            for _ in reference.blocks
        )
        self.norm_out = LayerNorm(self.dim, norm_eps=eps, bias=True, mesh_device=mesh_device)
        self.proj_out = Linear(self.dim, reference.proj_out.out_features, bias=True, mesh_device=mesh_device)

        grids = {shift: self._grid(shift) for shift in set(self.shifts)}
        # Every grid pads to one window count that divides over the devices; pad windows hold only pad rows.
        self.windows = -(-max(g["windows"] for g in grids.values()) // self.sp) * self.sp
        for grid in grids.values():
            pad = self.windows - grid["windows"]
            grid["tokens"] = torch.cat([grid["tokens"], torch.full((pad, grid["tokens"].shape[1]), -1)])
            grid["valid"] = torch.cat([grid["valid"], torch.zeros(pad, grid["valid"].shape[1], dtype=torch.bool)])
            grid["windows"] = self.windows
        self._grids = grids
        self._suffix_pad = self._replicated(torch.zeros(1, self.heads, _SUFFIX_BLOCK - _TILE, self.head_dim))
        self._zero_rows = self._replicated(torch.zeros(1, 1, _TILE, self.dim))
        self.sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=256,
            k_chunk_size=_SUFFIX_BLOCK,
            exp_approx_mode=False,
        )
        suffix_scores = torch.full((1, 1, 1, _SUFFIX_BLOCK), _MASKED)
        suffix_scores[..., : self.num_suffix] = 0.0
        self._suffix_score_mask = self._replicated(suffix_scores)
        if rgb is not None:
            # proj_out columns are (channel, frame, y, x); every column takes its channel's revert, times 255.
            per_channel = reference.proj_out.out_features // 3
            scale, bias = (t.float().reshape(3).repeat_interleave(per_channel) * 255.0 for t in rgb)
            self._rgb_scale = self._replicated(scale.reshape(1, 1, 1, -1), ttnn.float32)
            self._rgb_bias = self._replicated(bias.reshape(1, 1, 1, -1), ttnn.float32)
        cached_counts = sorted(
            {min(t, self.cache_latents) for t in range(0, self.cache_latents + self.chunk, self.chunk)}
        )
        self._masks = {(c, shift): self._build_mask(c, shift) for c in cached_counts for shift in self._grids}
        self._moves = {(a, b): self._build_move(a, b) for a in self._grids for b in self._grids if a != b}
        cache_shape = (
            self.windows // self.sp,
            self.heads,
            self.cache_latents * self.window * self.window,
            self.head_dim,
        )
        self._cache = [
            tuple(
                ttnn.zeros(cache_shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device) for _ in "kv"
            )
            for _ in reference.blocks
        ]
        self.t = 0

    # ------------------------------------------------------------------ weights

    def load_reference_weights(self) -> None:
        ref = self.ref
        # Denormalize, the 1x1x1 `post_quant_conv` and `x_embedder` are one affine map of a latent voxel.
        conv_w = ref.post_quant_conv.weight.detach().float().flatten(1)
        conv_b = ref.post_quant_conv.bias.detach().float()
        embed_w = ref.x_embedder.weight.detach().float()
        embed_b = ref.x_embedder.bias.detach().float()
        std, mean = ref.latents_std.float(), ref.latents_mean.float()
        state = {
            "embed.weight": embed_w @ conv_w * std,
            "embed.bias": embed_w @ (conv_w @ mean + conv_b) + embed_b,
            "norm_out.weight": ref.norm_out.weight.detach(),
            "norm_out.bias": ref.norm_out.bias.detach(),
            "proj_out.weight": ref.proj_out.weight.detach(),
            "proj_out.bias": ref.proj_out.bias.detach(),
        }
        for index, block in enumerate(ref.blocks):
            block_state = {k: v.detach() for k, v in _student_state(block).items()}
            for key, value in _student_block_state(block_state, self.heads, self.head_dim).items():
                state[f"blocks.{index}.{key}"] = value
        self.load_torch_state_dict(state)

    # ------------------------------------------------------------------ host tables

    def _replicated(self, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) -> ttnn.Tensor:
        return from_torch(t, device=self.mesh_device, dtype=dtype, layout=layout, mesh_axes=[..., None, None])

    def _sharded(self, t: torch.Tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, dim: int = 0) -> ttnn.Tensor:
        """`t` split along `dim` over the window axis (whole windows per device), replicated over the other."""
        if self.sp == 1:
            return self._replicated(t, dtype, layout)
        axes = [None] * t.dim()
        axes[dim] = self.sp_axis
        return from_torch(t, device=self.mesh_device, dtype=dtype, layout=layout, mesh_axes=axes)

    def _local_indices(self, values: torch.Tensor) -> ttnn.Tensor:
        """Per-device index lists: `values` is split evenly over the window axis."""
        return self._sharded(values.to(torch.int32).reshape(1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, dim=1)

    def _gather_windows(self, t: ttnn.Tensor, dim: int) -> ttnn.Tensor:
        if self.sp == 1:
            return t
        return self.ccl_manager.all_gather(t, dim=dim, mesh_axis=self.sp_axis, use_hyperparams=False)

    def _grid(self, shift: int) -> dict:
        """Window grid of one shift: the token of every window row (-1 for pad rows) and the spatial validity."""
        H, W, ws, C = self.H, self.W, self.window, self.chunk
        rows_h, rows_w = (H + shift + ws - 1) // ws, (W + shift + ws - 1) // ws
        ly, lx = torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")
        tokens, valid = [], []
        for a in range(rows_h):
            for b in range(rows_w):
                y, x = (a * ws + ly - shift).flatten(), (b * ws + lx - shift).flatten()
                inside = (y >= 0) & (y < H) & (x >= 0) & (x < W)
                spatial = torch.where(inside, y * W + x, -1)
                tokens.append(torch.cat([torch.where(inside, t * H * W + spatial, -1) for t in range(C)]))
                valid.append(inside)
        return {"windows": rows_h * rows_w, "tokens": torch.stack(tokens), "valid": torch.stack(valid)}

    def _build_mask(self, cached: int, shift: int) -> dict[str, ttnn.Tensor]:
        """Additive key masks of one `(cached latents, shift)`: window attention and the suffix's window scores.

        Keys are `[cache (cache_latents) ; current (chunk) ; suffix block]`; the oldest `cache_latents - cached` cache
        latents are not produced yet.
        """
        grid = self._grids[shift]
        latents = self.cache_latents + self.chunk
        produced = torch.arange(latents) >= self.cache_latents - cached
        valid = (grid["valid"][:, None, :] & produced[None, :, None]).reshape(grid["windows"], -1)
        keys = torch.where(valid, 0.0, _MASKED)
        suffix = torch.full((grid["windows"], _SUFFIX_BLOCK), _MASKED)
        suffix[:, : self.num_suffix] = 0.0
        window = torch.cat([keys, suffix], dim=1)[:, None, None, :].expand(-1, 1, self.rows, -1)
        return {
            "window": self._sharded(window.contiguous()),
            "suffix_scores": self._sharded(keys[:, None, None, :].contiguous()),
        }

    def _build_move(self, source: int, target: int) -> ttnn.Tensor:
        """Row gather from `source`'s window grid to `target`'s; target pad rows read an appended zero row."""
        src, dst = self._grids[source], self._grids[target]
        row_of = torch.full((self.chunk * self.H * self.W,), -1, dtype=torch.long)
        flat = src["tokens"].flatten()
        row_of[flat[flat >= 0]] = (flat >= 0).nonzero().flatten()
        zero = src["tokens"].numel()
        wanted = dst["tokens"].flatten()
        index = torch.where(wanted >= 0, row_of[wanted.clamp(min=0)], zero)
        return self._local_indices(index)

    def _rope(self, t0: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        # Window-local ids are the window position itself, so a `window x window` grid at shift 0 covers every window.
        cos, sin = self.ref._ids(t0, self.chunk, self.window, self.window, 0, torch.float32, torch.device("cpu"))
        return self._tt_rope(cos, sin)

    def _suffix_rope(self, t0: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        cos, sin = self.ref._suffix_ids(t0, self.num_suffix, torch.float32, torch.device("cpu"))
        cos, sin = cos.reshape(-1, cos.shape[-1]), sin.reshape(-1, sin.shape[-1])
        pad = _TILE - self.num_suffix
        cos = torch.cat([cos, torch.ones(pad, cos.shape[-1])])
        sin = torch.cat([sin, torch.zeros(pad, sin.shape[-1])])
        return self._tt_rope(cos, sin)

    def _tt_rope(self, cos: torch.Tensor, sin: torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Reference `(N, rotary)` tables in the interleaved lane basis the fused norm+RoPE uses (`rope_minimax_h3`)."""
        cos, sin = cos.reshape(-1, cos.shape[-1]).float(), sin.reshape(-1, sin.shape[-1]).float()
        half = cos.shape[-1] // 2
        pad = self.head_dim - cos.shape[-1]
        cos = torch.cat([cos[:, :half].repeat_interleave(2, dim=1), torch.ones(cos.shape[0], pad)], dim=1)
        sin = torch.cat([sin[:, :half].repeat_interleave(2, dim=1), torch.zeros(sin.shape[0], pad)], dim=1)
        return tuple(self._replicated(t.reshape(1, 1, *t.shape)) for t in (cos, sin))

    # ------------------------------------------------------------------ stream

    def init_state(self) -> "CausalViTDecoderTT":
        # The cache keeps its buffers; a new stream's masks hide whatever an earlier stream left in them.
        self.t = 0
        return self

    def _relayout(self, x: ttnn.Tensor, source: int, target: int) -> ttnn.Tensor:
        local = x.shape[0]
        flat = self._gather_windows(ttnn.reshape(x, (1, 1, local * self.rows, self.dim)), dim=2)
        table = ttnn.to_layout(ttnn.concat([flat, self._zero_rows], dim=2), ttnn.ROW_MAJOR_LAYOUT)
        moved = ttnn.embedding(self._moves[(source, target)], table, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(table)
        return ttnn.reshape(moved, (local, 1, self.rows, self.dim))

    def _suffix_attention(self, qs, k_all, v_all, ks_block, vs_block, key_mask, scale):
        """Suffix queries over every window's keys and the suffix, as one softmax across the score blocks.

        Each device reduces its own windows to `(max, denominator, numerator)`; the per-device partials are rescaled
        to the global max and summed. The suffix's own keys are added once, after the combine.
        """
        local = k_all.shape[0]
        q_rep = ttnn.repeat(qs, ttnn.Shape([local, 1, 1, 1]))
        s_w = ttnn.add(ttnn.multiply(ttnn.matmul(q_rep, k_all, transpose_b=True), scale), key_mask)
        m = ttnn.max(ttnn.max(s_w, dim=-1, keepdim=True), dim=0, keepdim=True)
        e_w = ttnn.exp(ttnn.subtract(s_w, m))
        denominator = ttnn.sum(ttnn.sum(e_w, dim=-1, keepdim=True), dim=0, keepdim=True)
        numerator = ttnn.sum(ttnn.matmul(e_w, v_all), dim=0, keepdim=True)
        if self.sp > 1:
            # One gather of `[numerator | denominator | max]` per layer instead of three.
            packed = self._gather_windows(ttnn.concat([numerator, denominator, m], dim=-1), dim=0)
            d = self.head_dim
            shape = list(packed.shape)
            numerator = ttnn.slice(packed, [0, 0, 0, 0], shape[:3] + [d])
            denominator = ttnn.slice(packed, [0, 0, 0, d], shape[:3] + [d + 1])
            m_all = ttnn.slice(packed, [0, 0, 0, d + 1], shape[:3] + [d + 2])
            m = ttnn.max(m_all, dim=0, keepdim=True)
            weight = ttnn.exp(ttnn.subtract(m_all, m))
            denominator = ttnn.sum(ttnn.multiply(denominator, weight), dim=0, keepdim=True)
            numerator = ttnn.sum(ttnn.multiply(numerator, weight), dim=0, keepdim=True)

        s_s = ttnn.add(ttnn.multiply(ttnn.matmul(qs, ks_block, transpose_b=True), scale), self._suffix_score_mask)
        m_new = ttnn.maximum(m, ttnn.max(s_s, dim=-1, keepdim=True))
        carry = ttnn.exp(ttnn.subtract(m, m_new))
        e_s = ttnn.exp(ttnn.subtract(s_s, m_new))
        denominator = ttnn.add(ttnn.multiply(denominator, carry), ttnn.sum(e_s, dim=-1, keepdim=True))
        numerator = ttnn.add(ttnn.multiply(numerator, carry), ttnn.matmul(e_s, vs_block))
        return ttnn.divide(numerator, denominator)

    def _layer(self, index: int, block, x, suf, rope, suffix_rope, masks):
        cached_k, cached_v = self._cache[index]
        attn = block.attn
        windows = x.shape[0]

        def qkv(hidden, cos, sin):
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                attn.to_qkv(hidden), num_heads=self.heads, num_kv_heads=self.heads, transpose_k_heads=False
            )
            return attn._norm_rope(q, cos, sin), attn._norm_rope(k, cos, sin), v

        q, k, v = qkv(block.norm1(x), *rope)
        qs, ks, vs = qkv(block.norm1(suf), *suffix_rope)
        ks_block = ttnn.concat([ks, self._suffix_pad], dim=2)
        vs_block = ttnn.concat([vs, self._suffix_pad], dim=2)
        k_all = ttnn.concat([cached_k, k], dim=2)
        v_all = ttnn.concat([cached_v, v], dim=2)

        per_window = ttnn.Shape([windows, 1, 1, 1])
        o = ttnn.transformer.scaled_dot_product_attention(
            q,
            ttnn.concat([k_all, ttnn.repeat(ks_block, per_window)], dim=2),
            ttnn.concat([v_all, ttnn.repeat(vs_block, per_window)], dim=2),
            attn_mask=masks["window"],
            is_causal=False,
            program_config=self.sdpa_config,
            compute_kernel_config=attn.sdpa_compute_kernel_config,
        )
        o = ttnn.reshape(ttnn.transformer.concatenate_heads(o), (windows, 1, self.rows, self.dim))
        o_s = self._suffix_attention(
            qs, k_all, v_all, ks_block, vs_block, masks["suffix_scores"], 1.0 / math.sqrt(self.head_dim)
        )
        o_s = ttnn.reshape(ttnn.transformer.concatenate_heads(o_s), suf.shape)

        x = ttnn.add(x, attn.to_out(o))
        suf = ttnn.add(suf, attn.to_out(o_s))
        x = ttnn.add(x, block.ff2(block.ff1(block.norm2(x))))
        suf = ttnn.add(suf, block.ff2(block.ff1(block.norm2(suf))))

        rows = k_all.shape[2]
        start = rows - cached_k.shape[2]
        for source, cache in ((k_all, cached_k), (v_all, cached_v)):
            newest = ttnn.slice(source, [0, 0, start, 0], [windows, self.heads, rows, self.head_dim])
            ttnn.copy(newest, cache)
            ttnn.deallocate(newest)
        return x, suf

    def _windows_to_torch(self, t: ttnn.Tensor) -> torch.Tensor:
        """All windows in order: one device per window shard, from the first index of the replicated axis."""
        if self.sp == 1:
            return local_device_to_torch(t)
        shards = {}
        for coord, device_tensor in zip(t.tensor_topology().mesh_coords(), ttnn.get_device_tensors(t)):
            other = coord[1 - self.sp_axis]
            if other == 0:
                shards[coord[self.sp_axis]] = ttnn.to_torch(device_tensor)
        return torch.cat([shards[i] for i in range(self.sp)])

    @traced_function(device=lambda self: self.mesh_device, clone_prep_inputs=False, prep_run=False)
    def _device_chunk(self, latents, suf, rope_cos, rope_sin, suffix_cos, suffix_sin, cached: int):
        x = self.embed(latents)
        shift = self.shifts[0]
        for index, block in enumerate(self.blocks):
            if self.shifts[index] != shift:
                x = self._relayout(x, shift, self.shifts[index])
                shift = self.shifts[index]
            x, suf = self._layer(
                index, block, x, suf, (rope_cos, rope_sin), (suffix_cos, suffix_sin), self._masks[(cached, shift)]
            )
        out = self.proj_out(self.norm_out(x))
        if self.emits_rgb:
            out = ttnn.add(ttnn.multiply(ttnn.typecast(out, ttnn.float32), self._rgb_scale), self._rgb_bias)
            # The uint8 typecast truncates, like `byte()`, and wraps out of range, hence the clamp first.
            out = ttnn.typecast(ttnn.clamp(out, 0.0, 255.0), ttnn.uint8)
        # Untilized on device: a host-side untilize of the 37.7M output values costs more than the transfer.
        return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)

    def release_trace(self) -> None:
        """Drop the captured chunk; required before anything long-lived is allocated on these devices."""
        tracer = type(self)._device_chunk._tracers.get(self)
        if tracer is not None:
            tracer.release_trace()

    def decode_chunk(self, z: torch.Tensor) -> torch.Tensor:
        """`StreamingCausalViT`'s stream interface."""
        return self.forward(z)

    @torch.no_grad()
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """`z`: normalized latents `(1, chunk, C, H, W)`. Returns decode_base pixels `(1, 3, frames, 16H, 16W)`, or
        uint8 RGB in the same layout with `rgb`."""
        if tuple(z.shape[1:2] + z.shape[-2:]) != (self.chunk, self.H, self.W):
            raise ValueError(f"expected {self.chunk} latents of {self.H}x{self.W}, got {tuple(z.shape)}")
        t0 = self.t
        cached = min(t0, self.cache_latents)
        first = self._grids[self.shifts[0]]
        tokens = z[0].permute(0, 2, 3, 1).reshape(-1, z.shape[2]).float()
        window_rows = first["tokens"].flatten()
        latents = torch.zeros(window_rows.numel(), z.shape[2])
        latents[window_rows >= 0] = tokens[window_rows[window_rows >= 0]]
        register = self.ref.register_tokens.detach().float()
        suffix = torch.zeros(1, 1, _TILE, self.dim)
        suffix[0, 0, : register.shape[1]] = register[0]

        # Once the cache is full every chunk has one shape, so the steady state replays a captured trace.
        out = self._device_chunk(
            self._sharded(latents.reshape(first["windows"], 1, self.rows, -1)),
            self._replicated(suffix),
            *self._rope(t0),
            *self._suffix_rope(t0),
            cached=cached,
            traced=self.trace and cached == self.cache_latents,
        )
        out = self._windows_to_torch(out).reshape(-1, self.ref.proj_out.out_features)
        if not self.emits_rgb:
            out = out.float()
        rows = self._grids[self.shifts[-1]]["tokens"].flatten()
        pixels = torch.empty(self.chunk * self.H * self.W, out.shape[-1], dtype=out.dtype)
        pixels[rows[rows >= 0]] = out[rows >= 0]
        self.t = t0 + self.chunk
        return _decode_tail(self.ref, pixels.reshape(self.chunk, self.H, self.W, -1), t0)


def _student_state(block: torch.nn.Module) -> dict[str, torch.Tensor]:
    return dict(block.state_dict())


def _decode_tail(ref: torch.nn.Module, out: torch.Tensor, t0: int) -> torch.Tensor:
    """The reference's unpatchify and alignment-frame trim for one chunk of `proj_out` rows `(C, H, W, 3*pt*p*p)`."""
    from causal_vit_decoder import OUT_CH, PATCH, PATCH_T

    C, H, W = out.shape[:3]
    out = out.view(1, C, H, W, OUT_CH, PATCH_T, PATCH, PATCH).permute(0, 4, 1, 5, 2, 6, 3, 7)
    out = out.reshape(1, OUT_CH, C * PATCH_T, H * PATCH, W * PATCH)
    keep = torch.ones(C * PATCH_T, dtype=torch.bool)
    for i in range(C):
        if (t0 + i) % ref.block_latents == 0:
            keep[i * PATCH_T : i * PATCH_T + ref.frames_to_trim] = False
    return out[:, :, keep]
