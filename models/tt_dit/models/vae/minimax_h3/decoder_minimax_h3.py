# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The MiniMax-H3 visual VAE decoder: a 36-layer non-causal ViT, inner dim 2048.

Unlike ``vae_ltx.py`` / ``vae_wan2_1.py``, which are CNN decoders, this one is a plain
transformer, so it reuses ``layers/linear.py`` and the attention machinery rather than
needing new conv primitives. Every latent voxel becomes one token; with tiling on, one
decode call is always a ``(1, 24, 7, 16, 16)`` latent, i.e. ``7*16*16 = 1792`` patches
plus a 5-token suffix, and ``proj_out`` expands each token into a
``4 x 16 x 16`` block of 3-channel pixels.

Four easily-missed details, all verified against the pinned reference:

* **``scale1`` / ``scale2`` are per-channel LayerScale multipliers**, not norms:
  ``h = h + attn(norm1(h)) * scale1`` then ``h = h + ff(norm2(h)) * scale2``.
* **The cls token is ``torch.zeros_like``, not a parameter.** The sequence is
  ``[patches, register_tokens(4), zero_cls(1)]`` and all five suffix rows take position
  id 0 on every axis, so their RoPE is the identity.
* The reference computes ``norm1``/``norm2`` and the q/k norms **in fp32** whatever the
  compute dtype.
* ``norm1``/``norm2`` are RMSNorm eps 1e-5, **weight-only**, so tt_dit's ``RMSNorm``
  needs ``bias=False``; the q/k norms have ``elementwise_affine=False`` and therefore no
  parameters at all.

RoPE rotates only 48 of each head's 64 lanes and pairs lane *i* with *i + 24*. See
``rope_minimax_h3.py``: the q/k weight rows are permuted once at load time so
``ttnn.alt_complex_rotate90`` computes exactly the reference rotation with no slicing.

The 1792 patches are tile-aligned but 1797 is not, so the suffix is padded out to a full
tile and the pad columns are masked in attention. Without the mask those rows are not
neutral -- they would corrupt every softmax.
"""

from __future__ import annotations

import torch

import ttnn

from ....layers.linear import Linear
from ....layers.module import Module, ModuleList, Parameter
from ....layers.normalization import LayerNorm, RMSNorm
from .rope_minimax_h3 import head_lane_permutation, rope_tables

TILE = 32


def padded_sequence_length(num_patches: int, num_suffix_tokens: int) -> int:
    """Round ``num_patches + num_suffix_tokens`` up to a whole tile."""
    total = num_patches + num_suffix_tokens
    return ((total + TILE - 1) // TILE) * TILE


class MiniMaxH3ViTAttention(Module):
    """Self-attention over all tokens, with per-head parameter-free q/k RMS and partial RoPE.

    The checkpoint stores ``to_q`` / ``to_k`` / ``to_v`` separately; they are fused into a
    single projection at load time, and the q/k halves carry the RoPE lane permute.
    """

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.mesh_device = mesh_device

        inner = num_heads * head_dim
        self.to_qkv = Linear(dim, 3 * inner, bias=True, mesh_device=mesh_device, dtype=dtype)
        self.to_out = Linear(inner, dim, bias=True, mesh_device=mesh_device, dtype=dtype)

        # Both `attention_wan.py:120` and `attention_ltx.py:158` configure their SDPA; this
        # follows them -- HiFi2 with fp32 accumulation off is the standard SDPA setting
        # there, and it roughly doubles matmul throughput against the HiFi4 default.
        # Swept at the decoder's exact SDPA shape ([1, 32, 1824, 64] bf16) with a single-op
        # Chosen by a min-of-20 op benchmark; whole-decoder wall clock jitters too much to
        # resolve it. q=k=192 with HiFi2 is ~2.95x the default blocking, 128 is slightly worse,
        # and 256 and above hang the sweep. SDPA is ~40 % of layer device time.
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=192,
            k_chunk_size=192,
            exp_approx_mode=False,  # False is more correct, matching wan/ltx
        )
        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
        )

        # The elementwise and norm ops all default to HiFi4, which the profile shows costs
        # 21.5 % of layer device time (BinaryNg 13.2 %, LayerNorm 5.1 %, Typecast 3.2 %,
        # Unary 1.6 %). None of them is a matmul; HiFi4 buys nothing here. fp32 accumulation
        # stays on for the q/k RMS, which the reference computes in fp32.
        self.elementwise_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Fuse q/k/v, permute the q/k lanes for RoPE, and flatten ``to_out.0``."""
        if "to_q.weight" in state:
            permutation = head_lane_permutation(self.head_dim, 0.75)

            def permute_heads(tensor: torch.Tensor) -> torch.Tensor:
                # Rows are (head, lane); permute lanes within each head.
                if tensor.dim() == 2:
                    reshaped = tensor.view(self.num_heads, self.head_dim, tensor.shape[-1])
                    return reshaped.index_select(1, permutation).reshape_as(tensor)
                reshaped = tensor.view(self.num_heads, self.head_dim)
                return reshaped.index_select(1, permutation).reshape_as(tensor)

            state["to_qkv.weight"] = torch.cat(
                [
                    permute_heads(state.pop("to_q.weight")),
                    permute_heads(state.pop("to_k.weight")),
                    state.pop("to_v.weight"),
                ],
                dim=0,
            )
            state["to_qkv.bias"] = torch.cat(
                [permute_heads(state.pop("to_q.bias")), permute_heads(state.pop("to_k.bias")), state.pop("to_v.bias")],
                dim=0,
            )
        for suffix in ("weight", "bias"):
            key = f"to_out.0.{suffix}"
            if key in state:
                state[f"to_out.{suffix}"] = state.pop(key)

    def _rms(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Parameter-free RMS over the head dim, computed in fp32 like the reference."""
        original = x.get_dtype()
        if original != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        x = ttnn.rms_norm(x, epsilon=self.eps, compute_kernel_config=self.elementwise_compute_kernel_config)
        return ttnn.typecast(x, original) if original != ttnn.float32 else x

    def forward(
        self,
        x: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.to_qkv(x)

        # One fused op rather than chunk + 3 reshapes + 3 permutes: profiled per layer, that
        # hand-rolled head plumbing costs 25.8 % of device time in ReshapeView (800 us mean)
        # and 10.1 % in Transpose -- 36 % on data movement, more than every matmul combined.
        # wan (attention_wan.py:401) and ltx (attention_ltx.py:463)
        # both use this op. transpose_k_heads=False because SDPA wants K as [B,H,S,D], and
        # the layout it expects, [Q all heads | K all heads | V all heads], is exactly what
        # _prepare_torch_state's cat([q, k, v], dim=0) already produces.
        query, key, value = ttnn.experimental.nlp_create_qkv_heads(
            ttnn.reshape(qkv, (batch, 1, seq_len, qkv.shape[-1])),
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
        )

        query = self._rms(query)
        key = self._rms(key)

        # Partial RoPE: the lane permute at load time makes this a full-width op.
        query = ttnn.add(ttnn.mul(query, rope_cos), ttnn.mul(ttnn.alt_complex_rotate90(query), rope_sin))
        key = ttnn.add(ttnn.mul(key, rope_cos), ttnn.mul(ttnn.alt_complex_rotate90(key), rope_sin))

        attended = ttnn.transformer.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            is_causal=False,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        attended = ttnn.reshape(
            ttnn.experimental.nlp_concat_heads(attended), (batch, seq_len, self.num_heads * self.head_dim)
        )
        return self.to_out(attended)


class MiniMaxH3TransformerBlock(Module):
    """``h += attn(norm1(h)) * scale1``; ``h += ff(norm2(h)) * scale2``."""

    def __init__(
        self,
        dim: int,
        *,
        num_heads: int,
        head_dim: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        # Weight-only RMSNorm: tt_dit defaults bias=True and every H3 norm has no bias.
        self.norm1 = RMSNorm(dim, norm_eps=eps, bias=False, mesh_device=mesh_device, dtype=dtype)
        self.attn = MiniMaxH3ViTAttention(
            dim, num_heads=num_heads, head_dim=head_dim, eps=eps, mesh_device=mesh_device, dtype=dtype
        )
        self.norm2 = RMSNorm(dim, norm_eps=eps, bias=False, mesh_device=mesh_device, dtype=dtype)
        inner = dim * ffn_mult
        # The checkpoint's ``ff.net.0.proj`` packs ``[value; gate]``, which is exactly
        # tt_dit's swiglu convention, so the halves load as-is. This is unlike the H3
        # *DiT*, whose ``fc1`` halves must be swapped -- that swap came from the raw
        # MiniMax layout, not the diffusers-converted one, and applying it here would
        # silently corrupt every FFN. Verified against ``diffusers.FeedForward``: the
        # first half is the value (out = W2(first * silu(second))), and the swapped
        # order does not reproduce the reference, so exactly one order is right.
        self.ff1 = Linear(dim, inner, bias=True, activation_fn="swiglu", mesh_device=mesh_device, dtype=dtype)
        self.ff2 = Linear(inner, dim, bias=True, mesh_device=mesh_device, dtype=dtype)
        # LayerScale, initialised to zeros in the reference and trained.
        # No scale1/scale2 Parameters: LayerScale is folded into to_out / ff2 at load time
        # (see _prepare_torch_state), which is exact and saves two multiplies per layer.

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for index, name in ((0, "ff1"), (2, "ff2")):
            source = f"ff.net.{index}.proj" if index == 0 else f"ff.net.{index}"
            for suffix in ("weight", "bias"):
                key = f"{source}.{suffix}"
                if key in state:
                    state[f"{name}.{suffix}"] = state.pop(key)
        # Fold LayerScale into the preceding projection. `to_out(x) * scale1` is
        # `x @ (W_out * scale1) + b_out * scale1` because scale is per *output* channel, so
        # this is exact, not an approximation. Same for scale2 into ff2. Removes two
        # elementwise multiplies per layer from BinaryNg, which is 22 calls and 13.2 % of
        # layer device time.
        #
        # Torch layout here is [out, in], so scale multiplies rows; the Linear's own
        # _prepare_torch_state transposes afterwards.
        for scale_name, target in (("scale1", "attn.to_out.0"), ("scale2", "ff2")):
            scale = state.pop(scale_name, None)
            if scale is None:
                continue
            flat = scale.reshape(-1)
            weight_key, bias_key = f"{target}.weight", f"{target}.bias"
            if weight_key in state:
                state[weight_key] = state[weight_key] * flat.unsqueeze(1)
            if bias_key in state:
                state[bias_key] = state[bias_key] * flat

    def forward(
        self,
        x: ttnn.Tensor,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        x = ttnn.add(x, self.attn(self.norm1(x), rope_cos, rope_sin, attention_mask))
        return ttnn.add(x, self.ff2(self.ff1(self.norm2(x))))


class MiniMaxH3ViTDecoder3d(Module):
    """Latent voxels to pixels: ``proj_in -> 36 blocks -> norm_out -> proj_out -> unpatchify``.

    Shape-specialised on ``(num_frames, height, width)`` of the *latent* tile, because the
    RoPE tables, the padded suffix and the attention mask are all constants for a given
    shape -- and with tiling on there is only ever one shape.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        patch_size_t: int = 4,
        num_layers: int = 36,
        num_heads: int = 32,
        head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        eps: float = 1e-5,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        dim = num_heads * head_dim
        self.dim = dim
        self.latent_shape = (num_frames, height, width)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.num_register_tokens = num_register_tokens
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.num_patches = num_frames * height * width
        # +1 for the zero cls token the reference appends after the register tokens.
        self.num_suffix_tokens = num_register_tokens + 1
        self.seq_len = padded_sequence_length(self.num_patches, self.num_suffix_tokens)

        self.proj_in = Linear(in_channels, dim, bias=True, mesh_device=mesh_device, dtype=dtype)
        self.transformer_blocks = ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    ffn_mult=ffn_mult,
                    eps=eps,
                    mesh_device=mesh_device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = LayerNorm(dim, norm_eps=eps, bias=True, mesh_device=mesh_device)
        self.proj_out = Linear(
            dim, out_channels * patch_size_t * patch_size * patch_size, bias=True, mesh_device=mesh_device, dtype=dtype
        )

        # One constant covering the register tokens, the zero cls token, and the tile pad.
        self.suffix = Parameter(total_shape=[1, self.seq_len - self.num_patches, dim], device=mesh_device, dtype=dtype)

        cos, sin = rope_tables(
            num_frames,
            height,
            width,
            num_suffix_tokens=self.seq_len - self.num_patches,
            attention_head_dim=head_dim,
            rope_dim_ratio=rope_dim_ratio,
            theta=rope_theta,
            permuted=True,
        )
        self.rope_cos = Parameter(total_shape=[1, 1, self.seq_len, head_dim], device=mesh_device, dtype=dtype)
        self.rope_sin = Parameter(total_shape=[1, 1, self.seq_len, head_dim], device=mesh_device, dtype=dtype)
        self._rope_host = (cos.reshape(1, 1, self.seq_len, head_dim), sin.reshape(1, 1, self.seq_len, head_dim))

        # Mask the tile-pad columns: they are not neutral in a softmax.
        mask = torch.zeros(1, 1, self.seq_len, self.seq_len)
        valid = self.num_patches + self.num_suffix_tokens
        if valid < self.seq_len:
            mask[..., valid:] = float("-inf")
        self.attention_mask = Parameter(total_shape=[1, 1, self.seq_len, self.seq_len], device=mesh_device, dtype=dtype)
        self._mask_host = mask

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Build the fused suffix constant, and the RoPE / mask constants.

        ``register_tokens`` is ``(1, 4, dim)``; the cls token is a runtime zero in the
        reference, and the remaining rows are tile padding, so all three fold into one
        parameter.
        """
        register = state.pop("register_tokens", None)
        suffix_rows = self.seq_len - self.num_patches
        if register is not None:
            pad_rows = suffix_rows - register.shape[1]
            state["suffix"] = torch.cat([register, register.new_zeros((1, pad_rows, register.shape[2]))], dim=1)
        cos, sin = self._rope_host
        state["rope_cos"] = cos
        state["rope_sin"] = sin
        state["attention_mask"] = self._mask_host

    def forward(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """``(1, num_patches, in_channels)`` latent tokens to ``(1, seq_len, C*pt*p*p)``.

        The caller flattens the latent voxel grid to tokens and unpatchifies the result;
        it already owns the tiling, hence the host-side reshapes. :func:`unpatchify` is the
        tail, and it crops the suffix rows off before reshaping.
        """
        assert (
            tokens.shape[1] == self.num_patches
        ), f"expected {self.num_patches} patch tokens for latent {self.latent_shape}, got {tokens.shape[1]}"
        hidden = self.proj_in(tokens)
        # One concat covers the register tokens, the zero cls token and the tile pad.
        hidden = ttnn.concat([hidden, self.suffix.data], dim=1)
        for block in self.transformer_blocks:
            hidden = block(hidden, self.rope_cos.data, self.rope_sin.data, self.attention_mask.data)
        return self.proj_out(self.norm_out(hidden))


def unpatchify(
    tokens: torch.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    out_channels: int = 3,
    patch_size: int = 16,
    patch_size_t: int = 4,
) -> torch.Tensor:
    """``(1, num_patches, C*pt*p*p)`` to ``(1, C, T*pt, H*p, W*p)`` -- the reference's tail."""
    batch = tokens.shape[0]
    tokens = tokens[:, : num_frames * height * width, :]
    tokens = tokens.view(batch, num_frames, height, width, out_channels, patch_size_t, patch_size, patch_size)
    tokens = tokens.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return tokens.reshape(batch, out_channels, num_frames * patch_size_t, height * patch_size, width * patch_size)
