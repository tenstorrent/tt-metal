# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of the HunyuanImage-3.0 SigLIP2 vision encoder + aligner.
#
# Upstream reference
# ------------------
#   HunyuanImage-3.0/hunyuan_image_3/siglip2.py
#     Siglip2VisionEmbeddings   lines 54-152
#     Siglip2Attention          lines 155-228
#     Siglip2MLP                lines 301-313
#     Siglip2EncoderLayer       lines 316-363
#     Siglip2Encoder            lines 366-444
#     Siglip2VisionTransformer  lines 478-540
#     LightProjector            lines 543-564
#
# HunyuanImage I2I call site (modeling_hunyuan_image_3.py)
# --------------------------------------------------------
#     image_embeds = self.vision_model(pixel_values, attention_mask,
#                                      spatial_shapes=...).last_hidden_state
#     image_embeds = self.vision_aligner(image_embeds)
#
# Checkpoint weight prefixes (safetensors keys)
# ---------------------------------------------
#     vision_model.embeddings.*
#     vision_model.encoder.layers.{i}.*
#     vision_model.post_layernorm.*
#     vision_aligner.layers.*          (mlp_gelu, depth=2: Linear-GELU-Linear)
#
# Out of scope for I2I (pooler output unused):
#     vision_model.head.*
#
# Implementation plan (do in this order)
# --------------------------------------
#   1. ref/vision/siglip2.py
#        Copy/adapt upstream siglip2.py as the PyTorch golden for PCC tests.
#
#   2. tests/pcc/test_siglip2_embeddings.py
#        PCC: patch_embedding + resize_positional_embeddings + add.
#        Pos resize uses device matmul (R @ pos_grid); R is cached on device
#        after one-time host build via prewarm_pos_geometries().
#
#   3. HunyuanTtSiglip2Attention (single layer)
#        q_proj / k_proj / v_proj / out_proj via ttnn.linear.
#        SDPA via ttnn.transformer.scaled_dot_product_attention with additive mask.
#        Tests: tests/pcc/test_siglip2_attention.py against ref layer 0.
#
#   4. HunyuanTtSiglip2EncoderLayer
#        Pre-norm LayerNorm -> attn -> residual -> pre-norm -> MLP -> residual.
#        LayerNorm: start with torch reference on host, then ttnn.layer_norm if needed.
#        Tests: tests/pcc/test_siglip2_encoder_layer.py (layer 0 weights).
#
#   5. HunyuanTtSiglip2Encoder + HunyuanTtSiglip2Vision
#        Stack 27 layers; add post_layernorm.
#        Gate at HY_VIT_NUM_LAYERS env (same pattern as HY_NUM_LAYERS in backbone tests).
#        Tests: tests/pcc/test_siglip2_vision.py (full vision_model, no head).
#
#   6. HunyuanTtLightProjector
#        Two Linear+GELU blocks (1152 -> 4096 -> 4096). Reuse ttnn.linear + ttnn.gelu.
#        Tests: tests/pcc/test_siglip2_aligner.py.
#
#   7. End-to-end vision + aligner
#        tests/pcc/test_siglip2_e2e.py with real Siglip2ImageProcessor inputs.
#
#   8. Pipeline hook (later milestone)
#        instantiate_vit_image_tokens() scatter into the Hunyuan sequence.
#
# Device inputs (Siglip2VisionInputs — all ttnn.Tensor on device)
# -------------------------------------------------------------
#   pixel_values:          [B, max_num_patches, 3 * patch_size * patch_size]
#   spatial_shapes_hw:     ((token_h, token_w), ...) per batch item (Python tuples)
#   pixel_attention_mask:  [B, max_num_patches]  (1 = real patch, 0 = pad)
#
# Config: siglip2-so400m-patch16-naflex (bundled ref/tokenizer/assets/config.json)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import ttnn
from models.common.lightweightmodule import LightweightModule

# Defaults from ref/tokenizer/assets/config.json -> "vit" + "vit_aligner"
VIT_CONFIG = {
    "hidden_size": 1152,
    "intermediate_size": 4304,
    "num_attention_heads": 16,
    "num_hidden_layers": 27,
    "num_channels": 3,
    "patch_size": 16,
    "num_patches": 256,
    "layer_norm_eps": 1e-6,
    "attention_dropout": 0.0,
    "hidden_act": "gelu_pytorch_tanh",
}

ALIGNER_CONFIG = {
    "projector_type": "mlp_gelu",
    "input_dim": 1152,
    "n_embed": 4096,
    "depth": 2,
}

WEIGHT_PREFIX_VISION = "vision_model"
WEIGHT_PREFIX_ALIGNER = "vision_aligner"


@dataclass(frozen=True)
class Siglip2VisionInputs:
    """Device-resident input bundle for the vision encoder.

    All tensors are on-device ``ttnn.Tensor`` values. ``spatial_shapes_hw`` is
    plain Python metadata (token_h, token_w per batch item). ``attention_mask_4d``
    is built on device via ``create`` so forwards never rebuild it on host.
    """

    pixel_values: ttnn.Tensor
    spatial_shapes_hw: tuple[tuple[int, int], ...]
    pixel_attention_mask: ttnn.Tensor
    attention_mask_4d: ttnn.Tensor

    @classmethod
    def create(
        cls,
        pixel_values: ttnn.Tensor,
        spatial_shapes_hw: tuple[tuple[int, int], ...],
        pixel_attention_mask: ttnn.Tensor,
        *,
        attention_mask_4d: ttnn.Tensor | None = None,
    ) -> "Siglip2VisionInputs":
        """Bundle on-device tensors for vision forward (no host tensors)."""
        if attention_mask_4d is None:
            attention_mask_4d = build_siglip2_attention_mask(pixel_attention_mask)
        return cls(
            pixel_values=pixel_values,
            spatial_shapes_hw=spatial_shapes_hw,
            pixel_attention_mask=pixel_attention_mask,
            attention_mask_4d=attention_mask_4d,
        )


def build_siglip2_attention_mask(
    pixel_attention_mask: ttnn.Tensor,
    *,
    neg_fill: float = -1e30,
) -> ttnn.Tensor:
    """Device-side additive SDPA mask [B, 1, S, S] from an on-device padding mask.

    Operates entirely on TTNN tensors: the [B, S] padding mask (already on device)
    is expanded to the [B, 1, S, S] additive mask with ttnn ops only (no Torch,
    no from_torch). Equivalent to prepare_4d_attention_mask:
    additive[b, 0, :, j] = 0 if key j is valid else neg_fill.
    """
    batch, seq_len = pixel_attention_mask.shape
    # [B, S] -> [B, 1, 1, S] (key axis last); reshape is a view, input is caller-owned.
    mask = ttnn.reshape(pixel_attention_mask, [batch, 1, 1, seq_len])
    # inv = 1 - mask  -> 0 for valid keys, 1 for padded keys
    inv = ttnn.add(ttnn.multiply(mask, -1.0), 1.0)
    # additive = inv * neg_fill -> 0 for valid keys, neg_fill for padded keys
    additive = ttnn.multiply(inv, neg_fill)
    ttnn.deallocate(inv)
    # broadcast over the query axis: [B, 1, 1, S] -> [B, 1, S, S]
    additive = ttnn.repeat(additive, (1, 1, seq_len, 1))
    return additive


def _pos_geometry_key(spatial_shapes_hw: tuple[tuple[int, int], ...], max_length: int) -> tuple:
    flat = tuple(coord for hw in spatial_shapes_hw for coord in hw)
    return (max_length, flat)


def resize_positional_embeddings_tt(
    device,
    pos_grid_tt: ttnn.Tensor,
    spatial_shapes_hw: tuple[tuple[int, int], ...],
    max_length: int,
    *,
    r_cache: dict | None = None,
    num_patches: int = VIT_CONFIG["num_patches"],
    dtype=ttnn.bfloat16,
    compute_kernel_config=None,
) -> ttnn.Tensor:
    """Device naflex pos-embedding resize: resized = R @ pos_grid (one matmul/image).

    Resize matrices R [max_length, num_patches] are cached on device in
    ``r_cache`` (keyed by (height, width, max_length)). R is built once per
    unique geometry at prewarm/init (ref golden helper + ``ttnn.from_torch``),
    then stays on device. Returns [B, max_length, hidden] on device.
    """
    from models.experimental.hunyuan_image_3_0.ref.vision.siglip2 import build_resize_matrix_host

    if r_cache is None:
        r_cache = {}

    outs = []
    for height, width in spatial_shapes_hw:
        r_key = (height, width, max_length)
        r_tt = r_cache.get(r_key)
        if r_tt is None:
            r = build_resize_matrix_host(num_patches, height, width, max_length)
            r_tt = ttnn.from_torch(
                r, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            r_cache[r_key] = r_tt
        o = ttnn.matmul(
            r_tt, pos_grid_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=compute_kernel_config
        )  # [max_length, hidden]
        outs.append(ttnn.reshape(o, [1, max_length, o.shape[-1]]))

    if len(outs) == 1:
        return outs[0]
    return ttnn.concat(outs, dim=0)


class HunyuanTtSiglip2VisionEmbeddings(LightweightModule):
    """Patch linear + naflex positional embeddings -> [B, S, H]."""

    def __init__(
        self,
        device,
        state_dict: dict[str, Any],
        *,
        hidden_size: int = VIT_CONFIG["hidden_size"],
        patch_size: int = VIT_CONFIG["patch_size"],
        num_channels: int = VIT_CONFIG["num_channels"],
        num_patches: int = VIT_CONFIG["num_patches"],
        weight_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.position_embedding_size = int(num_patches**0.5)

        in_features = num_channels * patch_size * patch_size
        patch_w = state_dict["embeddings.patch_embedding.weight"]  # [H, in]
        patch_b = state_dict["embeddings.patch_embedding.bias"]  # [H]
        self._patch_weight = ttnn.from_torch(
            patch_w.transpose(0, 1).contiguous(),
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._patch_bias = ttnn.from_torch(
            patch_b.reshape(1, -1).contiguous(),
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Positional grid as a device weight [num_patches, hidden] (moe pattern). The
        # naflex resize is applied on device via resize_positional_embeddings_tt:
        # resized = R @ pos_grid, with R built per-image on host (size-only, RoPE-style).
        self._pos_grid = ttnn.from_torch(
            state_dict["embeddings.position_embedding.weight"].contiguous(),
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        self._weight_dtype = weight_dtype
        self._pos_compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # Device caches — populated at init (prewarm) or on first use per geometry.
        # After warmup, forward() does zero host work and zero host->device transfers.
        self._r_cache: dict[tuple[int, int, int], ttnn.Tensor] = {}
        self._pos_cache: dict[tuple, ttnn.Tensor] = {}

    def prewarm_pos_geometries(
        self,
        geometries: list[tuple[int, int, int]],
    ) -> None:
        """Pre-upload resize matrices R and positional embeddings for known geometries.

        Each entry is ``(token_h, token_w, max_length)``. Call once after construction
        (e.g. at model load) so inference never touches the host for pos resize.

        Example::

            embeddings.prewarm_pos_geometries([(64, 64, 256), (48, 48, 256)])
        """
        for token_h, token_w, max_length in geometries:
            self.position_embeddings(((token_h, token_w),), max_length)

    def position_embeddings(
        self,
        spatial_shapes_hw: tuple[tuple[int, int], ...],
        max_length: int,
    ) -> ttnn.Tensor:
        """Cached device positional embeddings [B, max_length, H] for this geometry.

        Uses device-resident R matrices (``_r_cache``) and caches the matmul result
        in ``_pos_cache``. Returned tensor is cache-owned — do NOT deallocate.
        """
        key = _pos_geometry_key(spatial_shapes_hw, max_length)
        cached = self._pos_cache.get(key)
        if cached is None:
            cached = resize_positional_embeddings_tt(
                self.device,
                self._pos_grid,
                spatial_shapes_hw,
                max_length,
                r_cache=self._r_cache,
                num_patches=self.num_patches,
                dtype=self._weight_dtype,
                compute_kernel_config=self._pos_compute_cfg,
            )
            self._pos_cache[key] = cached
        return cached

    def forward(self, inputs: Siglip2VisionInputs) -> ttnn.Tensor:
        """Device-side bundle -> embeddings [B, S, H] on device. Pure TTNN.

        Positional embeddings come from the per-geometry cache (computed once), so
        no Torch runs here after warmup. Inputs are caller/cache-owned and are not
        deallocated by this module.
        """
        pixel_values = inputs.pixel_values
        seq_len = pixel_values.shape[1]
        pos = self.position_embeddings(inputs.spatial_shapes_hw, seq_len)

        # patch projection: Linear(pixel_values) [B, S, patch_dim] -> [B, S, H].
        patch = ttnn.linear(pixel_values, self._patch_weight, bias=self._patch_bias)
        # patch_embeds + resized positional embeddings (pos is cache-owned).
        out = ttnn.add(patch, pos)
        ttnn.deallocate(patch)
        return out


def _ensure_bsh(x: ttnn.Tensor) -> ttnn.Tensor:
    """Normalize activations to [B, S, H]. Some ops (nlp_concat_heads, layer_norm) emit [B, 1, S, H]."""
    if len(x.shape) == 4 and x.shape[1] == 1:
        return ttnn.reshape(x, [x.shape[0], x.shape[2], x.shape[3]])
    return x


def _pad_head_dim_with_zeros(
    weight: ttnn.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
    on_out_features: bool,
) -> ttnn.Tensor:
    """Pad each head from head_dim -> padded_head_dim with zeros, on device.

    head_dim=72 is not tile-aligned. Rather than padding on host with torch.zeros,
    the zero block is built with ttnn.zeros and concatenated on device so weight
    loading stays purely TTNN (a single ttnn.from_torch per weight upstream).

    on_out_features=True : weight is [in, num_heads*head_dim] (q/k/v proj, bias)
                           -> [in, num_heads*padded_head_dim]
    on_out_features=False: weight is [num_heads*head_dim, out] (out_proj input dim)
                           -> [num_heads*padded_head_dim, out]
    """
    pad = padded_head_dim - head_dim
    device = weight.device()
    dtype = weight.get_dtype()
    weight = ttnn.to_layout(weight, ttnn.ROW_MAJOR_LAYOUT)

    if on_out_features:
        leading = weight.shape[0]
        weight = ttnn.reshape(weight, [leading, num_heads, head_dim])
        if pad:
            zeros = ttnn.zeros([leading, num_heads, pad], dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            weight = ttnn.concat([weight, zeros], dim=2)
        weight = ttnn.reshape(weight, [leading, num_heads * padded_head_dim])
    else:
        trailing = weight.shape[1]
        weight = ttnn.reshape(weight, [num_heads, head_dim, trailing])
        if pad:
            zeros = ttnn.zeros([num_heads, pad, trailing], dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
            weight = ttnn.concat([weight, zeros], dim=1)
        weight = ttnn.reshape(weight, [num_heads * padded_head_dim, trailing])

    return ttnn.to_layout(weight, ttnn.TILE_LAYOUT)


class HunyuanTtSiglip2Attention(LightweightModule):
    """Multi-head self-attention (16 heads, head_dim=72, no GQA)."""

    def __init__(
        self,
        device,
        state_dict: dict[str, Any],
        *,
        layer_idx: int,
        hidden_size: int = VIT_CONFIG["hidden_size"],
        num_heads: int = VIT_CONFIG["num_attention_heads"],
        weight_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        prefix = f"encoder.layers.{layer_idx}.self_attn"

        # head_dim=72 is not tile-aligned. To keep all on-device tensors tile-aligned
        # we pad each head to padded_head_dim (=96). The projection weights are loaded
        # once with ttnn.from_torch, then the per-head zero padding is applied on device
        # (ttnn.zeros + concat), so the head split sees a clean num_heads*96 layout
        # (real data in [:72], zeros in [72:96]).
        self.padded_head_dim = ((self.head_dim + 31) // 32) * 32
        self.padded_qkv_dim = num_heads * self.padded_head_dim  # 1536

        def _load_w(name: str):
            # ttnn.linear expects weight as [in, out]
            w = state_dict[f"{prefix}.{name}.weight"].transpose(0, 1).contiguous()
            return ttnn.from_torch(w, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)

        def _load_b(name: str):
            b = state_dict[f"{prefix}.{name}.bias"].reshape(1, -1).contiguous()
            return ttnn.from_torch(b, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)

        def _pad_out(t):  # pad the num_heads*head_dim feature axis
            return _pad_head_dim_with_zeros(
                t,
                num_heads=num_heads,
                head_dim=self.head_dim,
                padded_head_dim=self.padded_head_dim,
                on_out_features=True,
            )

        # q/k/v: pad the OUTPUT feature dim (head-major) to num_heads*padded_head_dim.
        self.q_proj_w = _pad_out(_load_w("q_proj"))
        self.k_proj_w = _pad_out(_load_w("k_proj"))
        self.v_proj_w = _pad_out(_load_w("v_proj"))
        self.q_proj_b = _pad_out(_load_b("q_proj"))
        self.k_proj_b = _pad_out(_load_b("k_proj"))
        self.v_proj_b = _pad_out(_load_b("v_proj"))

        # out_proj: pad the INPUT feature dim (merged padded heads) so the head pad
        # regions contribute 0; the output dim and bias are unchanged.
        self.out_proj_w = _pad_head_dim_with_zeros(
            _load_w("out_proj"),
            num_heads=num_heads,
            head_dim=self.head_dim,
            padded_head_dim=self.padded_head_dim,
            on_out_features=False,
        )
        self.out_proj_b = _load_b("out_proj")

        # SigLIP2 scales by head_dim**-0.5. Pass it explicitly to SDPA so the padded
        # head_dim (96) cannot change the scale (real head_dim is 72).
        self.scale = self.head_dim**-0.5

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, hidden_states: ttnn.Tensor, attention_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """hidden_states [B, S, H] device -> attention output [B, S, H] device.

        Plain bidirectional MHA (no GQA, no RoPE). Input/output stay on device.
        """
        hidden_states = _ensure_bsh(hidden_states)
        bsz = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        # ---- 1. Q/K/V projections: [B, S, H] each ----
        q = ttnn.linear(
            hidden_states,
            self.q_proj_w,
            bias=self.q_proj_b,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        k = ttnn.linear(
            hidden_states,
            self.k_proj_w,
            bias=self.k_proj_b,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        v = ttnn.linear(
            hidden_states,
            self.v_proj_w,
            bias=self.v_proj_b,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # ---- 2. Split into heads: [B, num_heads, S, padded_head_dim] ----
        # Each proj already outputs num_heads*padded_head_dim (head-major, pad zeros),
        # so the fused [B,1,S,3*padded_qkv_dim] splits into clean tile-aligned heads.
        # num_kv_heads == num_heads (no GQA).
        qkv = ttnn.concat([q, k, v], dim=-1)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        qkv_4d = ttnn.reshape(qkv, [bsz, 1, seq_len, qkv.shape[-1]])
        qh, kh, vh = ttnn.experimental.nlp_create_qkv_heads(
            qkv_4d,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        qkv_4d.deallocate(True)

        # ---- 3. SDPA (bidirectional, explicit scale) ----
        attn = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            is_causal=False,
            attn_mask=attention_mask,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config,
        )
        qh.deallocate(True)
        kh.deallocate(True)
        vh.deallocate(True)

        # ---- 4. Merge heads: [B, S, H] ----
        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = _ensure_bsh(attn)

        # ---- 5. Output projection ----
        out = ttnn.linear(
            attn,
            self.out_proj_w,
            bias=self.out_proj_b,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn)
        return out


class HunyuanTtSiglip2MLP(LightweightModule):
    """fc1 -> gelu_pytorch_tanh -> fc2."""

    def __init__(
        self,
        device,
        state_dict: dict[str, Any],
        *,
        layer_idx: int,
        hidden_size: int = VIT_CONFIG["hidden_size"],
        intermediate_size: int = VIT_CONFIG["intermediate_size"],
        weight_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        prefix = f"encoder.layers.{layer_idx}.mlp"
        w0 = state_dict[f"{prefix}.fc1.weight"].transpose(0, 1).contiguous()
        b0 = state_dict[f"{prefix}.fc1.bias"].reshape(1, -1).contiguous()
        w1 = state_dict[f"{prefix}.fc2.weight"].transpose(0, 1).contiguous()
        b1 = state_dict[f"{prefix}.fc2.bias"].reshape(1, -1).contiguous()
        self.fc1_w = ttnn.from_torch(w0, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.fc1_b = ttnn.from_torch(b0, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.fc2_w = ttnn.from_torch(w1, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.fc2_b = ttnn.from_torch(b1, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[B, S, H] device -> [B, S, H] device. fc1 -> gelu_pytorch_tanh -> fc2."""
        x = _ensure_bsh(x)
        h = ttnn.linear(
            x,
            self.fc1_w,
            bias=self.fc1_b,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        # gelu_pytorch_tanh == nn.GELU(approximate="tanh")
        h = ttnn.gelu(h, fast_and_approximate_mode=True)
        out = ttnn.linear(
            h,
            self.fc2_w,
            bias=self.fc2_b,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(h)
        return out


class HunyuanTtSiglip2EncoderLayer(LightweightModule):
    """Pre-LN transformer block."""

    def __init__(
        self,
        device,
        state_dict: dict[str, Any],
        *,
        layer_idx: int,
        weight_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = HunyuanTtSiglip2Attention(device, state_dict, layer_idx=layer_idx, weight_dtype=weight_dtype)
        self.mlp = HunyuanTtSiglip2MLP(device, state_dict, layer_idx=layer_idx, weight_dtype=weight_dtype)

        # LayerNorm gamma/beta loaded as device weights (moe pattern: weights-only
        # from_torch in __init__). Standard ttnn.layer_norm convention is [1, dim] TILE.
        def _load_norm(name: str):
            w = state_dict[f"encoder.layers.{layer_idx}.{name}.weight"].reshape(1, -1).contiguous()
            b = state_dict[f"encoder.layers.{layer_idx}.{name}.bias"].reshape(1, -1).contiguous()
            return (
                ttnn.from_torch(w, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device),
                ttnn.from_torch(b, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device),
            )

        self.ln1_weight, self.ln1_bias = _load_norm("layer_norm1")
        self.ln2_weight, self.ln2_bias = _load_norm("layer_norm2")
        self.layer_norm_eps = VIT_CONFIG["layer_norm_eps"]

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, hidden_states: ttnn.Tensor, attention_mask: ttnn.Tensor | None) -> ttnn.Tensor:
        """Pre-LN block: x + attn(LN1(x)); x + mlp(LN2(x)). Pure TTNN on device."""
        hidden_states = _ensure_bsh(hidden_states)
        # ---- attention sub-block ----
        residual = hidden_states
        normed = ttnn.layer_norm(
            hidden_states,
            weight=self.ln1_weight,
            bias=self.ln1_bias,
            epsilon=self.layer_norm_eps,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = self.self_attn(normed, attention_mask)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(residual, attn_out)
        ttnn.deallocate(residual)
        ttnn.deallocate(attn_out)

        # ---- mlp sub-block ----
        residual = hidden_states
        normed = ttnn.layer_norm(
            hidden_states,
            weight=self.ln2_weight,
            bias=self.ln2_bias,
            epsilon=self.layer_norm_eps,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mlp_out = self.mlp(normed)
        ttnn.deallocate(normed)
        hidden_states = ttnn.add(residual, mlp_out)
        ttnn.deallocate(residual)
        ttnn.deallocate(mlp_out)
        return hidden_states


class HunyuanTtSiglip2Encoder(LightweightModule):
    """Stack of num_hidden_layers encoder blocks."""

    def __init__(
        self,
        device,
        state_dict: dict[str, Any],
        *,
        num_layers: int = VIT_CONFIG["num_hidden_layers"],
        weight_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.layers = [
            HunyuanTtSiglip2EncoderLayer(device, state_dict, layer_idx=i, weight_dtype=weight_dtype)
            for i in range(num_layers)
        ]

    def forward(self, hidden_states: ttnn.Tensor, attention_mask: ttnn.Tensor) -> ttnn.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return hidden_states


class HunyuanTtSiglip2Vision(LightweightModule):
    """Siglip2VisionTransformer without pooling head (I2I uses last_hidden_state only)."""

    def __init__(
        self,
        device,
        state_dict: dict[str, Any],
        *,
        num_layers: int = VIT_CONFIG["num_hidden_layers"],
        weight_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        self.embeddings = HunyuanTtSiglip2VisionEmbeddings(device, state_dict, weight_dtype=weight_dtype)
        self.encoder = HunyuanTtSiglip2Encoder(device, state_dict, num_layers=num_layers, weight_dtype=weight_dtype)
        # post_layernorm gamma/beta as device weights (moe pattern), [1, dim] TILE.
        post_w = state_dict["post_layernorm.weight"].reshape(1, -1).contiguous()
        post_b = state_dict["post_layernorm.bias"].reshape(1, -1).contiguous()
        self.post_ln_weight = ttnn.from_torch(post_w, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.post_ln_bias = ttnn.from_torch(post_b, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.layer_norm_eps = VIT_CONFIG["layer_norm_eps"]

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def prewarm_pos_geometries(self, geometries: list[tuple[int, int, int]]) -> None:
        """Pre-cache positional embeddings for known (token_h, token_w, max_length) tuples."""
        self.embeddings.prewarm_pos_geometries(geometries)

    def forward(
        self,
        inputs: Siglip2VisionInputs,
        *,
        attention_mask: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Returns last_hidden_state [B, S, 1152] on device."""
        if attention_mask is None:
            attention_mask = inputs.attention_mask_4d
        hidden = self.embeddings(inputs)
        hidden = self.encoder(hidden, attention_mask)
        out = ttnn.layer_norm(
            hidden,
            weight=self.post_ln_weight,
            bias=self.post_ln_bias,
            epsilon=self.layer_norm_eps,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return _ensure_bsh(out)


class HunyuanTtLightProjector(LightweightModule):
    """vision_aligner: mlp_gelu depth=2, 1152 -> 4096 -> 4096."""

    def __init__(
        self,
        device,
        state_dict: dict[str, Any],
        *,
        weight_dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.device = device
        # Sequential: layers.0 (Linear), layers.1 (GELU), layers.2 (Linear)
        w0 = state_dict["layers.0.weight"].transpose(0, 1).contiguous()
        b0 = state_dict["layers.0.bias"].reshape(1, -1).contiguous()
        w2 = state_dict["layers.2.weight"].transpose(0, 1).contiguous()
        b2 = state_dict["layers.2.bias"].reshape(1, -1).contiguous()
        self.w0 = ttnn.from_torch(w0, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.b0 = ttnn.from_torch(b0, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.w2 = ttnn.from_torch(w2, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        self.b2 = ttnn.from_torch(b2, dtype=weight_dtype, layout=ttnn.TILE_LAYOUT, device=device)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """[B, S, 1152] device -> [B, S, 4096] device. Linear -> GELU -> Linear."""
        x = _ensure_bsh(x)
        h = ttnn.linear(
            x,
            self.w0,
            bias=self.b0,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        # Reference uses nn.GELU() (erf), not gelu_pytorch_tanh.
        h = ttnn.gelu(h, fast_and_approximate_mode=False)
        out = ttnn.linear(
            h,
            self.w2,
            bias=self.b2,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(h)
        return out


def forward_vision_with_aligner(
    vision: HunyuanTtSiglip2Vision,
    aligner: HunyuanTtLightProjector,
    inputs: Siglip2VisionInputs,
) -> ttnn.Tensor:
    """Mirror modeling_hunyuan_image_3._forward_vision_encoder on device."""
    return aligner(vision(inputs))
