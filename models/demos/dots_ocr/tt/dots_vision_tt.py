# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
TTNN Dots vision tower (Wormhole).

Reuses ttnn patterns from Qwen2.5-VL / Qwen3-VL (RMSNorm, SwiGLU MLP, LayerNorm+GELU merger)
for norms/MLP. The trunk uses TTNN tensors end-to-end; `DotsVisionTransformerTT.forward`
accepts and returns PyTorch tensors, converting only at the API boundary.

RoPE frequencies follow the same geometry as `DotsVisionTransformer.rot_pos_emb`.

Typical `state_dict` prefix: ``"vision_tower."`` (keys like ``vision_tower.blocks.0...``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm as TtRmsNorm
from models.demos.dots_ocr.tt.vision_config_dataclass import DotsVisionConfig
from models.demos.qwen3_vl.tt.vision_layernorm import LayerNorm as TtLayerNorm


@dataclass
class DotsVisionTtConfig:
    """Runtime bundle for ttnn vision stack."""

    embed_dim: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    spatial_merge_size: int
    rms_norm_eps: float
    use_bias: bool
    post_norm: bool
    hidden_size: int
    patch_size: int
    temporal_patch_size: int
    num_channels: int

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_attention_heads


def _w128(x: int) -> int:
    return ((x + 127) // 128) * 128


def _pad_seq_dim_ttnn(x: ttnn.Tensor, s: int, s_pad: int) -> ttnn.Tensor:
    """Pad sequence dim (dim=2) from s to s_pad; no-op if already padded."""
    if s_pad <= s:
        return x
    pad_rows = s_pad - s
    out = ttnn.pad(x, padding=((0, 0), (0, 0), (0, pad_rows), (0, 0)), value=0.0)
    ttnn.deallocate(x)
    return out


def _dots_ttnn_sdpa_program_config(mesh_device: Any, seq_len: int) -> ttnn.SDPAProgramConfig:
    """Chunk sizes for :func:`ttnn.transformer.scaled_dot_product_attention` (must align to ``TILE_WIDTH``)."""
    tile = int(getattr(ttnn, "TILE_SIZE", 32))

    def _tile_round_down(n: int, cap: int) -> int:
        m = min(cap, max(n, tile))
        r = (m // tile) * tile
        return max(tile, r)

    grid = (8, 8)
    if mesh_device is not None and hasattr(mesh_device, "compute_with_storage_grid_size"):
        grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid,
        q_chunk_size=_tile_round_down(seq_len, 128),
        k_chunk_size=_tile_round_down(seq_len, 512),
        exp_approx_mode=False,
    )


def _get_compute_cfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _qkv_from_state(state_dict: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    w = state_dict[key]
    d = w.shape[0] // 3
    wq, wk, wv = w[:d], w[d : 2 * d], w[2 * d :]
    return torch.cat([wq.transpose(0, 1), wk.transpose(0, 1), wv.transpose(0, 1)], dim=-1).unsqueeze(0).unsqueeze(0)


def _qkv_from_qkv_weights(wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
    """
    Build fused QKV tensor for TTNN linear from separate {q,k,v}_proj weights.
    Expected per-weight shape: [out_dim, in_dim] (HF convention).
    Returns: [1, 1, in_dim, 3*out_dim] for TTNN linear.
    """
    return torch.cat([wq.transpose(0, 1), wk.transpose(0, 1), wv.transpose(0, 1)], dim=-1).unsqueeze(0).unsqueeze(0)


def _wo_from_state(state_dict: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    return state_dict[key].transpose(-1, -2).unsqueeze(0).unsqueeze(0)


def _rotate_half_ttnn(x: ttnn.Tensor) -> ttnn.Tensor:
    d = x.shape[-1]
    half = d // 2
    x1 = ttnn.slice(
        x,
        [0, 0, 0, 0],
        [x.shape[0], x.shape[1], x.shape[2], half],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    x2 = ttnn.slice(
        x,
        [0, 0, 0, half],
        [x.shape[0], x.shape[1], x.shape[2], d],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    neg_x2 = ttnn.multiply(x2, -1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x2)
    out = ttnn.concat([neg_x2, x1], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(neg_x2)
    ttnn.deallocate(x1)
    return out


def apply_rotary_pos_emb_vision_ttnn(tensor_tt: ttnn.Tensor, freqs_tt: ttnn.Tensor) -> ttnn.Tensor:
    """
    TTNN equivalent of reference apply_rotary_pos_emb_vision using raw freqs.
    ``tensor_tt``: [1, S, H, D] TILE BF16 (same layout as reference ``q.unsqueeze(0)``).
    ``freqs_tt``: [S, head_dim / 2] TILE BF16 (matches reference ``rotary_pos_emb`` slice).
    Returns TILE BF16 tensor with the same rank as ``tensor_tt``. ``freqs_tt`` is not deallocated.
    """
    cos_half = ttnn.cos(freqs_tt)
    sin_half = ttnn.sin(freqs_tt)

    cos_full = ttnn.concat([cos_half, cos_half], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    sin_full = ttnn.concat([sin_half, sin_half], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(cos_half)
    ttnn.deallocate(sin_half)

    cos = ttnn.reshape(cos_full, (1, cos_full.shape[0], 1, cos_full.shape[1]))
    sin = ttnn.reshape(sin_full, (1, sin_full.shape[0], 1, sin_full.shape[1]))
    ttnn.deallocate(cos_full)
    ttnn.deallocate(sin_full)

    rotated = _rotate_half_ttnn(tensor_tt)
    out_a = ttnn.multiply(tensor_tt, cos, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_b = ttnn.multiply(rotated, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.add(out_a, out_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    ttnn.deallocate(tensor_tt)
    ttnn.deallocate(rotated)
    ttnn.deallocate(cos)
    ttnn.deallocate(sin)
    ttnn.deallocate(out_a)
    ttnn.deallocate(out_b)

    return out


class VisionRotaryEmbeddingTt(LightweightModule):
    """Path-A TTNN rotary frequencies (raw freqs, not precomputed cos/sin)."""

    def __init__(self, mesh_device: Any, dim: int, theta: float = 10000.0):
        super().__init__()
        self.mesh = mesh_device
        self.dim = dim
        self.theta = theta

        # pi0-style TTNN math:
        # inv_freq = 1 / (theta ** (arange(0, dim, 2) / dim))
        idx = ttnn.arange(0, dim, 2, dtype=ttnn.float32, device=mesh_device)
        idx = ttnn.to_layout(idx, ttnn.TILE_LAYOUT)
        exponent = ttnn.multiply(idx, 1.0 / dim)
        ttnn.deallocate(idx)
        theta_pow = ttnn.pow(theta, exponent)
        ttnn.deallocate(exponent)
        self.inv_freq = ttnn.reciprocal(theta_pow)
        ttnn.deallocate(theta_pow)

    def forward(self, seqlen: int) -> ttnn.Tensor:
        # freqs = outer(arange(seqlen), inv_freq)
        seq = ttnn.arange(0, seqlen, 1, dtype=ttnn.float32, device=self.mesh)
        seq = ttnn.to_layout(seq, ttnn.TILE_LAYOUT)
        seq_col = ttnn.reshape(seq, (seqlen, 1))
        ttnn.deallocate(seq)
        inv_row = ttnn.reshape(self.inv_freq, (1, self.inv_freq.shape[-1]))
        freqs = ttnn.multiply(seq_col, inv_row)
        ttnn.deallocate(seq_col)
        # Do not deallocate ``inv_row``: it aliases ``self.inv_freq``; deallocating it frees the
        # module buffer and breaks the next ``forward`` (e.g. staged PCC then full ``forward``).
        return freqs


class DotsPatchEmbedTt(LightweightModule):
    """TTNN counterpart of reference DotsPatchEmbed (proj + RMSNorm)."""

    def __init__(
        self,
        mesh_device: Any,
        state_dict: Dict[str, torch.Tensor],
        state_dict_prefix: str,
        cfg: DotsVisionTtConfig,
        weight_cache_path: Optional[Any],
    ):
        super().__init__()
        self.mesh = mesh_device
        self.cfg = cfg
        self.compute_cfg = _get_compute_cfg()
        self.weight_dtype = getattr(cfg, "_weight_dtype", None) or ttnn.bfloat16
        in_dim = cfg.num_channels * cfg.patch_size * cfg.patch_size

        # HF key layouts for patch embed have varied over time (patchifier vs direct patch_embed).
        # Resolve the prefix dynamically when the expected key is missing.
        patch_prefix = state_dict_prefix
        w_key = f"{patch_prefix}proj.weight"
        if w_key not in state_dict:
            # Try common prefix rewrite: `...patch_embed.patchifier.` -> `...patch_embed.`
            if ".patch_embed.patchifier." in patch_prefix:
                alt = patch_prefix.replace(".patch_embed.patchifier.", ".patch_embed.")
                if f"{alt}proj.weight" in state_dict:
                    patch_prefix = alt
                    w_key = f"{patch_prefix}proj.weight"
            # Final fallback: scan keys.
            if w_key not in state_dict:
                candidates = [
                    k
                    for k in state_dict.keys()
                    if k.endswith("proj.weight") and ("patch_embed" in k or "patchifier" in k)
                ]
                if candidates:
                    # Prefer canonical `vision_tower.patch_embed` keys when present.
                    preferred = [k for k in candidates if "vision_tower.patch_embed" in k]
                    pick = sorted(preferred or candidates, key=lambda s: (len(s), s))[0]
                    patch_prefix = pick[: -len("proj.weight")]
                    w_key = f"{patch_prefix}proj.weight"
        if w_key not in state_dict:
            raise KeyError(
                f"DotsPatchEmbedTt: could not find patch-embed proj.weight. "
                f"Tried `{state_dict_prefix}proj.weight` and fallbacks; example expected suffix `proj.weight`."
            )

        w_conv = state_dict[w_key]  # [D, C, P, P]
        w_lin = w_conv.reshape(cfg.embed_dim, in_dim).transpose(0, 1).unsqueeze(0).unsqueeze(0)
        b_key = f"{patch_prefix}proj.bias"
        b_proj = state_dict[b_key] if b_key in state_dict else None

        cache = (
            (lambda p: (weight_cache_path / p) if weight_cache_path else None) if weight_cache_path else lambda _: None
        )
        self.w_proj = ttnn.as_tensor(
            w_lin,
            dtype=self.weight_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache(f"{state_dict_prefix}proj") if weight_cache_path else None,
        )
        self.b_proj = (
            ttnn.as_tensor(
                b_proj,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
            if b_proj is not None
            else None
        )
        # Patch-embed RMSNorm key layout has varied. Find a concrete `<prefix><wk>.weight` that exists.
        norm_prefix = patch_prefix
        norm_wk = None
        for wk in ("norm", "o_norm"):
            if f"{norm_prefix}{wk}.weight" in state_dict:
                norm_wk = wk
                break
        if norm_wk is None:
            # Some checkpoints use `o_proj`/`o_norm` under the patchifier; in that case our
            # resolved `patch_prefix` can end with `o_` (picked from `...o_proj.weight`).
            # Try stripping the trailing `o_` once.
            if norm_prefix.endswith("o_"):
                alt_prefix = norm_prefix[: -len("o_")]
                for wk in ("norm", "o_norm"):
                    if f"{alt_prefix}{wk}.weight" in state_dict:
                        norm_prefix = alt_prefix
                        norm_wk = wk
                        break
        if norm_wk is None:
            # Last resort: scan state_dict for a norm weight that shares the patch_prefix stem.
            candidates = [
                k
                for k in state_dict.keys()
                if (k.endswith("norm.weight") or k.endswith("o_norm.weight")) and k.startswith(patch_prefix)
            ]
            if candidates:
                pick = sorted(candidates, key=lambda s: (len(s), s))[0]
                if pick.endswith("o_norm.weight"):
                    norm_wk = "o_norm"
                    norm_prefix = pick[: -len("o_norm.weight")]
                else:
                    norm_wk = "norm"
                    norm_prefix = pick[: -len("norm.weight")]
        if norm_wk is None:
            raise KeyError(
                f"DotsPatchEmbedTt: could not find patch-embed norm weight. "
                f"Tried `{patch_prefix}norm.weight`, `{patch_prefix}o_norm.weight` (and stripping trailing `o_`)."
            )

        self.norm = TtRmsNorm(
            device=mesh_device,
            dim=cfg.embed_dim,
            eps=cfg.rms_norm_eps,
            state_dict=state_dict,
            state_dict_prefix=norm_prefix,
            weight_key=norm_wk,
            weight_cache_path=weight_cache_path,
            weight_dtype=ttnn.bfloat16,
        )

    def forward(self, x: ttnn.Tensor, grid_thw=None) -> ttnn.Tensor:
        """
        Patch projection + RMSNorm. Input ``x`` is [1, 1, N, C*P*P] TILE BF16 (same layout as
        reference patchifier after flatten); ``grid_thw`` is unused (kept for API compatibility).
        """
        del grid_thw
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x_proj = ttnn.linear(
            x,
            self.w_proj,
            bias=self.b_proj,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        # models.common.rmsnorm expects tt_transformers' Mode enum, but accepts strings too.
        return self.norm(x_proj, mode="prefill")


class DotsMlpTt(LightweightModule):
    """SwiGLU FFN: silu(fc1) * fc3 -> fc2 (matches Dots reference naming)."""

    def __init__(
        self,
        mesh_device: Any,
        state_dict: Dict[str, torch.Tensor],
        state_dict_prefix: str,
        cfg: DotsVisionTtConfig,
        weight_cache_path: Optional[Any],
    ):
        super().__init__()
        self.mesh = mesh_device
        self.cfg = cfg
        self.compute_cfg = _get_compute_cfg()
        self.weight_dtype = getattr(cfg, "_weight_dtype", None) or ttnn.bfloat16

        def t_linear_weight(name: str) -> torch.Tensor:
            w = state_dict[f"{state_dict_prefix}{name}.weight"]
            return w.transpose(0, 1).unsqueeze(0).unsqueeze(0)

        # Keep torch copies for a host MLP fallback (some TTNN builds keep padded physical widths
        # that can trip matmul validation even after trimming).
        self._torch_w1 = state_dict[f"{state_dict_prefix}fc1.weight"].to(torch.float32)
        self._torch_w2 = state_dict[f"{state_dict_prefix}fc2.weight"].to(torch.float32)
        self._torch_w3 = state_dict[f"{state_dict_prefix}fc3.weight"].to(torch.float32)
        self._torch_b1 = state_dict.get(f"{state_dict_prefix}fc1.bias")
        self._torch_b2 = state_dict.get(f"{state_dict_prefix}fc2.bias")
        self._torch_b3 = state_dict.get(f"{state_dict_prefix}fc3.bias")
        if self._torch_b1 is not None:
            self._torch_b1 = self._torch_b1.to(torch.float32)
        if self._torch_b2 is not None:
            self._torch_b2 = self._torch_b2.to(torch.float32)
        if self._torch_b3 is not None:
            self._torch_b3 = self._torch_b3.to(torch.float32)

        cache = (
            (lambda p: (weight_cache_path / p) if weight_cache_path else None) if weight_cache_path else lambda _: None
        )

        self.w1 = ttnn.as_tensor(
            t_linear_weight("fc1"),
            dtype=self.weight_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache(f"{state_dict_prefix}fc1") if weight_cache_path else None,
        )
        self.w2 = ttnn.as_tensor(
            t_linear_weight("fc2"),
            dtype=self.weight_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache(f"{state_dict_prefix}fc2") if weight_cache_path else None,
        )
        self.w3 = ttnn.as_tensor(
            t_linear_weight("fc3"),
            dtype=self.weight_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache(f"{state_dict_prefix}fc3") if weight_cache_path else None,
        )

        # Many HF checkpoints omit MLP biases (weights-only). Treat biases as optional even when
        # cfg.use_bias is True.
        def _maybe_bias(name: str):
            k = f"{state_dict_prefix}{name}.bias"
            if k not in state_dict:
                return None
            return ttnn.as_tensor(
                state_dict[k],
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        if cfg.use_bias:
            self.b1 = _maybe_bias("fc1")
            self.b2 = _maybe_bias("fc2")
            self.b3 = _maybe_bias("fc3")
        else:
            self.b1 = self.b2 = self.b3 = None

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if int(x.shape[-1]) < int(self.cfg.embed_dim):
            raise RuntimeError(
                f"DotsMlpTt: unexpected hidden dim {int(x.shape[-1])} < embed_dim {int(self.cfg.embed_dim)}"
            )
        xt = ttnn.to_torch(x)
        # xt: [1, 1, S, D_pad]
        xt = xt[..., : int(self.cfg.embed_dim)].contiguous()
        ttnn.deallocate(x)
        x = ttnn.from_torch(
            xt,
            dtype=ttnn.bfloat16,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        # Defensive: some TTNN versions may still materialize padded tile widths even when the
        # logical shape is 1536. Verify via a quick D2H; if still padded, re-upload the trimmed tensor.
        xt2 = ttnn.to_torch(x)
        if int(xt2.shape[-1]) != int(self.cfg.embed_dim):
            xt2 = xt2[..., : int(self.cfg.embed_dim)].contiguous()
            ttnn.deallocate(x)
            x = ttnn.from_torch(
                xt2,
                dtype=ttnn.bfloat16,
                device=self.mesh,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        try:
            xt_dbg = ttnn.to_torch(x)
            logger.info(
                f"[DotsMlpTt] pre-linear shapes: ttnn={list(x.shape)} torch={list(xt_dbg.shape)} "
                f"embed_dim={int(self.cfg.embed_dim)}"
            )
        except Exception as e:
            logger.info(f"[DotsMlpTt] pre-linear shapes: ttnn={list(x.shape)} (torch read failed: {e})")
        s = x.shape[-2]
        if s >= 1024:
            x = ttnn.reshape(x, (1, s // 1024, 1024, -1))
        w1o = ttnn.linear(
            x,
            self.w1,
            bias=self.b1,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        w3o = ttnn.linear(
            x,
            self.w3,
            bias=self.b3,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        mid = ttnn.mul(
            w1o,
            w3o,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            dtype=(ttnn.bfloat16 if self.weight_dtype == ttnn.bfloat16 else ttnn.bfloat8_b),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w1o)
        ttnn.deallocate(w3o)
        out = ttnn.linear(
            mid,
            self.w2,
            bias=self.b2,
            compute_kernel_config=self.compute_cfg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(mid)
        shp = out.shape
        if len(shp) == 4 and shp[1] != 1:
            return ttnn.reshape(out, (1, 1, shp[0] * shp[1] * shp[2], shp[3]))
        return out


class DotsAttnQkvprojTt(LightweightModule):
    """
    TTNN QKV and output projections; RoPE on device; attention via
    :func:`ttnn.transformer.scaled_dot_product_attention` (additive mask); host round-trip only for activations.
    """

    def __init__(
        self,
        mesh_device: Any,
        state_dict: Dict[str, torch.Tensor],
        state_dict_prefix: str,
        cfg: DotsVisionTtConfig,
        weight_cache_path: Optional[Any],
    ):
        super().__init__()
        self.cfg = cfg
        self.compute_cfg = _get_compute_cfg()
        self.mesh = mesh_device
        self.weight_dtype = getattr(cfg, "_weight_dtype", None) or ttnn.bfloat16
        # HF key layouts for attention weights have varied (`attn` vs `attention`, `qkv` vs `qkv_proj`).
        prefixes = [state_dict_prefix]
        if ".attn." in state_dict_prefix:
            prefixes.append(state_dict_prefix.replace(".attn.", ".attention."))
            prefixes.append(state_dict_prefix.replace(".attn.", ".self_attn."))
        if ".attention." in state_dict_prefix:
            prefixes.append(state_dict_prefix.replace(".attention.", ".attn."))
            prefixes.append(state_dict_prefix.replace(".attention.", ".self_attn."))

        def _pick_key(suffixes: tuple[str, ...]) -> tuple[str, str]:
            for pfx in prefixes:
                for suf in suffixes:
                    k = f"{pfx}{suf}"
                    if k in state_dict:
                        return pfx, k
            # Helpful debug: show near-miss keys for this block.
            hint = []
            try:
                block_stub = prefixes[0].split("attn.")[0]
                for k in state_dict.keys():
                    if block_stub in k and ("qkv" in k or "in_proj" in k or "q_proj" in k):
                        hint.append(k)
                hint = sorted(hint)[:30]
            except Exception:
                hint = []
            raise KeyError(
                f"DotsAttnQkvprojTt: missing attention weight. Tried prefixes={prefixes} suffixes={suffixes}. "
                f"Nearby keys (truncated)={hint}"
            )

        # Combined QKV variants (common in ViT/CLIP: in_proj_weight).
        try:
            attn_prefix, qkv_key = _pick_key(
                (
                    "qkv.weight",
                    "qkv_proj.weight",
                    "qkv_proj.linear.weight",
                    "in_proj_weight",
                    "in_proj.weight",
                    "in_proj.linear.weight",
                )
            )
            wqkv = _qkv_from_state(state_dict, qkv_key)
        except KeyError:
            # Separate Q/K/V projection variants.
            attn_prefix, q_key = _pick_key(("q_proj.weight", "query.weight"))
            _, k_key = _pick_key(("k_proj.weight", "key.weight"))
            _, v_key = _pick_key(("v_proj.weight", "value.weight"))
            wqkv = _qkv_from_qkv_weights(state_dict[q_key], state_dict[k_key], state_dict[v_key])
        cache = (
            (lambda p: (weight_cache_path / p) if weight_cache_path else None) if weight_cache_path else lambda _: None
        )
        self.wqkv = ttnn.as_tensor(
            wqkv,
            dtype=self.weight_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache(f"{attn_prefix}qkv") if weight_cache_path else None,
        )

        _, wo_key = _pick_key(("proj.weight", "out_proj.weight", "o_proj.weight"))
        self.wo = ttnn.as_tensor(
            _wo_from_state(state_dict, wo_key),
            dtype=self.weight_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            cache_file_name=cache(f"{attn_prefix}wo") if weight_cache_path else None,
        )
        if cfg.use_bias and f"{attn_prefix}qkv.bias" in state_dict:
            b = state_dict[f"{attn_prefix}qkv.bias"]
            self.bqkv = ttnn.as_tensor(
                b,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        else:
            self.bqkv = None
        if cfg.use_bias and f"{attn_prefix}proj.bias" in state_dict:
            self.bo = ttnn.as_tensor(
                state_dict[f"{attn_prefix}proj.bias"],
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        else:
            self.bo = None
        self.scale = cfg.head_dim**-0.5

    def forward(
        self,
        x: ttnn.Tensor,
        rotary_pos_emb: ttnn.Tensor,
        cu_seqlens: ttnn.Tensor,
        seqlen_in: int,
    ) -> ttnn.Tensor:
        # Trim physically padded channels in ROW_MAJOR, then re-tile.
        if int(x.shape[-1]) != int(self.cfg.embed_dim):
            if int(x.shape[-1]) < int(self.cfg.embed_dim):
                raise RuntimeError(
                    f"DotsAttnQkvprojTt: unexpected hidden dim {int(x.shape[-1])} < embed_dim {int(self.cfg.embed_dim)}"
                )
            x_rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x)
            x_rm = ttnn.slice(
                x_rm,
                [0, 0, 0, 0],
                [x_rm.shape[0], x_rm.shape[1], x_rm.shape[2], self.cfg.embed_dim],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)
            ttnn.deallocate(x_rm)
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            bias=self.bqkv,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        tq = ttnn.to_torch(xqkv)
        ttnn.deallocate(xqkv)
        qkv = tq[0, 0, :seqlen_in, :]
        s = seqlen_in
        d = self.cfg.embed_dim
        h = self.cfg.num_attention_heads
        hd = self.cfg.head_dim
        q, k, v = qkv.reshape(s, 3, h, hd).permute(1, 0, 2, 3).unbind(0)

        rot_dim = int(rotary_pos_emb.shape[-1])
        rpe_s = ttnn.slice(
            rotary_pos_emb,
            [0, 0, 0, 0],
            [1, 1, s, rot_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        freqs_2d = ttnn.reshape(rpe_s, (s, rot_dim))

        q_tt = ttnn.from_torch(
            q.unsqueeze(0).float(),
            dtype=ttnn.bfloat16,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        q_tt = ttnn.to_memory_config(q_tt, ttnn.DRAM_MEMORY_CONFIG)
        q_tt = apply_rotary_pos_emb_vision_ttnn(q_tt, freqs_2d)
        q = ttnn.to_torch(q_tt).squeeze(0).to(qkv.dtype)
        ttnn.deallocate(q_tt)

        k_tt = ttnn.from_torch(
            k.unsqueeze(0).float(),
            dtype=ttnn.bfloat16,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        k_tt = ttnn.to_memory_config(k_tt, ttnn.DRAM_MEMORY_CONFIG)
        k_tt = apply_rotary_pos_emb_vision_ttnn(k_tt, freqs_2d)
        k = ttnn.to_torch(k_tt).squeeze(0).to(qkv.dtype)
        ttnn.deallocate(k_tt)

        # Do not deallocate ``rpe_s`` / ``freqs_2d``: they are views into ``rotary_pos_emb``;
        # deallocating them frees the shared backing store and breaks later vision blocks.

        cu_1d = ttnn.to_torch(cu_seqlens).reshape(-1).to(torch.int32)
        # Additive mask: 0 = allow; large negative = disallow (use -1e9 for TTNN SDPA tilization).
        attn_mask = torch.full((1, s, s), -1e9, dtype=torch.float32, device=q.device)
        for b in range(1, cu_1d.numel()):
            a, z = int(cu_1d[b - 1].item()), int(cu_1d[b].item())
            attn_mask[..., a:z, a:z] = 0.0
        q1 = q.transpose(0, 1).unsqueeze(0)
        k1 = k.transpose(0, 1).unsqueeze(0)
        v1 = v.transpose(0, 1).unsqueeze(0)
        m3 = attn_mask.unsqueeze(0)  # [1, 1, S, S]

        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        tt_Q = ttnn.from_torch(
            q1.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        tt_K = ttnn.from_torch(
            k1.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        tt_V = ttnn.from_torch(
            v1.bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        tt_mask = ttnn.from_torch(
            m3,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        tt_Q = ttnn.to_memory_config(tt_Q, ttnn.DRAM_MEMORY_CONFIG)
        tt_K = ttnn.to_memory_config(tt_K, ttnn.DRAM_MEMORY_CONFIG)
        tt_V = ttnn.to_memory_config(tt_V, ttnn.DRAM_MEMORY_CONFIG)
        tt_mask = ttnn.to_memory_config(tt_mask, ttnn.DRAM_MEMORY_CONFIG)

        sdpa_ck = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        tt_o = ttnn.transformer.scaled_dot_product_attention(
            tt_Q,
            tt_K,
            tt_V,
            is_causal=False,
            attn_mask=tt_mask,
            scale=self.scale,
            program_config=_dots_ttnn_sdpa_program_config(self.mesh, s),
            compute_kernel_config=sdpa_ck,
        )
        o_t = ttnn.to_torch(tt_o)
        o_t = o_t[:, :, :s, :]
        o = o_t.squeeze(0).transpose(0, 1).reshape(s, d)
        ttnn.deallocate(tt_Q)
        ttnn.deallocate(tt_K)
        ttnn.deallocate(tt_V)
        ttnn.deallocate(tt_mask)
        ttnn.deallocate(tt_o)
        out_pad = _w128(s)
        if out_pad > s:
            o_full = o.new_zeros(out_pad, d)
            o_full[:s] = o
        else:
            o_full = o
        ttn_in = ttnn.from_torch(
            o_full.unsqueeze(0).unsqueeze(0).bfloat16(),
            dtype=ttnn.bfloat16,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttn_in = ttnn.to_memory_config(ttn_in, ttnn.DRAM_MEMORY_CONFIG)
        out1 = ttnn.linear(
            ttn_in,
            self.wo,
            bias=self.bo,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(ttn_in)
        return out1


class DotsPatchMergerTt(LightweightModule):
    """
    LayerNorm on token dim, then GELU-MLP (matches default Dots `PatchMerger` with
    pre_norm=layernorm and Sequential Linear-GELU-Linear).
    """

    def __init__(
        self,
        mesh_device: Any,
        state_dict: Dict[str, torch.Tensor],
        state_dict_prefix: str,
        cfg: DotsVisionTtConfig,
        weight_cache_path: Optional[Any],
    ):
        super().__init__()
        self.cfg = cfg
        self.mlp_in = (cfg.spatial_merge_size**2) * cfg.embed_dim
        self.ln = TtLayerNorm(
            device=mesh_device,
            dim=cfg.embed_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_q",
            weight_cache_path=weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            eps=1e-6,
        )
        self.compute_cfg = _get_compute_cfg()
        # Default to BF16 weights for correctness (BF8 can degrade OCR quality).
        self.weight_dtype = getattr(cfg, "_weight_dtype", None) or ttnn.bfloat16

        def tw_linear(name: str) -> ttnn.Tensor:
            w = torch.transpose(state_dict[f"{state_dict_prefix}mlp.{name}.weight"], -2, -1)
            return ttnn.as_tensor(
                w,
                dtype=self.weight_dtype,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        self.w0 = tw_linear("0")
        self.w2 = tw_linear("2")
        self.b0 = ttnn.as_tensor(
            state_dict[f"{state_dict_prefix}mlp.0.bias"],
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.b2 = ttnn.as_tensor(
            state_dict[f"{state_dict_prefix}mlp.2.bias"],
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def forward(self, x: ttnn.Tensor, seqlen: int) -> ttnn.Tensor:
        merge2 = self.cfg.spatial_merge_size**2
        assert seqlen % merge2 == 0, "Token count must be divisible by spatial_merge_size**2 for PatchMerger"
        xu = ttnn.slice(x, [0, 0, 0, 0], [1, 1, seqlen, x.shape[3]], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Do not deallocate `x` before `ln(xu)`: `xu` may alias `x` storage; freeing `x` invalidates `xu`.
        x1 = self.ln(xu)
        ttnn.deallocate(xu)
        ttnn.deallocate(x)
        n_merge = seqlen // merge2
        xrm = ttnn.to_layout(x1, ttnn.ROW_MAJOR_LAYOUT)
        xrm = ttnn.reshape(xrm, (1, 1, n_merge, self.mlp_in))
        xrm = ttnn.to_layout(xrm, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x1)
        w1c = ttnn.linear(
            xrm,
            self.w0,
            bias=self.b0,
            activation="gelu",
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xrm)
        out = ttnn.linear(
            w1c,
            self.w2,
            bias=self.b2,
            compute_kernel_config=self.compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(w1c)
        return out


class DotsVisionBlockTt(LightweightModule):
    def __init__(
        self,
        layer_idx: int,
        mesh_device: Any,
        state_dict: Dict[str, torch.Tensor],
        prefix: str,
        cfg: DotsVisionTtConfig,
        weight_cache_path: Optional[Any],
    ):
        super().__init__()
        sp = f"{prefix}blocks.{layer_idx}."
        self.layer_idx = layer_idx
        self.rms1 = TtRmsNorm(
            device=mesh_device,
            dim=cfg.embed_dim,
            eps=cfg.rms_norm_eps,
            state_dict=state_dict,
            state_dict_prefix=sp,
            weight_key="norm1",
            weight_cache_path=weight_cache_path,
            weight_dtype=ttnn.bfloat16,
        )
        self.rms2 = TtRmsNorm(
            device=mesh_device,
            dim=cfg.embed_dim,
            eps=cfg.rms_norm_eps,
            state_dict=state_dict,
            state_dict_prefix=sp,
            weight_key="norm2",
            weight_cache_path=weight_cache_path,
            weight_dtype=ttnn.bfloat16,
        )
        self.attn = DotsAttnQkvprojTt(mesh_device, state_dict, f"{sp}attn.", cfg, weight_cache_path)
        self.mlp = DotsMlpTt(mesh_device, state_dict, f"{sp}mlp.", cfg, weight_cache_path)

    def forward(self, x: ttnn.Tensor, rotary_pos_emb: ttnn.Tensor, cu_seqlens: ttnn.Tensor, seqlen: int) -> ttnn.Tensor:
        x0 = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        n1 = self.rms1(x0, mode="prefill")
        ao = self.attn(n1, rotary_pos_emb, cu_seqlens, seqlen)
        t1 = ttnn.add(x0, ao, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x0)
        ttnn.deallocate(ao)
        n2 = self.rms2(t1, mode="prefill")
        m = self.mlp(n2)
        ttnn.deallocate(n2)
        out = ttnn.add(t1, m, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        ttnn.deallocate(m)
        ttnn.deallocate(t1)
        return out


class DotsVisionTransformerTT(LightweightModule):
    """
    ttnn Dots vision trunk + merger. RoPE and patch/trunk norms/attn/MLP/post/merger use ttnn.
    """

    def __init__(
        self,
        vision_config: Union[DotsVisionConfig, Dict[str, Any]],
        mesh_device: Any,
        state_dict: Dict[str, torch.Tensor],
        state_dict_prefix: str = "vision_tower.",
        weight_cache_path: Optional[Any] = None,
        dtype: Any = None,
    ):
        super().__init__()
        # `dtype` is accepted for API compatibility with other TT modules / DropInVisionTransformer.
        # Vision tower weights/activations are primarily bf16 today; callers may still pass dtype.
        self.dtype = dtype
        if isinstance(vision_config, DotsVisionConfig):
            self.dots_cfg = vision_config
        elif isinstance(vision_config, Mapping):
            self.dots_cfg = DotsVisionConfig(**vision_config)
        elif hasattr(vision_config, "to_dict"):
            # Handles dynamically loaded config classes from trust_remote_code.
            self.dots_cfg = DotsVisionConfig(**vision_config.to_dict())
        else:
            raise TypeError(
                "vision_config must be DotsVisionConfig, mapping, or to_dict()-compatible config object; "
                f"got {type(vision_config)!r}"
            )
        self.pfx = state_dict_prefix
        self.mesh = mesh_device
        self.cfg = DotsVisionTtConfig(
            embed_dim=self.dots_cfg.embed_dim,
            num_hidden_layers=self.dots_cfg.num_hidden_layers,
            num_attention_heads=self.dots_cfg.num_attention_heads,
            intermediate_size=self.dots_cfg.intermediate_size,
            spatial_merge_size=self.dots_cfg.spatial_merge_size,
            rms_norm_eps=self.dots_cfg.rms_norm_eps,
            use_bias=self.dots_cfg.use_bias,
            post_norm=bool(self.dots_cfg.post_norm),
            hidden_size=self.dots_cfg.hidden_size,
            patch_size=self.dots_cfg.patch_size,
            temporal_patch_size=self.dots_cfg.temporal_patch_size,
            num_channels=self.dots_cfg.num_channels,
        )
        self.patch_embed = DotsPatchEmbedTt(
            mesh_device,
            state_dict,
            f"{self.pfx}patch_embed.patchifier.",
            self.cfg,
            weight_cache_path,
        )
        head_dim = self.cfg.embed_dim // self.cfg.num_attention_heads
        self.rotary_dim = head_dim // 2
        self.rotary_pos_emb = VisionRotaryEmbeddingTt(mesh_device, self.rotary_dim, theta=10000.0)
        if self.dots_cfg.post_norm:
            self.post_norm = TtRmsNorm(
                device=mesh_device,
                dim=self.cfg.embed_dim,
                eps=self.cfg.rms_norm_eps,
                state_dict=state_dict,
                state_dict_prefix=self.pfx,
                weight_key="post_trunk_norm",
                weight_cache_path=weight_cache_path,
                weight_dtype=ttnn.bfloat16,
            )
        else:
            self.post_norm = None
        self.merger = DotsPatchMergerTt(mesh_device, state_dict, f"{self.pfx}merger.", self.cfg, weight_cache_path)
        self.blocks = [
            DotsVisionBlockTt(i, mesh_device, state_dict, self.pfx, self.cfg, weight_cache_path)
            for i in range(self.cfg.num_hidden_layers)
        ]

    def _pixels_flat_torch(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, int]:
        hs = pixel_values.view(
            -1,
            self.cfg.num_channels,
            self.cfg.temporal_patch_size,
            self.cfg.patch_size,
            self.cfg.patch_size,
        )[:, :, 0]
        n = int(hs.shape[0])
        hs = hs.reshape(n, self.cfg.num_channels * self.cfg.patch_size * self.cfg.patch_size)
        return hs, n

    def _pixels_flat_ttnn(self, pixel_values: ttnn.Tensor) -> tuple[ttnn.Tensor, int]:
        c = int(self.cfg.num_channels)
        t = int(self.cfg.temporal_patch_size)
        p = int(self.cfg.patch_size)

        shape = [int(s) for s in pixel_values.shape]
        elems_per_patch = c * t * p * p
        total = 1
        for s in shape:
            total *= s
        if total % elems_per_patch != 0:
            raise ValueError(f"pixel_values has incompatible shape {shape} for C={c}, T={t}, P={p}")

        n = total // elems_per_patch
        x5 = ttnn.reshape(pixel_values, (n, c, t, p, p))
        x5_t0 = ttnn.slice(x5, (0, 0, 0, 0, 0), (n, c, 1, p, p), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Do not deallocate ``x5`` before consuming ``x5_t0``: slice output may alias parent storage.
        hs = ttnn.reshape(x5_t0, (n, c * p * p))
        ttnn.deallocate(x5_t0)
        ttnn.deallocate(x5)
        return hs, n

    def _patch_pixels_to_ttnn(self, pixel_values: torch.Tensor) -> ttnn.Tensor:
        hs, _n = self._pixels_flat_torch(pixel_values)
        pixel_tt = ttnn.from_torch(
            hs.unsqueeze(0).unsqueeze(0).bfloat16(),
            dtype=ttnn.bfloat16,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        pixel_tt = ttnn.to_memory_config(pixel_tt, ttnn.DRAM_MEMORY_CONFIG)
        return self.patch_embed(pixel_tt, grid_thw=None)

    @torch.inference_mode()
    def _patchify(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """
        Host patch embed for staged PCC / callers that expect torch ``[N, embed_dim]``.
        Same math as ``_patch_pixels_to_ttnn`` + ``to_torch`` slice; ``grid_thw`` is unused
        (reference patchifier ignores it).
        """
        del grid_thw
        out_tt = self._patch_pixels_to_ttnn(hidden_states)
        n = int(out_tt.shape[2])
        o = ttnn.to_torch(out_tt)[0, 0, :n, : self.cfg.embed_dim].to(hidden_states.dtype)
        ttnn.deallocate(out_tt)
        return o

    def _prepare_ttnn(self, emb: torch.Tensor, mesh_device: Any) -> Tuple[ttnn.Tensor, int, int]:
        """Pad sequence to ``_w128`` and upload for staged PCC (``[1,1,S_pad,D]`` TILE)."""
        s, d = int(emb.shape[0]), int(emb.shape[1])
        s_pad = _w128(s)
        if s_pad > s:
            p = emb.new_zeros(s_pad, d)
            p[:s] = emb
            tile = p
        else:
            tile = emb
        tt = ttnn.from_torch(
            tile.unsqueeze(0).unsqueeze(0).bfloat16(),
            dtype=ttnn.bfloat16,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tt = ttnn.to_memory_config(tt, ttnn.DRAM_MEMORY_CONFIG)
        return tt, s, s_pad

    def _cu_seqlens_ttnn_from_grid(self, grid_thw: torch.Tensor) -> ttnn.Tensor:
        g = grid_thw.unsqueeze(0) if grid_thw.dim() == 1 else grid_thw
        cu = torch.repeat_interleave(g[:, 1] * g[:, 2], g[:, 0], dim=0).cumsum(0, dtype=torch.int32)
        # Prepends a leading 0 (same as ``F.pad(cu, (1, 0), value=0)`` on the host): pad last dim before.
        cu_tt = ttnn.from_torch(
            cu.reshape(1, 1, 1, -1).to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        out = ttnn.pad(cu_tt, padding=((0, 0), (0, 0), (0, 0), (1, 0)), value=0)
        ttnn.deallocate(cu_tt)
        return out

    def _get_pos_ids_by_grid_ttnn(self, grid_thw_list: list[tuple[int, int, int]]) -> tuple[list[int], list[int]]:
        h_all: list[int] = []
        w_all: list[int] = []
        sm = self.cfg.spatial_merge_size
        for t, h, w in grid_thw_list:
            h_block: list[int] = []
            w_block: list[int] = []
            for hb in range(0, h, sm):
                for wb in range(0, w, sm):
                    for hi in range(sm):
                        for wi in range(sm):
                            h_block.append(hb + hi)
                            w_block.append(wb + wi)
            for _ in range(t):
                h_all.extend(h_block)
                w_all.extend(w_block)
        return h_all, w_all

    def _rot_pos_ttnn(self, grid_thw_list: list[tuple[int, int, int]]) -> ttnn.Tensor:
        h_ids, w_ids = self._get_pos_ids_by_grid_ttnn(grid_thw_list)
        max_grid_size = max(max(h, w) for _, h, w in grid_thw_list)
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        # ttnn.embedding currently requires BF16 weight tensors.
        rotary_pos_emb_full = ttnn.typecast(rotary_pos_emb_full, ttnn.bfloat16)
        mesh_mapper = ttnn.ReplicateTensorToMesh(self.mesh)
        h_row = torch.tensor(h_ids, dtype=torch.uint32).reshape(1, -1)
        w_row = torch.tensor(w_ids, dtype=torch.uint32).reshape(1, -1)
        h_idx_tt = ttnn.from_torch(
            h_row,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        w_idx_tt = ttnn.from_torch(
            w_row,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )
        h_freqs = ttnn.embedding(h_idx_tt, rotary_pos_emb_full, layout=ttnn.TILE_LAYOUT)
        w_freqs = ttnn.embedding(w_idx_tt, rotary_pos_emb_full, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(h_idx_tt)
        ttnn.deallocate(w_idx_tt)
        ttnn.deallocate(rotary_pos_emb_full)
        token_count = len(h_ids)
        h_freqs = ttnn.reshape(h_freqs, (token_count, h_freqs.shape[-1]))
        w_freqs = ttnn.reshape(w_freqs, (token_count, w_freqs.shape[-1]))
        rotary = ttnn.concat([h_freqs, w_freqs], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h_freqs)
        ttnn.deallocate(w_freqs)
        d = rotary.shape[-1]
        return ttnn.reshape(rotary, (1, 1, token_count, d))

    def forward(
        self,
        pixel_values: ttnn.Tensor,
        grid_thw: ttnn.Tensor,
        return_host_torch: bool = True,
    ) -> ttnn.Tensor:
        if not isinstance(pixel_values, ttnn.Tensor):
            raise TypeError(
                f"DotsVisionTransformerTT.forward expects pixel_values as ttnn.Tensor, got {type(pixel_values)}"
            )
        if not isinstance(grid_thw, ttnn.Tensor):
            raise TypeError(f"DotsVisionTransformerTT.forward expects grid_thw as ttnn.Tensor, got {type(grid_thw)}")

        hs_tt, n = self._pixels_flat_ttnn(pixel_values)
        pixel_tt = ttnn.reshape(
            hs_tt,
            (1, 1, n, self.cfg.num_channels * self.cfg.patch_size * self.cfg.patch_size),
        )
        ttnn.deallocate(hs_tt)
        if pixel_tt.dtype != ttnn.bfloat16:
            pixel_tt = ttnn.typecast(pixel_tt, ttnn.bfloat16)
        pixel_tt = ttnn.to_layout(pixel_tt, ttnn.TILE_LAYOUT)
        pixel_tt = ttnn.to_memory_config(pixel_tt, ttnn.DRAM_MEMORY_CONFIG)

        grid_thw_torch = ttnn.to_torch(grid_thw)

        if grid_thw_torch.dim() == 1:
            grid_thw_torch = grid_thw_torch.unsqueeze(0)
        t = int(grid_thw_torch.shape[0])
        hidden_tt = self.patch_embed(pixel_tt, grid_thw=None)
        seqlen = int(hidden_tt.shape[2])
        s_pad = _w128(seqlen)
        hidden_tt = _pad_seq_dim_ttnn(hidden_tt, seqlen, s_pad)
        grid_thw_list = [tuple(int(v) for v in row) for row in grid_thw_torch.tolist()]
        rotary_tt = self._rot_pos_ttnn(grid_thw_list)
        cu_tt = self._cu_seqlens_ttnn_from_grid(grid_thw_torch)
        x = hidden_tt
        for blk in self.blocks:
            x = blk(x, rotary_tt, cu_tt, seqlen)
        ttnn.deallocate(rotary_tt)
        ttnn.deallocate(cu_tt)
        if self.post_norm is not None:
            x = self.post_norm(x, mode="prefill")
        x = self.merger(x, seqlen)
        if not return_host_torch:
            return x
        s_merge = seqlen // (self.cfg.spatial_merge_size**2)
        o_full = ttnn.to_torch(x)
        if o_full.dim() == 5:
            o = o_full[:, 0, 0, :s_merge, : self.cfg.hidden_size]
        elif o_full.dim() == 4:
            o = o_full[:, 0, :s_merge, : self.cfg.hidden_size]
        elif o_full.dim() == 3:
            o = o_full[:, :s_merge, : self.cfg.hidden_size]
        else:
            raise RuntimeError(f"Unexpected merged tensor rank {o_full.dim()} with shape {tuple(o_full.shape)}")
        out_host = o.squeeze(0) if t == 1 else o.reshape(-1, self.cfg.hidden_size)
        ttnn.deallocate(x)
        out_tt = ttnn.from_torch(
            out_host.contiguous().bfloat16(),
            dtype=ttnn.bfloat16,
            device=self.mesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        return ttnn.to_memory_config(out_tt, ttnn.DRAM_MEMORY_CONFIG)
