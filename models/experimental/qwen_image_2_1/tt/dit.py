# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen-Image-2.1 single-stream DiT on one Blackhole with TTNN ops.

Two entry points mirror the reference's prefix-KV-cache scheme:
  * prefix_kv(text_embeds)  - runs the (t = 0 modulated) text tokens through the 32 blocks with causal
                              attention and returns per-layer post-RoPE K/V (the "extract" step).
  * step(latents, cond, kv) - runs the 4096 image tokens attending to [image; cached text] and returns the
                              velocity prediction (the "cached" step). All tokens of the step share one
                              modulation row, so adaLN folds into layer_norm's gamma and the gated residual
                              folds into the fused matmul+addcmul op.
"""
from __future__ import annotations

import math
import os as _os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

import ttnn
from models.tt_dit.utils.matmul import get_matmul_config

from ..common import rope as rope_mod
from ..common.config import DIT, DiTConfig
from ..common.schedule import StepConditioning, TimeConditioning
from ..common.weights import LazyCheckpoint, fuse_qkv, linear_to_mm, swiglu_interleave

TILE = 32
_FID = {"lofi": ttnn.MathFidelity.LoFi, "hifi2": ttnn.MathFidelity.HiFi2, "hifi4": ttnn.MathFidelity.HiFi4}
SDPA_SCALE_FOLD = 1.0 / math.sqrt(DIT.head_dim)  # folded into norm_q at load; attention runs with scale=1.0
# QWEN_SDPA_FOLD=1 folds 1/sqrt(head_dim) into norm_q so attention runs with scale=1.0 and the SDPA op skips its
# hidden mask multiply on the masked editing path (-12/-18 ms/step). Off by default: step-level accuracy is identical
# (LoFi step PCC 0.99934 vs 0.99927 unfolded) but the chaotic two-image edit lands elsewhere (0.971 vs 0.980), and the
# mask-free editing path (long_prefix) has no mask to multiply, so the fold buys nothing there.
FOLD_SDPA_SCALE = _os.environ.get("QWEN_SDPA_FOLD", "0") == "1"
PREFIX_QCHUNK = int(_os.environ.get("QWEN_PREFIX_QCHUNK", "256"))  # SDPA q chunk of the prefix pass
PREFIX_CHUNK = int(
    _os.environ.get("QWEN_PREFIX_KCHUNK", "256")
)  # SDPA k chunk; the prefix is padded to a multiple of it


def _pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    if t.shape[-2] == rows:
        return t
    out = torch.zeros(*t.shape[:-2], rows, t.shape[-1], dtype=t.dtype)
    out[..., : t.shape[-2], :] = t
    return out


@dataclass
class DiTPrecision:
    """Dtype / fidelity policy."""

    weight_dtype: ttnn.DataType = ttnn.bfloat16  # matmul weights (bf16 or bfloat8_b)
    act_dtype: ttnn.DataType = ttnn.bfloat16
    mm_fidelity: ttnn.MathFidelity = (
        ttnn.MathFidelity.LoFi
    )  # 40-step latents PCC 0.995 (HiFi2: 0.997) at -19 % step time
    sdpa_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi2  # LoFi SDPA drops PCC to 0.98-0.995 with no speed gain
    norm_fp32_acc: bool = True  # layer_norm with fp32 accumulation off is 1.8x faster; accuracy checked by tests
    sdpa_q_chunk: int = 512  # tests/sweep_sdpa.py: q512/k256 3.19 ms vs q256/k512 4.11 ms per layer
    sdpa_k_chunk: int = 256
    sdpa_mask_q_chunk: int = 256  # masked (long-prefix) path: the mask chunk CB must also fit L1
    sdpa_mask_k_chunk: int = 256
    prefix_sdpa_fidelity: ttnn.MathFidelity = (
        ttnn.MathFidelity.HiFi2
    )  # attention fidelity of the once-per-prompt prefix pass
    prefix_mm_fidelity: ttnn.MathFidelity = _FID[
        _os.environ.get("QWEN_PREFIX_MM_FID", "hifi2")
    ]  # matmul fidelity of the prefix pass (once per prompt; the step keeps mm_fidelity)
    edit_mm_fidelity: ttnn.MathFidelity = _FID[
        _os.environ.get("QWEN_EDIT_MM_FID", "hifi2")
    ]  # matmul fidelity of the step when condition images are present (tests/diag_step_ops.py: LoFi matmuls
    # carry 14-30x the bf16 rounding error and the two-image edit amplifies it; HiFi2 is 2x, HiFi4 1x)
    ln_fidelity: ttnn.MathFidelity = _FID[
        _os.environ.get("QWEN_LN_FID", "hifi4")
    ]  # layer_norm: HiFi2 leaves 4x the bf16 rounding error, HiFi4 1.2x (diag_step_ops.py)
    t2i_sdpa: str = _os.environ.get("QWEN_T2I_SDPA", "plain")  # "plain": streaming bf16-acc SDPA over the concatenated
    # [image ; text] K/V at the logical length; "joint": the joint SDPA op with fp32 accumulation
    fused_qknorm: bool = _os.environ.get("QWEN_FUSED_QKNORM", "1") == "1"  # split QKV matmul +
    # dit_fused_distributed_rmsnorm(per_head_norm, rope) for q and k instead of head split + rms_norm + rope


class TTLayer:
    """Device weights of one transformer block."""

    def __init__(self, dev, ckpt: LazyCheckpoint, idx: int, prec: DiTPrecision, cfg: DiTConfig = DIT):
        p = f"transformer_blocks.{idx}."
        g = lambda k: ckpt.get(p + k, torch.bfloat16)
        wd = prec.weight_dtype
        mem = ttnn.DRAM_MEMORY_CONFIG
        self.wqkv = ttnn.from_torch(
            fuse_qkv(g("attn.to_q.weight"), g("attn.to_k.weight"), g("attn.to_v.weight")),
            dtype=wd,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        # optionally (FOLD_SDPA_SCALE) 1/sqrt(head_dim) is folded into norm_q and attention runs with scale=1.0
        self.norm_q = ttnn.from_torch(
            (g("attn.norm_q.weight").float() * (SDPA_SCALE_FOLD if FOLD_SDPA_SCALE else 1.0))
            .to(torch.bfloat16)
            .reshape(1, 1, 1, cfg.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        self.norm_k = ttnn.from_torch(
            g("attn.norm_k.weight").reshape(1, 1, 1, cfg.head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        # the fused per-head RMSNorm+RoPE op takes the weight repeated for every head
        self.norm_q_fused = ttnn.from_torch(
            (g("attn.norm_q.weight").float() * (SDPA_SCALE_FOLD if FOLD_SDPA_SCALE else 1.0))
            .to(torch.bfloat16)
            .repeat(cfg.heads)
            .reshape(1, 1, 1, cfg.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        self.norm_k_fused = ttnn.from_torch(
            g("attn.norm_k.weight").repeat(cfg.heads).reshape(1, 1, 1, cfg.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        self.wo = ttnn.from_torch(
            linear_to_mm(g("attn.to_out.0.weight")), dtype=wd, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem
        )
        self.w_gateup = ttnn.from_torch(
            swiglu_interleave(g("img_mlp.gate_layer.weight"), g("img_mlp.proj.weight")),
            dtype=wd,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        self.w_down = ttnn.from_torch(
            linear_to_mm(g("img_mlp.out.weight")), dtype=wd, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem
        )


@dataclass
class DeviceCond:
    """One step's modulation rows on device (logical [1, 1, 1, 4096] each, bf16 TILE)."""

    one_plus_scale1: ttnn.Tensor
    tanh_gate1: ttnn.Tensor
    one_plus_scale2: ttnn.Tensor
    tanh_gate2: ttnn.Tensor
    one_plus_scale_out: ttnn.Tensor

    @classmethod
    def from_host(cls, dev, sc: StepConditioning) -> "DeviceCond":
        f = lambda t: ttnn.from_torch(
            t.reshape(1, 1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return cls(
            f(sc.one_plus_scale1), f(sc.tanh_gate1), f(sc.one_plus_scale2), f(sc.tanh_gate2), f(sc.one_plus_scale_out)
        )

    def update(self, sc: StepConditioning):
        """Overwrite in place (keeps device addresses stable for trace replay)."""
        for name in ("one_plus_scale1", "tanh_gate1", "one_plus_scale2", "tanh_gate2", "one_plus_scale_out"):
            host = ttnn.from_torch(
                getattr(sc, name).reshape(1, 1, 1, -1).to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(host, getattr(self, name))


@dataclass
class RopeTables:
    cos: ttnn.Tensor  # [1, 1, S, 128] bf16 TILE
    sin: ttnn.Tensor
    trans_mat: ttnn.Tensor  # [1, 1, 32, 32]


class QwenImageDiT:
    def __init__(
        self,
        dev,
        ckpt: LazyCheckpoint,
        prec: Optional[DiTPrecision] = None,
        cfg: DiTConfig = DIT,
        layers: Optional[int] = None,
    ):
        self.dev = dev
        self.cfg = cfg
        self.prec = prec or DiTPrecision()
        self.grid = dev.compute_with_storage_grid_size()
        mem = ttnn.DRAM_MEMORY_CONFIG
        wd = self.prec.weight_dtype
        g = lambda k: ckpt.get(k, torch.bfloat16)
        layer_count = cfg.num_layers if layers is None else layers
        self.layers: List[TTLayer] = [TTLayer(dev, ckpt, i, self.prec, cfg) for i in range(layer_count)]
        self.img_in = ttnn.from_torch(
            linear_to_mm(g("img_in.weight")),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        self.proj_out = ttnn.from_torch(
            linear_to_mm(g("proj_out.weight")),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        # text projection (zero-centered RMSNorm: effective scale = w + 1)
        self.txt_norm_w = ttnn.from_torch(
            (g("txt_in.text_norm.weight").float() + 1.0).to(torch.bfloat16).reshape(1, 1, 1, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
        )
        self.txt_in_w = ttnn.from_torch(
            linear_to_mm(g("txt_in.in_layer.weight")), dtype=wd, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem
        )
        self.txt_out_w = ttnn.from_torch(
            linear_to_mm(g("txt_in.out_layer.weight")), dtype=wd, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem
        )
        self.time_cond = TimeConditioning.from_ckpt(ckpt)
        self.trans_mat = ttnn.from_torch(
            rope_mod.rot_transformation_mat(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=mem,
        )

        # attention scale: 1/sqrt(head_dim), or 1.0 when FOLD_SDPA_SCALE folded it into norm_q (TTLayer)
        self.sdpa_scale = 1.0 if FOLD_SDPA_SCALE else SDPA_SCALE_FOLD
        arch = dev.arch()
        self.ck_mm = ttnn.init_device_compute_kernel_config(
            arch, math_fidelity=self.prec.mm_fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        self.ck_norm = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.ck_mm_edit = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=self.prec.edit_mm_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.ck_ln = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=self.prec.ln_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=self.prec.norm_fp32_acc,
            packer_l1_acc=False,
        )
        self.ck_sdpa = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=self.prec.sdpa_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # plain SDPA without fp32 accumulation selects the kernel's streaming path (bf16 intermediates, dst 8 tiles)
        self.ck_sdpa_stream = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=self.prec.sdpa_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        self.ck_rope = ttnn.init_device_compute_kernel_config(
            arch,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self._mm_cfg_cache: Dict[Tuple[int, int, int, str], object] = {}
        self._tuned = None

    # ------------------------------------------------------------------ helpers
    def _tuned_table(self):
        """Load the published matmul blocking table for this grid and precision, if present."""
        if self._tuned is None:
            import json
            import os

            self._tuned = {}
            dt = "bf16" if self.prec.weight_dtype == ttnn.bfloat16 else "bfp8"
            fid = {
                ttnn.MathFidelity.HiFi2: "hifi2",
                ttnn.MathFidelity.LoFi: "lofi",
                ttnn.MathFidelity.HiFi4: "hifi4",
            }[self.prec.mm_fidelity]
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # exact (dtype, fidelity) table first, else the hifi2 table of the same dtype (blockings transfer)
            candidates = [
                f"matmul_configs_{self.grid.x}x{self.grid.y}_{dt}_{fid}.json",
                f"matmul_configs_{self.grid.x}x{self.grid.y}_{dt}_hifi2.json",
            ]
            found = None
            for name in candidates:
                path = os.path.join(here, "configs", name)
                if os.path.exists(path):
                    found = path
                    break
            if found:
                with open(found) as source:
                    for r in json.load(source).values():
                        self._tuned[(r["M"], r["K"], r["N"], r["variant"])] = r
        return self._tuned

    def _mm_cfg(self, M: int, K: int, N: int, variant: str = "plain"):
        key = (M, K, N, variant)
        if key not in self._mm_cfg_cache:
            r = self._tuned_table().get((M, K, N, variant))
            if r is not None:
                self._mm_cfg_cache[key] = ttnn.MinimalMatmulConfig(
                    M_block_size=r["M_block_size"],
                    K_block_size=r["K_block_size"],
                    N_block_size=r["N_block_size"],
                    subblock_h=r["subblock_h"],
                    subblock_w=r["subblock_w"],
                    compute_with_storage_grid_size=self.grid,
                )
            else:
                self._mm_cfg_cache[key] = get_matmul_config(M, K, N, self.grid)
        return self._mm_cfg_cache[key]

    def _cast_in(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """minimal_matmul needs matching dtypes: cast a bf16 activation to the weight dtype (no-op for bf16)."""
        if x.dtype == self.prec.weight_dtype:
            return x
        y = ttnn.typecast(x, self.prec.weight_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x)
        return y

    def _mm(self, x: ttnn.Tensor, w: ttnn.Tensor, M: int, K: int, N: int, fuse_swiglu: bool = False, dtype=None):
        x = self._cast_in(x)
        return ttnn.experimental.minimal_matmul(
            x,
            w,
            config=self._mm_cfg(M, K, N, "swiglu" if fuse_swiglu else "plain"),
            compute_kernel_config=self.ck_mm,
            dtype=dtype or self.prec.act_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            fuse_swiglu=fuse_swiglu,
        )

    def _mm_gated_residual(
        self, x: ttnn.Tensor, w: ttnn.Tensor, residual: ttnn.Tensor, gate_row: ttnn.Tensor, M: int, K: int, N: int
    ):
        """residual + tanh(gate) * (x @ w), computed inside the matmul kernel."""
        x = self._cast_in(x)
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            x,
            w,
            1.0,
            residual,
            gate_row,
            config=self._mm_cfg(M, K, N, "addcmul"),
            compute_kernel_config=self.ck_mm,
            dtype=self.prec.act_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _ln_mod(self, x: ttnn.Tensor, one_plus_scale: ttnn.Tensor) -> ttnn.Tensor:
        """LayerNorm(no affine) * (1 + scale) == layer_norm with gamma = 1 + scale."""
        return ttnn.layer_norm(
            x,
            epsilon=self.cfg.eps,
            weight=one_plus_scale,
            compute_kernel_config=self.ck_ln,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # Standard SDPA uses the port's measured approximate exponential policy. The streaming path
    # retains accurate mode; its separate kernel already honored False in the original runtime.
    def _sdpa_cfg(self, q_chunk: int, k_chunk: int, *, exp_approx_mode: bool):
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.grid,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=exp_approx_mode,
        )

    def rope_tables(self, cos: torch.Tensor, sin: torch.Tensor) -> RopeTables:
        """cos/sin [S, 128] bf16 -> device tables (S padded to a tile multiple)."""
        S = cos.shape[0]
        Sp = (S + TILE - 1) // TILE * TILE
        f = lambda t: ttnn.from_torch(
            _pad_rows(t, Sp).reshape(1, 1, Sp, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return RopeTables(f(cos), f(sin), self.trans_mat)

    def _qkv(self, x: ttnn.Tensor, layer: TTLayer, S: int, rt: RopeTables):
        """x [1,1,S,4096] -> q,k,v [1,32,S,128] with per-head RMSNorm on q/k and RoPE."""
        c = self.cfg
        if self.prec.fused_qknorm and rt.cos.shape[-2] == S and S >= 256:
            return self._qkv_fused(x, layer, S, rt)
        fused = self._mm(x, layer.wqkv, S, c.hidden, 3 * c.hidden)
        if len(fused.shape) != 4:
            fused = ttnn.reshape(fused, [1, 1, S, 3 * c.hidden])
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            fused,
            num_heads=c.heads,
            num_kv_heads=c.heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(fused)
        qn = ttnn.rms_norm(
            q,
            epsilon=c.eps,
            weight=layer.norm_q,
            compute_kernel_config=self.ck_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        kn = ttnn.rms_norm(
            k,
            epsilon=c.eps,
            weight=layer.norm_k,
            compute_kernel_config=self.ck_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(k)
        qr = ttnn.experimental.rotary_embedding_llama(
            qn, rt.cos, rt.sin, rt.trans_mat, is_decode_mode=False, compute_kernel_config=self.ck_rope
        )
        ttnn.deallocate(qn)
        kr = ttnn.experimental.rotary_embedding_llama(
            kn, rt.cos, rt.sin, rt.trans_mat, is_decode_mode=False, compute_kernel_config=self.ck_rope
        )
        ttnn.deallocate(kn)
        return qr, kr, v

    def _qkv_fused(self, x: ttnn.Tensor, layer: TTLayer, S: int, rt: RopeTables):
        """Split QKV matmul (3 x [1,1,S,4096]) -> fused per-head RMSNorm+RoPE for q and k, head split for v.
        Replaces head split + 2 rms_norm + 2 rope (5 launches) with 2 fused launches + 1 split."""
        c = self.cfg
        x = self._cast_in(x)
        qc, kc, vc = ttnn.experimental.minimal_matmul_split(
            x,
            layer.wqkv,
            chunks=3,
            dim=-1,
            config=self._mm_cfg(S, c.hidden, 3 * c.hidden, "plain"),
            compute_kernel_config=self.ck_mm,
            dtype=self.prec.act_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if len(qc.shape) != 4:  # minimal_matmul(_split) may hand back rank-3 chunks
            qc, kc, vc = (ttnn.reshape(t, [1, 1, S, c.hidden]) for t in (qc, kc, vc))
        outs = []
        for chunk, w in ((qc, layer.norm_q_fused), (kc, layer.norm_k_fused)):
            outs.append(
                ttnn.experimental.dit_fused_distributed_rmsnorm(
                    chunk,
                    0,
                    self.dev,
                    [],
                    epsilon=c.eps,
                    num_heads_per_device=c.heads,
                    per_head_norm=True,
                    weight=w,
                    transformation_mat=rt.trans_mat,
                    rope_cos=rt.cos,
                    rope_sin=rt.sin,
                    compute_kernel_config=self.ck_norm,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )
            ttnn.deallocate(chunk)
        v, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            vc, num_heads=c.heads, num_kv_heads=0, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(vc)
        return outs[0], outs[1], v

    # ------------------------------------------------------------------ text prefix
    def text_project(self, text_embeds: torch.Tensor) -> Tuple[ttnn.Tensor, int]:
        """txt_in(): [T, 4096] encoder hidden states -> device [1,1,Tpad,4096] bf16; returns (tensor, T)."""
        T = text_embeds.shape[0]
        Tp = (T + TILE - 1) // TILE * TILE
        x = ttnn.from_torch(
            _pad_rows(text_embeds.to(torch.bfloat16), Tp).reshape(1, 1, Tp, -1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        h = ttnn.rms_norm(
            x,
            epsilon=self.cfg.eps,
            weight=self.txt_norm_w,
            compute_kernel_config=self.ck_norm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        h1 = ttnn.linear(
            h,
            self.txt_in_w,
            compute_kernel_config=self.ck_mm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(h)
        h1a = ttnn.gelu(h1, fast_and_approximate_mode=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(h1)
        out = ttnn.linear(
            h1a,
            self.txt_out_w,
            compute_kernel_config=self.ck_mm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(h1a)
        return out, T

    def prefix_kv(self, text_embeds: torch.Tensor, rt_text: RopeTables, cond0: DeviceCond, return_hidden: bool = False):
        """Run the text tokens (modulated from t = 0) through all blocks with causal attention.
        Returns list of (k, v) per layer, each [1, 32, T, 128] (logical T, tile padded), and T."""
        h, T = self.text_project(text_embeds)
        Tp = h.shape[-2]
        c = self.cfg
        kv: List[Tuple[ttnn.Tensor, ttnn.Tensor]] = []
        for layer in self.layers:
            x = self._ln_mod(h, cond0.one_plus_scale1)
            q, k, v = self._qkv(x, layer, Tp, rt_text)
            ttnn.deallocate(x)
            # joint path: keep the real T rows of K/V (logical T, tile padded; the joint op masks by logical length)
            # plain path: keep all Tp rows (logical == padded) so the step's [image ; text] concat stays tile aligned;
            # the step slices the concatenation to the logical S + T rows before the SDPA
            if self.prec.t2i_sdpa == "plain":
                kv.append((k, v))  # all Tp rows (logical == padded); the step slices [image ; text] to S + T
            else:
                k_keep = (
                    ttnn.slice(k, [0, 0, 0, 0], [1, c.heads, T, c.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    if T != Tp
                    else k
                )
                v_keep = (
                    ttnn.slice(v, [0, 0, 0, 0], [1, c.heads, T, c.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    if T != Tp
                    else v
                )
                kv.append((k_keep, v_keep))
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=True,
                scale=self.sdpa_scale,
                program_config=self._sdpa_cfg(TILE, TILE, exp_approx_mode=True),
                compute_kernel_config=self.ck_sdpa,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(q)
            if T != Tp and self.prec.t2i_sdpa != "plain":  # on the plain path k, v are the kept K/V
                ttnn.deallocate(k)
                ttnn.deallocate(v)
            a = ttnn.transformer.concatenate_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn)
            h_new = self._mm_gated_residual(a, layer.wo, h, cond0.tanh_gate1, Tp, c.hidden, c.hidden)
            ttnn.deallocate(a)
            ttnn.deallocate(h)
            h = h_new
            x2 = self._ln_mod(h, cond0.one_plus_scale2)
            m = self._mm(
                x2, layer.w_gateup, Tp, c.hidden, 2 * c.mlp_hidden, fuse_swiglu=True, dtype=self.prec.weight_dtype
            )
            ttnn.deallocate(x2)
            h_new = self._mm_gated_residual(m, layer.w_down, h, cond0.tanh_gate2, Tp, c.mlp_hidden, c.hidden)
            ttnn.deallocate(m)
            ttnn.deallocate(h)
            h = h_new
        if return_hidden:
            return kv, T, h
        ttnn.deallocate(h)
        return kv, T

    # ------------------------------------------------------------------ general prefix (text + condition images)
    @staticmethod
    def block_causal_mask(segments, padded_len: int) -> torch.Tensor:
        """Additive attention bias [1, 1, P_pad, P_pad] (0 = attend, -1e9 = masked) for a prefix made of
        ("text", n) and ("image", h, w) segments: allowed(q, k) = (k <= q) or same image block; keys beyond the
        logical length are masked. Mirrors diffusers' block-causal mask for the condition part."""
        ids = []
        blk = 0
        for seg in segments:
            if seg[0] == "text":
                ids.extend([-1] * seg[1])
            else:
                ids.extend([blk] * (seg[1] * seg[2]))
                blk += 1
        P = len(ids)
        ids_t = torch.tensor(ids + [-2] * (padded_len - P))
        q = torch.arange(padded_len)[:, None]
        k = torch.arange(padded_len)[None, :]
        same_block = (ids_t[:, None] == ids_t[None, :]) & (ids_t[:, None] >= 0)
        allowed = ((k <= q) | same_block) & (k < P)
        bias = torch.where(allowed, 0.0, -1e9).to(torch.bfloat16)
        return bias.reshape(1, 1, padded_len, padded_len)

    def prefix_embed(self, segments, text_rows: torch.Tensor, cond_latents: torch.Tensor) -> Tuple[ttnn.Tensor, int]:
        """Prefix hidden states [1, 1, P_pad, 4096]: txt_in() of the text rows and img_in() of the packed
        condition latents, interleaved in segment order. text_rows: [n_text_total, 4096] (encoder hidden states
        of the text positions only, in order); cond_latents: [sum(h*w), 64] normalized latents in order."""
        c = self.cfg
        n_text = sum(seg[1] for seg in segments if seg[0] == "text")
        n_img = sum(seg[1] * seg[2] for seg in segments if seg[0] == "image")
        assert text_rows.shape[0] == n_text, (text_rows.shape, n_text)
        assert cond_latents.shape[0] == n_img, (cond_latents.shape, n_img)
        txt_host = None
        if n_text:
            txt_dev, _ = self.text_project(text_rows)
            txt_host = ttnn.to_torch(txt_dev).reshape(-1, c.hidden)[:n_text]
            ttnn.deallocate(txt_dev)
        img_host = None
        if n_img:
            lat = ttnn.from_torch(
                _pad_rows(cond_latents.to(torch.bfloat16), (n_img + TILE - 1) // TILE * TILE).reshape(
                    1, 1, -1, c.in_channels
                ),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.dev,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            emb = self.embed_latents(lat, lat.shape[-2])
            img_host = ttnn.to_torch(emb).reshape(-1, c.hidden)[:n_img]
            ttnn.deallocate(lat)
            ttnn.deallocate(emb)
        rows = []
        ti = ii = 0
        for seg in segments:
            if seg[0] == "text":
                rows.append(txt_host[ti : ti + seg[1]])
                ti += seg[1]
            else:
                n = seg[1] * seg[2]
                rows.append(img_host[ii : ii + n])
                ii += n
        h_host = torch.cat(rows, 0)
        P = h_host.shape[0]
        # multiple of PREFIX_CHUNK so the prefix SDPA runs 256-row chunks (32-row chunks lost accuracy on 8k prefixes)
        Pp = (P + PREFIX_CHUNK - 1) // PREFIX_CHUNK * PREFIX_CHUNK
        h = ttnn.from_torch(
            _pad_rows(h_host, Pp).reshape(1, 1, Pp, c.hidden),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return h, P

    def prefix_kv_segments(
        self, segments, text_rows: torch.Tensor, cond_latents: torch.Tensor, rt_prefix: RopeTables, cond0: DeviceCond
    ):
        """Prefix pass for an arbitrary [text | condition image | text ...] layout (modulated from t = 0), with a
        block-causal attention bias. Returns per-layer (k, v) of length P_pad (tile padded; mask the tail with
        step_key_mask) and the logical prefix length P."""
        c = self.cfg
        h, P = self.prefix_embed(segments, text_rows, cond_latents)
        Pp = h.shape[-2]
        bias = ttnn.from_torch(
            self.block_causal_mask(segments, Pp),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        chunk = PREFIX_CHUNK if Pp % PREFIX_CHUNK == 0 else 32
        qchunk = PREFIX_QCHUNK if Pp % PREFIX_QCHUNK == 0 else chunk
        # the prefix runs once per prompt: HiFi2 matmuls here even when the step uses LoFi
        ck_saved, ck_sdpa_saved = self.ck_mm, self.ck_sdpa
        self.ck_mm = ttnn.init_device_compute_kernel_config(
            self.dev.arch(),
            math_fidelity=self.prec.prefix_mm_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.ck_sdpa = ttnn.init_device_compute_kernel_config(
            self.dev.arch(),
            math_fidelity=self.prec.prefix_sdpa_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        kv: List[Tuple[ttnn.Tensor, ttnn.Tensor]] = []
        for layer in self.layers:
            x = self._ln_mod(h, cond0.one_plus_scale1)
            q, k, v = self._qkv(x, layer, Pp, rt_prefix)
            ttnn.deallocate(x)
            # keep the tile-PADDED K/V (logical length Pp): the step concatenates [image ; prefix] and masks the
            # padding columns with step_key_mask, and the plain SDPA needs mask width == K length exactly
            kv.append((k, v))
            attn = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=bias,
                is_causal=False,
                scale=self.sdpa_scale,
                program_config=self._sdpa_cfg(qchunk, chunk, exp_approx_mode=True),
                compute_kernel_config=self.ck_sdpa,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(q)
            a = ttnn.transformer.concatenate_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn)
            h_new = self._mm_gated_residual(a, layer.wo, h, cond0.tanh_gate1, Pp, c.hidden, c.hidden)
            ttnn.deallocate(a)
            ttnn.deallocate(h)
            h = h_new
            x2 = self._ln_mod(h, cond0.one_plus_scale2)
            m = self._mm(
                x2, layer.w_gateup, Pp, c.hidden, 2 * c.mlp_hidden, fuse_swiglu=True, dtype=self.prec.weight_dtype
            )
            ttnn.deallocate(x2)
            h_new = self._mm_gated_residual(m, layer.w_down, h, cond0.tanh_gate2, Pp, c.mlp_hidden, c.hidden)
            ttnn.deallocate(m)
            ttnn.deallocate(h)
            h = h_new
        ttnn.deallocate(h)
        ttnn.deallocate(bias)
        self.ck_mm, self.ck_sdpa = ck_saved, ck_sdpa_saved
        return kv, P

    @staticmethod
    def step_key_mask(S: int, P: int, P_pad: int) -> torch.Tensor:
        """Additive bias [1, 1, S, S + P_pad] masking the prefix's tile padding (long-prefix step path)."""
        bias = torch.zeros(1, 1, S, S + P_pad, dtype=torch.bfloat16)
        if P_pad > P:
            bias[..., S + P :] = -1e9
        return bias

    # ------------------------------------------------------------------ image step
    def embed_latents(self, latents_bf16: ttnn.Tensor, S: int) -> ttnn.Tensor:
        """img_in: [1,1,S,64] -> [1,1,S,4096]."""
        return ttnn.linear(
            latents_bf16,
            self.img_in,
            compute_kernel_config=self.ck_mm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )

    def step(
        self,
        latents_bf16: ttnn.Tensor,
        cond: DeviceCond,
        kv: List[Tuple[ttnn.Tensor, ttnn.Tensor]],
        rt_img: RopeTables,
        n_text: int,
        return_hidden_taps: Optional[List[int]] = None,
        key_mask: Optional[ttnn.Tensor] = None,
        long_prefix: bool = False,
    ):
        """Cached step: latents [1,1,S,64] bf16 on device -> velocity [1,1,S,64] bf16.
        Condition-image prefixes take one of two paths: long_prefix=True (default for editing) concatenates
        [image ; prefix] K/V at the LOGICAL prefix length and runs plain SDPA without a mask (the kernel masks the
        tile padding itself); key_mask (legacy) is the additive [1,1,S,S+P_pad] bias over tile-padded K/V."""
        c = self.cfg
        S = latents_bf16.shape[-2]
        ck_saved = self.ck_mm
        if key_mask is not None or long_prefix:
            self.ck_mm = self.ck_mm_edit  # editing step: higher matmul fidelity (see DiTPrecision.edit_mm_fidelity)
        try:
            return self._step_body(latents_bf16, cond, kv, rt_img, n_text, return_hidden_taps, key_mask, S, long_prefix)
        finally:
            self.ck_mm = ck_saved

    def _step_body(self, latents_bf16, cond, kv, rt_img, n_text, return_hidden_taps, key_mask, S, long_prefix=False):
        c = self.cfg
        h = self.embed_latents(latents_bf16, S)
        taps = {}
        # dummy joint queries (their outputs are discarded); reuse the cached K rows to get the right shape
        for li, layer in enumerate(self.layers):
            x = self._ln_mod(h, cond.one_plus_scale1)
            q, k, v = self._qkv(x, layer, S, rt_img)
            ttnn.deallocate(x)
            tk, tv = kv[li]
            if long_prefix:
                # condition images: K/V = [image ; prefix] at the logical prefix length P = n_text; the cached
                # prefix K/V hold round_up(P, 32) rows, so the aligned concat is followed by a slice to the logical
                # length only when P is not a tile multiple (the SDPA kernel masks its own tile padding)
                kf = ttnn.concat([k, tk], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                vf = ttnn.concat([v, tv], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                if tk.shape[-2] != n_text:
                    kl = ttnn.slice(
                        kf, [0, 0, 0, 0], [1, c.heads, S + n_text, c.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    vl = ttnn.slice(
                        vf, [0, 0, 0, 0], [1, c.heads, S + n_text, c.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    ttnn.deallocate(kf)
                    ttnn.deallocate(vf)
                    kf, vf = kl, vl
                attn = ttnn.transformer.scaled_dot_product_attention(
                    q,
                    kf,
                    vf,
                    is_causal=False,
                    scale=self.sdpa_scale,
                    program_config=self._sdpa_cfg(
                        self.prec.sdpa_mask_q_chunk, self.prec.sdpa_mask_k_chunk, exp_approx_mode=True
                    ),
                    compute_kernel_config=self.ck_sdpa,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(q)
                ttnn.deallocate(kf)
                ttnn.deallocate(vf)
            elif key_mask is None and self.prec.t2i_sdpa == "plain":
                # text prefix: [image ; text] K/V (tile-aligned concat of the Tp-row text K/V), sliced to the logical
                # S + T rows, then plain SDPA on the streaming (bf16-accumulate) kernel; -16 ms/step cold vs the joint op
                kf = ttnn.concat([k, tk], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                vf = ttnn.concat([v, tv], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                if tk.shape[-2] != n_text:
                    kl = ttnn.slice(
                        kf, [0, 0, 0, 0], [1, c.heads, S + n_text, c.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    vl = ttnn.slice(
                        vf, [0, 0, 0, 0], [1, c.heads, S + n_text, c.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
                    )
                    ttnn.deallocate(kf)
                    ttnn.deallocate(vf)
                    kf, vf = kl, vl
                attn = ttnn.transformer.scaled_dot_product_attention(
                    q,
                    kf,
                    vf,
                    is_causal=False,
                    scale=self.sdpa_scale,
                    program_config=self._sdpa_cfg(
                        self.prec.sdpa_q_chunk, self.prec.sdpa_k_chunk, exp_approx_mode=False
                    ),
                    compute_kernel_config=self.ck_sdpa_stream,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(q)
                ttnn.deallocate(kf)
                ttnn.deallocate(vf)
            elif key_mask is None:
                # short text prefix: joint SDPA attaches the cached text K/V (tile padding masked logically)
                attn, _joint = ttnn.transformer.joint_scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    tk,
                    tk,
                    tv,
                    joint_strategy="rear",
                    scale=1.0 if self.sdpa_scale == 1.0 else None,  # None = the op's default 1/sqrt(d)
                    program_config=self._sdpa_cfg(
                        self.prec.sdpa_q_chunk, self.prec.sdpa_k_chunk, exp_approx_mode=False
                    ),
                    compute_kernel_config=self.ck_sdpa,
                )
                ttnn.deallocate(_joint)
                ttnn.deallocate(q)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
            else:
                # long prefix (condition images): [image ; prefix] K/V + key mask; the joint op would spend a
                # dummy query row per prefix token (2x the attention work at P ~ 4k)
                kf = ttnn.concat([k, tk], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                vf = ttnn.concat([v, tv], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                attn = ttnn.transformer.scaled_dot_product_attention(
                    q,
                    kf,
                    vf,
                    attn_mask=key_mask,
                    is_causal=False,
                    scale=self.sdpa_scale,
                    program_config=self._sdpa_cfg(
                        self.prec.sdpa_mask_q_chunk, self.prec.sdpa_mask_k_chunk, exp_approx_mode=True
                    ),
                    compute_kernel_config=self.ck_sdpa,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(q)
                ttnn.deallocate(kf)
                ttnn.deallocate(vf)
            a = ttnn.transformer.concatenate_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn)
            h_new = self._mm_gated_residual(a, layer.wo, h, cond.tanh_gate1, S, c.hidden, c.hidden)
            ttnn.deallocate(a)
            ttnn.deallocate(h)
            h = h_new
            x2 = self._ln_mod(h, cond.one_plus_scale2)
            m = self._mm(
                x2, layer.w_gateup, S, c.hidden, 2 * c.mlp_hidden, fuse_swiglu=True, dtype=self.prec.weight_dtype
            )
            ttnn.deallocate(x2)
            h_new = self._mm_gated_residual(m, layer.w_down, h, cond.tanh_gate2, S, c.mlp_hidden, c.hidden)
            ttnn.deallocate(m)
            ttnn.deallocate(h)
            h = h_new
            if return_hidden_taps is not None and li in return_hidden_taps:
                taps[li] = ttnn.to_torch(h)
        xo = self._ln_mod(h, cond.one_plus_scale_out)
        ttnn.deallocate(h)
        out = ttnn.linear(
            xo,
            self.proj_out,
            compute_kernel_config=self.ck_mm,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(xo)
        if return_hidden_taps is not None:
            return out, taps
        return out
