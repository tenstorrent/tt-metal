# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any

import torch
import ttnn
from models.common.rmsnorm import RMSNorm
from models.experimental.voxtraltts.utils.config_helpers import COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC
from models.tt_transformers.tt.common import Mode


class VoxtralAcousticRMSNorm(RMSNorm):
    """``models.common.rmsnorm.RMSNorm`` with acoustic FM HiFi4 kernel (no common API changes)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.compute_kernel_config_hifi2 = COMPUTE_KERNEL_CONFIG_VOXTRAL_ACOUSTIC


class VoxtralTTRMSNorm:
    """Thin Voxtral adapter over shared TT RMSNorm implementation."""

    def __init__(
        self,
        device,
        dim: int,
        state_dict: dict[str, torch.Tensor],
        weight_key: str,
        eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path: Path | None = None,
    ) -> None:
        if weight_key in state_dict and f"{weight_key}.weight" not in state_dict:
            state_dict = {**state_dict, f"{weight_key}.weight": state_dict[weight_key]}

        self.inner = RMSNorm(
            device=device,
            dim=dim,
            state_dict=state_dict,
            weight_key=weight_key,
            eps=eps,
            weight_dtype=weight_dtype,
            weight_cache_path=weight_cache_path,
            is_distributed=False,
        )

    def __call__(self, x: ttnn.Tensor, mode: Mode | str = Mode.DECODE, **kwargs: Any) -> ttnn.Tensor:
        return self.inner(x, mode=mode, **kwargs)


class VoxtralTextRMSNorm:
    """FP32-promoting RMSNorm matching HF MistralRMSNorm exactly.

    HF's ``MistralRMSNorm`` promotes the input tensor to FP32 *before* computing the
    variance, then casts the normalised result back to bf16 and multiplies by the
    (bf16) weight. The shared ``models.common.rmsnorm.RMSNorm`` uses bf16 inputs with
    an fp32 *accumulator*, which means the per-element ``x*x`` multiplication loses
    bf16 precision before accumulation — a small but compounding error over the 32
    text-model layers.

    This class implements the HF behaviour explicitly with ttnn primitives:

    1. ``ttnn.typecast(x, fp32)`` — promote input
    2. ``ttnn.multiply(x_fp32, x_fp32)`` — squared values in fp32
    3. ``ttnn.mean(dim=-1, keepdim=True)`` — variance in fp32
    4. ``ttnn.rsqrt(variance + eps)`` — inverse-sqrt in fp32
    5. ``ttnn.multiply(x_fp32, rsqrt)`` — normalise in fp32 (broadcast over last dim)
    6. ``ttnn.typecast(result, bf16)`` — cast back
    7. ``ttnn.multiply(normalised, weight_bf16)`` — final weight multiply in bf16

    Matches HF MistralRMSNorm's forward exactly. Same interface as :class:`VoxtralTTRMSNorm`
    so it can be drop-in replaced into a tt_transformers-built text model.

    Note: each call adds ~5 extra ttnn ops vs the fused ``ttnn.rms_norm`` kernel, so this
    is slower per-call. Use only where the precision matters (typically the text-model
    layer norms feeding into attention / MLP).
    """

    def __init__(
        self,
        device,
        dim: int,
        state_dict: dict[str, torch.Tensor],
        weight_key: str,
        eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path: Path | None = None,
    ) -> None:
        # Resolve weight from state_dict using either ``weight_key`` or ``weight_key.weight``.
        if f"{weight_key}.weight" in state_dict:
            weight = state_dict[f"{weight_key}.weight"]
        elif weight_key in state_dict:
            weight = state_dict[weight_key]
        else:
            raise KeyError(f"VoxtralTextRMSNorm: weight not found at {weight_key!r} or {weight_key}.weight")

        weight_torch = weight.reshape(1, 1, 1, -1).to(torch.bfloat16).contiguous()
        self.weight_tt = ttnn.from_torch(
            weight_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.eps = float(eps)
        self.dim = int(dim)
        self.mesh_device = device

        # Fused-kernel path (opt-in, default ON). The manual __call__ below spends ~8 ttnn ops per
        # norm; with 53 norms/decode-token that is ~hundreds of tiny ops whose host-dispatch GAP
        # (~100us each) dominates decode wall-clock (and RTF) far more than their compute. The fused
        # ``ttnn.rms_norm`` does square+mean+rsqrt+normalize+weight in ONE kernel. We still promote the
        # input to fp32 first so the x*x square is fp32 (HF-faithful — the bf16 square is exactly what
        # this class was created to avoid), and use fp32_dest_acc for the variance accumulation.
        # Disable with VOXTRAL_TEXT_FUSED_RMSNORM=0 to fall back to the explicit decomposition.
        self.use_fused = os.environ.get("VOXTRAL_TEXT_FUSED_RMSNORM", "1") == "1"
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def __call__(self, x: ttnn.Tensor, mode: Mode | str = Mode.DECODE, **kwargs: Any) -> ttnn.Tensor:
        """Returns ``weight * normalize_fp32(x)`` with the bf16/fp32 cast pattern HF uses."""
        if isinstance(mode, str):
            mode = Mode(mode)

        norm_config = kwargs.get("norm_config")
        sharded_output_config = norm_config.get("sharded_output_config") if norm_config else None
        output_mem_config = norm_config.get("output_mem_config") if norm_config else None

        if self.use_fused:
            return self._forward_fused(x, mode, sharded_output_config, output_mem_config)

        # Match DistributedNorm: decode activations are width-sharded in L1.
        input_mem_cfg = (
            sharded_output_config if mode == Mode.DECODE and sharded_output_config else ttnn.DRAM_MEMORY_CONFIG
        )
        x = ttnn.to_memory_config(x, input_mem_cfg)
        in_sharded = mode == Mode.DECODE and sharded_output_config is not None
        if in_sharded:
            x = ttnn.sharded_to_interleaved(x)

        # Capture input dtype so we can restore it at the end (matches HF's behaviour).
        input_dtype = x.dtype
        dram_mem_cfg = ttnn.DRAM_MEMORY_CONFIG

        # Step 1: promote to fp32
        x_fp32 = ttnn.typecast(x, ttnn.float32, memory_config=dram_mem_cfg)

        # Step 2: x ** 2 in fp32
        x_sq = ttnn.multiply(x_fp32, x_fp32, dtype=ttnn.float32, memory_config=dram_mem_cfg)

        # Step 3: variance = mean(x_sq, dim=-1, keepdim=True)
        variance = ttnn.mean(x_sq, dim=-1, keepdim=True)
        if x_sq.is_allocated():
            ttnn.deallocate(x_sq)

        # Step 4: rsqrt(variance + eps)
        var_eps = ttnn.add(variance, self.eps, dtype=ttnn.float32, memory_config=dram_mem_cfg)
        if variance.is_allocated():
            ttnn.deallocate(variance)
        rsqrt = ttnn.rsqrt(var_eps, memory_config=dram_mem_cfg)
        if var_eps.is_allocated():
            ttnn.deallocate(var_eps)

        # Step 5: x_fp32 * rsqrt  (broadcast scalar-per-row over last dim)
        normalised_fp32 = ttnn.multiply(x_fp32, rsqrt, dtype=ttnn.float32, memory_config=dram_mem_cfg)
        if x_fp32.is_allocated():
            ttnn.deallocate(x_fp32)
        if rsqrt.is_allocated():
            ttnn.deallocate(rsqrt)

        # Step 6: cast back to input dtype (typically bf16)
        normalised_in = ttnn.typecast(normalised_fp32, input_dtype, memory_config=dram_mem_cfg)
        if normalised_fp32.is_allocated():
            ttnn.deallocate(normalised_fp32)

        # Step 7: weight * normalised (bf16 * bf16, broadcast over last dim)
        out = ttnn.multiply(
            normalised_in,
            self.weight_tt,
            dtype=ttnn.bfloat16,
            memory_config=dram_mem_cfg,
        )
        if normalised_in.is_allocated():
            ttnn.deallocate(normalised_in)

        if in_sharded:
            out = ttnn.to_memory_config(out, sharded_output_config)
        elif output_mem_config is not None:
            out = ttnn.to_memory_config(out, output_mem_config)
        return out

    def _forward_fused(self, x, mode, sharded_output_config, output_mem_config):
        """Single fused ``ttnn.rms_norm`` (fp32 square via fp32 input + fp32_dest_acc).

        Same numerics intent as the manual path (fp32 promote → variance → rsqrt → normalize →
        bf16 weight-multiply) but in one kernel instead of ~8 ops, killing the per-op dispatch gap.
        """
        input_mem_cfg = (
            sharded_output_config if mode == Mode.DECODE and sharded_output_config else ttnn.DRAM_MEMORY_CONFIG
        )
        x = ttnn.to_memory_config(x, input_mem_cfg)
        in_sharded = mode == Mode.DECODE and sharded_output_config is not None
        if in_sharded:
            x = ttnn.sharded_to_interleaved(x)

        input_dtype = x.dtype
        # Promote to fp32 so the kernel's x*x is computed in fp32 (HF-faithful — the precision this
        # class exists to preserve). fp32_dest_acc keeps the variance reduction in fp32 too.
        x_fp32 = ttnn.typecast(x, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.rms_norm(
            x_fp32,
            epsilon=self.eps,
            weight=self.weight_tt,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if x_fp32.is_allocated():
            ttnn.deallocate(x_fp32)
        if out.dtype != input_dtype:
            out = ttnn.typecast(out, input_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        if in_sharded:
            out = ttnn.to_memory_config(out, sharded_output_config)
        elif output_mem_config is not None:
            out = ttnn.to_memory_config(out, output_mem_config)
        return out
