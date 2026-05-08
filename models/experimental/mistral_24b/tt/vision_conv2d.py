# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
This is the modified version of the vision_patch_conv2d for the Mistral-Small-3.1-24B-Instruct-2503 model.
We have modified the llama_patch_conv2d to be compatible with the Mistral-Small-3.1-24B-Instruct-2503 model.

Prototype: patch-embedding linear memory (MatmulDeviceOperation) can be tuned via env:
- TT_MISTRAL_VISION_PATCH_EMBED_MEM=dram|l1|height_sharded (default dram)
- TT_MISTRAL_VISION_PATCH_EMBED_WEIGHT_L1=0|1 (optional; pins linear weights in L1 at init)
- TT_MISTRAL_VISION_PATCH_EMBED_LOG=0|1 (prints tensor buffer/layout before/after linear)

Downstream note: `MistralVisionTower.forward` runs transpose/reshape/concat then `ln_pre`. The `simplified_rms`
path (`_simplified_rmsnorm`) always starts with `ttnn.sharded_to_interleaved(inp, DRAM_MEMORY_CONFIG)` before
`pow`/`mean` (see `models/experimental/mistral_24b/tt/rmsnorm.py`), which typically lands activations in DRAM early
in the norm regardless of whether `inp` was interleaved or sharded—so that line is the first obvious
end-to-end DRAM spill after the patch MLP unless the op elides copies for already-interleaved DRAM tensors.
"""

from __future__ import annotations

import math
import os
from typing import Literal, Tuple

import torch
import ttnn
from loguru import logger

from models.common.lightweightmodule import LightweightModule
from models.experimental.oft.tt.common import infer_out_subblock

try:
    from tracy import signpost
except ImportError:

    def signpost(*args, **kwargs):
        pass


_PATCH_MEM_MODE = Literal["dram", "l1", "height_sharded"]


def _patch_embed_mem_mode() -> _PATCH_MEM_MODE:
    v = os.environ.get("TT_MISTRAL_VISION_PATCH_EMBED_MEM", "dram").strip().lower()
    if v in ("dram", "l1", "height_sharded"):
        return v  # type: ignore[return-value]
    logger.warning(
        f"TT_MISTRAL_VISION_PATCH_EMBED_MEM={v!r} invalid; use dram|l1|height_sharded. Falling back to dram."
    )
    return "dram"


def _patch_embed_log_enabled() -> bool:
    return os.environ.get("TT_MISTRAL_VISION_PATCH_EMBED_LOG", "0") == "1"


def _log_tt_tensor(tag: str, tensor: ttnn.Tensor) -> None:
    if not _patch_embed_log_enabled():
        return
    try:
        mc = ttnn.get_memory_config(tensor)
        logger.info(
            f"[patch_embed] {tag}: shape={tuple(tensor.shape)} padded={tuple(tensor.padded_shape)} "
            f"layout={tensor.layout} memory_config={mc}"
        )
    except Exception as exc:  # pragma: no cover
        logger.info(f"[patch_embed] {tag}: <log failed: {exc}>")


def _reshape_to_mm4(x: ttnn.Tensor) -> Tuple[ttnn.Tensor, int, int]:
    """Return (x_4d, nhw, in_ch) with x_4d shaped [1, 1, nhw, in_ch] for matmul helpers."""
    sh = tuple(x.shape)
    if len(sh) == 4:
        _b0, _b1, nhw, in_ch = sh
        return x, int(nhw), int(in_ch)
    if len(sh) == 3:
        _b, nhw, in_ch = sh
        return ttnn.reshape(x, (1, 1, nhw, in_ch)), int(nhw), int(in_ch)
    raise RuntimeError(f"patch_embed: expected rank 3 or 4 TTNN tensor, got shape {sh}")


def _height_sharded_program_and_memcfgs(
    mesh_device,
    nhw: int,
    in_ch: int,
    out_ch: int,
) -> Tuple[ttnn.MemoryConfig, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig, ttnn.MemoryConfig]:
    """Height-shard inputs/outputs in L1 with a 1D multicast matmul config (weights typically stay DRAM)."""
    compute_grid = mesh_device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
    total_cores = core_grid.x * core_grid.y
    n_tiles_h = (int(nhw) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    per_core_M = max(1, math.ceil(n_tiles_h / total_cores))
    k_tiles = (int(in_ch) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE
    per_core_N = max(1, math.ceil(int(out_ch) / ttnn.TILE_SIZE))
    in0_block_w = 1
    for w in range(min(8, k_tiles), 0, -1):
        if k_tiles % w == 0:
            in0_block_w = w
            break
    out_sub_h, out_sub_w = infer_out_subblock(per_core_M, per_core_N, dtype=ttnn.bfloat16)
    out_block_h = per_core_M
    out_block_w = per_core_N
    shard_height = per_core_M * ttnn.TILE_SIZE
    in_shard_width = (int(in_ch) + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
    out_shard_width = per_core_N * ttnn.TILE_SIZE
    memory_config_in = ttnn.create_sharded_memory_config(
        (shard_height, in_shard_width),
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    memory_config_out = ttnn.create_sharded_memory_config(
        (shard_height, out_shard_width),
        core_grid,
        ttnn.ShardStrategy.HEIGHT,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(core_grid.x, core_grid.y),
        in0_block_w=in0_block_w,
        out_subblock_h=out_sub_h,
        out_subblock_w=out_sub_w,
        out_block_h=out_block_h,
        out_block_w=out_block_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
    )
    return memory_config_in, program_config, memory_config_out


class TtMistralConv2dPatch(LightweightModule):
    """Conv2D Patching layer.
    Arguments:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Size of convolution kernel.
        stride (default 1): Stride for convolution.
        bias (default False): Use bias in Conv2d.
    Input: (bsz, in_channels, width, height)
    Output: (bsz, num_tokens, out_channels)
    """

    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        dtype,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.num_devices = self.mesh_device.get_num_devices()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self._patch_embed_mem_mode = _patch_embed_mem_mode()
        self._pin_linear_weight_l1 = os.environ.get("TT_MISTRAL_VISION_PATCH_EMBED_WEIGHT_L1", "0") == "1"
        weight_mem = ttnn.L1_MEMORY_CONFIG if self._pin_linear_weight_l1 else ttnn.DRAM_MEMORY_CONFIG

        self.bias = (
            ttnn.as_tensor(
                torch.reshape(state_dict[f"{state_dict_prefix}_linear.bias"], (1, -1)),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            if bias
            else None
        )

        self._unfold = torch.nn.Unfold(kernel_size=self.kernel_size, stride=self.stride)

        weight = state_dict[f"{state_dict_prefix}_linear.weight"]
        if weight.ndim == 4:
            weight = weight.reshape(out_channels, -1).T

        self._linear_weight = ttnn.as_tensor(
            weight,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=weight_mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(self, x: torch.Tensor):
        signpost("Mistral24B::PatchEmbedding::Start", f"input_shape={tuple(x.shape)}")
        signpost("Mistral24B::PatchEmbedding::Unfold::Start")
        x = self._unfold(x)
        x = x.permute(0, 2, 1)
        signpost("Mistral24B::PatchEmbedding::Unfold::End", f"unfold_shape={tuple(x.shape)}")

        act_mem = ttnn.DRAM_MEMORY_CONFIG
        if self._patch_embed_mem_mode == "l1":
            act_mem = ttnn.L1_MEMORY_CONFIG

        signpost("Mistral24B::DeviceTransfer::PatchInputAsTensor::Start", self._patch_embed_mem_mode)
        x_tt = ttnn.as_tensor(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=act_mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        signpost("Mistral24B::DeviceTransfer::PatchInputAsTensor::End", self._patch_embed_mem_mode)
        _log_tt_tensor("activations after as_tensor", x_tt)
        _log_tt_tensor("linear weight", self._linear_weight)

        linear_mode = self._patch_embed_mem_mode
        if linear_mode == "height_sharded" and tuple(x_tt.shape)[0] != 1:
            logger.warning(
                "[patch_embed] height_sharded path supports batch size 1 only; using dram linear for this forward."
            )
            linear_mode = "dram"

        if linear_mode == "height_sharded":
            x_4d, nhw, in_ch = _reshape_to_mm4(x_tt)
            out_ch = int(self._linear_weight.shape[-1])
            mem_in, program_config, mem_out = _height_sharded_program_and_memcfgs(self.mesh_device, nhw, in_ch, out_ch)
            signpost("Mistral24B::PatchEmbedding::ReshardToL1::Start", "height_sharded")
            x_s = ttnn.to_memory_config(x_4d, memory_config=mem_in, dtype=ttnn.bfloat16)
            signpost("Mistral24B::PatchEmbedding::ReshardToL1::End", "height_sharded")
            _log_tt_tensor("activations resharded height L1", x_s)
            signpost("Mistral24B::PatchEmbedding::Matmul::Start", "height_sharded")
            out = ttnn.linear(
                x_s,
                self._linear_weight,
                bias=self.bias,
                dtype=ttnn.bfloat16,
                memory_config=mem_out,
                program_config=program_config,
                compute_kernel_config=self.compute_kernel_config,
            )
            signpost("Mistral24B::PatchEmbedding::Matmul::End", "height_sharded")
            if len(tuple(x_tt.shape)) == 3:
                out = ttnn.reshape(out, (x_tt.shape[0], out.shape[-2], out.shape[-1]))
        else:
            out_mem = ttnn.DRAM_MEMORY_CONFIG if linear_mode == "dram" else ttnn.L1_MEMORY_CONFIG
            signpost("Mistral24B::PatchEmbedding::Matmul::Start", linear_mode)
            out = ttnn.linear(
                x_tt,
                self._linear_weight,
                bias=self.bias,
                dtype=ttnn.bfloat16,
                memory_config=out_mem,
                compute_kernel_config=self.compute_kernel_config,
                core_grid=ttnn.CoreGrid(y=8, x=8),
            )
            signpost("Mistral24B::PatchEmbedding::Matmul::End", linear_mode)

        _log_tt_tensor("patch linear output", out)
        if _patch_embed_log_enabled():
            logger.info(
                f"[patch_embed] mem_mode={self._patch_embed_mem_mode} linear_mode={linear_mode} "
                f"weight_l1={self._pin_linear_weight_l1}"
            )

        signpost("Mistral24B::PatchEmbedding::End", f"output_shape={tuple(out.shape)}")
        return out
