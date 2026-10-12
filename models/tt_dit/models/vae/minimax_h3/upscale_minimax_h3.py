# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Bicubic resize of the stitched VAE canvas on device, before the YUV readback.

Bicubic resize is separable, so it is two matmuls with fixed interpolation matrices: one along the
width, one along the height. Each device of the mesh computes one output patch, so the result is
already sharded the way `fast_device_to_host_yuv` reads it: H over mesh rows, W over mesh columns.

The strip stitch leaves each device with a horizontal band of the canvas (`1/mesh_rows` of the
rows, full width). The width matmul runs on that band directly, keeping only this device's output
columns, and an all-gather over the mesh rows then gives every device all input rows of its
columns. Each output pixel therefore sees exactly the input pixels it would see on host: there is
no halo to exchange and no seam at the device boundaries.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn

TILE = 32


def bicubic_matrix(size_in: int, size_out: int) -> torch.Tensor:
    """`(size_out, size_in)` matrix `M` with `M @ x` equal to torch's bicubic resize of `x` along one axis.

    Built by resizing an identity, so it has torch's exact taps (a = -0.75) and border clamping
    (`align_corners=False`).
    """
    eye = torch.eye(size_in, dtype=torch.float64).reshape(size_in, 1, 1, size_in)
    out = F.interpolate(eye, size=(1, size_out), mode="bicubic", align_corners=False)
    return out.reshape(size_in, size_out).T.contiguous()


def padded(n: int) -> int:
    return -(-n // TILE) * TILE


def transposed_blocks(matrix: torch.Tensor, parts: int) -> torch.Tensor:
    """`matrix.T` cut into `parts` equal column blocks, each zero-padded to a tile multiple.

    Block `k` holds the weights of output rows `k * per : (k + 1) * per` of `matrix`, so sharding
    the result's last dim `parts` ways gives each device the weights of its own output patch.
    """
    size_out, size_in = matrix.shape
    per = size_out // parts
    blocks = matrix.T.reshape(size_in, parts, per)
    return F.pad(blocks, (0, padded(per) - per)).reshape(size_in, parts * padded(per))


def upscale_blocks_reference(
    canvas: torch.Tensor, out_hw: tuple[int, int], mesh_shape: tuple[int, int]
) -> torch.Tensor:
    """Torch model of `CanvasUpscaler`: each device's patch from its row band, then the patches assembled.

    `canvas` is `(..., H, W)`; returns `(..., out_h, out_w)`.
    """
    mesh_rows, mesh_cols = mesh_shape
    in_h, in_w = canvas.shape[-2:]
    out_h, out_w = out_hw
    per_h, per_w = out_h // mesh_rows, out_w // mesh_cols
    width = transposed_blocks(bicubic_matrix(in_w, out_w), mesh_cols).to(canvas.dtype)
    height = transposed_blocks(bicubic_matrix(in_h, out_h), mesh_rows).to(canvas.dtype)
    bands = canvas.chunk(mesh_rows, dim=-2)
    rows = []
    for r in range(mesh_rows):
        patches = []
        for c in range(mesh_cols):
            width_c = width[:, c * padded(per_w) : (c + 1) * padded(per_w)]
            height_r = height[:, r * padded(per_h) : (r + 1) * padded(per_h)]
            gathered = torch.cat([band @ width_c for band in bands], dim=-2)
            patch = (gathered.transpose(-2, -1) @ height_r).transpose(-2, -1)
            patches.append(patch[..., :per_h, :per_w])
        rows.append(torch.cat(patches, dim=-1))
    return torch.cat(rows, dim=-2)


class CanvasUpscaler:
    """Resizes a row-partitioned `(1, C, T, H / mesh_rows, W)` canvas to a 2D-sharded `(out_h, out_w)` one.

    Output per device: `(1, C, T, out_h / mesh_rows, out_w / mesh_cols)` bfloat16 ROW_MAJOR, clamped
    to `[-1, 1]`: the input `fast_device_to_host_yuv` takes.
    """

    def __init__(self, mesh_device: ttnn.MeshDevice, in_hw: tuple[int, int], out_hw: tuple[int, int]) -> None:
        mesh_rows, mesh_cols = tuple(mesh_device.shape)
        (in_h, in_w), (out_h, out_w) = in_hw, out_hw
        if in_h % mesh_rows or out_h % mesh_rows or out_w % mesh_cols:
            raise ValueError(f"{in_hw} -> {out_hw} does not split over the {mesh_rows}x{mesh_cols} mesh")
        self.per_h, self.per_w = out_h // mesh_rows, out_w // mesh_cols
        if self.per_h % 2 or self.per_w % 2:
            raise ValueError(f"per-device patch {self.per_h}x{self.per_w} must be even for 4:2:0")
        self.mesh_device = mesh_device
        self.in_hw, self.out_hw = in_hw, out_hw
        shape = (mesh_rows, mesh_cols)

        def upload(matrix, dims):
            return ttnn.from_torch(
                matrix.float(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=shape, dims=dims),
            )

        self._width = upload(transposed_blocks(bicubic_matrix(in_w, out_w), mesh_cols), (None, -1))
        self._height = upload(transposed_blocks(bicubic_matrix(in_h, out_h), mesh_rows), (-1, None))
        # fp32 accumulation: the taps are signed, so bf16 sums would cost the PSNR the resize is held to.
        self._compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, canvas_rows: ttnn.Tensor) -> ttnn.Tensor:
        _, channels, frames, rows, width = tuple(canvas_rows.shape)
        if (rows * tuple(self.mesh_device.shape)[0], width) != tuple(self.in_hw):
            raise ValueError(f"canvas band {rows}x{width} does not match the upscaler's {self.in_hw} input")
        x = canvas_rows
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        if x.dtype != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        x = ttnn.reshape(x, (1, channels * frames, rows, width))
        x = ttnn.matmul(x, self._width, dtype=ttnn.float32, compute_kernel_config=self._compute)
        x = ttnn.all_gather(x, 2, cluster_axis=0, topology=ttnn.Topology.Ring)
        x = ttnn.transpose(x, -2, -1)
        x = ttnn.matmul(x, self._height, dtype=ttnn.float32, compute_kernel_config=self._compute)
        x = ttnn.transpose(x, -2, -1)
        x = ttnn.clamp(x, min=-1.0, max=1.0)
        x = ttnn.typecast(x, ttnn.bfloat16)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.slice(x, (0, 0, 0, 0), (1, channels * frames, self.per_h, self.per_w))
        return ttnn.reshape(x, (1, channels, frames, self.per_h, self.per_w))
