# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The vision model: `KimiK3VisionModel` in the reference.

`TT_MoonViT_Vision_Tower` is `MoonViT3dPretrainedModel` (patch embed, 27 blocks, final norm, patch
merger). `TT_MoonViT_Vision_Model` adds the projector (`mm_projector`), so it maps image patches to
language-model embeddings.

One still image per call. Its patches are put in window order and padded to `config.max_patches`
rows on the host (`host_prep.prepare_host_inputs`, built fast by `host_fill`), written split along
`sp_axis` into device inputs allocated once (`DeviceInput`, so a trace can replay them), run through
the tower, merger and projector, and gathered back with the padding dropped. The merged tokens come out in the
reference's order. Weights are replicated and activations are DRAM interleaved.

Child names match the reference, so `load_torch_state_dict(reference.state_dict())` loads it, and
`module.load_cached` caches it.
"""

from __future__ import annotations

import numpy as np
import torch

import ttnn

from .ccl import ParallelFactor, TT_CCL, ccl_slot_for_layer
from . import host_fill
from .device_input import DeviceInput
from .image_prep import check_pixels
from .model_config import MoonViTConfig, VisionParallel, check_grid, compute_kernel_config, resolve_parallel
from .module import Module, ModuleList
from .patch_embed import TT_Patch_Embed
from .patch_merger import TT_Patch_Merger
from .projector import TT_Projector
from .rms_norm import RMSNorm
from .rope import RopeTensors, TT_Rope
from .vision_block import TT_Vision_Block

# The numpy dtypes that hold bf16 values bit for bit: their raw 16 bits, or ml_dtypes' bfloat16.
BF16_HOLDERS = ("uint16", "int16", "bfloat16")


def check_output_rows(rows: np.ndarray) -> None:
    """Raise unless `rows` can take `read_output`'s copy: a writable, C-contiguous, non-empty 2D numpy array of one of
    `BF16_HOLDERS`. TypeError for another type or dtype, ValueError for the rest."""
    if not isinstance(rows, np.ndarray) or rows.dtype.name not in BF16_HOLDERS:
        raise TypeError(
            f"rows must be a numpy array of {', '.join(BF16_HOLDERS)} that holds bf16 values,"
            f" got {getattr(rows, 'dtype', type(rows).__name__)}"
        )
    if rows.ndim != 2 or rows.shape[0] == 0 or not (rows.flags.c_contiguous and rows.flags.writeable):
        raise ValueError(
            f"rows must be a writable, C-contiguous [logical_n, C] array with logical_n > 0, got shape {rows.shape},"
            f" C-contiguous {rows.flags.c_contiguous}, writable {rows.flags.writeable}"
        )


class TT_MoonViT_Encoder(Module):
    def __init__(
        self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, parallel: VisionParallel, compute_kernel_config
    ) -> None:
        super().__init__()
        n = config.num_hidden_layers
        self.blocks = ModuleList(
            TT_Vision_Block(
                config,
                mesh_device=mesh_device,
                parallel=parallel,
                compute_kernel_config=compute_kernel_config,
                ccl_slot=ccl_slot_for_layer(layer, n),
            )
            for layer in range(n)
        )
        # Reads the fp32 residual stream, hands bf16 to the merger.
        self.final_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            mesh_device=mesh_device,
            compute_kernel_config=compute_kernel_config,
            output_dtype=ttnn.bfloat16,
        )

    def forward(self, x: ttnn.Tensor, rope: RopeTensors, logical_n: ttnn.Tensor) -> ttnn.Tensor:
        """`MoonViT3dEncoder.forward`: every block, then the final norm. [1, 1, rows, hidden_size] ->
        the same shape. Each block's output is freed once the next has read it; `x` is left
        allocated: the caller owns it."""
        h = x
        for block in self.blocks:
            out = block(h, rope, logical_n)
            if h is not x:
                ttnn.deallocate(h)
            h = out
        out = self.final_layernorm(h)
        if h is not x:
            ttnn.deallocate(h)
        return out


class TT_MoonViT_Vision_Tower(Module):
    def __init__(
        self, config: MoonViTConfig, *, mesh_device: ttnn.MeshDevice, parallel: VisionParallel, compute_kernel_config
    ) -> None:
        super().__init__()
        self.patch_embed = TT_Patch_Embed(config, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config)
        self.encoder = TT_MoonViT_Encoder(
            config, mesh_device=mesh_device, parallel=parallel, compute_kernel_config=compute_kernel_config
        )
        self.merger = TT_Patch_Merger(config)

    def forward(
        self, patches: ttnn.Tensor, pos_emb: ttnn.Tensor, rope: RopeTensors, logical_n: ttnn.Tensor
    ) -> ttnn.Tensor:
        embedded = self.patch_embed(patches, pos_emb)
        encoded = self.encoder(embedded, rope, logical_n)
        ttnn.deallocate(embedded)
        merged = self.merger(encoded)
        ttnn.deallocate(encoded)
        return merged


class TT_MoonViT_Vision_Model(Module):
    def __init__(
        self,
        config: MoonViTConfig,
        *,
        mesh_device: ttnn.MeshDevice,
        sequence_parallel: ParallelFactor | None = None,
        ccl: TT_CCL | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mesh_device = mesh_device
        self.parallel = resolve_parallel(mesh_device, sequence_parallel, ccl, config)
        ckc = compute_kernel_config(mesh_device)
        self.rope = TT_Rope(config, mesh_device=mesh_device, parallel=self.parallel)
        self.vision_tower = TT_MoonViT_Vision_Tower(
            config, mesh_device=mesh_device, parallel=self.parallel, compute_kernel_config=ckc
        )
        self.mm_projector = TT_Projector(config, mesh_device=mesh_device, compute_kernel_config=ckc)
        sp_axis = self.parallel.sp_axis
        rows = config.max_patches
        # Row major: the patch embed tiles it on device (patch_embed.py).
        self.patches = DeviceInput(
            (1, 1, rows, config.padded_patch_dim),
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        # fp32: added into the fp32 residual stream (patch_embed.py).
        self.pos_emb = DeviceInput(
            (1, 1, rows, config.hidden_size), mesh_device=mesh_device, sp_axis=sp_axis, dtype=ttnn.float32
        )
        self.logical_n = DeviceInput(
            (1, 1, 1, 1), mesh_device=mesh_device, sp_axis=None, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        # The grid whose position embeddings, RoPE tables and logical_n the chips hold: they depend on nothing else,
        # so an image of the same grid writes only its patches.
        self._grid: tuple[int, int] | None = None
        self._pos_table: host_fill.PosEmbedTable | None = None
        # read_output's host buffer for one chip's shard: (output shape, uint16 rows, host tensor that borrows them).
        self._readback: tuple | None = None

    def prepare_inputs(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> dict:
        """Write one image's inputs, split along `sp_axis`, into the same device tensors on every call: the values of
        `host_prep.prepare_host_inputs`, built only for the image's rows (`host_fill`) and written into the staging
        buffers of the `DeviceInput`s, whose padding rows stay zero.

        Returns `patches` [1, 1, S/sp, padded_patch_dim], `pos_emb` [1, 1, S/sp, hidden], `rope` and
        `logical_n` (the unpadded length, [1, 1, 1, 1] uint32), rows in window order.
        """
        self._pos_emb_weight()
        h, w = check_grid(grid_thws, self.config)
        expected = (h * w, self.config.in_channels, self.config.patch_size, self.config.patch_size)
        if tuple(pixel_values.shape) != expected:
            raise ValueError(f"pixel_values {tuple(pixel_values.shape)} != {expected} for grid {h}x{w}")
        self.patches.fill_rows(h * w, lambda rows: host_fill.patch_values_into(rows, pixel_values, h, w, self.config))
        return self._write_grid(h, w)

    def prepare_image(self, pixels: np.ndarray) -> dict:
        """`prepare_inputs(*image_prep.pixels_to_patches(pixels, config))` from `image_prep.resize_image`'s uint8
        [H, W, 3] pixels, with the same device inputs, bit for bit: the patches are normalized straight into their
        bf16 rows, with no float32 `pixel_values`."""
        self._pos_emb_weight()
        h, w = check_pixels(pixels, self.config)
        self.patches.fill_rows(h * w, lambda rows: host_fill.normalized_patches_into(rows, pixels, self.config))
        return self._write_grid(h, w)

    def _pos_emb_weight(self) -> torch.Tensor:
        weight = self.vision_tower.patch_embed.pos_emb_weight
        if weight is None:
            raise RuntimeError("weights not loaded; call load_torch_state_dict or load_cached first")
        return weight

    def _write_grid(self, h: int, w: int) -> dict:
        """Write the inputs that depend only on the grid, unless the chips hold this grid's already, from these
        weights."""
        weight = self._pos_emb_weight()
        if self._pos_table is None or self._pos_table.weight is not weight:  # first call, or weights reloaded
            self._pos_table = host_fill.PosEmbedTable(weight, self.config)
            self._grid = None
        if self._grid != (h, w):
            self._grid = None  # until every write below has completed
            self.pos_emb.write_rows(self._pos_table.window_rows(h, w))
            self.rope.write_rows(*host_fill.rope_window_rows(self.config, h, w))
            self.logical_n.write(torch.tensor(h * w, dtype=torch.int64))
            self._grid = (h, w)
        return {
            "patches": self.patches.tensor,
            "pos_emb": self.pos_emb.tensor,
            "rope": self.rope.tensors(),
            "logical_n": self.logical_n.tensor,
        }

    def _data_coords(self, rows_per_chip: int, logical_n: int) -> list:
        """The mesh coordinates of the chips that hold rows 0..logical_n-1, in order: the first line of chips along
        `sp_axis`, cut at the last chip with real rows; any other lines hold replicas. Without sequence parallelism,
        the first replica."""
        sp_axis = self.parallel.sp_axis
        if sp_axis is None:
            return [ttnn.MeshCoordinate(0, 0)]
        chips = -(-logical_n // rows_per_chip)
        return [ttnn.MeshCoordinate(*((0, k) if sp_axis == 1 else (k, 0))) for k in range(chips)]

    def gather_output(self, out: ttnn.Tensor, *, logical_n: int) -> torch.Tensor:
        """Concatenate the shards along `sp_axis` and drop the padding -> (logical_n, C). The merger
        keeps the reference's token order, so no reordering is needed. Without sequence parallelism the
        first replica is kept.

        Only the chips with real rows are read, in one batch (`Tensor.cpu_shards`); each shard is converted on its
        own and joined with `torch.cat`, which is faster than `ttnn.to_torch` with a mesh composer."""
        shards = out.cpu_shards(self._data_coords(out.shape[2], logical_n))
        return torch.cat([ttnn.to_torch(shard)[0, 0] for shard in shards])[:logical_n]

    def read_output(self, out: ttnn.Tensor, rows: np.ndarray) -> None:
        """Copy the first `len(rows)` rows of `forward`'s output into `rows`, chip by chip.

        rows: writable, C-contiguous [logical_n, C] array that holds bf16 bits (uint16, int16 or ml_dtypes' bfloat16),
        for example the rows of a stored value in shared memory. Each chip's real rows are copied straight to their
        place, with no tensor of the whole output and no host untilize. Only the chips that hold real rows are read:
        when that is the first chip alone, into a host buffer kept across calls (`_read_first_chip`); otherwise in one
        batch (`Tensor.cpu_shards`). Chips that hold only padding are never read. The reads release the GIL.

        Raises TypeError or ValueError (`check_output_rows`) for a bad `rows`, and ValueError for an `out` that is not
        forward's output or holds fewer than `logical_n` rows of width C, all before reading the chips.
        """
        check_output_rows(rows)
        if out.layout != ttnn.ROW_MAJOR_LAYOUT or out.dtype != ttnn.bfloat16:
            raise ValueError(f"expected forward's bf16 row-major output, got {out.dtype} {out.layout}")
        logical_n, width = rows.shape
        rows_per_chip = out.shape[2]
        if width != out.shape[3]:
            raise ValueError(f"rows of width {width} for an output of width {out.shape[3]}")
        capacity = rows_per_chip * self.parallel.sp_factor
        if logical_n > capacity:
            raise ValueError(f"{logical_n} rows for an output of {capacity}")
        rows = rows.view(np.uint16)  # the same bits, whichever of BF16_HOLDERS rows is
        coords = self._data_coords(rows_per_chip, logical_n)
        if len(coords) == 1:
            # The first chip holds every real row: read it into the host buffer kept for this.
            shards = [self._read_first_chip(out)]
        else:
            # Row major with no tile padding, so the padded shape is the logical one; unlike `to_torch`, it hands back
            # the host buffer without a copy.
            shards = [s.to_torch_with_padded_shape()[0, 0].view(torch.uint16).numpy() for s in out.cpu_shards(coords)]
        # TODO: read straight into `rows` (copy_device_to_host_tensor into a host tensor that borrows its memory) to
        # skip this copy, but the watcher's NOC check aborts that read.
        for i, shard in enumerate(shards):
            start = i * rows_per_chip
            n = min(rows_per_chip, logical_n - start)
            np.copyto(rows[start : start + n], shard[:n])

    def _read_first_chip(self, out: ttnn.Tensor) -> np.ndarray:
        """Read the shard of `out` on the first chip, mesh coordinate (0, 0), into a host buffer allocated once ->
        uint16 [rows per chip, C], valid until the next call. Saves allocating, zeroing and freeing a buffer of the
        whole shard on every read, as `cpu_shards` does.

        The buffer is borrowed by a host tensor of `out`'s topology (`ttnn.from_local_host_shards`) whose shards all
        alias it: a copy from the first chip's own device tensor writes only that chip's shard."""
        shape = tuple(out.shape)
        if self._readback is None or self._readback[0] != shape:
            self._readback = None  # drop the old buffer first
            rows_per_chip, width = shape[2], shape[3]
            buffer = np.empty(rows_per_chip * width, np.uint16)
            spec = ttnn.TensorSpec(
                ttnn.Shape([1, 1, rows_per_chip, width]), out.dtype, out.layout, ttnn.DRAM_MEMORY_CONFIG
            )
            shard = ttnn.host_tensor_from_bytes(buffer.view(np.uint8), spec)
            host = ttnn.from_local_host_shards(
                {coord: shard for coord in out.device_coords()}, self.mesh_device, out.tensor_topology()
            )
            self._readback = (shape, buffer.reshape(rows_per_chip, width), host)
        _, buffer, host = self._readback
        ttnn.copy_device_to_host_tensor(ttnn.get_device_tensors(out)[0], host)
        return buffer

    def forward(
        self, patches: ttnn.Tensor, pos_emb: ttnn.Tensor, rope: RopeTensors, logical_n: ttnn.Tensor
    ) -> ttnn.Tensor:
        """-> bf16 [1, 1, S/(4*sp), C], row major. Untilized on the chips, since the output always goes to the host:
        inside a trace, so `read_output` copies plain rows with no host untilize."""
        x = self.vision_tower(patches, pos_emb, rope, logical_n)
        out = self.mm_projector(x)
        ttnn.deallocate(x)
        rows = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out)
        return rows

    def run(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        """Host in, host out: (h*w, 3, 14, 14), [[1, h, w]] -> (h*w/4, text_hidden_size)."""
        h, w = check_grid(grid_thws, self.config)
        inputs = self.prepare_inputs(pixel_values, grid_thws)
        merged_n = h * w // self.config.merge_area
        return self.gather_output(self.forward(**inputs), logical_n=merged_n)
