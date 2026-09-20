# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The video VAE's tile blend and unpatchify, on device.

Why this exists: the host path reads *overlapping* pixel tiles back and blends them on host. The
assembled canvas is far smaller than the tiles that produce it, so most of that transfer is overlap
that gets averaged away.

Why it is not simply a weighted accumulation: the reference `stitch_tiles` is **sequential and
asymmetric**. For an interior tile the corner region is `b*L + (1-b)*(a*A + (1-a)*T)` where `L` is the
*unblended* left tile and the diagonal tile never appears. A separable ramp formulation gives an
O(1) error over roughly a ninth of every frame, which surfaces as visible seams. So this mirrors the
reference order exactly, tile by tile, rather than reformulating it.

The blend runs in whatever dtype its tiles carry. float32 is what the host path it replaces used
(`.float()` before `stitch_tiles`), which is how the existing PCC and roundtrip-PSNR gates carried
over unchanged. bfloat16 is what the decoder actually emits and halves the bytes the blend moves;
both are gated in `test_stitch_device_minimax_h3.py`.
"""

from __future__ import annotations

import torch

import ttnn


class DeviceTileStitcher:
    """Blends a grid of decoded tiles into one clip, on device, caching its ramp weights.

    The weights depend only on `(extent, axis, shape)`, and a decode reuses the same geometry for
    every chunk, so they are built once and reused for the whole video.

    Everything here stays ROW_MAJOR. This grid's derived overlaps are [96, 80, 80] by height and
    [80, 80, 80, 80, 64, 64] by width, so the blend slices begin at 176 and 80 -- neither a multiple
    of 32 -- and `ttnn.slice` drops to untilize -> row-major -> retilize for exactly that case. The
    trims then hand `ttnn.concat` extents of 80 and 176, which is tile padding on the concat dim and
    triggers the same fallback again. `binary_ng` takes ROW_MAJOR operands and keeps the layout on
    output, so the arithmetic is unaffected.

    The ramp is built in its tiles' own dtype. bfloat16 tiles meeting a float32 ramp is what produced
    garbage-scale output on this ttnn, so the two are never allowed to disagree.
    """

    def __init__(self, mesh_device: ttnn.MeshDevice) -> None:
        self.mesh_device = mesh_device
        self._ramps: dict[tuple, ttnn.Tensor] = {}

    def _ramp(self, plane: tuple[int, int], rank: int, dim: int) -> ttnn.Tensor:
        """`weight_a = 1 - i/extent` along `dim`, over the trailing `plane = (H, W)`; leading dims broadcast."""
        key = (plane, dim)
        if key not in self._ramps:
            extent = plane[dim - (rank - 2)]
            positions = torch.arange(extent, dtype=torch.float32)
            view = [1] * rank
            view[dim] = extent
            slab = [1] * (rank - 2) + list(plane)
            weight_a = (1 - positions / extent).view(view).expand(slab).contiguous()
            self._ramps[key] = ttnn.from_torch(
                weight_a, dtype=ttnn.float32, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
            )
        return self._ramps[key]

    def bind_ramps(self, extents, tile: int) -> None:
        """Build every ramp up front (before trace capture); one allocated mid-request gets clobbered by replays."""
        for extent in extents:
            self._ramp((extent, tile), 5, 3)
            self._ramp((tile, extent), 5, 4)

    @staticmethod
    def _slice(tensor: ttnn.Tensor, dim: int, start: int, stop: int) -> ttnn.Tensor:
        rank = len(tensor.shape)
        dim = dim % rank
        starts = [0] * rank
        stops = list(tensor.shape)
        starts[dim], stops[dim] = start, stop
        return ttnn.slice(tensor, starts, stops)

    def blend(self, a: ttnn.Tensor, b: ttnn.Tensor, blend_extent: int, dim: int) -> ttnn.Tensor:
        """Cross-fade `a`'s tail into `b`'s head along `dim`; result is shaped like `b`.

        Mirrors the host `blend` in `vae_minimax_h3.py`, including its early return when the extent
        covers all of `b`.
        """
        rank = len(b.shape)
        axis = dim % rank
        blend_extent = min(a.shape[axis], b.shape[axis], blend_extent)

        # An operand the extent covers whole is used as is; slicing it would only copy it.
        tail_a = (
            a if blend_extent == a.shape[axis] else self._slice(a, axis, a.shape[axis] - blend_extent, a.shape[axis])
        )
        head_b = self._slice(b, axis, 0, blend_extent)
        weight_a = self._ramp((head_b.shape[-2], head_b.shape[-1]), rank, axis)
        blended = ttnn.add(head_b, ttnn.multiply(ttnn.subtract(tail_a, head_b), weight_a))

        if blend_extent == b.shape[axis]:
            return blended
        rest = self._slice(b, axis, blend_extent, b.shape[axis])
        return ttnn.concat([blended, rest], dim=axis)

    def stitch(
        self,
        tiles: list[list[ttnn.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> ttnn.Tensor:
        """Device mirror of `stitch_tiles`, in the same order and with the same trims.

        Note `tiles[i - 1][j]` and `row[j - 1]` are the **original** tiles, not previously blended
        ones. That asymmetry is the reference's and is reproduced here.
        """
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self.blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = self._slice(tile, -2, 0, tile.shape[-2] - height_overlaps[i])
                if j < len(row) - 1:
                    tile = self._slice(tile, -1, 0, tile.shape[-1] - width_overlaps[j])
                result_row.append(tile)
            result_rows.append(ttnn.concat(result_row, dim=-1))
        return ttnn.concat(result_rows, dim=-2)


class StripTileStitcher(DeviceTileStitcher):
    """`stitch` split along the mesh, so no device blends canvas it never reads back: gather a tile column, H-blend
    and trim, keep this device's rows; gather those row strips, W-blend and trim, keep its columns. Same bits out."""

    # Both stages work off the gathered stacks (tile index in dim 0): each blend operand and kept run is sliced once and
    # each column or row concatenated once, instead of copying tiles out, blending and trimming the result again.

    @staticmethod
    def _sub(stack: ttnn.Tensor, index: int, dim: int, start: int, stop: int) -> ttnn.Tensor:
        """Entry `index` of a stacked tensor (dim 0), sliced to `[start, stop)` along `dim`."""
        rank = len(stack.shape)
        dim = dim % rank
        starts = [0] * rank
        stops = list(stack.shape)
        starts[0], stops[0] = index, index + 1
        starts[dim], stops[dim] = start, stop
        return ttnn.slice(stack, starts, stops)

    @staticmethod
    def _corner(stack: ttnn.Tensor, index: int, rows: int, cols: int) -> ttnn.Tensor:
        """Entry `index`: the top `rows` rows of the last `cols` columns."""
        rank = len(stack.shape)
        starts = [0] * rank
        stops = list(stack.shape)
        starts[0], stops[0] = index, index + 1
        starts[-1] = stack.shape[-1] - cols
        stops[-2] = rows
        return ttnn.slice(stack, starts, stops)

    def _blend_pieces(
        self, a: ttnn.Tensor, a_index: int, b: ttnn.Tensor, b_index: int, extent: int, axis: int, keep: int
    ) -> list[ttnn.Tensor]:
        """Pieces of `blend(a[a_index], b[b_index])` along `axis` with `b` kept to `keep`: the blended overlap, then b's
        `[extent, keep)` run as its own slice; same operands and ops as `blend`, with no concat and no trim."""
        axis = axis % len(b.shape)
        extent = min(a.shape[axis], b.shape[axis], extent)
        tail_a = self._sub(a, a_index, axis, a.shape[axis] - extent, a.shape[axis])
        head_b = self._sub(b, b_index, axis, 0, extent)
        weight_a = self._ramp((head_b.shape[-2], head_b.shape[-1]), len(head_b.shape), axis)
        pieces = [ttnn.add(head_b, ttnn.multiply(ttnn.subtract(tail_a, head_b), weight_a))]
        if keep > extent:
            pieces.append(self._sub(b, b_index, axis, extent, keep))
        return pieces

    def column(
        self, column: ttnn.Tensor, rows: int, height_overlaps: list[int], edge_width: int
    ) -> tuple[ttnn.Tensor, ttnn.Tensor | None]:
        """H-blend one gathered tile column (entries `0..rows`) and trim as `stitch` would; returns `(strip, edge)`.
        `strip` is the column at canvas height. `edge` is the last `edge_width` columns of the ORIGINAL tiles at those
        rows: the next column's W-blend reads it in place of `strip`, as `stitch` blends against `row[j - 1]`, never
        the blended tile. `None` when `edge_width` is 0 (single-column grid, no W seam)."""
        height = column.shape[-2]
        pieces, edges = [], []
        for i in range(rows):
            keep = height - (height_overlaps[i] if i < rows - 1 else 0)
            if i == 0:
                pieces.append(self._sub(column, 0, -2, 0, keep))
            else:
                pieces.extend(self._blend_pieces(column, i - 1, column, i, height_overlaps[i - 1], -2, keep))
            if edge_width:
                edges.append(self._corner(column, i, rows=keep, cols=edge_width))
        strip = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=-2)
        if not edges:
            return strip, None
        return strip, edges[0] if len(edges) == 1 else ttnn.concat(edges, dim=-2)

    def row(
        self, strips: ttnn.Tensor, edges: ttnn.Tensor | None, first: int, cols: int, width_overlaps: list[int]
    ) -> ttnn.Tensor:
        """W-blend gathered row strips `first..first+cols` left to right and trim as `stitch` would; `edges` holds the
        matching un-blended edge strips (same indices), and the blend of column j reads `edges[first + j - 1]`."""
        width = strips.shape[-1]
        pieces = []
        for j in range(cols):
            keep = width - (width_overlaps[j] if j < cols - 1 else 0)
            if j == 0:
                pieces.append(self._sub(strips, first, -1, 0, keep))
            else:
                pieces.extend(
                    self._blend_pieces(edges, first + j - 1, strips, first + j, width_overlaps[j - 1], -1, keep)
                )
        return pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=-1)


class NeighborTileBlender:
    """Gather-free stitch: each tile cross-fades in place from its raw up/left neighbour halos.

    Halos come from a per-axis neighbour exchange; the blend is two matmuls against per-device constant
    weights. Requires a grid-aligned wave (tile ``(chunk k, r, c)`` on device ``(r, k * grid_cols + c)``).
    """

    def __init__(self, mesh_device: ttnn.MeshDevice, ccl_manager) -> None:
        assert ccl_manager is not None, "the neighbour exchange needs a CCLManager"
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self._weights: dict[tuple, tuple] = {}
        self._compute = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )

    @staticmethod
    def _band_matrix(pad: int, extent: int, overlap: int) -> torch.Tensor:
        """``(pad+extent, extent)`` identity with the first ``overlap`` outputs cross-faded from the halo."""
        m = torch.zeros(pad + extent, extent, dtype=torch.float32)
        for k in range(extent):
            if k < overlap:
                m[pad - overlap + k, k] = 1.0 - k / overlap
                m[pad + k, k] = k / overlap
            else:
                m[pad + k, k] = 1.0
        return m

    def _geometry_weights(
        self,
        *,
        grid_rows: int,
        grid_cols: int,
        y_overlaps: list[int],
        x_overlaps: list[int],
        chunks_per_wave: int,
        tile_h: int,
        tile_w: int,
        hpad: int,
        wpad: int,
    ) -> tuple:
        mesh_rows, mesh_cols = tuple(self.mesh_device.shape)
        key = (
            mesh_rows,
            mesh_cols,
            grid_rows,
            grid_cols,
            tuple(y_overlaps),
            tuple(x_overlaps),
            chunks_per_wave,
            tile_h,
            tile_w,
        )
        if key not in self._weights:
            mv_shards, nh_shards = [], []
            for shard in range(mesh_rows * mesh_cols):
                r, mc = shard // mesh_cols, shard % mesh_cols
                k, c = mc // grid_cols, mc % grid_cols
                valid = r < grid_rows and k < chunks_per_wave
                hov = y_overlaps[r - 1] if valid and r > 0 else 0
                wov = x_overlaps[c - 1] if valid and c > 0 else 0
                mv_shards.append(self._band_matrix(hpad, tile_h, hov))
                nh_shards.append(self._band_matrix(wpad, tile_w, wov))

            def upload(shards):
                stacked = torch.stack(shards, dim=0)
                tensor = ttnn.from_torch(
                    stacked,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
                )
                return ttnn.reshape(tensor, tuple(stacked.shape[1:]))

            self._weights[key] = (upload(mv_shards), upload(nh_shards))
        return self._weights[key]

    def _exchange(self, x: ttnn.Tensor, *, pad: int, axis: int) -> ttnn.Tensor:
        """Prepend the previous device-along-``axis``'s trailing ``pad`` rows of ``x`` (dim 1)."""
        num_links = max(1, min(int(x.shape[0]), self.ccl_manager.num_links))
        return self.ccl_manager.neighbor_pad_persistent_buffer(
            x,
            dims=[1],
            pad_left=[pad],
            pad_right=[0],
            padding_mode="zeros",
            axes=[axis],
            neighbor_sems=[self.ccl_manager.get_np_ping_pong_semaphore(axis)],
            num_links=[num_links],
        )

    def blend_wave(
        self,
        pixels: ttnn.Tensor,
        *,
        grid_rows: int,
        grid_cols: int,
        y_overlaps: list[int],
        x_overlaps: list[int],
        chunks_per_wave: int,
    ) -> ttnn.Tensor:
        """``(1, C, F, H, W)`` ROW_MAJOR raw tiles in, blended fp32 tiles out, same shape."""
        _, channels, frames, tile_h, tile_w = (int(d) for d in pixels.shape)
        hpad = max(y_overlaps) if y_overlaps else 0
        wpad = max(x_overlaps) if x_overlaps else 0
        mv, nh = (
            self._geometry_weights(
                grid_rows=grid_rows,
                grid_cols=grid_cols,
                y_overlaps=y_overlaps,
                x_overlaps=x_overlaps,
                chunks_per_wave=chunks_per_wave,
                tile_h=tile_h,
                tile_w=tile_w,
                hpad=hpad,
                wpad=wpad,
            )
            if hpad or wpad
            else (None, None)
        )

        def tiled_f32(t: ttnn.Tensor) -> ttnn.Tensor:
            return ttnn.typecast(ttnn.to_layout(t, ttnn.TILE_LAYOUT), ttnn.float32)

        x = ttnn.reshape(pixels, (channels * frames, tile_h, tile_w))

        if hpad:
            stacked = self._exchange(x, pad=hpad, axis=0)
            v_t = ttnn.matmul(tiled_f32(ttnn.permute(stacked, (0, 2, 1))), mv, compute_kernel_config=self._compute)
            v = ttnn.permute(ttnn.to_layout(v_t, ttnn.ROW_MAJOR_LAYOUT), (0, 2, 1))
        else:
            v = ttnn.typecast(ttnn.to_layout(x, ttnn.TILE_LAYOUT), ttnn.float32)
            v = ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT)

        if not wpad:
            return ttnn.reshape(v, (1, channels, frames, tile_h, tile_w))

        halo_t = self._exchange(ttnn.permute(x, (0, 2, 1)), pad=wpad, axis=1)
        halo_t = ttnn.slice(halo_t, [0, 0, 0], [channels * frames, wpad, tile_h])
        halo = ttnn.permute(halo_t, (0, 2, 1))
        halo = ttnn.to_layout(
            ttnn.typecast(ttnn.to_layout(halo, ttnn.TILE_LAYOUT), ttnn.float32), ttnn.ROW_MAJOR_LAYOUT
        )
        stacked = ttnn.concat([halo, v], dim=2)
        out = ttnn.matmul(ttnn.to_layout(stacked, ttnn.TILE_LAYOUT), nh, compute_kernel_config=self._compute)
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(out, (1, channels, frames, tile_h, tile_w))


def unpatchify_device(
    tokens: ttnn.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    out_channels: int = 3,
    patch_size: int = 16,
    patch_size_t: int = 4,
) -> ttnn.Tensor:
    """Device mirror of `decoder_minimax_h3.unpatchify`: tokens to `(1, C, T*pt, H*p, W*p)`.

    `ttnn.permute` handles the 8-dimensional `(B,T,H,W,C,pt,p,p) -> (B,C,T,pt,H,p,W,p)` permutation
    directly, which is what makes keeping the tiles on device possible at all.
    """
    batch = tokens.shape[0]
    tokens = ttnn.slice(tokens, [0, 0, 0], [batch, num_frames * height * width, tokens.shape[-1]])
    tokens = ttnn.reshape(
        tokens, (batch, num_frames, height, width, out_channels, patch_size_t, patch_size, patch_size)
    )
    tokens = ttnn.permute(tokens, (0, 4, 1, 5, 2, 6, 3, 7))
    return ttnn.reshape(
        tokens, (batch, out_channels, num_frames * patch_size_t, height * patch_size, width * patch_size)
    )
