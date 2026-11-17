# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Panoptic-DeepLab 1x1 convolutions on the generated genify_matmul (#65150).

With TT_GENIFY_MATMUL=1 (models/experimental/op_genify/genify_matmul_adapter.py) the conv helpers of
TtBottleneck, TtASPP and TtDeepLabV3PlusHead (TtPanopticDeepLabInsEmbedHead inherits it) build a
GenifyConv1x1 instead of a TtConv2d for every 1x1 convolution; every other convolution keeps TtConv2d.
Unset, `create_conv_layer` returns the stock TtConv2d, so the stock path is unchanged.

A 1x1 conv is `relu?(A @ W + b)` on the flattened activation A = [1, 1, N*Ho*Wo, Cin]:
  * weight W = conv weight[:, :, 0, 0].T as a [Cin, Cout] TILE tensor in DRAM, in the conv's weights
    dtype (bfp8), prepared once at build time from the same host tensors the stock conv reads;
  * bias b = [1, Cout] TILE in the weights dtype (the stock conv also prepares its bias in that dtype);
  * the stock compute config (LoFi, fp32 DEST, approx) and output dtype (bfp8).
A is read where it lives: an L1 height- or block-sharded activation in place (the output keeps A's
sharding, as the stock conv's does), an interleaved one (DRAM or L1) through the interleaved path
(the output goes to L1 interleaved). A sharded layout genify_matmul rejects is resharded to one it
accepts, with a warning; nothing falls back to the stock conv.
A stride-2 1x1 conv (the res3.0 / res4.0 shortcuts) subsamples A first, in TILE layout:
  * rows: a tile-aligned slice of a [N, Ho, 2W, C] view keeps the even image rows;
  * columns: a selection matmul on genify_matmul, S' @ group for every group of SUBSAMPLE_GROUP input
    pixels (S'[j, 2j] = 1, a batched B), which copies each kept pixel exactly (see SUBSAMPLE_COMPUTE).
A stays in the activation's dtype (bfp8): the kept values are the input's own values. Both ops are glue;
the profile reports them apart from the conv's matmul. A row-major or oddly shaped input takes the
generic path instead: untilize, strided row-major slice, tilize (bf16; 115 vs 10 us on res3.0).
"""

import torch
import ttnn
from loguru import logger

from models.experimental.op_genify import genify_matmul_adapter as genify
from models.tt_cnn.tt.builder import TtConv2d, to_compute_config

TILE = 32

# Stride-2 column subsample = a selection matmul over groups of SUBSAMPLE_GROUP input pixels (a multiple
# of 32 * stride, so the kept pixels of a group fill whole tiles). Device time after the 17.6 us rows
# slice on res3.0 (res4.0), models/experimental/op_genify/integration/pdl_test_subsample.py:
#   untilize + strided row-major slice + tilize  115 (42) us
#   selection over a whole image row (K = W)     22.7 (7.5) us at HiFi4, 16.5 (6.3) us at HiFi2
#   selection over groups of 64 pixels (K = 64)  9.7 (4.8) us at HiFi2  <- used; stock's I2S + halo + move: 36 (19)
# HiFi2 is exact (checked bit for bit; LoFi is not): S = 1.0 is in0, which sits in SrcB, whose top bits
# every fidelity uses whole; the activation sits in SrcA, whose 8 significant bits take 2 phases. With
# one nonzero product per output the fp32 sum is exact.
SUBSAMPLE_GROUP = 64
SUBSAMPLE_COMPUTE = ttnn.ComputeConfigDescriptor(
    math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True, math_approx_mode=False
)


def is_genify_conv(configuration) -> bool:
    """True when this conv runs on genify_matmul: the switch is on and the conv is a plain 1x1 conv."""
    return (
        genify.enabled()
        and tuple(configuration.kernel_size) == (1, 1)
        and all(p == 0 for p in tuple(configuration.padding))
        and configuration.groups == 1
    )


def create_conv_layer(configuration, device, conv_path=None):
    """The conv object a PDL helper builds: GenifyConv1x1 for a 1x1 conv when the switch is on, else TtConv2d."""
    if is_genify_conv(configuration):
        return GenifyConv1x1(configuration, device, conv_path=conv_path)
    return TtConv2d(configuration, device)


def _activation(act):
    if act is None:
        return None
    if act.op_type == ttnn.UnaryOpType.RELU:
        return "relu"
    raise NotImplementedError(f"genify 1x1 conv: no genify_matmul epilogue for activation {act}")


def _describe(t):
    """Short placement string for logs: e.g. 'L1 HEIGHT_SHARDED [320, 128] x103' or 'DRAM INTERLEAVED'."""
    mc = t.memory_config()
    where = "L1" if mc.buffer_type == ttnn.BufferType.L1 else "DRAM"
    layout = str(mc.memory_layout).split(".")[-1]
    desc = f"{where} {layout} {t.dtype} {list(t.shape)}"
    if mc.shard_spec is not None:
        spec = mc.shard_spec
        box = spec.grid.bounding_box()
        orient = "RM" if spec.orientation == ttnn.ShardOrientation.ROW_MAJOR else "CM"
        desc += (
            f" shard {list(spec.shape)} on {spec.grid.num_cores()} cores "
            f"(box {box.grid_size().x}x{box.grid_size().y}, {orient})"
        )
    return desc


def _in_l1(t):
    return t.memory_config().buffer_type == ttnn.BufferType.L1


def _same_buffer(a, b):
    try:
        return a.buffer_address() == b.buffer_address()
    except Exception:  # noqa: BLE001 - per-core allocations have no single address; treat as distinct
        return False


class GenifyConv1x1:
    """A 1x1 conv (bias, optional fused ReLU, any stride) on genify_matmul, with TtConv2d's __call__ contract."""

    def __init__(self, configuration, device, conv_path=None):
        cfg = configuration
        self.configuration = cfg
        self.device = device
        self.conv_path = conv_path or "conv1x1"
        if cfg.bias is None:
            raise ValueError(f"{self.conv_path}: genify 1x1 conv expects a bias (BN is fused into every PDL conv)")
        cin, cout = cfg.in_channels, cfg.out_channels
        if cin % TILE or cout % TILE:
            raise ValueError(f"{self.conv_path}: genify_matmul needs tile-aligned K and N, got {cin} -> {cout}")
        self.activation = _activation(cfg.activation)
        self.output_dtype = cfg.output_dtype
        # Same compute config the stock TtConv2d passes to ttnn.conv2d, as a ComputeConfigDescriptor.
        self.compute_config = genify.compute_config(to_compute_config(cfg, device))

        # Weights once, from the same host tensors the stock conv prepares: W = w[:, :, 0, 0].T, b = [1, Cout].
        w = ttnn.to_torch(cfg.weight).float()
        if tuple(w.shape) != (cout, cin, 1, 1):
            raise ValueError(f"{self.conv_path}: unexpected 1x1 weight shape {tuple(w.shape)}")
        self.weight = ttnn.from_torch(
            w.reshape(cout, cin).t().contiguous(),
            dtype=cfg.weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        b = ttnn.to_torch(cfg.bias).float().reshape(1, cout)
        self.bias = ttnn.from_torch(
            b, dtype=cfg.weights_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        sh, sw = tuple(cfg.stride)
        self.stride = (sh, sw)
        self.out_h = (cfg.input_height - 1) // sh + 1
        self.out_w = (cfg.input_width - 1) // sw + 1
        self.select = self._selection_matrix(device) if self.stride != (1, 1) else None
        self.calls = 0
        self.reshards = []  # (input placement, rejection, resharded placement), one per rejected call

    def _selection_matrix(self, device):
        """S' [N, groups, G / sw, G] with S'[j, sw * j] = 1 (G = SUBSAMPLE_GROUP), or None when the shape
        does not allow the TILE-layout subsample (then the generic row-major path runs)."""
        cfg = self.configuration
        n, h, w = cfg.batch_size, cfg.input_height, cfg.input_width
        sh, sw = self.stride
        g = SUBSAMPLE_GROUP
        if h != self.out_h * sh or w % TILE or w % g or g % (TILE * sw):
            return None
        s = torch.zeros(g // sw, g)
        s[torch.arange(g // sw), sw * torch.arange(g // sw)] = 1.0
        s = s.expand(n, self.out_h * (w // g), g // sw, g).contiguous()
        return ttnn.from_torch(
            s, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    # ------------------------------------------------------------------ input preparation

    def _flatten(self, x):
        """[N, H, W, C] or [1, 1, NHW, C] -> [1, 1, NHW, C] TILE (a view for tile-aligned interleaved / flat tensors)."""
        cfg = self.configuration
        m = cfg.batch_size * cfg.input_height * cfg.input_width
        a = x
        if list(a.shape) != [1, 1, m, cfg.in_channels]:
            a = ttnn.reshape(a, (1, 1, m, cfg.in_channels))
        if a.layout != ttnn.TILE_LAYOUT:
            # The stock 1x1 conv tilizes a row-major activation too (conv2d tilize_with_optional_deallocation).
            a = ttnn.to_layout(a, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        return a

    def _subsample(self, x):
        """Stride (sh, sw) 1x1 conv input: every sh-th image row and sw-th column -> [1, 1, N*Ho*Wo, C] TILE in L1."""
        cfg = self.configuration
        n, h, w, c = cfg.batch_size, cfg.input_height, cfg.input_width, cfg.in_channels
        sh, sw = self.stride
        ho, wo = self.out_h, self.out_w
        if self.select is not None and x.layout == ttnn.TILE_LAYOUT and not x.is_sharded():
            # Rows: in a [N, Ho, sh*W, C] view each group of sh image rows is one row block whose first W
            # matrix rows are the kept image row, so a tile-aligned slice keeps it (any dtype, no RM trip).
            xr = x
            if sh > 1:
                pairs = ttnn.reshape(x, (n, ho, sh * w, c))
                xr = ttnn.slice(pairs, [0, 0, 0, 0], [n, ho, w, c], memory_config=ttnn.L1_MEMORY_CONFIG)
            # Columns: S' @ each group of SUBSAMPLE_GROUP pixels (batched B) keeps every sw-th pixel exactly.
            g = SUBSAMPLE_GROUP
            groups = ttnn.reshape(xr, (n, ho * (w // g), g, c))
            a = genify.matmul(
                self.select,
                groups,
                dtype=x.dtype,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=SUBSAMPLE_COMPUTE,
            )
            if not _same_buffer(xr, x):
                ttnn.deallocate(xr)
            return ttnn.reshape(a, (1, 1, n * ho * wo, c))
        return self._subsample_row_major(x)

    def _subsample_row_major(self, x):
        """Generic stride subsample: untilize, strided row-major slice (stepped slices reject bfp8; the
        untilize gives bf16), tilize -> [1, 1, N*Ho*Wo, C] bf16 TILE in L1."""
        cfg = self.configuration
        n, h, w, c = cfg.batch_size, cfg.input_height, cfg.input_width, cfg.in_channels
        sh, sw = self.stride
        ho, wo = self.out_h, self.out_w
        x_in = x
        temps = []
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            temps.append(x)
        xr = ttnn.reshape(x, (n, h, w, c))
        x_rm = ttnn.to_layout(xr, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        if x_rm.dtype == ttnn.bfloat8_b:
            x_rm = ttnn.typecast(x_rm, ttnn.bfloat16)
        temps.append(x_rm)
        xs = ttnn.slice(x_rm, [0, 0, 0, 0], [n, h, w, c], [1, sh, sw, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        temps.append(xs)
        a = ttnn.to_layout(xs, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        a = ttnn.reshape(a, (1, 1, n * ho * wo, c))
        # Free the temporaries, never the caller's activation (a row-major input makes to_layout a no-op).
        for t in temps:
            if not _same_buffer(t, a) and not _same_buffer(t, x_in):
                ttnn.deallocate(t)
        return a

    def _accepted_shard_config(self, a):
        """A sharded placement genify_matmul accepts, close to A's: same grid and orientation, whole-tile shards."""
        mc = a.memory_config()
        spec = mc.shard_spec
        m_tiles = -(-a.shape[-2] // TILE)
        k_tiles = a.shape[-1] // TILE
        if mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            cores = spec.grid.num_cores()
            shard_m = -(-m_tiles // cores)
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(spec.grid, [shard_m * TILE, k_tiles * TILE], spec.orientation),
            )
        if mc.memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            box = spec.grid.bounding_box()
            cols, rows = box.grid_size().x, box.grid_size().y
            if spec.orientation != ttnn.ShardOrientation.ROW_MAJOR:
                cols, rows = rows, cols
            # K over grid columns, M over grid rows, the grid exactly covering A (R3's layout gate).
            while cols > 1 and -(-k_tiles // -(-k_tiles // cols)) != cols:
                cols -= 1
            shard_k = -(-k_tiles // cols)
            while rows > 1 and -(-m_tiles // -(-m_tiles // rows)) != rows:
                rows -= 1
            shard_m = -(-m_tiles // rows)
            start = box.start
            grid = ttnn.CoreRangeSet([ttnn.CoreRange(start, ttnn.CoreCoord(start.x + cols - 1, start.y + rows - 1))])
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [shard_m * TILE, shard_k * TILE], ttnn.ShardOrientation.ROW_MAJOR),
            )
        raise NotImplementedError(f"{self.conv_path}: no genify placement for {_describe(a)}")

    def _accept(self, a, x):
        """A as genify_matmul accepts it: unchanged when it validates, else resharded (never the stock conv)."""
        from ttnn.operations._op_contract import UnsupportedAxisValue
        from ttnn.operations.genify_matmul import validate

        if not a.is_sharded():
            return a
        try:
            validate(
                a,
                self.weight,
                bias=self.bias,
                activation=self.activation,
                compute_kernel_config=self.compute_config,
            )
            return a
        except UnsupportedAxisValue as e:
            mc = self._accepted_shard_config(a)
            before = _describe(a)
            resharded = ttnn.to_memory_config(a, mc)
            if not _same_buffer(a, x):
                ttnn.deallocate(a)
            a = resharded
            self.reshards.append((before, str(e), _describe(a)))
            logger.warning(
                f"[genify] {self.conv_path}: genify_matmul rejects A {before} ({e}); resharded to {_describe(a)}"
            )
            return a

    # ------------------------------------------------------------------ call

    def __call__(self, x, return_output_dim: bool = False):
        cfg = self.configuration
        x_desc = _describe(x)
        a = self._subsample(x) if self.stride != (1, 1) else self._flatten(x)
        a = self._accept(a, x)
        out = genify.matmul(
            a,
            self.weight,
            bias=self.bias,
            activation=self.activation,
            dtype=self.output_dtype,
            memory_config=None if a.is_sharded() else ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.compute_config,
        )
        if self.calls == 0:
            logger.info(
                f"[genify] {self.conv_path}: M={a.shape[-2]} K={cfg.in_channels} N={cfg.out_channels} "
                f"stride={self.stride} act={self.activation} | in {x_desc} | A {_describe(a)} | out {_describe(out)}"
            )
        self.calls += 1
        # Free what the stock conv frees: its own temporaries, and the activation when deallocate_activation
        # is set and the activation is not in DRAM (conv2d never frees a DRAM input).
        if not _same_buffer(a, x):
            ttnn.deallocate(a)
        if cfg.deallocate_activation and _in_l1(x):
            ttnn.deallocate(x)
        if return_output_dim:
            return out, (self.out_h, self.out_w)
        return out
