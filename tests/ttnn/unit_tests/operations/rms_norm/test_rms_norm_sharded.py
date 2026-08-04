# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 acceptance: sharded placements.

Pins the three things the golden suite can pass *without*, and which nothing
else in the tree would catch:

1. **Nativeness, checked against the DATAFLOW rather than the output values.**
   A local shard read back through a `TensorAccessor` produces the right answer
   (each core happens to hold the bytes it needs), so a golden cell goes green
   either way.  `test_sharded_input_cb_is_zero_copy` asserts on the
   ProgramDescriptor instead: the x / out CBs must be tensor-BACKED, which is
   only true of `ttnn.cb_descriptor_from_sharded_tensor`.
2. **The cross-core width combine's edge geometries** — a ragged width tail
   (`Wt` not a multiple of the shard's tile width), a group spanning more than
   one physical grid row, and a many-group BLOCK grid.
3. **`GRID_W > 1` on an interleaved input** — the Lamp-L1 knob whose guard this
   refinement deleted.  It ships at its byte-identical 1, so without this test
   the whole interleaved width-split path would be dead code.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod
from eval.sharding import auto_shard_config
from ttnn.operations.rms_norm import rms_norm

_ML = ttnn.TensorMemoryLayout


def _cfg(fp32_acc=True, fidelity=ttnn.MathFidelity.HiFi4):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = fidelity
    c.fp32_dest_acc_en = fp32_acc
    c.math_approx_mode = False
    return c


def _ref(x, gamma=None, eps=1e-6):
    xf = x.to(torch.float32)
    out = xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def _check(got, expected, *, pcc_min=0.995, rms_max=0.04, label=""):
    a, e = got.float().flatten(), expected.float().flatten()
    pcc = torch.corrcoef(torch.stack([a, e]))[0, 1].item()
    rms = ((a - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
    assert pcc >= pcc_min and rms <= rms_max, f"{label}: pcc={pcc:.6f} (min {pcc_min}) rms={rms:.5f} (max {rms_max})"
    return pcc, rms


def _run(
    device,
    shape,
    memory_layout,
    *,
    layout=ttnn.TILE_LAYOUT,
    gamma=True,
    fp32_acc=True,
    poison=None,
    gamma_layout=None,
):
    """Build a legal shard for `shape`, run the op, return (got, expected)."""
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    mc = None
    if memory_layout != _ML.INTERLEAVED:
        mc = auto_shard_config(list(shape), memory_layout, layout=layout, dtype=ttnn.bfloat16, device=device)
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc)

    gd = g = None
    if gamma:
        gl = gamma_layout if gamma_layout is not None else layout
        g = torch.randn(shape[-1], dtype=torch.bfloat16)
        gt = g.reshape(1, 1, 1, -1) if gl == ttnn.TILE_LAYOUT else g
        gd = ttnn.from_torch(gt, dtype=ttnn.bfloat16, layout=gl, device=device)

    if poison is not None:
        # The reference is built from the LOGICAL tensor, so a padded-width reduce
        # (or a pad-tile leak on a ragged width shard) diverges hard instead of by
        # a sub-noise margin.
        xd = ttnn.fill_implicit_tile_padding(xd, poison)
        if gd is not None and layout == ttnn.TILE_LAYOUT:
            gd = ttnn.fill_implicit_tile_padding(gd, poison)

    out = rms_norm(
        xd,
        gamma=gd,
        compute_kernel_config=_cfg(fp32_acc),
        memory_config=(xd.memory_config() if mc is not None else None),
    )
    assert out.memory_config().memory_layout == (memory_layout if mc is not None else out.memory_config().memory_layout)
    return ttnn.to_torch(out), _ref(x, g)


# ---------------------------------------------------------------------------
# 1. Nativeness — asserted on the descriptor, not on the output
# ---------------------------------------------------------------------------

_NATIVE_SHAPES = [
    ((1, 1, 256, 512), _ML.HEIGHT_SHARDED, "height_8_cores"),
    ((1, 1, 32, 2048), _ML.WIDTH_SHARDED, "width_64_cores_2_grid_rows"),
    ((1, 1, 256, 512), _ML.BLOCK_SHARDED, "block_8x8"),
]


@pytest.mark.parametrize("shape,memory_layout,label", _NATIVE_SHAPES, ids=[c[2] for c in _NATIVE_SHAPES])
def test_sharded_input_cb_is_zero_copy(device, shape, memory_layout, label):
    """A TILE shard must be consumed IN PLACE, never re-read through an accessor.

    `CBDescriptor.has_buffer()` is true only for a tensor-backed (globally
    allocated) CB, i.e. exactly what `cb_descriptor_from_sharded_tensor` builds.
    This is the assertion the golden suite cannot make: the accessor path
    produces identical numbers, so only the descriptor distinguishes them.
    """
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    mc = auto_shard_config(list(shape), memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mc)
    od = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, mc)
    gd = ttnn.from_torch(
        torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    desc = pdmod.create_program_descriptor(xd, od, gamma=gd, compute_kernel_config=_cfg())
    backed = {}
    for cb in desc.cbs:
        for fd in cb.format_descriptors:
            backed[fd.buffer_index] = cb.has_buffer()

    assert backed.get(pdmod.CB_INPUT_TILES) is True, (
        f"{label}: cb_input_tiles is NOT backed on the input shard -- the op is reading a "
        f"local shard through a TensorAccessor, which tolerates the placement instead of "
        f"implementing it (op_requirements.md Refinement 2, 'Native or nothing')"
    )
    assert backed.get(pdmod.CB_OUTPUT_TILES) is True, f"{label}: cb_output_tiles is not backed on the output shard"


def test_interleaved_input_cb_is_not_backed(device):
    """The inverse guard: an interleaved build must NOT claim a backing buffer."""
    shape = (1, 1, 64, 128)
    torch.manual_seed(0)
    xd = ttnn.from_torch(
        torch.randn(*shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    od = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    desc = pdmod.create_program_descriptor(xd, od, compute_kernel_config=_cfg())
    for cb in desc.cbs:
        assert not cb.has_buffer(), "an interleaved build must allocate every CB in the arena"


# ---------------------------------------------------------------------------
# 2. Correctness across the three schemes and their edge geometries
# ---------------------------------------------------------------------------

# (shape, id) -- chosen for the geometry they force, not for size:
#   ragged tails, groups spanning >1 physical grid row, many groups, tiny grids,
#   H- and W-non-aligned, and a tall shard that forces multiple row-blocks.
_SCHEME_SHAPES = [
    ((1, 1, 256, 512), "aligned_8_or_64_cores"),
    ((1, 1, 224, 72), "w_non_aligned_prime_rows"),
    ((1, 1, 17, 50), "h_and_w_non_aligned_tiny"),
    ((1, 1, 32, 4064), "ragged_width_tail_wt127"),
    ((1, 1, 32, 7168), "ragged_width_tail_wt224"),
    ((1, 1, 3232, 96), "tall_shard_many_row_blocks"),
    ((3, 5, 224, 736), "rank4_multibatch_rt105"),
    ((7, 224, 3072), "rank3_wide"),
    ((1024, 1024), "rank2_square"),
]


@pytest.mark.parametrize(
    "memory_layout",
    [_ML.HEIGHT_SHARDED, _ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED],
    ids=["height", "width", "block"],
)
@pytest.mark.parametrize("shape,label", _SCHEME_SHAPES, ids=[c[1] for c in _SCHEME_SHAPES])
def test_sharded_tile_schemes(device, shape, label, memory_layout):
    got, expected = _run(device, shape, memory_layout)
    _check(got, expected, label=f"{label}/{memory_layout}")


@pytest.mark.parametrize(
    "memory_layout",
    [_ML.INTERLEAVED, _ML.HEIGHT_SHARDED, _ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED],
    ids=["interleaved", "height", "width", "block"],
)
@pytest.mark.parametrize("shape,label", _SCHEME_SHAPES[:4], ids=[c[1] for c in _SCHEME_SHAPES[:4]])
def test_sharded_row_major(device, shape, label, memory_layout):
    """ROW_MAJOR activations under every placement.

    INTERLEAVED and HEIGHT go through SCHEME_ROWS (a height shard spans the full
    row, so the tensor's page IS the stick and the accessor addresses rows
    exactly).  WIDTH and BLOCK are Refinement 2b's BAND scheme: their page is a
    row SEGMENT, so no accessor read can reach a row -- each core stages the band
    it already holds out of its own L1 instead.
    """
    got, expected = _run(device, shape, memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT)
    _check(got, expected, label=f"rm/{label}/{memory_layout}")


# ---------------------------------------------------------------------------
# 2b. The ROW_MAJOR BAND scheme (Refinement 2b)
# ---------------------------------------------------------------------------

# Shapes chosen for the BAND GEOMETRY they force, which is what this scheme turns
# on -- the shard granule is L1_align/elem_size (8 for bf16, 4 for fp32), so the
# per-core band is generally NOT a whole tile column and generally does NOT start
# on one.  auto_shard_config picks the geometry, so these are the shapes whose
# geometry is interesting:
#   W=512  bf16 WIDTH -> [.., 8]   : band = 1/4 of a tile column, delta 0/8/16/24
#   W=512  bf16 BLOCK -> [.., 48]  : band spans TWO tile columns, delta 0 or 16
#   W=3072 bf16 WIDTH -> [.., 32]  : band == exactly one tile column, delta 0
#                                    (the contiguous fast path: one read per tile-row)
#   W=1000 / 50 / 47   : non-tile-aligned W, so the last band is short as well
#   W=64   BLOCK       : tiny width, many row blocks
_BAND_SHAPES = [
    ((1, 1, 256, 512), "sub_tile_band_delta_cycles"),
    ((1, 1, 224, 3072), "band_is_one_whole_tile_column"),
    ((1, 1, 224, 1000), "w_non_aligned_short_last_band"),
    ((1, 1, 32, 50), "tiny_w_non_aligned"),
    ((4, 8, 32, 47), "rank4_multibatch_w_non_aligned"),
    ((1, 1, 3232, 96), "very_tall_many_row_blocks"),
    ((7136, 736), "rank2_tall_92_band_group"),
    ((1, 224, 11008), "rank3_wide_ffn"),
]


@pytest.mark.parametrize(
    "memory_layout",
    [_ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED],
    ids=["width", "block"],
)
@pytest.mark.parametrize("shape,label", _BAND_SHAPES, ids=[c[1] for c in _BAND_SHAPES])
def test_row_major_width_band(device, shape, label, memory_layout):
    """An RM shard that cuts the WIDTH axis is ANSWERED, not refused.

    Before Refinement 2b these two cells were op-side EXCLUSIONS: the shard's page
    is a row SEGMENT, so a stick read keyed on the page index landed inside one
    segment and ran off the end of the shard (measured PCC 0.005).  The BAND
    scheme never reads a row -- the cross-core combine sums per-row PARTIALS
    elementwise, so each core reduces only the band it already holds.
    """
    got, expected = _run(device, shape, memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT)
    _check(got, expected, label=f"band/{label}/{memory_layout}")


@pytest.mark.parametrize(
    "memory_layout",
    [_ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED],
    ids=["width", "block"],
)
@pytest.mark.parametrize("gamma_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["g_rm", "g_tile"])
def test_row_major_width_band_gamma_layouts(device, memory_layout, gamma_layout):
    """gamma at BOTH layouts on the BAND scheme -- the alignment regression net.

    The band is staged in the tensor's GLOBAL tile frame precisely so gamma stays
    fetchable: a DRAM read whose source offset is not 64-byte aligned is SILENTLY
    TRUNCATED to the alignment, so staging at the band's own byte offset handed
    three cores in four the wrong weights (measured: bands 1..3 of an 8-element
    shard all received gamma[0..8), PCC 0.32 with a gamma that is 1.0 everywhere
    it was checked -- i.e. invisible to a spot check).  A random gamma makes the
    mismatch a whole-tensor PCC collapse.
    """
    got, expected = _run(
        device, (1, 1, 256, 512), memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT, gamma_layout=gamma_layout
    )
    _check(got, expected, label=f"band_gamma/{gamma_layout}/{memory_layout}")


@pytest.mark.parametrize(
    "memory_layout",
    [_ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED],
    ids=["width", "block"],
)
def test_row_major_width_band_is_local(device, memory_layout):
    """The band must be read from THIS core's own L1, never through an accessor.

    Asserted on the descriptor, because the golden suite cannot see the
    difference: an accessor path that happened to address the right segments
    would produce identical numbers.  The reader's BAND compile-time arg is the
    switch between "stage from x_addr + local_stick * shard_row_bytes" (local L1,
    zero DRAM traffic) and the interleaved accessor dataflow.
    """
    shape = (1, 1, 256, 512)
    mc = auto_shard_config(list(shape), memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    xd = ttnn.from_torch(
        torch.randn(*shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mc,
    )
    od = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, mc)
    gd = ttnn.from_torch(
        torch.randn(shape[-1], dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    desc = pdmod.create_program_descriptor(xd, od, gamma=gd, compute_kernel_config=_cfg())
    reader = next(k for k in desc.kernels if "reader" in k.kernel_source)
    writer = next(k for k in desc.kernels if "writer" in k.kernel_source)
    assert reader.compile_time_args[pdmod.READER_CT_BAND] == 1, (
        f"{memory_layout}: the reader is NOT on the BAND path -- an RM width shard is being read "
        f"through a TensorAccessor, which cannot address a row at all (its page is a row segment)"
    )
    assert (
        writer.compile_time_args[pdmod.WRITER_CT_OUT_SHARD_ROW_BYTES] != 0
    ), f"{memory_layout}: the writer is not writing back into the resident output shard"


@pytest.mark.parametrize(
    "memory_layout",
    [_ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED],
    ids=["width", "block"],
)
@pytest.mark.parametrize(
    "out_mc,out_id",
    [(ttnn.DRAM_MEMORY_CONFIG, "dram"), (ttnn.L1_MEMORY_CONFIG, "l1_interleaved")],
    ids=["out_dram", "out_l1_interleaved"],
)
def test_row_major_width_band_non_matching_output(device, memory_layout, out_mc, out_id):
    """A BAND input with an output placement that is NOT the input's shard.

    Then the band cannot be written in place, and the writer addresses the output
    through the accessor at the band's byte offset instead -- legal exactly because
    an interleaved ROW_MAJOR page IS a whole stick.  Nothing else covers this: the
    golden harness always requests an output shard matching the input's.
    """
    shape = (1, 1, 256, 512)
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    g = torch.randn(shape[-1], dtype=torch.bfloat16)
    mc = auto_shard_config(list(shape), memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device)
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    gd = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(xd, gamma=gd, compute_kernel_config=_cfg(), memory_config=out_mc))
    _check(out, _ref(x, g), label=f"band_out/{out_id}/{memory_layout}")


@pytest.mark.parametrize(
    "memory_layout",
    [_ML.INTERLEAVED, _ML.HEIGHT_SHARDED, _ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED],
    ids=["interleaved", "height", "width", "block"],
)
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32, 40), (1, 1, 32, 72), (1, 1, 32, 200), (1, 1, 224, 72), (1, 1, 40, 40)],
    ids=["wt2", "wt3", "wt7", "wt3_manyrows", "wt2_h_and_w"],
)
def test_sharded_pad_poison(device, shape, memory_layout):
    """A narrow, POISONED width under every placement.

    One tile of padding is 11-38% of these rows, so folding it into the reduce is
    a 6-27% error rather than a sub-noise one -- and the poison makes a leak
    catastrophic. This is the only test that can catch a masked-reduce regression
    on the width-split path, where the partial-W mask has to land on ONE core.
    """
    got, expected = _run(device, shape, memory_layout, fp32_acc=False, poison=1000.0)
    _check(got, expected, label=f"poison/{shape}/{memory_layout}")


@pytest.mark.parametrize(
    "memory_layout",
    [_ML.HEIGHT_SHARDED, _ML.WIDTH_SHARDED, _ML.BLOCK_SHARDED],
    ids=["height", "width", "block"],
)
def test_sharded_no_gamma(device, memory_layout):
    got, expected = _run(device, (1, 1, 256, 512), memory_layout, gamma=False)
    _check(got, expected, label=f"no_gamma/{memory_layout}")


def test_sharded_wide_w_keeps_the_reduce_datapath(device):
    """D8: a resident shard must not switch the AccumulateViaAdd precision fix off.

    A HEIGHT-sharded (1,1,160,11008) holds a 344-tile shard, which squeezes
    WT_CHUNK far below D7's crossover of 4.  Gating the datapath on WT_CHUNK
    instead of the whole reduce dim brought Refinement 1b's rms 0.127 straight
    back on exactly the cells 1b closed, so this is the regression net for D8.
    """
    got, expected = _run(device, (1, 1, 160, 11008), _ML.HEIGHT_SHARDED, fp32_acc=False)
    _check(got, expected, label="wide_w_height_shard")


# ---------------------------------------------------------------------------
# 3. GRID_W — the interleaved width split (Lamp L1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("grid_w", [1, 2, 8], ids=["gw1_default", "gw2", "gw8"])
@pytest.mark.parametrize("shape", [(1, 1, 32, 2048), (1, 1, 96, 1024)], ids=["rt1_decode", "rt3"])
def test_interleaved_width_split_knob(device, shape, grid_w):
    """`GRID_W > 1` runs the SAME combine on an interleaved input.

    This refinement deleted the knob's NotImplementedError guard; it ships at its
    byte-identical 1, so this is what keeps the path live rather than dead code.
    Refinement 3 turns it to fill the grid on an `Rt = 1` decode profile.
    """
    saved = pdmod.GRID_W
    try:
        pdmod.GRID_W = grid_w
        got, expected = _run(device, shape, _ML.INTERLEAVED, fp32_acc=False)
        _check(got, expected, label=f"grid_w={grid_w}/{shape}")
    finally:
        pdmod.GRID_W = saved
