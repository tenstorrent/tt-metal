# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""g6_clone over every SUPPORTED (dtype, layout) pair, aligned and partial-tile shapes.

The op has no compute kernel and no compute_kernel_config, so this matrix is also its precision
matrix: the only precision axis is the dtype, and every cell is graded by bit equality of the
logical elements. Partial-tile TILE shapes run `logical_rows_partial_tile`; the shape list covers
each of its four page kinds, a last tile column of <= 16 and > 16 logical columns, a last tile row
of <= 16 and > 16 logical rows, and several images so the walk's tile row wraps.
"""

import importlib

import pytest
import torch
import ttnn

from ttnn.operations.g6_clone import g6_clone

op_module = importlib.import_module("ttnn.operations.g6_clone.g6_clone")
descriptor_module = importlib.import_module("ttnn.operations.g6_clone.g6_clone_program_descriptor")

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.int32: torch.int32,
    ttnn.uint32: torch.uint32,
}
_BITS_DTYPE = {2: torch.int16, 4: torch.int32}

DTYPES = [
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, id="fp32"),
    pytest.param(ttnn.int32, id="int32"),
    pytest.param(ttnn.uint32, id="uint32"),
]
LAYOUTS = [
    pytest.param(ttnn.TILE_LAYOUT, id="tile"),
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="rm"),
]
SHAPES = [
    pytest.param((1, 1, 32, 32), id="32x32_aligned"),
    pytest.param((1, 1, 64, 128), id="64x128_aligned"),
    pytest.param((2, 3, 96, 160), id="2x3x96x160_aligned_multi_image"),
    pytest.param((1, 1, 32, 48), id="32x48_W_non_aligned_cols16"),
    pytest.param((1, 1, 48, 32), id="48x32_H_non_aligned_rows16"),
    pytest.param((1, 1, 48, 48), id="48x48_both_non_aligned"),
    pytest.param((3, 40, 50), id="3x40x50_all_kinds_rows8_cols18"),
    pytest.param((5, 17, 33), id="5x17x33_rows17_cols1"),
    pytest.param((2, 1, 3000), id="2x1x3000_one_row_images"),
    pytest.param((1, 5120), id="1x5120_rank2_one_row"),
    pytest.param((37,), id="37_rank1"),
]


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(_BITS_DTYPE[tensor.element_size()])


def _random_bits(shape, torch_dtype) -> torch.Tensor:
    bits_dtype = _BITS_DTYPE[torch_dtype.itemsize]
    info = torch.iinfo(bits_dtype)
    return torch.randint(info.min, info.max + 1, tuple(shape), dtype=torch.int64).to(bits_dtype).view(torch_dtype)


def _upload(torch_input: torch.Tensor, device, dtype, layout) -> ttnn.Tensor:
    # preserve_nan_values builds the tile layout on the host, so every bit pattern reaches DRAM.
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        preserve_nan_values=True,
    )
    if not torch.equal(_bits(ttnn.to_torch(ttnn_input)), _bits(torch_input)):
        pytest.fail("HARNESS PRECONDITION (not the op): the uploaded input does not hold the host bits")
    return ttnn_input


def _assert_clone(ttnn_output, ttnn_input, torch_input, dtype, layout) -> None:
    assert ttnn.is_tensor_storage_on_device(ttnn_output)
    assert list(ttnn_output.shape) == list(torch_input.shape), f"shape {list(ttnn_output.shape)}"
    assert ttnn_output.dtype == dtype and ttnn_output.layout == layout
    out_mc = ttnn_output.memory_config()
    assert out_mc.buffer_type == ttnn.BufferType.DRAM
    assert out_mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert ttnn_output.buffer_address() != ttnn_input.buffer_address(), "output aliases the input buffer"
    got, want = _bits(ttnn.to_torch(ttnn_output)), _bits(torch_input)
    if not torch.equal(got, want):
        differ = got != want
        first = tuple(int(i) for i in differ.nonzero()[0])
        pytest.fail(f"{int(differ.sum())} of {differ.numel()} logical elements differ; first at {first}")


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_g6_clone_layout_matrix(device, dtype, layout, shape):
    torch.manual_seed(42)
    torch_input = _random_bits(shape, _TORCH_DTYPE[dtype])
    ttnn_input = _upload(torch_input, device, dtype, layout)
    ttnn_output = g6_clone(ttnn_input)
    _assert_clone(ttnn_output, ttnn_input, torch_input, dtype, layout)


def _poisoned(torch_input: torch.Tensor, device, dtype, poison_bits: int):
    """x at its logical shape over a buffer whose tile padding holds `poison_bits`; returns
    (view, backing), and the caller keeps backing alive while the view is used."""
    logical = list(torch_input.shape)
    padded = logical[:-2] + [-(-logical[-2] // 32) * 32, -(-logical[-1] // 32) * 32]
    full_bits = torch.full(padded, poison_bits, dtype=_BITS_DTYPE[torch_input.element_size()])
    full_bits[..., : logical[-2], : logical[-1]] = _bits(torch_input)
    backing = _upload(full_bits.view(torch_input.dtype), device, dtype, ttnn.TILE_LAYOUT)
    view = ttnn.reshape(backing, ttnn.Shape(logical), ttnn.Shape(padded))
    if view.buffer_address() != backing.buffer_address():
        pytest.fail("HARNESS PRECONDITION (not the op): the poisoned view must share the padded buffer")
    return view, backing


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((3, 40, 50), id="3x40x50_all_kinds"),
        pytest.param((5, 17, 33), id="5x17x33_rows17_cols1"),
        pytest.param((1, 1, 1, 62080), id="1x1x1x62080_q8"),
    ],
)
@pytest.mark.parametrize(
    "dtype,poison_bits",
    [
        pytest.param(ttnn.bfloat16, 0x7FC0, id="bf16_nan"),
        pytest.param(ttnn.float32, 0x7FC00001, id="fp32_nan"),
        pytest.param(ttnn.int32, -0x5A5A5A5B, id="int32_pattern"),
    ],
)
def test_g6_clone_poisoned_padding_kinds(device, shape, dtype, poison_bits):
    """No logical element of any page kind depends on the input's tile padding."""
    torch.manual_seed(42)
    torch_input = _random_bits(shape, _TORCH_DTYPE[dtype])
    view, _backing = _poisoned(torch_input, device, dtype, poison_bits)
    assert op_module.g6_clone_plan(view)["regime"] == descriptor_module.REGIME_LOGICAL_ROWS
    ttnn_output = g6_clone(view)
    _assert_clone(ttnn_output, view, torch_input, dtype, ttnn.TILE_LAYOUT)


@pytest.mark.parametrize(
    "shape,dtype,layout,regime,kind_pieces",
    [
        # One logical row per bf16 tile: face rows 0 of faces 0 and 1 move as one span through the
        # 448 B gap (PIECE_MERGE_GAP_BYTES).
        ((1, 1, 1, 62080), ttnn.bfloat16, ttnn.TILE_LAYOUT, "logical_rows_partial_tile", (1, [(0, 576)])),
        ((1, 5120), ttnn.bfloat16, ttnn.TILE_LAYOUT, "logical_rows_partial_tile", (1, [(0, 576)])),
        # The f32 gap (960 B) stays split.
        ((1, 1, 1, 62080), ttnn.float32, ttnn.TILE_LAYOUT, "logical_rows_partial_tile", (1, [(0, 64), (1024, 64)])),
        # The one page of [1, 1] is the last tile row and column (kind 3).
        ((1, 1), ttnn.float32, ttnn.TILE_LAYOUT, "logical_rows_partial_tile", (3, [(0, 64)])),
        # Full pages run bank_affine_runs (11x10 Blackhole grid, 8 banks, dram_rows groups).
        ((1, 1, 32, 62080), ttnn.bfloat16, ttnn.TILE_LAYOUT, "bank_affine_runs", None),
        ((16, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT, "bank_affine_runs", None),
        ((16, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, "bank_affine_runs", None),
        # Every full-page plan runs bank_affine_runs on up to BANK_AFFINE_CORES_PER_BANK cores per
        # bank; a one-page plan is one core on bank 0's row.
        ((40, 32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, "bank_affine_runs", None),
        ((1,), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT, "bank_affine_runs", None),
    ],
)
def test_g6_clone_plan_regime(device, shape, dtype, layout, regime, kind_pieces):
    """The plan reports the regime each input runs and, for logical rows, the pieces of one page
    kind (1: the image's last tile row; 3: also its last tile column)."""
    x = ttnn.from_torch(
        torch.zeros(shape, dtype=_TORCH_DTYPE[dtype]),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    plan = op_module.g6_clone_plan(x)
    assert plan["regime"] == regime, plan["regime"]
    walk = plan["partial_tile_walk"]
    if kind_pieces is None:
        assert walk is None
    else:
        kind, pieces = kind_pieces
        assert walk["kinds"][kind] == pieces, walk["kinds"]


def test_g6_clone_plan_bank_affine_rows(device):
    """bank_affine_runs under dram_rows: every core takes a run of one bank's pages, sits on that
    bank's anchor row on the anchor's side, at most BANK_AFFINE_CORES_PER_BANK per bank, and both
    kernels address that NoC's endpoints; the cores form one rectangle per bank (BB-17)."""
    x = ttnn.from_torch(
        torch.zeros((16, 12, 128, 128), dtype=torch.float32),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    plan = op_module.g6_clone_plan(x)
    assert plan["regime"] == descriptor_module.REGIME_BANK_AFFINE
    num_banks = plan["num_banks"]
    assert plan["page_stride"] == num_banks
    assert plan["bank_endpoint_noc"] == descriptor_module.BANK_AFFINE_ENDPOINT_NOC
    # Shipped route: one VC, reads on NoC0 and writes on NoC1 (Refinement 3 measured the knobs null or worse).
    assert (plan["bank_vc_directions"], plan["bank_noc_split"]) == ((), "none")
    endpoint_noc = ttnn.NOC.NOC_0 if descriptor_module.BANK_AFFINE_ENDPOINT_NOC == 0 else ttnn.NOC.NOC_1
    anchors = ttnn.device.get_optimal_dram_bank_to_logical_worker_assignment(device, endpoint_noc)
    bank_cores = {}
    for core, _, start_page, _ in plan["roles"]:
        anchor = anchors[start_page % num_banks]
        assert core.y == anchor.y and core.x >= anchor.x, (core, anchor)
        bank_cores[start_page % num_banks] = bank_cores.get(start_page % num_banks, 0) + 1
    assert set(bank_cores.values()) == {descriptor_module.BANK_AFFINE_CORES_PER_BANK}, bank_cores
    # Banks on the two DRAM columns may share an anchor row, which merges their runs.
    assert len(plan["core_ranges"].ranges()) <= num_banks
    assert plan["core_ranges"].num_cores() == plan["num_cores"]


@pytest.mark.parametrize("blocks_per_core", [1, 3])
@pytest.mark.parametrize(
    "shape,dtype,layout",
    [
        # logical_rows: the rotated walk resets PartialTileWalk where it wraps.
        ((1, 1, 1, 62080), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        ((4, 100, 1000), ttnn.float32, ttnn.TILE_LAYOUT),
        # flat_page_range (BANK_AFFINE_CORES_PER_BANK = 0).
        ((8, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),
        ((16, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_g6_clone_bank_stagger(device, monkeypatch, shape, dtype, layout, blocks_per_core):
    """FLAT_BANK_STAGGER (off by default) stays bit-exact: each core's consecutive-page walk starts
    `rotate` pages in and wraps once, also inside a block of the depth-2 ring."""
    monkeypatch.setattr(descriptor_module, "FLAT_BANK_STAGGER", True)
    monkeypatch.setattr(descriptor_module, "BANK_AFFINE_CORES_PER_BANK", 0)
    monkeypatch.setattr(descriptor_module, "BLOCKS_PER_CORE_KNOB", blocks_per_core)
    monkeypatch.setattr(op_module, "_DESCRIPTOR_CACHE", {})
    torch.manual_seed(7)
    torch_input = _random_bits(shape, _TORCH_DTYPE[dtype])
    ttnn_input = _upload(torch_input, device, dtype, layout)
    plan = op_module.g6_clone_plan(ttnn_input)
    assert plan["bank_stagger"] and plan["page_stride"] == 1, plan
    ttnn_output = g6_clone(ttnn_input)
    _assert_clone(ttnn_output, ttnn_input, torch_input, dtype, layout)


@pytest.mark.parametrize(
    "vc_directions,vc_cycle,noc_split,blocks_per_core",
    [
        (("read", "write"), (1, 2, 3), "none", 1),
        (("read",), (0, 1, 2, 3), "none", 1),
        ((), (1, 2, 3), "alternate", 1),
        ((), (1, 2, 3), "half", 1),
        (("read", "write"), (1, 2, 3), "half", 3),
    ],
)
@pytest.mark.parametrize(
    "shape,dtype,layout",
    [
        ((8, 12, 128, 128), ttnn.float32, ttnn.TILE_LAYOUT),
        # One or two pages per core: "half" moves 0 or 1 page on the other NoC, "alternate" the whole run.
        ((16, 3, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_g6_clone_bank_route_knobs(
    device, monkeypatch, shape, dtype, layout, vc_directions, vc_cycle, noc_split, blocks_per_core
):
    """The NoC route knobs of bank_affine_runs (off by default) stay bit-exact: per-core VCs on either
    direction, and a block split over both NoCs under DM_DYNAMIC_NOC, also with ragged blocks of the
    depth-2 ring."""
    monkeypatch.setattr(descriptor_module, "BANK_AFFINE_PER_CORE_VC", vc_directions)
    monkeypatch.setattr(descriptor_module, "BANK_AFFINE_VC_CYCLE", vc_cycle)
    monkeypatch.setattr(descriptor_module, "BANK_AFFINE_NOC_SPLIT", noc_split)
    monkeypatch.setattr(descriptor_module, "BLOCKS_PER_CORE_KNOB", blocks_per_core)
    monkeypatch.setattr(op_module, "_DESCRIPTOR_CACHE", {})
    torch.manual_seed(11)
    torch_input = _random_bits(shape, _TORCH_DTYPE[dtype])
    ttnn_input = _upload(torch_input, device, dtype, layout)
    plan = op_module.g6_clone_plan(ttnn_input)
    assert plan["regime"] == descriptor_module.REGIME_BANK_AFFINE, plan["regime"]
    assert (plan["bank_vc_directions"], plan["bank_noc_split"]) == (vc_directions, noc_split)
    vcs = {vc for read_vc, write_vc, _ in plan["route_args"] for vc in (read_vc, write_vc)}
    assert vcs <= set(vc_cycle) | {descriptor_module.DEFAULT_UNICAST_VC}, vcs
    if noc_split != "none":
        assert any(other for _, _, other in plan["route_args"]) or plan["block_pages"] < 2, plan["route_args"]
    ttnn_output = g6_clone(ttnn_input)
    _assert_clone(ttnn_output, ttnn_input, torch_input, dtype, layout)


def test_g6_clone_bank_route_knobs_skip_flat(device, monkeypatch):
    """The route knobs apply only to bank runs: a consecutive-page walk keeps one VC and one NoC."""
    monkeypatch.setattr(descriptor_module, "BANK_AFFINE_PER_CORE_VC", ("read", "write"))
    monkeypatch.setattr(descriptor_module, "BANK_AFFINE_NOC_SPLIT", "half")
    monkeypatch.setattr(op_module, "_DESCRIPTOR_CACHE", {})
    torch.manual_seed(13)
    shape = (1, 1, 1, 62080)
    torch_input = _random_bits(shape, torch.bfloat16)
    ttnn_input = _upload(torch_input, device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    plan = op_module.g6_clone_plan(ttnn_input)
    assert plan["regime"] == descriptor_module.REGIME_LOGICAL_ROWS
    assert (plan["bank_vc_directions"], plan["bank_noc_split"]) == ((), "none")
    ttnn_output = g6_clone(ttnn_input)
    _assert_clone(ttnn_output, ttnn_input, torch_input, ttnn.bfloat16, ttnn.TILE_LAYOUT)


def test_g6_clone_cache_key_partial_vs_full(device, monkeypatch):
    """[1,1,1,62080] and [1,1,32,62080] share page count and page size; a one-row descriptor
    served to the full-tile shape would drop rows 1-31, so the cache key must tell them apart."""
    monkeypatch.setattr(op_module, "_DESCRIPTOR_CACHE", {})
    torch.manual_seed(42)
    for shape in [(1, 1, 1, 62080), (1, 1, 32, 62080), (1, 1, 1, 62080)]:
        torch_input = _random_bits(shape, torch.bfloat16)
        ttnn_input = _upload(torch_input, device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        ttnn_output = g6_clone(ttnn_input)
        _assert_clone(ttnn_output, ttnn_input, torch_input, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    assert len(op_module._DESCRIPTOR_CACHE) == 2


def test_g6_clone_zone_build(device, monkeypatch):
    """The zone build (G6_CLONE_ZONES) compiles the stage zones and stays bit-exact; plain and zone
    builds are separate descriptors, so toggling the variable never replays the other build."""
    monkeypatch.setattr(op_module, "_DESCRIPTOR_CACHE", {})
    torch.manual_seed(42)
    shape = (1, 1, 256, 128)
    torch_input = _random_bits(shape, torch.bfloat16)
    ttnn_input = _upload(torch_input, device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    for value, expected in [("", False), ("1", True), ("0", False)]:
        monkeypatch.setenv(descriptor_module.ZONE_BUILD_ENV, value)
        assert op_module.g6_clone_plan(ttnn_input)["zone_build"] is expected
        _assert_clone(g6_clone(ttnn_input), ttnn_input, torch_input, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    assert len(op_module._DESCRIPTOR_CACHE) == 2
