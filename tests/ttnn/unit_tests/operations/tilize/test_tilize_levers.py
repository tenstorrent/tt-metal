# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Structural pins for the lever ledger's `structurally-impossible` rows.

`/perf-ceiling-dm` Mode D allows exactly one argument-based closure — and only
when the argument is pinned by a passing test. These are those tests: each one
mechanically asserts the property that makes a catalog lever unbuildable (or
already-satisfied-by-construction) for tilize, so the ledger row can never
quietly outlive its premise.

DO NOT DELETE — `lever_ledger.json` names these tests by `assert_test`.
"""

from pathlib import Path

import pytest
import torch
import ttnn

from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd


KERNEL_DIR = Path(pd.__file__).parent / "kernels"


def _kernel_source(name):
    return (KERNEL_DIR / name).read_text()


def test_production_switches_ship_in_their_optimal_state():
    """Every lever ON, every classification ablation OFF — the shipped config.

    `LEVERS` / `ABLATE` are module-level dicts the bench mutates to build the
    counterfactual arms. Production must be the all-ON / all-OFF corner: an
    ablated arm produces deliberately WRONG output, and an OFF lever ships a
    measured-slower kernel. Pin the defaults so a stray edit (or a bench arm
    left behind in the file) cannot become the shipped behaviour.
    """
    assert pd.ABLATE == {"compute": 0, "dm": 0}, f"classification ablation is live: {pd.ABLATE}"
    off = {name: value for name, value in pd.LEVERS.items() if value != 1}
    assert not off, f"lever(s) shipped in their OFF (counterfactual) arm: {off}"


def test_cb_geometry_has_a_single_source():
    """The CB page/byte formula lives once, in cb_pages()/cb_bytes().

    Every consumer — the L1 ceiling inside derive_blocking(), the depth-2
    fallback, and both CBDescriptor sizes — reads those two functions, so
    turning CB_DEPTH / NT_BLK / WT_CHUNK lands in exactly one place. Pinned
    because a restated formula silently drops a knob (NT_BLK did, before it was
    routed through cb_bytes()).
    """
    assert pd.cb_pages(2, 8) == 2 * pd.NT_BLK * 8
    assert pd.cb_bytes(2, 8, 2048, 2048) == pd.cb_pages(2, 8) * (2048 + 2048)
    # the L1 ceiling must scale with NT_BLK: a per-tile-column cost that ignored
    # it would let a raised NT_BLK overflow the budget it was checked against.
    assert pd.cb_bytes(2, 1, 2048, 2048) == 2 * pd.NT_BLK * 4096

    source = Path(pd.__file__).read_text()
    body = source.split("def create_program_descriptor")[1]
    assert "cb_depth * NT_BLK" not in body, "CB page formula restated outside cb_pages()"


def test_b12_multicast_is_structurally_absent():
    """B12 (multicast): no operand is read by more than one core, ever.

    tilize is a pure permutation of byte positions: every input byte belongs to
    exactly one block on exactly one core under either split axis, so there is
    nothing to fan out. Pinned three ways: no semaphores in the program, no
    mcast primitive in any kernel, and no mcast_pipe include.
    """
    sources = {name: _kernel_source(name) for name in ("tilize_reader.cpp", "tilize_compute.cpp", "tilize_writer.cpp")}
    for name, src in sources.items():
        assert "multicast" not in src, f"{name} issues a multicast"
        assert "mcast_pipe" not in src, f"{name} includes the mcast pipe helper"
        assert "noc_semaphore" not in src, f"{name} uses a semaphore handshake"


def test_b12_program_has_no_semaphores(device):
    """The same claim, from the host side: the descriptor declares no semaphore."""
    torch_input = torch.zeros([1, 1, 64, 64], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    from ttnn.operations.tilize.tilize import validate

    plan = validate(tt_input)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(plan.target), plan.out_dtype, ttnn.TILE_LAYOUT, device, plan.out_memory_config
    )
    descriptor = pd.create_program_descriptor(tt_input, out, plan)
    assert len(descriptor.semaphores) == 0


def test_c17_in_place_is_structurally_impossible(device):
    """C17 (in-place / no-copy): input and output are different LAYOUTS.

    A ROW_MAJOR source and a TILE destination are different byte orderings of
    the same values, so the op can never alias one buffer onto the other — the
    two buffers are always distinct.
    """
    torch_input = torch.randn([1, 1, 64, 64]).to(torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    out = tilize(tt_input)
    assert tt_input.layout == ttnn.ROW_MAJOR_LAYOUT and out.layout == ttnn.TILE_LAYOUT
    assert out.buffer_address() != tt_input.buffer_address()


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("w", [64, 2048, 16384], ids=["w64", "w2048", "w16384"])
def test_b11_every_transaction_is_dram_aligned(device, dtype, w):
    """B11 (alignment): misalignment cannot occur — nothing to fix.

    Both transaction sizes are structurally 32 B multiples: the read chunk is
    `WT_CHUNK * 32 * elem` and the write is a whole tile page.
    """
    elem = 2 if dtype == ttnn.bfloat16 else 4
    grid = device.compute_with_storage_grid_size()
    in_tile_bytes = out_tile_bytes = 32 * 32 * elem
    wt_chunk, _, _ = pd.derive_blocking(1, w // 32, in_tile_bytes, out_tile_bytes, grid.x * grid.y, 2, 32)

    row_bytes = wt_chunk * 32 * elem
    assert row_bytes % ttnn.get_dram_alignment() == 0
    assert out_tile_bytes % ttnn.get_dram_alignment() == 0


def test_f26_lossless_fp32_tilize_is_never_requested():
    """F26: the tiled output is consumed by FPU ops, so a lossless unpack path
    buys nothing downstream. The compute kernel must never ask for it."""
    src = _kernel_source("tilize_compute.cpp")
    assert "Fp32Mode::Lossless" not in src
    assert "Fp32Mode::Fast" in src


def test_f27_no_math_fidelity_sensitive_op():
    """F27 (math_fidelity): tilize performs NO arithmetic — the LLK reinterprets
    byte positions. There is no multiply whose fidelity could be lowered."""
    src = _kernel_source("tilize_compute.cpp")
    for math_api in ("matmul", "reduce", "add_tiles", "mul_tiles", "sub_tiles", "MathFidelity"):
        assert math_api not in src, f"compute kernel uses {math_api} — F27 would then apply"


@pytest.mark.parametrize("w, nt_h", [(2048, 64), (16384, 1), (1024, 256), (64, 1)])
def test_a4_no_cliff_core_width(device, w, nt_h):
    """A4 (cliff-core specialization): every block has the SAME width.

    `n_chunks` is snapped to an exact divisor of WT, so there is no partial-width
    block and therefore no cliff core to specialize — the block-count remainder
    is absorbed by split_work_to_cores' two groups running the same kernel.
    """
    grid = device.compute_with_storage_grid_size()
    tile_bytes = 32 * 32 * 2
    wt = w // 32
    wt_chunk, n_chunks, num_blocks = pd.derive_blocking(nt_h, wt, tile_bytes, tile_bytes, grid.x * grid.y, 2, 32)
    assert wt_chunk * n_chunks == wt
    assert num_blocks == nt_h * n_chunks
