# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — physical shard placements (HEIGHT / WIDTH / BLOCK).

Pins the three things this refinement introduced that nothing else guards:

1. Correctness on each scheme, in both layouts. The schemes exercise DIFFERENT
   reduction topologies, which is the whole reason they are one refinement rather
   than three: HEIGHT cuts the independent `row` axis (w_group_size == 1, the
   combine degenerates to a local copy), WIDTH cuts the dependent `hidden` axis
   (the whole shard grid is ONE cross-core reduction group), BLOCK cuts both (one
   grid row of the shard rectangle is a group).

2. The ZERO-COPY contract. A TILE shard's block CB must be PINNED over the
   resident L1 buffer, not re-read through a TensorAccessor. That distinction is
   invisible to a numerical test — an accessor read of a core's own shard returns
   the right bytes — so it is asserted STRUCTURALLY, on the descriptor.

3. Sub-tile per-core hidden slices. A ROW_MAJOR WIDTH/BLOCK shard's width granule
   is the L1 alignment, not a tile, so a core's slice can be 8-16 elements of a
   32-column tile: EVERY core then carries a ragged tail (per-core `partial_w`,
   not just the core owning the tensor's last tile) and its gamma slice starts at
   an offset that is neither tile- nor DRAM-aligned. Both were real bugs found
   here (PCC 0.28 / 0.44), so both are pinned.

The device fixture comes from conftest.py (module-scoped) — never opened here.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from eval.sharding import auto_shard_config, shard_config
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import (
    CB_INPUT_TILES,
    CB_OUTPUT_TILES,
    READER_ACCESSOR_CT_BASE,
    create_program_descriptor,
)

ML = ttnn.TensorMemoryLayout
_SCHEMES = [ML.HEIGHT_SHARDED, ML.WIDTH_SHARDED, ML.BLOCK_SHARDED]


def _torch_rms_norm(x, gamma=None, eps=1e-6):
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.float().reshape(-1)
    return out


def _pcc(got, expected):
    a = got.float().flatten()
    b = expected.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _run(device, shape, memory_layout, layout, *, gamma=True, dtype=ttnn.bfloat16, pin=None):
    """Dispatch on a sharded input, requesting the matching sharded output."""
    torch.manual_seed(0)
    torch_dtype = torch.bfloat16 if dtype != ttnn.float32 else torch.float32
    torch_input = torch.randn(shape, dtype=torch_dtype)
    torch_gamma = torch.randn(shape[-1], dtype=torch_dtype) if gamma else None

    if pin is not None:
        memory_config = shard_config(pin[0], pin[1], memory_layout, layout=layout, dtype=dtype, device=device)
    else:
        memory_config = auto_shard_config(list(shape), memory_layout, layout=layout, dtype=dtype, device=device)

    ttnn_input = ttnn.from_torch(torch_input, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    ttnn_gamma = (
        ttnn.from_torch(torch_gamma.reshape(1, 1, 1, -1), dtype=dtype, layout=layout, device=device) if gamma else None
    )
    ttnn_output = rms_norm(ttnn_input, gamma=ttnn_gamma, memory_config=ttnn_input.memory_config())
    return ttnn_input, ttnn_output, ttnn.to_torch(ttnn_output), _torch_rms_norm(torch_input, torch_gamma)


# ---------------------------------------------------------------------------
# 1. Correctness per scheme x layout
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("memory_layout", _SCHEMES, ids=lambda m: str(m).split(".")[-1])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("shape", [(1, 1, 256, 512), (1, 1, 32, 2048)], ids=["256x512", "32x2048"])
def test_rms_norm_sharded_schemes(device, shape, memory_layout, layout):
    _, _, got, expected = _run(device, shape, memory_layout, layout)
    assert _pcc(got, expected) > 0.995, f"PCC {_pcc(got, expected)}"


@pytest.mark.parametrize("memory_layout", _SCHEMES, ids=lambda m: str(m).split(".")[-1])
def test_rms_norm_sharded_no_gamma(device, memory_layout):
    _, _, got, expected = _run(device, (1, 1, 256, 512), memory_layout, ttnn.TILE_LAYOUT, gamma=False)
    assert _pcc(got, expected) > 0.995


@pytest.mark.parametrize("memory_layout", _SCHEMES, ids=lambda m: str(m).split(".")[-1])
def test_rms_norm_sharded_output_inherits_shard_spec(device, memory_layout):
    """The op honours a sharded `memory_config`: the output IS sharded, same spec."""
    ttnn_input, ttnn_output, _, _ = _run(device, (1, 1, 256, 512), memory_layout, ttnn.TILE_LAYOUT)
    out_config = ttnn_output.memory_config()
    assert out_config.memory_layout == memory_layout
    assert list(out_config.shard_spec.shape) == list(ttnn_input.memory_config().shard_spec.shape)


# ---------------------------------------------------------------------------
# 2. The zero-copy contract (structural — invisible to a numerical check)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("memory_layout", _SCHEMES, ids=lambda m: str(m).split(".")[-1])
def test_rms_norm_tile_shard_is_consumed_in_place(device, memory_layout):
    """A TILE shard's block CBs are PINNED over the resident L1 buffers.

    This is what makes the placement IMPLEMENTED rather than merely tolerated: an
    accessor read of a core's own shard would return the same bytes (so every
    numerical test would still pass) while re-fetching over the NoC data that is
    already in L1. Assert it on the descriptor instead.
    """
    shape = (1, 1, 256, 512)
    memory_config = auto_shard_config(
        list(shape), memory_layout, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ttnn_input = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    ttnn_gamma = ttnn.from_torch(
        torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    ttnn_output = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn_input.memory_config()
    )
    descriptor = create_program_descriptor(
        ttnn_input,
        ttnn_gamma,
        ttnn_output,
        epsilon=1e-6,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(),
    )

    pinned = {}
    for cb in descriptor.cbs:
        for fmt in cb.format_descriptors:
            if cb.has_buffer():
                pinned[int(fmt.buffer_index)] = int(ttnn.get_cb_address(cb))

    assert CB_INPUT_TILES in pinned, "cb_input_tiles must be pinned over the resident input shard"
    assert CB_OUTPUT_TILES in pinned, "cb_output_tiles must be pinned over the resident output shard"
    assert pinned[CB_INPUT_TILES] == ttnn_input.buffer_address()
    assert pinned[CB_OUTPUT_TILES] == ttnn_output.buffer_address()


# ---------------------------------------------------------------------------
# 3. Sub-tile per-core hidden slices (two real bugs found here)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("memory_layout", [ML.WIDTH_SHARDED, ML.BLOCK_SHARDED], ids=lambda m: str(m).split(".")[-1])
@pytest.mark.parametrize("shape", [(1, 1, 256, 512), (1, 1, 224, 1000)], ids=["256x512", "224x1000"])
def test_rms_norm_row_major_shard_sub_tile_slices(device, shape, memory_layout):
    """ROW_MAJOR WIDTH/BLOCK: a core's hidden slice can be 8-16 elements wide.

    Every core then has a ragged tail (per-core `partial_w`) AND a gamma slice at
    an offset that is neither tile- nor DRAM-aligned. Regression for the two bugs
    that produced PCC 0.28 (gamma read truncated to the 64-byte DRAM alignment)
    and PCC 0.44 (the shifted re-read hitting a sub-L1-alignment offset).
    """
    _, _, got, expected = _run(device, shape, memory_layout, ttnn.ROW_MAJOR_LAYOUT)
    assert _pcc(got, expected) > 0.995, f"PCC {_pcc(got, expected)}"


def test_rms_norm_sharded_mixed_gamma_dtype(device):
    """fp32 activations x bf16 gamma on a sub-tile ROW_MAJOR WIDTH shard.

    The gamma byte offset is then a multiple of the GAMMA element size only, so it
    need not even be L1-aligned — the case the CPU-side shift exists for.
    """
    shape = (1, 1, 32, 64)
    torch.manual_seed(0)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_gamma = torch.randn(shape[-1], dtype=torch.bfloat16)
    memory_config = auto_shard_config(
        list(shape), ML.WIDTH_SHARDED, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32, device=device
    )
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma.reshape(1, 1, 1, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    got = ttnn.to_torch(rms_norm(ttnn_input, gamma=ttnn_gamma, memory_config=ttnn_input.memory_config()))
    assert _pcc(got, _torch_rms_norm(torch_input, torch_gamma)) > 0.999


# ---------------------------------------------------------------------------
# 4. The pinned perf geometries Refinement 5 measures
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize(
    "shape", [(1, 1, 32, 4064), (1, 1, 96, 6144), (1, 1, 992, 3000)], ids=["w4064", "w6144", "w3000"]
)
def test_rms_norm_height_shard_wide_w(device, shape, layout):
    """Refinement 2b — HEIGHT_SHARDED with a hidden slice too wide to hold whole.

    HEIGHT cuts the INDEPENDENT `row` axis, so `w_group_size == 1` and
    `core_w_tiles == tensor_w_tiles` by construction: the caller has pinned the very
    knob the residency solve would otherwise raise, and past C ~ 127 the resident
    shards plus the C-wide streaming CBs no longer fit L1. The op then chunks the
    hidden axis (`w_chunk_tiles`) instead of refusing. Both layouts are covered
    because they chunk DIFFERENT buffers: TILE pins the block CBs over the shards
    and chunks gamma/normed; ROW_MAJOR additionally chunks the tilize/untilize
    staging pair and lays the block out chunk-major.
    """
    _, _, got, expected = _run(device, shape, ML.HEIGHT_SHARDED, layout)
    assert _pcc(got, expected) > 0.999, f"PCC {_pcc(got, expected)}"


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
def test_rms_norm_hidden_chunking_is_a_live_knob(device, layout):
    """`w_chunk_tiles` is turned ONLY when the whole hidden slice does not fit.

    Structural, because it is invisible to a numerical check: a narrow geometry must
    keep the byte-identical single-chunk schedule (chunk == the whole hidden slice),
    and a wide HEIGHT shard must actually come back with a SMALLER chunk. Reads the
    knob off the reader's compile-time args, which is where all three kernels get it.
    """

    def chunk_and_width(shape):
        memory_config = auto_shard_config(
            list(shape), ML.HEIGHT_SHARDED, layout=layout, dtype=ttnn.bfloat16, device=device
        )
        ttnn_input = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=device,
            memory_config=memory_config,
        )
        ttnn_gamma = ttnn.from_torch(
            torch.randn(1, 1, 1, shape[-1], dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=device,
        )
        ttnn_output = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), ttnn.bfloat16, layout, device, memory_config
        )
        descriptor = create_program_descriptor(
            ttnn_input, ttnn_gamma, ttnn_output, epsilon=1e-6, compute_kernel_config=ttnn.ComputeConfigDescriptor()
        )
        reader_ct = list(descriptor.kernels[0].compile_time_args)
        # reader CT layout: [C, tensor_w_tiles, ..., w_chunk_tiles] — see
        # READER_ACCESSOR_CT_BASE in the program descriptor.
        return reader_ct[READER_ACCESSOR_CT_BASE - 1], reader_ct[0]

    narrow_chunk, narrow_c = chunk_and_width((1, 1, 256, 512))
    assert narrow_chunk == narrow_c, "a geometry that fits must keep the single-chunk (unchunked) schedule"

    wide_chunk, wide_c = chunk_and_width((1, 1, 96, 6144))
    assert wide_chunk < wide_c, f"a wide HEIGHT shard must chunk the hidden axis (got {wide_chunk} of {wide_c})"
    assert wide_chunk >= 1


@pytest.mark.parametrize(
    "shape,shard_shape,core_grid,memory_layout",
    [
        ((1, 1, 32, 1024), [32, 128], (8, 1), ML.WIDTH_SHARDED),
        ((1, 1, 32, 2304), [32, 256], (9, 1), ML.WIDTH_SHARDED),
        ((1, 1, 32, 5120), [32, 160], (8, 4), ML.WIDTH_SHARDED),
        ((1, 1, 32, 7168), [32, 256], (7, 4), ML.WIDTH_SHARDED),
        ((1, 1, 8192, 1024), [1024, 128], (8, 8), ML.BLOCK_SHARDED),
    ],
    ids=["w1024", "w2304", "w5120", "w7168", "block8192x1024"],
)
def test_rms_norm_pinned_perf_shard_geometries(device, shape, shard_shape, core_grid, memory_layout):
    """feature_spec's five `group="perf"` sharded geometries must at least RUN.

    Refinement 5 measures these; they are pinned here so a later change cannot
    silently break the geometry it is supposed to be optimizing.
    """
    _, _, got, expected = _run(device, shape, memory_layout, ttnn.TILE_LAYOUT, pin=(shard_shape, core_grid))
    assert _pcc(got, expected) > 0.9995, f"PCC {_pcc(got, expected)}"
