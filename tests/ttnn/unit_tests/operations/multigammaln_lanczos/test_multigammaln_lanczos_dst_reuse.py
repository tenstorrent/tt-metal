# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Refinement 1 — Reuse DST across lgamma iterations.

The Phase-0 kernel evaluated each of the 4 Lanczos lgamma sub-evaluations in
its own ``tile_regs_acquire``/``release`` block and round-tripped the running
global accumulator through an L1 ``cb_accumulator`` between iterations:

    push 5×  /  wait 5×  /  pop 5×   per output element

Refinement 1 collapses that to a SINGLE ``tile_regs_acquire`` per tile that
keeps the global accumulator resident in DST (D0) across all 4 iterations.
``cb_accumulator`` is no longer needed and has been removed from the program
descriptor entirely.

This file pins those structural & precision invariants so a future change
that re-introduces an intermediate L1 round-trip trips a test immediately,
not silently in a precision regression weeks later.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc

from ttnn.operations.multigammaln_lanczos import multigammaln_lanczos
from ttnn.operations.multigammaln_lanczos.multigammaln_lanczos_program_descriptor import (
    create_program_descriptor,
)


SAFE_LO = 2.0
SAFE_HI = 10.0


def _safe_input(shape, seed: int = 13) -> torch.Tensor:
    torch.manual_seed(seed)
    u = torch.rand(shape, dtype=torch.float32)
    return SAFE_LO + (SAFE_HI - SAFE_LO) * u


def _torch_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.special.multigammaln(x.double(), 4).float()


def _build_program_descriptor(device, shape):
    """Build (input_tensor, output_tensor, program_descriptor) for inspection."""
    torch_input = _safe_input(shape)
    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(shape)),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.DRAM_MEMORY_CONFIG,
    )
    return input_tensor, output_tensor, create_program_descriptor(input_tensor, output_tensor)


# -----------------------------------------------------------------------------
# Structural invariants — the changes that ARE the refinement
# -----------------------------------------------------------------------------


def test_program_descriptor_has_no_cb_accumulator(device):
    """The Phase-0 cb_accumulator (intermediate L1 CB at index 24) must not
    exist in the descriptor any more. The global accumulator now lives in DST."""
    _, _, pd = _build_program_descriptor(device, (1, 1, 32, 32))

    # Phase 0 wired 3 CBs (input, output, accumulator). Refinement 1 → 2 CBs.
    assert len(pd.cbs) == 2, (
        "Expected exactly 2 CBs after Refinement 1 (cb_input_tiles, "
        f"cb_output_tiles); got {len(pd.cbs)}. A re-introduced "
        "cb_accumulator would re-add the L1 round-trip the refinement eliminated."
    )

    # No CB should be bound to the legacy intermediate slot (24 in op_design.md).
    used_indices = []
    for cb in pd.cbs:
        for fmt in cb.format_descriptors:
            used_indices.append(fmt.buffer_index)
    assert 24 not in used_indices, (
        f"CB index 24 (legacy cb_accumulator slot) is still in use: {used_indices}. "
        "Refinement 1 removed cb_accumulator — adding it back regresses the refinement."
    )
    # Standard input/output slots must still be there.
    assert 0 in used_indices, f"cb_input_tiles (index 0) missing from descriptor; got {used_indices}"
    assert 16 in used_indices, f"cb_output_tiles (index 16) missing from descriptor; got {used_indices}"


def test_compute_kernel_has_two_cb_compile_time_args(device):
    """Compute kernel CT args must be [cb_input_tiles, cb_output_tiles] only —
    the third entry (cb_accumulator) was removed by Refinement 1."""
    _, _, pd = _build_program_descriptor(device, (1, 1, 32, 32))

    # Find the compute kernel (only one with .cpp containing 'compute').
    compute_kernels = [k for k in pd.kernels if "compute" in k.kernel_source]
    assert len(compute_kernels) == 1, f"Expected one compute kernel; got {len(compute_kernels)}"
    compute_kernel = compute_kernels[0]

    assert list(compute_kernel.compile_time_args) == [0, 16], (
        f"Compute kernel CT args = {list(compute_kernel.compile_time_args)}; "
        "expected [0 (cb_input_tiles), 16 (cb_output_tiles)] after Refinement 1. "
        "A third entry would indicate cb_accumulator was re-added."
    )


def test_compute_config_unpack_to_dest_fp32_on_input_only(device):
    """UnpackToDestFp32 must remain on cb_input_tiles (re-read multiple times
    per output element). After Refinement 1, cb_accumulator no longer exists,
    so its entry must NOT be UnpackToDestFp32 (would be a no-op on a non-existent
    CB but would also signal stale config)."""
    _, _, pd = _build_program_descriptor(device, (1, 1, 32, 32))

    compute_kernels = [k for k in pd.kernels if "compute" in k.kernel_source]
    assert len(compute_kernels) == 1
    config = compute_kernels[0].config

    modes = list(config.unpack_to_dest_mode)
    assert modes[0] == ttnn.UnpackToDestMode.UnpackToDestFp32, (
        "cb_input_tiles (index 0) MUST be UnpackToDestFp32 — the kernel re-reads it "
        "many times per output element and the unpacker would otherwise truncate to TF32."
    )
    # cb_accumulator (index 24) is gone; the slot should be Default (carries no signal).
    assert modes[24] == ttnn.UnpackToDestMode.Default, (
        f"Slot 24 (legacy cb_accumulator) should be Default; got {modes[24]}. "
        "A non-default value implies cb_accumulator is still being configured."
    )


# -----------------------------------------------------------------------------
# Behavioural correctness on the new code path
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        # Single tile — one full lgamma×4 accumulation cycle in DST.
        pytest.param((1, 1, 32, 32), id="single_tile"),
        # Multiple tiles per core — exercises the per-tile re-acquire cycle.
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        # Multi-batch — touches every core in the grid.
        pytest.param((2, 4, 64, 128), id="multi_batch"),
    ],
)
def test_dst_resident_accumulator_matches_reference(device, shape):
    """The DST-resident global accumulator must produce the same answer as the
    fp64 reference within the same precision envelope as Phase 0."""
    torch_input = _safe_input(shape, seed=29)
    expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual = ttnn.to_torch(multigammaln_lanczos(ttnn_input)).float()

    diff = (actual - expected).float()
    max_abs = diff.abs().max().item()
    expected_rms = expected.float().pow(2).mean().sqrt().item()
    rel_rms = diff.pow(2).mean().sqrt().item() / (expected_rms + 1e-12)

    print(f"\n[dst_reuse] shape={tuple(shape)} max_abs={max_abs:.6g} rel_rms={rel_rms:.6g}")

    # Same precision floor as test_multigammaln_lanczos_precision_baseline.
    pcc_passed, pcc_msg = check_with_pcc(expected, actual, pcc=0.999)
    assert pcc_passed, f"PCC<0.999 for shape={shape}: {pcc_msg}"
    assert max_abs < 0.05, f"max_abs={max_abs} regressed; expected < 0.05 for shape={shape}"
    assert rel_rms < 5e-4, f"rel_rms={rel_rms} regressed; expected < 5e-4 for shape={shape}"


def test_d1_corrupt_and_reload_correctness(device):
    """
    The kernel corrupts D1 (= a) for the (a-0.5)*log(a+4.5) multiply and then
    reloads D1 = a + offset[k] from cb_input_tiles before pole zeroing.

    This test stresses BOTH halves of that contract:
      * a=2.0 hits the pole exactly inside lgamma(x) → pole zeroing must fire
        on the CORRECT reloaded value of a, not on the corrupted D1=1.5.
        If we accidentally compared (a-0.5 != 1.0) instead of (a != 1.0),
        we'd zero out lgamma evaluations that should not be zeroed.
      * Multiple iterations k=0..3 mean the reload+offset arithmetic must
        recover the exact original `a` for the next iteration's pole check.
    """
    shape = (1, 1, 32, 32)
    # Mix of pole-hitting and off-pole values to stress the mask logic.
    # x=2.0  → a=2.0 at k=0 (pole), a=1.5/1.0/0.5 at k=1/2/3 (the a=1 pole at k=2!)
    # x=2.5  → a=2.5 at k=0, a=2.0/1.5/1.0 at k=1/2/3 (a=2 at k=1 and a=1 at k=3 both poles)
    # x=4.0  → a=4.0/3.5/3.0/2.5 (no pole)
    # x=10.0 → a=10.0/9.5/9.0/8.5 (no pole, high values stress (a-0.5)*log(a+4.5))
    values = torch.tensor([2.0, 2.5, 4.0, 10.0], dtype=torch.float32)
    # Tile the values to fill the 32x32 tile.
    torch_input = values.repeat_interleave(8).repeat(32).reshape(shape)

    expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual = ttnn.to_torch(multigammaln_lanczos(ttnn_input)).float()

    assert torch.isfinite(actual).all(), (
        f"Non-finite output — pole zeroing failed on D1 corruption/reload contract.\n"
        f"actual flat[:8] = {actual.flatten()[:8].tolist()}"
    )

    max_abs = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, rtol=0.1, atol=0.5), (
        f"D1 corrupt+reload regression: max_abs={max_abs:.6f}\n"
        f"actual.flat[:8]   = {actual.flatten()[:8].tolist()}\n"
        f"expected.flat[:8] = {expected.flatten()[:8].tolist()}"
    )

    # PCC remains the tight gate — pole-mask bugs typically also show up as
    # NaN/Inf in PCC's mean/variance pipeline.
    assert_with_pcc(expected, actual, pcc=0.999)


def test_single_acquire_block_stress(device):
    """The new kernel does ONE long tile_regs_acquire block per tile (≈ 150
    SFPU ops). Run a large multi-core shape to verify nothing in the long
    block trips a math/pack synchronisation issue at scale."""
    # 65 tiles total (forces both core_group_1 and core_group_2 to be non-empty
    # on an 8x8 grid) — same shape as test_uneven_work_split, but bigger
    # individual work units to stress the per-tile single-acquire block.
    shape = (1, 1, 64, 65 * 32)
    torch_input = _safe_input(shape, seed=101)
    expected = _torch_reference(torch_input)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    actual = ttnn.to_torch(multigammaln_lanczos(ttnn_input)).float()

    assert_with_pcc(expected, actual, pcc=0.999)
    assert torch.allclose(actual, expected, rtol=0.1, atol=0.5)
