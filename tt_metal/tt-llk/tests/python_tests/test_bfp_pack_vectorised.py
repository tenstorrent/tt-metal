# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The vectorised BFP packer must agree byte-for-byte with the per-element reference.

``helpers.pack`` keeps two descriptions of BFP_b quantisation: the per-element loop
(``_bfp_collect_blocks`` driving ``float_to_bfp{8,4,2}_block``) and the numpy pass the
packers actually call (``_bfp_quantize_blocks``). This file is what makes that safe --
it asserts the two produce identical bytes, so the reference stays the authority on the
format and the fast path stays free to change.

The comparison is on **bytes, not values**. A tolerance-based check passes on an
implementation that gets the mantissa width wrong: the reference keeps bits 9..14 of
the bf16 word (dropping the mantissa LSB) for a 7-bit explicit magnitude, and the
"obvious" 8-bit ``(bf16 & 0x7F) | 0x80`` is within tolerance while emitting different
bytes. That mistake is only caught bytewise.
"""

import numpy as np
import pytest
import torch
from helpers.pack import (
    _bfp_collect_blocks,
    _bfp_prepare_blocks,
    _bfp_quantize_blocks,
    float_to_bfp2_block,
    float_to_bfp4_block,
    float_to_bfp8_block,
    pack_bfp2_b,
    pack_bfp4_b,
    pack_bfp8_b,
)

BLOCK_SIZE = 16


def _assert_datums_equal(actual, expected, what):
    """Compare two int lists, reporting only the first few mismatches.

    ``assert actual == expected`` on 1024-element lists makes pytest build a full list
    diff per failing case, which across this file's matrix is slow enough to be
    mistaken for a hang. Report the count and the first three positions instead.
    """
    if actual == expected:
        return
    assert len(actual) == len(
        expected
    ), f"{what} count differs: got {len(actual)}, expected {len(expected)}"
    bad = [i for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
    detail = ", ".join(
        f"[{i}] got {hex(actual[i])} expected {hex(expected[i])}" for i in bad[:3]
    )
    raise AssertionError(
        f"{what} diverged from the reference in {len(bad)}/{len(expected)} "
        f"positions: {detail}"
    )


# (magnitude_bits, reference block fn, public packer) per BFP_b width.
BFP_WIDTHS = [
    pytest.param(7, float_to_bfp8_block, pack_bfp8_b, id="bfp8_b"),
    pytest.param(3, float_to_bfp4_block, pack_bfp4_b, id="bfp4_b"),
    pytest.param(1, float_to_bfp2_block, pack_bfp2_b, id="bfp2_b"),
]


def _populations():
    """Tensors that exercise the paths where the two implementations could diverge.

    The shared exponent is a per-block max, so what matters is the *spread* within a
    block: a block whose members share an exponent never exercises the mantissa shift,
    and a block spanning a wide range drives the shift past the 7-bit magnitude (which
    is where numpy's undefined over-wide shift would bite if it were not clipped).
    """
    torch.manual_seed(0)
    n = 1024  # one full tile
    pops = {
        # Ordinary signed data: small deltas, exercises the rounding guard bit.
        "uniform": torch.rand(n) * 10 - 5,
        # Wide dynamic range *within* each block -> large exponent deltas.
        "wide_range": torch.randn(n) * 1e3,
        # Everything tiny: shared exponent near the subnormal floor.
        "subnormal": torch.rand(n) * 1e-30,
        # Everything huge: shared exponent near the top of the range.
        "huge": torch.rand(n) * 3e38,
        # Exact powers of two: every mantissa is 0x40, ties land on the guard bit.
        "powers_of_two": torch.tensor(
            [2.0 ** ((i % 60) - 30) for i in range(n)], dtype=torch.float32
        ),
        # All zeros: shared exponent 0, every magnitude flushes, -0 must not survive.
        "zeros": torch.zeros(n),
        # Alternating +-0 next to real values: the negative-zero flush rule.
        "signed_zeros": torch.tensor(
            [0.0, -0.0, 1.0, -1.0] * (n // 4), dtype=torch.float32
        ),
        # Every block pairs a near-max value with a near-min one, forcing an intra-block
        # exponent delta far past the 7-bit magnitude width. This is the only population
        # that drives the over-wide mantissa shift; test_wide_delta_path_is_exercised
        # asserts it still does, so the clip in _bfp_quantize_blocks stays covered.
        "extreme_delta": torch.tensor(
            [1e38 if i % BLOCK_SIZE == 0 else 1e-38 for i in range(n)],
            dtype=torch.float32,
        ),
    }
    # Every population additionally gets the specials seeded into distinct blocks, so
    # a NaN/inf shared exponent (0xFF) cannot mask a divergence elsewhere in the tile.
    specials = [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-45, -1e-45]
    for name, base in list(pops.items()):
        seeded = base.clone().to(torch.float32)
        for i, v in enumerate(specials):
            seeded[i * BLOCK_SIZE + i] = v  # one special per block, staggered
        pops[f"{name}+specials"] = seeded
    return pops


POPULATIONS = _populations()


@pytest.mark.parametrize("magnitude_bits,reference_block_fn,packer", BFP_WIDTHS)
@pytest.mark.parametrize("population", sorted(POPULATIONS), ids=lambda p: p)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_faces,face_r_dim", [(4, 16), (2, 16), (1, 16), (4, 1)])
def test_vectorised_matches_reference(
    magnitude_bits, reference_block_fn, packer, population, dtype, num_faces, face_r_dim
):
    """Exponents and per-datum mantissas must match the reference exactly."""
    tensor = POPULATIONS[population].to(dtype)
    flattened = _bfp_prepare_blocks(tensor, BLOCK_SIZE, num_faces, face_r_dim)

    ref_exponents, ref_mantissas = _bfp_collect_blocks(
        flattened, BLOCK_SIZE, reference_block_fn
    )
    vec_exponents, vec_mantissas = _bfp_quantize_blocks(
        flattened, BLOCK_SIZE, magnitude_bits
    )

    # Compared via an explicit first-mismatch report rather than `assert a == b`:
    # these are 1024-element lists, and pytest's list-diff rewriting on a failure is
    # slow enough across this file's matrix to look like a hang.
    _assert_datums_equal(vec_exponents, ref_exponents, "shared exponent")
    _assert_datums_equal(vec_mantissas, ref_mantissas, "mantissa datum")
    # The packers return Python ints in 0..255; anything else breaks bytes(...).
    assert all(isinstance(v, int) and 0 <= v <= 0xFF for v in vec_exponents)
    assert all(isinstance(v, int) and 0 <= v <= 0xFF for v in vec_mantissas)
    # And the public packer's byte stream must be convertible, which is how every
    # caller consumes it (bytes(pack_bfp8_b(...))).
    assert isinstance(
        bytes(packer(tensor, num_faces=num_faces, face_r_dim=face_r_dim)), bytes
    )


@pytest.mark.parametrize("magnitude_bits,reference_block_fn,packer", BFP_WIDTHS)
def test_public_packer_byte_stream_matches_reference(
    magnitude_bits, reference_block_fn, packer
):
    """End-to-end: the public packer's full byte list, including the bit-packing.

    ``_bfp_quantize_blocks`` returns one entry per datum; BFP4 folds two datums into a
    byte and BFP2 four, and that folding is done in the packers. This checks the folded
    result, which the per-datum comparison above cannot.
    """
    tensor = POPULATIONS["wide_range+specials"].to(torch.float32)
    flattened = _bfp_prepare_blocks(tensor, BLOCK_SIZE, 4, 16)
    ref_exponents, ref_datums = _bfp_collect_blocks(
        flattened, BLOCK_SIZE, reference_block_fn
    )

    datums_per_byte = 8 // (magnitude_bits + 1)
    if datums_per_byte == 1:
        expected = ref_exponents + ref_datums
    else:
        folded = []
        for i in range(0, len(ref_datums), datums_per_byte):
            byte = 0
            for k in range(datums_per_byte):
                byte |= ref_datums[i + k] << (k * (magnitude_bits + 1))
            folded.append(byte)
        expected = ref_exponents + folded

    assert packer(tensor) == expected


def test_negative_zero_never_leaves_a_sign_only_mantissa():
    """A sign-only mantissa decodes to -inf in hardware, so it must never be emitted.

    Asserted directly rather than left to the reference comparison: if both
    implementations regressed together the comparison would still pass, and this is the
    one property whose violation silently corrupts a tile.
    """
    for magnitude_bits in (7, 3, 1):
        sign_only = 1 << magnitude_bits
        for population in ("zeros", "signed_zeros", "subnormal+specials"):
            flattened = _bfp_prepare_blocks(
                POPULATIONS[population].to(torch.float32), BLOCK_SIZE, 4, 16
            )
            _, mantissas = _bfp_quantize_blocks(flattened, BLOCK_SIZE, magnitude_bits)
            assert sign_only not in mantissas, (
                f"{population} produced a sign-only {magnitude_bits + 1}-bit mantissa "
                f"({hex(sign_only)}), which the hardware unpacker decodes to -inf"
            )


def test_wide_delta_path_is_exercised():
    """The populations must actually drive an intra-block exponent delta past 7 bits.

    ``_bfp_quantize_blocks`` clips the mantissa shift so the result does not depend on
    how the array library defines an over-wide shift. That clip is only meaningful if
    some block really has a delta that large -- and a mutation removing the clip passes
    the agreement tests unless one does. Assert the precondition rather than assuming
    the populations keep providing it.
    """
    flattened = _bfp_prepare_blocks(
        POPULATIONS["extreme_delta"].to(torch.float32), BLOCK_SIZE, 4, 16
    )
    bf16 = (
        flattened.to(torch.float32).contiguous().numpy().view(np.uint32) >> 16
    ).astype(np.int32)
    exponent = ((bf16 >> 7) & 0xFF).reshape(-1, BLOCK_SIZE)
    max_delta = int((exponent.max(axis=1, keepdims=True) - exponent).max())
    assert max_delta > 7, (
        f"no block exceeds a 7-bit mantissa shift (max delta {max_delta}); the shift "
        "clip in _bfp_quantize_blocks is no longer covered by these populations"
    )
