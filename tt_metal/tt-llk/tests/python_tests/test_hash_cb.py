# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Standalone LLK test for the SFPU-backed CB hash (23-bit FNV23).
#
# The kernel folds each INT32 tile into 32 per-lane FNV23 accumulators on the
# SFPU (SFPLOAD / SFPXOR / multiply — SFPMUL24 on BH, shift-and-add on WH —
# masked to 23 bits), then writes the
# accumulators back into DEST row 0 (the rest of the tile zeroed) and lets the
# standard packer move the tile to L1. The host XOR-folds the whole result tile:
# the zeroed rows contribute nothing, so the fold equals XOR(32 accumulators) —
# a deterministic, input-sensitive fingerprint that is independent of the SFPU
# lane <-> tile-position permutation. This uses only the proven
# SFPU -> DEST -> PACK -> L1 path (no DEST debug-bus read-back).
#
# The tests verify:
#   1. The kernel runs and produces a non-zero 23-bit fingerprint.
#   2. Two runs on the same input produce bit-identical results (determinism).
#   3. Distinct inputs produce distinct fingerprints (discrimination).
#   4. The fold and the store-back target DEST slot 0 even when an SFPU op on
#      another slot ran earlier in the same DEST window -- before the hash init
#      (production order) or between the fold and the store (slot pinning) --
#      in either half-sync bank (dest_bank 1 runs one prior math/pack window
#      first, so the bank base the pin must restore is DEST_REGISTER_HALF_SIZE).
#
# A bit-exact Python golden is intentionally omitted: the SFPU lane ordering and
# the DEST/pack datum handling are hardware-specific. The properties above are
# what matter for nondeterminism bisection.

import pytest
import torch
from helpers.chip_architecture import ChipArchitecture
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    HASH_CB_DEST_BANK,
    HASH_CB_SFPU_OP,
    TILE_COUNT,
    generate_input_dim,
)

MASK23 = 0x7FFFFF
# Words in one 32x32 result tile; the hash tile is buffer_Res[0], a prior window's
# pass-through of the input (dest_bank 1) is buffer_Res[1].
TILE_WORDS = 32 * 32

# Architectures that support the SFPU hash variant.
SFPU_HASH_ARCHS = {ChipArchitecture.BLACKHOLE, ChipArchitecture.WORMHOLE}


def _skip_if_unsupported():
    if TestConfig.CHIP_ARCH not in SFPU_HASH_ARCHS:
        pytest.skip(f"SFPU hash not supported on {TestConfig.CHIP_ARCH}")


def _build_hash_kernel(
    formats,
    input_dimensions,
    src_A,
    src_B,
    tile_cnt_A,
    tile_cnt_B,
    dest_acc,
    sfpu_op_position=None,
    sfpu_op_dst_index=0,
    dest_bank=0,
):
    """Build (do not run) one SFPU hash kernel variant.

    ``sfpu_op_position`` runs a whole-tile SFPU op on DEST slot ``sfpu_op_dst_index``
    inside the same DEST window: ``"before_init"`` (before the hash init, the
    production order) or ``"before_store"`` (between the fold loop and the store).
    ``dest_bank`` is the half-sync bank the hash window runs in; bank 1 is reached
    by one prior math/pack window that passes the input tile through to
    ``buffer_Res[1]``.
    """
    return TestConfig(
        "sources/hash_cb_test.cpp",
        formats,
        templates=[
            generate_input_dim(input_dimensions, input_dimensions),
            HASH_CB_SFPU_OP(position=sfpu_op_position, dst_index=sfpu_op_dst_index),
            HASH_CB_DEST_BANK(bank=dest_bank),
        ],
        runtimes=[TILE_COUNT(tile_cnt_A)],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=1
            + dest_bank,  # hash tile (+ the prior window's pass-through)
        ),
        dest_acc=dest_acc,
        unpack_to_dest=formats.input_format.is_32_bit(),
    )


def _hash_of(res_from_L1):
    """23-bit hash from the packed result: the fold of the hash tile (buffer_Res[0])."""
    # The hash tile holds the 32 per-lane accumulators (DEST row 0) plus an
    # even number of identical packer-transformed zero words (rows 1..31). XOR-
    # folding the whole tile cancels the zero words and yields XOR(32 accumulators)
    # — a deterministic, input-sensitive fingerprint independent of lane layout.
    return _xor_fold(res_from_L1[:TILE_WORDS])


def _finish_hash_kernel(configuration):
    """Run a built variant and return the 23-bit hash."""
    return _hash_of(configuration.run().result)


def _run_hash_kernel(
    formats,
    input_dimensions,
    src_A,
    src_B,
    tile_cnt_A,
    tile_cnt_B,
    dest_acc,
    dest_bank=0,
):
    """Build and run the SFPU hash kernel and return the 23-bit hash."""
    return _finish_hash_kernel(
        _build_hash_kernel(
            formats,
            input_dimensions,
            src_A,
            src_B,
            tile_cnt_A,
            tile_cnt_B,
            dest_acc,
            dest_bank=dest_bank,
        )
    )


def _xor_fold(words):
    """XOR-fold a flat sequence of 32-bit words (order-independent)."""
    h = 0
    for v in words:
        h ^= int(v) & 0xFFFFFFFF
    return h


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_tiles=[1],
    seed=[42],
    dest_acc=[DestAccumulation.Yes],
    dest_bank=[0, 1],
)
def test_hash_cb_sfpu(formats, num_tiles, seed, dest_acc, dest_bank):
    """Verify the SFPU hash kernel runs and produces a valid 23-bit result.

    ``dest_bank`` 1 runs the hash in the second half-sync bank (one prior
    math/pack window), as hash_cb_sfpu does every other window in production.
    """
    _skip_if_unsupported()

    torch.manual_seed(seed)

    input_dimensions = [32 * num_tiles, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    hw_hash = _run_hash_kernel(
        formats,
        input_dimensions,
        src_A,
        src_B,
        tile_cnt_A,
        tile_cnt_B,
        dest_acc,
        dest_bank=dest_bank,
    )

    # Sanity: hash should be non-zero and fit in 23 bits (each lane accumulator
    # is masked to 23 bits in the kernel, and the zeroed rows cancel under XOR).
    # The 23-bit bound holds because this kernel packs Int32 -> Int32 identity
    # (raw bitcast), so the accumulators round-trip verbatim; a non-identity pack
    # format could push bits above bit 22, so only the non-zero / determinism /
    # discrimination properties are guaranteed in general.
    assert (
        hw_hash != 0
    ), "SFPU hash returned zero — kernel likely did not execute correctly"
    assert hw_hash <= MASK23, f"SFPU hash {hex(hw_hash)} exceeds 23-bit range"


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_tiles=[1],
    seed=[7],
    dest_acc=[DestAccumulation.Yes],
)
def test_hash_cb_sfpu_determinism(formats, num_tiles, seed, dest_acc):
    """Two kernel runs on the same input must produce bit-identical hashes."""
    _skip_if_unsupported()

    torch.manual_seed(seed)
    input_dimensions = [32 * num_tiles, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    h1 = _run_hash_kernel(
        formats, input_dimensions, src_A, src_B, tile_cnt_A, tile_cnt_B, dest_acc
    )
    h2 = _run_hash_kernel(
        formats, input_dimensions, src_A, src_B, tile_cnt_A, tile_cnt_B, dest_acc
    )
    assert h1 == h2, f"SFPU hash non-deterministic across runs: {hex(h1)} vs {hex(h2)}"


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_tiles=[1],
    seed=[42],
    dest_acc=[DestAccumulation.Yes],
)
def test_hash_cb_sfpu_discriminates(formats, num_tiles, seed, dest_acc):
    """Distinct inputs must produce distinct fingerprints."""
    _skip_if_unsupported()

    input_dimensions = [32 * num_tiles, 32]

    torch.manual_seed(seed)
    a_A, a_tc_A, a_B, a_tc_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )
    torch.manual_seed(seed + 1)
    b_A, b_tc_A, b_B, b_tc_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    ha = _run_hash_kernel(formats, input_dimensions, a_A, a_B, a_tc_A, a_tc_B, dest_acc)
    hb = _run_hash_kernel(formats, input_dimensions, b_A, b_B, b_tc_A, b_tc_B, dest_acc)
    assert ha != hb, f"SFPU hash collided on distinct inputs: {hex(ha)}"


@parametrize(
    formats=input_output_formats([DataFormat.Int32]),
    num_tiles=[1],
    seed=[42],
    dest_acc=[DestAccumulation.Yes],
    sfpu_op_position=["before_init", "before_store"],
    sfpu_op_dst_index=[1, 2],
    dest_bank=[0, 1],
)
def test_hash_cb_sfpu_after_sfpu_op_on_other_slot(
    formats,
    num_tiles,
    seed,
    dest_acc,
    sfpu_op_position,
    sfpu_op_dst_index,
    dest_bank,
):
    """The hash must fold and store DEST slot 0 after an SFPU op on another slot.

    An eltwise SFPU op at dst_index N leaves the math DEST offset at base + 64 * N
    for the rest of the DEST window, while the unpack-to-dest datacopy writes slot 0
    at an absolute address. The hash has to pin the offset back to the bank base
    itself, in the fold (``before_init``: the op precedes the hash, as in
    hash_cb_sfpu) and again in the store-back (``before_store``: a standalone caller
    issues the op between the fold and the store); otherwise it folds slot N and/or
    stores the accumulators into slot N, and the packed slot 0 is the untouched
    input tile.

    ``dest_bank`` 1 runs the hash window with ``dest_offset_id == 1`` (one prior
    math/pack window), where the bank base is DEST_REGISTER_HALF_SIZE: a pin to a
    literal 0 would pass every bank-0 case and fold/store bank 0 here, so the
    packed bank-1 slot 0 would again be the untouched input.
    """
    _skip_if_unsupported()

    torch.manual_seed(seed)
    input_dimensions = [32 * num_tiles, 32]
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    reference_cfg = _build_hash_kernel(
        formats, input_dimensions, src_A, src_B, tile_cnt_A, tile_cnt_B, dest_acc
    )
    perturbed_cfg = _build_hash_kernel(
        formats,
        input_dimensions,
        src_A,
        src_B,
        tile_cnt_A,
        tile_cnt_B,
        dest_acc,
        sfpu_op_position=sfpu_op_position,
        sfpu_op_dst_index=sfpu_op_dst_index,
        dest_bank=dest_bank,
    )
    # prepare() both before running either: under --compile-producer run() skips
    # as soon as the first variant is built, so the second would never emit its ELF.
    reference_cfg.prepare()
    perturbed_cfg.prepare()

    reference = _finish_hash_kernel(reference_cfg)
    perturbed_result = perturbed_cfg.run().result
    hw_hash = _hash_of(perturbed_result)

    # The packed tile is the raw input exactly when the hash folded/stored another
    # slot and never touched slot 0 (Int32 -> Int32 pack is a bitcast).
    raw_input_fold = _xor_fold(src_A.flatten())
    if dest_bank:
        # The prior window packed its datacopy of the input from bank 0: it must be
        # the raw input, which also pins down what "untouched input" folds to.
        prior_fold = _xor_fold(perturbed_result[TILE_WORDS : 2 * TILE_WORDS])
        assert prior_fold == raw_input_fold, (
            f"prior-window pass-through of the input is not the raw input "
            f"(fold {hex(prior_fold)} != {hex(raw_input_fold)})"
        )
    where = (
        f"an SFPU op on DEST slot {sfpu_op_dst_index} ({sfpu_op_position}) "
        f"in half-sync bank {dest_bank}"
    )
    assert hw_hash != raw_input_fold, (
        f"after {where}, the packed slot 0 is the untouched input tile "
        f"(fold {hex(hw_hash)}): the hash folded and/or stored another DEST slot"
    )
    assert hw_hash == reference, (
        f"SFPU hash after {where}: {hex(hw_hash)} != {hex(reference)} "
        "(dedicated-window reference)"
    )
