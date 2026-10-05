# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the heavyweight golden's operation chains.

No kernel, no device. The chains decide which src format each unpack and Dest-feedback
step lands in when the caller does not name one, and that choice has to match what the
harness programs or the golden computes on values the kernel never saw.
"""

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generator.heavyweight.operations.quasar_operations.quasar_eltwise import (
    QuasarEltwiseBinaryGolden,
)
from helpers.golden_generator.heavyweight.operations.quasar_operations.quasar_reuse_dest import (
    QuasarEltwiseBinaryReuseDestGolden,
)
from helpers.llk_params import EltwiseBinaryReuseDestType, MathFidelity, MathOperation

TILE = 1024


def _elwmul(in_format, out_format, a, b, **kwargs):
    golden = QuasarEltwiseBinaryGolden(MathOperation.Elwmul, MathFidelity.HiFi4)
    return golden.run(
        [torch.full((TILE,), a), torch.full((TILE,), b)],
        in_format,
        out_format,
        **kwargs,
    ).float()


def test_float32_input_under_a_float16_dest_unpacks_into_float16():
    """The harness lands Float32 in a Float16 src for a Float16 output, and that src
    flushes 2^-16 before the multiply. A Tf32 src would keep it and give 2^-14."""
    out = _elwmul(
        DataFormat.Float32,
        DataFormat.Float16,
        2.0**-16,
        4.0,
        dest_format=DataFormat.Float16,
    )
    assert out.tolist() == [0.0] * TILE


@pytest.mark.parametrize(
    "out_format, kwargs",
    [
        (DataFormat.Float16_b, dict(dest_format=DataFormat.Float16_b)),
        (DataFormat.Float32, dict(dest_acc=True)),
    ],
    ids=["float16_b_dest", "dest_acc"],
)
def test_float32_input_keeps_the_8_bit_exponent_range_otherwise(out_format, kwargs):
    out = _elwmul(DataFormat.Float32, out_format, 2.0**-16, 4.0, **kwargs)
    assert out.tolist() == [2.0**-14] * TILE


# ---------------------------------------------------------------------------
# reuse_dest block geometry


def _reuse_dest(tiles_a, tiles_b=None, **kwargs):
    golden = QuasarEltwiseBinaryReuseDestGolden(
        MathOperation.Elwadd, MathFidelity.LoFi, EltwiseBinaryReuseDestType.DEST_TO_SRCA
    )
    a = torch.cat([torch.full((TILE,), float(v)) for v in tiles_a])
    b = torch.cat([torch.full((TILE,), float(v)) for v in (tiles_b or tiles_a)])
    return golden.run([a, b], DataFormat.Float16_b, DataFormat.Float16_b, **kwargs)


@pytest.mark.parametrize(
    "tiles, kwargs",
    [
        ([1] * 3, dict(inner_dim=2)),  # tile 2 would go unused
        ([1] * 6, dict(inner_dim=2, output_tiles_in_block=2)),  # would read tile 6
        ([1] * 2, dict(inner_dim=0)),
        ([1] * 2, dict(inner_dim=2, output_tiles_in_block=-1)),
    ],
    ids=["leftover_tile", "past_the_end", "zero_inner_dim", "negative_block"],
)
def test_an_incomplete_reuse_dest_block_is_refused(tiles, kwargs):
    with pytest.raises(ValueError):
        _reuse_dest(tiles, **kwargs)


def test_reuse_dest_operands_must_match_in_size():
    with pytest.raises(ValueError, match="differ in size"):
        _reuse_dest([1] * 4, [1] * 2, inner_dim=2)


def test_a_complete_reuse_dest_block_folds_every_tile():
    """inner_dim=3, output_tiles_in_block=2: output j folds inputs j, j+2, j+4.

    DEST_TO_SRCA with Elwadd: Dest = seed(A[j]) + B[j], then Dest += B[j+2], then
    Dest += B[j+4]. Values are small integers, so every step is exact in bf16."""
    a = [1, 2, 10, 20, 100, 200]
    out = (
        _reuse_dest(a, a, inner_dim=3, output_tiles_in_block=2).float().reshape(2, TILE)
    )
    assert out[0].unique().tolist() == [1 + 1 + 10 + 100]
    assert out[1].unique().tolist() == [2 + 2 + 20 + 200]


# ---------------------------------------------------------------------------
# Golden.run input geometry


def _eltwise_add(*operands, **kwargs):
    golden = QuasarEltwiseBinaryGolden(MathOperation.Elwadd, MathFidelity.LoFi)
    return golden.run(
        list(operands), DataFormat.Float16_b, DataFormat.Float16_b, **kwargs
    )


@pytest.mark.parametrize("blocked", [False, True], ids=["one_tile", "blocked"])
@pytest.mark.parametrize("extra", [1, TILE // 2], ids=["one_datum", "half_tile"])
def test_a_trailing_partial_tile_is_refused_on_both_paths(blocked, extra):
    tiles = 2 if blocked else 1
    x = torch.ones(tiles * TILE + extra)
    with pytest.raises(ValueError, match="whole number"):
        _eltwise_add(x, x, num_tiles_per_output=tiles)


@pytest.mark.parametrize("blocked", [False, True], ids=["one_tile", "blocked"])
def test_operands_of_different_sizes_are_refused(blocked):
    tiles = 2 if blocked else 1
    with pytest.raises(ValueError, match="differ in size"):
        _eltwise_add(torch.ones(2 * TILE), torch.ones(TILE), num_tiles_per_output=tiles)


def test_whole_tiles_still_run_on_both_paths():
    a, b = torch.full((2 * TILE,), 1.0), torch.full((2 * TILE,), 2.0)
    assert _eltwise_add(a[:TILE], b[:TILE]).float().unique().tolist() == [3.0]
    assert _eltwise_add(a, b, num_tiles_per_output=2).float().unique().tolist() == [6.0]


# ---------------------------------------------------------------------------
# Golden.run_l1 inputs


def _config(golden, operands, tiles_per_output=1):
    _, cfg = golden._make_config(
        DataFormat.Float16_b,
        DataFormat.Float16_b,
        operands=operands,
        geometry=dict(num_faces=4, face_r_dim=16),
        tiles_per_output=tiles_per_output,
        dest_format=None,
        dest_acc=False,
        pack_effects={},
    )
    return cfg


def _l1(golden, value):
    return golden.blocks.pack_to_l1(
        torch.full((TILE,), float(value)), DataFormat.Float16_b
    )


def _unpacked(golden, l1):
    return (
        golden.blocks.unpack_from_l1(l1, DataFormat.Float16_b).float().unique().tolist()
    )


def test_run_l1_takes_a_sequence_for_a_single_tile_chain():
    golden = QuasarEltwiseBinaryGolden(MathOperation.Elwadd, MathFidelity.LoFi)
    out = golden.run_l1([_l1(golden, 1), _l1(golden, 2)], _config(golden, 2))
    assert _unpacked(golden, out) == [3.0]


def test_run_l1_takes_named_tile_slots_for_a_folding_chain():
    golden = QuasarEltwiseBinaryGolden(MathOperation.Elwadd, MathFidelity.LoFi)
    buffers = {
        golden.source(0, 0): _l1(golden, 1),
        golden.source(1, 0): _l1(golden, 2),
        golden.source(0, 1): _l1(golden, 10),
        golden.source(1, 1): _l1(golden, 20),
    }
    out = golden.run_l1(buffers, _config(golden, 2, tiles_per_output=2))
    assert _unpacked(golden, out) == [33.0]


def test_run_l1_takes_the_reuse_dest_seed_by_name():
    golden = QuasarEltwiseBinaryReuseDestGolden(
        MathOperation.Elwadd, MathFidelity.LoFi, EltwiseBinaryReuseDestType.DEST_TO_SRCA
    )
    # DEST_TO_SRCA feeds A back from Dest, so the chain reads only the seed and B.
    buffers = {
        golden.SEED: _l1(golden, 1),
        golden.source(1, 0): _l1(golden, 2),
        golden.source(1, 1): _l1(golden, 20),
    }
    out = golden.run_l1(buffers, _config(golden, 2, tiles_per_output=2))
    # Dest = seed + B0, then Dest += B1: 1 + 2 + 20.
    assert _unpacked(golden, out) == [23.0]


def test_run_l1_names_the_inputs_a_sequence_cannot_supply():
    golden = QuasarEltwiseBinaryGolden(MathOperation.Elwadd, MathFidelity.LoFi)
    with pytest.raises(ValueError, match="missing.*in0_t1"):
        golden.run_l1(
            [_l1(golden, 1), _l1(golden, 2)], _config(golden, 2, tiles_per_output=2)
        )


def test_run_l1_refuses_a_buffer_the_chain_never_reads():
    golden = QuasarEltwiseBinaryGolden(MathOperation.Elwadd, MathFidelity.LoFi)
    buffers = [_l1(golden, 1), _l1(golden, 2), _l1(golden, 3)]
    with pytest.raises(ValueError, match="never reads.*in2"):
        golden.run_l1(buffers, _config(golden, 2))
