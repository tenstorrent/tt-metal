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
from helpers.golden_generator.heavyweight.data_transfer_blocks import (
    UnmodelledHardwareWarning,
)
from helpers.golden_generator.heavyweight.operations.blackhole_operations.blackhole_matmul import (
    BlackholeMatmulGolden,
)
from helpers.golden_generator.heavyweight.operations.quasar_operations.quasar_datacopy import (
    QuasarDataCopyGolden,
)
from helpers.golden_generator.heavyweight.operations.quasar_operations.quasar_eltwise import (
    QuasarEltwiseBinaryGolden,
)
from helpers.golden_generator.heavyweight.operations.quasar_operations.quasar_matmul import (
    QuasarMatmulGolden,
)
from helpers.golden_generator.heavyweight.operations.quasar_operations.quasar_reuse_dest import (
    QuasarEltwiseBinaryReuseDestGolden,
)
from helpers.golden_generator.heavyweight.operations.wormhole_operations.wormhole_eltwise import (
    WormholeEltwiseBinaryGolden,
)
from helpers.golden_generator.heavyweight.operations.wormhole_operations.wormhole_matmul import (
    WormholeMatmulGolden,
)
from helpers.llk_params import EltwiseBinaryReuseDestType, MathFidelity, MathOperation
from helpers.tile_constants import MAX_TILE_ELEMENTS
from helpers.tilize_untilize import tilize_block, untilize_block

TILE = MAX_TILE_ELEMENTS


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


# ---------------------------------------------------------------------------
# MatmulGolden


def _tilize(matrix):
    """A logical 32x32 matrix as the face-ordered tile a src register holds."""
    return tilize_block(
        matrix.reshape(-1).float(),
        stimuli_format=DataFormat.Float32,
        dimensions=[32, 32],
        tile_dimensions=[32, 32],
    ).flatten()


def _untilize(values):
    return untilize_block(
        values.float(),
        stimuli_format=DataFormat.Float32,
        dimensions=[32, 32],
        tile_dimensions=[32, 32],
    ).reshape(32, 32)


def test_matmul_computes_arg0_at_arg1_not_its_transpose():
    """Operand routing and product order have to agree, and both are load-bearing.

    `OPERAND_REGISTERS` puts arg0 in SrcB and `_product` computes SrcB @ SrcA.
    Flipping one without the other transposes the result -- wrong everywhere but
    still plausible, which is how it was last caught. A non-symmetric operand is
    what makes the two distinguishable at all.
    """
    # Small integers and a one-hot shift: every product is exact, so this
    # tests operand order rather than the fidelity split's accuracy.
    a = torch.arange(1.0, 32 * 32 + 1).reshape(32, 32) % 7 - 3
    b = torch.eye(32).roll(1, dims=1)
    out = QuasarMatmulGolden(MathFidelity.HiFi4).run(
        [_tilize(a), _tilize(b)],
        DataFormat.Float32,
        DataFormat.Float32,
        dest_acc=True,
    )
    got = _untilize(out.flatten())
    assert torch.equal(got, a @ b), "not arg0 @ arg1"
    assert not torch.equal(got, b @ a), "operands are interchangeable here"


def test_matmul_writes_dest_once_per_k_face_per_phase():
    """One MVMUL is D[8,16] += B[8,16] * A[16,16], so a 32-wide K takes two,
    and the MOP nests them inside the fidelity loop (phase-outer, K-face-inner).
    Summing all 32 K terms before one Dest write would drop a rounding per phase."""
    golden = QuasarMatmulGolden(MathFidelity.HiFi4)
    golden.run([torch.ones(TILE)] * 2, DataFormat.Float16_b, DataFormat.Float16_b)
    math_steps = [s.name for s in golden.last_chain if s.name.startswith("matmul")]
    assert math_steps == [
        f"matmul[p{p}k{k}]" + ("+=" if (p, k) != (0, 0) else "")
        for p in range(4)
        for k in range(2)
    ]


def test_matmul_refuses_a_tile_that_is_not_32x32():
    golden = QuasarMatmulGolden(MathFidelity.LoFi)
    with pytest.raises(ValueError, match="one 32x32 tile at a time"):
        golden.run(
            [torch.ones(512)] * 2,
            DataFormat.Float16_b,
            DataFormat.Float16_b,
            num_faces=2,
        )


def test_matmul_fidelity_changes_the_answer():
    """LoFi keeps 7 mantissa bits of each operand, HiFi4 all four phases, so a
    product needing the low bits must differ between them."""
    torch.manual_seed(0)
    a = torch.randn(32, 32)
    b = torch.randn(32, 32)
    runs = {
        fid: QuasarMatmulGolden(fid).run(
            [_tilize(a), _tilize(b)],
            DataFormat.Float32,
            DataFormat.Float32,
            dest_acc=True,
        )
        for fid in (MathFidelity.LoFi, MathFidelity.HiFi4)
    }
    assert not torch.equal(runs[MathFidelity.LoFi], runs[MathFidelity.HiFi4])
    exact = a @ b
    assert (_untilize(runs[MathFidelity.HiFi4].flatten()) - exact).abs().max() < (
        _untilize(runs[MathFidelity.LoFi].flatten()) - exact
    ).abs().max()


# ---------------------------------------------------------------------------
# DataCopyGolden, the architecture bindings, and chain checking


def test_datacopy_round_trips_its_input():
    out = QuasarDataCopyGolden().run(
        [torch.full((TILE,), 1.5)], DataFormat.Float16_b, DataFormat.Float16_b
    )
    assert out.flatten().tolist() == [1.5] * TILE


def test_a_second_operand_a_chain_never_reads_is_refused():
    """The single-tile path checks this the way run_l1 and the blocked path do."""
    with pytest.raises(ValueError, match="never reads"):
        QuasarDataCopyGolden().run(
            [torch.ones(TILE), torch.ones(TILE)],
            DataFormat.Float16_b,
            DataFormat.Float16_b,
        )


@pytest.mark.parametrize(
    "golden_class",
    [WormholeMatmulGolden, BlackholeMatmulGolden],
    ids=lambda c: c.__name__,
)
def test_wormhole_and_blackhole_warn_that_fidelity_is_unmodelled(golden_class):
    """MANTISSA_SPLIT is unset there, so the multiply is exact and math_fidelity
    is ignored. The warning is the only signal, and it has to survive
    pytest.ini's ignore::UserWarning -- hence a RuntimeWarning subclass."""
    golden = golden_class(MathFidelity.HiFi4)
    with pytest.warns(UnmodelledHardwareWarning, match="exact product"):
        golden.run([torch.ones(TILE)] * 2, DataFormat.Float16_b, DataFormat.Float16_b)


def test_wormhole_promotes_the_outlier_combination_to_a_32_bit_dest():
    """TestConfig forces dest_acc on for an exponent-B input with a Float16
    output on every architecture but Quasar, so the device runs a 32-bit Dest
    whatever the test asked for. Quasar is exempt and keeps the bf16 Dest."""
    a = torch.full((TILE,), 1 + 2**-7)
    b = torch.ones(TILE)
    wh = WormholeEltwiseBinaryGolden(MathOperation.Elwadd, MathFidelity.HiFi4)
    qs = QuasarEltwiseBinaryGolden(MathOperation.Elwadd, MathFidelity.HiFi4)
    # Elwadd, so no unmodelled-split warning: the mantissa split applies to a
    # multiply, and eltwise only warns for Elwmul.
    out_wh = wh.run([a, b], DataFormat.Float16_b, DataFormat.Float16)
    out_qs = qs.run([a, b], DataFormat.Float16_b, DataFormat.Float16)
    assert wh.last_dest_format is DataFormat.Float32
    assert out_wh.flatten()[0].item() == 1 + 1 + 2**-7
    assert qs.last_dest_format is DataFormat.Float16_b
    assert out_qs.flatten()[0].item() == 2.0


def test_dry_run_catches_a_step_reading_a_register_nothing_wrote():
    golden = QuasarMatmulGolden(MathFidelity.LoFi)
    chain = golden.build_chain(_config(golden, 2))
    assert chain.dry_run(["in0", "in1"]) == []
    assert chain.dry_run(["in0"]), "a missing input should be reported"


def test_last_chain_and_dest_format_are_readable_before_any_run():
    golden = QuasarMatmulGolden(MathFidelity.LoFi)
    assert golden.last_chain is None
    assert golden.last_dest_format is None
