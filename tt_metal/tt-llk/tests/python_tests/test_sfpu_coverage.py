# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lane EU LLK-coverage-expansion correctness vehicle.

Races the corpus-uncovered SFPU kernels (manifest class D-ABSENT: zero
dispatch anywhere under tests/ before this lane) against fresh semantic
bodies through sources/sfpu_coverage_test.cpp, with identical stimuli, golden
and tolerance per op (the S4/ED conversion discipline).  The production
kernels are byte-untouched — every selector lives test-side (LLK-pristine
rule R7).

Dst vector-row layout (PROVEN on silicon + pinned sim, lane FM 2026-08-22 —
golden re-spec per the CX cast precedent; supersedes lane EU's chunk-linear
assumption): in BOTH observed Dst modes (bf16 dest_acc=No AND Int32
dest_acc=Yes/unpack_to_dest), the SFPU vector-row PAIR dst_reg[2k] /
dst_reg[2k+1] holds the EVEN / ODD elements of the flat face-major
64-element span flat[64k : 64(k+1)] — NOT two consecutive 32-element
chunks.  Proofs (evidence laneFM-evidence-20260822/coverage-diag):

* rotate90 (bf16): tagged-stimulus runs — device and pinned-sim results
  both match ``out[i] = -in[i^1]`` (i even) / ``in[i^1]`` (i odd) at
  1024/1024 elements; the chunk-linear model matches 4/1024.
* int_sum ROW (Int32): tagged sim run decodes exactly to the interleave
  model (chunk c: even positions = the 4-vector sum, odd positions
  untouched); int_sum COL is LAYOUT-BLIND — all its summand offsets are
  even, making chunk-linear and interleaved predictions elementwise
  identical — so its earlier silicon PASS proved nothing about layout.

Row-structured goldens therefore use _vector_rows()/_from_vector_rows()
below.  Lane-uniform (elementwise) goldens are layout-independent.
"""

import struct

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import ApproximationMode, DestAccumulation, format_dict
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    COVERAGE_OP,
    COVERAGE_SUBOP,
    FRESH_CPP_IMPL,
    SFPU_UNARY_SCALAR,
)
from helpers.utils import passed_test


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


# Fixed op parameters — must match sources/sfpu_coverage_{test,perf}.cpp.
_BITWISE_SCALAR = 0x5A5A0FF0  # bit31 = 0 so 2's-comp == sign-magnitude == raw
_ADD_RSQRT_EPS = 0.5
_SMOOTHSTEP_EDGE0 = -0.5
_SMOOTHSTEP_INV_DELTA = 1.0
_ZERO_PAD_VALID_ROWS = 24
_TILED_PROD_ROWS = 9  # the production kernel's ITERATIONS(8)+1 row walk
_SKF_BANK_MASK = 0x3F
_SKF_MY_BANK = 5
_SKF_GLOBAL_BANK_SHIFT = 10
_SKF_WITHIN_BANK_MASK = 0x3FF
_SKF_OUT_SHIFT = 0

_IMPL_IDS = ["production", "fresh_cpp"]


def _vector_rows(flat):
    """(32, 32) view whose row v is SFPU vector row dst_reg[v] under the
    proven Dst layout (both 16-bit and 32-bit modes): rows 2k/2k+1 =
    even/odd elements of the flat 64-element span k (module docstring).
    Lane order within a row is not observable by (nor relevant to)
    lane-uniform goldens."""
    return flat.reshape(16, 32, 2).permute(0, 2, 1).reshape(32, 32)


def _from_vector_rows(rows):
    """Inverse of _vector_rows: vector-row tensor back to flat layout."""
    return rows.reshape(16, 2, 32).permute(0, 2, 1).reshape(1024)


def _run_coverage(
    op,
    fresh_cpp_impl,
    golden_fn,
    formats,
    dest_acc,
    subop=0,
    scalar_bits=None,
    spec_A=None,
    spec_B=None,
    atol=None,
    rtol=None,
):
    """Drive one coverage-op variant and gate it against *golden_fn*.

    golden_fn(src_A, src_B) -> flat golden tensor in the output format's
    value domain (row-structured goldens reshape to (32, 32) per the module
    docstring's Dst-row mapping).
    """
    torch.manual_seed(0)

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=[32, 32],
        stimuli_format_B=formats.input_format,
        input_dimensions_B=[32, 32],
        spec_A=spec_A,
        spec_B=spec_B,
    )

    # laneJO formal-equivalence witness-check hook (see test_sfpu_binary.py):
    # LANEJO_SRC_OVERRIDE holds {"src_A","src_B"} tensors replayed verbatim.
    import os as _lanejo_os

    _lanejo_src = _lanejo_os.environ.get("LANEJO_SRC_OVERRIDE")
    if _lanejo_src:
        _lanejo_t = torch.load(_lanejo_src)
        src_A = _lanejo_t["src_A"].to(src_A.dtype).reshape(src_A.shape)
        src_B = _lanejo_t["src_B"].to(src_B.dtype).reshape(src_B.shape)

    golden = golden_fn(src_A.flatten(), src_B.flatten())

    templates = [
        COVERAGE_OP(op),
        COVERAGE_SUBOP(subop),
        FRESH_CPP_IMPL(fresh_cpp_impl),
        APPROX_MODE(ApproximationMode.No),
    ]
    if scalar_bits is not None:
        templates.append(SFPU_UNARY_SCALAR(scalar_bits))

    configuration = TestConfig(
        "sources/sfpu_coverage_test.cpp",
        formats,
        templates=templates,
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_A.flatten(),
            formats.input_format,
            src_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        unpack_to_dest=formats.input_format.is_32_bit(),
        dest_acc=dest_acc,
        compile_time_formats=True,
    )

    res_from_L1 = configuration.run().result
    res_from_L1 = res_from_L1[:1024]

    assert len(res_from_L1) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    golden_tensor = golden.to(torch_format).flatten()
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format).flatten()

    # laneJO witness-check hook (paired with LANEJO_SRC_OVERRIDE above).
    _lanejo_dump = _lanejo_os.environ.get("LANEJO_DUMP")
    if _lanejo_dump:
        torch.save({"src_A": src_A, "src_B": src_B, "result": res_tensor}, _lanejo_dump)
    if _lanejo_os.environ.get("LANEJO_SKIP_ASSERT") == "1":
        return

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
        custom_atol=atol,
        custom_rtol=rtol,
    ), "Assert against golden failed"


_BF16 = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
_INT32 = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)

# Small non-negative int stimuli: keeps every input/output representation-
# agnostic (bit31 = 0 in 2's complement AND sign-magnitude) and int-sum
# accumulations far from overflow.
_INT_SPEC = StimuliSpec.uniform(low=0.0, high=1000.0)
# Index-shaped stimuli for the sparse-k filter: full 16-bit field span so both
# the bank hit and miss branches are exercised.
_SKF_SPEC = StimuliSpec.uniform(low=0.0, high=float(2**16 - 1))


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=_IMPL_IDS)
def test_sfpu_coverage_rotate90(fresh_cpp_impl):
    """metal alt_complex_rotate90 vs fresh: (re, im) row pairs -> (-im, re).

    Vector-row pairs under the PROVEN 16-bit Dst layout (module docstring):
    dst_reg[2k]/dst_reg[2k+1] are the even/odd elements of flat span k, so
    the rotation is elementwise ``out[i] = -in[i^1]`` (even i) /
    ``in[i^1]`` (odd i).  Bit-preserving contract, so the gate is exact."""

    def golden(a, _b):
        rows = _vector_rows(a).clone()
        out = rows.clone()
        out[0::2] = -rows[1::2]
        out[1::2] = rows[0::2]
        return _from_vector_rows(out)

    _run_coverage(
        1, fresh_cpp_impl, golden, _BF16, DestAccumulation.No, atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=_IMPL_IDS)
@pytest.mark.parametrize("subop", [0, 1, 2], ids=["AND", "OR", "XOR"])
def test_sfpu_coverage_unary_bitwise(subop, fresh_cpp_impl):
    """metal unary bitwise (and/or/xor with an int scalar) vs fresh.

    Exact integer contract on non-negative int32 stimuli (bit31 = 0 keeps the
    dest sign-magnitude/2's-complement question out of the contract)."""

    def golden(a, _b):
        x = a.to(torch.int64)
        if subop == 0:
            y = x & _BITWISE_SCALAR
        elif subop == 1:
            y = x | _BITWISE_SCALAR
        else:
            y = x ^ _BITWISE_SCALAR
        return y.to(torch.int32)

    _run_coverage(
        2,
        fresh_cpp_impl,
        golden,
        _INT32,
        DestAccumulation.Yes,
        subop=subop,
        scalar_bits=_BITWISE_SCALAR,
        spec_A=_INT_SPEC,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=_IMPL_IDS)
def test_sfpu_coverage_add_rsqrt(fresh_cpp_impl):
    """metal experimental add_rsqrt (RMSNorm idiom) vs fresh: 1/sqrt(x+eps).

    Both arms are quadratic-refined Newton rsqrt approximations of the same
    fp32 target; the bf16 store quantum (~2^-8 relative) dominates their
    residuals, so the default-magnitude 5% rtol + 0.05 atol gate is ample and
    still rejects a wrong epsilon or a missing refinement step."""

    def golden(a, _b):
        return torch.rsqrt(a.to(torch.float32) + _ADD_RSQRT_EPS)

    _run_coverage(
        3,
        fresh_cpp_impl,
        golden,
        _BF16,
        DestAccumulation.No,
        scalar_bits=_bits(_ADD_RSQRT_EPS),
        spec_A=StimuliSpec.uniform(low=0.05, high=6.0),
        atol=0.05,
        rtol=0.05,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=_IMPL_IDS)
def test_sfpu_coverage_smoothstep(fresh_cpp_impl):
    """metal experimental smoothstep vs fresh: t^2(3-2t) on the clamped ramp.

    Fixed edges (-0.5, 0.5) => inv_delta 1; smooth polynomial in [0, 1], the
    format-default bf16 tolerance carries it."""

    def golden(a, _b):
        t = torch.clamp(
            (a.to(torch.float32) - _SMOOTHSTEP_EDGE0) * _SMOOTHSTEP_INV_DELTA, 0.0, 1.0
        )
        return t * t * (3.0 - 2.0 * t)

    _run_coverage(
        4,
        fresh_cpp_impl,
        golden,
        _BF16,
        DestAccumulation.No,
        spec_A=StimuliSpec.uniform(low=-1.0, high=1.0),
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=_IMPL_IDS)
def test_sfpu_coverage_tiled_prod(fresh_cpp_impl):
    """metal tiled_prod vs fresh: running elementwise product down vector rows.

    The production walk covers ITERATIONS+1 = 9 rows per call (its documented
    off-by-one); rows 9..31 pass through.  Stimuli in [0.5, 1.5] bound the
    9-term product well inside bf16 range."""

    def golden(a, _b):
        rows = _vector_rows(a).to(torch.float32)
        out = rows.clone()
        run = torch.ones(32)
        for r in range(_TILED_PROD_ROWS):
            run = run * rows[r]
            out[r] = run
        return _from_vector_rows(out)

    _run_coverage(
        5,
        fresh_cpp_impl,
        golden,
        _BF16,
        DestAccumulation.No,
        spec_A=StimuliSpec.uniform(low=0.5, high=1.5),
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=_IMPL_IDS)
def test_sfpu_coverage_zero_pad(fresh_cpp_impl):
    """legacy experimental zero_pad vs fresh: scrub rows [VALID, 32) to +0.0.

    Pure constant store — exact gate.  Vector rows via the proven 16-bit
    layout helper; with VALID=24 (even) the zeroed region happens to be the
    same flat[768:1024] the old chunk-linear assumption gave — which is why
    this row passed on silicon while rotate90/tiledprod failed."""

    def golden(a, _b):
        rows = _vector_rows(a).clone()
        rows[_ZERO_PAD_VALID_ROWS:] = 0.0
        return _from_vector_rows(rows)

    _run_coverage(
        6, fresh_cpp_impl, golden, _BF16, DestAccumulation.No, atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=_IMPL_IDS)
def test_sfpu_coverage_sparse_k_filter(fresh_cpp_impl):
    """legacy experimental sparse_k_filter vs fresh: bank-addressed index
    filter, y = bank-hit ? (within-bank slot + 1) << shift : 0.

    Exact integer contract; the 16-bit stimuli span exercises both branches
    (bank field = bits [10, 16))."""

    def golden(a, _b):
        x = a.to(torch.int64)
        bank = (x >> _SKF_GLOBAL_BANK_SHIFT) & _SKF_BANK_MASK
        local = x & _SKF_WITHIN_BANK_MASK
        y = torch.where(
            bank == _SKF_MY_BANK, (local + 1) << _SKF_OUT_SHIFT, torch.zeros_like(x)
        )
        return y.to(torch.int32)

    _run_coverage(
        7,
        fresh_cpp_impl,
        golden,
        _INT32,
        DestAccumulation.Yes,
        spec_A=_SKF_SPEC,
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=_IMPL_IDS)
def test_sfpu_coverage_custom_add(fresh_cpp_impl):
    """metal experimental custom_add vs fresh: two-tile elementwise a + b.

    Second operand rides buffer_B into Dst tile 1; format-default bf16
    tolerance absorbs the hand arm's truncating store vs the golden's RNE."""

    def golden(a, b):
        return a.to(torch.float32) + b.to(torch.float32)

    _run_coverage(8, fresh_cpp_impl, golden, _BF16, DestAccumulation.No)


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=_IMPL_IDS)
def test_sfpu_coverage_copy_dest(fresh_cpp_impl):
    """metal copy_dest_values vs fresh: identity move tile 0 -> tile 1.

    The packed output is Dst tile 1; a bf16 load/store round-trip is exact,
    so the gate is exact."""

    def golden(a, _b):
        return a.clone()

    _run_coverage(
        9, fresh_cpp_impl, golden, _BF16, DestAccumulation.No, atol=0.0, rtol=0.0
    )


@pytest.mark.parametrize("fresh_cpp_impl", [0, 1], ids=_IMPL_IDS)
@pytest.mark.parametrize("subop", [0, 1], ids=["COL", "ROW"])
def test_sfpu_coverage_int_sum(subop, fresh_cpp_impl):
    """metal int_sum vs fresh: strided in-tile int32 reductions.

    COL: rows {i, i+2..i+6, i+16..i+22} summed into row i for i in {0, 1};
    ROW: rows {i, i+1, i+8, i+9} summed into row i for i in {0, 2, 4, 6};
    all other rows unchanged.  Exact integer contract on small non-negative
    stimuli (sums stay far below 2^31)."""

    def golden(a, _b):
        # Vector-row basis via the proven interleaved layout (module
        # docstring).  COL is layout-blind (all-even offsets); ROW is not —
        # its chunk-linear form failed the pinned sim before the re-spec.
        rows = _vector_rows(a).to(torch.int64)
        out = rows.clone()
        if subop == 0:
            for i in range(2):
                acc = rows[i].clone()
                for j in (2, 4, 6, 16, 18, 20, 22):
                    acc = acc + rows[i + j]
                out[i] = acc
        else:
            for i in range(0, 8, 2):
                out[i] = rows[i] + rows[i + 1] + rows[i + 8] + rows[i + 9]
        return _from_vector_rows(out).to(torch.int32)

    _run_coverage(
        10,
        fresh_cpp_impl,
        golden,
        _INT32,
        DestAccumulation.Yes,
        subop=subop,
        spec_A=_INT_SPEC,
        atol=0.0,
        rtol=0.0,
    )
