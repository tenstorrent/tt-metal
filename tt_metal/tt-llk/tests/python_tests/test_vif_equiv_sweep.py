# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bit-exact equivalence capture for the branch ldjurovic/vif_optimizations_wh.

The branch rewrites several Wormhole metal SFPU kernels from predicated (`v_if`)
form into SFPSWAP min/max, `copysgn`, or a cheaper condition code. Every one of
those rewrites is claimed to be *functionally identical*, not merely close, so
the check has to be a bit-for-bit comparison over the whole input domain rather
than a tolerance against a golden.

This module captures raw output **bit patterns** for each affected kernel and
dumps them to .npz tagged by $SFPU_EQUIV_TAG. Run it once with the baseline
headers checked out and once with the branch headers, then diff the two dumps
with `compare_vif_equiv.py`. Anything that differs in a single bit fails.

Bits, not floats: NaN != NaN under float comparison, and the NaN *payload* the
kernel emits is exactly the kind of thing a sign-bit rewrite could change. The
dumps therefore store int32/int16 views and compare with ==.

Domain coverage:

* float kernels -- all 65,279 distinct finite bf16 values (StimuliSpec.ulp_sweep)
  plus the 256 exp==0xFF patterns (2 infinities, 254 NaNs) run separately, since
  the sweep generator sorts and dedupes and would drop them.
* int32 kernels (calculate_comp_int) -- 2^32 is not enumerable, so a crafted set:
  every 16-bit low pattern, both sign-magnitude and two's-complement boundaries,
  and a fixed-seed random sample. Comparison to zero only reads the sign and the
  zero-ness, so this covers every equivalence class the kernel can distinguish.

Wormhole only; the rewrites are in hw/ckernels/wormhole_b0.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import wormhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import TILE_DIMENSIONS
from helpers.llk_params import (
    ApproximationMode,
    BlocksCalculationAlgorithm,
    DestAccumulation,
    FastMode,
    MathOperation,
    format_dict,
)
from helpers.param_config import get_num_blocks_and_num_tiles_in_block
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    CLAMP_NEGATIVE,
    FAST_MODE,
    MATH_OP,
    NUM_BLOCKS,
    NUM_TILES_IN_BLOCK,
    TILE_COUNT,
    DestSync,
    generate_input_dim,
)

_FACE_ELEMENTS = 16 * 16

# ─────────────────────────────────────────────────────────────────────────────
# Kernels the branch touches, grouped by the input domain they need.
#
# relu_min / relu_max are deliberately absent: the tt-llk harness routes
# MathOperation.ReluMin/ReluMax to tt-llk's own _relu_min_/_relu_max_ in
# tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h, not to the metal
# relu_min/relu_max this branch edits. Those two are covered from ttnn instead.
# ─────────────────────────────────────────────────────────────────────────────

_FLOAT_OPS = {
    "sign": MathOperation.Sign,
    "heaviside": MathOperation.Heaviside,
    "hardshrink": MathOperation.Hardshrink,
    "unary_eq": MathOperation.UnaryEq,
    "unary_ne": MathOperation.UnaryNe,
    "unary_gt": MathOperation.UnaryGt,
    "unary_lt": MathOperation.UnaryLt,
    "unary_ge": MathOperation.UnaryGe,
    "unary_le": MathOperation.UnaryLe,
}

_INT_OPS = {
    "eqz_int": MathOperation.EqualZero,
    "nez_int": MathOperation.NotEqualZero,
    "ltz_int": MathOperation.LessThanZero,
    "gtz_int": MathOperation.GreaterThanZero,
    "lez_int": MathOperation.LessThanEqualZero,
    "gez_int": MathOperation.GreaterThanEqualZero,
}

# bf16 in / fp32 out is the sensitive pair: it shows the kernel's own result
# before the pack quantises it, so a one-bit difference cannot hide.
_FLOAT_FORMATS = {
    "bf16_to_fp32": InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32),
    "bf16_to_bf16": InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
}

_INT_FORMAT = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)


def _nonfinite_patterns():
    """The 256 bf16 patterns with exp == 0xFF: 2 infinities and 254 NaNs."""
    bits = torch.arange(0, 2**16, dtype=torch.int32)
    vals = (bits.to(torch.int16)).view(torch.bfloat16).to(torch.float32)
    sel = ~torch.isfinite(vals)
    out = vals[sel]
    assert out.numel() == _FACE_ELEMENTS
    return out


# StimuliSpec.custom writes its values at the start of ONE face and zero-fills the
# rest, so a long list has to be handed over as custom_faces: 256 values per face,
# 4 faces per tile. 4 tiles = 16 faces = 4096 values, which sits comfortably inside
# the DEST budget.
_INT_TILES = 4
_INT_FACES = _INT_TILES * 4
_INT_VALUES = _INT_FACES * _FACE_ELEMENTS


def _int32_probe_values():
    """Crafted int32 domain for the comparison-to-zero kernels.

    Comparison to zero only distinguishes {negative, zero, positive}, but the
    kernel reaches that verdict through sign-magnitude / two's-complement
    handling, so the set pins every representation boundary explicitly and then
    fills the rest with a fixed-seed random bulk sample across the full range.
    """
    special = [
        0,
        1,
        -1,
        2,
        -2,
        0x7FFFFFFF,  # INT_MAX
        -0x80000000,  # INT_MIN, unrepresentable in sign-magnitude
        0x40000000,
        -0x40000000,
        0x00FFFFFF,
        -0x00FFFFFF,
        0x00010000,
        -0x00010000,
        0x7FFFFFFE,
        -0x7FFFFFFF,
        0x0000FFFF,
        -0x0000FFFF,
        0x00008000,
        -0x00008000,
    ]
    # Dense small magnitudes either side of zero: the region where sign-magnitude
    # and two's-complement encodings diverge most cheaply.
    dense = [v for m in range(1, 513) for v in (m, -m)]
    rng = np.random.default_rng(20260902)
    remaining = _INT_VALUES - len(special) - len(dense)
    assert remaining > 0
    bulk = rng.integers(
        -(2**31), 2**31 - 1, size=remaining, dtype=np.int64, endpoint=True
    ).tolist()
    vals = special + dense + bulk
    assert len(vals) == _INT_VALUES
    return vals


def _int32_face_spec(vals):
    return StimuliSpec.custom_faces(
        {
            f: vals[f * _FACE_ELEMENTS : (f + 1) * _FACE_ELEMENTS]
            for f in range(_INT_FACES)
        }
    )


def _unpack_to_dest(input_format: DataFormat, dest_acc: DestAccumulation) -> bool:
    return input_format.is_32_bit() and dest_acc == DestAccumulation.Yes


def _drive(mathop, formats, spec_A, num_tiles, dest_acc):
    """Run one kernel over one stimulus set and return the raw result tensor."""
    input_dimensions = [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * num_tiles]

    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        spec_A=spec_A,
    )

    num_blocks, num_tiles_in_block = get_num_blocks_and_num_tiles_in_block(
        DestSync.Half,
        dest_acc,
        formats,
        [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * tile_cnt_A],
        TILE_DIMENSIONS,
        BlocksCalculationAlgorithm.Standard,
    )

    configuration = TestConfig(
        "sources/eltwise_unary_sfpu_test.cpp",
        formats,
        templates=[
            generate_input_dim(
                [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * tile_cnt_A],
                [TILE_DIMENSIONS[0], TILE_DIMENSIONS[1] * tile_cnt_A],
            ),
            APPROX_MODE(ApproximationMode.No),
            FAST_MODE(FastMode.No),
            CLAMP_NEGATIVE(True),
            MATH_OP(mathop=mathop),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_BLOCKS(num_blocks),
            NUM_TILES_IN_BLOCK(num_tiles_in_block),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=_unpack_to_dest(formats.input_format, dest_acc),
    )
    res = configuration.run().result
    return src_A, torch.tensor(res, dtype=format_dict[formats.output_format])


def _as_bits(t: torch.Tensor, fmt: DataFormat) -> np.ndarray:
    """Reinterpret a result tensor as an integer bit pattern array."""
    if fmt == DataFormat.Float32:
        return t.to(torch.float32).view(torch.int32).numpy().astype(np.int64)
    if fmt == DataFormat.Float16_b:
        return (
            t.to(torch.float32).to(torch.bfloat16).view(torch.int16).numpy()
        ).astype(np.int64)
    if fmt == DataFormat.Int32:
        return t.to(torch.int32).numpy().astype(np.int64)
    raise AssertionError(f"no bit view for {fmt}")


def _dump(tag, name, x_bits, y_bits):
    out_dir = os.environ["SFPU_EQUIV_OUT"]
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{tag}__{name}.npz"
    np.savez_compressed(path, x=x_bits, y=y_bits)
    print(f"\nwrote {path} ({y_bits.size} points)")


def _tag():
    tag = os.environ.get("SFPU_EQUIV_TAG")
    if not tag or not os.environ.get("SFPU_EQUIV_OUT"):
        pytest.skip("set SFPU_EQUIV_OUT and SFPU_EQUIV_TAG")
    return tag


@wormhole_only
@pytest.mark.parametrize("domain", ("finite", "nonfinite"))
@pytest.mark.parametrize("fmt_name", list(_FLOAT_FORMATS))
@pytest.mark.parametrize("op_name", list(_FLOAT_OPS))
def test_equiv_float(op_name, fmt_name, domain):
    tag = _tag()
    formats = _FLOAT_FORMATS[fmt_name]

    if domain == "finite":
        spec_A = StimuliSpec.ulp_sweep(low=-math.inf, high=math.inf)
        num_tiles, expected = 64, 65279
    else:
        spec_A = StimuliSpec.custom(values=_nonfinite_patterns().tolist(), seed=0)
        num_tiles, expected = 1, _FACE_ELEMENTS

    src_A, res = _drive(
        _FLOAT_OPS[op_name], formats, spec_A, num_tiles, DestAccumulation.Yes
    )

    x_bits = _as_bits(src_A.to(torch.float32), DataFormat.Float16_b)[:expected]
    y_bits = _as_bits(res, formats.output_format)[:expected]
    _dump(tag, f"{op_name}__{fmt_name}__{domain}", x_bits, y_bits)


@wormhole_only
@pytest.mark.parametrize("op_name", list(_INT_OPS))
def test_equiv_int32(op_name):
    tag = _tag()
    spec_A = _int32_face_spec(_int32_probe_values())
    src_A, res = _drive(
        _INT_OPS[op_name], _INT_FORMAT, spec_A, _INT_TILES, DestAccumulation.Yes
    )

    x_bits = _as_bits(src_A, DataFormat.Int32)
    y_bits = _as_bits(res, DataFormat.Int32)
    _dump(tag, f"{op_name}__int32", x_bits, y_bits)
