# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
H128 (1x128) Hadamard transform LLK test (Blackhole only).

Every operand is a bf16 face. The 128 input values live in face rows 0..7 (row-major, so
element k is row k//16, column k%16) and the result comes back the same way. H_16 fills
its whole face.

Most tests here drive x with signs only (+-1), which lets the result be compared for
equality rather than against a tolerance. Only normalized variants bring in tolerance.
"""

import math

import pytest
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import HadamardH128Golden, get_golden_generator
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import StimuliMode, TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    HADAMARD,
    MATH_FIDELITY,
    NUM_FACES,
    TILE_COUNT,
)
from helpers.utils import tolerances

pytestmark = blackhole_only

FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)

H128_LEN = 128
FACE_ELEMENTS = 16 * 16
TILE_SLOT_ELEMENTS = 1024

NORM_SCALE = 1.0 / math.sqrt(128.0)

# All four fidelities. LoFi issues one MVMUL per matmul pass; HiFi2/3/4 all issue two with the
# asymmetric phase step, so they are expected to be indistinguishable in most tests.
FIDELITIES = [
    MathFidelity.LoFi,
    MathFidelity.HiFi2,
    MathFidelity.HiFi3,
    MathFidelity.HiFi4,
]


def _sylvester(order):
    """The Sylvester Hadamard matrix, as float32."""
    return HadamardH128Golden.sylvester(order)


def _tile(values):
    """A 1024-element L1 tile slot whose face 0 holds `values`."""
    tile = torch.zeros(TILE_SLOT_ELEMENTS, dtype=torch.bfloat16)
    values = values.reshape(-1).to(torch.bfloat16)
    tile[: values.numel()] = values
    return tile


def _h16_tile():
    return _tile(_sylvester(16))


def _input_tile(x, padding=None):
    """The input tile for one transform. `padding` optionally places arbitrary values in unused datums."""
    tile = _tile(x)
    if padding is not None:
        tile[H128_LEN:FACE_ELEMENTS] = padding.reshape(-1).to(torch.bfloat16)
    return tile


def _signs(seed, count=1):
    """`count` input vectors of +-1."""
    generator = torch.Generator().manual_seed(seed)
    bits = torch.randint(
        0, 2, (count, H128_LEN), generator=generator, dtype=torch.int32
    )
    return (bits * 2 - 1).to(torch.float32)


def _config(
    inputs,
    normalize=True,
    fidelity=MathFidelity.HiFi4,
    dest_sync=DestSync.Half,
    dest_acc=DestAccumulation.No,
    formats=FORMATS,
    h16_tile_index=0,
    padding=None,
):
    """One kernel variant. `inputs` is a [num_tiles, 128] tensor, one transform per row."""
    num_tiles = inputs.shape[0]

    # buffer_A is the h16 operand and buffer_B the inputs, matching the LLKs' own operand naming
    # (operandA -> SrcB is h16, operandB -> SrcA is the input). A non-zero h16_tile_index needs
    # preceding slots to exist, so pad buffer_A with tiles the op must never read.
    h16_tiles = [_tile(torch.full((FACE_ELEMENTS,), 7.0))] * h16_tile_index + [
        _h16_tile()
    ]

    return TestConfig(
        "sources/hadamard_test.cpp",
        formats,
        templates=[
            HADAMARD(hadamard_normalize=normalize, h16_tile_index=h16_tile_index),
            MATH_FIDELITY(fidelity),
            DEST_SYNC(dest_sync),
        ],
        runtimes=[TILE_COUNT(num_tiles), NUM_FACES(1, 1, 1)],
        variant_stimuli=StimuliConfig(
            torch.cat(h16_tiles),
            formats.input_format,
            torch.cat([_input_tile(x, padding) for x in inputs]),
            formats.input_format,
            formats.output_format,
            tile_count_A=len(h16_tiles),
            tile_count_B=num_tiles,
            tile_count_res=num_tiles,
            num_faces=1,
        ),
        dest_acc=dest_acc,
    )


def _run(*configurations):
    """Run one or more variants and return their results as [num_tiles, 128] each.

    This exists for test_hadamard_h128_fidelity_precision, since it compares two variants
    against each other: every variant has to be built before any of them runs.
    """
    if len(configurations) > 1 and TestConfig.STIMULI_MODE != StimuliMode.INLINE:
        # The stimuli cache keys on the pytest node id, one slot per test, so this would
        # collapse into comparing a stimulus with itself under that.
        pytest.skip("this test cannot use the stimuli cache")
    for configuration in configurations:
        configuration.prepare()
    results = [_transforms(c.run().result) for c in configurations]
    return results[0] if len(results) == 1 else results


def _transforms(result):
    """The packed results as [num_tiles, 128], float32."""
    faces = torch.tensor(result, dtype=torch.float32).reshape(-1, FACE_ELEMENTS)
    return faces[:, :H128_LEN]


def _golden(inputs, normalize):
    golden = get_golden_generator(HadamardH128Golden)
    return torch.stack([golden(x, normalize=normalize) for x in inputs])


def _assert_exact(device, inputs):
    expected = _golden(inputs, normalize=False)
    assert torch.equal(device, expected), (
        "H128 result is not exactly H_128 @ x\n"
        f"first mismatching row {int((device != expected).any(dim=1).nonzero()[0])}\n"
        f"device={device[(device != expected).any(dim=1)][0][:16]}\n"
        f"golden={expected[(device != expected).any(dim=1)][0][:16]}"
    )


def _assert_close(device, expected, output_format, what):
    tolerance = tolerances[output_format]
    error = (device - expected).abs()
    bound = tolerance.atol + tolerance.rtol * expected.abs()
    assert bool((error <= bound).all()), (
        f"{what}: max error {float(error.max())} exceeds the "
        f"{output_format.name} tolerance (atol={tolerance.atol}, rtol={tolerance.rtol})"
    )


@parametrize(fidelity=FIDELITIES, normalize=[False, True], num_tiles=[1, 8])
def test_hadamard_h128(fidelity, normalize, num_tiles):
    inputs = _signs(seed=101, count=num_tiles)

    device = _run(_config(inputs, normalize=normalize, fidelity=fidelity))

    if normalize:
        _assert_close(
            device,
            _golden(inputs, normalize=True),
            FORMATS.output_format,
            "normalized H128",
        )
    else:
        _assert_exact(device, inputs)


@parametrize(
    exponent=[-8, -1, 0, 1, 8], fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4]
)
def test_hadamard_h128_exponents(exponent, fidelity):
    inputs = _signs(seed=303) * (2.0**exponent)
    device = _run(_config(inputs, normalize=False, fidelity=fidelity))
    _assert_exact(device, inputs)


# The 128 inputs occupy face rows 0..7 and rows 8..15 are ignored. The latter is tested here.
def test_hadamard_h128_ignores_l1_padding():
    inputs = _signs(seed=404)
    padding = torch.full((FACE_ELEMENTS - H128_LEN,), -1024.0)
    device = _run(_config(inputs, normalize=False, padding=padding))
    _assert_exact(device, inputs)


# The unpack init preprograms config context 1's SrcA base from base + tile_size * h16_tile_index
# once, and phase 2 then streams H_16 from it with no per-tile CFG write. Putting H_16 at index 1
# exercises the address arithmetic.
def test_hadamard_h128_h16_tile_index():
    inputs = _signs(seed=505, count=2)

    device = _run(_config(inputs, normalize=False, h16_tile_index=1))

    _assert_exact(device, inputs)


# Full bf16 mantissa, so fidelity is relevant. LoFi keeps 4 of SrcA's 7 stored mantissa bits on MM1
# and drops the intermediate's lowest bit on MM2, while HiFi pairs ({0,1} for MM1, {0,2} for MM2)
# reconstruct both operands whole. HiFi must be strictly closer to the exact transform than LoFi,
# and LoFi must still be within mantissa truncation bounds.
def test_hadamard_h128_fidelity_precision():
    generator = torch.Generator().manual_seed(606)
    inputs = torch.randn(1, H128_LEN, generator=generator, dtype=torch.float32)
    inputs = inputs.to(torch.bfloat16).to(torch.float32)
    expected = _golden(inputs, normalize=False)

    lofi, hifi = _run(
        _config(inputs, normalize=False, fidelity=MathFidelity.LoFi),
        _config(inputs, normalize=False, fidelity=MathFidelity.HiFi4),
    )

    lofi_error = float((lofi - expected).abs().max())
    hifi_error = float((hifi - expected).abs().max())

    # Dropping the low 3 of 7 stored mantissa bits costs at most 2^-4 relative per input.
    # Summing 128 of them bounds the absolute error by that times sum|x|.
    bound = (2.0**-4) * float(inputs.abs().sum())
    assert (
        lofi_error <= bound
    ), f"LoFi error {lofi_error} exceeds the mantissa-truncation bound {bound}"
    assert (
        hifi_error < lofi_error
    ), f"HiFi4 is not more accurate than LoFi (hifi={hifi_error}, lofi={lofi_error})"


# The configuration the Compute API documents: bfp8_b out, and both DEST sync modes. bfp8_b shares
# one exponent per 16 datums, so this is a tolerance check even for the +-1 stimuli.
@parametrize(dest_sync=[DestSync.Half, DestSync.Full], normalize=[False, True])
def test_hadamard_h128_shipping_config(dest_sync, normalize):
    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Bfp8_b)
    inputs = _signs(seed=707, count=2)

    device = _run(
        _config(inputs, normalize=normalize, dest_sync=dest_sync, formats=formats)
    )

    _assert_close(
        device,
        _golden(inputs, normalize=normalize),
        formats.output_format,
        f"bfp8_b H128 (dest_sync={dest_sync.name})",
    )


FP32_FORMATS = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float32)


# 32-bit DEST is reachable from the Compute API: hadamard_h128_init takes fp32_dest_acc_en and defaults
# it to DST_ACCUM_MODE, so any kernel built with fp32 accumulation lands here. The matmul half of the op
# is correct under it, which is what this covers. `normalize` is unsupported for this configuration.
@parametrize(num_tiles=[1, 2, 4])
def test_hadamard_h128_dest_acc(num_tiles):
    (num_tiles,) = num_tiles
    inputs = _signs(seed=808, count=num_tiles)

    device = _run(
        _config(
            inputs,
            normalize=False,
            dest_acc=DestAccumulation.Yes,
            formats=FP32_FORMATS,
        )
    )

    _assert_exact(device, inputs)
