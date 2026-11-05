# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""GGUF weight decoders on the SFPU and the DEST-fed matmul (Blackhole).

Covers sfpu/experimental/ckernel_sfpu_gguf_decode.h (all 25 formats), experimental/ckernel_gguf_page.h (the fp32 scale
copies and the codebook grid lookups, which the kernel runs on the pack thread) and
experimental/llk_{math,unpack}_dest_srca_mm.h, through sources/gguf_decode_mm_test.cpp:
  - test_gguf_decode: random ggml blocks -> records -> pages -> decoded Float32 [K][N] tiles, equal to ggml's
    dequantization (gguf-py) value for value;
  - test_gguf_decode_mm: the same weights decoded in DEST and multiplied with bf16 in0 rows;
  - test_dest_srca_mm: the matmul alone, on Float32 in1 tiles unpacked to DEST.
The matmul golden truncates in1 to TF32 (MOVD2A) and multiplies each fidelity phase's operand bits as the FPU does.
"""

import math

import numpy as np
import torch
from conftest import blackhole_only
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.gguf_records import (
    FORMATS,
    dequantize,
    level_table,
    pages,
    random_blocks,
    records,
    tiles_kn,
)
from helpers.golden_generators import FidelityMasking
from helpers.llk_params import DestAccumulation, DestSync, MathFidelity
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import DEST_SYNC, GGUF_DECODE_MM, MATH_FIDELITY
from helpers.utils import passed_test

pytestmark = blackhole_only

TILE = 32
TILE_BYTES_32BIT = 4096
TILE_BYTES_BF16 = 2048
FIDELITY_PHASES = {
    MathFidelity.LoFi: 1,
    MathFidelity.HiFi2: 2,
    MathFidelity.HiFi3: 3,
    MathFidelity.HiFi4: 4,
}


def _k_unit(fmt):
    """The smallest K that holds whole groups and whole blocks."""
    return math.lcm(TILE * fmt.group_tiles, fmt.block_elems)


def _decode_shapes(fmt):
    """[N, K]: one N tile and one group, and two N tiles over at least eight K tiles."""
    unit = _k_unit(fmt)
    return [(TILE, unit), (2 * TILE, max(2 * unit, 8 * TILE))]


def _tilize(matrix: np.ndarray) -> np.ndarray:
    """[32, 32] -> the four 16x16 faces in L1 order."""
    return matrix.reshape(2, 16, 2, 16).transpose(0, 2, 1, 3).reshape(-1)


def _untilize(flat: torch.Tensor, tiles: int) -> torch.Tensor:
    return (
        flat.reshape(tiles, 2, 2, 16, 16)
        .permute(0, 1, 3, 2, 4)
        .reshape(tiles, TILE, TILE)
    )


def _weight_pages(fmt, blocks):
    """All pages of the weight, N tile by N tile, as one L1 image."""
    return b"".join(
        b"".join(tile_pages) for tile_pages in pages(fmt, records(fmt, blocks))
    )


def _in0_tiles(in0: np.ndarray) -> bytes:
    """[rows, K] bf16 values -> K / 32 tiles of 32 rows (rows past `rows` zero), 4 faces each."""
    rows, k = in0.shape
    padded = np.zeros((TILE, k), dtype=np.float32)
    padded[:rows] = in0
    bits = torch.from_numpy(padded).to(torch.bfloat16).view(torch.int16).numpy()
    return b"".join(
        _tilize(bits[:, TILE * kt : TILE * (kt + 1)]).tobytes()
        for kt in range(k // TILE)
    )


def _run(
    fmt_id,
    kt,
    nt,
    a_image,
    a_format,
    c_image,
    output_tiles,
    matmul=False,
    in0_rows=32,
    in0_image=None,
    fidelity=MathFidelity.HiFi2,
):
    in0_faces = 2 if in0_rows <= 16 else 4
    in0_image = in0_image if in0_image is not None else bytes(TILE_BYTES_BF16)
    configuration = TestConfig(
        "sources/gguf_decode_mm_test.cpp",
        InputOutputFormat(a_format, DataFormat.Float32),
        templates=[
            GGUF_DECODE_MM(
                fmt_id=fmt_id,
                kt=kt,
                nt=nt,
                matmul=matmul,
                in0_rows=in0_rows,
                in0_faces=in0_faces,
            ),
            MATH_FIDELITY(fidelity),
            DEST_SYNC(DestSync.Full),
        ],
        # Prepacked L1 images: the tile counts only reserve each operand's region.
        variant_stimuli=StimuliConfig(
            buffer_A=a_image,
            stimuli_A_format=a_format,
            tile_count_A=-(-len(a_image) // TILE_BYTES_32BIT),
            buffer_B=in0_image,
            stimuli_B_format=DataFormat.Float16_b,
            tile_count_B=len(in0_image) // TILE_BYTES_BF16,
            buffer_C=c_image,
            stimuli_C_format=DataFormat.UInt32,
            tile_count_C=-(-len(c_image) // TILE_BYTES_32BIT),
            stimuli_res_format=DataFormat.Float32,
            tile_count_res=output_tiles,
        ),
        dest_acc=DestAccumulation.Yes,
        disable_format_inference=True,
    )
    result = torch.tensor(configuration.run().result, dtype=torch.float32)
    return _untilize(result, output_tiles)


def _table_image(fmt) -> bytes:
    return level_table(fmt).tobytes() if fmt.codebook else bytes(16)


def _tf32_matmul_golden(
    in0: torch.Tensor, in1: torch.Tensor, fidelity: MathFidelity
) -> torch.Tensor:
    """in0 [M, K] @ in1 [K, N] as the FPU computes it from SrcB = in0 and SrcA = in1 truncated to TF32: the sum over
    the fidelity phases of the products of each phase's mantissa bits, accumulated in fp64.
    """
    masking = FidelityMasking()
    result = torch.zeros(in0.shape[0], in1.shape[1], dtype=torch.float64)
    for phase in range(FIDELITY_PHASES[fidelity]):
        srca, srcb = masking._apply_fidelity_masking(
            DataFormat.Float32, in1, in0, phase
        )
        result += srcb.to(torch.float64) @ srca.to(torch.float64)
    return result


def _assert_matmul(
    device: torch.Tensor, in0: torch.Tensor, in1: torch.Tensor, fidelity: MathFidelity
):
    """device: the output tiles [nt][32][32]; in0 [rows, K]; in1 [K, N]."""
    rows = in0.shape[0]
    nt = in1.shape[1] // TILE
    golden = torch.zeros(TILE, nt * TILE, dtype=torch.float64)
    golden[:rows] = _tf32_matmul_golden(in0, in1, fidelity)
    result = device.permute(1, 0, 2).reshape(TILE, nt * TILE).to(torch.float64)
    # FPU accumulation error only: the golden models the operands bit for bit.
    magnitude = torch.zeros_like(golden)
    magnitude[:rows] = in0.abs().to(torch.float64) @ in1.abs().to(torch.float64)
    atol = 2.0**-16 * magnitude.max().item()
    worst = (result - golden).abs().argmax().item()
    row, col = divmod(worst, golden.shape[1])
    assert passed_test(
        golden, result, DataFormat.Float32, custom_atol=atol, custom_rtol=2.0**-16
    ), (
        f"largest difference at row {row}, column {col}: device {result[row, col].item()}, golden {golden[row, col].item()}"
        f" (atol {atol})"
    )


@parametrize(fmt=list(FORMATS), shape=lambda fmt: _decode_shapes(FORMATS[fmt]))
def test_gguf_decode(fmt, shape):
    f = FORMATS[fmt]
    n, k = shape
    nt, kt = n // TILE, k // TILE
    blocks = random_blocks(f, n, k, np.random.default_rng(1000 + f.id))
    device = _run(
        f.id,
        kt,
        nt,
        _weight_pages(f, blocks),
        DataFormat.UInt32,
        _table_image(f),
        nt * kt,
    )

    weights = dequantize(f, blocks)
    for t in range(nt):
        for j in range(kt):
            golden = torch.from_numpy(np.ascontiguousarray(tiles_kn(weights, t, j)))
            # Equal values; a zero may differ in sign (q * S + M against ggml's (q - m) * d).
            assert torch.equal(device[t * kt + j], golden), (
                f"{fmt}: tile (n {t}, k {j}) differs at {torch.nonzero(device[t * kt + j] != golden)[:8].tolist()}:"
                f" device {device[t * kt + j][device[t * kt + j] != golden][:8].tolist()},"
                f" golden {golden[device[t * kt + j] != golden][:8].tolist()}"
            )


@parametrize(fmt=list(FORMATS), rows=[1, 13, 32])
def test_gguf_decode_mm(fmt, rows):
    f = FORMATS[fmt]
    n, k = 2 * TILE, max(_k_unit(f), 8 * TILE)
    nt, kt = n // TILE, k // TILE
    rng = np.random.default_rng(2000 + 7 * f.id + rows)
    blocks = random_blocks(f, n, k, rng, wide_scales=False)
    in0 = torch.from_numpy(
        rng.uniform(-1.0, 1.0, size=(rows, k)).astype(np.float32)
    ).to(torch.bfloat16)
    in0_rows = 8 if rows <= 8 else 16 if rows <= 16 else 32
    device = _run(
        f.id,
        kt,
        nt,
        _weight_pages(f, blocks),
        DataFormat.UInt32,
        _table_image(f),
        nt,
        matmul=True,
        in0_rows=in0_rows,
        in0_image=_in0_tiles(in0.to(torch.float32).numpy()),
    )
    in1 = torch.from_numpy(dequantize(f, blocks)).T.contiguous()
    _assert_matmul(device, in0.to(torch.float32), in1, MathFidelity.HiFi2)


@parametrize(
    fidelity=[MathFidelity.LoFi, MathFidelity.HiFi2, MathFidelity.HiFi4],
    rows=[5, 16, 32],
    kt=[1, 3, 8],
)
def test_dest_srca_mm(fidelity, rows, kt):
    nt = 2
    rng = np.random.default_rng(3000 + 31 * kt + rows)
    # Full fp32 mantissas, so that the TF32 truncation of in1 matters.
    in1 = torch.from_numpy(
        rng.standard_normal((kt * TILE, nt * TILE)).astype(np.float32)
    )
    in0 = torch.from_numpy(
        rng.uniform(-1.0, 1.0, size=(rows, kt * TILE)).astype(np.float32)
    ).to(torch.bfloat16)
    image = b"".join(
        _tilize(
            in1[TILE * j : TILE * (j + 1), TILE * t : TILE * (t + 1)].numpy()
        ).tobytes()
        for t in range(nt)
        for j in range(kt)
    )
    in0_rows = 8 if rows <= 8 else 16 if rows <= 16 else 32
    device = _run(
        0,
        kt,
        nt,
        image,
        DataFormat.Float32,
        bytes(16),
        nt,
        matmul=True,
        in0_rows=in0_rows,
        in0_image=_in0_tiles(in0.to(torch.float32).numpy()),
        fidelity=fidelity,
    )
    _assert_matmul(device, in0.to(torch.float32), in1, fidelity)
