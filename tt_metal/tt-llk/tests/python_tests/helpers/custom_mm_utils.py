# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared layout/golden helpers for the custom-matmul family of advance tests.

`test_custom_mm.py`, `test_compressed_custom_mm.py`, `test_sdpa_custom_mm.py` and
`test_sdpa_custom_mm_reuse_dest_srcb.py` all drive a partial-in0 custom matmul, and previously each
carried its own copy of the CT_DIMS / KT_DIMS / IN0_ROWS grid plus its own (wrong) idea of how in0
is laid out in L1. Both live here once.

Two layouts appear in this family, and neither is a plain tilized tile:

1. `pack_in0_faces` -- custom_mm / compressed_custom_mm / sdpa_custom_mm.
   `_llk_unpack_AB_custom_mm_init_` sets `unpB_x_end = unpB_face_r_dim * FACE_C_DIM - 1` and issues two
   UNPACRs per k-tile, with the SrcB L1 base programmed once and advanced by counters whose stride is a
   single face, `datum_size * FACE_C_DIM * face_r_dim`. So in0 is `kt_dim * 2` DENSELY packed faces of
   `in0_rows x 16`, not `kt_dim` padded 32x32 tiles -- an in0 k-tile is `64 * in0_rows` bytes, and
   TILE_SIZE_UNPACK_A is dead on this path because `tile_index_b` is 0. This matches what the
   silicon-validated `compressed_utils.run_compressed` emits.

2. `pack_sdpa_dest_tile` / `sdpa_dest_tile_golden` -- sdpa_custom_mm_reuse_dest_srcb.
   That primitive reads in0 out of DEST, and every tile it touches is 16 DEST rows: an 8x32 logical tile
   packed into ONE 16x16 face, rows 0-7 holding logical columns 0-15 and rows 8-15 holding columns 16-31
   (the demo's convention -- "Each tile is 8x32, which is the same as a full 16x16 face", sdpa.h:317).
"""

import torch

from .tile_constants import FACE_C_DIM, MAX_FACE_R_DIM

# Header contract: ct_dim 1..16, kt_dim even (2..256), in0 rows in {1, 2, 4, 8}.
CT_DIMS = [1, 2, 4, 8, 16]
KT_DIMS = [2, 4]
IN0_ROWS = [1, 2, 4, 8]

# sdpa_custom_mm and sdpa_custom_mm_reuse_dest_srcb do NOT sweep in0 rows: their addrmod helpers hardcode
# `constexpr std::uint32_t face_r_dim = 8` / an 8-row dest step, so 8 is the only shape they implement.
IN0_ROWS_SDPA = 8

# K-aware absolute floor for the goldens below. A single LoFi MVMUL accumulates the K-deep sum in a bf16
# dest, so noise grows ~linearly per K-tile -- a floor that Float16_b's default atol (0.05) is too tight
# for at large kt. Same calibration as compressed_utils.run_compressed; PCC remains the real gate.
FLOAT16B_DEFAULT_ATOL = 0.05
ACC_ATOL_PER_KT = 0.005


def matmul_grid(ct_dims=None, kt_dims=None, in0_rows=None):
    """The (ct_dim, kt_dim, in0_rows) sweep shared by the custom_mm-style tests."""
    combos = []
    for ct in ct_dims if ct_dims is not None else CT_DIMS:
        for kt in kt_dims if kt_dims is not None else KT_DIMS:
            for rows in in0_rows if in0_rows is not None else IN0_ROWS:
                combos.append((ct, kt, rows))
    return combos


def matmul_acc_atol(golden, kt_dim):
    """Scale the absolute tolerance by kt * mean|nonzero golden| (never below the format default)."""
    active = golden.abs().flatten()
    active = active[active > 0]
    mean_active = active.mean().item() if active.numel() else 0.0
    return max(FLOAT16B_DEFAULT_ATOL, ACC_ATOL_PER_KT * kt_dim * mean_active)


def pack_in0_faces(in0, kt_dim, stimuli_format):
    """Layout 1: in0 [in0_rows, kt_dim*32] -> kt_dim*2 dense (in0_rows x FACE_C_DIM) faces, row-major.

    Returns the packed L1 image (bytes), not a tensor, because this operand has no tile
    geometry StimuliConfig can stride by: an in0 k-tile is 64 * in0_rows bytes, while
    `StimuliConfig.write_matrix` walks the host buffer at MAX_TILE_ELEMENTS per tile and packs
    num_faces * face_r_dim * FACE_C_DIM datums out of each stride. A bytes buffer is written to
    L1 verbatim (`StimuliConfig._write_prepacked`), which is byte-for-byte what the
    silicon-validated `compressed_utils.run_compressed` emits for its own in0.
    """
    from .llk_params import format_dict
    from .stimuli_config import StimuliConfig

    pack_function = StimuliConfig.get_packer(stimuli_format)
    if pack_function is None:
        raise ValueError(f"Unsupported in0 format: {stimuli_format.name}")

    faces = torch.cat(
        [
            in0[:, i * FACE_C_DIM : (i + 1) * FACE_C_DIM].reshape(-1)
            for i in range(kt_dim * 2)
        ]
    ).to(format_dict[stimuli_format])
    return bytes(pack_function(faces))


def _to_dest_face(block, torch_format):
    """One 8x32 logical block -> a 16x16 DEST face: rows 0-7 = cols 0-15, rows 8-15 = cols 16-31."""
    top = block[:, :FACE_C_DIM]
    bottom = block[:, FACE_C_DIM : 2 * FACE_C_DIM]
    return torch.cat([top, bottom], dim=0).to(torch_format)  # [16, 16]


def pack_sdpa_dest_tile(in0, kt_dim, torch_format):
    """Layout 2 (input side): in0 [8, kt_dim*32] -> one 4-face 32x32 tile, face i == K-tile i.

    A single A2D datacopy of the returned tile lands face i at DEST rows 16*i, which is exactly where
    `_llk_math_sdpa_custom_mm_reuse_dest_srcb_` reads K-tile i (src_index + i*16). Unused face slots are
    zero-filled so the tile is a well-formed 1024-datum tile regardless of kt_dim.
    """
    faces = []
    for k in range(4):
        if k < kt_dim:
            faces.append(
                _to_dest_face(in0[:, k * 32 : (k + 1) * 32], torch_format).reshape(-1)
            )
        else:
            faces.append(torch.zeros(MAX_FACE_R_DIM * FACE_C_DIM, dtype=torch_format))
    return torch.cat(faces)


def sdpa_dest_tile_golden(out, torch_format):
    """Layout 2 (output side): an 8x(nt_dim*32) result -> the flat order the packer writes back.

    One 16x16 face per output tile, in the same rows-0-7/rows-8-15 split as `pack_sdpa_dest_tile`.
    """
    nt_dim = out.shape[1] // 32
    return torch.cat(
        [
            _to_dest_face(out[:, n * 32 : (n + 1) * 32], torch_format).reshape(-1)
            for n in range(nt_dim)
        ]
    )


def matmul_lofi_golden(in0, in1, formats, in0_dimensions, in1_dimensions):
    """Row-major LoFi matmul golden, shape (in0_rows, ct_dim*32).

    MatmulGolden rather than a raw torch matmul because the FPU runs LoFi here: it truncates the
    SrcA/SrcB mantissas before multiplying, which biases a K-deep sum of positive values low by ~2%
    -- far outside atol if the golden multiplies at full bf16 precision. Instantiated directly rather
    than through `get_golden_generator`: the harness swaps in a DummyGoldenGenerator during
    compile-producer, whose zeros(1024) would break the reshape for these narrow outputs.
    """
    from .golden_generators import MatmulGolden
    from .llk_params import MathFidelity

    return MatmulGolden()(
        in0,
        in1,
        formats.output_format,
        MathFidelity.LoFi,
        input_A_dimensions=in0_dimensions,
        input_B_dimensions=in1_dimensions,
        tilize=False,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    ).reshape(in0_dimensions[0], in1_dimensions[1])


def dense_result_rowmajor(res_tensor, ct_dim, in0_rows):
    """Readback for the dense 2-face layout -> row-major (in0_rows, ct_dim*32).

    The device packs ct_dim tiles; within a tile the two 16-column faces (in0_rows x FACE_C_DIM,
    row-major) sit contiguously, then padding out to the full L1 tile. Drop the per-tile padding and
    reorder. Same reduction as the silicon-validated `compressed_utils.run_compressed`.
    """
    faces_per_tile = 32 // FACE_C_DIM
    per_tile = res_tensor.reshape(ct_dim, -1)[:, : in0_rows * 32]
    return (
        per_tile.reshape(ct_dim, faces_per_tile, in0_rows, FACE_C_DIM)
        .permute(2, 0, 1, 3)
        .reshape(in0_rows, ct_dim * 32)
    )


def face_result_leading(res_tensor, tile_cnt):
    """Readback for the single-16x16-face layout: keep the leading face of each L1 tile.

    Used by the two sdpa matmuls, whose output tile is one 16x16 DEST face written at the start of a
    full 32x32 L1 tile. StimuliConfig's num_faces would narrow the readback for us, but it narrows the
    buffer_A / buffer_B *writes* too, which would truncate the operands.
    """
    face_datums = MAX_FACE_R_DIM * FACE_C_DIM
    tile_datums = len(res_tensor) // tile_cnt
    return torch.cat(
        [
            res_tensor[n * tile_datums : n * tile_datums + face_datums]
            for n in range(tile_cnt)
        ]
    )
