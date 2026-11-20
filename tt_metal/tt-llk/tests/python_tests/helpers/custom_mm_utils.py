# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Layout helpers for the compressed custom-matmul advance test.

`test_compressed_custom_mm.py` drives a partial-in0 custom matmul whose in0 is NOT a plain tilized
tile: `_llk_unpack_AB_custom_mm_init_` sets `unpB_x_end = unpB_face_r_dim * FACE_C_DIM - 1` and issues
two UNPACRs per k-tile, with the SrcB L1 base programmed once and advanced by counters whose stride is
a single face, `datum_size * FACE_C_DIM * face_r_dim`. So in0 is `kt_dim * 2` DENSELY packed faces of
`in0_rows x 16`, not `kt_dim` padded 32x32 tiles -- an in0 k-tile is `64 * in0_rows` bytes, and
TILE_SIZE_UNPACK_A is dead on this path because `tile_index_b` is 0. This matches what the
silicon-validated `compressed_utils.run_compressed` emits.

The sibling tests that shared these helpers (`custom_mm`, `sdpa_custom_mm`,
`sdpa_custom_mm_reuse_dest_srcb`) are owned by #53361, which targets the promoted `experimental/`
headers; the DEST-tile layout they needed went with them.

The SrcB face-walk helpers at the end are shared by `test_custom_mm.py`, `test_sdpa_custom_mm.py` and
`perf_custom_mm.py`.
"""

import torch

from .tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM, FACE_C_DIM

# Header contract: ct_dim 1..16, kt_dim even (2..254), in0 rows in {1, 2, 4, 8}.
CT_DIMS = [1, 2, 4, 8, 16]
KT_DIMS = [2, 4]
# NOT rt_dim: the header contract pins rt_dim at 1, and in0 is a single partial tile. This axis is
# in0's *face* row count -- the value that goes to the primitives' unpB_face_r_dim / the harness's
# IN_FACE_DIMS(in0_face_r_dim=...) -- so it is named after that parameter, not after the tile grid.
IN0_FACE_R_DIMS = [1, 2, 4, 8]


def matmul_grid(ct_dims=None, kt_dims=None, in0_face_r_dims=None):
    """The (ct_dim, kt_dim, in0_face_r_dim) sweep shared by the custom_mm-style tests."""
    combos = []
    for ct in ct_dims if ct_dims is not None else CT_DIMS:
        for kt in kt_dims if kt_dims is not None else KT_DIMS:
            for rows in (
                in0_face_r_dims if in0_face_r_dims is not None else IN0_FACE_R_DIMS
            ):
                combos.append((ct, kt, rows))
    return combos


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


# Matches CUSTOM_MM_MAX_KT_PER_WALK in llk_unpack_AB_custom_mm.h
CUSTOM_MM_MAX_KT_PER_WALK = 255


def face_walk_starts(kt):
    """First k-tile of each SrcB face walk, split as _llk_unpack_AB_custom_mm_run_ splits kt."""
    num_walks = -(-kt // CUSTOM_MM_MAX_KT_PER_WALK)
    base_kt, long_walks = divmod(kt, num_walks)
    starts, start = [], 0
    for walk in range(num_walks):
        starts.append(start)
        start += base_kt + (1 if walk < long_walks else 0)
    return starts


def face_walk_operands(M, kt, ct):
    """Exact bf16 integer operands. A is +-1 only on faces at the ends, the walk boundaries and every 256th face;
    B is random in {-2, -1, 1, 2}.
    """
    torch.manual_seed(0)
    num_faces = 2 * kt
    sentinels = {0, 1, num_faces - 2, num_faces - 1}
    for walk_start in face_walk_starts(kt)[1:]:
        face = 2 * walk_start
        sentinels |= {face - 2, face - 1, face, face + 1}
    sentinels |= set(range(255, num_faces, 256))

    torch_a = torch.zeros((M, kt * DEFAULT_TILE_R_DIM), dtype=torch.float32)
    for face in sentinels:
        value = 1.0 if (face // 256) % 2 == 0 else -1.0
        for m in range(M):
            torch_a[m, face * FACE_C_DIM + (face + m) % FACE_C_DIM] = value
    shape = (kt * DEFAULT_TILE_R_DIM, ct * DEFAULT_TILE_C_DIM)
    magnitude = torch.randint(1, 3, shape)
    sign = torch.randint(0, 2, shape) * 2 - 1
    torch_b = (magnitude * sign).to(torch.float32)
    return torch_a, torch_b
