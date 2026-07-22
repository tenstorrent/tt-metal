# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from helpers.compressed_utils import (  # noqa: F401 (autouse fixture — imported to activate in this module)
    DEEPSEEK_T420,
    FMT_CODE,
    assign_clustered,
    assign_interleaved,
    assign_random,
    assign_tile_matched,
    compressed_mm_include_paths,
    generate_refined_face_assignment,
    run_compressed,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.tile_constants import FACE_C_DIM


def promote_assignment(assignment, ct):
    # Promote bfp0 faces (to the smallest present format, or bfp2 if all-bfp0) so the
    # assignment meets the two face-unpack constraints: (1) every aligned 4-row block
    # holds >= 4 non-zero faces — at most one first-in-block face per 4-face meta; and
    # (2) the total non-zero count is a multiple of 4. ct is in 16x16 faces (= N // 16).
    promote_code = min((c for c in assignment if c != 0), default=FMT_CODE["bfp2"])

    def fill(
        indices, need
    ):  # promote up to `need` bfp0 faces in `indices` to promote_code
        for i in indices:
            if need <= 0:
                break
            if assignment[i] == 0:
                assignment[i] = promote_code
                need -= 1

    # 1) each aligned 4-row block needs >= 4 non-zero faces
    block_faces = 4 * ct
    for start in range(0, len(assignment), block_faces):
        block = range(start, start + block_faces)
        fill(block, 4 - sum(assignment[i] != 0 for i in block))

    # 2) total non-zero count must be a multiple of 4
    fill(
        range(len(assignment)), (4 - sum(c != 0 for c in assignment) % 4) % 4
    )  # faces to add to reach a multiple of 4

    return assignment


def pack_b(faces, chunk_faces=192):
    # Pack the assignment faces into the chunked split layout the unpack path streams.
    # faces = [(code, full_bytes)] in assignment order, full = 16B exps + mantissas.
    # bfp0 faces are empty and dropped; the rest are split into chunk_faces-sized chunks
    # (one B base address per chunk). Within a chunk, each present format (bfp2 then bfp4)
    # writes its exps (padded to a 1-mod-(man-1) word count) then its mantissas.
    #
    # Returns (packed_bytes, chunk_info). chunk_info[k] = (bfp2_off, bfp2_y_off, bfp4_off,
    # bfp4_y_off): per-format exp-section 16B-word offset and Y_OFF (the SET_Y stride,
    # (exp_words-1)//(man_words-1)), or (0, 0) if absent. encode_meta bakes these into
    # each chunk's base address + Y_OFF.
    nz_faces = [(code, full) for code, full in faces if code != 0]
    EXP = 16  # bytes of shared exps per face (one 16B word)
    out = bytearray()
    chunk_info = []
    for start in range(0, len(nz_faces), chunk_faces):
        chunk = nz_faces[start : start + chunk_faces]
        info = {}  # code -> (exp_word_off, y_off); (0, 0) if absent
        # man_words = 16B mantissa words per face
        for code, man_words in ((FMT_CODE["bfp2"], 4), (FMT_CODE["bfp4"], 8)):
            fmt_faces = [full for c, full in chunk if c == code]
            if not fmt_faces:
                info[code] = (0, 0)  # absent -> context unused
                continue
            off = len(out) // 16  # 16B-word offset of this exp section
            m = man_words - 1  # exp-section stride: 3 (bfp2) / 7 (bfp4)
            words = len(fmt_faces) + (
                (1 - len(fmt_faces)) % m
            )  # pad exps to 1 mod (man-1)
            out += b"".join(f[:EXP] for f in fmt_faces)  # exponents
            out += b"\x00" * (EXP * (words - len(fmt_faces)))  # exp padding
            out += b"".join(f[EXP:] for f in fmt_faces)  # mantissas
            info[code] = (
                off,
                (words - 1) // m,
            )  # (offset, Y_OFF); exact since words ≡ 1 mod m
        chunk_info.append(info[FMT_CODE["bfp2"]] + info[FMT_CODE["bfp4"]])
    return bytes(out), chunk_info


def meta_math_header(m, ct):
    # Header (bits[1:0] of a 6-bit math meta) for the 4-face block m. ct is in 16x16
    # faces (= N // 16): a face-row holds ct faces, a meta holds 4, and a face's kt
    # (K-face-row) index is face // ct. The header VALUES are regime-specific (the kernel
    # keys off ct_dim = ct//2) and must match the decode tables in
    # llk_math_face_compressed_mm.h:
    #     ct == 2      (CT_DIM == 1, "one"):  0=endInc, 1=endClr
    #     ct % 4 == 2  (CT_DIM odd):          0=nopB,   1=midInc, 2=endInc, 3=endClr
    #     ct % 4 == 0  (CT_DIM even):         0=nopB,   1=incB,   2=clrB
    # A meta gets a non-nopB header only when a face-row ends inside its 4 faces; the
    # value is then chosen by that row's kt index mod 4.

    # The meta's 4 linear face indexes placed on the ct-wide face grid.
    faces = [4 * m + j for j in range(4)]
    rows = [f // ct for f in faces]  # kt (K-face-row) index of each face
    cols = [f % ct for f in faces]  # column within the face-row

    # A face-row ends at column ct-1. Record where (if anywhere) such a boundary
    # lands inside this meta: position 3 is the meta's last face ("end"); position
    # 1 is a row ending mid-meta with the next row continuing ("mid").
    ends_at = [j for j in range(4) if cols[j] == ct - 1]

    if ct == 2:  # one regime: meta spans two face-rows
        assert ends_at == [1, 3]  # a mid (L1) and an end (L2) boundary
        return {1: 0, 3: 1}[rows[3] % 4]  # L2 row is odd: endInc / endClr
    elif ct % 4 == 2:  # odd regime
        assert ends_at in ([], [1], [3])  # at most one boundary, at pos 1 or 3
        if not ends_at:
            return 0  # nopB
        row = rows[ends_at[0]]
        assert (ends_at[0] == 1) == (
            row % 2 == 0
        )  # even rows split mid-meta, odd rows at end
        return {0: 1, 2: 1, 1: 2, 3: 3}[row % 4]  # 0/2 midInc, 1 endInc, 3 endClr
    else:  # even regime
        assert ends_at in ([], [3])  # a face-row only ends on the meta's last face
        if not ends_at:
            return 0  # nopB
        return {0: 1, 1: 1, 2: 1, 3: 2}[rows[3] % 4]  # 0/1/2 incB, 3 clrB


def encode_meta(assignment, ct, kt, chunk_info):
    # Build the face-compressed meta buffer, laid out as:
    #   math_words | iters | address_words | index_words
    # ct/kt are face columns/rows (N//16, K//16); chunk_info is pack_b's per-chunk
    # (bfp2, bfp4) exp-section offset + Y_OFF.

    # buffer_B base in L1_ADDRESS units (= byte/16 - 1): B follows A, which
    # is kt//2 (= K//32) 32x32 Float16_b tiles of 2048 B each.
    buf_a_addr = (
        StimuliConfig.STIMULI_L1_ADDRESS_DEBUG
        if StimuliConfig.WITH_COVERAGE
        else StimuliConfig.STIMULI_L1_ADDRESS_PERF
    )
    buf_b_words = (buf_a_addr + 2048 * (kt // 2)) // 16 - 1

    # Math region: one 6-bit meta per 4-face block, 5 metas per 32-bit word. A meta is
    # (face_bits << 2) | header — face_bits marks its non-zero faces, header is the
    # regime-specific value from meta_math_header.
    num_metas = len(assignment) // 4
    math_words = [0] * ((num_metas + 4) // 5)
    for m in range(num_metas):
        u, slot = divmod(m, 5)
        faces = assignment[4 * m : 4 * m + 4]
        face_bits = sum((1 << j) for j, code in enumerate(faces) if code != 0)
        meta6 = (face_bits << 2) | (meta_math_header(m, ct) & 0b11)
        math_words[u] |= meta6 << (6 * slot)

    # Non-zero faces in assignment order as (fmt, first_in_block): fmt 0=bfp2 / 1=bfp4
    # (bfp0 skipped, bfp8 disallowed). first_in_block marks the first non-zero face of each
    # aligned 4-row block; it drives the one B activation load per block (kt//4 blocks,
    # each given >= 4 non-zero faces by promote_assignment).
    nz_faces = []
    seen_block = -1
    for i, code in enumerate(assignment):
        if code == 0:
            continue
        block = i // ct // 4  # aligned 4-row block index
        first_in_block = block != seen_block  # first non-zero face of this block
        seen_block = block
        nz_faces.append((1 if code == FMT_CODE["bfp4"] else 0, first_in_block))
    blocks = sum(f[1] for f in nz_faces)
    assert blocks == kt // 4, f"{blocks} activation loads, expected kt//4 = {kt // 4}"

    nonzero_faces = len(nz_faces)  # one entry per non-zero face
    assert (
        nonzero_faces % 4 == 0
    ), f"non-zero face count {nonzero_faces} must be a multiple of 4"
    iters = nonzero_faces // 4  # unpack steps: 4 non-zero faces per index meta
    full_iters = iters // 6

    # Address section: one (bfp2, bfp4) base-address pair per 192-face chunk, which the
    # double-buffered unpacker reloads as it streams. The kernel reads full_iters//8 + 1
    # pairs — the last is a never-used lookahead when full_iters is a multiple of 8 (the
    # (0,0,0,0) pad covers it). Each word = (Y_OFF << 24) | ((base - Y_OFF) & 0xFFFFFF),
    # base = buf_b_words + chunk offset; the kernel re-adds Y_OFF via SETADC SET_Y so the
    # read starts at the exp-section base (absent format -> (0,0) = base, no stride).
    def addr_word(off, y_off):
        base = buf_b_words + off  # absolute 16B-word base
        return (y_off << 24) | ((base - y_off) & 0x00FFFFFF)

    address_words = []
    chunk_info = chunk_info + [(0, 0, 0, 0)]  # pad for the lookahead pair
    for k in range(full_iters // 8 + 1):
        b2_off, b2_y, b4_off, b4_y = chunk_info[k]
        address_words.append(addr_word(b2_off, b2_y))
        address_words.append(addr_word(b4_off, b4_y))

    # Index words (meta_ptr[2:]): each 32-bit word packs 6 unpack metas at stride 5 — meta
    # j occupies bits [5j+1 .. 5j+5] = [hasB, face0..3]. Bit 0 replicates the previous
    # word's last face as prev_fmt (word 0: its own first face), so each 6-bit lookup
    # (prev_fmt | hasB | face0..3) overlaps. hasB = 1 iff a face in the meta is first_in_block.
    index_words = [0] * ((iters + 5) // 6)
    prev_fmt = nz_faces[0][0]  # word 0's bit 0 replicates its own first face
    for k in range(iters):  # k = unpack meta index
        w, j = divmod(k, 6)
        base = (
            5 * j + 1
        )  # meta j's fields start at bit `base`; bit 0 of the word is prev_fmt
        if j == 0:
            index_words[w] |= prev_fmt
        first_in_blocks = 0
        for i, face in enumerate(nz_faces[4 * k : 4 * k + 4]):
            index_words[w] |= face[0] << (base + 1 + i)
            prev_fmt = face[0]
            first_in_blocks += face[1]
        assert (
            first_in_blocks <= 1
        ), f"unpack meta {k}: {first_in_blocks} first-in-block faces (expected 0 or 1)"
        index_words[w] |= first_in_blocks << base

    words = math_words + [iters] + address_words + index_words
    return np.array(words, dtype=np.uint32).tobytes()


COMPRESSION_GRANULARITY = FACE_C_DIM
SUPPORTED_M = {1, 8}
# The face unpack path supports only bfp2/bfp4 (bfp0 = skipped zeros); no bfp8.
SUPPORTED_FORMATS = {FMT_CODE[f] for f in ("bfp0", "bfp2", "bfp4")}


def run_face_compressed(M, K, N, assignment, pcc_threshold=None):
    run_compressed(
        M,
        K,
        N,
        assignment,
        "sources/matmul_face_compressed_test.cpp",
        COMPRESSION_GRANULARITY,
        SUPPORTED_M,
        SUPPORTED_FORMATS,
        promote_assignment,
        pack_b,
        encode_meta,
        pcc_threshold=pcc_threshold,
    )


BASE_SHAPES = [
    (1, 64, 32),  #   2x1
    (1, 64, 64),  #   2x2
    (1, 256, 32),  #   8x1
    (1, 256, 128),  #   8x4
    (1, 512, 256),  #  16x8
    (1, 7168, 32),  # 224x1
    (1, 7168, 64),  # 224x2
    # (1, 7168, 256), # 224x8 OOM
]

DEEPSEEK_SHAPES = [
    (1, 256, 64),  #  8x2
    (1, 896, 32),  # 28x1
    (1, 256, 224),  #  8x7
    (1, 1792, 32),  # 56x1
]

EXT_SHAPES = [
    # (1,  128, 512), #   4x16
    (1, 512, 128),  #  16x 4
    (1, 1536, 128),  #  48x 4
    (1, 2048, 32),  #  64x 1
    (1, 3584, 32),  # 112x 1
    # (1, 7168, 160), # 224x 5
    (1, 8192, 64),  # 256x 2
    # (8,  256, 512), #   8x16
    # (8,  512, 512), #  16x16
    (8, 576, 256),  #  18x 8
    # (8,  576, 512), #  18x16
]

SINGLE_FORMATS = [
    # ("bfp8",), unsupported
    ("bfp4",),
    ("bfp2",),
]

BASE_MULTI_FORMATS = [
    ("bfp4", "bfp2"),
    ("bfp4", "bfp0"),
    ("bfp2", "bfp0"),
    ("bfp4", "bfp2", "bfp0"),
]

EXT_MULTI_FORMATS = [
    ("bfp8", "bfp4"),
    ("bfp8", "bfp2"),
    ("bfp8", "bfp0"),
    ("bfp8", "bfp4", "bfp2"),
    ("bfp8", "bfp4", "bfp0"),
    ("bfp8", "bfp2", "bfp0"),
    ("bfp8", "bfp4", "bfp2", "bfp0"),
]

FACE_SWITCH_MULTS = [1.0, 2.0, 3.0]

SHAPES = BASE_SHAPES + DEEPSEEK_SHAPES + EXT_SHAPES
MULTI_FORMATS = BASE_MULTI_FORMATS  # EXT_MULTI_FORMATS are not supported


@parametrize(
    shape=SHAPES,
    formats=SINGLE_FORMATS,
)
def test_matmul_face_compressed_single(shape, formats):
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, COMPRESSION_GRANULARITY)
    run_face_compressed(M, K, N, assignment)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
)
def test_matmul_face_compressed_random(shape, formats):
    M, K, N = shape
    assignment = assign_random(K, N, formats, COMPRESSION_GRANULARITY)
    run_face_compressed(M, K, N, assignment)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
)
def test_matmul_face_compressed_clustered(shape, formats):
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, COMPRESSION_GRANULARITY)
    run_face_compressed(M, K, N, assignment)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
    interleave_n=[1, 2, 4, 8, 16, 32],
)
def test_matmul_face_compressed_interleaved(shape, formats, interleave_n):
    M, K, N = shape
    assignment = assign_interleaved(
        K, N, formats, COMPRESSION_GRANULARITY, interleave_n
    )
    run_face_compressed(M, K, N, assignment)


# ---------------------------------------------------------------------------
# Tile-matched variants: the SAME 32x32-tile assignment as the custom_compressed
# test, expanded 2x2 to 16x16 faces, so the face kernel decompresses identical B
# compression (parametrized 1:1 with the tile test — a controlled comparison).
# Fully-bfp0 blocks get densified by promote_assignment (>= 4 non-zero faces per
# aligned 4-row block), so assignments match except there.


@parametrize(
    shape=SHAPES,
    formats=SINGLE_FORMATS,
)
def test_matmul_face_compressed_single_matched(shape, formats):
    M, K, N = shape
    assignment = assign_tile_matched(assign_clustered, K, N, formats)
    run_face_compressed(M, K, N, assignment)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
)
def test_matmul_face_compressed_random_matched(shape, formats):
    M, K, N = shape
    assignment = assign_tile_matched(assign_random, K, N, formats)
    run_face_compressed(M, K, N, assignment)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
)
def test_matmul_face_compressed_clustered_matched(shape, formats):
    M, K, N = shape
    assignment = assign_tile_matched(assign_clustered, K, N, formats)
    run_face_compressed(M, K, N, assignment)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
    interleave_n=[1, 2, 4, 8, 16, 32],
)
def test_matmul_face_compressed_interleaved_matched(shape, formats, interleave_n):
    M, K, N = shape
    assignment = assign_tile_matched(assign_interleaved, K, N, formats, interleave_n)
    run_face_compressed(M, K, N, assignment)


# ---------------------------------------------------------------------------
# Realistic DeepSeek-R1 assignments swept by face-refinement density (switch_mult),
# from DEEPSEEK_T420's per-format stats (see helpers/compressed_utils.py):
#   * 1.0  — tile-granular baseline: 32x32 tiles expanded homogeneously to faces;
#            identical to the native tile run in test_matmul_custom_compressed_deepseek.
#   * >1.0 — face-granular: sub-tile precision flips at switch_mult x the baseline
#            switch count (2.0 isotropic estimate, 3.0 vertical/zigzag bracket).
# Exact-count (not sampled) so the small shapes below still hit the target shares.


@parametrize(
    shape=DEEPSEEK_SHAPES,
    switch_mult=FACE_SWITCH_MULTS,
    seed=[0],
)
def test_matmul_face_compressed_deepseek(shape, switch_mult, seed):
    M, K, N = shape
    assignment = generate_refined_face_assignment(
        K, N, DEEPSEEK_T420, switch_mult=switch_mult, seed=seed
    )
    run_face_compressed(M, K, N, assignment)
