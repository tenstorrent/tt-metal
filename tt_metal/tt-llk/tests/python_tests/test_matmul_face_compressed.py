# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import random
from collections import namedtuple
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import MatmulGolden
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.logger import logger
from helpers.pack import pack_bfp2_b, pack_bfp4_b, pack_bfp8_b, pack_bfp16
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import CRK_TILE_DIMM, IN_FACE_DIMS, NUM_FACES
from helpers.unpack import unpack_bfp2_b, unpack_bfp4_b, unpack_bfp8_b
from helpers.utils import passed_test
from ttexalens.tt_exalens_lib import write_to_device

EXTRA_INCLUDES = [
    f"-I../../../models/demos/deepseek_v3_b1/kernel_includes/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib",
    f"-I../../../models/demos/deepseek_v3_b1/kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api",
]


@pytest.fixture(autouse=True)
def compressed_mm_include_paths():
    added = [inc for inc in EXTRA_INCLUDES if inc not in TestConfig.INCLUDES]
    TestConfig.INCLUDES.extend(added)
    yield
    for inc in added:
        TestConfig.INCLUDES.remove(inc)


FACE_DIM = 16
FMT_CODE = {"bfp8": 3, "bfp4": 2, "bfp2": 1, "bfp0": 0}
# 16-byte words per 16x16 face: 1 shared-exponent word, plus mantissa words
# (bits-per-element * 256 / 8 / 16 = bits-per-element * 2).
FACE_EXP_WORDS = 1
FACE_MAN_WORDS = {0: 0, 1: 4, 2: 8, 3: 16}  # bfp0 / bfp2 / bfp4 / bfp8

# One non-zero unpack face: format bit (0=bfp2, 1=bfp4), its (row, col) on the
# ct-wide face grid (face-row-major), first_in_row (the first non-zero face of its
# face-row), and first_in_block (the first non-zero face of its aligned 4-row
# block). The hasB rule keys off first_in_block.
Face = namedtuple("Face", ["fmt", "row", "col", "first_in_row", "first_in_block"])


class CompressedStimuliConfig(StimuliConfig):
    def __init__(self, K, N, packed_a, packed_b, packed_meta):
        super().__init__(
            buffer_A=torch.zeros(1, dtype=torch.float32),  # placeholder
            stimuli_A_format=DataFormat.Float16_b,
            tile_count_A=K // 32,
            buffer_B=torch.zeros(1, dtype=torch.float32),  # placeholder
            stimuli_B_format=DataFormat.Bfp8_b,
            tile_count_B=(K // 32) * (N // 32),
            buffer_C=torch.zeros(1, dtype=torch.int32),  # placeholder
            stimuli_C_format=DataFormat.UInt32,
            tile_count_C=(len(packed_meta) + 4095) // 4096,
            stimuli_res_format=DataFormat.Float16_b,
            tile_count_res=N // 32,
        )
        self._packed_a = packed_a
        self._packed_b = packed_b
        self._packed_meta = packed_meta

    def write(self, location: str = "0,0"):
        write_to_device(location, self.buf_a_addr, self._packed_a)
        write_to_device(location, self.buf_b_addr, self._packed_b)
        write_to_device(location, self.buf_c_addr, self._packed_meta)


@dataclass
class CompressedMatmulStimulus:
    torch_a: torch.Tensor
    torch_b: torch.Tensor
    b_dequant: torch.Tensor
    golden: torch.Tensor
    tile_counts: list
    packed_a: bytes
    packed_b_standard: bytes
    packed_b_split: bytes
    meta: bytes


def meta_math_header(m, ct):
    """Header (bits[1:0] of a 6-bit math meta) for the 4-face block m.

    ``ct`` is in 16x16 faces (= N // 16): a face-row holds ct faces, a meta holds
    4, and the kt (K-face-row) index of a face is face // ct. The kernel regime
    keys off the 32-tile column count ct_dim = ct//2, and the header VALUES are
    regime-specific — they must match the decode tables in
    llk_math_face_compressed_mm.h:
        ct == 2      (CT_DIM == 1, "one"):  0=endInc, 1=endClr
        ct % 4 == 2  (CT_DIM odd):          0=nopB, 1=midInc, 2=endInc, 3=endClr
        ct % 4 == 0  (CT_DIM even):         0=nopB, 1=incB,   2=clrB

    A meta gets a non-nopB header only when a face-row ends inside its 4 faces;
    the value is then chosen by that row's kt index mod 4.
    """
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
        assert (ends_at[0] == 1) == (row % 2 == 0)  # even rows split mid-meta, odd rows at end
        return {0: 1, 2: 1, 1: 2, 3: 3}[row % 4]  # 0/2 midInc, 1 endInc, 3 endClr
    else:  # even regime
        assert ends_at in ([], [3])  # a face-row only ends on the meta's last face
        if not ends_at:
            return 0  # nopB
        return {0: 1, 1: 1, 2: 1, 3: 2}[rows[3] % 4]  # 0/1/2 incB, 3 clrB


def meta_unpack_hasb(k, nz_faces):
    """hasB bit for unpack meta k: 1 if exactly one of the meta's 4 faces is the
    first face of an aligned 4-row block (triggering an SbbbB activation load), 0
    if none. ``nz_faces`` is the Face(...) stream; meta k spans nz_faces[4k:4k+4].
    Two first-in-block faces in a single meta is impossible (asserted)."""
    n = sum(f.first_in_block for f in nz_faces[4 * k : 4 * k + 4])
    assert n <= 1, f"unpack meta {k}: {n} first-in-block faces (expected 0 or 1)"
    return n


def encode_meta(assignment, ct, tile_counts, chunk_info, buf_b_words):
    assert len(assignment) % 4 == 0, "assignment must be a multiple of 4"
    assert ct % 2 == 0, "ct must be even"

    num_metas = len(assignment) // 4
    math_words = [0] * ((num_metas + 4) // 5)
    for m in range(num_metas):
        faces = assignment[4 * m : 4 * m + 4]
        face_bits = sum((1 << j) for j, code in enumerate(faces) if int(code) != 0)
        meta6 = (face_bits << 2) | (meta_math_header(m, ct) & 0b11)
        u, slot = divmod(m, 5)
        math_words[u] |= meta6 << (6 * slot)

    # --- unpack region ---  layout after the math region:
    #   [iters word][address section][index words]
    # B is packed in 192-face chunks (see pack_b_chunked); the address section
    # carries one (bfp2, bfp4) exp-section base-address pair per chunk, which the
    # double-buffered unpacker reloads as it streams.
    assert tile_counts[FMT_CODE["bfp8"]] == 0, "unpack path supports only bfp2/bfp4 (no bfp8)"

    nonzero_faces = sum(tile_counts) - tile_counts[FMT_CODE["bfp0"]]
    assert nonzero_faces % 4 == 0, f"non-zero face count {nonzero_faces} must be a multiple of 4"
    iters = nonzero_faces // 4  # unpack steps: 4 non-zero faces / index

    # Unpack index payload (meta_ptr[2:]). Only non-zero faces are unpacked, in
    # assignment order; 0 = bfp2, 1 = bfp4 (bfp0 skipped, bfp8 disallowed). Each
    # face carries (row, col), first_in_row and first_in_block for the hasB rule.
    # Each 32-bit word holds 6 metas at stride 5: meta j occupies bits
    # [5j+1 .. 5j+5] = [hasB, face0, face1, face2, face3] (low->high). Bit 0 of the
    # word replicates the previous word's last face as prev_fmt (its own first face
    # for word 0), so each 6-bit lookup (prev_fmt | hasB | face0..3) overlaps.
    #
    # first_in_block (the first non-zero face of each aligned 4-row block) drives
    # the one SbbbB activation load per block. There are kt//4 blocks (kt =
    # face-rows = K//16) and every block must hold at least one non-zero face.
    nz_faces = []
    seen_row = seen_block = -1
    for i, code in enumerate(assignment):
        if int(code) == 0:
            continue
        row, col = divmod(i, ct)
        first_in_row = row != seen_row
        first_in_block = first_in_row and row // 4 != seen_block
        if first_in_block:
            seen_block = row // 4
        seen_row = row
        nz_faces.append(Face(1 if int(code) == FMT_CODE["bfp4"] else 0, row, col, first_in_row, first_in_block))
    assert len(nz_faces) == nonzero_faces

    kt = len(assignment) // ct  # face-rows (= K // 16)
    blocks = sum(f.first_in_block for f in nz_faces)
    assert (
        blocks == kt // 4
    ), f"{blocks} activation loads, expected kt//4 = {kt // 4} (each 4-row block needs a non-zero face)"

    index_words = [0] * ((iters + 5) // 6)
    for k in range(iters):  # k = unpack meta index
        w, j = divmod(k, 6)
        base = 5 * j + 1  # bit 0 is the word's prev_fmt
        index_words[w] |= meta_unpack_hasb(k, nz_faces) << base
        for i, face in enumerate(nz_faces[4 * k : 4 * k + 4]):
            index_words[w] |= face.fmt << (base + 1 + i)
    for w in range(len(index_words)):
        first_face = 24 * w  # global non-zero face index of meta 6w
        index_words[w] |= nz_faces[first_face - 1].fmt if w else nz_faces[0].fmt

    # Address section (sits between the iters word and the index words — the
    # kernel reads it at pre_meta_ptr[1 .. 2*(full_iters/8)+2]). One (bfp2, bfp4)
    # base-address pair per chunk: chunk k -> word[1+2k]=bfp2 (cntx0/2),
    # word[2+2k]=bfp4 (cntx1/3). The kernel always reads full_iters//8 + 1 pairs;
    # the last is a never-used lookahead when full_iters is a multiple of 8.
    #
    # Each word = (Y_OFF << 24) | ((base - Y_OFF) & 0x00FFFFFF), all in 16B words.
    # base = buf_b_words + chunk exp-section offset (absolute, kernel uses it raw).
    # Y_OFF = (exp_section_words - 1) / (man_words - 1) — an exact integer because
    # exp sizes are 1 mod (man-1); the kernel re-adds it via SETADC SET_Y so the
    # read starts at the exp-section base.
    full_iters = iters // 6

    def addr_word(off, exp_words, code):
        base = buf_b_words + off  # absolute 16B-word base
        if exp_words == 0:  # format absent -> context unused
            return base & 0x00FFFFFF
        y_off = (exp_words - 1) // (FACE_MAN_WORDS[code] - 1)
        return (y_off << 24) | ((base - y_off) & 0x00FFFFFF)

    address_words = []
    for k in range(full_iters // 8 + 1):
        b2_off, b2_words, b4_off, b4_words = chunk_info[k] if k < len(chunk_info) else (0, 0, 0, 0)
        address_words.append(addr_word(b2_off, b2_words, FMT_CODE["bfp2"]))
        address_words.append(addr_word(b4_off, b4_words, FMT_CODE["bfp4"]))

    words = math_words + [iters] + address_words + index_words
    return np.array(words, dtype=np.uint32).tobytes()


def tilize(tile, tile_dim, face_dim):
    assert tile.shape == (tile_dim, tile_dim)
    num_faces = tile_dim // face_dim
    res = []
    for i in range(num_faces):
        for j in range(num_faces):
            res += tile[i * face_dim : (i + 1) * face_dim, j * face_dim : (j + 1) * face_dim].flatten().tolist()
    return torch.tensor(res, dtype=tile.dtype)


def untilize(tile, tile_dim, face_dim):
    assert tile.shape == (tile_dim * tile_dim,)
    num_faces = tile_dim // face_dim
    res = torch.zeros((tile_dim, tile_dim), dtype=tile.dtype)
    for i in range(num_faces):
        for j in range(num_faces):
            res[i * face_dim : (i + 1) * face_dim, j * face_dim : (j + 1) * face_dim] = tile[
                (i * num_faces + j) * face_dim * face_dim : (i * num_faces + j + 1) * face_dim * face_dim
            ].reshape(face_dim, face_dim)
    return res


def untilize_result(tensor, M, N, tile_dim, face_dim):
    tensor = torch.as_tensor(tensor, dtype=torch.bfloat16)
    ct = N // tile_dim
    faces_per_tile = tile_dim // face_dim  # column-faces per output tile (2 for 32/16)
    slot = len(tensor) // ct  # per-tile element span in the result
    res = torch.zeros((M, N), dtype=torch.bfloat16)
    for m in range(N // face_dim):  # faces across the full output width
        c, within = divmod(m, faces_per_tile)  # which tile, which face within it
        base = c * slot + within * M * face_dim
        res[:, m * face_dim : (m + 1) * face_dim] = tensor[base : base + M * face_dim].reshape(M, face_dim)
    return res


def pack_bfp_tile(tile, code, tile_dim, face_dim):
    num_faces = (tile_dim // face_dim) ** 2
    if code == 3:  # bfp8
        return bytes(pack_bfp8_b(tilize(tile, tile_dim, face_dim), num_faces=num_faces))
    if code == 2:  # bfp4
        return bytes(pack_bfp4_b(tilize(tile, tile_dim, face_dim), num_faces=num_faces))
    if code == 1:  # bfp2
        return bytes(pack_bfp2_b(tilize(tile, tile_dim, face_dim), num_faces=num_faces))
    if code == 0:  # bfp0
        return b""
    raise ValueError(f"Unsupported format {code}")


def unpack_bfp_tile(tile, code, tile_dim, face_dim):
    num_faces = (tile_dim // face_dim) ** 2
    if code == 3:  # bfp8
        return untilize(unpack_bfp8_b(tile, num_faces=num_faces), tile_dim, face_dim)
    if code == 2:  # bfp4
        return untilize(unpack_bfp4_b(tile, num_faces=num_faces), tile_dim, face_dim)
    if code == 1:  # bfp2
        return untilize(unpack_bfp2_b(tile, num_faces=num_faces), tile_dim, face_dim)
    if code == 0:  # bfp0
        return untilize(torch.zeros((tile_dim * tile_dim,), dtype=torch.bfloat16), tile_dim, face_dim)
    raise ValueError(f"Unsupported format {code}")


def pack_a(tensor, M, K, face_dim):
    assert tensor.shape == (M, K)
    width_faces = K // face_dim
    res = b""
    for i in range(width_faces):
        res += pack_bfp16(tensor[:, i * face_dim : (i + 1) * face_dim])
    return res


def promote_assignment(assignment, ct):
    """Promote bfp0 faces to the smallest available format so the assignment meets
    the kernel's unpack constraints:
      * every aligned 4-row block holds >= 4 non-zero faces — so consecutive
        block starts are >= 4 non-zero faces apart, i.e. at most one
        first-in-block face per 4-face unpack meta; and
      * the total non-zero face count is a multiple of 4 (one unpack meta = 4).
    ``ct`` is in 16x16 faces (= N // 16). Returns a new assignment that the golden
    is then built from. No-op when the assignment already satisfies both."""
    assignment = list(int(c) for c in assignment)
    available = sorted({c for c in assignment if c != 0})
    if not available:
        return assignment  # all bfp0 -> empty-B case, caller skips
    promote_code = available[0]  # smallest (lowest-precision) available format
    block_faces = 4 * ct

    # 1) each aligned 4-row block needs >= 4 non-zero faces
    for start in range(0, len(assignment), block_faces):
        block = range(start, min(start + block_faces, len(assignment)))
        need = 4 - sum(1 for i in block if assignment[i] != 0)
        for i in block:
            if need <= 0:
                break
            if assignment[i] == 0:
                assignment[i] = promote_code
                need -= 1

    # 2) total non-zero count must be a multiple of 4
    need = (-sum(1 for c in assignment if c != 0)) % 4
    for i in range(len(assignment)):
        if need <= 0:
            break
        if assignment[i] == 0:
            assignment[i] = promote_code
            need -= 1

    return assignment


CHUNK_FACES = 192  # max non-zero faces unpacked against a single base address


def round_exp_words(count, code):
    """Pad an exponent section of ``count`` faces (1 16B word each) up to a size
    that is 1 mod (man_words-1) — 3 for bfp2, 7 for bfp4 — with minimal padding."""
    m = FACE_MAN_WORDS[code] - 1
    return count + ((1 - count) % m)


def pack_b_chunked(nz_faces):
    """Pack the non-zero faces into the chunked split layout the unpack streams.

    ``nz_faces`` is [(code, full_bytes)] in assignment order, full = 16B exps +
    mantissas. Faces are grouped into CHUNK_FACES blocks (one base address per
    chunk). Within a chunk, for each format that is present (order bfp2, bfp4):
    its exponents padded up to a 1-mod-(man-1) word count, then its mantissas.
    An absent format contributes nothing (no section, no padding).

    Returns (packed_bytes, chunk_info), where chunk_info[k] =
    (bfp2_off, bfp2_exp_words, bfp4_off, bfp4_exp_words): each format's exp-section
    16B-word offset and padded word count within chunk k (0, 0 when the format is
    absent). encode_meta turns these into the per-chunk base address + Y_OFF."""
    EXP = FACE_EXP_WORDS * 16  # bytes of shared exps per face (=16)
    out = bytearray()
    chunk_info = []
    for start in range(0, len(nz_faces), CHUNK_FACES):
        chunk = nz_faces[start : start + CHUNK_FACES]
        info = {}  # code -> (exp_word_off, exp_words)
        for code in (FMT_CODE["bfp2"], FMT_CODE["bfp4"]):
            faces = [full for c, full in chunk if c == code]
            if not faces:
                info[code] = (0, 0)  # absent -> context unused
                continue
            off = len(out) // 16  # 16B-word offset of this exp section
            words = round_exp_words(len(faces), code)
            out += b"".join(f[:EXP] for f in faces)  # exponents
            out += b"\x00" * (EXP * (words - len(faces)))  # exp padding
            out += b"".join(f[EXP:] for f in faces)  # mantissas
            info[code] = (off, words)
        b2, b4 = info[FMT_CODE["bfp2"]], info[FMT_CODE["bfp4"]]
        chunk_info.append((b2[0], b2[1], b4[0], b4[1]))
    return bytes(out), chunk_info


def generate_compressed_matmul_golden(M, K, N, assignment, tile_dim, seed=0):
    assert K % tile_dim == 0 and N % tile_dim == 0, "K and N must be multiples of tile_dim"
    kt = K // tile_dim
    ct = N // tile_dim
    num_tiles = kt * ct
    assert len(assignment) == num_tiles, f"assignment has {len(assignment)} entries, expected kt*ct={num_tiles}"
    assignment = promote_assignment(assignment, ct)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N), dtype=torch.bfloat16)

    b_dequant = torch.zeros((K, N), dtype=torch.bfloat16)

    packed_b_standard = b""
    nz_faces = []  # (code, full) for non-zero faces, in assignment order
    tile_counts = [0] * len(FMT_CODE)

    for i, code in enumerate(assignment):
        k, c = divmod(i, ct)
        tile_counts[code] += 1
        tile = torch_b[k * tile_dim : (k + 1) * tile_dim, c * tile_dim : (c + 1) * tile_dim]

        full = pack_bfp_tile(tile, code, tile_dim=tile_dim, face_dim=FACE_DIM)
        packed_b_standard += full
        b_dequant[k * tile_dim : (k + 1) * tile_dim, c * tile_dim : (c + 1) * tile_dim] = unpack_bfp_tile(
            full, code, tile_dim=tile_dim, face_dim=FACE_DIM
        )

        if code != 0:
            nz_faces.append((code, full))

    # Chunked split layout consumed by the unpack path (see pack_b_chunked).
    # chunk_info feeds encode_meta's per-chunk base addresses + Y_OFF.
    packed_b_split, chunk_info = pack_b_chunked(nz_faces)

    # buffer_B base in L1_ADDRESS units (= byte_addr/16 - 1 on Blackhole),
    # recreated like StimuliConfig._calculate_tile_sizes: buf_a_addr +
    # tile_size_A * tile_count_A, with tile_size_A = Float16_b 32x32 = 2048 B and
    # tile_count_A = K//32. Lets encode_meta bake absolute chunk base addresses at
    # gold time.
    buf_a_addr = (
        StimuliConfig.STIMULI_L1_ADDRESS_DEBUG if StimuliConfig.WITH_COVERAGE else StimuliConfig.STIMULI_L1_ADDRESS_PERF
    )
    buf_b_words = (buf_a_addr + 2048 * (K // 32)) // 16 - 1

    # Golden via the standard MatmulGolden path (models LoFi fidelity masking,
    # accumulation, etc.). Feed the already-bfp-quantized b_dequant as Float16_b
    # so MatmulGolden does NOT re-quantize it, and keep tilize=False — tile
    # layout is handled on the device-side path (pack_bfp_tile / untilize_result),
    # so the golden stays row-major. Compressed custom MM is LoFi-only.
    #
    # Instantiate MatmulGolden directly rather than via get_golden_generator: the
    # harness swaps the registered generator for a DummyGoldenGenerator
    # (zeros(1024)) during compile-producer, which would break the reshape below
    # for narrow-M outputs (M*N != 1024).
    matmul_golden = MatmulGolden()
    golden = matmul_golden(
        torch_a,
        b_dequant,
        DataFormat.Float16_b,
        MathFidelity.LoFi,
        input_A_dimensions=[M, K],
        input_B_dimensions=[K, N],
        tilize=False,
        input_A_format=DataFormat.Float16_b,
        input_B_format=DataFormat.Float16_b,
    ).reshape(M, N)

    return CompressedMatmulStimulus(
        torch_a=torch_a,
        torch_b=torch_b,
        b_dequant=b_dequant,
        golden=golden,
        tile_counts=tile_counts,
        packed_a=pack_a(torch_a, M, K, face_dim=FACE_DIM),
        packed_b_standard=packed_b_standard,
        packed_b_split=packed_b_split,
        meta=encode_meta(assignment, ct, tile_counts, chunk_info, buf_b_words),
    )


def run_compressed(M, K, N, assignment, tile_dim, pcc_threshold=None):
    assert M in {1, 2, 4, 8}, "compressed_custom_mm requires M in {1, 2, 4, 8}"
    assert K % 64 == 0 and K // 32 >= 2, "kt_dim must be an even number >= 2"
    assert 1 <= N // 32 <= 16, "ct_dim must be in [1, 16]"
    kt = K // tile_dim
    ct = N // tile_dim
    output_format = DataFormat.Float16_b

    # Stimulus + quantization-aware golden: B is packed (tilized) per the
    # assignment and the golden is A @ dequant(quant(B)), so the BFP rounding
    # error is folded into the golden rather than charged against PCC.
    stim = generate_compressed_matmul_golden(M, K, N, assignment, tile_dim)

    # bfp0 tiles pack to zero bytes; an assignment with no non-bfp0 tiles produces
    # an empty B buffer the device kernel can't consume (faults). Skip those.
    if sum(stim.tile_counts) - stim.tile_counts[FMT_CODE["bfp0"]] == 0:
        pytest.skip("no non-zero (non-bfp0) tiles in assignment — empty B buffer")

    config = CompressedStimuliConfig(K, N, stim.packed_a, stim.packed_b_split, stim.meta)

    formats = InputOutputFormat(
        input_format=DataFormat.Float16_b,
        output_format=output_format,
        input_format_B=DataFormat.Bfp8_b,
    )

    configuration = TestConfig(
        "sources/matmul_face_compressed_test.cpp",
        formats,
        templates=[
            CRK_TILE_DIMM(c_dimm=ct // 2, r_dimm=1, k_dimm=kt // 2),
        ],
        runtimes=[
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
            IN_FACE_DIMS(in0_face_r_dim=M),
        ],
        variant_stimuli=config,
        dest_acc=DestAccumulation.No,
    )

    res = configuration.run().result
    # Result tiles are 32-wide hardware tiles (regardless of the 16x16 compression
    # faces), so untilize with tile_dim=32, not the compression TILE_DIM.
    res_tensor = untilize_result(res, M, N, 32, FACE_DIM)

    # abs_diff = (stim.golden.to(torch.float32) - res_tensor.to(torch.float32)).abs()
    # logger.info("golden [{}x{}]:\n{}", M, N, stim.golden)
    # logger.info("result [{}x{}]:\n{}", M, N, res_tensor)
    # logger.info("abs diff (max={:.4f}, mean={:.4f}):\n{}", abs_diff.max().item(), abs_diff.mean().item(), abs_diff)

    golden = stim.golden

    # K-aware tolerance. The single LoFi MVMUL accumulates the K-deep sum in a
    # bf16 dest; that accumulation noise grows ~linearly per K-tile and lands as an
    # absolute floor on small-magnitude outputs. Float16_b's default atol (0.05) is
    # right for shallow K but far too tight once kt is large, so scale the absolute
    # floor by kt * mean|golden| (never below the default). rtol stays at the
    # default — the residual is an absolute floor, not a relative error. PCC
    # (>=0.997 even at K=7168) remains the correctness gate.
    #
    # The coefficient is calibrated across ALL bfp formats: the worst case is the
    # coarse low-bit mixes (bfp2/bfp0), where coarse quantization shrinks
    # mean|golden| but not the noise floor (worst observed need ~0.0034 at kt=16).
    # 0.005 gives ~1.5x margin over that.
    #
    # Mean is over NON-ZERO golden only: bfp0 tiles zero whole output columns, and
    # those structural zeros (which pass trivially) deflate mean|golden| below the
    # magnitude of the *active* dot products the floor must cover — exactly what made
    # interleaved (bfp2,bfp0) trip on tiny cancellation outputs. Excluding zeros can
    # only raise the floor on sparse outputs, never lower it, so no dense case shifts.
    FLOAT16B_DEFAULT_ATOL = 0.05
    ACC_ATOL_PER_KT = 0.005
    # Depth is the K-tile count (K // 32) — matches the kernel's KT_DIM and the per-K-tile noise model
    # above. (The local `kt`/`ct` are per-face counts = K // 16, so scale by K // 32 here, not `kt`.)
    active_golden = stim.golden.abs()
    active_golden = active_golden[active_golden > 0]
    mean_active = active_golden.mean().item() if active_golden.numel() else 0.0
    acc_atol = max(FLOAT16B_DEFAULT_ATOL, ACC_ATOL_PER_KT * (K // 32) * mean_active)

    # Profiling run: with device print on, the kernel's pmp loop re-runs the matmul
    # (accumulating into dest), so the result is intentionally NOT bit-correct — the
    # payload is the pmp Zone/Sync timing prints, not the output. Skip the PCC gate.
    if TestConfig.DEVICE_PRINT_ENABLED:
        logger.warning("PERF run (device print on) — skipping correctness check for shape=(M={}, K={}, N={})", M, K, N)
        return

    assert passed_test(
        golden,
        res_tensor,
        output_format,
        custom_atol=acc_atol,
        custom_pcc_threshold=pcc_threshold,
        print_pcc=True,
    ), f"compressed matmul failed for shape=(M={M}, K={K}, N={N})"


def assign_random(K, N, formats, tile_dim):
    """Randomly assign ``formats`` across all (k, c) tiles in row-major order.
    Deterministic across test invocations (fixed seed) for reproducible PCC."""
    codes = [FMT_CODE[f] for f in formats]
    num_tiles = (K // tile_dim) * (N // tile_dim)
    rng = random.Random(0)
    return [rng.choice(codes) for _ in range(num_tiles)]


def assign_clustered(K, N, formats, tile_dim):
    """Assign ``formats`` in contiguous, roughly equal-length runs over the
    (k, c) row-major tile order — each format occupies one cluster."""
    codes = [FMT_CODE[f] for f in formats]
    num_tiles = (K // tile_dim) * (N // tile_dim)
    num_runs = min(len(codes), num_tiles)
    run_length = num_tiles // num_runs
    last_run_length = num_tiles - run_length * (num_runs - 1)
    assignment = []
    for i in range(num_runs - 1):
        assignment.extend([codes[i]] * run_length)
    assignment.extend([codes[num_runs - 1]] * last_run_length)
    return assignment


def assign_interleaved(K, N, formats, interleave_n, tile_dim):
    """Cycle through ``formats`` in blocks of ``interleave_n`` tiles over the
    (k, c) row-major tile order, for any number of formats."""
    codes = [FMT_CODE[f] for f in formats]
    num_tiles = (K // tile_dim) * (N // tile_dim)
    num_blocks = (num_tiles + interleave_n - 1) // interleave_n
    assignment = []
    for i in range(num_blocks - 1):
        assignment.extend([codes[i % len(codes)]] * interleave_n)
    assignment.extend([codes[(num_blocks - 1) % len(codes)]] * (num_tiles - len(assignment)))
    return assignment


TILE_DIM = 16

BASE_SHAPES = [
    (1, 64, 32),
    (1, 64, 64),
    (1, 256, 32),
    (1, 256, 128),
    (1, 512, 256),
    (1, 7168, 32),
    (1, 7168, 64),
    # (1, 7168, 256), OOM
]

DS_SHAPES = [
    (1, 256, 64),
    (1, 896, 256),
]

EXT_SHAPES = [
    # (1,  128, 512),
    (1, 256, 256),
    (1, 512, 128),
    (1, 1536, 128),
    (1, 2048, 32),
    (1, 2048, 64),
    (1, 3584, 32),
    # (1, 7168, 160),
    (1, 8192, 64),
    # (8,  256, 512),
    # (8,  512, 512),
    (8, 576, 256),
    # (8,  576, 512),
]

SINGLE_FORMATS = [
    # ("bfp8",),
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
    # ("bfp4", "bfp2"), BASE
    # ("bfp4", "bfp0"), BASE
    # ("bfp2", "bfp0"), BASE
    ("bfp8", "bfp4", "bfp2"),
    ("bfp8", "bfp4", "bfp0"),
    ("bfp8", "bfp2", "bfp0"),
    # ("bfp4", "bfp2", "bfp0"), BASE
    ("bfp8", "bfp4", "bfp2", "bfp0"),
]

SHAPES = BASE_SHAPES + DS_SHAPES + EXT_SHAPES
MULTI_FORMATS = BASE_MULTI_FORMATS


@parametrize(
    shape=SHAPES,
    formats=SINGLE_FORMATS,
)
def test_matmul_custom_compressed_single(shape, formats):
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, tile_dim=TILE_DIM)
    run_compressed(M, K, N, assignment, tile_dim=TILE_DIM)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
)
def test_matmul_custom_compressed_random(shape, formats):
    M, K, N = shape
    assignment = assign_random(K, N, formats, tile_dim=TILE_DIM)
    run_compressed(M, K, N, assignment, tile_dim=TILE_DIM)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
)
def test_matmul_custom_compressed_clustered(shape, formats):
    M, K, N = shape
    assignment = assign_clustered(K, N, formats, tile_dim=TILE_DIM)
    run_compressed(M, K, N, assignment, tile_dim=TILE_DIM)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
    interleave_n=[1, 2, 4, 8, 16, 32],
)
def test_matmul_custom_compressed_interleaved(shape, formats, interleave_n):
    M, K, N = shape
    assignment = assign_interleaved(K, N, formats, interleave_n, tile_dim=TILE_DIM)
    run_compressed(M, K, N, assignment, tile_dim=TILE_DIM)


# ---------------------------------------------------------------------------
# Tile-matched variants: run the SAME logical assignment as the 32x32-tile
# (custom_compressed) test, expanded to 16x16-face granularity. Each tile's
# format is replicated across the 2x2 block of faces it covers, so the face
# kernel decompresses exactly the same B compression as the tile kernel — making
# face-vs-tile a controlled comparison (parametrized identically to the tile
# test, so the rows pair 1:1).
#
# NOTE: the face kernel needs >=1 non-zero face per aligned 4-face-row block (one
# activation load per block); the tile kernel has no such per-block constraint.
# So where a full 2-tile-row block is bfp0, generate_compressed_matmul_golden's
# promote_assignment densifies a few faces — the assignments match exactly except
# in those sparse-bfp0 blocks (all single-format and dense-mix cases are exact).


def assign_tile_matched(assign_fn, K, N, formats, **kwargs):
    """Generate a 32x32-tile assignment with ``assign_fn`` (the same generator the
    tile/custom_compressed test uses) at tile_dim=32, then expand it to 16x16-face
    granularity: replicate each tile's format across its 2x2 face block (duplicate
    each element along columns, then duplicate each row). The result is row-major
    over (K//16, N//16) faces, matching what run_compressed expects at TILE_DIM=16."""
    tiles = assign_fn(K, N, formats, tile_dim=32, **kwargs)
    # Promote at TILE granularity first, mirroring the tile (custom_compressed) test:
    # an all-bfp0 tile assignment would otherwise expand to an all-bfp0 face assignment,
    # which the face kernel can't consume. The face-side promote_assignment can't fix it
    # (it bails when no non-bfp0 format is available), and run_compressed's empty-B skip
    # happens only after encode_meta has already asserted. Promoting the first tile to
    # bfp2 here keeps the expanded assignment identical to the tile test's promoted one.
    if all(int(c) == FMT_CODE["bfp0"] for c in tiles):
        tiles = list(tiles)
        tiles[0] = FMT_CODE["bfp2"]
    arr = np.array(tiles, dtype=np.int64).reshape(K // 32, N // 32)
    return arr.repeat(2, axis=0).repeat(2, axis=1).flatten().tolist()


@parametrize(
    shape=SHAPES,
    formats=SINGLE_FORMATS,
)
def test_matmul_face_compressed_single_tile_matched(shape, formats):
    M, K, N = shape
    assignment = assign_tile_matched(assign_clustered, K, N, formats)
    run_compressed(M, K, N, assignment, tile_dim=TILE_DIM)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
)
def test_matmul_face_compressed_random_tile_matched(shape, formats):
    M, K, N = shape
    assignment = assign_tile_matched(assign_random, K, N, formats)
    run_compressed(M, K, N, assignment, tile_dim=TILE_DIM)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
)
def test_matmul_face_compressed_clustered_tile_matched(shape, formats):
    M, K, N = shape
    assignment = assign_tile_matched(assign_clustered, K, N, formats)
    run_compressed(M, K, N, assignment, tile_dim=TILE_DIM)


@parametrize(
    shape=SHAPES,
    formats=MULTI_FORMATS,
    interleave_n=[1, 2, 4, 8, 16, 32],
)
def test_matmul_face_compressed_interleaved_tile_matched(shape, formats, interleave_n):
    M, K, N = shape
    assignment = assign_tile_matched(assign_interleaved, K, N, formats, interleave_n=interleave_n)
    run_compressed(M, K, N, assignment, tile_dim=TILE_DIM)
