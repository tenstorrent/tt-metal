# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import random
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


def encode_meta(assignment, ct):
    total = len(assignment)
    num_u32 = max(1, (total + 9) // 10)
    meta = [0] * num_u32
    prev_fmt = 0
    for i in range(total):
        c = i % ct
        u, j = divmod(i, 10)
        if j == 0:
            meta[u] |= prev_fmt & 0b11
        fmt = int(assignment[i]) & 0b11
        use_b = 1 if c == 0 else 0
        meta[u] |= use_b << (3 * j + 2)
        meta[u] |= fmt << (3 * j + 3)
        prev_fmt = fmt
    return np.array(meta, dtype=np.uint32).tobytes()


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


def promote_assignment(assignment):
    """If every tile is bfp0, the packed B buffer is empty and the kernel faults —
    promote a single tile to bfp2 so there is at least one non-zero tile to unpack."""
    if all(int(code) == FMT_CODE["bfp0"] for code in assignment):
        assignment = list(assignment)
        assignment[0] = FMT_CODE["bfp2"]
    return assignment


def generate_compressed_matmul_golden(M, K, N, assignment, tile_dim, seed=0):
    assert K % tile_dim == 0 and N % tile_dim == 0, "K and N must be multiples of tile_dim"
    kt = K // tile_dim
    ct = N // tile_dim
    num_tiles = kt * ct
    assert len(assignment) == num_tiles, f"assignment has {len(assignment)} entries, expected kt*ct={num_tiles}"
    assignment = promote_assignment(assignment)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N), dtype=torch.bfloat16)

    b_dequant = torch.zeros((K, N), dtype=torch.bfloat16)

    packed_b_standard = b""
    packed_b_split = b""
    grouped_exp = [b"" for _ in range(len(FMT_CODE))]
    grouped_man = [b"" for _ in range(len(FMT_CODE))]
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
            grouped_exp[code] += full[: tile_dim * tile_dim // 16]
            grouped_man[code] += full[tile_dim * tile_dim // 16 :]

    max_exp_len = max(len(sec) for sec in grouped_exp)

    for code in FMT_CODE.values():
        if tile_counts[code] == 0 or code == 0:
            continue
        packed_b_split += grouped_exp[code]
        packed_b_split += b"\x00" * (max_exp_len - len(grouped_exp[code]))
        packed_b_split += grouped_man[code]

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
        meta=encode_meta(assignment, ct),
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
    # generate_compressed_matmul_golden promotes an all-bfp0 assignment to one bfp2
    # tile, so the packed B buffer is never empty (no need to skip those cases).
    stim = generate_compressed_matmul_golden(M, K, N, assignment, tile_dim)

    config = CompressedStimuliConfig(K, N, stim.packed_a, stim.packed_b_standard, stim.meta)

    formats = InputOutputFormat(
        input_format=DataFormat.Float16_b,
        output_format=output_format,
        input_format_B=DataFormat.Bfp8_b,
    )

    configuration = TestConfig(
        "sources/matmul_custom_compressed_test.cpp",
        formats,
        templates=[],
        runtimes=[
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
            CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
            IN_FACE_DIMS(in0_face_r_dim=M),
        ],
        variant_stimuli=config,
        dest_acc=DestAccumulation.No,
    )

    res = configuration.run().result
    res_tensor = untilize_result(res, M, N, tile_dim, FACE_DIM)

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
    # (kt == K // 32 here since TILE_DIM == 32, matching the per-K-tile noise model.)
    FLOAT16B_DEFAULT_ATOL = 0.05
    ACC_ATOL_PER_KT = 0.005
    active_golden = stim.golden.abs()
    active_golden = active_golden[active_golden > 0]
    mean_active = active_golden.mean().item() if active_golden.numel() else 0.0
    acc_atol = max(FLOAT16B_DEFAULT_ATOL, ACC_ATOL_PER_KT * kt * mean_active)

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


TILE_DIM = 32

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
