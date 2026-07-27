# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# Shared utilities for the compressed matmul tests (tile + face), in two parts: the
# B-compression assignment generators (the exact-count DeepSeek-R1 generators + the
# simple assign_* helpers) and the shared device harness (stimulus config, bfp
# tilize/pack, and the run_compressed driver parameterized by a per-test kernel source
# + strategy hooks). run_compressed is the entry point; the detail is on each function.

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import MatmulGolden
from helpers.llk_params import DestAccumulation, MathFidelity
from helpers.logger import logger
from helpers.pack import pack_bfp2_b, pack_bfp4_b, pack_bfp8_b, pack_bfp16
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import CRK_TILE_DIMM, IN_FACE_DIMS, NUM_FACES
from helpers.tile_constants import DEFAULT_TILE_C_DIM, DEFAULT_TILE_R_DIM, FACE_C_DIM
from helpers.tilize_untilize import tilize, untilize
from helpers.unpack import unpack_bfp2_b, unpack_bfp4_b, unpack_bfp8_b
from helpers.utils import passed_test
from ttexalens.tt_exalens_lib import write_to_device

# -----------------------------------------------------------------------------
# Extra includes fixture
# -----------------------------------------------------------------------------

# Extra include dirs the compressed-mm kernels need (the deepseek vendored llk_lib
# + its metal llk_api). Added to TestConfig.INCLUDES for the duration of each test
# by the autouse fixture below; import the fixture into a test module to activate it.
EXTRA_INCLUDES = [
    "-I../../../models/demos/deepseek_v3_b1/kernel_includes/tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib",
    "-I../../../models/demos/deepseek_v3_b1/kernel_includes/tt_metal/hw/ckernels/blackhole/metal/llk_api",
]


@pytest.fixture(autouse=True)
def compressed_mm_include_paths():
    added = [inc for inc in EXTRA_INCLUDES if inc not in TestConfig.INCLUDES]
    TestConfig.INCLUDES.extend(added)
    yield
    for inc in added:
        TestConfig.INCLUDES.remove(inc)


# -----------------------------------------------------------------------------
# Simple assignment generators
# -----------------------------------------------------------------------------

FMT_CODE = {"bfp8": 3, "bfp4": 2, "bfp2": 1, "bfp0": 0}
# bfp0 is the zero/skip tile, and its code being 0 is relied on throughout: a unit is
# tested for "is a zero tile" simply as ``code == 0`` (non-zero as ``code != 0``).
assert FMT_CODE["bfp0"] == 0, "bfp0 code must be 0 (used as the zero-tile sentinel)"


def assign_random(K, N, formats, granularity):
    # Randomly assign formats across the (K//granularity) x (N//granularity) row-major
    # units. Deterministic (fixed seed) for reproducible PCC.
    codes = [FMT_CODE[f] for f in formats]
    num = (K // granularity) * (N // granularity)
    rng = random.Random(0)
    return [rng.choice(codes) for _ in range(num)]


def assign_clustered(K, N, formats, granularity):
    # Assign formats in contiguous, roughly equal-length runs over the (K//granularity)
    # x (N//granularity) row-major units — one cluster per format.
    codes = [FMT_CODE[f] for f in formats]
    num = (K // granularity) * (N // granularity)
    num_runs = min(len(codes), num)
    run_length = num // num_runs
    last_run_length = num - run_length * (num_runs - 1)
    assignment = []
    for i in range(num_runs - 1):
        assignment.extend([codes[i]] * run_length)
    assignment.extend([codes[num_runs - 1]] * last_run_length)
    return assignment


def assign_interleaved(K, N, formats, granularity, interleave_n):
    # Cycle through formats in blocks of interleave_n units over the (K//granularity) x
    # (N//granularity) row-major grid, for any number of formats.
    codes = [FMT_CODE[f] for f in formats]
    num = (K // granularity) * (N // granularity)
    num_blocks = (num + interleave_n - 1) // interleave_n
    assignment = []
    for i in range(num_blocks - 1):
        assignment.extend([codes[i % len(codes)]] * interleave_n)
    assignment.extend([codes[(num_blocks - 1) % len(codes)]] * (num - len(assignment)))
    return assignment


def expand_tiles_to_faces(assignment, kt, ct):
    # Homogeneous 2x2 expansion of a 32x32-tile assignment to 16x16 faces, row-major
    # over (2*kt, 2*ct) — every face inherits its tile's format.
    arr = np.asarray(assignment, dtype=np.int64).reshape(kt, ct)
    return arr.repeat(2, axis=0).repeat(2, axis=1).flatten().tolist()


def assign_tile_matched(assign_fn, K, N, formats, *args):
    # Build a (K//32) x (N//32) 32x32-tile assignment with ``assign_fn`` (as the
    # custom_compressed test does), then expand 2x2 to 16x16 faces so the face kernel
    # runs the identical compression — a controlled face-vs-tile comparison.
    kt, ct = K // DEFAULT_TILE_R_DIM, N // DEFAULT_TILE_C_DIM
    tiles = assign_fn(K, N, formats, DEFAULT_TILE_C_DIM, *args)
    return expand_tiles_to_faces(tiles, kt, ct)


# -----------------------------------------------------------------------------
# Statistical assignment generators
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CompressionStats:
    # Per-precision compression profile at a fixed granularity, ordered high->low
    # precision. shares: fraction per precision (renormalized). mean_run: mean run
    # length in granularity-sized units. granularity: the unit the stats describe
    # (32 = 32x32 tile, 16 = 16x16 face).
    names: tuple
    shares: tuple
    mean_run: tuple
    granularity: int

    def __post_init__(self):
        assert len(self.names) == len(self.shares) == len(self.mean_run) >= 1
        assert all(s >= 0 for s in self.shares) and sum(self.shares) > 0
        assert all(
            r >= 1.0 for r in self.mean_run
        ), "mean_run is in units, must be >= 1"

    @property
    def codes(self):
        # kernel format code (FMT_CODE) per precision level
        return [FMT_CODE[name] for name in self.names]


DEEPSEEK_T420 = CompressionStats(
    names=("bfp4", "bfp2", "bfp0"),
    shares=(0.629, 0.269, 0.103),
    mean_run=(18.51, 5.91, 7.54),
    granularity=DEFAULT_TILE_C_DIM,  # 32 — measured on 32x32 tiles
)

# ---------------------------------------------------------------------------
# Exact-count generators: plant EXACTLY round(share_i * n) tiles of each format
# (largest-remainder rounding — counts sum to n and the mix matches `shares` at any
# size, incl. the small kt*ct shapes here). mean_run only lays the counts into runs,
# ordered to keep adjacent runs different-format where feasible (no directional
# preference — the transition preferences of a Markov model are dropped). Feed to
# expand_tiles_to_faces (tile-granular) or generate_refined_face_assignment.
# ---------------------------------------------------------------------------


def exact_counts(n, shares):
    # Integer per-class counts summing to n, proportional to shares (largest-remainder:
    # floor all, then hand the leftover to the largest fractional parts).
    pi = np.asarray(shares, dtype=float)
    pi = pi / pi.sum()
    raw = pi * n
    counts = np.floor(raw).astype(np.int64)
    leftover = n - counts.sum()
    for c in np.argsort(-(raw - counts))[:leftover]:
        counts[c] += 1
    return counts.tolist()


def partition_into_runs(count, mean_run, rng):
    # Run lengths (each >=1, geometric-shaped, averaging mean_run) summing exactly to count.
    if count <= 0:
        return []
    k = min(count, max(1, round(count / mean_run)))
    lengths = [max(1, int(rng.geometric(min(1.0, 1.0 / mean_run)))) for _ in range(k)]
    # Correct the total to `count` exactly, keeping every run >= 1.
    diff = count - sum(lengths)
    while diff != 0:
        j = rng.integers(k)
        if diff > 0:
            lengths[j] += 1
            diff -= 1
        elif lengths[j] > 1:
            lengths[j] -= 1
            diff += 1
    return lengths


def order_runs(by_code, rng):
    # Order the runs (by_code: FMT_CODE -> list of run lengths) into (code, length) pairs
    # so adjacent codes differ where feasible: greedily take from the code with the most
    # remaining runs that isn't the previous (ties random). A code's runs keep their order;
    # if one code holds > half the runs a same-code pair is forced (just merges two runs).
    order = []
    prev = -1
    total = sum(len(q) for q in by_code.values())
    for _ in range(total):
        cands = [code for code in by_code if by_code[code] and code != prev]
        if not cands:
            cands = [code for code in by_code if by_code[code]]
        top = max(len(by_code[code]) for code in cands)
        pick = int(
            rng.choice([code for code in cands if len(by_code[code]) == top])
        )  # -> output code
        order.append((pick, by_code[pick].pop(0)))
        prev = pick
    return order


def exact_codes(kt, ct, stats, rng):
    # Row-major (kt x ct) grid of FMT_CODEs: exact per-format counts (round(share_i * n),
    # largest-remainder) chopped into geometric runs averaging stats.mean_run[i], then
    # ordered so adjacent formats differ where feasible.
    n = kt * ct
    counts = exact_counts(n, stats.shares)
    by_code = {}  # FMT_CODE -> list of run lengths for that format
    for c in range(len(counts)):
        by_code[stats.codes[c]] = partition_into_runs(counts[c], stats.mean_run[c], rng)
    out = []
    for code, length in order_runs(by_code, rng):
        out += [code] * length
    assert len(out) == n
    return out


def inject_interior_flips(faces, switch_mult, codes, rng):
    # In-place: flip isolated interior faces to a different format until the switch
    # count reaches switch_mult x the homogeneous baseline. Each flip lands where both
    # neighbours share its format (A A A -> A B A, +2 switches); the replacement is drawn
    # uniformly from the other formats, flips kept >2 apart to stay isolated. Needs >=2
    # formats.
    if switch_mult <= 1.0 or len(codes) < 2:
        return
    baseline = int((faces[1:] != faces[:-1]).sum())  # format switches, row-major
    n_flips = round((switch_mult - 1.0) * baseline / 2.0)
    if n_flips <= 0:
        return
    interior = (faces[1:-1] == faces[:-2]) & (faces[1:-1] == faces[2:])
    cand = np.nonzero(interior)[0] + 1
    rng.shuffle(cand)
    used = []
    for i in cand:
        if len(used) >= n_flips:
            break
        if all(abs(i - u) > 2 for u in used):  # keep each flip isolated
            other = [c for c in codes if c != faces[i]]
            faces[i] = rng.choice(other)
            used.append(i)


def generate_exact_assignment(K, N, stats, seed=0):
    # Exact-count assignment over the (K // g) x (N // g) grid, g = stats.granularity:
    # per-format counts round(share_i * n) exactly, run lengths averaging stats.mean_run.
    # Unrefined — use directly for tile stats (g=32) or face stats (g=16).
    rng = np.random.default_rng(seed)
    kt, ct = K // stats.granularity, N // stats.granularity
    return exact_codes(kt, ct, stats, rng)


def generate_refined_face_assignment(K, N, stats, switch_mult=2.0, seed=0):
    # Face (16x16) assignment refined from a TILE-granular substrate: the exact tile
    # assignment expanded 2x2 to faces, then isolated sub-tile flips at density
    # switch_mult (1.0 = plain expansion; the flips make it no longer exact-count).
    # Models a face assignment when only tile stats exist — with real face-granular stats
    # use generate_exact_assignment (g=16) instead. Same seed => same tile substrate as
    # the tile run (controlled face-vs-tile comparison).
    assert (
        stats.granularity == DEFAULT_TILE_C_DIM
    ), "refinement expects tile-granular stats"
    rng = np.random.default_rng(seed)
    kt, ct = K // stats.granularity, N // stats.granularity
    tiles = exact_codes(kt, ct, stats, rng)
    faces = np.asarray(expand_tiles_to_faces(tiles, kt, ct), dtype=np.int64)
    inject_interior_flips(faces, switch_mult, stats.codes, rng)
    return faces.tolist()


# -----------------------------------------------------------------------------
# Shared test harness
# -----------------------------------------------------------------------------


class CompressedStimuliConfig(StimuliConfig):
    def __init__(self, kt, ct, packed_a, packed_b, packed_meta):
        super().__init__(
            buffer_A=torch.zeros(1, dtype=torch.float32),  # placeholder
            stimuli_A_format=DataFormat.Float16_b,
            tile_count_A=kt,
            buffer_B=torch.zeros(1, dtype=torch.float32),  # placeholder
            stimuli_B_format=DataFormat.Bfp8_b,
            tile_count_B=kt * ct,
            buffer_C=torch.zeros(1, dtype=torch.int32),  # placeholder
            stimuli_C_format=DataFormat.UInt32,
            tile_count_C=(len(packed_meta) + 4095) // 4096,
            stimuli_res_format=DataFormat.Float16_b,
            tile_count_res=ct,
        )
        self.packed_a = packed_a
        self.packed_b = packed_b
        self.packed_meta = packed_meta

    def write(self, location: str = "0,0"):
        write_to_device(location, self.buf_a_addr, self.packed_a)
        write_to_device(location, self.buf_b_addr, self.packed_b)
        write_to_device(location, self.buf_c_addr, self.packed_meta)


def pack_bfp_tile(tile, code, tile_dim, face_dim=FACE_C_DIM):
    num_faces = (tile_dim // face_dim) ** 2
    faces = tilize(
        tile.reshape(-1), tile_dimensions=[tile_dim, tile_dim]
    )  # row-major -> face-major
    if code == FMT_CODE["bfp8"]:
        return bytes(pack_bfp8_b(faces, num_faces=num_faces))
    if code == FMT_CODE["bfp4"]:
        return bytes(pack_bfp4_b(faces, num_faces=num_faces))
    if code == FMT_CODE["bfp2"]:
        return bytes(pack_bfp2_b(faces, num_faces=num_faces))
    if code == FMT_CODE["bfp0"]:
        return b""
    raise ValueError(f"Unsupported format {code}")


def unpack_bfp_tile(tile, code, tile_dim, face_dim=FACE_C_DIM):
    num_faces = (tile_dim // face_dim) ** 2
    if code == FMT_CODE["bfp0"]:
        return torch.zeros((tile_dim, tile_dim), dtype=torch.bfloat16)
    if code == FMT_CODE["bfp8"]:
        faces = unpack_bfp8_b(tile, num_faces=num_faces)
    elif code == FMT_CODE["bfp4"]:
        faces = unpack_bfp4_b(tile, num_faces=num_faces)
    elif code == FMT_CODE["bfp2"]:
        faces = unpack_bfp2_b(tile, num_faces=num_faces)
    else:
        raise ValueError(f"Unsupported format {code}")
    return untilize(faces, tile_dimensions=[tile_dim, tile_dim]).reshape(
        tile_dim, tile_dim
    )  # face-major -> row-major


def run_compressed(
    M,
    K,
    N,
    assignment,
    kernel,
    granularity,
    supported_M,
    supported_formats,
    promote,
    pack_b,
    make_meta,
    pcc_threshold=None,
):
    # Shared driver for both compressed-matmul tests. Kernel-specific parts are passed
    # flat: kernel (C++ source), granularity (assignment unit — 32 tile / 16 face),
    # supported_M / supported_formats (input gates), and three strategy hooks:
    #   promote(assignment, cu) -> assignment       # fix up an all-bfp0 assignment
    #   pack_b(tiles) -> (packed_b, aux)             # tiles = [(code, packed_bytes)]
    #   make_meta(assignment, cu, ku, aux) -> meta   # aux is pack_b's side data
    # ku/cu are the assignment row/col counts (K//granularity, N//granularity), which
    # differ from the 32-tile grid kt/ct (K//32, N//32) on the face kernel.
    kt, ct = K // DEFAULT_TILE_R_DIM, N // DEFAULT_TILE_C_DIM  # 32-tile kernel grid
    ku, cu = (
        K // granularity,
        N // granularity,
    )  # assignment rows/cols (= kt/ct on tile, 2x on face)
    assert M in supported_M, f"M must be in {supported_M}"
    assert kt >= 2 and kt % 2 == 0, "kt_dim must be an even number >= 2"
    assert 1 <= ct <= 16, "ct_dim must be in [1, 16]"
    assert (
        K % DEFAULT_TILE_R_DIM == 0 and N % DEFAULT_TILE_C_DIM == 0
    ), "K and N must be multiples of the 32x32 kernel tile"
    assert (
        K % granularity == 0 and N % granularity == 0
    ), "K and N must be multiples of the assignment unit"
    assert (
        len(assignment) == ku * cu
    ), f"assignment has {len(assignment)} entries, expected {ku * cu}"
    assert all(
        code in supported_formats for code in assignment
    ), f"assignment uses a format outside {supported_formats}"

    dist = Counter(assignment)
    parts = [
        f"{name}={round(100 * dist[c] / len(assignment))}%"
        for name, c in FMT_CODE.items()
        if dist[c]
    ]
    logger.info(
        "assignment (M={} K={} N={}): {} (n={})",
        M,
        K,
        N,
        ", ".join(parts),
        len(assignment),
    )

    assignment = promote(assignment, cu)

    # Stimulus + quantization-aware golden: B is packed per the assignment and
    # b_dequant = dequant(quant(B)), so BFP rounding is folded into the golden rather
    # than charged against PCC. The per-kernel pack_b / make_meta turn the packed
    # tiles into the device B buffer and its meta.
    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, N), dtype=torch.bfloat16)
    b_dequant = torch.zeros((K, N), dtype=torch.bfloat16)
    tiles = []  # (code, packed_bytes) per assignment unit, in row-major order
    for i, code in enumerate(assignment):
        r, c = divmod(i, cu)
        blk = torch_b[
            r * granularity : (r + 1) * granularity,
            c * granularity : (c + 1) * granularity,
        ]
        full = pack_bfp_tile(blk, code, tile_dim=granularity)
        b_dequant[
            r * granularity : (r + 1) * granularity,
            c * granularity : (c + 1) * granularity,
        ] = unpack_bfp_tile(full, code, tile_dim=granularity)
        tiles.append((code, full))

    packed_a = b""
    for i in range(kt * 2):
        packed_a += pack_bfp16(torch_a[:, i * FACE_C_DIM : (i + 1) * FACE_C_DIM])
    packed_b, aux = pack_b(tiles)
    meta = make_meta(assignment, cu, ku, aux)

    # Golden via the standard MatmulGolden path (LoFi). Feed the already-bfp-quantized
    # b_dequant as Float16_b so MatmulGolden does NOT re-quantize it, and keep
    # tilize=False — tile layout is handled device-side (pack_bfp_tile / the result
    # reorder below), so the golden stays row-major. Instantiate MatmulGolden directly
    # rather than via get_golden_generator: the harness swaps in a DummyGoldenGenerator
    # (zeros(1024)) during compile-producer, which would break the reshape for narrow-M.
    golden = MatmulGolden()(
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

    configuration = TestConfig(
        kernel,
        InputOutputFormat(
            input_format=DataFormat.Float16_b,
            output_format=DataFormat.Float16_b,
            input_format_B=DataFormat.Bfp8_b,
        ),
        templates=[
            CRK_TILE_DIMM(c_dimm=ct, r_dimm=1, k_dimm=kt),
        ],
        runtimes=[
            NUM_FACES(num_faces=2, num_faces_A=2, num_faces_B=4),
            IN_FACE_DIMS(in0_face_r_dim=M),
        ],
        variant_stimuli=CompressedStimuliConfig(kt, ct, packed_a, packed_b, meta),
        dest_acc=DestAccumulation.No,
    )

    res = configuration.run().result
    # Device packs the result as ct = N//32 tiles; within a tile the two 16-col faces
    # (M x FACE_C_DIM, row-major) sit contiguously, then padding out to a full 32-row tile.
    # Drop the per-tile padding (span = len // ct) and reorder to row-major (M, N).
    res_tensor = torch.as_tensor(res, dtype=torch.bfloat16)
    ct = N // DEFAULT_TILE_C_DIM
    faces_per_tile = DEFAULT_TILE_C_DIM // FACE_C_DIM
    per_tile = res_tensor.reshape(ct, -1)[:, : M * DEFAULT_TILE_C_DIM]
    res_tensor = (
        per_tile.reshape(ct, faces_per_tile, M, FACE_C_DIM)
        .permute(2, 0, 1, 3)
        .reshape(M, N)
    )

    # K-aware absolute floor: the single LoFi MVMUL accumulates the K-deep sum in a
    # bf16 dest, so noise grows ~linearly per K-tile — a floor on small outputs that
    # Float16_b's default atol (0.05) is too tight for at large kt. Scale it by
    # kt * mean|nonzero golden| (never below default; rtol unchanged; PCC is the real
    # gate). 0.005 is calibrated across formats (worst ~0.0034 at kt=16, bfp2/bfp0);
    # mean excludes bfp0's structural zeros, which would otherwise deflate it.
    FLOAT16B_DEFAULT_ATOL = 0.05
    ACC_ATOL_PER_KT = 0.005
    active_golden = golden.abs()
    active_golden = active_golden[active_golden > 0]
    mean_active = active_golden.mean().item() if active_golden.numel() else 0.0
    acc_atol = max(FLOAT16B_DEFAULT_ATOL, ACC_ATOL_PER_KT * kt * mean_active)

    assert passed_test(
        golden,
        res_tensor,
        DataFormat.Float16_b,
        custom_atol=acc_atol,
        custom_pcc_threshold=pcc_threshold,
        print_pcc=True,
    ), f"compressed matmul failed for shape=(M={M}, K={K}, N={N})"
