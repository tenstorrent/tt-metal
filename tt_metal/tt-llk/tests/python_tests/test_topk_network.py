# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
TopK bitonic network sweep (sources/topk_network_test.cpp).

test_topk.py drives the full topk pipeline at K = 32 and checks only the final top-K tile.
This module runs the network entry points on one freshly loaded 2-tile slab per 32-row tile
row and reads back all four tiles (2 value + 2 index), sweeping what the pipeline test pins:

- network op: local sort alone (phases 0..logK-1), or local sort + merge + rebuild;
- K in {2..64} (local sort) / {4..64} (merge + rebuild), rebuild skip_second in {0, 1};
- local-sort direction idir in {0, 1} (the same idir drives merge and rebuild);
- sort mode unstable / comparator-stable in 16-bit and 32-bit DEST, fused and rank-stamped
  keys in 32-bit DEST;
- two value populations: uniform bf16, and a tie-heavy set with signed zeros, bf16
  denormals, infinities and NaNs.

Every run checks that each output (value, index) pair is an input pair and that every row
still holds a permutation of its 64 datums. Local sorts additionally check that each run of K
datums is ordered (alternating direction per run, as the bitonic merge expects; runs of K <= 8
come out descending-first for either idir, since idir only enters at phase 3) and, for the
three stable modes at K = 64, that the whole row matches torch's stable argsort.
"""

from dataclasses import dataclass

import numpy as np
import pytest
import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    INPUT_DIMENSIONS,
    TILE_COUNT,
    TemplateParameter,
)
from helpers.utils import _RECORD_TEST_ORDER
from test_topk import (
    _canon_bits,
    _extract_topk_values_and_indices,
    prepare_input_tensor_for_topk,
    transform_result_tensor_to_right_form,
)

# Quasar has no topk SFPU implementation.
pytestmark = [skip_for_quasar]

NUM_ROWS = (
    64  # two tile rows: the second call per kernel runs the warm replay-cache path
)
NUM_COLS = 128  # 64 values + 64 indices per row -> one 2-tile slab per tile row
W_VALUES = NUM_COLS // 2
TILES_PER_ROW = 4

FORMATS = input_output_formats([DataFormat.Float16_b], same=True)

# (sort mode, 32-bit DEST): fused / rank-stamped keys only exist in 32-bit DEST.
MODE_DEST = [
    ("unstable", DestAccumulation.No),
    ("unstable", DestAccumulation.Yes),
    ("stable", DestAccumulation.No),
    ("stable", DestAccumulation.Yes),
    ("fused", DestAccumulation.Yes),
    ("rank_stamped", DestAccumulation.Yes),
]
MODE_DEST_IDS = [
    f"{m}-{'fp32' if d == DestAccumulation.Yes else 'fp16'}dest" for m, d in MODE_DEST
]

POPULATIONS = ["uniform", "ties_specials"]

NETWORKS = [f"sort_k{k}" for k in (2, 4, 8, 16, 32, 64)] + [
    f"merge_rebuild_k{k}_skip{s}" for k in (4, 8, 16, 32, 64) for s in (0, 1)
]


def _parse_network(network: str):
    network_op = 0 if network.startswith("sort_") else 1
    K = int(network.split("_k")[1].split("_")[0])
    skip_second = int(network.split("_skip")[1]) if network_op == 1 else 0
    return network_op, K, skip_second


@dataclass
class TOPK_NETWORK(TemplateParameter):
    network_op: int = 0  # 0 = local sort, 1 = local sort + merge + rebuild
    k: int = 32
    idir: int = 0
    stable_sort: bool = False
    fused_stable: bool = False
    rank_stamped: bool = False
    rebuild_skip_second: int = 0
    raw_fp32: bool = False

    def convert_to_cpp(self) -> str:
        logk = self.k.bit_length() - 1
        return "\n".join(
            [
                f"constexpr int TOPK_NETWORK_OP = {self.network_op};",
                f"constexpr int TOPK_K = {self.k};",
                f"constexpr int TOPK_LOGK = {logk};",
                f"constexpr int TOPK_IDIR = {self.idir};",
                f"constexpr bool TOPK_STABLE_SORT = {str(self.stable_sort).lower()};",
                f"constexpr bool TOPK_FUSED_STABLE = {str(self.fused_stable).lower()};",
                f"constexpr bool TOPK_RANK_STAMPED = {str(self.rank_stamped).lower()};",
                f"constexpr int TOPK_REBUILD_SKIP_SECOND = {self.rebuild_skip_second};",
                f"constexpr bool TOPK_RAW_FP32 = {str(self.raw_fp32).lower()};",
            ]
        )


def _value_bits(population: str) -> torch.Tensor:
    """[NUM_ROWS, W_VALUES] bf16 bit patterns (uint16), deterministic per population."""
    gen = torch.Generator().manual_seed(0x70C0 + POPULATIONS.index(population))
    if population == "uniform":
        vals = (torch.rand(NUM_ROWS, W_VALUES, generator=gen) * 8.0 - 4.0).to(
            torch.bfloat16
        )
        return vals.view(torch.uint16)
    levels = torch.tensor(
        [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0], dtype=torch.bfloat16
    ).view(torch.uint16)
    specials = torch.tensor(
        [0x8000, 0x0001, 0x007F, 0x8001, 0x7F80, 0xFF80, 0x7FC0, 0xFFC0],
        dtype=torch.int32,
    ).to(torch.uint16)
    pool = torch.cat([levels, specials]).view(torch.int16)
    pick = torch.randint(0, pool.numel(), (NUM_ROWS, W_VALUES), generator=gen)
    return pool[pick].view(torch.uint16)


def _bits_to_float(bits_u16: torch.Tensor) -> torch.Tensor:
    return bits_u16.view(torch.bfloat16).to(torch.float32)


def _stable_sort_key(
    raw_bits_u16: torch.Tensor, canon_bits_u16: torch.Tensor, fp32_dest: bool
) -> torch.Tensor:
    """Order the network sorts by. +-0 and denormals are +0 by the time it runs. A NaN has
    become the same-sign infinity already in 32-bit DEST, but in 16-bit DEST it only does so on
    the way out, so there it still ranks beyond every real infinity of its sign."""
    key = _bits_to_float(canon_bits_u16).to(torch.float64)
    if fp32_dest:
        return key
    key = torch.where(torch.isinf(key), torch.sign(key) * 1e300, key)
    raw = _bits_to_float(raw_bits_u16)
    nan_sign = torch.where((raw_bits_u16.to(torch.int32) & 0x8000) != 0, -1.0, 1.0).to(
        torch.float64
    )
    return torch.where(torch.isnan(raw), nan_sign * float("inf"), key)


@pytest.mark.parametrize("mode_dest", MODE_DEST, ids=MODE_DEST_IDS)
@parametrize(
    formats=FORMATS,
    network=NETWORKS,
    idir=[0, 1],
    population=POPULATIONS,
)
def test_topk_network(formats, mode_dest, network, idir, population):
    sort_mode, dest_acc = mode_dest
    network_op, K, skip_second = _parse_network(network)

    value_bits = _value_bits(population)
    src_rows = torch.zeros(NUM_ROWS, NUM_COLS, dtype=torch.bfloat16)
    src_rows[:, :W_VALUES] = value_bits.view(torch.bfloat16)
    src_A = prepare_input_tensor_for_topk(
        src_rows.flatten(), formats, [NUM_ROWS, NUM_COLS]
    )
    tile_count = (NUM_ROWS // 32) * TILES_PER_ROW

    configuration = TestConfig(
        test_name="sources/topk_network_test.cpp",
        formats=formats,
        templates=[
            DEST_SYNC(),
            TOPK_NETWORK(
                network_op=network_op,
                k=K,
                idir=idir,
                stable_sort=sort_mode == "stable",
                fused_stable=sort_mode == "fused",
                rank_stamped=sort_mode == "rank_stamped",
                rebuild_skip_second=skip_second,
            ),
        ],
        runtimes=[
            INPUT_DIMENSIONS(NUM_ROWS // 32, NUM_COLS // 32),
            TILE_COUNT(tile_count),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_A.clone(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
    )

    res = configuration.run().result
    if _RECORD_TEST_ORDER:
        return

    res_tensor = torch.tensor(res, dtype=format_dict[formats.output_format])
    res_tensor = transform_result_tensor_to_right_form(
        res_tensor, formats, W_VALUES, [NUM_ROWS, NUM_COLS]
    )
    out_bits, out_idx = _extract_topk_values_and_indices(
        res_tensor, formats, NUM_ROWS, W_VALUES
    )

    # Both engines canonicalise the datacopied bf16 slab (+-0 / denormals -> +0, NaN -> same-sign inf).
    canon = _canon_bits(value_bits)
    for row in range(NUM_ROWS):
        idx = out_idx[row]
        assert torch.equal(
            torch.sort(idx).values, torch.arange(W_VALUES)
        ), f"row {row}: indices are not a permutation: {idx.tolist()}"
        expected_bits = canon[row].view(torch.int16)[idx].view(torch.uint16)
        assert torch.equal(
            out_bits[row], expected_bits
        ), f"row {row}: value/index pairing broken\n  values {out_bits[row].tolist()}\n  expect {expected_bits.tolist()}"

        if network_op != 0:
            continue
        vals = _bits_to_float(out_bits[row])
        for run in range(W_VALUES // K):
            seg = vals[run * K : (run + 1) * K]
            # idir only enters the network at phase 3: sorts of K <= 8 (phases 0..2) always
            # come out descending-first, whatever idir says.
            first_descending = idir == 0 or K <= 8
            descending = first_descending == (run % 2 == 0)
            ordered = (
                torch.all(seg[:-1] >= seg[1:])
                if descending
                else torch.all(seg[:-1] <= seg[1:])
            )
            assert (
                ordered
            ), f"row {row} run {run}: not {'descending' if descending else 'ascending'}: {seg.tolist()}"
        if K == W_VALUES and sort_mode != "unstable":
            golden = torch.argsort(
                _stable_sort_key(
                    value_bits[row], canon[row], dest_acc == DestAccumulation.Yes
                ),
                descending=(idir == 0),
                stable=True,
            )
            assert torch.equal(
                idx, golden
            ), f"row {row}: stable order mismatch\n  got    {idx.tolist()}\n  expect {golden.tolist()}"


# =============================================================================
# Raw fp32 words in 32-bit DEST (test_topk_network_fp32_raw)
# =============================================================================
#
# The bf16 slab above reaches DEST through the SrcA datacopy, which canonicalises signed
# zeros, denormals and NaNs before the network sees them. Here Float32 value words and
# integer index words are unpacked straight into 32-bit DEST (no transpose: the network sorts
# the DEST columns of the untransposed tiles), so the comparator network sees every fp32 bit
# class. Index word j names flat position j of the tile row's 2048 value words.

FP32_POPULATIONS = ["uniform_fp32", "specials_fp32"]
_FP32_WORDS_PER_ROW = 2 * 1024

_FP32_SPECIALS = [
    0x00000000,
    0x80000000,  # +-0
    0x00000001,
    0x80000001,
    0x007FFFFF,
    0x807FFFFF,
    0x00012345,
    0x80012345,  # denormals
    0x00800000,
    0x80800000,  # +-FLT_MIN
    0x3F800000,
    0xBF800000,
    0x40000000,
    0xC0000000,
    0x3F800001,
    0xBF800001,  # +-1, +-2, next-after
    0x7F7FFFFF,
    0xFF7FFFFF,  # +-FLT_MAX
    0x7F800000,
    0xFF800000,  # +-inf
    0x7FC00000,
    0xFFC00000,
    0x7F800001,
    0xFF800001,
    0x7FFFFFFF,
    0xFFFFFFFF,  # NaNs
]


def _fp32_value_words(population: str, tile_rows: int) -> np.ndarray:
    rng = np.random.default_rng(0xF32 + FP32_POPULATIONS.index(population))
    n = tile_rows * _FP32_WORDS_PER_ROW
    if population == "uniform_fp32":
        return (
            (rng.random(n, dtype=np.float32) * 8.0 - 4.0)
            .astype(np.float32)
            .view(np.uint32)
        )
    pool = np.array(_FP32_SPECIALS, dtype=np.uint32)
    return pool[rng.integers(0, pool.size, n)]


def _is_zero_or_denormal(words: np.ndarray) -> np.ndarray:
    return (words & 0x7F800000) == 0


@pytest.mark.parametrize("sort_mode", ["unstable", "stable"])
@parametrize(
    formats=input_output_formats([DataFormat.Float32], same=True),
    network=NETWORKS,
    idir=[0, 1],
    population=FP32_POPULATIONS,
)
def test_topk_network_fp32_raw(formats, sort_mode, network, idir, population):
    network_op, K, skip_second = _parse_network(network)
    tile_rows = NUM_ROWS // 32

    values = _fp32_value_words(population, tile_rows).reshape(
        tile_rows, _FP32_WORDS_PER_ROW
    )
    indices = np.broadcast_to(
        np.arange(_FP32_WORDS_PER_ROW, dtype=np.uint32), values.shape
    )
    l1_words = np.concatenate([values, indices], axis=1).reshape(
        -1
    )  # per tile row: 2 value + 2 index tiles
    src_A = torch.from_numpy(l1_words.view(np.float32).copy())
    tile_count = tile_rows * TILES_PER_ROW

    configuration = TestConfig(
        test_name="sources/topk_network_test.cpp",
        formats=formats,
        templates=[
            DEST_SYNC(),
            TOPK_NETWORK(
                network_op=network_op,
                k=K,
                idir=idir,
                stable_sort=sort_mode == "stable",
                rebuild_skip_second=skip_second,
                raw_fp32=True,
            ),
        ],
        runtimes=[
            INPUT_DIMENSIONS(NUM_ROWS // 32, NUM_COLS // 32),
            TILE_COUNT(tile_count),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_A.clone(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_count,
            tile_count_B=tile_count,
            tile_count_res=tile_count,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )

    configuration.run()
    if _RECORD_TEST_ORDER:
        return
    raw = configuration.variant_stimuli.collect_raw_result_bytes(
        TestConfig.TENSIX_LOCATION
    )
    out = np.frombuffer(raw, dtype=np.uint32).reshape(
        tile_rows, 2 * _FP32_WORDS_PER_ROW
    )

    for row in range(tile_rows):
        out_vals = out[row, :_FP32_WORDS_PER_ROW]
        out_idx = out[row, _FP32_WORDS_PER_ROW:]
        assert np.array_equal(
            np.sort(out_idx), np.arange(_FP32_WORDS_PER_ROW)
        ), f"tile row {row}: indices are not a permutation"
        src = values[row][out_idx]
        # Lanes whose input was a zero or denormal may legitimately come back as a (signed) zero
        # (the stable engine folds -0.0, the DEST load path may flush); every other lane must
        # keep its exact word.
        exact = ~_is_zero_or_denormal(src)
        mismatched = np.nonzero(exact & (out_vals != src))[0]
        assert mismatched.size == 0, (
            f"tile row {row}: value/index pairing broken at {mismatched[:8].tolist()}: "
            f"got {[hex(v) for v in out_vals[mismatched[:8]]]} expected {[hex(v) for v in src[mismatched[:8]]]}"
        )
