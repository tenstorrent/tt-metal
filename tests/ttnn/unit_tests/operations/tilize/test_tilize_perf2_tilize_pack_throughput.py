# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Perf 2 / tilize_pack_throughput harness: LOOSE_CASES (+ dtype / shape overrides) x kernel dirs.

TPT_CASES="9,10,b9,x0"   "N"  = LOOSE_CASES[N]
                         "d<tag>:<in>:<out>" = <tag> with dtypes overridden (bf16 f32 bf8 bf4 u8 u16 u32 i32)
                         "bN" = LOOSE_CASES[N] with output_dtype bfloat8_b
                         "xN" = EXTRA[N] below (domain sweep scenarios)
TPT_VARIANTS="head,tilize_pack_throughput/kernels_s4"   "head" = the op's kernels/, else a dir
                         relative to ttnn/ttnn/operations/tilize/perf_experiments/
Every run is golden-checked (helpers.check_output) AND, when "head" ran first on the same case,
bit-compared (torch.equal) against head's readback of the same seeded input.
Run under `scripts/run_safe_pytest.sh --profile ... -s`; label with p2_breakdown/label_ns.py
(prints `P2 case=.. variant=..` lines). Opt-in: TILIZE_PERF_EXPERIMENTS=1.
"""
import os
from pathlib import Path

import pytest
import torch

import ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from eval.feature_matrix import cartesian
from eval.golden_tests.tilize import helpers
from eval.golden_tests.tilize import feature_spec as fs
from eval.golden_tests.tilize.feature_spec import LOOSE_CASES, TARGET
from ttnn.operations.tilize import INPUT_TAGGERS  # type: ignore

EXP = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/tilize/perf_experiments"
CASES = os.environ.get("TPT_CASES", "9,10").split(",")  # ',' separated; dtype tags use ':'
VARIANTS = os.environ.get("TPT_VARIANTS", "head").split(",")
CHECK = os.environ.get("TPT_CHECK", "1") != "0"  # 0 = ablated (incorrect) variants
# Force the user's ComputeKernelConfig knobs (as if the caller had passed them): regime coverage only.
FULL_SYNC = os.environ.get("TPT_FULL_SYNC", "0") == "1"  # dst_full_sync_en=True
FP32_DEST = os.environ.get("TPT_FP32_DEST", "0") == "1"  # fp32_dest_acc_en=True where allowed

_BF = dict(dtype=ttnn.bfloat16, output_dtype=ttnn.bfloat16)
_G64 = fs._crs(((0, 0), (7, 7)))
_G16 = fs._crs(((0, 0), (3, 3)))
_G32 = fs._crs(((0, 0), (7, 3)))
EXTRA = [
    # x0: HEIGHT resident, 2 tile-rows x 16 tiles per core
    {
        "inputs": (
            {
                "input_shape": [1, 1, 4096, 512],
                "shard_api": "legacy_2d",
                "in": fs._sh(fs._L1, _G64, (64, 512), fs._ROW, fs._HEIGHT),
                "out": fs._sh(fs._L1, _G64, (64, 512), fs._ROW, fs._HEIGHT),
            },
        ),
        **_BF,
    },
    # x1: HEIGHT resident, 1 tile-row x 32 tiles (block_width 32)
    {
        "inputs": (
            {
                "input_shape": [1, 1, 2048, 1024],
                "shard_api": "legacy_2d",
                "in": fs._sh(fs._L1, _G64, (32, 1024), fs._ROW, fs._HEIGHT),
                "out": fs._sh(fs._L1, _G64, (32, 1024), fs._ROW, fs._HEIGHT),
            },
        ),
        **_BF,
    },
    # x2: HEIGHT resident, 1 tile-row x 2 tiles (narrow)
    {
        "inputs": (
            {
                "input_shape": [1, 1, 2048, 64],
                "shard_api": "legacy_2d",
                "in": fs._sh(fs._L1, _G64, (32, 64), fs._ROW, fs._HEIGHT),
                "out": fs._sh(fs._L1, _G64, (32, 64), fs._ROW, fs._HEIGHT),
            },
        ),
        **_BF,
    },
    # x3: HEIGHT resident, 4 tile-rows x 5 tiles (odd block width)
    {
        "inputs": (
            {
                "input_shape": [1, 1, 8192, 160],
                "shard_api": "legacy_2d",
                "in": fs._sh(fs._L1, _G64, (128, 160), fs._ROW, fs._HEIGHT),
                "out": fs._sh(fs._L1, _G64, (128, 160), fs._ROW, fs._HEIGHT),
            },
        ),
        **_BF,
    },
    # x4: HEIGHT resident, 1 tile-row x 9 tiles (odd, > DEST half)
    {
        "inputs": (
            {
                "input_shape": [1, 1, 2048, 288],
                "shard_api": "legacy_2d",
                "in": fs._sh(fs._L1, _G64, (32, 288), fs._ROW, fs._HEIGHT),
                "out": fs._sh(fs._L1, _G64, (32, 288), fs._ROW, fs._HEIGHT),
            },
        ),
        **_BF,
    },
    # x5: DRAM interleaved wide [1,1,8192,256]
    {
        "inputs": (
            {"input_shape": [1, 1, 8192, 256], "shard_api": "none", "in": fs._il(fs._DRAM), "out": fs._il(fs._DRAM)},
        ),
        **_BF,
    },
    # x6: HEIGHT resident, 1 tile-row x 3 tiles
    {
        "inputs": (
            {
                "input_shape": [1, 1, 2048, 96],
                "shard_api": "legacy_2d",
                "in": fs._sh(fs._L1, _G64, (32, 96), fs._ROW, fs._HEIGHT),
                "out": fs._sh(fs._L1, _G64, (32, 96), fs._ROW, fs._HEIGHT),
            },
        ),
        **_BF,
    },
    # x7: tiny output tiles (tile_height 16) DRAM -> DRAM
    {
        "inputs": (
            {
                "input_shape": [1, 1, 2048, 512],
                "shard_api": "none",
                "tile_height": 16,
                "in": fs._il(fs._DRAM),
                "out": fs._il(fs._DRAM),
            },
        ),
        **_BF,
    },
    # x8: retile 32 -> 16 DRAM -> DRAM
    {
        "inputs": (
            {
                "input_shape": [1, 1, 2048, 512],
                "shard_api": "none",
                "tile_height": 16,
                "in_tile_height": 32,
                "in": fs._il(fs._DRAM),
                "out": fs._il(fs._DRAM),
            },
        ),
        **_BF,
    },
    # x9: low_l1 focus shape
    {
        "inputs": (
            {
                "input_shape": [1, 1, 16384, 64],
                "shard_api": "none",
                "low_l1": True,
                "in": fs._il(fs._DRAM),
                "out": fs._il(fs._DRAM),
            },
        ),
        **_BF,
    },
    # x10: padded (auto, zero) unaligned DRAM
    {
        "inputs": (
            {
                "input_shape": [1, 1, 1000, 500],
                "shard_api": "none",
                "pad_mode": "auto",
                "pad_value": 0.0,
                "in": fs._il(fs._DRAM),
                "out": fs._il(fs._DRAM),
            },
        ),
        **_BF,
    },
    # x11: wide DRAM [1,1,2048,1024]
    {
        "inputs": (
            {"input_shape": [1, 1, 2048, 1024], "shard_api": "none", "in": fs._il(fs._DRAM), "out": fs._il(fs._DRAM)},
        ),
        **_BF,
    },
]

_HEAD_OUT = {}


_DT = {
    "bf16": ttnn.bfloat16,
    "f32": ttnn.float32,
    "bf8": ttnn.bfloat8_b,
    "bf4": ttnn.bfloat4_b,
    "u8": ttnn.uint8,
    "u16": ttnn.uint16,
    "u32": ttnn.uint32,
    "i32": ttnn.int32,
}


def _case(tag):
    if tag.startswith("d"):  # "d<base tag>:<in dtype>:<out dtype>", e.g. d9:u8:u8 or dx2:f32:bf16
        base, din, dout = tag[1:].split(":")
        c = _case(base)
        c["dtype"], c["output_dtype"] = _DT[din], _DT[dout]
        return c
    if tag.startswith("x"):
        return dict(EXTRA[int(tag[1:])])
    if tag.startswith("b"):
        c = dict(LOOSE_CASES[int(tag[1:])])
        c["output_dtype"] = ttnn.bfloat8_b
        return c
    return dict(LOOSE_CASES[int(tag)])


def _axes(case):
    dt = case.get("dtype", ttnn.bfloat16)
    odt = case.get("output_dtype", dt)
    return next(
        a for a in cartesian(TARGET, INPUT_TAGGERS, case["inputs"]) if a["dtype"] == dt and a["output_dtype"] == odt
    )


PAIRS = [(c, v) for c in CASES for v in VARIANTS]


@pytest.mark.parametrize("case_tag,variant", PAIRS, ids=[f"{c}-{v.split('/')[-1]}" for c, v in PAIRS])
def test_tpt_variant(device, monkeypatch, case_tag, variant):
    if variant != "head":
        d = Path(variant)
        monkeypatch.setattr(pd, "KERNEL_DIR", d if d.is_absolute() else EXP / d)
    case = _case(case_tag)
    axes = _axes(case)
    if not CHECK:
        monkeypatch.setattr(helpers, "check_output", lambda *a, **k: None)
    if FULL_SYNC or FP32_DEST:
        orig_init = pd.NumericConfig.__init__

        def forced_init(self, in_dtype, out_dtype, compute_kernel_config=None):
            ckc = ttnn.WormholeComputeKernelConfig(fp32_dest_acc_en=FP32_DEST, dst_full_sync_en=FULL_SYNC)
            orig_init(self, in_dtype, out_dtype, ckc)

        monkeypatch.setattr(pd.NumericConfig, "__init__", forced_init)
    torch.manual_seed(1234)
    _, tt_out = helpers.run_tilize(case["inputs"], device=device, extras=case.get("extras"), **axes)
    ttnn.synchronize_device(device)
    rb = ttnn.to_torch(tt_out)
    if variant == "head":
        _HEAD_OUT[case_tag] = rb
    elif CHECK and case_tag in _HEAD_OUT:
        assert torch.equal(rb.float(), _HEAD_OUT[case_tag].float()), f"{variant} != head on case {case_tag}"
    print(f"P2 case={case_tag} variant={variant} done")
