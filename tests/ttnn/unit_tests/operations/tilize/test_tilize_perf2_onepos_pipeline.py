"""Perf 2 onepos_pipeline harness: column sub-block streaming, graduate vs HEAD A/B under --profile.

TILIZE_OP_CASES="7,8,4,d4096x256"   LOOSE_CASES indices and/or DOMAIN names (below)
TILIZE_OP_VARIANTS="head,grad"      comma-separated tokens  <name>[#<rep>][@KNOB=value[+KNOB=value]]
    head   the op's kernels/ + descriptor (HEAD)
    grad   perf_experiments/onepos_pipeline/graduate/ (descriptor copy + its kernels/, loaded as a module)
    #<rep> a repeat label (same program, interleaved A/B/A/B in one session)
    @...   descriptor-knob overrides on that module (Python literals), e.g.
           grad@ONEPOS_SUB_BLOCK_TILES=0 (off: HEAD's programs), grad@SUB_BLOCK_ONE_POSITION_ONLY=False
TILIZE_OP_CHECK=0                   skip the golden contract check (never used for candidates)
Opt-in: TILIZE_PERF_EXPERIMENTS=1.
"""
import ast
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

import ttnn
import ttnn.operations.tilize.tilize_program_descriptor as pd
from eval.feature_matrix import cartesian
from eval.golden_tests.tilize import helpers
from eval.golden_tests.tilize.feature_spec import LOOSE_CASES, TARGET, _crs, _il, _sh
from ttnn.operations.tilize import INPUT_TAGGERS  # type: ignore

EXP = Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/tilize/perf_experiments/onepos_pipeline"
GRAD = EXP / "graduate"

_L1, _DRAM = ttnn.BufferType.L1, ttnn.BufferType.DRAM
_HS = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_G64 = _crs(((0, 0), (7, 7)))
_BF = dict(dtype=ttnn.bfloat16, output_dtype=ttnn.bfloat16)
_FP = dict(dtype=ttnn.float32, output_dtype=ttnn.float32)


def _case(shape, i, o, extra=None, **dt):
    return {
        "inputs": (
            {
                "input_shape": shape,
                "shard_api": "legacy_2d" if "sharded" in (i["kind"], o["kind"]) else "none",
                "in": i,
                "out": o,
                **(extra or {}),
            },
        ),
        **(dt or _BF),
    }


DOMAIN = {
    # DRAM -> DRAM, 64 Tensix cores x 1 tile-row (one position), by tile-row width
    "d32x16384": _case([1, 1, 32, 16384], _il(_DRAM), _il(_DRAM)),
    "d64x8192": _case([1, 1, 64, 8192], _il(_DRAM), _il(_DRAM)),
    "d2048x128": _case([1, 1, 2048, 128], _il(_DRAM), _il(_DRAM)),
    "d2048x256": _case([1, 1, 2048, 256], _il(_DRAM), _il(_DRAM)),
    "d2048x512": _case([1, 1, 2048, 512], _il(_DRAM), _il(_DRAM)),
    "d2048x1024": _case([1, 1, 2048, 1024], _il(_DRAM), _il(_DRAM)),
    "d2048x2048": _case([1, 1, 2048, 2048], _il(_DRAM), _il(_DRAM)),
    # 2-D split one-position
    "d64x4096": _case([1, 1, 64, 4096], _il(_DRAM), _il(_DRAM)),
    # L1-interleaved source (co-read engages at every width)
    "l1_2048x256": _case([1, 1, 2048, 256], _il(_L1), _il(_DRAM)),
    "l1_2048x512": _case([1, 1, 2048, 512], _il(_L1), _il(_DRAM)),
    # resident input, other widths
    "hs_2048x256": _case([1, 1, 2048, 256], _sh(_L1, _G64, (32, 256), _ROW, _HS), _il(_DRAM)),
    "hs_2048x1024": _case([1, 1, 2048, 1024], _sh(_L1, _G64, (32, 1024), _ROW, _HS), _il(_DRAM)),
    # resident input -> L1 interleaved output
    "hs_2048x512_l1o": _case([1, 1, 2048, 512], _sh(_L1, _G64, (32, 512), _ROW, _HS), _il(_L1)),
    # fp32 (slow lossless tilize path)
    "fp32_hs_2048x512": _case([1, 1, 2048, 512], _sh(_L1, _G64, (32, 512), _ROW, _HS), _il(_DRAM), **_FP),
    "fp32_d2048x512": _case([1, 1, 2048, 512], _il(_DRAM), _il(_DRAM), **_FP),
    "fp32_hso_2048x512": _case([1, 1, 2048, 512], _il(_DRAM), _sh(_L1, _G64, (32, 512), _ROW, _HS), **_FP),
    # resident output, other widths
    "hso_2048x1024": _case([1, 1, 2048, 1024], _il(_DRAM), _sh(_L1, _G64, (32, 1024), _ROW, _HS)),
    # correctness-only coverage of the other paths a one-position walk can take
    "pad_2000x500": _case([1, 1, 2000, 500], _il(_DRAM), _il(_DRAM), extra={"pad_mode": "auto", "pad_value": -7}),
    "tiny16_1024x512": _case([1, 1, 1024, 512], _il(_DRAM), _il(_DRAM), extra={"tile_height": 16}),
    "retile_1024x512": _case(
        [1, 1, 1024, 512], _il(_DRAM), _il(_DRAM), extra={"tile_height": 16, "in_tile_height": 32}
    ),
    "lowl1_2048x256": _case([1, 1, 2048, 256], _il(_DRAM), _il(_DRAM), extra={"low_l1": True}),
    "int32_hs_2048x512": _case(
        [1, 1, 2048, 512], _sh(_L1, _G64, (32, 512), _ROW, _HS), _il(_DRAM), dtype=ttnn.int32, output_dtype=ttnn.int32
    ),
    "u16_d2048x512": _case([1, 1, 2048, 512], _il(_DRAM), _il(_DRAM), dtype=ttnn.uint16, output_dtype=ttnn.uint16),
    "u8_d2048x512": _case([1, 1, 2048, 512], _il(_DRAM), _il(_DRAM), dtype=ttnn.uint8, output_dtype=ttnn.uint8),
    "bfp8_hs_2048x512": _case(
        [1, 1, 2048, 512],
        _sh(_L1, _G64, (32, 512), _ROW, _HS),
        _il(_DRAM),
        dtype=ttnn.bfloat16,
        output_dtype=ttnn.bfloat8_b,
    ),
    "fp32_bf16_hs_2048x512": _case(
        [1, 1, 2048, 512],
        _sh(_L1, _G64, (32, 512), _ROW, _HS),
        _il(_DRAM),
        dtype=ttnn.float32,
        output_dtype=ttnn.bfloat16,
    ),
    "ragged_2d_64x4000": _case([1, 1, 64, 4000], _il(_DRAM), _il(_DRAM), extra={"pad_mode": "auto", "pad_value": 0}),
    "tiny8_512x512": _case([1, 1, 512, 512], _il(_DRAM), _il(_DRAM), extra={"tile_height": 8}),
    "tiny1_64x512": _case([1, 1, 64, 512], _il(_DRAM), _il(_DRAM), extra={"tile_height": 1}),
    "mixed_3200x512": _case([1, 1, 3200, 512], _il(_DRAM), _il(_DRAM)),  # 28 one-position + 36 two-position cores
    # multi-position walks (block_width >= 4): the one-position restriction must be earned
    "d4096x512": _case([1, 1, 4096, 512], _il(_DRAM), _il(_DRAM)),  # 2 positions, bw 16
    "d4096x256": _case([1, 1, 4096, 256], _il(_DRAM), _il(_DRAM)),  # 2 positions, bw 8
    "d8192x256": _case([1, 1, 8192, 256], _il(_DRAM), _il(_DRAM)),  # 4 positions, bw 8
    "d16384x128": _case([1, 1, 16384, 128], _il(_DRAM), _il(_DRAM)),  # 8 positions, bw 4, 2 rows / quantum
    "d8192x512": _case([1, 1, 8192, 512], _il(_DRAM), _il(_DRAM)),  # 4 positions, bw 16
    "hs_4096x512": _case([1, 1, 4096, 512], _sh(_L1, _G64, (64, 512), _ROW, _HS), _il(_DRAM)),  # 2-row shard
    "tiny1_512x512": _case([1, 1, 512, 512], _il(_DRAM), _il(_DRAM), extra={"tile_height": 1}),  # 8 rows, 4 / quantum
    "tiny16_4096x512": _case([1, 1, 4096, 512], _il(_DRAM), _il(_DRAM), extra={"tile_height": 16}),  # 4 rows
    "fp32_d4096x512": _case([1, 1, 4096, 512], _il(_DRAM), _il(_DRAM), **_FP),  # slow tilize path, 2 rows
    "hs_8192x256": _case([1, 1, 8192, 256], _sh(_L1, _G64, (128, 256), _ROW, _HS), _il(_DRAM)),  # 4-row shard
    "l1_4096x512": _case([1, 1, 4096, 512], _il(_L1), _il(_DRAM)),  # L1-interleaved source, 2 rows
    "lowl1_8192x512": _case([1, 1, 8192, 512], _il(_DRAM), _il(_DRAM), extra={"low_l1": True}),
    "retile_2048x512": _case(
        [1, 1, 2048, 512], _il(_DRAM), _il(_DRAM), extra={"tile_height": 16, "in_tile_height": 32}
    ),
    "d4096x1024": _case([1, 1, 4096, 1024], _il(_DRAM), _il(_DRAM)),  # 2 rows, bw 32
    "hs_4096x512_l1o": _case([1, 1, 4096, 512], _sh(_L1, _G64, (64, 512), _ROW, _HS), _il(_L1)),
    # odd block_width (merged 3-tile last sub-block)
    "d2048x160": _case([1, 1, 2048, 160], _il(_DRAM), _il(_DRAM)),
    "hs_2048x224": _case([1, 1, 2048, 224], _sh(_L1, _G64, (32, 224), _ROW, _HS), _il(_DRAM)),
}

CASES = os.environ.get("TILIZE_OP_CASES", "7").split(",")
VARIANTS = os.environ.get("TILIZE_OP_VARIANTS", "head").split(",")
CHECK = os.environ.get("TILIZE_OP_CHECK", "1") != "0"
# TILIZE_OP_BITEXACT=1: every non-head call also runs HEAD's program on the same input into a second
# output tensor, and the two readbacks must be torch.equal (bit-exact vs HEAD, beyond the golden
# contract's PCC on casts). Adds generic ops: correctness runs only, never under --profile.
BITEXACT = os.environ.get("TILIZE_OP_BITEXACT", "0") == "1"


def _get_case(key):
    return LOOSE_CASES[int(key)] if key.isdigit() else DOMAIN[key]


def _axes(case):
    inputs = case["inputs"]
    dt = case.get("dtype", ttnn.bfloat16)
    odt = case.get("output_dtype", dt)
    return next(a for a in cartesian(TARGET, INPUT_TAGGERS, inputs) if a["dtype"] == dt and a["output_dtype"] == odt)


_GRAD_MOD = None


def _grad_module():
    global _GRAD_MOD
    if _GRAD_MOD is None:
        spec = importlib.util.spec_from_file_location("tilize_pd_onepos_grad", GRAD / "tilize_program_descriptor.py")
        _GRAD_MOD = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_GRAD_MOD)
        assert _GRAD_MOD.KERNEL_DIR == GRAD / "kernels", _GRAD_MOD.KERNEL_DIR
    return _GRAD_MOD


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("key", CASES)
def test_onepos_variant(device, monkeypatch, key, variant):
    tmod = sys.modules["ttnn.operations.tilize.tilize"]
    name, _, knobs = variant.partition("@")
    name = name.split("#")[0]
    if name == "grad":
        mod = _grad_module()
        monkeypatch.setattr(tmod, "create_program_descriptor", mod.create_program_descriptor)
    else:
        assert name == "head", name
        mod = pd
    for knob in filter(None, knobs.split("+")):
        k, _, v = knob.partition("=")
        assert hasattr(mod, k), k
        monkeypatch.setattr(mod, k, ast.literal_eval(v))
    log = []
    pairs = []
    orig = tmod.create_program_descriptor

    def spy(*a, **kw):
        if BITEXACT and name != "head":
            inp, out = a[0], a[1]
            ref = tmod._allocate_output(
                ttnn.Shape(list(out.shape)), out.dtype, inp.device(), out.memory_config(), tile_h=kw.get("tile_h", 32)
            )
            ttnn.generic_op([inp, ref], pd.create_program_descriptor(inp, ref, **kw))
            pairs.append((ref, out))
        desc = orig(*a, **kw)
        by = {Path(k.kernel_source).name: k for k in desc.kernels}
        log.append(
            "defines w=%s c=%s bw=%d co_read=%d"
            % (
                [d for d in by["tilize_writer.cpp"].defines if d[0] != "KERNEL_PERF_ZONES"],
                [d for d in by["tilize_compute.cpp"].defines if d[0] != "KERNEL_PERF_ZONES"],
                by["tilize_reader.cpp"].compile_time_args[1],
                by["tilize_reader.cpp"].compile_time_args[28],
            )
        )
        # Program digest (kernel file names, defines, CT / RT args, CB sizes): an OFF variant must
        # reproduce HEAD's digest exactly.
        parts = []
        for k in desc.kernels:
            rts = [list(k.runtime_args[c.x][c.y]) for c in ttnn.corerange_to_cores(k.core_ranges, None, True)]
            parts.append((Path(k.kernel_source).name, sorted(k.defines), list(k.compile_time_args), rts))
        parts.append([cb.total_size for cb in desc.cbs])
        log.append("digest=" + hashlib.sha1(repr(parts).encode()).hexdigest()[:12])
        return desc

    monkeypatch.setattr(tmod, "create_program_descriptor", spy)
    case = _get_case(key)
    if not CHECK:
        monkeypatch.setattr(helpers, "check_output", lambda *a, **k: None)
    helpers.run_tilize(case["inputs"], device=device, extras=case.get("extras"), **_axes(case))
    ttnn.synchronize_device(device)
    for ref, out in pairs:
        r, o = ttnn.to_torch(ref), ttnn.to_torch(out)
        assert r.shape == o.shape and torch.equal(r, o), f"not bit-exact vs HEAD: {key} {variant}"
    if pairs:
        log.append(f"bitexact_vs_head={len(pairs)}")
    print(f"ONEPOS {log}")
    print(f"P2 case={key} variant={variant} nops={sum(e.startswith('digest=') for e in log)} done")
