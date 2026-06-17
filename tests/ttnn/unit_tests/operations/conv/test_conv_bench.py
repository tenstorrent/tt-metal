# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ╔══════════════════════════════════════════════════════════════════════════════════════════╗
# ║ conv_bench — conv kernel profiling harness (conv_bench branch, pure test scaffolding).      ║
# ║                                                                                            ║
# ║ Purpose: profile the conv compute kernel WITH and WITHOUT the matmul helper / subblock      ║
# ║ relaxation, to understand why the matmul helper gives a perf win but conv doesn't.          ║
# ║                                                                                            ║
# ║ Edit CONFIG below (shapes / dtypes / flags / optional manual subblock), then run ONE mode:  ║
# ║                                                                                            ║
# ║   TT_CONV_BENCH_MODE=main       bash scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv_bench.py   ║
# ║   TT_CONV_BENCH_MODE=helper_sbm bash scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv_bench.py   ║
# ║   TT_CONV_BENCH_MODE=helper_trm bash scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/conv/test_conv_bench.py   ║
# ║                                                                                            ║
# ║   main       = main's verbatim no-helper conv kernel (SubblockMajor, hand-written matmul).  ║
# ║   helper_sbm = this branch's kernel, TileRowMajor auto-select forced OFF (pure SBM + pin).  ║
# ║   helper_trm = this branch's kernel with TRM+pin engaged on the conv's REAL output layout   ║
# ║                (ROW_MAJOR and TILE both); pin stays ON; relaxed subblock is tuner-derived.  ║
# ║                Add TT_CONV_BENCH_FORCE_TRM=1 to skip the production ROI gate so any         ║
# ║                hard-eligible conv runs TRM. Hard constraints stay: HEIGHT_SHARDED, no bias, ║
# ║                packer_l1_acc, bf16/fp32 weights, TILE-out partials alias. Ineligible convs  ║
# ║                fall back to helper_sbm with a CONV_BENCH fallback log (no FATAL).           ║
# ║                                                                                            ║
# ║ Optional manual subblock (overrides the host SBM tuner pick; validated with TT_FATAL —      ║
# ║ the TRM relaxed subblock stays tuner-derived in the factory and cannot be set manually):    ║
# ║   TT_CONV_BENCH_SUBBLOCK_H=1 TT_CONV_BENCH_SUBBLOCK_W=2 TT_CONV_BENCH_MODE=helper_sbm ...    ║
# ║                                                                                            ║
# ║ IDIOT-PROOFING (the harness fails loudly rather than let you misread a result):             ║
# ║   • output_layout / packer_l1_acc / weights_dtype are REAL per-conv via CB_* env (defaults:  ║
# ║     tile out, l1_acc on) so all three modes match how models run the conv.                  ║
# ║   • helper_trm ineligible convs run SBM and the dispatch log shows trm_fallback_sbm=true —   ║
# ║     a helper_trm row equal to helper_sbm means FALLBACK, not a null result.                 ║
# ║   • width-sharded / 1D-depthwise convs fatal (bench supports HEIGHT/BLOCK sharded only).     ║
# ║   • every run prints CONV_BENCH[...] lines: tuner SubblockMajor vs TileRowMajor picks (host) ║
# ║     and USING + trm_pin/trm_forced/trm_fallback_sbm (factory dispatch).                     ║
# ║   • run_conv checks PCC vs torch every run, so a mis-wired mode fails loudly (never a         ║
# ║     silently-wrong perf number).                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════╝
import os
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, BS


# ════════════════════════════════════ EDIT ME ════════════════════════════════════
# GH #45995 measurement note: the CONFIG values below are the harness defaults (unchanged).
# Each field is ALSO overridable via a CB_* env var so a whole candidate sweep can be driven
# by env (alongside the mode env TT_CONV_BENCH_MODE) without re-editing the file per run.
# This is pure test scaffolding — no kernel / tuner / factory / harness-logic is touched.
def _e(name, default, cast):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


_SHARD = {"HS": HS, "BS": BS}
_DTYPE = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "float32": ttnn.float32}
_FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}
_LAYOUT = {"tile": ttnn.TILE_LAYOUT, "row_major": ttnn.ROW_MAJOR_LAYOUT}


def _bool(v):
    return str(v).lower() in ("1", "true", "yes")


def _opt_int(v):
    return None if str(v).lower() == "none" else int(v)


def _pad(v):
    return tuple(int(x) for x in v.split(","))


CONFIG = dict(
    batch_size=_e("CB_BATCH", 1, int),
    output_channels=_e(
        "CB_OUT_CH", 256, int
    ),  # height-sharded per_core_N = out_channels / 32 (tiles). >DST to make helper_trm differ.
    input_channels=_e("CB_IN_CH", 256, int),
    input_height=_e("CB_H", 14, int),
    input_width=_e("CB_W", 14, int),
    filter=_e("CB_FILTER", 3, int),  # square kernel (filter x filter)
    stride=_e("CB_STRIDE", 1, int),
    padding=_e("CB_PAD", (1, 1, 1, 1), _pad),  # (top, bottom, left, right)
    shard_layout=_e("CB_SHARD", HS, lambda v: _SHARD[v]),  # HS or BS only (width-sharded fatals in bench mode)
    act_block_h_override=_e(
        "CB_ABH", None, _opt_int
    ),  # None, or a multiple of 32 (e.g. 64) to grow act_block_h (=> more M-subblocks)
    input_dtype=_e("CB_IN_DTYPE", ttnn.bfloat16, lambda v: _DTYPE[v]),  # bfloat16 | bfloat8_b | float32
    output_dtype=_e(
        "CB_OUT_DTYPE", ttnn.bfloat16, lambda v: _DTYPE[v]
    ),  # bfloat16 | float32  (bfloat8_b is illegal with ROW_MAJOR output)
    fp32_accum=_e(
        "CB_FP32_ACCUM", True, _bool
    ),  # True => DST capacity 4 (helps force out_subblock_w < per_core_N => real helper_trm diff)
    math_fidelity=_e("CB_FIDELITY", ttnn.MathFidelity.HiFi4, lambda v: _FID[v]),
    has_bias=_e("CB_BIAS", True, _bool),
    # Real per-conv settings (NO LONGER harness-forced — the bench wiring now passes these through so
    # main/helper_sbm baselines match how the model actually runs the conv):
    weights_dtype=_e(
        "CB_WEIGHTS_DTYPE", None, lambda v: None if str(v).lower() == "none" else _DTYPE[v]
    ),  # bf8/bf16/fp32/none
    output_layout=_e(
        "CB_OUT_LAYOUT", ttnn.TILE_LAYOUT, lambda v: _LAYOUT[v]
    ),  # tile (real default) | row_major (relaxation regime)
    packer_l1_acc=_e("CB_L1_ACC", True, _bool),  # real default ON; set false for the row-major relaxation regime
)
# ══════════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_conv_bench(device, torch_tensor_map):
    c = CONFIG
    config_override = {"act_block_h": c["act_block_h_override"]} if c["act_block_h_override"] else None
    run_conv(
        device,
        torch_tensor_map,
        c["math_fidelity"],
        c["output_dtype"],
        c["weights_dtype"],  # real per-conv weights dtype (None -> fp32)
        c["batch_size"],
        c["output_channels"],
        c["input_channels"],
        c["input_height"],
        c["input_width"],
        c["filter"],
        c["filter"],
        c["stride"],
        c["stride"],
        c["padding"],
        config_override,
        shard_layout=c["shard_layout"],
        has_bias=c["has_bias"],
        fp32_accum=c["fp32_accum"],
        # Real per-conv settings (no longer harness-forced): main/helper_sbm baselines now match how the
        # model runs the conv. For the row-major relaxation study set CB_OUT_LAYOUT=row_major + CB_L1_ACC=false.
        output_layout=c["output_layout"],
        packer_l1_acc=c["packer_l1_acc"],
        run_twice=True,
        input_dtype=c["input_dtype"],
    )
