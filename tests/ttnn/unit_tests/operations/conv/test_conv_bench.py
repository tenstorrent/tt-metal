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
# ║   helper_sbm = matmul-helper kernel, SubblockMajor (subblock_w == per_core_N enforced).     ║
# ║   helper_trm = matmul-helper kernel, TileRowMajor (subblock_w == per_core_N relaxed).       ║
# ║                                                                                            ║
# ║ Optional manual subblock (overrides the auto-tuner; validated with TT_FATAL):               ║
# ║   TT_CONV_BENCH_SUBBLOCK_H=2 TT_CONV_BENCH_SUBBLOCK_W=2 TT_CONV_BENCH_MODE=helper_trm ...    ║
# ║                                                                                            ║
# ║ IDIOT-PROOFING (the harness TT_FATALs loudly rather than let you misread a result):         ║
# ║   • output_layout is FORCED to ROW_MAJOR and packer_l1_acc is FORCED OFF in every mode      ║
# ║     (so the 3 modes are a fair, bug-free comparison — do NOT change those below).           ║
# ║   • helper_trm on a shape where out_subblock_w == per_core_N (weight_num_subblocks==1)       ║
# ║     fatals: TileRowMajor would be identical to SubblockMajor (no-op). To make helper_trm     ║
# ║     actually differ, the tuner must pick out_subblock_w < per_core_N — that needs            ║
# ║     per_core_N > DST capacity (DST = 4 with fp32_accum=True, 8 with fp32_accum=False), i.e.  ║
# ║     enough output channels (per_core_N = out_channels/32 tiles on height-sharded).           ║
# ║   • width-sharded / 1D-depthwise convs fatal (bench supports HEIGHT/BLOCK sharded only).     ║
# ║   • every run prints a CONV_BENCH[...] line: the mode, per_core_M/N, what the tuner WOULD     ║
# ║     pick for SubblockMajor vs TileRowMajor, the out_subblock actually used, and l1_acc state.║
# ║   • run_conv checks PCC vs torch every run, so a mis-wired mode fails loudly (never a         ║
# ║     silently-wrong perf number).                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════╝
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.conv.test_conv2d import run_conv, torch_tensor_map, HS, BS

# ════════════════════════════════════ EDIT ME ════════════════════════════════════
CONFIG = dict(
    batch_size=1,
    output_channels=256,  # height-sharded per_core_N = out_channels / 32 (tiles). >DST to make helper_trm differ.
    input_channels=256,
    input_height=14,
    input_width=14,
    filter=3,  # square kernel (filter x filter)
    stride=1,
    padding=(1, 1, 1, 1),  # (top, bottom, left, right)
    shard_layout=HS,  # HS or BS only (width-sharded fatals in bench mode)
    act_block_h_override=None,  # None, or a multiple of 32 (e.g. 64) to grow act_block_h (=> more M-subblocks)
    input_dtype=ttnn.bfloat16,  # bfloat16 | bfloat8_b | float32
    output_dtype=ttnn.bfloat16,  # bfloat16 | float32  (bfloat8_b is illegal with ROW_MAJOR output)
    fp32_accum=True,  # True => DST capacity 4 (helps force out_subblock_w < per_core_N => real helper_trm diff)
    math_fidelity=ttnn.MathFidelity.HiFi4,
    has_bias=True,
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
        None,  # weights_dtype (defaults from config)
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
        # DO NOT change these two — the C++ harness forces them anyway; set here so run_conv's PCC
        # comparison matches what the op actually produces (ROW_MAJOR) and the modes stay a fair pair.
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        packer_l1_acc=False,
        run_twice=True,
        input_dtype=c["input_dtype"],
    )
