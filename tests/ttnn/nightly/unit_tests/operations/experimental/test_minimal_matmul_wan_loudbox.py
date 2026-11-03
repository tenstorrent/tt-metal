# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os
import statistics

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import comp_pcc
from models.tt_dit.utils.matmul import get_matmul_config

# Wan2.2 T2V A14B per-device matmuls on a Blackhole LoudBox (dim 5120, ffn 13824, 480p).
# 2x4 preset (SP=4, TP=2): local M = 8192, N is the TP-sharded width.
# 2x2 preset (SP=2, TP=2): local M = 16384.
PRESET_M = {"2x4": 8192, "2x2": 16384}
PRESETS = list(PRESET_M)

QKV_N = 7680
OUT_N = 2560
FF1_N = 6912
CROSS_KV_M = 512
CROSS_KV_N = 5120
DIM = 5120
FF2_K = 6912

PCC_MIN = 0.9995
RMSE_MAX = 0.02
WARMUP = 2


def compute_config(device, approx):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=approx,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def quality(golden, out):
    _, pcc = comp_pcc(golden, out)
    rmse = torch.nn.functional.mse_loss(golden, out).sqrt().item() / golden.std().item()
    logger.info(f"PCC {pcc:.7f} relative RMSE {rmse:.5f}")
    return pcc, rmse


def to_tt(device, t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def matmul_config(device, M, K, N):
    # The model's heuristic blocking (16 M tiles) overflows the 1.5 MB L1 with the fp32 intermediate at these M,
    # so the generic 8x8x8 / 2x2 fallback of get_matmul_config is used.
    return get_matmul_config(M, K, N, device.compute_with_storage_grid_size())


def torch_inputs(M, K, N, bias=True):
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.float32)
    w = torch.randn((K, N), dtype=torch.float32)
    b = torch.randn((1, N), dtype=torch.float32) if bias else None
    return x, w, b


def profile_ns(device, call, n):
    from tests.ttnn.profiling.realtime_profiler_utils import profile_realtime_program

    for _ in range(WARMUP):
        call()
    ttnn.synchronize_device(device)
    ns = []
    for _ in range(n):
        _, rec = profile_realtime_program(device, call, collect_all=True)
        mm = [r for r in rec if any("minimal_matmul" in k for k in r["kernel_sources"])]
        assert len(mm) == 1, f"expected one minimal_matmul program record, got {len(mm)} of {len(rec)}"
        ns.append(mm[0]["duration_ns"])
    return ns


def report(tag, ns):
    for i, v in enumerate(ns):
        print(f"MMAB-SAMPLE {tag} i={i} duration_ns={v:.0f}")
    print(f"MMAB-MEDIAN {tag} duration_ns={statistics.median(ns):.0f}")


def linear_case(device, M, K, N, approx, activation=None, check=True):
    x, w, b = torch_inputs(M, K, N)
    tt_x, tt_w, tt_b = to_tt(device, x), to_tt(device, w), to_tt(device, b)
    cfg = matmul_config(device, M, K, N)
    ckc = compute_config(device, approx)

    def call():
        return ttnn.experimental.minimal_matmul(
            input_tensor=tt_x,
            weight_tensor=tt_w,
            bias_tensor=tt_b,
            config=cfg,
            fused_activation=activation,
            compute_kernel_config=ckc,
        )

    return call, (x, w, b)


def torch_linear(x, w, b, gelu_tanh=False):
    with torch.no_grad():
        o = x @ w + b
        if gelu_tanh:
            o = torch.nn.functional.gelu(o, approximate="tanh")
    return o


def check_quality(golden, out, pcc_min=PCC_MIN, rmse_max=RMSE_MAX):
    pcc, rmse = quality(golden, out)
    assert pcc > pcc_min
    assert rmse < rmse_max


LINEAR_CASES = {
    # id: (K, N, math approx, gelu_tanh)
    "cross_q_out": (DIM, OUT_N, True, False),
    "ff1_gelu_tanh": (DIM, FF1_N, False, True),
    "ff2": (FF2_K, DIM, False, False),
}


def split_case(device, M, K, N, chunks):
    x, w, b = torch_inputs(M, K, N)
    tt_x, tt_w, tt_b = to_tt(device, x), to_tt(device, w), to_tt(device, b)
    cfg = matmul_config(device, M, K, N)
    ckc = compute_config(device, True)

    def call():
        return ttnn.experimental.minimal_matmul_split(
            tt_x, tt_w, chunks=chunks, dim=-1, bias_tensor=tt_b, compute_kernel_config=ckc, config=cfg
        )

    return call, (x, w, b)


# The torch references and device uploads live in fixtures so that the per-test timeout covers only the op and the
# comparison.
@pytest.fixture
def linear_prepared(device, preset, case):
    K, N, approx, gelu = LINEAR_CASES[case]
    activation = ttnn.UnaryOpType.GELU_TANH if gelu else None
    call, (x, w, b) = linear_case(device, PRESET_M[preset], K, N, approx, activation)
    return call, torch_linear(x, w, b, gelu)


@pytest.fixture
def qkv_prepared(device, preset):
    call, (x, w, b) = split_case(device, PRESET_M[preset], DIM, QKV_N, 3)
    return call, torch.chunk(torch_linear(x, w, b), 3, dim=-1)


@pytest.fixture
def cross_kv_prepared(device):
    call, (x, w, b) = split_case(device, CROSS_KV_M, DIM, CROSS_KV_N, 2)
    return call, torch.chunk(torch_linear(x, w, b), 2, dim=-1)


@pytest.fixture
def out_addcmul_prepared(device, preset):
    return addcmul_case(device, PRESET_M[preset], DIM, OUT_N)


@pytest.fixture
def out_addcmul_full_gate_prepared(device):
    return addcmul_case(device, PRESET_M["2x4"], DIM, OUT_N, full_gate=True)


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("case", list(LINEAR_CASES))
def test_wan_linear(linear_prepared, preset, case):
    call, golden = linear_prepared
    check_quality(golden, ttnn.to_torch(call()))


@pytest.mark.parametrize("preset", PRESETS)
def test_wan_qkv_split(qkv_prepared, preset):
    call, golden = qkv_prepared
    outs = call()
    assert len(outs) == len(golden)
    for g, o in zip(golden, outs):
        check_quality(g, ttnn.to_torch(o))


def test_wan_cross_kv_split(cross_kv_prepared):
    call, golden = cross_kv_prepared
    outs = call()
    assert len(outs) == len(golden)
    for g, o in zip(golden, outs):
        check_quality(g, ttnn.to_torch(o))


def addcmul_case(device, M, K, N, full_gate=False):
    x, w, b = torch_inputs(M, K, N)
    a = torch.randn((M, N), dtype=torch.float32)
    g = torch.randn((M, N) if full_gate else (1, N), dtype=torch.float32)
    tt = dict(x=to_tt(device, x), w=to_tt(device, w), b=to_tt(device, b), a=to_tt(device, a), g=to_tt(device, g))
    cfg = matmul_config(device, M, K, N)
    ckc = compute_config(device, True)

    def call():
        return ttnn.experimental.dit_minimal_matmul_addcmul_fused(
            tt["x"], tt["w"], 1.0, tt["a"], tt["g"], bias_tensor=tt["b"], config=cfg, compute_kernel_config=ckc
        )

    with torch.no_grad():
        golden = torch.addcmul(a, torch_linear(x, w, b), g, value=1.0)
    return call, golden


@pytest.mark.parametrize("preset", PRESETS)
def test_wan_out_addcmul(out_addcmul_prepared, preset):
    call, golden = out_addcmul_prepared
    check_quality(golden, ttnn.to_torch(call()))


def test_wan_out_addcmul_full_gate(out_addcmul_full_gate_prepared):
    call, golden = out_addcmul_full_gate_prepared
    check_quality(golden, ttnn.to_torch(call()))


def perf_n():
    return int(os.environ.get("MMAB_N", "10"))


@pytest.mark.timeout(600)
@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("case", list(LINEAR_CASES))
def test_wan_linear_perf(device, preset, case):
    K, N, approx, gelu = LINEAR_CASES[case]
    M = PRESET_M[preset]
    activation = ttnn.UnaryOpType.GELU_TANH if gelu else None
    call, _ = linear_case(device, M, K, N, approx, activation)
    report(f"{case}-{preset}", profile_ns(device, call, perf_n()))


@pytest.mark.timeout(600)
@pytest.mark.parametrize("preset", PRESETS)
def test_wan_qkv_split_perf(device, preset):
    call, _ = split_case(device, PRESET_M[preset], DIM, QKV_N, 3)
    report(f"qkv_split-{preset}", profile_ns(device, call, perf_n()))


@pytest.mark.timeout(600)
def test_wan_cross_kv_split_perf(device):
    call, _ = split_case(device, CROSS_KV_M, DIM, CROSS_KV_N, 2)
    report("cross_kv_split-2x4", profile_ns(device, call, perf_n()))


@pytest.mark.timeout(600)
@pytest.mark.parametrize("preset", PRESETS)
def test_wan_out_addcmul_perf(device, preset):
    call, _ = addcmul_case(device, PRESET_M[preset], DIM, OUT_N)
    report(f"out_addcmul-{preset}", profile_ns(device, call, perf_n()))
