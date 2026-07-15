# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# mm_help profiling harness — isolated single SDPA prefill (causal) for the main-vs-mm_help5
# A/B. Underscore-prefixed so ambient `pytest tests/...` does NOT collect it; run EXPLICITLY
# under Tracy on each branch-build:
#
#   SD_B=1 SD_NH=10 SD_NKV=10 SD_S=9472 SD_D=128 SD_QCHUNK=256 SD_KCHUNK=256 \
#   python -m tracy -r -p -v -m pytest tests/ttnn/unit_tests/operations/sdpa/_sdpa_mmhelp_bench.py
#
# Calls ttnn.transformer.scaled_dot_product_attention(is_causal=True) twice (cold + warm)
# so the extractor takes the warm ScaledDotProductAttentionDeviceOperation. Causal PCC vs
# torch SDPA is checked every run. Why-fields come from the Tracy CSV ATTRIBUTES column.
#
# Env: SD_B, SD_NH, SD_NKV, SD_S, SD_D, SD_QCHUNK, SD_KCHUNK, grid SD_GX/SD_GY (0->full),
#      SD_DTYPE (bfloat16), SD_FID (HiFi2), SD_FP32 (false), SD_EXP_APPROX (false).
import os
import pytest
import torch
import ttnn
from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

_DT = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b, "float32": ttnn.float32}
_FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2, "HiFi4": ttnn.MathFidelity.HiFi4}


def _e(name, default, cast):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


def _bool(v):
    return str(v).lower() in ("1", "true", "yes")


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


CFG = dict(
    b=_e("SD_B", 1, int), nh=_e("SD_NH", 10, int), nkv=_e("SD_NKV", 10, int),
    s=_e("SD_S", 9472, int), d=_e("SD_D", 128, int),
    q_chunk=_e("SD_QCHUNK", 256, int), k_chunk=_e("SD_KCHUNK", 256, int),
    gx=_e("SD_GX", 0, int), gy=_e("SD_GY", 0, int),
    dtype=_e("SD_DTYPE", ttnn.bfloat16, lambda v: _DT[v]),
    fidelity=_e("SD_FID", ttnn.MathFidelity.HiFi2, lambda v: _FID[v]),
    fp32=_e("SD_FP32", False, _bool),
    exp_approx=_e("SD_EXP_APPROX", False, _bool),
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_sdpa_prefill_mmhelp_bench(device):
    c = CFG
    torch.manual_seed(1234)
    b, nh, nkv, s, d = c["b"], c["nh"], c["nkv"], c["s"], c["d"]
    assert nh % nkv == 0, "nkv must divide nh"
    dg = device.compute_with_storage_grid_size()
    gx = c["gx"] if c["gx"] else dg.x
    gy = c["gy"] if c["gy"] else dg.y
    logger.info(f"sdpa_prefill b={b} nh={nh} nkv={nkv} s={s} d={d} q={c['q_chunk']} k={c['k_chunk']} grid={gx}x{gy}")

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        q_chunk_size=c["q_chunk"],
        k_chunk_size=c["k_chunk"],
        exp_approx_mode=c["exp_approx"],
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=c["fidelity"],
        math_approx_mode=False,
        fp32_dest_acc_en=c["fp32"],
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)
    mc = ttnn.DRAM_MEMORY_CONFIG
    tt_Q = ttnn.from_torch(Q, dtype=c["dtype"], layout=ttnn.TILE_LAYOUT, memory_config=mc, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=c["dtype"], layout=ttnn.TILE_LAYOUT, memory_config=mc, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=c["dtype"], layout=ttnn.TILE_LAYOUT, memory_config=mc, device=device, pad_value=0.0)

    tt_back = None
    for _ in range(2):  # cold + warm; extractor takes the warm op
        tt_back = ttnn.transformer.scaled_dot_product_attention(
            tt_Q, tt_K, tt_V, is_causal=True,
            program_config=program_config, compute_kernel_config=compute_kernel_config,
        )

    out = ttnn.to_torch(tt_back)[:, :, :s, :]
    K_rep = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    V_rep = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K_rep, V_rep, is_causal=True)
    out_pass, out_pcc = comp_pcc(gt, out, 0.994)
    logger.info(f"sdpa_prefill torch vs ttnn: {out_pcc}")
    assert out_pass
