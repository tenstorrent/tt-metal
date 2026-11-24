# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# The minimal_matmul calls of the Wan2.2 pipeline that test_minimal_matmul_llk2.py does not cover, on the LLK 2.0 kernel
# (TT_MINIMAL_MATMUL_LLK2=1), configured the way Wan configures them (models/tt_dit: get_matmul_config, its compute
# configs, dtypes): QKV split, cross-attention to_kv split, proj_out (Float32 in and out), patch embedding (one K block),
# and the text embedder (gelu_tanh). Fewer rows than the model's 16384 per device: the blocking and the kernel's paths are
# the same, the run is shorter.

import pytest
import torch
import ttnn

from models.tt_dit.utils.matmul import get_matmul_config
from tests.ttnn.nightly.unit_tests.operations.experimental.test_minimal_matmul import assert_quality


@pytest.fixture(autouse=True)
def llk2_minimal_matmul(monkeypatch):
    monkeypatch.setenv("TT_MINIMAL_MATMUL_LLK2", "1")


def compute_config(device, fidelity, l1_acc=True):
    return ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=l1_acc
    )


def to_tt(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, device=device, layout=ttnn.TILE_LAYOUT)


CASES = {
    # name: (M, K, N, chunks, activation, in0 dtype, out dtype, fidelity, packer_l1_acc, use Wan's table config)
    "qkv_split": (2048, 5120, 7680, 3, None, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True, True),
    "cross_to_kv_split": (512, 5120, 5120, 2, None, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True, True),
    "proj_out_fp32": (2048, 5120, 64, 1, None, ttnn.float32, ttnn.float32, ttnn.MathFidelity.HiFi4, True, True),
    "patch_embed": (2048, 64, 5120, 1, None, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi4, False, False),
    "text_embed_gelu": (
        512,
        4096,
        5120,
        1,
        "gelu_tanh",
        ttnn.bfloat16,
        ttnn.bfloat16,
        ttnn.MathFidelity.HiFi2,
        True,
        True,
    ),
    "text_embed_2": (512, 5120, 5120, 1, None, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True, True),
    # The T5 encoder's linears have no bias: the switch leaves them on the 1.0 kernel (the LLK 2.0 one needs a bias).
    "t5_no_bias": (512, 4096, 4096, 1, None, ttnn.bfloat16, ttnn.bfloat16, ttnn.MathFidelity.HiFi2, True, True),
}


@pytest.mark.parametrize("name", list(CASES.keys()))
def test_llk2_wan_variant(device, name):
    M, K, N, chunks, act, in0_dtype, out_dtype, fidelity, l1_acc, table = CASES[name]
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.float32)
    w = torch.randn((K, N), dtype=torch.float32) / K**0.5
    b = None if name == "t5_no_bias" else torch.randn((1, N), dtype=torch.float32)
    expected = x @ w + (b if b is not None else 0)
    if act == "gelu_tanh":
        expected = torch.nn.functional.gelu(expected, approximate="tanh")

    grid = device.compute_with_storage_grid_size()
    config = get_matmul_config(M, K, N, grid) if table else None
    common = dict(
        bias_tensor=to_tt(b, device, ttnn.bfloat16) if b is not None else None,
        compute_kernel_config=compute_config(device, fidelity, l1_acc),
        config=config,
        dtype=out_dtype,
    )
    tt_x, tt_w = to_tt(x, device, in0_dtype), to_tt(w, device, ttnn.bfloat16)
    if chunks > 1:
        outs = ttnn.experimental.minimal_matmul_split(tt_x, tt_w, chunks=chunks, dim=-1, **common)
        got = torch.cat([ttnn.to_torch(o).float() for o in outs], dim=-1)
    else:
        out = ttnn.experimental.minimal_matmul(
            input_tensor=tt_x,
            weight_tensor=tt_w,
            fused_activation=ttnn.UnaryOpType.GELU_TANH if act == "gelu_tanh" else None,
            **common,
        )
        got = ttnn.to_torch(out).float()
    check = assert_quality(expected, got)
    assert check["pcc"] > 0.9995, check
    assert check["relative_rmse"] < 0.02, check
