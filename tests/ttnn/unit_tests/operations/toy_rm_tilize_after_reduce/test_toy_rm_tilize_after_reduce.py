# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Probe whether REDUCE_ROW corrupts the preserved tilized input tiles.

The op performs tilize -> reduce, discards the reduce result, and writes the
tilized tiles after reduce has run. If this output is corrupted, the handoff is
damaging the tilize result in L1. If it stays correct while the mean result is
bad, the reduce path itself is the stronger suspect.
"""

import pytest
import torch
import ttnn

from ttnn.operations.toy_rm_tilize_after_reduce import toy_rm_tilize_after_reduce


TEST_CASES = [
    pytest.param(torch.bfloat16, {"fp32_dest_acc_en": False}, 0, False, id="bf16_fp32_dest_off"),
    pytest.param(
        torch.bfloat16,
        {"fp32_dest_acc_en": True},
        0,
        False,
        marks=pytest.mark.xfail(
            reason="Expected repro: if reduce reprogramming corrupts preserved tilized tiles, snapshot mismatches input"
        ),
        id="bf16_fp32_dest_on",
    ),
    pytest.param(torch.bfloat16, {"fp32_dest_acc_en": True}, 400, False, id="bf16_fp32_dest_on_nops400"),
    pytest.param(torch.float32, {"fp32_dest_acc_en": True}, 0, False, id="fp32_input_fp32_dest_on"),
]


def to_device(tensor_torch, device):
    ttnn_dtype = {
        torch.bfloat16: ttnn.bfloat16,
        torch.float32: ttnn.float32,
    }[tensor_torch.dtype]
    return ttnn.from_torch(
        tensor_torch,
        dtype=ttnn_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


@pytest.mark.parametrize("torch_dtype, compute_kernel_config, post_tilize_nops, insert_tensix_sync", TEST_CASES)
def test_toy_rm_tilize_after_reduce(device, torch_dtype, compute_kernel_config, post_tilize_nops, insert_tensix_sync):
    torch.manual_seed(42)

    shape = (1, 1, 64, 128)
    x_torch = torch.randn(shape, dtype=torch_dtype)

    x_tt = to_device(x_torch, device)
    result_tt = toy_rm_tilize_after_reduce(
        x_tt,
        compute_kernel_config=compute_kernel_config,
        post_tilize_nops=post_tilize_nops,
        insert_tensix_sync=insert_tensix_sync,
    )
    result_torch = ttnn.to_torch(result_tt)

    diff = (result_torch.float() - x_torch.float()).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()

    print(
        f"\nshape={shape} config={compute_kernel_config} "
        f"dtype={torch_dtype} "
        f"post_tilize_nops={post_tilize_nops} insert_tensix_sync={insert_tensix_sync} "
        f"max_abs_err={max_abs_err:.6f} mean_abs_err={mean_abs_err:.6f}"
    )

    atol = 0.01 if torch_dtype == torch.bfloat16 else 0.002
    rtol = 0.01 if torch_dtype == torch.bfloat16 else 0.001

    assert torch.allclose(result_torch.float(), x_torch.float(), rtol=rtol, atol=atol), (
        f"Tilized snapshot mismatch with dtype={torch_dtype}, config={compute_kernel_config}, "
        f"post_tilize_nops={post_tilize_nops}, insert_tensix_sync={insert_tensix_sync}: "
        f"max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}"
    )


def test_toy_rm_tilize_after_reduce_fp32_input_on_vs_delayed(device):
    torch.manual_seed(42)

    shape = (1, 1, 64, 128)
    x_torch = torch.randn(shape, dtype=torch.float32)
    x_tt = to_device(x_torch, device)
    result_on = ttnn.to_torch(
        toy_rm_tilize_after_reduce(
            x_tt,
            compute_kernel_config={"fp32_dest_acc_en": True},
            post_tilize_nops=0,
        )
    ).float()

    x_tt = to_device(x_torch, device)
    result_delayed = ttnn.to_torch(
        toy_rm_tilize_after_reduce(
            x_tt,
            compute_kernel_config={"fp32_dest_acc_en": True},
            post_tilize_nops=400,
        )
    ).float()

    diff = (result_on - result_delayed).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()

    print(
        f"\nfp32_input snapshot ON_vs_delayed post_tilize_nops=0_vs_400 "
        f"max_abs_err={max_abs_err:.6f} mean_abs_err={mean_abs_err:.6f}"
    )

    assert torch.allclose(result_on, result_delayed, rtol=1e-3, atol=0.002), (
        f"Float32 tilized snapshot should not show the bf16-style race signature: "
        f"max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}"
    )
