# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Focused compute-config repro for the row-major fast-tilize -> reduce path.

The op intentionally keeps only:
  row-major reader -> tilize -> row reduce(mean) -> tiled writer

Only the first output column is expected to contain valid values.
"""

import pytest
import torch
import ttnn

from ttnn.operations.toy_rm_row_mean import toy_rm_row_mean


TEST_CASES = [
    pytest.param(
        torch.bfloat16,
        {"fp32_dest_acc_en": False},
        0,
        False,
        id="bf16_fp32_dest_off",
    ),
    pytest.param(
        torch.bfloat16,
        {"fp32_dest_acc_en": True},
        0,
        False,
        marks=pytest.mark.xfail(
            reason="Known debug reproducer: fast tilize -> reduce can corrupt output with fp32 dest accumulation"
        ),
        id="bf16_fp32_dest_on",
    ),
    pytest.param(
        torch.bfloat16,
        {"fp32_dest_acc_en": True},
        100,
        False,
        id="bf16_fp32_dest_on_nops100",
    ),
    pytest.param(
        torch.float32,
        {"fp32_dest_acc_en": True},
        0,
        False,
        id="fp32_input_fp32_dest_on",
    ),
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


def torch_row_mean(x):
    return x.float().mean(dim=-1, keepdim=True)


@pytest.mark.parametrize("torch_dtype, compute_kernel_config, post_tilize_nops, insert_tensix_sync", TEST_CASES)
def test_toy_rm_row_mean_compute_config(
    device, torch_dtype, compute_kernel_config, post_tilize_nops, insert_tensix_sync
):
    torch.manual_seed(42)

    shape = (1, 1, 64, 128)
    x_torch = torch.randn(shape, dtype=torch_dtype)
    expected = torch_row_mean(x_torch)

    x_tt = to_device(x_torch, device)
    result_tt = toy_rm_row_mean(
        x_tt,
        compute_kernel_config=compute_kernel_config,
        post_tilize_nops=post_tilize_nops,
        insert_tensix_sync=insert_tensix_sync,
    )
    result_torch = ttnn.to_torch(result_tt)

    actual = result_torch[..., :1].float()
    max_abs_err = (actual - expected).abs().max().item()
    mean_abs_err = (actual - expected).abs().mean().item()

    print(
        f"\nshape={shape} config={compute_kernel_config} "
        f"dtype={torch_dtype} "
        f"post_tilize_nops={post_tilize_nops} insert_tensix_sync={insert_tensix_sync} "
        f"max_abs_err={max_abs_err:.6f} mean_abs_err={mean_abs_err:.6f}"
    )

    atol = 0.1 if torch_dtype == torch.bfloat16 else 1e-3
    rtol = 0.05 if torch_dtype == torch.bfloat16 else 1e-3

    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), (
        f"Mismatch with dtype={torch_dtype}, config={compute_kernel_config}, post_tilize_nops={post_tilize_nops}, "
        f"insert_tensix_sync={insert_tensix_sync}: "
        f"max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}"
    )


def test_toy_rm_row_mean_fp32_input_on_vs_delayed(device):
    torch.manual_seed(42)

    shape = (1, 1, 64, 128)
    x_torch = torch.randn(shape, dtype=torch.float32)
    x_tt = to_device(x_torch, device)
    result_on = ttnn.to_torch(
        toy_rm_row_mean(
            x_tt,
            compute_kernel_config={"fp32_dest_acc_en": True},
            post_tilize_nops=0,
        )
    )[..., :1].float()

    x_tt = to_device(x_torch, device)
    result_delayed = ttnn.to_torch(
        toy_rm_row_mean(
            x_tt,
            compute_kernel_config={"fp32_dest_acc_en": True},
            post_tilize_nops=100,
        )
    )[..., :1].float()

    diff = (result_on - result_delayed).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()

    print(
        f"\nfp32_input ON_vs_delayed post_tilize_nops=0_vs_100 "
        f"max_abs_err={max_abs_err:.6f} mean_abs_err={mean_abs_err:.6f}"
    )

    assert torch.allclose(result_on, result_delayed, rtol=1e-3, atol=1e-3), (
        f"Float32 input should not show the bf16-style race signature: "
        f"max_abs_err={max_abs_err:.6f}, mean_abs_err={mean_abs_err:.6f}"
    )
