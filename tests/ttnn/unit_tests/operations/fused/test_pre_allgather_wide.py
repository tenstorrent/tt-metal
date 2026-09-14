# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Wide-input coverage for {rms,layer}_norm_pre_all_gather (#53094).

Widths past the L1 budget make the op stream each row in chunks through a CB-backed running
partial instead of holding full-width buffers; these tests cross that threshold (which used to
TT_THROW on Blackhole at Wt=128 with fp32_dest_acc_en) and check the stats against torch.

Max relative error is asserted rather than PCC: the sum(x^2) stat is near-constant across rows
(mean ~W, spread ~sqrt(2W)), so output-dtype quantization alone drags PCC below any tight
threshold while the values are correctly rounded. The fp32-stats variants remove the output
rounding entirely and act as a canary for the running partial's precision across chunks.
"""

import pytest
import torch
import ttnn

TILE = 32


def golden_stats(x, is_rmsnorm):
    stats = [x.pow(2).sum(-1, keepdim=True)]
    if not is_rmsnorm:
        stats.append(x.sum(-1, keepdim=True))
    return stats


def run_pre_all_gather(device, is_rmsnorm, w, h, input_dtype, stats_dtype, fp32_acc, has_residual):
    torch.manual_seed(1234)
    shape = (1, 1, h, w)
    torch_x = torch.randn(shape)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_acc,
        packer_l1_acc=False,
    )

    x = ttnn.from_torch(torch_x, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    op = ttnn.rms_norm_pre_all_gather if is_rmsnorm else ttnn.layer_norm_pre_all_gather
    kwargs = {"compute_kernel_config": compute_kernel_config, "dtype": stats_dtype}

    # Round each operand through the on-device dtype before summing: the fused pre-add computes
    # bf16(x) + bf16(res) exactly in fp32, not bf16(x + res).
    def round_in(t):
        return t.bfloat16().float() if input_dtype == ttnn.bfloat16 else t

    torch_golden_input = round_in(torch_x)
    if has_residual:
        torch_res = torch.randn(shape)
        kwargs["residual_input_tensor"] = ttnn.from_torch(
            torch_res, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        torch_golden_input = torch_golden_input + round_in(torch_res)

    # Two iterations to exercise the program cache.
    for _ in range(2):
        tt_stats = op(x, **kwargs)
        stats = ttnn.to_torch(tt_stats)
        tt_stats.deallocate(True)

    # Tolerances calibrated against the pre-change kernel (bit-identical where both run):
    # sum(x^2) is well-conditioned — one output ulp plus FPU operand truncation; sum(x) is
    # cancellation-dominated (near-zero sums against O(1) terms), so it gets a looser bound
    # relative to a clamped denominator. The 16-bit running total (fp32_acc off) drifts ~3%.
    for i, golden in enumerate(golden_stats(torch_golden_input, is_rmsnorm)):
        if not fp32_acc:
            if i > 0:
                continue  # near-cancelling sum(x) is meaningless under the 16-bit running total
            tol = 6e-2
        elif i == 0:
            tol = 8e-3 if stats_dtype == ttnn.bfloat16 else 2e-3
        else:
            tol = 2e-2
        actual = stats[..., i * TILE : i * TILE + 1]
        rel_err = ((actual - golden).abs() / golden.abs().clamp(min=1.0)).max().item()
        assert rel_err <= tol, f"stat {i}: max rel err {rel_err} > {tol}"


@pytest.mark.parametrize("is_rmsnorm", [True, False], ids=["rmsnorm", "layernorm"])
@pytest.mark.parametrize(
    "w",
    [1024, 4096, 4128],
    ids=["w1024_full_width", "w4096_chunked", "w4128_uneven_last_chunk"],
)
@pytest.mark.parametrize("h", [32, 128], ids=["h32", "h128"])
@pytest.mark.parametrize("stats_dtype", [ttnn.bfloat16, ttnn.float32], ids=["stats_bf16", "stats_fp32"])
@pytest.mark.parametrize("has_residual", [False, True], ids=["no_residual", "residual"])
def test_pre_all_gather_wide_bf16(device, is_rmsnorm, w, h, stats_dtype, has_residual):
    run_pre_all_gather(device, is_rmsnorm, w, h, ttnn.bfloat16, stats_dtype, fp32_acc=True, has_residual=has_residual)


@pytest.mark.parametrize("is_rmsnorm", [True, False], ids=["rmsnorm", "layernorm"])
@pytest.mark.parametrize("w", [1024, 4096], ids=["w1024_full_width", "w4096_chunked"])
def test_pre_all_gather_wide_fp32(device, is_rmsnorm, w):
    run_pre_all_gather(device, is_rmsnorm, w, 32, ttnn.float32, ttnn.bfloat16, fp32_acc=True, has_residual=False)


@pytest.mark.parametrize("is_rmsnorm", [True, False], ids=["rmsnorm", "layernorm"])
def test_pre_all_gather_no_fp32_acc(device, is_rmsnorm):
    # bf16 accumulation: narrow width only — wide rows lose precision in the 16-bit running
    # total regardless of chunking (which bf16 buffers don't trigger at these widths anyway).
    run_pre_all_gather(device, is_rmsnorm, 1024, 32, ttnn.bfloat16, ttnn.bfloat16, fp32_acc=False, has_residual=False)
