# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""group_norm regression: a caller-supplied input_mask at a non-tile-aligned H*W with dirty padding.

Self-contained: pure ttnn + torch, no weights, checkpoints, network access, or shared fixtures beyond
the standard `device`. Shapes are hardcoded, not imported.

Reproduces the defect fixed by metal #52924. group_norm used to reduce over the implicit tile-padding
rows and back-correct *assuming they held zeros*; ttnn guarantees no such thing -- `reshape` and `slice`
both leave the padded region dirty. So at a non-tile-aligned H*W the group statistics were computed
over garbage.

Complements the coverage in test_group_norm_DRAM.py, which builds its masks internally: this file
exercises the caller-supplied `input_mask` path (with and without `rows_in_last_tile`) on the DRAM grid,
at a large channel count.

Verified on both sides of the fix (see the measured table in test_..._garbage_padding):
  at/after bf68d31a5a2  -> all params pass
  at bf68d31a5a2^       -> garbage_padding fails (max abs err 1.881) and padding_is_ignored fails
                           (delta 1.958), while clean_padding still passes as the control
To reproduce the broken side:
  git checkout bf68d31a5a2^ -- ttnn/cpp/ttnn/operations/normalization/groupnorm/
  rm -rf built && ./build_metal.sh --enable-ccache     # ~13 min; restore with `git checkout HEAD --`
"""

import pytest
import torch
from loguru import logger

import ttnn
from ttnn.operations.normalization import dram_group_norm_virtual_columns

# GroupNorm(32, 1024). H*W=259 = 8*32 + 3, so the final row-tile holds 3 valid rows and 29 padding
# rows -- a small valid remainder, which maximises the damage when the padding leaks into the stats.
N, C, HW, G = 1, 1024, 259, 32
PADDED = -(-HW // 32) * 32  # 288 -> 29 implicit tile-padding rows
EPS = 1e-5
KERNEL_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
)


def _torch_ref(real):
    """group_norm over the HW logical rows only, via NCHW."""
    x = real.view(N, HW, C).permute(0, 2, 1).reshape(N, C, 1, HW).float()
    out = torch.nn.functional.group_norm(x, G, eps=EPS)
    return out.permute(0, 2, 3, 1).reshape(N, 1, HW, C)


def _input_mask(device, cols, rows_in_last_tile):
    """Caller-supplied mask. rows_in_last_tile arrived with #52924, so degrade on older builds:
    the point of this file is to run against bf68d31a5a2^ and show the numbers, and a TypeError
    would hide them. rows_in_last_tile=0 needs no new kwarg and exercises the derive path."""
    if rows_in_last_tile:
        try:
            host = ttnn.create_group_norm_input_mask(C, G, cols, ttnn.bfloat16, rows_in_last_tile=rows_in_last_tile)
        except TypeError:
            pytest.skip("build predates metal #52924: create_group_norm_input_mask has no rows_in_last_tile")
    else:
        host = ttnn.create_group_norm_input_mask(C, G, cols, ttnn.bfloat16)
    return ttnn.to_device(host, device)


def _run(device, real, padding_value, rows_in_last_tile):
    """Run group_norm with padding_value filling the tile-padding rows."""
    grid = ttnn.determine_expected_group_norm_dram_grid_size(
        device=device, num_channels=C, num_groups=G, input_nhw=HW, num_batches=N
    )
    cols = dram_group_norm_virtual_columns(grid, C, G)
    mask = _input_mask(device, cols, rows_in_last_tile)
    gamma, beta = (
        ttnn.from_torch(
            ttnn.create_group_norm_weight_bias_rm(t, C, cols),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        for t in (torch.ones(C), torch.zeros(C))
    )

    # Fill the padding rows, then reinterpret as HW logical rows *preserving* that content.
    # from_torch on an HW-row tensor would zero them -- which is exactly why the pre-existing
    # group_norm suite never caught this.
    buf = torch.zeros((N, 1, PADDED, C), dtype=torch.bfloat16)
    buf[:, :, :HW, :] = real
    buf[:, :, HW:, :] = padding_value
    x = ttnn.from_torch(
        buf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    x = ttnn.reshape(x, ttnn.Shape([N, 1, HW, C]), ttnn.Shape([N, 1, PADDED, C]))

    out = ttnn.group_norm(
        x,
        num_groups=G,
        epsilon=EPS,
        input_mask=mask,
        weight=gamma,
        bias=beta,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=grid,
        inplace=False,
        output_layout=ttnn.TILE_LAYOUT,
        compute_kernel_config=KERNEL_CONFIG,
    )
    return ttnn.to_torch(ttnn.from_device(out)).float()[:, :, :HW, :]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
# rows_in_last_tile is what a caller supplying its own mask must set; 0 makes group_norm derive the
# second mask set itself (costing a host build + upload per call). Both must give the same correct
# answer -- i.e. passing the kwarg is a performance choice, with no numerical effect.
@pytest.mark.parametrize("rows_in_last_tile", [HW % 32, 0], ids=["mask_with_rows", "mask_derived"])
@pytest.mark.parametrize("padding_value", [0.0, 7.0], ids=["clean_padding", "garbage_padding"])
def test_group_norm_caller_mask_garbage_padding(device, rows_in_last_tile, padding_value):
    """Dirty tile-padding rows must not move group_norm away from the torch reference."""
    torch.manual_seed(0)
    real = torch.rand((N, 1, HW, C), dtype=torch.bfloat16)
    ref = _torch_ref(real)
    out = _run(device, real, padding_value, rows_in_last_tile)

    max_abs_err = (out - ref).abs().max().item()
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    logger.info(
        f"H*W={HW} padding={padding_value} rows_in_last_tile={rows_in_last_tile}: "
        f"max_abs_err={max_abs_err} pcc={pcc}"
    )
    # Gate on max abs error, NOT PCC. The damage is confined to the final row-tile (29 of 259 rows),
    # so a global statistic barely registers it: measured on this exact case at bf68d31a5a2^, PCC is
    # 0.9999310 broken vs 0.9999919 fixed -- a PCC gate at 0.9999 passes the broken kernel. max abs
    # error separates by 22x. Measured, same seed and shape:
    #   fixed      padding 0.0 / 7.0 / 100.0 -> 0.0845827 (identical to the last digit)
    #   bf68d31a5a2^ padding 0.0 -> 0.0923952 (zeros satisfy the old back-correction)
    #   bf68d31a5a2^ padding 7.0 -> 1.8811113  <-- the defect
    # Tile-aligned controls at the same C/G: 0.0260 at H*W=256, 0.0477 at H*W=288.
    assert max_abs_err < 0.15, (
        f"max abs error {max_abs_err} at padding={padding_value} (pcc {pcc}); group_norm must exclude "
        f"the tile-padding rows from both accumulation passes (see #52685 / metal #52924)"
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("rows_in_last_tile", [HW % 32, 0], ids=["mask_with_rows", "mask_derived"])
def test_group_norm_caller_mask_padding_is_ignored(device, rows_in_last_tile):
    """Same logical rows with different padding bytes must be bit-identical, not merely close."""
    torch.manual_seed(0)
    real = torch.rand((N, 1, HW, C), dtype=torch.bfloat16)
    baseline = _run(device, real, 0.0, rows_in_last_tile)
    for padding_value in (7.0, -3.5, 100.0):
        other = _run(device, real, padding_value, rows_in_last_tile)
        assert torch.equal(baseline, other), (
            f"group_norm output changed when tile padding went 0.0 -> {padding_value} (max delta "
            f"{(baseline - other).abs().max().item()}); the padding rows must not reach either "
            f"accumulation pass (see #52685 / metal #52924)"
        )
