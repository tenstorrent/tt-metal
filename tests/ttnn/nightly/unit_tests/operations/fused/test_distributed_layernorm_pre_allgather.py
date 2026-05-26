# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from models.common.utility_functions import tt2torch_tensor, torch2tt_tensor

import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal, comp_allclose_and_pcc
from tests.ttnn.utils_for_testing import assert_equal, assert_numeric_metrics, tt_dtype_to_torch_dtype

TEST_PADDING_VALUE = -42


def reference(x, n_devices, is_rmsnorm):
    S = x[0].shape[2]
    B = x[0].shape[0]
    sumxs = []
    sumx2s = []
    # Distributed processing
    for chunk in x:
        sumx_local = torch.sum(chunk, dim=-1, keepdim=True)
        sumx2_local = torch.sum(torch.square(chunk), dim=-1, keepdim=True)
        sumxs.append(sumx_local)
        sumx2s.append(sumx2_local)

    # pad with zeros as for tiles
    output = []
    for i in range(n_devices):
        if is_rmsnorm:
            output.append(torch.concat([sumx2s[i], torch.zeros([B, 1, S, 31])], dim=-1))
        else:
            output.append(
                torch.concat([sumx2s[i], torch.zeros([B, 1, S, 31]), sumxs[i], torch.zeros([B, 1, S, 31])], dim=-1)
            )

    return output


def referencefp32(x, n_devices, is_rmsnorm):
    S = x[0].shape[2]
    B = x[0].shape[0]
    sumxs = []
    sumx2s = []
    # Distributed processing
    for chunk in x:
        chunk = chunk.to(torch.float32)
        count_local = chunk.shape[-1]
        sumx_local = torch.sum(chunk, dim=-1, keepdim=True)
        sumx2_local = torch.sum(torch.square(chunk), dim=-1, keepdim=True)

        sumxs.append(sumx_local)
        sumx2s.append(sumx2_local)

    # pad with zeros as for tiles
    output = []
    for i in range(n_devices):
        if is_rmsnorm:
            output.append(torch.concat([sumx2s[i], torch.zeros([B, 1, S, 31])], dim=-1))
        else:
            output.append(
                torch.concat([sumx2s[i], torch.zeros([B, 1, S, 31]), sumxs[i], torch.zeros([B, 1, S, 31])], dim=-1)
            )

    return output


def ln_pre_allgather_op(xs, n_devices, is_rmsnorm, out_dtpe, kernel_config):
    tt_out = []
    for d in range(n_devices):
        if is_rmsnorm:
            tt_out.append(ttnn.rms_norm_pre_all_gather(xs[d], compute_kernel_config=kernel_config, dtype=out_dtpe))
        else:
            tt_out.append(ttnn.layer_norm_pre_all_gather(xs[d], compute_kernel_config=kernel_config, dtype=out_dtpe))
    return tt_out


def run_pre_all_gather_residual_pcc(device, inp_shape, op_name):
    """Stats from {layer,rms}_norm_pre_all_gather with residual_input_tensor (input + residual).

    BFLOAT16, HiFi4, fp32_dest_acc on pre.
    Golden stats match torch reductions on the fused tensor input + residual.
    """
    torch.manual_seed(41467)
    is_rmsnorm = op_name == "rms_norm"

    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    torch_inp = torch.randn(inp_shape, dtype=torch.bfloat16)
    torch_res = torch.randn(inp_shape, dtype=torch.bfloat16)
    combined = torch_inp + torch_res

    out_torch = reference(combined.chunk(1, dim=-1), 1, is_rmsnorm)[0]

    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    tt_inp = torch2tt_tensor(
        torch_inp,
        tt_dtype=ttnn.bfloat16,
        tt_device=device,
        tt_layout=ttnn.TILE_LAYOUT,
        tt_memory_config=dram_memcfg,
    )
    tt_inp = ttnn.fill_implicit_tile_padding(tt_inp, TEST_PADDING_VALUE)
    tt_res = torch2tt_tensor(
        torch_res,
        tt_dtype=ttnn.bfloat16,
        tt_device=device,
        tt_layout=ttnn.TILE_LAYOUT,
        tt_memory_config=dram_memcfg,
    )

    pre_op = ttnn.rms_norm_pre_all_gather if is_rmsnorm else ttnn.layer_norm_pre_all_gather
    tt_stats = pre_op(
        tt_inp,
        residual_input_tensor=tt_res,
        dtype=ttnn.bfloat16,
        compute_kernel_config=kernel_config,
        memory_config=dram_memcfg,
    )

    tt_output_host = tt2torch_tensor(tt_stats)
    failures = []
    n_devices = 1
    reduction_width = inp_shape[-1] // n_devices

    def _log_errors(label, passing, output_str):
        logger.debug(f"tt vs torch {label} = {output_str}")
        if not passing:
            failures.append(f"{label}: {output_str}")

    device_offset = 0
    # sum(x^2) lives in column 0 of the first stats tile (same layout as run_layernorm_part_1)
    _log_errors(
        "sum(x^2) residual path",
        *comp_allclose_and_pcc(
            out_torch[:, :, :, 0 + device_offset],
            tt_output_host[:, :, :, 0 + device_offset],
            rtol=1e-02 * reduction_width,
            atol=0,
            pcc=0.999,
        ),
    )
    _log_errors(
        "padding 1 residual path",
        *comp_equal(
            out_torch[:, :, :, 1 + device_offset : 32 + device_offset],
            tt_output_host[:, :, :, 1 + device_offset : 32 + device_offset],
        ),
    )

    # rmsnorm output is one tile wide; layernorm has a second tile holding sum(x).
    if not is_rmsnorm:
        # sum(x) lives in column 0 of the second stats tile (offset 32)
        _log_errors(
            "sum(x) residual path",
            *comp_allclose_and_pcc(
                out_torch[:, :, :, 32 + device_offset],
                tt_output_host[:, :, :, 32 + device_offset],
                rtol=1e-02 * reduction_width,
                atol=0,
                pcc=0.9999,
            ),
        )
        _log_errors(
            "padding 2 residual path",
            *comp_equal(
                out_torch[:, :, :, 33 + device_offset : 64 + device_offset],
                tt_output_host[:, :, :, 33 + device_offset : 64 + device_offset],
            ),
        )

    assert not failures, "tt vs torch comparison(s) failed:\n  " + "\n  ".join(failures)


def run_layernorm_pre_post_gamma_only_pcc(device, use_pre_all_gather: bool):
    """End-to-end LayerNorm (pre_all_gather -> post_all_gather) with weight only, no bias.

    Regression for the former TT_FATAL that required beta whenever layernorm used gamma
    (see layernorm_post_all_gather_device_operation). Shape (1, 1, 37, 72)
    """
    torch.manual_seed(42)

    inp_shape = (1, 1, 37, 72)
    hidden_size = inp_shape[-1]
    epsilon = 1e-5

    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    x = torch.randn(inp_shape, dtype=torch.bfloat16)
    gamma_torch = torch.rand(1, 1, 1, hidden_size, dtype=torch.bfloat16) * 2 - 1

    ref_out = torch.nn.functional.layer_norm(
        x.float(),
        (hidden_size,),
        gamma_torch.reshape(hidden_size).float(),
        bias=None,
        eps=epsilon,
    ).to(torch.bfloat16)

    pre_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    post_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_inp = torch2tt_tensor(
        x,
        tt_dtype=ttnn.bfloat16,
        tt_device=device,
        tt_layout=ttnn.TILE_LAYOUT,
        tt_memory_config=dram_memcfg,
    )
    tt_inp = ttnn.fill_implicit_tile_padding(tt_inp, TEST_PADDING_VALUE)

    if use_pre_all_gather:
        tt_stats = ln_pre_allgather_op([tt_inp], 1, False, ttnn.bfloat16, pre_kernel_config)[0]
    else:
        stats_torch = reference(x.chunk(1, dim=-1), 1, False)[0]
        tt_stats = torch2tt_tensor(
            stats_torch,
            tt_dtype=ttnn.bfloat16,
            tt_device=device,
            tt_layout=ttnn.TILE_LAYOUT,
            tt_memory_config=dram_memcfg,
        )

    tt_gamma = torch2tt_tensor(
        gamma_torch,
        tt_dtype=ttnn.bfloat16,
        tt_device=device,
        tt_layout=ttnn.TILE_LAYOUT,
        tt_memory_config=dram_memcfg,
    )

    tt_out = ttnn.layer_norm_post_all_gather(
        tt_inp,
        tt_stats,
        epsilon=epsilon,
        weight=tt_gamma,
        compute_kernel_config=post_kernel_config,
        dtype=ttnn.bfloat16,
        memory_config=dram_memcfg,
    )

    tt_out_host = tt2torch_tensor(tt_out)[..., :hidden_size]

    passing, output_str = comp_allclose_and_pcc(
        ref_out,
        tt_out_host,
        rtol=0.5,
        atol=25,
        pcc=0.99,
    )
    logger.debug(f"layernorm pre+post (gamma only) vs torch = {output_str}")
    assert passing, output_str


def run_layernorm_part_1(inp_shape, n_devices, is_rmsnorm, input_dtype, output_dtype, device):
    # Set print options
    torch.set_printoptions(threshold=100)

    torch.manual_seed(42)

    if input_dtype == ttnn.float32:
        canon_inp = torch.randn(inp_shape)
    else:
        canon_inp = torch.randn(inp_shape).bfloat16()

    # Get per-chunk inputs
    inp_chunked = canon_inp.chunk(n_devices, dim=-1)

    # Reference
    out_torch = reference(inp_chunked, n_devices, is_rmsnorm)
    out_torch = torch.concat(out_torch, -1)

    # out_torchfp32 = referencefp32(inp_chunked, n_devices, is_rmsnorm)
    # out_torchfp32 = torch.concat(out_torchfp32, -1)

    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    tt_inp = []
    for d in range(n_devices):
        tt_inp.append(
            torch2tt_tensor(
                inp_chunked[d],
                tt_dtype=input_dtype,
                tt_device=device,
                tt_layout=ttnn.TILE_LAYOUT,
                tt_memory_config=dram_memcfg,
            )
        )

    # LN pre all gather OP
    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,  # Highest fidelity
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    tt_out = ln_pre_allgather_op(tt_inp, n_devices, is_rmsnorm, output_dtype, kernel_config)
    tt_output_host = torch.concat([tt2torch_tensor(tt_o) for tt_o in tt_out], -1)

    all_passing = True

    reduction_width = inp_shape[-1] // n_devices

    for i in range(n_devices):
        device_offset = i * 32 if is_rmsnorm else i * 64
        # Compare sum(xˆ2)

        # print("out_torch sum(x^2):")
        # print(out_torch[0, 0, :64, 0 + device_offset])

        # print("tt_output_host sum(x^2):")
        # print(tt_output_host[0, 0, :64, 0 + device_offset])

        passing, output_str = comp_allclose_and_pcc(
            out_torch[:, :, :, 0 + device_offset],
            tt_output_host[:, :, :, 0 + device_offset],
            rtol=1e-01 * reduction_width,  # Issue 9908: large error in reduce, set new rtol target when fixed!
            atol=0,
            pcc=0.9,
        )
        logger.debug(f"tt vs torch sum(xˆ2) = {output_str}")

        all_passing &= passing

        # Check if zeros are same
        passing, output_str = comp_equal(
            out_torch[:, :, :, 1 + device_offset : 32 + device_offset],
            tt_output_host[:, :, :, 1 + device_offset : 32 + device_offset],
        )
        logger.debug(f"tt vs torch padding 1 = {output_str}")

        all_passing &= passing

        if not is_rmsnorm:
            # Compare sum(x)
            # print("out_torch sum(x):")
            # print(out_torch[0, 0, :64, 32 + device_offset])

            # print("tt_output_host sum(x):")
            # print(tt_output_host[0, 0, :64, 32 + device_offset])

            passing, output_str = comp_allclose_and_pcc(
                out_torch[:, :, :, 32 + device_offset],
                tt_output_host[:, :, :, 32 + device_offset],
                rtol=5e-01
                * reduction_width,  # Issue 9908: large error in reduce, set new rtol and atol target when fixed!
                atol=10,
                pcc=0.97,
            )
            logger.debug(f"tt vs torch sum(x) = {output_str}")

            all_passing &= passing

            # Check if zeros are same
            passing, output_str = comp_equal(
                out_torch[:, :, :, 33 + device_offset : 64 + device_offset],
                tt_output_host[:, :, :, 33 + device_offset : 64 + device_offset],
            )
            logger.debug(f"tt vs torch padding 2 = {output_str}")

            all_passing &= passing

    assert all_passing


@pytest.mark.parametrize(
    "input_dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "output_dtype",
    (ttnn.bfloat16, ttnn.bfloat8_b),
    ids=["BFLOAT16", "BFLOAT8_B"],
)
@pytest.mark.parametrize(
    "inp_shape",
    [
        (1, 1, 32, 8192),
        (1, 1, 128, 8192),
        (1, 1, 2048, 8192),
        (1, 1, 8192, 8192),
        (2, 1, 128, 8192),
        (1, 1, 128, 2048),
    ],
)
@pytest.mark.parametrize(
    "n_devices",
    [4, 8],
)
@pytest.mark.parametrize(
    "is_rmsnorm",
    [True, False],
    ids=["rmsnorm", "layernorm"],
)
def test_layernorm_part_1_with_program_cache(inp_shape, n_devices, is_rmsnorm, input_dtype, output_dtype, device):
    run_layernorm_part_1(inp_shape, n_devices, is_rmsnorm, input_dtype, output_dtype, device)


@pytest.mark.parametrize(
    "input_dtype",
    [ttnn.bfloat16],
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "output_dtype",
    [ttnn.bfloat16],
    ids=["BFLOAT16"],
)
@pytest.mark.parametrize(
    "inp_shape",
    [
        (1, 1, 2048, 8192),
    ],
)
@pytest.mark.parametrize(
    "n_devices",
    [8],
)
@pytest.mark.parametrize(
    "is_rmsnorm",
    [True, False],
    ids=["rmsnorm", "layernorm"],
)
def test_layernorm_part_1_with_program_cache2(inp_shape, n_devices, is_rmsnorm, input_dtype, output_dtype, device):
    dummy_tensors = []

    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    for i in range(2):
        if i > 0:
            dummy_tensors.append(
                torch2tt_tensor(
                    torch.randn(inp_shape),
                    tt_dtype=input_dtype,
                    tt_device=device,
                    tt_layout=ttnn.TILE_LAYOUT,
                    tt_memory_config=dram_memcfg,
                )
            )
        run_layernorm_part_1(inp_shape, n_devices, is_rmsnorm, input_dtype, output_dtype, device)

    assert device.num_program_cache_entries() == 1, "Program cache should have only one entry" + str(
        device.num_program_cache_entries()
    )


@pytest.mark.parametrize(
    "use_pre_all_gather",
    [True, False],
    ids=["via_pre_allgather", "stats_manual"],
)
def test_layernorm_pre_post_gamma_only_pcc(use_pre_all_gather, device):
    """layer_norm_post_all_gather with gamma and no bias; PCC vs torch reference."""
    run_layernorm_pre_post_gamma_only_pcc(device, use_pre_all_gather)


@pytest.mark.parametrize(
    "inp_shape",
    [(1, 1, 32, 128), (1, 1, 24, 42), (1, 1, 24, 38)],
)
@pytest.mark.parametrize("op_name", ["layer_norm", "rms_norm"])
def test_layernorm_pre_all_gather_residual_pcc(device, op_name, inp_shape):
    """{layer,rms}_norm_pre_all_gather with residual_input_tensor; PCC vs torch reference."""
    run_pre_all_gather_residual_pcc(device, inp_shape, op_name)


@pytest.mark.parametrize(
    "inp_shape",
    [(1, 1, 32, 42)],  # W=42 → padded W=64, 22 cols of implicit tile padding per row
)
def test_layernorm_pre_all_gather_residual_padding_zeroed(device, inp_shape):
    """Both input and residual implicit tile padding must not contaminate the layernorm stats.

    The kernel reads input and residual tile-by-tile and adds them inside the tile,
    including the implicit padded columns past the logical width. A correct op must
    zero both tensors' implicit padding. We "poison" both tensors
    with a large constant; if either zeroing is missing, the per-row sum(x^2) will
    be off by ~22 * poison^2, which a tight tolerance will catch.
    """
    torch.manual_seed(0)

    POISON = 100.0  # large enough that 22 * POISON^2 dominates any correct stat value
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    torch_inp = torch.randn(inp_shape, dtype=torch.bfloat16)
    torch_res = torch.randn(inp_shape, dtype=torch.bfloat16)
    combined = torch_inp + torch_res

    out_torch = reference(combined.chunk(1, dim=-1), 1, False)[0]

    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    tt_inp = torch2tt_tensor(
        torch_inp, tt_dtype=ttnn.bfloat16, tt_device=device, tt_layout=ttnn.TILE_LAYOUT, tt_memory_config=dram_memcfg
    )
    # Poison input's implicit tile padding. A correct op must ignore these values.
    tt_inp = ttnn.fill_implicit_tile_padding(tt_inp, POISON)
    tt_res = torch2tt_tensor(
        torch_res, tt_dtype=ttnn.bfloat16, tt_device=device, tt_layout=ttnn.TILE_LAYOUT, tt_memory_config=dram_memcfg
    )
    # Poison residual's implicit tile padding. A correct op must ignore these values.
    tt_res = ttnn.fill_implicit_tile_padding(tt_res, POISON)

    tt_stats = ttnn.layer_norm_pre_all_gather(
        tt_inp,
        residual_input_tensor=tt_res,
        dtype=ttnn.bfloat16,
        compute_kernel_config=kernel_config,
        memory_config=dram_memcfg,
    )

    tt_output_host = tt2torch_tensor(tt_stats)

    # Both out_torch and tt_output_host are shape (B, 1, S, 64): two 32-wide tiles where
    # each stat is packed into column 0 of its tile and the remaining 31 columns are zero.
    # reference() constructs out_torch in this layout using the logical-width torch tensor
    # (W=42 values, no tile padding), so out_torch[:,0,0,0] = sum(combined_logical**2) and
    # out_torch[:,0,0,32] = sum(combined_logical).
    # If the op correctly zeroes both tensors' implicit padding before the add, the kernel
    # also reduces over 42 real values (+ 22 zeros), matching the torch sums. If either
    # padding was not zeroed, the POISON values in the padded region would inflate the TT
    # sums, causing the assertions below to fail.
    assert_numeric_metrics(
        out_torch[:, :, :, 0].float(),
        tt_output_host[:, :, :, 0].float(),
        rtol=0.01,
        atol=0.01,
        pcc_threshold=0.9999,
    )
    # Column 32 = start of the second tile where sum(x) lives.
    assert_numeric_metrics(
        out_torch[:, :, :, 32].float(),
        tt_output_host[:, :, :, 32].float(),
        rtol=0.01,
        atol=0.02,
        pcc_threshold=0.9999,
    )


@pytest.mark.parametrize(
    "inp_dtype, res_dtype",
    [
        (ttnn.bfloat16, ttnn.float32),
        (ttnn.float32, ttnn.bfloat16),
    ],
    ids=["bf16_inp_fp32_res", "fp32_inp_bf16_res"],
)
def test_layernorm_pre_all_gather_residual_mismatched_dtype(device, inp_dtype, res_dtype):
    """Input and residual with different dtypes must produce correct result.

    Different dtypes have different per-tile byte sizes (bfloat16=2048B, float32=4096B).
    Each operand's CB must be sized in its own data format; the LLK's add_tiles
    handles per-operand format conversion. For example, sizing cb_res with the input's tile
    size would silently truncate or overrun the residual reads.

    """
    torch.manual_seed(41512)
    inp_shape = (1, 1, 32, 128)

    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG

    torch_inp = torch.randn(inp_shape, dtype=tt_dtype_to_torch_dtype[inp_dtype])
    torch_res = torch.randn(inp_shape, dtype=tt_dtype_to_torch_dtype[res_dtype])
    combined = torch_inp.float() + torch_res.float()

    out_torch = referencefp32(combined.chunk(1, dim=-1), 1, False)[0]

    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    tt_inp = torch2tt_tensor(
        torch_inp, tt_dtype=inp_dtype, tt_device=device, tt_layout=ttnn.TILE_LAYOUT, tt_memory_config=dram_memcfg
    )
    tt_res = torch2tt_tensor(
        torch_res, tt_dtype=res_dtype, tt_device=device, tt_layout=ttnn.TILE_LAYOUT, tt_memory_config=dram_memcfg
    )

    tt_stats = ttnn.layer_norm_pre_all_gather(
        tt_inp,
        residual_input_tensor=tt_res,
        dtype=ttnn.bfloat16,
        compute_kernel_config=kernel_config,
        memory_config=dram_memcfg,
    )

    tt_output_host = tt2torch_tensor(tt_stats)

    # Column 0 = sum(x^2) of the first stats tile, column 32 = sum(x) of the second tile.
    assert_numeric_metrics(
        out_torch[:, :, :, 0].float(),
        tt_output_host[:, :, :, 0].float(),
        rtol=0.01,
        atol=0.1,
        pcc_threshold=0.999,
    )
    assert_numeric_metrics(
        out_torch[:, :, :, 32].float(),
        tt_output_host[:, :, :, 32].float(),
        rtol=0.01,
        atol=0.1,
        pcc_threshold=0.9999,
    )


@pytest.mark.parametrize(
    "inp_shape, res_shape",
    [
        # Both shapes have different logical W but the same tile-aligned padded W.
        # Validation should compare logical shapes rather than padded, and reject mismatched shapes.
        #
        # Residual wider than the input.
        ((1, 1, 32, 38), (1, 1, 32, 62)),  # W=38 and W=62 both pad to W=64
        ((1, 1, 32, 1), (1, 1, 32, 31)),  # W=1  and W=31 both pad to W=32
        #
        # Residual narrower than the input.
        ((1, 1, 32, 62), (1, 1, 32, 38)),  # W=62 and W=38 both pad to W=64
        ((1, 1, 32, 31), (1, 1, 32, 1)),  # W=31 and W=1  both pad to W=32
    ],
    ids=["wider_res_2tile", "wider_res_1tile", "narrower_res_2tile", "narrower_res_1tile"],
)
@pytest.mark.parametrize(
    "op_name",
    ["layer_norm_pre_all_gather", "layer_norm"],
)
def test_residual_logical_shape_mismatch_rejected(device, op_name, inp_shape, res_shape):
    """Residual with a different logical shape from the input must be rejected.

    If validation compares padded_shape, which is tile-aligned and can therefore
    be identical for some logical shapes that are different, the mismatch would pass silently.
    Depending on which tensor is wider, the consequences differ:
    - Residual wider than input: residual's real elements beyond input's logical
      width get added to input's zero-padding, inflating sum(x^2) and sum(x) with
      values that have no counterpart in the input; likely not what the caller intended.
    - Residual narrower than input: input's real elements beyond residual's logical
      width get added to residual's zero-padding (equivalent to silently zero-extending
      the residual). The stats are internally self-consistent, but semantically wrong:
      the caller passed a residual of width W_res intending to add it to an input of
      width W_inp > W_res. The op should reject this rather than silently zero-fill.
    """
    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG
    torch.manual_seed(2)

    tt_inp = torch2tt_tensor(
        torch.randn(inp_shape, dtype=torch.bfloat16),
        tt_dtype=ttnn.bfloat16,
        tt_device=device,
        tt_layout=ttnn.TILE_LAYOUT,
        tt_memory_config=dram_memcfg,
    )
    tt_res = torch2tt_tensor(
        torch.randn(res_shape, dtype=torch.bfloat16),
        tt_dtype=ttnn.bfloat16,
        tt_device=device,
        tt_layout=ttnn.TILE_LAYOUT,
        tt_memory_config=dram_memcfg,
    )

    with pytest.raises(RuntimeError):
        if op_name == "layer_norm_pre_all_gather":
            ttnn.layer_norm_pre_all_gather(
                tt_inp, residual_input_tensor=tt_res, dtype=ttnn.bfloat16, memory_config=dram_memcfg
            )
        else:
            ttnn.layer_norm(tt_inp, epsilon=1e-5, residual_input_tensor=tt_res, memory_config=dram_memcfg)


@pytest.mark.parametrize(
    "inp_shape",
    [
        (1, 1, 32, 128),
        (1, 1, 32, 1024),
        pytest.param(
            (1, 1, 37, 72),
            marks=pytest.mark.xfail(
                strict=True,
                reason="Welford pre-all-gather overflows to inf on non-tile-aligned H shapes; "
                "torch comparison fails on mean and var. Issue #43935",
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "inp_dtype, stats_dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat16),
        (ttnn.float32, ttnn.float32),
    ],
    ids=["bf16_inp_bf16_stats", "fp32_inp_fp32_stats"],
)
def test_layernorm_pre_all_gather_welford_residual(device, inp_shape, inp_dtype, stats_dtype):
    """Welford pre_all_gather, both FUSE_PRE_ADD and no-residual paths.

    Both paths go through LayerNormPreAllGatherWelfordProgramFactory. The fp32 tolerance is
    tight enough to catch e.g. the welford finalize buffer (cb_x2) being held in bfloat16
    rather than desired float32; the bf16 tolerance is loosened to the bf16 noise floor.

    1. Fused-pre-add output (input, residual) vs torch reference for mean and variance.
    2. No-residual output (manually-pre-added input) vs the same torch reference. Independently
       exercises the welford path without FUSE_PRE_ADD set, so a regression isolated to either
       the residual CB plumbing or the welford-only CBs (e.g., cb_x2 width/format) is caught.
    3. Fused-pre-add output vs manually-pre-added output, tight against each other. Catches
       fused-add-specific bugs whose magnitude is below the bf16 torch tolerance, since the
       shared kernel/quantization noise cancels between the two paths.
    """
    torch.manual_seed(0)

    dram_memcfg = ttnn.DRAM_MEMORY_CONFIG
    w = inp_shape[-1]
    torch_dtype = tt_dtype_to_torch_dtype[inp_dtype]

    # The two device inputs to compare: separate (input, residual) for the kernel-fused
    # add path, and a single host-pre-added tensor for the non-fused path.
    torch_inp = torch.randn(inp_shape, dtype=torch_dtype)
    torch_res = torch.randn(inp_shape, dtype=torch_dtype)
    torch_combined = torch_inp + torch_res

    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    tt_inp = torch2tt_tensor(
        torch_inp, tt_dtype=inp_dtype, tt_device=device, tt_layout=ttnn.TILE_LAYOUT, tt_memory_config=dram_memcfg
    )
    tt_res = torch2tt_tensor(
        torch_res, tt_dtype=inp_dtype, tt_device=device, tt_layout=ttnn.TILE_LAYOUT, tt_memory_config=dram_memcfg
    )
    tt_combined = torch2tt_tensor(
        torch_combined,
        tt_dtype=inp_dtype,
        tt_device=device,
        tt_layout=ttnn.TILE_LAYOUT,
        tt_memory_config=dram_memcfg,
    )

    program_config = ttnn.LayerNormDefaultProgramConfig(use_welford=True)
    grid = device.compute_with_storage_grid_size()
    core_range_set = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    recip_tensor = ttnn.create_layer_norm_reciprocals(device, core_range_set, w)

    # Passing residual_input_tensor causes the program factory to set the FUSE_PRE_ADD
    # compile-time define, so the kernel is compiled with the in-kernel add path: it reads
    # tt_inp and tt_res into separate CBs and adds them via add_tiles before the welford pass.
    stats_fused = ttnn.layer_norm_pre_all_gather(
        tt_inp,
        residual_input_tensor=tt_res,
        dtype=stats_dtype,
        compute_kernel_config=kernel_config,
        program_config=program_config,
        memory_config=dram_memcfg,
        recip_tensor=recip_tensor,
    )
    # No residual_input_tensor is passed, so the program factory does not set FUSE_PRE_ADD
    # and the kernel is compiled without the in-kernel add path entirely. The kernel sees
    # a single, already-summed input. Mathematically equivalent to stats_fused.
    stats_combined = ttnn.layer_norm_pre_all_gather(
        tt_combined,
        dtype=stats_dtype,
        compute_kernel_config=kernel_config,
        program_config=program_config,
        memory_config=dram_memcfg,
        recip_tensor=recip_tensor,
    )

    out_fused = tt2torch_tensor(stats_fused)
    out_combined = tt2torch_tensor(stats_combined)

    # Welford output layout: per-row mean lives in tile 0 column 0,
    # per-row variance in tile 1 column 0 (= overall column 32).
    torch_combined_fp32 = torch_combined.float()
    torch_mean = torch_combined_fp32.mean(dim=-1, keepdim=False)
    torch_var = torch_combined_fp32.var(dim=-1, keepdim=False, unbiased=False)
    fused_mean = out_fused[..., 0]
    fused_var = out_fused[..., 32]
    combined_mean = out_combined[..., 0]
    combined_var = out_combined[..., 32]

    # Tolerance is tight enough in fp32 to catch welford CB width/format regressions
    # (e.g., cb_x2 held in bfloat16 in fp32-dest-acc mode); loosened to the bf16 noise floor
    # for bf16 stats.
    if stats_dtype == ttnn.float32:
        atol = 0.002
        rtol = 0.003
        pcc = 0.9999
    else:
        atol = 0.01
        rtol = 0.01
        pcc = 0.999
    # Run every check independently and collect results, so a single test report identifies
    # all failing comparisons instead of stopping at the first one. Each entry is
    # (label, expected, actual). Labels describe what the comparison covers.
    checks = [
        # Fused-pre-add path (FUSE_PRE_ADD set) vs torch reference.
        ("fused vs torch: mean", torch_mean, fused_mean),
        ("fused vs torch: var", torch_var, fused_var),
        # No-residual path (FUSE_PRE_ADD unset) vs the same torch reference. Independently
        # exercises the welford path without the in-kernel add, so a regression isolated to
        # either side is caught.
        ("combined vs torch: mean", torch_mean, combined_mean),
        ("combined vs torch: var", torch_var, combined_var),
        # Fused-pre-add output vs manually-pre-added output. Catches fused-add-specific bugs
        # whose magnitude is below the bf16 torch tolerance: the shared kernel/quantization
        # noise that limits that tolerance cancels between the two paths.
        ("fused vs combined: mean", combined_mean, fused_mean),
        ("fused vs combined: var", combined_var, fused_var),
    ]

    failures = []
    for label, expected, actual in checks:
        passed, message = assert_numeric_metrics(
            expected, actual, rtol=rtol, atol=atol, pcc_threshold=pcc, assert_on_fail=False
        )
        if not passed:
            failures.append(f"[{label}] {message}")

    assert not failures, "Welford comparison(s) failed:\n\n  " + "\n\n  ".join(failures)


@pytest.mark.parametrize(
    "inp_shape",
    [(1, 1, 37, 72)],
)
def test_pre_allgather_ignores_implicit_tile_padding(device, inp_shape):
    """layer_norm_pre_all_gather stats match for ttnn.ones vs torch2tt_tensor."""

    tt_from_torch = torch2tt_tensor(
        torch.ones(inp_shape, dtype=torch.bfloat16),
        tt_dtype=ttnn.bfloat16,
        tt_device=device,
        tt_layout=ttnn.TILE_LAYOUT,
    )
    tt_ones = ttnn.ones(
        shape=inp_shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    stats_from_torch = ttnn.layer_norm_pre_all_gather(
        tt_from_torch,
        dtype=ttnn.bfloat16,
    )
    stats_from_ones = ttnn.layer_norm_pre_all_gather(
        tt_ones,
        dtype=ttnn.bfloat16,
    )

    out_from_torch = tt2torch_tensor(stats_from_torch)
    out_from_ones = tt2torch_tensor(stats_from_ones)

    # test for equivalance
    assert_equal(out_from_torch, out_from_ones)
