# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Discriminating fp32-reload precision test for llama_all_gather_matmul_async.

REQUIRES a 6U Galaxy (8x4 mesh, unharvested 7x10 grid) — the fused op is mesh/CCL based.
It was NOT runnable on the single-device machine the fix was developed on, so this test is
provided for you to run on Galaxy hardware. It reuses the existing, proven scaffolding
(`run_llama_all_gather_matmul_impl`) unchanged and only:
  1. injects ill-conditioned data via monkeypatch (in0 gets a large common offset; in1's
     columns sum to zero over K), so the offset cancels from the true result but bloats the
     mid-accumulation partials, and
  2. selects a config that actually exercises the fp32 cross-block reload:
       output_dtype = float32       -> the intermediate K-partials are Float32
       fp32_acc_mode = True         -> fp32 DEST accumulation
       packer_l1_acc = False        -> partials are reloaded via copy_block_matmul_partials
                                       (with packer_l1_acc=True there is no reload, so the fix
                                        is inactive -- note the *shipped* llama configs use
                                        packer_l1_acc=True, so this is a synthetic config
                                        specifically to expose/validate the reload path).

Validate fails-before/passes-after:
  * built with the fix (unpack_to_dest_mode line in llama_1d_mm_fusion.cpp) -> PASS
  * comment out that line, rebuild -> FAIL
The op's own validation uses comp_pcc at 0.99. Expect a moderate gap (the fused kernel uses
4 multicast K-blocks), similar to the single-device gather_in0 case (unfixed ~0.9 -> fixed pass).

Tuning notes if it doesn't discriminate cleanly on your hardware:
  * increase OFFSET (bigger partials -> worse unfixed) but keep it modest if in0_dtype is bf16
    (a too-large offset is lost in bf16 input quantization and caps even the fixed PCC);
  * or set in0_dtype/in1_dtype to ttnn.float32 to remove the input-quantization cap (if the
    op supports fp32 activations on your build).
"""

import pytest
import torch

import ttnn

from tests.ttnn.unit_tests.operations.ccl.test_llama_all_gather_matmul import (
    run_llama_all_gather_matmul_impl,
)
from tests.ttnn.unit_tests.operations.ccl.test_ccl_async_TG_llama import (
    BINARY_MULT_CRS,
    RING_CRS,
    PREFETCHER_NOC1_GRID,
)

OFFSET = 1000.0
M_ROWS = 32  # the run uses M=32; in0 has M at dim -2, in1 has K at dim -2 (used to tell them apart)


@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize(
    "cluster_axis, num_links, input_num_cores, input_core_range_set, output_num_cores, output_core_range_set",
    [(1, 3, 30, BINARY_MULT_CRS, 24, RING_CRS)],
    ids=["binary_mult"],
)
@pytest.mark.parametrize("has_bias", [False], ids=["no_bias"])
def test_llama_all_gather_matmul_fp32_reload_precision(
    mesh_device,
    cluster_axis,
    num_links,
    input_num_cores,
    input_core_range_set,
    output_num_cores,
    output_core_range_set,
    has_bias,
    function_level_defaults,
    ensure_devices_tg,
    monkeypatch,
):
    # Inject ill-conditioned data without touching the shared impl: in0 (M at dim -2) gets the
    # common offset; in1 (K at dim -2) has its columns zeroed over K so the offset cancels.
    orig_randn = torch.randn

    def patched_randn(*args, **kwargs):
        t = orig_randn(*args, **kwargs)
        if t.dim() >= 2 and t.shape[-2] == M_ROWS:  # in0: [*cluster, M, K_per_device]
            return t + OFFSET
        if t.dim() >= 2:  # in1: [*cluster, K, N] -> make each output column sum to 0 over K
            return t - t.mean(dim=-2, keepdim=True)
        return t

    monkeypatch.setattr(torch, "randn", patched_randn)

    run_llama_all_gather_matmul_impl(
        mesh_device,
        1,  # B
        M_ROWS,  # M
        3200,  # K
        1280,  # N
        cluster_axis,
        ttnn.bfloat16,  # in0_dtype (matches shipped op support; offset kept modest for bf16)
        ttnn.bfloat16,  # in1_dtype
        num_links,
        input_num_cores,
        input_core_range_set,
        output_num_cores,
        output_core_range_set,
        ttnn.float32,  # output_dtype -> Float32 intermediate partials (exercises the fp32 reload)
        ttnn.MathFidelity.HiFi4,
        has_bias,
        True,  # fp32_acc_mode
        False,  # packer_l1_acc = False -> reload path (the case the fix addresses)
        PREFETCHER_NOC1_GRID,
        False,  # in1_is_dram_interleaved
        num_iters=1,
        trace_mode=False,
        validate_all=False,
    )
