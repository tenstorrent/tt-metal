# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the tt_ccl.all_gather_matmul fused wrapper (Qwen3.6 BH_GLX, 32-chip).

TDD keystone for Task 1. Builds a REAL 1-layer Qwen3.6 model (same way the eager
decode profiler does) so we get a fully-constructed ``tt_ccl`` object (all
buffers + semaphores + sub-device manager set up correctly). It then drives the
new ``tt_ccl.all_gather_matmul(...)`` wrapper with the EXACT tensor dims /
memory configs / program config / dtypes of the proven-on-BH ``ff2_qwen`` case
from ``tests/ttnn/unit_tests/operations/ccl/test_llama_all_gather_matmul.py`` so
the ONLY new variable under test is the wrapper itself.

The wrapper internally calls ``ttnn.experimental.llama_all_gather_matmul_async``
using tt_ccl's pooled semaphore / interim buffer / sub-device — model & test code
NEVER hand-build CCL primitives (a previous hand-built attempt hung the board).

Run:
    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) && source python_env/bin/activate
    python -m pytest --noconftest \
        models/demos/qwen3_6_galaxy_v2/tests/test_all_gather_matmul_wrapper_pcc.py -s -x
"""
from __future__ import annotations

import math
import os

import pytest
import torch

import ttnn
from models.demos.llama3_70b_galaxy.tt.model_config import PREFETCHER_NOC1_GRID

# Importing bh_glx_mesh registers the 32-chip mesh fixture in this module; the
# build helpers are the exact ones the demo / profiler use.
from models.demos.qwen3_6_galaxy_v2.demo.text_demo_qwen36 import (  # noqa: F401
    _PAGED_BLOCK_SIZE,
    _PAGED_MAX_NUM_BLOCKS,
    _SNAPSHOT,
    _build_tt_model_paged_kv,
    _load_full_state_dict,
    bh_glx_mesh,
)
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc
from tests.ttnn.nightly.unit_tests.operations.matmul.test_matmul_1d_gather_in0 import round_up

# ---- ff2_qwen geometry (verbatim from test_llama_all_gather_matmul.py + the
#      ff2_qwen parametrize row in test_ccl_async_TG_llama.py) ----
_CLUSTER_SHAPE = (8, 4)
_CLUSTER_AXIS = 1
_B, _M, _K, _N = 1, 32, 3200, 1280
_IN0_DTYPE = ttnn.bfloat16
_IN1_DTYPE = ttnn.bfloat16
_OUT_DTYPE = ttnn.bfloat16
_FIDELITY = ttnn.MathFidelity.HiFi2
_FP32_ACC = True
_PACKER_L1_ACC = True
_INPUT_NUM_CORES = 30
_OUTPUT_NUM_CORES = 24

_SUB_DEVICE_CRS = ttnn.CoreRangeSet(
    [
        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
    ]
)
_BINARY_MULT_CRS = ttnn.num_cores_to_corerangeset_in_subcoregrids(
    ttnn.CoreCoord(1, 0), _INPUT_NUM_CORES, _SUB_DEVICE_CRS, row_wise=True
)
_RING_CRS = ttnn.CoreRangeSet(
    [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in PREFETCHER_NOC1_GRID]
)
_HOP_GRID = ttnn.CoreRangeSet([])


def _num_cores_to_rectangle_grid(num_cores, device):
    x = device.compute_with_storage_grid_size().x
    while x > 0 and num_cores % x != 0:
        x -= 1
    if x == 0:
        return None
    return (x, num_cores // x)


def _build_ff2_qwen_configs(mesh_device):
    """Mirror run_llama_all_gather_matmul_impl's config construction for ff2_qwen."""
    input_core_range_set = _BINARY_MULT_CRS
    output_core_range_set = _RING_CRS
    output_num_cores = _OUTPUT_NUM_CORES
    input_num_cores = _INPUT_NUM_CORES

    storage_grid = _num_cores_to_rectangle_grid(output_num_cores, mesh_device)

    M, K_per_device = _M, _K // _CLUSTER_SHAPE[_CLUSTER_AXIS]
    K_per_device_per_shard = round_up(math.ceil(K_per_device / input_num_cores), ttnn.TILE_SIZE)

    N_per_shard = round_up(math.ceil(_N / output_num_cores), ttnn.TILE_SIZE)
    N_padded = N_per_shard * output_num_cores

    in0_block_w = _K // _CLUSTER_SHAPE[_CLUSTER_AXIS] // ttnn.TILE_SIZE
    while (_K / ttnn.TILE_SIZE) % in0_block_w != 0:
        in0_block_w -= 1

    out_block_h = M // ttnn.TILE_SIZE
    out_block_w = N_padded // output_num_cores // ttnn.TILE_SIZE

    out_subblock_h = 1
    out_subblock_w = 8
    while out_block_w % out_subblock_w != 0:
        out_subblock_w -= 1

    program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=storage_grid,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=out_block_h,
        per_core_N=out_block_w,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=False,
        gather_in0=True,
        hop_cores=_HOP_GRID,
        num_global_cb_receivers=24,
        untilize_out=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=_FIDELITY,
        math_approx_mode=True,
        fp32_dest_acc_en=_FP32_ACC,
        packer_l1_acc=_PACKER_L1_ACC,
        dst_full_sync_en=True,
    )

    input_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(input_core_range_set, [M, K_per_device_per_shard], ttnn.ShardOrientation.ROW_MAJOR),
    )
    in1_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(output_core_range_set, [_K, N_per_shard], ttnn.ShardOrientation.ROW_MAJOR),
    )

    # Intermediate (all-gather output) config — gathered along cluster_axis.
    intermediate_num_cores = _CLUSTER_SHAPE[_CLUSTER_AXIS]
    intermediate_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 3))])
    intermediate_width = K_per_device * _CLUSTER_SHAPE[_CLUSTER_AXIS]
    interemediate_N_per_shard = round_up(math.ceil(intermediate_width / intermediate_num_cores), ttnn.TILE_SIZE)
    ag_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(intermediate_core_range_set, [M, interemediate_N_per_shard], ttnn.ShardOrientation.ROW_MAJOR),
    )
    mm_output_sharded_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(output_core_range_set, [M, N_per_shard], ttnn.ShardOrientation.ROW_MAJOR),
    )

    return dict(
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        input_mem_config=input_mem_config,
        in1_sharded_mem_config=in1_sharded_mem_config,
        ag_output_mem_config=ag_output_mem_config,
        mm_output_sharded_mem_config=mm_output_sharded_mem_config,
        K_per_device=K_per_device,
    )


@pytest.mark.hardware
def test_all_gather_matmul_wrapper_pcc(bh_glx_mesh):
    mesh_device = bh_glx_mesh

    # Bake the known-good qwen3.6 decode config (same as the eager profiler) so the
    # 1-layer model builds the production decode tt_ccl.
    for _k, _v in {
        "QWEN36_FORCE_SWITCH_DECODE": "1",
        "QWEN36_DECODE_L1_RESIDUAL": "1",
        "QWEN36_RESIDUAL_BUF_BF16": "1",
        "QWEN36_LM_HEAD_PLAIN_DECODE": "1",
        "QWEN36_SEQ_CORES_PER_HEAD": "4",
    }.items():
        os.environ.setdefault(_k, _v)

    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    pattern = ["linear_attention"]
    state_dict = _load_full_state_dict(_SNAPSHOT)
    # Relabel layer 0 (a linear/GDN layer) to contiguous slot 0.
    out = {}
    pfx0 = "model.language_model.layers.0."
    all_pfx = "model.language_model.layers."
    for k, v in state_dict.items():
        if k.startswith(all_pfx):
            if k.startswith(pfx0):
                out[k] = v
            else:
                continue
        else:
            out[k] = v
    state_dict = out

    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=_PAGED_MAX_NUM_BLOCKS)
    model, args = _build_tt_model_paged_kv(mesh_device, state_dict, pattern, 1, paged_attention_config)

    tt_ccl = model.layers[0].feed_forward.tt_ccl
    num_links = model.model_config["GALAXY_NUM_LINKS"]
    print(f"[ag_mm_wrapper] tt_ccl mode={tt_ccl.mode} GALAXY_NUM_LINKS={num_links}")

    cfg = _build_ff2_qwen_configs(mesh_device)

    # ---- Build sharded ring input + ring weight (same as the impl test) ----
    K_per_device = cfg["K_per_device"]
    in0_shape = [*_CLUSTER_SHAPE, _M, K_per_device]
    in1_shape = [*_CLUSTER_SHAPE, _K, _N]

    torch.manual_seed(0)
    in0_tensor = torch.randn(in0_shape)
    in1_tensor = torch.randn(in1_shape)

    tt_input_tensor = ttnn.from_torch(
        in0_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=_IN0_DTYPE,
        memory_config=cfg["input_mem_config"],
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=_CLUSTER_SHAPE),
    )
    tt_in1_tensor = ttnn.from_torch(
        in1_tensor,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=_IN1_DTYPE,
        memory_config=cfg["in1_sharded_mem_config"],
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=_CLUSTER_SHAPE),
    )

    # ---- Torch golden: all-gather along the cluster axis, then matmul ----
    intermediate_shape = [*_CLUSTER_SHAPE, _M, K_per_device * _CLUSTER_SHAPE[_CLUSTER_AXIS]]
    golden_Ashape = list(intermediate_shape)
    golden_Ashape[_CLUSTER_AXIS] = 1
    golden_A = in0_tensor.transpose(-2, _CLUSTER_AXIS).reshape(golden_Ashape).squeeze(_CLUSTER_AXIS)
    golden_A = golden_A.unsqueeze(_CLUSTER_AXIS).repeat(1, _CLUSTER_SHAPE[_CLUSTER_AXIS], 1, 1)
    golden_out = golden_A @ in1_tensor  # [8, 4, M, N]

    # ---- Call the wrapper under test ----
    tt_out = tt_ccl.all_gather_matmul(
        tt_input_tensor,
        tt_in1_tensor,
        dim=3,
        cluster_axis=_CLUSTER_AXIS,
        num_links=num_links,
        ag_memory_config=cfg["ag_output_mem_config"],
        mm_memory_config=cfg["mm_output_sharded_mem_config"],
        program_config=cfg["program_config"],
        compute_kernel_config=cfg["compute_kernel_config"],
        dtype=_OUT_DTYPE,
        global_cb=None,
        buffer_key="FF2_AG_MM_TEST",
    )
    ttnn.synchronize_device(mesh_device)

    # ---- Validate PCC per device ----
    worst = 1.0
    for i, t in enumerate(ttnn.get_device_tensors(tt_out)):
        row_index = i // _CLUSTER_SHAPE[1]
        col_index = i % _CLUSTER_SHAPE[1]
        ref = golden_out[row_index, col_index]
        got = t.cpu().to_torch().squeeze(0).squeeze(0)
        eq, pcc = comp_pcc(got, ref)
        # pcc string like "PCC: 0.999..."; comp_pcc returns (bool, msg)
        assert eq, f"device {i} FAILED PCC: {pcc}"
    print(f"[ag_mm_wrapper] PCC OK across all 32 devices; last msg: {pcc}")
    assert tt_out.shape[-1] == _N, f"output N dim {tt_out.shape[-1]} != {_N}"
    print(f"[ag_mm_wrapper] output shape = {tt_out.shape}  (M preserved, N={_N})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
