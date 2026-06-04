# SPDX-License-Identifier: Apache-2.0
"""Standalone MoE-block test extracted from the generated decode graph (main.py).

Scope: the MoE block of layer 1 — exactly the code between the tracy signposts
`moe_start` and `moe_end` in `main.py._main` (i.e. everything from just after
attention ends through the end of the MoE block).

Data flow across the boundaries (computed by SSA live-in/live-out analysis of
`_main`):

  live-in  activations (loaded from moe_io/, captured during an e2e run):
      ttnn_add_22, ttnn_reshape_117, ttnn_rms_in_4d_3
  live-in  weight-derived consteval constants (recomputed from the SAME weights
      via consteval__main, so no separate dump is needed):
      ce_cache__main (full dict), var_67, var_70, var_76
  live-out tensors (the MoE block's outputs, golden captured during e2e):
      ttnn_add_27          (residual + MoE result)
      ttnn_rms_in_4d_4     (its reshape feeding the lm-head norm)

The MoE block itself is copied verbatim into run_moe_block() below.
We rerun ONLY that block on the captured inputs and PCC-check its outputs
against the captured goldens. Because it is the identical op sequence on
identical inputs, the expected PCC is 1.0.

Capture the moe_io/ tensors first with:  python3 main_capture.py

Then run (TT_METAL_CCACHE_KERNEL_SUPPORT enables ccache for JIT kernel
compilation — see MOE_TEST_TIMINGS.md):

    cd deepseek_codegen/graph_0
    source /home/ubuntu/tt-metal/python_env/bin/activate
    export TT_METAL_HOME=/home/ubuntu/tt-metal
    export PYTHONPATH=/home/ubuntu/tt-metal
    export ARCH_NAME=blackhole
    export TT_METAL_CCACHE_KERNEL_SUPPORT=1
    python3 moe_test.py
"""

import os
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent  # .../graph_0
os.chdir(THIS_DIR)
sys.path.insert(0, str(THIS_DIR))

import ttnn  # noqa: E402
import utils  # noqa: E402
import main as M  # noqa: E402

MESH = (4, 8)
MOE_IO = THIS_DIR / "moe_io"
CE_DIR = MOE_IO / "ce_cache"
PCC_FLOOR = 0.99

# Const-eval tensors the MoE block depends on (the only weight-derived inputs).
# Caching these to disk lets us skip BOTH the 27 GB weight load AND the full
# const-eval pass on every run. var_67/70/76 alias keys 6/17/37; the remaining
# keys are indexed directly inside the MoE region. The all_reduce semaphore
# pools are NOT tensors (can't be dumped) and are cheaply recreated each run.
CE_TENSOR_KEYS = [
    "main_const_eval_6",
    "main_const_eval_9",
    "main_const_eval_16",
    "main_const_eval_17",
    "main_const_eval_19",
    "main_const_eval_24",
    "main_const_eval_25",
    "main_const_eval_29",
    "main_const_eval_35",
    "main_const_eval_37",
    "main_const_eval_39",
    "main_const_eval_gate_up",
]


# ---------------------------------------------------------------------------
# Copied verbatim from main.py: rotating CCL semaphore-pool slot selector used
# by the MoE block's all_reduce_async calls.
# ---------------------------------------------------------------------------
_ccl_pool_slot_counter = [0]


def _ccl_next_slot():
    slot = _ccl_pool_slot_counter[0]
    _ccl_pool_slot_counter[0] = (slot + 1) % 2
    return slot


def run_moe_block(
    ttnn_add_22,
    ttnn_reshape_117,
    ttnn_rms_in_4d_3,
    var_67,
    var_70,
    var_76,
    ce_cache__main,
):
    """The MoE block of layer 1, copied verbatim from main.py._main (the code
    between the `moe_start` and `moe_end` tracy signposts). Inputs are the
    live-in tensors crossing the moe_start boundary; returns the two live-out
    tensors used downstream (ttnn_add_27, ttnn_rms_in_4d_4)."""
    ttnn_rms_stats_4d_3 = ttnn.rms_norm_pre_all_gather(
        ttnn_rms_in_4d_3,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn_rms_sliced_3 = ttnn.slice(
        ttnn_rms_stats_4d_3,
        [0, 0, 0, 0],
        [1, 1, 32, 1],
        [1, 1, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_stats_4d_3, False)
    ttnn_reshape_118 = ttnn.reshape(
        ttnn_rms_sliced_3,
        [1, 32, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rms_sliced_3, False)
    ttnn_all_gather_18 = ttnn.all_gather(
        input_tensor=ttnn_reshape_118,
        dim=0,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_118, False)
    ttnn_sum_15 = ttnn.sum(
        ttnn_all_gather_18,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_all_gather_18, False)
    ttnn_multiply_54 = ttnn.multiply(
        ttnn_sum_15,
        var_70,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_15, False)
    ttnn_add_23 = ttnn.add(
        ttnn_multiply_54,
        var_67,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_54, False)
    ttnn_rsqrt_3 = ttnn.rsqrt(
        ttnn_add_23,
        fast_and_approximate_mode=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_23, False)
    ttnn_reshape_119 = ttnn.reshape(
        ttnn_rsqrt_3,
        [32, 1, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_rsqrt_3, False)
    ttnn_multiply_55 = ttnn.multiply(
        ttnn_reshape_117,
        ttnn_reshape_119,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_119, False)
    ttnn.deallocate(ttnn_reshape_117, False)
    ttnn_typecast_78 = ttnn.multiply(
        ce_cache__main["main_const_eval_24"],
        ttnn_multiply_55,
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_multiply_55, False)
    ttnn_reshape_120 = ttnn.reshape(
        ttnn_typecast_78,
        [32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_typecast_79 = ttnn.typecast(
        ttnn_reshape_120,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_matmul_27 = ttnn.matmul(
        ttnn_typecast_79,
        ce_cache__main["main_const_eval_25"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.FLOAT32,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_typecast_79, False)
    ttnn_reshape_121 = ttnn.reshape(
        ttnn_matmul_27,
        [1, 1, 32, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_27, False)
    ttnn_reduce_scatter_9 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_121,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_121, False)
    ttnn_reshape_122 = ttnn.reshape(
        ttnn_reduce_scatter_9,
        [32, 32],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_9, False)
    ttnn_all_gather_19 = ttnn.all_gather(
        input_tensor=ttnn_reshape_122,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_122, False)
    # === E_router_fuse: deepseek_grouped_gate replaces ~50-op router block ===
    # Inputs:
    #   ttnn_all_gather_19  - FP32 TILE [32, 256]   gate logits (pre-sigmoid)
    #   main_const_eval_9   - FP32 TILE [1,  256]   routing bias
    # Outputs (must keep these variable names for downstream rewiring):
    #   ttnn_multiply_58    - FP32       [32, 1, 8] normalized scaled weights
    #   ttnn_typecast_86    - INT32      [32, 8]    selected expert indices
    _dgg_scores_bf16 = ttnn.typecast(
        ttnn_all_gather_19,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_19, False)
    _dgg_scores_4d = ttnn.reshape(
        _dgg_scores_bf16,
        [1, 1, 32, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_dgg_scores_bf16, False)
    _dgg_bias_bf16 = ttnn.typecast(
        ce_cache__main["main_const_eval_9"],
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    _dgg_bias_4d = ttnn.reshape(
        _dgg_bias_bf16,
        [1, 1, 1, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_dgg_bias_bf16, False)
    _dgg_bias_bcast = ttnn.repeat(
        _dgg_bias_4d,
        ttnn.Shape([1, 1, 32, 1]),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_dgg_bias_4d, False)
    _dgg_weights_bf16, _dgg_indices_u16 = ttnn.experimental.deepseek_grouped_gate(
        _dgg_scores_4d,
        _dgg_bias_bcast,
        n_groups=8,
        summed_experts_per_group=2,
        topk_groups=4,
        n_activated_experts=8,
        route_scale=2.5,
        epsilon=1e-20,
    )
    ttnn.deallocate(_dgg_scores_4d, False)
    ttnn.deallocate(_dgg_bias_bcast, False)
    # E45: drop the BF16→FP32 typecast on _dgg_weights — feed BF16 weights to matmul_29
    ttnn_multiply_58 = ttnn.reshape(
        _dgg_weights_bf16,
        [32, 1, 8],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_dgg_weights_bf16, False)
    _dgg_indices_i32 = ttnn.typecast(
        _dgg_indices_u16,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_dgg_indices_u16, False)
    ttnn_typecast_86 = ttnn.reshape(
        _dgg_indices_i32,
        [32, 8],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(_dgg_indices_i32, False)
    # === END E_router_fuse ===
    ttnn_reshape_140 = ttnn.reshape(
        ttnn_typecast_86,
        [32, 8, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_eq_0 = ttnn.eq(
        ttnn_reshape_140,
        ce_cache__main["main_const_eval_35"],
        dtype=ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_140, False)
    ttnn_typecast_92 = ttnn.typecast(
        ttnn_eq_0,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    # E45: matmul_29 in BF16 — input A (weights) is now BF16, input B (eq mask) skips typecast
    ttnn_matmul_29 = ttnn.matmul(
        ttnn_multiply_58,
        ttnn_eq_0,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_eq_0, False)
    ttnn.deallocate(ttnn_multiply_58, False)
    ttnn_reshape_141 = ttnn.reshape(
        ttnn_matmul_29,
        [1, 32, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_concat_22 = ttnn.concat(
        [ttnn_reshape_141, ttnn_reshape_141, ttnn_reshape_141, ttnn_reshape_141],
        1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_141, False)
    ttnn_all_gather_24 = ttnn.all_gather(
        input_tensor=ttnn_concat_22,
        dim=1,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_concat_22, False)
    ttnn_reshape_142 = ttnn.reshape(
        ttnn_all_gather_24,
        [1, 1, 512, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_24, False)
    ttnn_reshape_143 = ttnn.reshape(
        ttnn_typecast_78,
        [32, 1, 1, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_78, False)
    ttnn_reshape_144 = ttnn.reshape(
        ttnn_typecast_86,
        [32, 1, 1, 8],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_86, False)
    ttnn_all_gather_25 = ttnn.all_gather(
        input_tensor=ttnn_reshape_143,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_143, False)
    ttnn_all_gather_26 = ttnn.all_gather(
        input_tensor=ttnn_all_gather_25,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_all_gather_25, False)
    ttnn_all_gather_27 = ttnn.all_gather(
        input_tensor=ttnn_reshape_144,
        dim=0,
        cluster_axis=0,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_144, False)
    ttnn_to_layout_255 = ttnn.to_layout(ttnn_all_gather_26, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_all_gather_26, False)
    ttnn_typecast_93 = ttnn.typecast(
        ttnn_all_gather_27,
        ttnn.DataType.UINT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_27, False)
    # E40 site 1: on-device to_layout(ROW_MAJOR) replaces from_device→to_layout→to_device round-trip
    ttnn_to_device_65 = ttnn.to_layout(
        ttnn_typecast_93,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_93, False)
    v_92, v_93 = ttnn.all_to_all_dispatch(
        input_tensor=ttnn_to_layout_255,
        expert_indices_tensor=ttnn_to_device_65,
        expert_mapping_tensor=var_76,
        cluster_axis=0,
        memory_config=None,
    )
    ttnn.deallocate(ttnn_to_device_65, False)
    ttnn.deallocate(ttnn_to_layout_255, False)
    ttnn_to_layout_257 = ttnn.to_layout(v_93, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(v_93, False)
    ttnn_typecast_94 = ttnn.typecast(
        ttnn_to_layout_257,
        ttnn.DataType.INT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_257, False)
    ttnn_to_layout_258 = ttnn.to_layout(v_92, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(v_92, False)
    ttnn_reshape_145 = ttnn.reshape(
        ttnn_typecast_94,
        [1, 1, 512, 8],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_94, False)
    # E45: typecast_95 (FP32→BF16) eliminated — reshape_142 is already BF16 after matmul_29 fold
    ttnn_to_layout_259 = ttnn.to_layout(ttnn_reshape_142, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_reshape_142, False)
    ttnn_typecast_96 = ttnn.typecast(
        ttnn_reshape_145,
        ttnn.DataType.UINT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_145, False)
    # E40 site 2: on-device to_layout(ROW_MAJOR) replaces from_device→to_layout→to_device round-trip
    ttnn_to_device_66 = ttnn.to_layout(
        ttnn_typecast_96,
        ttnn.Layout.ROW_MAJOR,
        None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_96, False)
    v_94, v_95 = ttnn.moe_expert_token_remap(
        topk_tensor=ttnn_to_layout_259,
        expert_mapping_tensor=var_76,
        expert_metadata_tensor=ttnn_to_device_66,
        reduction_size=32,
        memory_config=None,
    )
    ttnn.deallocate(v_94, False)
    ttnn.deallocate(ttnn_to_layout_259, False)
    ttnn_to_layout_261 = ttnn.to_layout(v_95, ttnn.Layout.TILE, None, memory_config=None)
    ttnn_typecast_97 = ttnn.typecast(
        ttnn_to_layout_261,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_261, False)
    ttnn_reshape_146 = ttnn.reshape(
        ttnn_to_layout_258,
        [16, 1, 32, 7168],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_258, False)
    ttnn_reshape_147 = ttnn.reshape(
        ttnn_typecast_97,
        [16, 1, 1, 8],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_97, False)
    ttnn_typecast_98 = ttnn.typecast(
        ttnn_reshape_147,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_147, False)
    ttnn_to_layout_262 = ttnn.to_layout(ttnn_typecast_98, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_typecast_98, False)
    ttnn_sparse_matmul_gate_up = ttnn.sparse_matmul(
        input_tensor_a=ttnn_reshape_146,
        input_tensor_b=ce_cache__main["main_const_eval_gate_up"],
        sparsity=ttnn_to_layout_262,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=2,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
            gather_in0=False,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=0,
            untilize_out=False,
        ),
        is_input_a_sparse=False,
        is_input_b_sparse=True,
        memory_config=None,
        dtype=None,
    )
    ttnn.deallocate(ttnn_to_layout_262, False)
    ttnn.deallocate(ttnn_reshape_146, False)
    ttnn_reshape_gate_up = ttnn.reshape(
        ttnn_sparse_matmul_gate_up,
        [16, 8, 32, 4096],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sparse_matmul_gate_up, False)
    ttnn_reshape_148 = ttnn.slice(
        ttnn_reshape_gate_up,
        [0, 0, 0, 0],
        [16, 8, 32, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn_reshape_149 = ttnn.slice(
        ttnn_reshape_gate_up,
        [0, 0, 0, 2048],
        [16, 8, 32, 4096],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_gate_up, False)
    ttnn_multiply_59 = ttnn.multiply(
        ttnn_reshape_148,
        ttnn_reshape_149,
        dtype=ttnn.DataType.BFLOAT16,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_149, False)
    ttnn.deallocate(ttnn_reshape_148, False)
    # E40 site 3: on-device typecast replaces from_device→typecast→to_device round-trip
    ttnn_to_device_67 = ttnn.typecast(
        v_95,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(v_95, False)
    ttnn_sparse_matmul_2 = ttnn.sparse_matmul(
        input_tensor_a=ttnn_multiply_59,
        input_tensor_b=ce_cache__main["main_const_eval_39"],
        sparsity=ttnn_to_device_67,
        program_config=ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 7),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            out_block_h=1,
            out_block_w=1,
            per_core_M=1,
            per_core_N=4,
            fuse_batch=False,
            fused_activation=None,
            mcast_in0=True,
            gather_in0=False,
            hop_cores=ttnn.CoreRangeSet([]),
            num_global_cb_receivers=0,
            untilize_out=False,
        ),
        is_input_a_sparse=True,
        is_input_b_sparse=False,
        memory_config=None,
        dtype=None,
    )
    ttnn.deallocate(ttnn_to_device_67, False)
    ttnn.deallocate(ttnn_multiply_59, False)
    ttnn_permute_44 = ttnn.permute(
        ttnn_sparse_matmul_2,
        [1, 0, 2, 3],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_sparse_matmul_2, False)
    ttnn_reshape_150 = ttnn.reshape(
        ttnn_permute_44,
        [8, 1, 512, 7168],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_44, False)
    ttnn_to_layout_263 = ttnn.to_layout(ttnn_reshape_150, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_reshape_150, False)
    ttnn_all_to_all_combine_0 = ttnn.all_to_all_combine(
        input_tensor=ttnn_to_layout_263,
        expert_metadata_tensor=ttnn_to_device_66,
        expert_mapping_tensor=var_76,
        cluster_axis=0,
        output_shard_dim=2,
        memory_config=None,
    )
    ttnn.deallocate(ttnn_to_layout_263, False)
    ttnn.deallocate(ttnn_to_device_66, False)
    ttnn_post_combine_tilized = ttnn.experimental.deepseek_moe_post_combine_tilize(
        ttnn_all_to_all_combine_0,
        output_memory_config=ttnn.MemoryConfig(
            buffer_type=ttnn.BufferType.L1,
            nd_shard_spec=ttnn.NdShardSpec(
                shard_shape=ttnn.Shape([32, 3584]),
                grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))]),
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
            ),
        ),
    )
    ttnn.deallocate(ttnn_all_to_all_combine_0, False)
    ttnn_to_layout_264 = ttnn.to_memory_config(
        ttnn_post_combine_tilized,
        ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_post_combine_tilized, False)
    # E49: fuse the post-MoE reduce_scatter_10 + all_gather_28 (both cluster_axis=1, dim=3)
    # into one ttnn.experimental.all_reduce_async using the rotating semaphore pool.
    _ar0_slot = _ccl_next_slot()
    ttnn_all_reduce_0 = ttnn.experimental.all_reduce_async(
        ttnn_to_layout_264,
        cluster_axis=1,
        mesh_device=utils.DeviceGetter.get_device((4, 8)),
        barrier_semaphores=ce_cache__main["main_const_eval_all_reduce_pool_barrier"][_ar0_slot],
        rs_global_semaphores=ce_cache__main["main_const_eval_all_reduce_pool_rs"][_ar0_slot],
        ag_global_semaphores=ce_cache__main["main_const_eval_all_reduce_pool_ag"][_ar0_slot],
        math_op=ttnn.ReduceType.Sum,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_264, False)
    ttnn_to_layout_265 = ttnn.to_layout(ttnn_all_reduce_0, ttnn.Layout.ROW_MAJOR, None, memory_config=None)
    ttnn.deallocate(ttnn_all_reduce_0, False)
    ttnn_mesh_partition_3 = ttnn.mesh_partition(
        input_tensor=ttnn_to_layout_265,
        dim=2,
        cluster_axis=0,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_265, False)
    ttnn_mesh_partition_4 = ttnn.mesh_partition(
        input_tensor=ttnn_mesh_partition_3,
        dim=3,
        cluster_axis=1,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_mesh_partition_3, False)
    ttnn_to_layout_266 = ttnn.to_layout(ttnn_mesh_partition_4, ttnn.Layout.TILE, None, memory_config=None)
    ttnn.deallocate(ttnn_mesh_partition_4, False)
    ttnn_typecast_100 = ttnn.typecast(
        ttnn_to_layout_266,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_to_layout_266, False)
    ttnn_reshape_151 = ttnn.reshape(
        ttnn_matmul_29,
        [32, 256, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_29, False)
    # E45: matmul_30 still wants FP32 inputs — cast the BF16 matmul_29 result back to FP32
    ttnn_reshape_151_fp32 = ttnn.typecast(
        ttnn_reshape_151,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_151, False)
    ttnn_matmul_30 = ttnn.matmul(
        ttnn_typecast_92,
        ttnn_reshape_151_fp32,
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.FLOAT32,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_151_fp32, False)
    ttnn.deallocate(ttnn_typecast_92, False)
    ttnn_reshape_152 = ttnn.reshape(
        ttnn_matmul_30,
        [32, 8],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_30, False)
    ttnn_permute_45 = ttnn.permute(
        ttnn_reshape_152,
        [1, 0],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        pad_value=0.0,
    )
    ttnn.deallocate(ttnn_reshape_152, False)
    ttnn_reshape_153 = ttnn.reshape(
        ttnn_permute_45,
        [8, 1, 32, 1],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_permute_45, False)
    ttnn_multiply_60 = ttnn.multiply(
        ttnn_typecast_100,
        ttnn_reshape_153,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_153, False)
    ttnn.deallocate(ttnn_typecast_100, False)
    ttnn_sum_18 = ttnn.sum(
        ttnn_multiply_60,
        [0],
        False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_multiply_60, False)
    ttnn_typecast_101 = ttnn.typecast(
        ttnn_sum_18,
        ttnn.DataType.BFLOAT16,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_sum_18, False)
    ttnn_matmul_31 = ttnn.matmul(
        ttnn_reshape_120,
        ce_cache__main["main_const_eval_19"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn_reshape_154 = ttnn.reshape(
        ttnn_matmul_31,
        [1, 1, 32, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_31, False)
    ttnn_reduce_scatter_11 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_154,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_154, False)
    ttnn_reshape_155 = ttnn.reshape(
        ttnn_reduce_scatter_11,
        [32, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_11, False)
    ttnn_all_gather_29 = ttnn.all_gather(
        input_tensor=ttnn_reshape_155,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_155, False)
    ttnn_typecast_102 = ttnn.typecast(
        ttnn_all_gather_29,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_29, False)
    ttnn_matmul_32 = ttnn.matmul(
        ttnn_reshape_120,
        ce_cache__main["main_const_eval_29"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_reshape_120, False)
    ttnn_reshape_156 = ttnn.reshape(
        ttnn_matmul_32,
        [1, 1, 32, 2048],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_32, False)
    ttnn_reduce_scatter_12 = ttnn.reduce_scatter(
        input_tensor=ttnn_reshape_156,
        dim=3,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    ttnn.deallocate(ttnn_reshape_156, False)
    ttnn_reshape_157 = ttnn.reshape(
        ttnn_reduce_scatter_12,
        [32, 256],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reduce_scatter_12, False)
    ttnn_all_gather_30 = ttnn.all_gather(
        input_tensor=ttnn_reshape_157,
        dim=1,
        cluster_axis=1,
        subdevice_id=None,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        num_links=None,
        topology=ttnn.Topology.Ring,
    )
    ttnn.deallocate(ttnn_reshape_157, False)
    ttnn_typecast_103 = ttnn.typecast(
        ttnn_all_gather_30,
        ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_all_gather_30, False)
    ttnn_typecast_104 = ttnn.multiply(
        ttnn_typecast_102,
        ttnn_typecast_103,
        dtype=ttnn.DataType.BFLOAT16,
        input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_typecast_103, False)
    ttnn.deallocate(ttnn_typecast_102, False)
    ttnn_matmul_33 = ttnn.matmul(
        ttnn_typecast_104,
        ce_cache__main["main_const_eval_16"],
        transpose_a=False,
        transpose_b=False,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
        dtype=ttnn.DataType.BFLOAT16,
        program_config=None,
        activation=None,
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=True
        ),
    )
    ttnn.deallocate(ttnn_typecast_104, False)
    ttnn_typecast_105 = ttnn.add(
        ttnn_matmul_33,
        ttnn_typecast_101,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_matmul_33, False)
    ttnn.deallocate(ttnn_typecast_101, False)
    ttnn_reshape_158 = ttnn.reshape(
        ttnn_add_22,
        [1, 32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_add_22, False)
    ttnn_add_27 = ttnn.add(
        ttnn_typecast_105,
        ttnn_reshape_158,
        dtype=ttnn.DataType.FLOAT32,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    ttnn.deallocate(ttnn_reshape_158, False)
    ttnn.deallocate(ttnn_typecast_105, False)
    ttnn_rms_in_4d_4 = ttnn.reshape(
        ttnn_add_27,
        [1, 1, 32, 896],
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None),
    )
    return ttnn_add_27, ttnn_rms_in_4d_4


def compute_pcc(golden: torch.Tensor, device_out: torch.Tensor) -> float:
    g = golden.to(torch.float32).flatten()
    d = device_out.to(torch.float32).flatten()
    gc = g - g.mean()
    dc = d - d.mean()
    denom = gc.norm() * dc.norm()
    if denom == 0:
        return 1.0 if torch.allclose(g, d, rtol=1e-2, atol=1e-2) else 0.0
    return max(-1.0, min(1.0, ((gc @ dc) / denom).item()))


def _dram():
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def _load_to_device(path, device):
    """Load a dumped tensorbin (preserves dtype/layout/mesh distribution) and
    place it on-device in interleaved DRAM (the config it was captured in)."""
    host_t = ttnn.load_tensor(str(path))
    return ttnn.to_device(host_t, device, _dram())


def _to_torch(t, device):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)).float()


def _ce_cache_ready() -> bool:
    return all((CE_DIR / f"{k}.tensorbin").exists() for k in CE_TENSOR_KEYS)


def _add_semaphore_pools(ce: dict) -> None:
    """Recreate the rotating all_reduce semaphore pools (not dumpable)."""
    pools = M.main_const_eval_all_reduce_semaphores()
    ce["main_const_eval_all_reduce_pool_barrier"] = pools[0]
    ce["main_const_eval_all_reduce_pool_rs"] = pools[1]
    ce["main_const_eval_all_reduce_pool_ag"] = pools[2]


def build_ce_cache(device) -> dict:
    """Slow path (run once): load weights + full const-eval, then dump the
    MoE block's const-eval tensor dependencies to disk."""
    print("Const-eval cache miss: loading weights + running const-eval once...")
    weights = M.load_weights_for__main()
    M.ce_cache__main = M.consteval__main(M.ce_cache__main, weights)
    CE_DIR.mkdir(parents=True, exist_ok=True)
    for k in CE_TENSOR_KEYS:
        ttnn.dump_tensor(str(CE_DIR / f"{k}.tensorbin"), M.ce_cache__main[k])
    print(f"  cached {len(CE_TENSOR_KEYS)} const-eval tensors -> {CE_DIR}")
    return M.ce_cache__main


def load_ce_cache(device) -> dict:
    """Fast path: load only the MoE block's const-eval tensors from disk and
    recreate the semaphore pools. Skips the weight load and const-eval."""
    print("Const-eval cache hit: loading cached const-eval tensors from disk...")
    ce = {}
    for k in CE_TENSOR_KEYS:
        ce[k] = ttnn.load_tensor(str(CE_DIR / f"{k}.tensorbin"), device=device)
    _add_semaphore_pools(ce)
    M.ce_cache__main = ce
    return ce


def main() -> int:
    import time

    rebuild = "--rebuild-ce-cache" in sys.argv
    timings = {}

    t = time.perf_counter()
    device = utils.DeviceGetter.get_device(MESH)
    timings["device_open"] = time.perf_counter() - t

    # 1) Const-eval tensors (the MoE block's only weight-derived inputs). Built
    #    once from the SAME weights, then cached so subsequent runs skip the
    #    27 GB weight load and the full const-eval pass.
    t = time.perf_counter()
    if _ce_cache_ready() and not rebuild:
        ce = load_ce_cache(device)
    else:
        ce = build_ce_cache(device)
    var_67 = ce["main_const_eval_6"]
    var_70 = ce["main_const_eval_17"]
    var_76 = ce["main_const_eval_37"]
    timings["ce_cache_load"] = time.perf_counter() - t

    # 2) Captured live-in activations.
    t = time.perf_counter()
    ttnn_add_22 = _load_to_device(MOE_IO / "in_ttnn_add_22.tensorbin", device)
    ttnn_reshape_117 = _load_to_device(MOE_IO / "in_ttnn_reshape_117.tensorbin", device)
    ttnn_rms_in_4d_3 = _load_to_device(MOE_IO / "in_ttnn_rms_in_4d_3.tensorbin", device)
    timings["input_load"] = time.perf_counter() - t

    # 3) Run ONLY the MoE block (copied verbatim into run_moe_block()). This is
    #    where JIT kernel compilation happens (on a cold kernel cache), so it is
    #    the phase that TT_METAL_CCACHE_KERNEL_SUPPORT accelerates. synchronize
    #    so the timing captures compile + execution, not just dispatch.
    t = time.perf_counter()
    out_add_27, out_rms_in_4d_4 = run_moe_block(
        ttnn_add_22,
        ttnn_reshape_117,
        ttnn_rms_in_4d_3,
        var_67,
        var_70,
        var_76,
        M.ce_cache__main,
    )
    ttnn.synchronize_device(device)
    timings["moe_block"] = time.perf_counter() - t
    outputs = {
        "ttnn_add_27": out_add_27,
        "ttnn_rms_in_4d_4": out_rms_in_4d_4,
    }

    # 4) PCC vs captured goldens.
    all_pass = True
    print("\n=== MoE block PCC (recomputed vs captured e2e golden) ===")
    for name, got in outputs.items():
        gold = _load_to_device(MOE_IO / f"out_{name}.tensorbin", device)
        a = _to_torch(got, device)
        b = _to_torch(gold, device)
        if a.shape != b.shape:
            print(f"  {name}: SHAPE MISMATCH got {tuple(a.shape)} vs golden {tuple(b.shape)}")
            all_pass = False
            continue
        pcc = compute_pcc(b, a)
        maxd = (a - b).abs().max().item()
        ok = pcc >= PCC_FLOOR
        all_pass &= ok
        print(f"  {name}: PCC={pcc:.6f} max|Δ|={maxd:.3e} shape={tuple(a.shape)} " f"{'PASS' if ok else 'FAIL'}")

    print(("\nPASS: MoE block matches e2e golden." if all_pass else "\nFAIL: MoE block diverged from e2e golden."))

    print("\n=== timings (s) ===")
    for k in ("device_open", "ce_cache_load", "input_load", "moe_block"):
        print(f"  {k:14s}: {timings[k]:8.3f}")
    print(
        f"  {'TOTAL_measured':14s}: {sum(timings.values()):8.3f}  "
        f"(TT_METAL_CCACHE_KERNEL_SUPPORT={'set' if os.environ.get('TT_METAL_CCACHE_KERNEL_SUPPORT') else 'unset'})"
    )

    if utils.DeviceGetter._instance is not None:
        ttnn.close_mesh_device(utils.DeviceGetter._instance)
        utils.DeviceGetter._instance = None
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
