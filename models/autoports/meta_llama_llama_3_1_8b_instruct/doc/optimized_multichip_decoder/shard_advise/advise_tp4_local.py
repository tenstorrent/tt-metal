# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shard-advisor capture for the exact TP4-local batch-one decode graph.

The advisor does not model TTNN collectives.  The two all-reduce boundaries
are identity dependencies here; hardware profiling and candidate tests cover
their placement and implementation.
"""

from __future__ import annotations

import torch

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.doc.optimized_decoder.shard_advise import (
    advise_llama as advisor_base,
)

BATCH = 1
MAX_CACHE_LEN = 128
CURRENT_POS = 18
HIDDEN = 4096
LOCAL_Q_HEADS = 8
LOCAL_KV_HEADS = 2
HEAD_DIM = 128
LOCAL_INTERMEDIATE = 3584


class _CaptureMesh:
    shape = (1, 4)

    @staticmethod
    def get_num_devices():
        return 4

    @staticmethod
    def compute_with_storage_grid_size():
        return ttnn.CoreCoord(11, 10)

    @staticmethod
    def dram_grid_size():
        return ttnn.CoreCoord(8, 1)


_CAPTURE_MESH = _CaptureMesh()
ttnn.open_mesh_device = lambda *args, **kwargs: _CAPTURE_MESH
ttnn.close_mesh_device = lambda *args, **kwargs: None


class _CaptureCCL:
    """Constructor-only stand-in; captured all-reduces are identities."""

    def __init__(self, mesh_device):
        self.mesh_device = mesh_device
        self.barrier_semaphore_handles = [[], [], [None, None]]


def _host_tensor(shape, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        torch.empty(shape, dtype=torch.bfloat16),
        device=_CAPTURE_MESH,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


_DECODER = None


def make_inputs(_device):
    global _DECODER

    advisor_base._install_generic_rotary_handler()
    from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.multichip_decoder import MultiChipConfig, MultiChipDecoder
    from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import OptimizationConfig

    policy = MultiChipConfig(
        optimized=OptimizationConfig(
            attention_weight_dtype=ttnn.bfloat16,
            gate_up_weight_dtype=ttnn.bfloat16,
            down_weight_dtype=ttnn.bfloat16,
            decode_matmul_strategy="dram_sharded",
            qkv_cores=16,
            output_cores=16,
            gate_up_cores=16,
            down_cores=16,
            residual_cores=16,
        )
    )
    _DECODER = MultiChipDecoder(
        multichip_config=policy,
        mesh_device=_CAPTURE_MESH,
        tt_ccl=_CaptureCCL(_CAPTURE_MESH),
        layer_idx=16,
        batch=BATCH,
        max_cache_len=MAX_CACHE_LEN,
        hidden_size=HIDDEN,
        num_heads=LOCAL_Q_HEADS,
        num_kv_heads=LOCAL_KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=LOCAL_INTERMEDIATE,
        rms_norm_eps=1.0e-5,
        global_num_heads=32,
        global_num_kv_heads=8,
        global_intermediate_size=14336,
        input_norm=_host_tensor((HIDDEN,)),
        post_attention_norm=_host_tensor((HIDDEN,)),
        qkv_weight=_host_tensor((HIDDEN, (LOCAL_Q_HEADS + 2 * LOCAL_KV_HEADS) * HEAD_DIM)),
        output_weight=_host_tensor((HIDDEN // 4, HIDDEN)),
        gate_weight=_host_tensor((HIDDEN, LOCAL_INTERMEDIATE)),
        up_weight=_host_tensor((HIDDEN, LOCAL_INTERMEDIATE)),
        down_weight=_host_tensor((LOCAL_INTERMEDIATE, HIDDEN)),
        rotary_cos=_host_tensor((1, 1, MAX_CACHE_LEN, HEAD_DIM)),
        rotary_sin=_host_tensor((1, 1, MAX_CACHE_LEN, HEAD_DIM)),
        position_indices=_host_tensor((MAX_CACHE_LEN, BATCH), layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32),
    )
    _DECODER._all_reduce_partial = lambda partial, **_kwargs: partial

    hidden = _host_tensor((1, BATCH, 1, HIDDEN))
    cache_shape = (BATCH, LOCAL_KV_HEADS, MAX_CACHE_LEN, HEAD_DIM)
    return hidden, _host_tensor(cache_shape), _host_tensor(cache_shape)


def decode(hidden, key_cache, value_cache):
    return _DECODER.decode_forward(hidden, key_cache, value_cache, current_pos=CURRENT_POS)
