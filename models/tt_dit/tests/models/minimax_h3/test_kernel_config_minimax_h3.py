# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 kernel-config canary: the vision tower's ring attention must fit the deployed ring, with margin.

The Qwen3-VL vision tower's `ring_joint_sdpa` is the largest program MiniMax-H3 compiles and has overflowed
the Tensix kernel-config ring twice on Wormhole (see `MINIMAX_H3_KERNEL_CONFIG_RING_BYTES`), each time from an
LLK change outside tt_dit. CI's "Minimax H3 Memory Test" runs this first: it compiles just that program, for
every shard-size variant the pipeline reaches, in about a minute and without weights, so a regression fails
there before the 2.5 h warmup (test_mock_oom_minimax_h3.py), which covers every other program.

The mesh opens with the deployed ring minus `GUARD_BYTES`, so this fails while deployments still have that much
left: a red run here is a warning, and the fix is to shrink the program or raise the ring in
`pipeline_minimax_h3.py`. The TT_FATAL lists the size per RISC.

Run with TT_METAL_MOCK_CLUSTER_DESC_PATH set to a 4x8 cluster descriptor (the mock needs no hardware, and its
architecture comes from the descriptor) and TT_METAL_WATCHER unset; without the mock every case skips. CI uses
wormhole/6u_cluster_desc/6u_cluster_desc.yaml; blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml
runs the Blackhole row (both under tt_metal/third_party/tt-cluster-descriptors/). Not covered: the 4x32 quad
(SP=32, a multi-host descriptor) and the windowed vision attention.
"""

from __future__ import annotations

import os

import pytest
import torch

import ttnn
from models.tt_dit.encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionAttention, resolve_vision_parallel
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import (
    MINIMAX_H3_KERNEL_CONFIG_RING_BYTES,
    MINIMAX_H3_VISION_PATCH_LADDER,
    resolve_mesh_preset,
)
from models.tt_dit.tests.encoders.qwen3vl.common import HIDDEN_SIZE, NUM_HEADS
from models.tt_dit.tests.models.minimax_h3.common import MESH_4X8_RING, MESH_4X8_RING_WH
from models.tt_dit.utils.tensor import bf16_tensor

pytestmark = pytest.mark.skipif(
    not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"),
    reason="mock-device-only test: set TT_METAL_MOCK_CLUSTER_DESC_PATH to a 4x8 cluster descriptor",
)

GUARD_BYTES = 1024

# Patch counts whose per-device shard (patches / SP=8) is 32..480 rows: the shard sets the flash chunk
# (min(shard, 128)) and whether the last chunk is partial, which is what changes the compiled kernels. Partial
# chunks of every remainder (32, 96) after one, two and three full ones, the full-chunk case and the sub-chunk
# shapes. The pipeline pads patches to SP * 32, so these are all reachable.
PATCHES = [256, 768, 1280, 1792, 2304, 2816, 3072, 3840]


@pytest.fixture
def device_params(device_params):
    """Open with the deployed ring less the guard band (the `worker_l1_size` that ring leaves)."""
    ring = MINIMAX_H3_KERNEL_CONFIG_RING_BYTES - GUARD_BYTES
    device_params["worker_l1_size"] = ttnn.device.get_max_worker_l1_unreserved_size() - ring
    return device_params


@pytest.mark.parametrize(
    ("mesh_device", "device_params"), [MESH_4X8_RING_WH, MESH_4X8_RING], indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("patches", PATCHES)
def test_vision_ring_attention_fits_kernel_config_ring(mesh_device, patches):
    shape = tuple(mesh_device.shape)
    preset = resolve_mesh_preset(shape)
    tp_axis, sp_axis = preset["tp_axis"], preset["sp_axis"]
    ccl = CCLManager(mesh_device=mesh_device, num_links=preset["num_links"], topology=preset["topology"])
    parallel = resolve_vision_parallel(
        mesh_device,
        EncoderParallelConfig(
            tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=shape[tp_axis]),
            sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=shape[sp_axis]),
        ),
        ccl,
    )
    # As `MiniMaxH3Pipeline._prepare_vision_tower` builds it; the gather capacity only sizes buffers.
    attn = Qwen3VlVisionAttention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        mesh_device=mesh_device,
        parallel=parallel,
        kv_gather_capacity=MINIMAX_H3_VISION_PATCH_LADDER[-1],
    )
    local_seq = patches // shape[sp_axis]
    # The mock ignores data: only shapes and configs reach the compiled program.
    x = torch.zeros(1, attn.num_local_heads, patches, attn.padded_head_dim)
    q, k, v = (bf16_tensor(x, device=mesh_device, mesh_axis=sp_axis, shard_dim=2) for _ in range(3))
    out = attn._ring_attention(q, k, v, local_seq)
    assert tuple(out.shape)[-2] == local_seq
