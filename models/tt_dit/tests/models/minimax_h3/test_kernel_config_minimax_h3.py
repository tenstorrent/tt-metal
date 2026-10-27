# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Fast check that the vision tower's ring attention fits the kernel-config ring.

It is the largest program MiniMax-H3 compiles and has overflowed the ring twice on Wormhole, each time from an
LLK change outside tt_dit. This compiles only that program, for every per-device shard size the pipeline
reaches, with the same mesh params as the gate tests, on a mock mesh: about a minute, no weights. CI runs it
at the start of the Minimax H3 Memory Test.

Run with TT_METAL_MOCK_CLUSTER_DESC_PATH set to a 4x8 cluster descriptor and TT_METAL_WATCHER unset, e.g.
tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_cluster_desc/6u_cluster_desc.yaml; without it every
case skips.
"""

import os

import pytest
import torch

from models.tt_dit.encoders.qwen3vl.vision_qwen3vl import Qwen3VlVisionAttention, resolve_vision_parallel
from models.tt_dit.parallel.config import EncoderParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.pipelines.minimax_h3.pipeline_minimax_h3 import MINIMAX_H3_VISION_PATCH_LADDER, resolve_mesh_preset
from models.tt_dit.tests.encoders.qwen3vl.common import HIDDEN_SIZE, NUM_HEADS
from models.tt_dit.tests.models.minimax_h3.common import MESH_4X8_RING, MESH_4X8_RING_WH
from models.tt_dit.utils.tensor import bf16_tensor

pytestmark = pytest.mark.skipif(
    not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"),
    reason="mock-device-only test: set TT_METAL_MOCK_CLUSTER_DESC_PATH to a 4x8 cluster descriptor",
)

# Per-device shards of 32..480 rows (patches / SP=8): partial and full 128-row flash chunks, which are what
# change the compiled kernels.
PATCHES = [256, 768, 1280, 1792, 2304, 2816, 3072, 3840]


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
    attn = Qwen3VlVisionAttention(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        mesh_device=mesh_device,
        parallel=parallel,
        kv_gather_capacity=MINIMAX_H3_VISION_PATCH_LADDER[-1],
    )
    local_seq = patches // shape[sp_axis]
    x = torch.zeros(1, attn.num_local_heads, patches, attn.padded_head_dim)  # the mock ignores data
    q, k, v = (bf16_tensor(x, device=mesh_device, mesh_axis=sp_axis, shard_dim=2) for _ in range(3))
    out = attn._ring_attention(q, k, v, local_seq)
    assert tuple(out.shape)[-2] == local_seq
