# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Safe eager profile for the SP=8 affine rank-release KDA protocol.

LoudBox has eight chips, so this is not a literal SP=8 x TP=4 layer.  It uses
the exact TP=4-rank state payload (eight local 128x128 heads) and global
T=5120, but retains a TP=1 output projection.  The resulting interval is a
scheduler baseline only: host stage fences and the missing TP=4 output CCL
must be removed before using it as a Galaxy latency prediction.
"""

from __future__ import annotations

import os

import pytest
import torch
from tracy import signpost

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.experimental.kimi_delta_attention.config import KDAConfig
from models.experimental.kimi_delta_attention.tests.test_factory import random_weights
from models.experimental.kimi_delta_attention.tt.sp_layer import SP8AffineTP1KimiDeltaAttention


pytestmark = [
    run_for_blackhole(),
    pytest.mark.perf,
    pytest.mark.timeout(0),
    pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True),
    pytest.mark.parametrize(
        "device_params",
        [
            {
                "l1_small_size": 24576,
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "trace_region_size": 256 * 1024 * 1024,
            }
        ],
        indirect=True,
    ),
]


def test_sp8_rank_release_eager_perf(mesh_device: ttnn.MeshDevice, monkeypatch: pytest.MonkeyPatch) -> None:
    """Measure safe rank release with the production TP4-rank state payload."""
    monkeypatch.setenv("KDA_SP8_PIPELINED_HANDOFFS", "1")
    rank_release = os.getenv("PERF_RANK_RELEASE", "1") == "1"
    if rank_release:
        monkeypatch.setenv("KDA_SP8_RANK_RELEASE", "1")
    else:
        monkeypatch.delenv("KDA_SP8_RANK_RELEASE", raising=False)
    config = KDAConfig(
        hidden_size=2304,
        num_heads=8,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel_size=4,
        norm_eps=1e-5,
        chunk_size=32,
    )
    sequence = int(os.getenv("PERF_SEQ", "5120"))
    if sequence % (8 * 128):
        raise ValueError(f"PERF_SEQ must be divisible by 1024, got {sequence}")
    layer = SP8AffineTP1KimiDeltaAttention(mesh_device, config, random_weights(config))
    layer.reset_state(batch_size=1)
    hidden = torch.randn(1, sequence, config.hidden_size, generator=torch.Generator().manual_seed(9217)).to(
        torch.bfloat16
    )
    span = sequence // 8
    spans = tuple(
        ttnn.from_torch(
            hidden[:, rank * span : (rank + 1) * span],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        for rank, device in enumerate(layer.span_devices)
    )
    warm_outputs = layer.forward(*spans)
    for output in warm_outputs:
        ttnn.deallocate(output)
    repetitions = int(os.getenv("PERF_REPS", "3"))
    outputs: list[tuple[ttnn.Tensor, ...]] = []
    mode = "rank_release" if rank_release else "eager"
    signpost(header=f"sp8_{mode}_eager_start")
    for _ in range(repetitions):
        outputs.append(layer.forward(*spans))
    signpost(header=f"sp8_{mode}_eager_stop")
    for rank_outputs in outputs:
        for output in rank_outputs:
            ttnn.deallocate(output)
