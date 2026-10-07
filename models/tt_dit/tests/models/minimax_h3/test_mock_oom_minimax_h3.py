# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Catches MiniMax-H3 ref2va L1 and DRAM failures without a Wormhole galaxy.

With TT_METAL_MOCK_CLUSTER_DESC_PATH set, tt-metal opens the cluster descriptor as a mock 4x8 mesh with the
galaxy's DRAM and L1 geometry, keeps the allocator, the circular-buffer validation and the kernel JIT, and
turns every device read and write into a no-op. The test builds the ref2va pipeline with serving's defaults
and runs its full init warmup. Both memory failures the pipeline can hit are raised on the host, so the mock
reproduces them with silicon's error text:
  - DRAM out of memory: raised by the allocator when a rung's forced request does not fit; it surfaces in the
    ladder walk, at the first rung that does not bind.
  - L1 overflow: raised by the static circular-buffer check when a program's CBs clash with L1 buffers or
    exceed L1; it surfaces in the walk or in the VAE, audio and prompt-encoder warms that compile the rest.
Either exception fails the test at once. A pass is the warmup completing. The mock cannot see hangs, timing
or data-dependent behaviour, and it allocates no kernel-binary DRAM, so its headroom reads about 60 MB per
device higher than silicon.

MINIMAX_H3_DRAM_PROBE=1 logs the per-owner DRAM attribution at each checkpoint. A large ccl_ping_pong or
pipeline.* owner names a leak; a large unattributed total points at transients and fragmentation.

Environment:
  - unset TT_METAL_WATCHER: CI exports it, and on the mock its device reads come back 0 and it aborts at
    mesh open.
  - Weights resolve from MINIMAX_H3_MODEL_PATH or $HF_HOME/hub; the resolver reads HF_HOME, not HF_HUB_CACHE.
    With TT_DIT_ALLOW_HF_DOWNLOAD=1 and HF_HUB_OFFLINE unset it fetches the ref2va set (transformer_ref,
    text_encoder, vae, audio_vae, about 144 GB); all four are requested here because an explicit directory
    is checked, not completed.
  - MINIMAX_H3_REQUIRE_WEIGHTS=1 fails instead of skipping when the weights can neither be found nor fetched.
  - The test skips unless the mock descriptor is set, so it never takes a galaxy.

CI ("Minimax H3 Ref2VA Memory Test" in tests/pipeline_reorg/models_unit_tests.yaml, wh_n150, tier 1) runs
from a cold kernel cache every time, and nearly all of the time is kernel JIT on the host CPU:
  - galaxy host: about 55 min cold, about 10 min warm.
  - N150 cloud VM: about 2.5 h cold (snapshot download 3 min, construction 7, ladder walk 44, VAE warm 8,
    audio warm 30, prompt-encoder warm about 40).
Hence the 3 h pytest marker and the 240 min job timeout. The shared cloud MLPerf NFS the N150 VMs mount at
/mnt/MLPerf/huggingface has held the snapshot since 2026-10-07; populating a mount needs a dispatch with
mlperf-read-only=false.
"""

from __future__ import annotations

import os

import pytest
from loguru import logger

import ttnn

from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.weights_minimax_h3 import resolve_weights_dir
from .common import MESH_4X8_RING_WH
from .common_av import weights_dir

MOCK_CLUSTER_DESC = os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")
pytestmark = pytest.mark.skipif(
    not MOCK_CLUSTER_DESC,
    reason="mock-device-only test: set TT_METAL_MOCK_CLUSTER_DESC_PATH to a Wormhole 6U cluster descriptor",
)

# Same L1_SMALL the ref2va gates open with (test_pipeline_ref2va_minimax_h3.MESHES, test_performance_minimax_h3
# `_REF2VA_L1_SMALL`): ref2va's taps=3 video encoder clashes with a larger pool. The mock reproduces that clash
# too: at 65536 the vision tower's windowed SDPA fails its static circular-buffer check 23 s after the mesh opens.
_L1_SMALL = 16384
MESHES = [
    pytest.param(shape, {**params, "l1_small_size": _L1_SMALL}, id=param.id, marks=param.marks)
    for param in [MESH_4X8_RING_WH]
    for shape, params in [param.values]
]


# Every partition the ref2va pipeline loads (create_pipeline resolves the same set). Resolving them all here
# matters with TT_DIT_ALLOW_HF_DOWNLOAD=1: the download fetches exactly the requested partitions, and the
# directory is then handed to create_pipeline as an explicit path, which is checked, not completed.
_REF2VA_PARTITIONS = ("transformer_ref", "text_encoder", "vae", "audio_vae")


def _weights_dir():
    """The ref2va snapshot. Locally a missing snapshot skips; with MINIMAX_H3_REQUIRE_WEIGHTS=1 (the CI entry)
    it raises WeightsNotFoundError instead, so a runner without the weights mount fails the gate rather than
    turning it green by skipping."""
    if os.environ.get("MINIMAX_H3_REQUIRE_WEIGHTS") == "1":
        return resolve_weights_dir(*_REF2VA_PARTITIONS)
    return weights_dir(*_REF2VA_PARTITIONS)


def _dram_line(mesh_device: ttnn.MeshDevice, label: str) -> str:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    gib = 1024**3
    return (
        f"DRAM[{label}] per device: allocated {view.total_bytes_allocated_per_bank * view.num_banks / gib:.2f} GiB, "
        f"free {view.total_bytes_free_per_bank * view.num_banks / gib:.2f} GiB, "
        f"largest contiguous {view.largest_contiguous_bytes_free_per_bank / 2**20:.1f} MiB/bank"
    )


# A cold kernel cache compiles every H3 program: ~55 min on a galaxy host, ~2.5 h on an N150 cloud VM (ladder walk
# 44 min, VAE warm 8, audio warm 30, prompt-encoder warm ~40); warm: ~10 min on the galaxy host.
@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESHES, indirect=["mesh_device", "device_params"])
def test_ref2va_warmup_fits_on_mock(mesh_device):
    """Run the ref2va preset's full init warmup exactly as serving does; any OOM or L1 clash fails it at once."""
    assert ttnn.get_arch_name() == "wormhole_b0", f"this check targets Wormhole; descriptor {MOCK_CLUSTER_DESC}"
    assert tuple(mesh_device.shape) == (4, 8), tuple(mesh_device.shape)

    # Serving's constructor arguments: the default yuv420 output keeps the VAE decode warm in the walk.
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=_weights_dir(),
        task="ref2va",
        warmup=False,
    )
    logger.info(_dram_line(mesh_device, "after construction"))

    # DEMO (do not merge): hold 1.5 GiB of DRAM per device across the walk, as if the DiT's resident
    # buffers had grown by that much, so the largest rungs no longer fit and the gate names them.
    import torch

    filler = ttnn.from_torch(  # noqa: F841  (kept alive for the whole walk)
        torch.zeros((24576, 32768), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    logger.info(_dram_line(mesh_device, "after the 1.5 GiB demo filler"))

    # Serving's warmup entry point, with the production warmup requests for every rung of this preset:
    # the ladder walk first, then the VAE, audio and prompt-encoder warms (compile-only, so they catch
    # static circular-buffer overflows), then trace capture (a no-op on the untraced Wormhole preset).
    pipeline._warmup_on_init()

    logger.info(_dram_line(mesh_device, "after the full warmup"))
    ladder = sorted(pipeline.bucket_ladder, reverse=True)
    logger.info(f"all {len(ladder)} rungs bound ({ladder[0]} .. {ladder[-1]} rows) on the mock mesh")
