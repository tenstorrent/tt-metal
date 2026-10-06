# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""DRAM and L1 budget check for MiniMax-H3 ref2va on a *mock* Wormhole 4x8 galaxy.

Every memory failure the pipeline can hit is raised on the host before dispatch: the allocator's
"Out of Memory" and the program's static circular-buffer check. With
``TT_METAL_MOCK_CLUSTER_DESC_PATH`` set, tt-metal keeps those paths and the kernel JIT and turns every
device read and write into a no-op, so this test replays serving's whole init warmup (the bucket
ladder walk that binds the DiT's buffers at every rung, then the VAE, audio and prompt-encoder warms
that compile every remaining program) on a box with no Tenstorrent hardware.

It is deliberately mock-only: it skips unless the mock descriptor is set, so it never spends galaxy
time. Point the descriptor at
``tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_cluster_desc/6u_cluster_desc.yaml`` (the
UMD example 6U descriptor derives a 32x1 system mesh and cannot open 4x8). Weights are still
required, from ``MINIMAX_H3_MODEL_PATH`` or the HuggingFace cache, and ``TT_DIT_CACHE_DIR`` makes
the per-rung stage reloads cheap. Set ``MINIMAX_H3_DRAM_PROBE=1`` in the environment to get the
per-owner DRAM attribution at every checkpoint; the probe reads that variable at import.

What the mock cannot see: per-program kernel-binary DRAM buffers are not allocated in mock mode, so
its headroom reads about 60 MB per device higher than silicon for a fully warmed H3, and nothing
data-dependent or timing-dependent runs. A pass here means every rung's allocations fit the
allocator and every warmed program's static circular buffers fit L1; a DRAM failure names the rungs
that do not fit, an L1 failure raises from the program that clashes.
"""

from __future__ import annotations

import os

import pytest
from loguru import logger

import ttnn

from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
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


def _dram_line(mesh_device: ttnn.MeshDevice, label: str) -> str:
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    gib = 1024**3
    return (
        f"DRAM[{label}] per device: allocated {view.total_bytes_allocated_per_bank * view.num_banks / gib:.2f} GiB, "
        f"free {view.total_bytes_free_per_bank * view.num_banks / gib:.2f} GiB, "
        f"largest contiguous {view.largest_contiguous_bytes_free_per_bank / 2**20:.1f} MiB/bank"
    )


@pytest.mark.timeout(7200)  # a cold kernel cache compiles every H3 program: ~55 min; warm: ~10 min
@pytest.mark.parametrize(("mesh_device", "device_params"), MESHES, indirect=["mesh_device", "device_params"])
def test_ref2va_warmup_fits_on_mock(mesh_device, monkeypatch):
    """Run the ref2va preset's full init warmup exactly as serving does and require every rung to bind."""
    assert ttnn.get_arch_name() == "wormhole_b0", f"this check targets Wormhole; descriptor {MOCK_CLUSTER_DESC}"
    assert tuple(mesh_device.shape) == (4, 8), tuple(mesh_device.shape)

    # Report-and-skip rungs that run out of DRAM instead of raising, so one walk names every unfittable rung.
    monkeypatch.setenv("MINIMAX_H3_WARMUP_SKIP_OOM_RUNGS", "1")

    # Serving's constructor arguments: the default yuv420 output keeps the VAE decode warm in the walk.
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir("transformer_ref"),
        task="ref2va",
        warmup=False,
    )
    logger.info(_dram_line(mesh_device, "after construction"))

    # Serving's warmup entry point, with the production warmup requests for every rung of this preset:
    # the ladder walk first, then the VAE, audio and prompt-encoder warms (compile-only, so they catch
    # static circular-buffer overflows), then trace capture (a no-op on the untraced Wormhole preset).
    pipeline._warmup_on_init()

    logger.info(_dram_line(mesh_device, "after the full warmup"))
    ladder = sorted(pipeline.bucket_ladder, reverse=True)
    assert not pipeline.unfittable_rungs, (
        f"rungs that do not fit in DRAM on the mock {tuple(mesh_device.shape)} mesh: "
        f"{sorted(pipeline.unfittable_rungs)} of ladder {ladder}; "
        "run with MINIMAX_H3_DRAM_PROBE=1 for the per-owner attribution at the failing rung"
    )
    logger.info(f"all {len(ladder)} rungs bound ({ladder[0]} .. {ladder[-1]} rows) on the mock mesh")
