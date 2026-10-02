# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Guards the Quasar TILE-materialization path used by the graph_ops harness (graph_case.build_tensor).

On Quasar, ttnn.from_torch(layout=TILE) routes to the MAINLINE device tilize (tilize_metal2.cpp). That
path faults on wide-short (1-tile-tall) tensors -- Neo0TRISC2 MEM_READ_NO_RESPONSE -- and, worse, leaks
state that faults a LATER tilize in the same run: test_linear_small_grid passes alone but hangs/faults when
run after test_embedding* in one pytest invocation (TT_METAL_FORCE_JIT_COMPILE=1 does not help, so it is
not the on-disk kernel cache; it is cross-op device/sim state). graph_case.build_tensor was changed to
materialize TILE floats via ttnn.experimental.quasar.tilize (RM upload + quasar tilize -- the path the
model uses), which is stable.

This test exercises that Quasar-safe path directly, as a SEQUENCE of differently-shaped tilizes in one
device session (tall then several wide-short, mirroring embedding-then-linear), round-tripping each back
through to_torch and PCC-checking it. It reproduces the cross-op ordering that broke the mainline path and
confirms quasar.tilize stays correct across repeated back-to-back invocations.

Run (Quasar sim, 2-node emulator, SLOW dispatch):
    TTSIM_QSR_TC_LEGACY_TRUNCATION_ALIAS=0 TT_METAL_SIMULATOR=~/sim/libttsim.so TT_METAL_SLOW_DISPATCH_MODE=1 \
        TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE="3,2" MESH_DEVICE=N150 \
        pytest models/experimental/llama32_1b_quasar/tests/debug_ops/test_quasar_tilize_sequence.py
"""

import pytest
import torch
from loguru import logger

import ttnn

# Tall first (like an embedding-weight tilize), then wide-short ones (the concat/linear inputs that fault
# the mainline path), then wide. All 1-batch, bf16. Kept small so the sim run is quick.
SHAPES = [
    (1, 1, 512, 64),  # tall-ish
    (1, 1, 32, 512),  # wide-short: from_torch(TILE) faults here on Quasar
    (1, 1, 32, 256),
    (1, 1, 32, 512),  # repeat wide-short back-to-back
    (1, 1, 64, 128),
    (1, 1, 32, 4352),  # wide (concat output width)
]


def _tile_bf16_dram(t_bf16, mesh_device):
    rm = ttnn.from_torch(
        t_bf16,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(mesh_device),
    )
    qt = getattr(getattr(ttnn.experimental, "quasar", None), "tilize", None)
    return (qt or ttnn.tilize)(rm, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)


def _pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if torch.allclose(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.timeout(3600)
def test_quasar_tilize_sequence(mesh_device):
    """A back-to-back sequence of differently-shaped quasar.tilize calls in one session: each must run
    (no MEM_READ_NO_RESPONSE) and round-trip correctly, and later tilizes must not be corrupted by earlier
    ones."""
    torch.manual_seed(0)
    for i, shape in enumerate(SHAPES):
        ref = torch.randn(*shape, dtype=torch.bfloat16)
        tt = _tile_bf16_dram(ref, mesh_device)
        ttnn.synchronize_device(mesh_device)
        back = ttnn.to_torch(tt)  # untilizes back to row-major
        assert tuple(back.shape) == shape, f"[{i}] shape {tuple(back.shape)} != {shape}"
        assert torch.isfinite(back.float()).all(), f"[{i}] shape {shape}: non-finite after tilize round-trip"
        pcc = _pcc(back, ref)
        logger.info(f"[tilize-seq {i}] shape={shape} PCC={pcc:.5f}")
        assert pcc > 0.999, f"[{i}] shape {shape}: tilize round-trip PCC {pcc}"
