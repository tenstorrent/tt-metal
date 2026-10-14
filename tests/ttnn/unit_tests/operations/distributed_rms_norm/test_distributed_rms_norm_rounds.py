# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 — R2 multi-round exchange (interleaved). Pins the regime's knobs beyond their defaults, through the
golden helper (builds x / gamma, dispatches, checks every device):

  EXCHANGE_ROUNDS  2 (default) / 4      more rounds than blocks in flight: rotating local-arrival counters wrap
  ROUND_LOOKAHEAD  0 / 1 (default) / 2  0 = the serial R1 schedule per round; 2 = three counters, x held 2 blocks
  GAMMA_PRESCALE   False                the interleaved fused-gamma normalize (the default plan now picks enough
                                        rounds for the prescale path to fit, so nothing else reaches it)
Shapes: (1,1,4096,1024) has row groups of 3 and 4 tile rows (ragged last round; a group sits out a round);
(1,1,640,1792) is the perf focus; (1,1,100,512) is h_non_aligned with 4 tile rows.
"""

import pytest
import ttnn

from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
from ttnn.operations.ccl import Topology
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

from .test_distributed_rms_norm import _FABRIC, _MESH

F32, BF16 = ttnn.float32, ttnn.bfloat16


def _run(mesh_device, shape, dtype, gamma_dtype, cluster_axis=1, num_links=2):
    if tuple(mesh_device.shape)[cluster_axis] < 2:
        pytest.skip(f"cluster_axis={cluster_axis} has size 1 on mesh {tuple(mesh_device.shape)}")
    run_distributed_rms_norm(
        (shape,),
        mesh_device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=True,
        gamma_mode="gamma" if gamma_dtype is not None else "no_gamma",
        gamma_dtype=gamma_dtype if gamma_dtype is not None else "none",
        memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,
        cluster_axis=cluster_axis,
        topology=Topology.Linear,
        num_links=num_links,
    )


CASES = [
    # (rounds, lookahead, prescale, shape, dtype, gamma dtype)
    (2, 1, True, (1, 1, 4096, 1024), BF16, None),
    (4, 1, True, (1, 1, 4096, 1024), BF16, BF16),
    (4, 1, True, (1, 1, 4096, 1024), F32, None),
    (4, 2, True, (1, 1, 4096, 1024), BF16, None),
    (4, 2, True, (1, 1, 100, 512), F32, F32),
    (4, 0, True, (1, 1, 640, 1792), BF16, BF16),
    (2, 1, False, (1, 1, 640, 1792), BF16, BF16),
    (4, 1, False, (1, 1, 4096, 1024), F32, F32),
]


@_MESH
@_FABRIC
@pytest.mark.parametrize(
    "rounds,lookahead,prescale,shape,dtype,gamma_dtype",
    CASES,
    ids=[f"R{r}-LA{la}-{'pre' if p else 'fused'}-{'x'.join(map(str, s))}-{d}-{g}" for r, la, p, s, d, g in CASES],
)
def test_rounds(mesh_device, monkeypatch, rounds, lookahead, prescale, shape, dtype, gamma_dtype):
    monkeypatch.setattr(pd, "EXCHANGE_ROUNDS", rounds)
    monkeypatch.setattr(pd, "ROUND_LOOKAHEAD", lookahead)
    monkeypatch.setattr(pd, "LOCAL_ARRIVAL_SEMS", 1 + lookahead)
    monkeypatch.setattr(pd, "SEM_LOCAL_COMBINE", pd.SEM_LOCAL_ARRIVALS + 1 + lookahead)
    monkeypatch.setattr(pd, "SEM_STATS_READY_BASE", pd.SEM_LOCAL_ARRIVALS + 2 + lookahead)
    monkeypatch.setattr(pd, "GAMMA_PRESCALE", prescale)
    for _ in range(2):  # the second call re-uses the re-armed counters and scratch of the first
        _run(mesh_device, shape, dtype, gamma_dtype)


@_MESH
@_FABRIC
def test_rounds_lift_p_resident(mesh_device):
    """(2, 2048, 4096) bf16 + gamma: prescale did not fit in one/two rounds (fused fallback); the plan now takes
    enough rounds for cb_x_resident + cb_xg (1 + lookahead one-row blocks) to fit."""
    mesh = tuple(mesh_device.shape)
    if mesh[1] < 2:
        pytest.skip("cluster_axis 1 has size 1")
    _run(mesh_device, (2, 2048, 4096), BF16, BF16)
