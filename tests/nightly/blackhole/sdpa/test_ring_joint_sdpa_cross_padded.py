"""Padded V2A cross configs so the large transport-bound shapes RUN on the ring8 (SP=8) mesh.

The committed cross test skips stage2_1080p_6s / _14s on SP=8 because their global q/kv dims
are not multiples of SP*32 (=256). Here we round q_global/kv_global UP to the nearest SP*32
while keeping the true logical_n, so the op attends only real keys (PCC unchanged) and we can
measure device kernel duration for the split-forward optimization on the exposed-transport case.
"""
import pytest

from tests.nightly.blackhole.sdpa.test_ring_joint_sdpa import (
    MESH_CONFIG,
    Topology,
    run_ring_joint_sdpa_cross,
    CROSS_PCC_THRESHOLD,
    CROSS_RMSE_THRESHOLD,
)


def _round_up(x, m):
    return ((x + m - 1) // m) * m


# Round to the SP=8 tile-shard granularity (8*32=256). q/kv padded; logical_n = true key count.
_SP = 8
_M = _SP * 32
# (id, nhq_total, head_dim, q_true, kv_true, logical_n)
_RAW = [
    ("ltx_v2a_stage1_6s", 32, 64, 256, 9728, 9690),
    ("ltx_v2a_stage2_1080p_6s", 32, 64, 256, 38784, 38760),
    ("ltx_v2a_stage2_1080p_14s", 32, 64, 384, 87808, 87720),
]
PADDED_CONFIGS = [
    pytest.param(nhq, d, _round_up(q, _M), _round_up(kv, _M), logical_n, id=name)
    for (name, nhq, d, q, kv, logical_n) in _RAW
]


@pytest.mark.parametrize("nhq_total, head_dim, q_global, kv_global, logical_n", PADDED_CONFIGS)
def test_cross_padded_ring8(nhq_total, head_dim, q_global, kv_global, logical_n):
    """is_cross ring SDPA at padded V2A shapes on the ring8 mesh (split-forward active)."""
    run_ring_joint_sdpa_cross(
        MESH_CONFIG,
        nhq_total,
        head_dim,
        q_global,
        kv_global,
        logical_n,
        q_chunk_sizes=[64, 128],
        tp_size=4,
        sp_size=8,
        topology=Topology.Ring,
        pcc_threshold=CROSS_PCC_THRESHOLD,
        rmse_threshold=CROSS_RMSE_THRESHOLD,
    )
