# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CCL device-kernel latency for TP=8 prefill: reduce_scatter + all_gather (Ring topology).

Shapes match pi0.5 prefill VLM (seq=768, hidden=2048, TP=8):
  RS input:  [1,1,768,2048]  — full hidden partial sum (down_proj / o_proj output)
  AG input:  [1,1,768,256]   — scattered slice (2048/tp=8), AG reassembles to [1,1,768,2048]

RS and AG are tested separately with their correct individual input shapes.

Run once and capture device-kernel duration via tracy:

  python -m tracy -p -r -n ccl_prefill_tp8 -o /tmp/tracy_ccl_tp8 \
    $(which pytest) models/experimental/pi0_5/tests/perf/test_ccl_prefill_tp8_perf.py -s

Parse ops_perf_results CSV: filter OP TYPE reduce_scatter / all_gather,
read DEVICE KERNEL DURATION [ns] — that is the authoritative prefill budget.

Requires 8 BH chips (24-31) + fabric.  Reset with tt-smi -glx_reset if needed.
"""

import os
from contextlib import contextmanager

import torch
import ttnn

from models.experimental.pi0_5.tt.tt_bh_glx.mesh_setup import open_prefill_tp4_mesh

# Expose chips 24-31 as devices 0-7 via TT_VISIBLE_DEVICES so open_prefill_tp4_mesh
# opens a 1×8 logical mesh over these physical chips under FABRIC_1D.
_VISIBLE_DEVICES = ",".join(str(i) for i in range(24, 32))

_TP = 8
_SEQ = 768
_HIDDEN = 2048

# RS input: full hidden partial sum; AG input: scattered slice after RS.
_RS_SHAPE = (1, 1, _SEQ, _HIDDEN)
_AG_SHAPE = (1, 1, _SEQ, _HIDDEN // _TP)  # [1,1,768,256]


@contextmanager
def _open_8chip_mesh():
    """Open chips 24-31 as a 1×8 TP=8 ring mesh under FABRIC_1D."""
    prev = os.environ.get("TT_VISIBLE_DEVICES")
    os.environ["TT_VISIBLE_DEVICES"] = _VISIBLE_DEVICES
    try:
        with open_prefill_tp4_mesh(tp=8, l1_small_size=24576) as mesh:
            yield mesh
    finally:
        if prev is None:
            os.environ.pop("TT_VISIBLE_DEVICES", None)
        else:
            os.environ["TT_VISIBLE_DEVICES"] = prev


# CCL kwargs matching stage_prefill_tp4.py exactly.
_RS = {
    "num_links": 2,
    "num_workers_per_link": 2,
    "memory_config": ttnn.L1_MEMORY_CONFIG,
    "num_buffers_per_channel": 2,
    "topology": ttnn.Topology.Ring,
}
_AG = {
    "num_links": 2,
    "num_workers_per_link": 4,
    "memory_config": ttnn.L1_MEMORY_CONFIG,
    "num_buffers_per_channel": 2,
    "topology": ttnn.Topology.Ring,
}


def _make_tensor(mesh, shape):
    return ttnn.from_torch(
        torch.randn(*shape).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def test_ccl_prefill_tp8_rs():
    """reduce_scatter at RS input [1,1,768,2048] on TP=8 Ring. Run under tracy for DK duration."""
    with _open_8chip_mesh() as mesh:
        x = _make_tensor(mesh, _RS_SHAPE)
        out = ttnn.reduce_scatter(x, dim=3, **_RS)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        print(f"\n  RS  input={list(_RS_SHAPE)}  output={list(out.shape)}  — check tracy for DK duration")


def test_ccl_prefill_tp8_ag():
    """all_gather at AG input [1,1,768,256] on TP=8 Ring. Run under tracy for DK duration."""
    with _open_8chip_mesh() as mesh:
        x = _make_tensor(mesh, _AG_SHAPE)
        out = ttnn.all_gather(x, dim=3, **_AG)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        print(f"\n  AG  input={list(_AG_SHAPE)}  output={list(out.shape)}  — check tracy for DK duration")
