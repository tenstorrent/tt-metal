# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CCL device-kernel latency for TP=8 prefill: reduce_scatter + all_gather (Ring topology).

Shapes match pi0.5 prefill MLP (seq=768, hidden=1024, mlp_dim=4096, TP=8):
  hidden   [1,1,768,1024]  — down-proj all_reduce target
  mlp_mid  [1,1,768,4096]  — gate/up scatter target

Run once and capture device-kernel duration via tracy:

  python -m tracy -p -r -n ccl_prefill_tp8 -o /tmp/tracy_ccl_tp8 \
    pytest models/experimental/pi0_5/tests/perf/test_ccl_prefill_tp8_perf.py -s

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


_SHAPES = {
    "hidden": (1, 1, 768, 1024),
    "mlp_mid": (1, 1, 768, 4096),
}

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


def _run_all(shape_name):
    shape = _SHAPES[shape_name]
    scatter_dim = len(shape) - 1
    with _open_8chip_mesh() as mesh:
        x = ttnn.from_torch(
            torch.randn(*shape).bfloat16(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        scattered = ttnn.reduce_scatter(x, scatter_dim, **_RS)
        out = ttnn.all_gather(scattered, scatter_dim, **_AG)
        ttnn.synchronize_device(mesh)
        ttnn.ReadDeviceProfiler(mesh)
        print(f"\n  shape={shape_name} {list(shape)}  RS+AG done — check tracy for device kernel duration")


def test_ccl_prefill_tp8_hidden():
    """RS + AG at hidden shape [1,1,768,1024] on TP=8 Ring. Run under tracy for DK duration."""
    _run_all("hidden")


def test_ccl_prefill_tp8_mlp_mid():
    """RS + AG at MLP-intermediate shape [1,1,768,4096] on TP=8 Ring. Run under tracy for DK duration."""
    _run_all("mlp_mid")
