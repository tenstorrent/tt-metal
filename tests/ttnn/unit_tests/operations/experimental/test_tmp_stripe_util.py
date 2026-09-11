# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY investigation harness -- not for commit.

Measures the same bounded TP gather twice: once as an ordinary contiguous shard (one stripe, which
leaves the bank-owned schedule eligible) and once with input_stripe_size set (several stripes, which
disqualifies bank ownership -- see the `single_stripe` gate in high_bw_all_gather_unicast_factory.cpp).

Both arms move the SAME bytes: with one stripe gathered_dim_size narrows the front of the shard, with
several it bounds whole stripes, and both land at gathered_dim_size/tp active rows per device. So the
median duration is directly comparable and the delta is the cost of losing the fast schedule.
"""

import json
import os
import statistics
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from tests.ttnn.unit_tests.operations.experimental.test_high_bw_all_gather import (
    _device_params,
    _make_tensor,
    _profile_high_bw_all_gather,
)

_OUT = Path(os.environ.get("STRIPE_UTIL_OUT", "/tmp/stripe_util_results.json"))

# Sized so the NON-striped arm clears bank_owned_min_link_bytes (1.5 MB/link) and stays under
# high_parallelism_min_link_bytes (32 MB/link), i.e. the 4-worker bank-owned tier.
SP, TP = 8, 4
ROWS_DEV = 512  # one chip's region of each chunk == the stripe extent
N_CHUNKS = 8
WIDTH = 576  # bf16 row -> 1152 B page, same page size as the GLM sparse path
NUM_SLOTS = 2
SLOT = 1
# Two operating points. 6 chunks puts the CONTIGUOUS arm over bank_owned_min_link_bytes (1.5 MB/link),
# so it gets bank ownership AND the 4-worker tier. 1 chunk sits below it, so both arms run 2 workers
# and the delta isolates the bank-owned schedule by itself.
_ACTIVE_CHUNK_CASES = [6, 1]


@run_for_blackhole("high_bw_all_gather requires Blackhole fabric")
@pytest.mark.parametrize("device_params", [_device_params(ttnn.FabricConfig.FABRIC_2D)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize("active_chunks", _ACTIVE_CHUNK_CASES, ids=["big_6chunks", "small_1chunk"])
@pytest.mark.parametrize("striped", [False, True], ids=["contiguous", "striped"])
def test_tmp_stripe_util(mesh_device, striped, active_chunks):
    sp, tp = tuple(mesh_device.shape)
    assert (sp, tp) == (SP, TP)
    chunk_global = ROWS_DEV * sp * tp
    rows_sp = chunk_global // sp

    torch.manual_seed(0)
    host = torch.rand((NUM_SLOTS, N_CHUNKS, chunk_global, WIDTH), dtype=torch.bfloat16)

    # Block-cyclic assignment: device d holds rows [d*ROWS_DEV, +ROWS_DEV) of EVERY chunk.
    shards = [
        host[:, :, d * ROWS_DEV : (d + 1) * ROWS_DEV].reshape(NUM_SLOTS, 1, N_CHUNKS * ROWS_DEV, WIDTH)
        for d in range(sp * tp)
    ]
    source = _make_tensor(
        mesh_device,
        torch.cat(shards, dim=2),
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
        ttnn.ShardTensorToMesh(mesh_device, dim=2),
    )
    dist_shape = ttnn.MeshShape(sp, tp)
    source.update_tensor_topology(
        ttnn.TensorTopology(
            dist_shape,
            [ttnn.PlacementShard(2), ttnn.PlacementShard(2)],
            [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist_shape)],
        )
    )

    sp_only = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(sp, tp), dims=[2, None])
    # Per DEVICE the output must be the worst-case gathered shape (N_CHUNKS*rows_sp rows), so the host
    # tensor carries sp times that and the sp_only mapper hands each device its own full-size slot.
    out_host = torch.zeros((1, 1, N_CHUNKS * rows_sp * sp, WIDTH), dtype=torch.bfloat16)
    output = _make_tensor(mesh_device, out_host, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, sp_only)

    gathered_dim_size = active_chunks * tp * ROWS_DEV
    stripe_kwargs = {"input_stripe_size": ROWS_DEV} if striped else {}

    def run():
        return ttnn.experimental.high_bw_all_gather(
            source,
            dim=2,
            output_tensor=output,
            num_links=2,
            cluster_axis=1,
            input_batch_index=SLOT,
            gathered_dim_size=gathered_dim_size,
            **stripe_kwargs,
        )

    # Prime the cached program (and prove the slot/extent patch on a cache hit), then measure.
    run()
    ttnn.synchronize_device(mesh_device)
    _profile_high_bw_all_gather(mesh_device, run)
    durations_ns = [_profile_high_bw_all_gather(mesh_device, run) for _ in range(7)]
    median_ns = statistics.median(durations_ns)

    page_size = WIDTH * 2
    active_rows_per_device = gathered_dim_size // tp
    received_bytes_per_device = active_rows_per_device * page_size * (tp - 1)
    bw_gbps = received_bytes_per_device / median_ns
    per_link_bytes = received_bytes_per_device / 2

    record = {
        "striped": striped,
        "active_chunks": active_chunks,
        "stripes": (N_CHUNKS if striped else 1),
        "median_ms": median_ns / 1e6,
        "samples_ms": [round(d / 1e6, 4) for d in durations_ns],
        "effective_receive_bw_gbps": bw_gbps,
        "active_rows_per_device": active_rows_per_device,
        "received_MB_per_device": received_bytes_per_device / 1e6,
        "per_link_MB": per_link_bytes / 1e6,
        "page_size_B": page_size,
    }
    print("STRIPE_UTIL " + json.dumps(record))

    results = {}
    if _OUT.exists():
        results = json.loads(_OUT.read_text())
    results[f"{'striped' if striped else 'contiguous'}_{active_chunks}ch"] = record
    _OUT.write_text(json.dumps(results, indent=2))
