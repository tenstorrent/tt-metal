# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.9C: mesh-only probe of tt-d-gen's per-layer D2H ack on this model's mesh (tests/packed/ladder_common.D2HAcks):
a metadata-only D2HStreamService + LayerAckService, ACK_N acks through outbound_socket_service_sync (in bursts of 105,
a 5-segment packed chunk's 21 layers), each burst drained from the layer-completion ring; between bursts the model's
compute / gather sub-device manager (tt/gather_hide.Split) is loaded and cleared, as a packed layer does. Prints the
host ms an ack op costs to enqueue and the burst's device time. No model load (BATCH_SHARED: the resident runner's
mesh)."""

from __future__ import annotations

import os
import time

import pytest

from models.demos.common.bringup.testing.harness import mesh_parametrize

BATCH_SHARED = True
BATCH_ITEM_VARS = ("ACK_N",)


@mesh_parametrize(timeout_s=600)
def test_d2h_ack_probe(mesh_device):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tests.packed.ladder_common import D2HAcks
    from models.demos.deepseek_v41_flash_d_p.tt.gather_hide import Split

    n = int(os.environ.get("ACK_N", "315") or 315)
    acks = D2HAcks(mesh_device)
    split = Split.of(mesh_device)
    enq, dev = [], []
    try:
        with acks:
            sent = 0
            while sent < n:
                k = min(105, n - sent)
                if split is not None:
                    mesh_device.load_sub_device_manager(split.mgr)
                    mesh_device.clear_loaded_sub_device_manager()
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                for _ in range(k):
                    ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(acks.d2h, metadata=acks.record)
                t1 = time.perf_counter()
                ttnn.synchronize_device(mesh_device)
                t2 = time.perf_counter()
                enq.append((t1 - t0) * 1e3 / k)
                dev.append((t2 - t0) * 1e3)
                sent += k
                missing = acks.drain(30.0)
                assert missing == 0, f"{missing} of {acks.sent} acks not seen"
    finally:
        err = acks.close()
    print(
        f"D2H ACK PROBE: {acks.sent} acks, {acks.popped} popped; host enqueue ms an ack {[round(x, 4) for x in enq]}; "
        f"burst wall to sync ms {[round(x, 2) for x in dev]}; split {'yes' if split is not None else 'no'}; "
        f"service error {err}",
        flush=True,
    )
    assert err is None, err
    assert acks.popped == acks.sent == n
