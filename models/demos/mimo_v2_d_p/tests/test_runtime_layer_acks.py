# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""MiMoPrefillRuntime layer acks (options.ack_sync): every layer acked exactly once, in order, only after the layer
finished on device, and the end-to-end chunk time with the deferred acks (ack layer L - 1 once layer L is enqueued: the
device queue never drains) vs the old synchronize-per-layer acks (emulated by a sink that synchronizes itself).

    scripts/run_safe_pytest.sh models/demos/mimo_v2_d_p/tests/test_runtime_layer_acks.py -s
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.mimo_v2_d_p.tests.mesh import MESH_PARAMS
from models.demos.mimo_v2_d_p.tt.tt_prefill_runtime import MiMoPrefillRuntime, MiMoRuntimeConfig

LAYERS = 6  # GA + dense, 4x SWA + MoE, GA + MoE


@pytest.mark.timeout(3600)
@MESH_PARAMS
@pytest.mark.parametrize("chunk", [1280, 4096], ids=["640tok", "2048tok"])
def test_runtime_layer_acks(mesh_device, device_params, chunk):
    seq = 8 * chunk
    rt = MiMoPrefillRuntime(
        mesh_device,
        MiMoRuntimeConfig(
            num_layers=LAYERS,
            max_seq_len=seq,
            chunk_size=chunk,
            mesh_shape=tuple(mesh_device.shape),
            fabric_config=device_params["fabric_config"],
        ),
    )
    kv = rt.allocate_kv_caches()
    rt.compile(kv)
    ids = torch.randint(0, 1000, (chunk,)).tolist()

    acks = []
    deferred_sink = lambda layer, rid: acks.append((layer, rid))

    def syncing_sink(layer, rid):  # the old behaviour: the device drained before every ack
        ttnn.synchronize_device(mesh_device)
        acks.append((layer, rid))

    # same chunk position (same context depth) for both modes, alternating
    times = {"deferred": [], "synced": []}
    for rep in range(6):
        mode = "deferred" if rep % 2 == 0 else "synced"
        rt.set_layer_completion_sink(deferred_sink if mode == "deferred" else syncing_sink)
        acks.clear()
        inp = rt.make_chunk_input(ids)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        rt.prefill_chunk(inp, kv, slot_id=0, actual_start=2 * chunk, actual_end=3 * chunk, request_id=rep)
        times[mode].append((time.perf_counter() - t0) * 1e3)
        assert acks == [(i, rep) for i in range(LAYERS)], acks
    deferred, synced = times["deferred"], times["synced"]
    med = lambda v: sorted(v)[len(v) // 2]
    logger.info(
        f"{chunk // mesh_device.shape[0]} tok/chip, {LAYERS} layers: deferred acks {med(deferred):.2f} ms / chunk, "
        f"synchronize per layer {med(synced):.2f} ms / chunk ({deferred} vs {synced})"
    )
