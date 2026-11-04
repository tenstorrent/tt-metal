# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The GDN lanes' verify rows with the verify-rows fold off (``QWEN38_FUSED_OFF=gdn_rows_scan``, with and without
``gdn_rows_wrap``) against the single stream's rows path run per lane on the same inputs and states, on the 1x4 mesh
with checkpoint layer 0: the forward pass's output rows and final state and the commit's recurrent state and FIR
history, bitwise, for one and four lanes; and four lanes in reverse order give each lane the same result.

Run:  QWEN38_FUSED_DEVICE_TEST=1 pytest models/demos/blackhole/qwen38_flash_next/tests/test_gdn_lanes_device.py
Env:  MODEL_WEIGHTS_DIR, MESH_DEVICE="(1, 4)"; no CI pipeline runs the model's device tests.
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.tests.helpers.tp_harness import (
    DEVICE_PARAMS,
    HIDDEN_SIZE,
    assert_tensors_equal,
    download_shards,
    upload,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE
from models.demos.blackhole.qwen38_flash_next.ttnn.gdn import Qwen38TTNNGDN, Qwen38TTNNGDNWeights

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires held four-die mesh")

ROWS = 5  # k = 4: the base row and four drafts per lane
ACCEPTED = (0, 2, 4, 1)  # drafts accepted per lane: lane u commits ACCEPTED[u] + 1 rows


def _assert_lane_equal(lane_shards, single_shards, label):
    for device, (got, want) in enumerate(zip(lane_shards, single_shards, strict=True)):
        assert_tensors_equal(got, want, f"{label}, device {device}")


class _Layer:
    """Layer 0 with the fold and the wrap off, a lanes state and a single-stream state of the same rows."""

    def __init__(self, harness, lanes: int) -> None:
        self.harness, self.mesh, self.lanes = harness, harness.mesh_device, lanes
        self.weights = Qwen38TTNNGDNWeights.from_checkpoint(
            harness.checkpoint,
            self.mesh,
            harness.contract,
            harness.component_cache_root,
            layer_index=0,
            tt_metal_sha=harness.tt_metal_sha,
        )
        self.module = Qwen38TTNNGDN(self.mesh, harness.contract, self.weights, collective_topology=harness.topology)
        self.constants = self.module.allocate_rows_constants(ROWS)
        self.lane_constants = self.module.allocate_lane_rows_constants(lanes, ROWS)
        self.lane_rows = self.module.allocate_lane_rows_state(self.constants, self.lane_constants)
        self.lane_state = self.module.allocate_lane_state(lanes)
        self.rows = self.module.allocate_rows_state(self.constants)
        self.state = self.module.allocate_state()

    def load(self, recurrent, history, state, history_buffer) -> None:
        for host, target, dtype, dim in (
            (recurrent, state.recurrent, ttnn.float32, 1),
            (history, history_buffer, ttnn.bfloat16, 3),
        ):
            source = upload(self.mesh, host, dtype, shard_dim=dim)
            gdn_module._copy_inplace(source, target, label="test state")
            ttnn.deallocate(source)

    def lanes_pass(self, hidden, recurrent, history, accepted):
        """The lanes' forward pass and commit of ``accepted[u] + 1`` rows per lane: the output tile, the final states,
        then the committed states and histories (host shards per device)."""

        self.load(recurrent, history, self.lane_state, self.lane_rows.history)
        tile = self.harness.upload_sharded(hidden)
        result = self.module.forward_rows_lanes(tile, self.lane_state, self.lane_rows)
        forward = download_shards(result.hidden_rows), download_shards(result.final_state)
        ttnn.deallocate(tile)
        ttnn.deallocate(result.hidden_rows)
        ttnn.deallocate(result.final_state)
        counts = upload(self.mesh, torch.tensor(accepted, dtype=torch.float32).reshape(1, -1, 1, 1), ttnn.float32)
        active = upload(self.mesh, torch.ones(1, self.lanes, 1, 1), ttnn.float32)
        selectors = gdn_module.build_rows_selectors_lanes(counts, active, self.lane_constants)
        self.module.commit_rows_lanes(self.lane_state, self.lane_rows, selectors)
        selectors.deallocate()
        ttnn.deallocate(counts)
        ttnn.deallocate(active)
        return forward + (download_shards(self.lane_state.recurrent), download_shards(self.lane_rows.history))

    def single_pass(self, hidden, recurrent, history, lane):
        """Lane ``lane``'s inputs through the single stream: the same four results for that lane."""

        self.load(recurrent[lane : lane + 1], history[:, lane : lane + 1], self.state, self.rows.history)
        rows = self.harness.upload_sharded(hidden[:, :, lane * ROWS : (lane + 1) * ROWS])
        result = self.module.forward_rows(rows, self.state, self.rows)
        forward = download_shards(result.hidden_rows), download_shards(result.final_state)
        ttnn.deallocate(rows)
        ttnn.deallocate(result.final_state)
        accepted = upload(self.mesh, torch.full((1, 1, 1, 1), float(ACCEPTED[lane])), ttnn.float32)
        selectors = gdn_module.build_rows_selectors(accepted, self.constants)
        self.module.commit_rows(self.state, self.rows, selectors)
        selectors.deallocate()
        ttnn.deallocate(accepted)
        return forward + (download_shards(self.state.recurrent), download_shards(self.rows.history))


def _inputs(lanes: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.zeros(1, 1, 32, HIDDEN_SIZE)
    hidden[:, :, : lanes * ROWS] = torch.randn(1, 1, lanes * ROWS, HIDDEN_SIZE, generator=generator)
    state_shape = (lanes, gdn_module.VALUE_HEADS, gdn_module.HEAD_DIM, gdn_module.HEAD_DIM)
    recurrent = torch.randn(state_shape, generator=generator)
    history = torch.zeros(1, lanes, 32, gdn_module.QKV_WIDTH)
    history[:, :, : gdn_module.CONV_HISTORY_ROWS] = torch.randn(
        1, lanes, gdn_module.CONV_HISTORY_ROWS, gdn_module.QKV_WIDTH, generator=generator
    )
    return hidden.to(torch.bfloat16), recurrent * 0.05, (history * 0.5).to(torch.bfloat16)


def _lane_results(results, lane):
    """Lane ``lane``'s part of the lanes' four results, in the single stream's shapes."""

    output, final, recurrent, history = results
    return (
        [shard[:, :, lane * ROWS : (lane + 1) * ROWS] for shard in output],
        [shard[lane : lane + 1] for shard in final],
        [shard[lane : lane + 1] for shard in recurrent],
        [shard[:, lane : lane + 1] for shard in history],
    )


NAMES = ("output rows", "final state", "committed state", "committed history")


@run_for_blackhole()
@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("lanes", [1, 4])
@pytest.mark.parametrize("off", ["gdn_rows_scan", "gdn_rows_scan,gdn_rows_wrap"], ids=["wrap", "chain"])
def test_gdn_lane_rows_are_the_single_stream_per_lane_with_the_fold_off(tp_harness, monkeypatch, off, lanes):
    monkeypatch.setenv("QWEN38_FUSED_OFF", off)
    layer = _Layer(tp_harness, lanes)
    hidden, recurrent, history = _inputs(lanes, 4100 + lanes)
    results = layer.lanes_pass(hidden, recurrent, history, ACCEPTED[:lanes])
    for lane in range(lanes):
        single = layer.single_pass(hidden, recurrent, history, lane)
        for name, got, want in zip(NAMES, _lane_results(results, lane), single, strict=True):
            _assert_lane_equal(got, want, f"{lanes} lanes, lane {lane}: {name}")
    if lanes > 1:
        # the same lanes in reverse order: slot B - 1 - u holds lane u and gives lane u's results
        order = list(reversed(range(lanes)))
        reordered = torch.zeros_like(hidden)
        for slot, lane in enumerate(order):
            reordered[:, :, slot * ROWS : (slot + 1) * ROWS] = hidden[:, :, lane * ROWS : (lane + 1) * ROWS]
        reversed_results = layer.lanes_pass(
            reordered, recurrent[order], history[:, order], [ACCEPTED[lane] for lane in order]
        )
        for slot, lane in enumerate(order):
            for name, got, want in zip(
                NAMES, _lane_results(reversed_results, slot), _lane_results(results, lane), strict=True
            ):
                _assert_lane_equal(got, want, f"reversed lanes, slot {slot} (lane {lane}): {name}")
