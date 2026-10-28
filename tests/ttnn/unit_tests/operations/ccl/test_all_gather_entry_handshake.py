# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""all_gather into a caller-supplied output keeps its entry handshake unless peers are proven done with that output.

Without the handshake a fast device writes its shard into a slower peer's output before the peer has read what the
previous gather put there (#58399). The handshake is dropped only when every earlier access to the output precedes the
start of an earlier gather on the same axis and queue: that gather has completed on the writer, which proves every
peer had started it.
"""

from contextlib import contextmanager

import pytest
import torch
import ttnn

ITERATIONS = 8
DELAY_CYCLES = 5_000_000  # a few milliseconds


@contextmanager
def queue_ordering(mode):
    ttnn._ttnn.core.set_queue_ordering(mode)
    try:
        yield
    finally:
        ttnn._ttnn.core.set_queue_ordering("off")


def gather(t, output=None):
    return ttnn.all_gather(t, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear, output_tensor=output)


def sharded(mesh_device, torch_tensor):
    return ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_the_handshake_is_dropped_only_with_proof(mesh_device):
    def without_handshake(output):
        return ttnn._ttnn.core.peers_may_write_without_handshake(mesh_device, [output], cluster_axis=1)

    x = sharded(mesh_device, torch.randn(1, 1, 64, 512))
    with queue_ordering("check"):
        one = gather(x)
        assert not without_handshake(one)  # the latest gather wrote it, after it started

        first, second = gather(x), gather(x)
        assert without_handshake(first)  # written before the latest gather started
        assert not without_handshake(second)

        ttnn.multiply(first, 1.0)
        assert not without_handshake(first)  # read since
        gather(x, second)
        assert without_handshake(first)  # the read precedes the latest gather
    assert not without_handshake(first)  # nothing is known with ordering off


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("ordering", ["off", "check"])
@pytest.mark.parametrize("outputs", [1, 2])
def test_a_lagging_peer_reads_every_gather(mesh_device, ordering, outputs):
    """Device 1 is delayed after each gather into the same output(s); every device must still read each one intact."""
    full = torch.randn(1, 1, 256, 2048)
    x = sharded(mesh_device, full)
    reference = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)).float()
    with queue_ordering(ordering):
        buffers = [gather(x) for _ in range(outputs)]
        ttnn.synchronize_device(mesh_device)
        results = []
        for i in range(ITERATIONS):
            scaled = ttnn.multiply(x, float((-1) ** i * (i + 1)))
            # With two outputs, the gather into the other one since this one was last read proves the peers are done.
            proven = ttnn._ttnn.core.peers_may_write_without_handshake(
                mesh_device, [buffers[i % outputs]], cluster_axis=1
            )
            assert proven == (ordering == "check" and outputs == 2)
            gathered = gather(scaled, buffers[i % outputs])
            ttnn.apply_device_delay(mesh_device, [[0, DELAY_CYCLES]])
            results.append(ttnn.multiply(gathered, 1.0))
            ttnn.deallocate(scaled)
        ttnn.synchronize_device(mesh_device)

    for i, result in enumerate(results):
        expected = reference * float((-1) ** i * (i + 1))
        for device, shard in enumerate(ttnn.get_device_tensors(result)):
            got = ttnn.to_torch(shard).float()
            error = ((got - expected).norm() / expected.norm()).item()
            assert error < 0.02, f"iteration {i}, device {device}: relative error {error}"


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 1 << 20}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_a_captured_gather_keeps_the_handshake(mesh_device):
    """A trace replays later, after whatever ran in between, so evidence from before the capture does not hold."""

    def without_handshake(output):
        return ttnn._ttnn.core.peers_may_write_without_handshake(mesh_device, [output], cluster_axis=1)

    x = sharded(mesh_device, torch.randn(1, 1, 64, 512))
    # Compile the gather into a given output with the handshake first: a trace cannot capture a program cache miss.
    gather(x, gather(x))
    ttnn.synchronize_device(mesh_device)
    with queue_ordering("check"):
        first, second = gather(x), gather(x)
        assert without_handshake(first)
        trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        assert not without_handshake(first)
        gather(x, first)
        ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
        ttnn.release_trace(mesh_device, trace)


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
@pytest.mark.parametrize("queues", [(0, 1), (0, 0)], ids=["two_queues", "one_queue"])
def test_gathers_from_two_streams_take_turns_on_the_link(mesh_device, queues):
    """Gathers on two sub-devices at once deadlock over the shared link; with ordering on, each waits for the last."""
    grid = mesh_device.compute_with_storage_grid_size()
    rest = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 2, grid.y - 1))})
    column = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(grid.x - 1, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    manager = mesh_device.create_sub_device_manager([ttnn.SubDevice([rest]), ttnn.SubDevice([column])], 0)
    mesh_device.load_sub_device_manager(manager)
    try:
        full = [torch.randn(1, 1, 1024, 4096) for _ in range(2)]
        inputs = [sharded(mesh_device, f) for f in full]
        references = [
            ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)).float() for t in inputs
        ]

        def gather_on(sub_device):
            with ttnn.command_queue(queues[sub_device]):
                return ttnn.all_gather(
                    inputs[sub_device],
                    dim=3,
                    cluster_axis=1,
                    subdevice_id=ttnn.SubDeviceId(sub_device),
                    topology=ttnn.Topology.Linear,
                )

        for sub_device in (0, 1):
            gather_on(sub_device)
            ttnn.synchronize_device(mesh_device)
        with queue_ordering("order"):
            outputs = [[gather_on(sub_device) for sub_device in (0, 1)] for _ in range(4)]
            ttnn.synchronize_device(mesh_device)
        for pair in outputs:
            for sub_device, output in enumerate(pair):
                for shard in ttnn.get_device_tensors(output):
                    got = ttnn.to_torch(shard).float()
                    assert ((got - references[sub_device]).norm() / references[sub_device].norm()).item() < 0.01
    finally:
        ttnn.synchronize_device(mesh_device)
        mesh_device.clear_loaded_sub_device_manager()
        mesh_device.remove_sub_device_manager(manager)
