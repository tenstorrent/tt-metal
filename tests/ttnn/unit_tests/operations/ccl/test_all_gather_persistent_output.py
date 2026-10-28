# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""all_gather into a caller-supplied output waits for its peers before writing into their copies of it.

A preallocated output exists on every device, but a slower peer may still be reading what the previous gather put
there. A fast device that writes its shard into the peer's copy without the entry handshake overwrites data the peer
has not read yet.
"""

import pytest
import torch
import ttnn

ITERATIONS = 8
DELAY_CYCLES = 5_000_000  # a few milliseconds


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
@pytest.mark.parametrize("outputs", [1, 2])
def test_a_lagging_peer_reads_every_gather(mesh_device, outputs):
    """Device 1 is delayed after each gather into the same output(s); every device must still read each one intact."""
    full = torch.randn(1, 1, 256, 2048)
    x = sharded(mesh_device, full)
    reference = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3)).float()

    def gather(t, output=None):
        return ttnn.all_gather(t, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear, output_tensor=output)

    buffers = [gather(x) for _ in range(outputs)]
    ttnn.synchronize_device(mesh_device)
    results = []
    for i in range(ITERATIONS):
        scaled = ttnn.multiply(x, float((-1) ** i * (i + 1)))
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
