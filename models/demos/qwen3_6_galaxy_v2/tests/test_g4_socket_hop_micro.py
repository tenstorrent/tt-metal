# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""G=4 inter-stage socket hop microbenchmark.

Times one activation-page transfer between two adjacent 1x4 submeshes on an
8x4 BH Galaxy parent mesh (models one pipeline stage boundary). Confirms socket
tax is negligible vs the ~13ms G=4 compute floor.

Run:
    export ARCH_NAME=wormhole_b0 TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && python -m pytest --noconftest -v -s \\
            models/demos/qwen3_6_galaxy_v2/tests/test_g4_socket_hop_micro.py
"""
from __future__ import annotations

import os
import time

import pytest
import torch

pytestmark_hardware = pytest.mark.skipif(
    os.environ.get("G4_RUN_DEVICE", "0") != "1",
    reason="Device tests disabled. Set G4_RUN_DEVICE=1 to run on silicon.",
)

import ttnn

_DIM = 5120
_N_STAGES = 8  # 64 layers / 8 layers per stage
_N_WARM = 20
pytestmark = [pytestmark_hardware]

_ACTIVATION_BYTES_B1 = 32 * _DIM * 2  # tile-padded M=32, bf16


@pytest.fixture(scope="module")
def bh_glx_mesh():
    ttnn.set_fabric_config(
        ttnn.FabricConfig.FABRIC_1D_RING,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
    )
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(8, 4),
        worker_l1_size=1345000,
    )
    yield mesh
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _build_socket_pair(sender, receiver, fifo_size: int):
    mesh_shape = sender.shape
    sender_cores = [ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0), ttnn.CoreCoord(3, 0)]
    recv_cores = [ttnn.CoreCoord(0, 1), ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1), ttnn.CoreCoord(3, 1)]
    connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        for i in range(len(sender_cores)):
            connections.append(
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(coord, sender_cores[i]),
                    ttnn.MeshCoreCoord(coord, recv_cores[i]),
                )
            )
    mem_cfg = ttnn.SocketMemoryConfig(ttnn.BufferType.DRAM, fifo_size)
    cfg = ttnn.SocketConfig(connections, mem_cfg)
    return ttnn.create_socket_pair(sender, receiver, cfg)


@pytest.mark.hardware
@pytest.mark.parametrize("batch_rows", [1, 4], ids=["B1", "B4"])
def test_g4_socket_hop_latency(bh_glx_mesh, batch_rows):
    """Measure send+recv latency for one activation page between 1x4 submeshes."""
    parent = bh_glx_mesh
    sender = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))
    receiver = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(1, 0))

    m = max(32, batch_rows)
    per_chip_shape = (1, 1, m, _DIM // 4)
    torch_input = torch.randn(1, 1, 1, *per_chip_shape[1:], dtype=torch.bfloat16) * 0.01
    torch_mesh = torch_input.expand(1, 4, *per_chip_shape[1:]).contiguous()

    input_tensor = ttnn.from_torch(
        torch_mesh,
        device=sender,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(sender, dims=(0, 1), mesh_shape=(1, 4)),
    )
    output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, receiver)

    fifo = max(_ACTIVATION_BYTES_B1 * batch_rows, 64 * 1024)
    send_sock, recv_sock = _build_socket_pair(sender, receiver, fifo)

    def _hop():
        ttnn.experimental.send_async(input_tensor, send_sock)
        ttnn.experimental.recv_async(output_tensor, recv_sock)
        ttnn.synchronize_device(sender)
        ttnn.synchronize_device(receiver)

    _hop()  # compile
    ttnn.synchronize_device(sender)
    ttnn.synchronize_device(receiver)

    t0 = time.perf_counter()
    for _ in range(_N_WARM):
        _hop()
    t1 = time.perf_counter()

    hop_ms = (t1 - t0) / _N_WARM * 1000.0
    pipeline_ms = hop_ms * (_N_STAGES - 1)

    print(
        f"\n[G4-SOCKET] batch={batch_rows} hop_ms={hop_ms:.4f} "
        f"x{_N_STAGES - 1}_stages={pipeline_ms:.3f}ms "
        f"(activation ~{m * _DIM * 2 / 1024:.1f} KiB per page)"
    )

    assert pipeline_ms < 1.0, f"Socket pipeline tax {pipeline_ms:.2f}ms too high vs ~13ms compute floor"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
