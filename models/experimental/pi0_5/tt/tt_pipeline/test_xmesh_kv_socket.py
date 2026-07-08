# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Cross-mesh prefix-KV socket transport (VLM TP-mesh -> denoise PP-meshes), eager + traced.

4 chips, 2x2 parent (from the mesh_device fixture, FABRIC_1D): KV (replicated, bf8) on a TP2 (1x2)
VLM submesh is socketed -- collinear/nearest chip, device-to-device, NO host -- to two PP (1x1)
denoise submeshes; validated eager AND captured in trace (send on the VLM trace, recv on each PP
trace), replayed with no deadlock.

Run:  pytest models/experimental/pi0_5/tt/tt_pipeline/test_xmesh_kv_socket.py -s
"""
import pytest
import torch
import ttnn

NKV, PREFIX, HD, PAGE, NREPLAY = 1, 1024, 256, 4096, 4

pytestmark = pytest.mark.skipif(ttnn.get_num_devices() < 4, reason="cross-mesh TP2->PP2 needs 4 chips")


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 134_217_728}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_xmesh_kv_socket(mesh_device):
    parent = mesh_device
    subs = []
    try:
        vlm = parent.create_submesh(ttnn.MeshShape(1, 2), ttnn.MeshCoordinate(0, 0))  # TP2
        pp = [parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(1, c)) for c in (0, 1)]  # PP2
        subs += [vlm, *pp]

        torch.manual_seed(0)
        kv = [torch.randn(1, NKV, PREFIX, HD) * 0.1 for _ in range(2)]
        rep = ttnn.ReplicateTensorToMesh(vlm)
        src = [
            ttnn.from_torch(
                k,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=vlm,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            for k in kv
        ]
        mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, PAGE * 4)

        socks, out = [], []
        for c in (0, 1):  # each PP stage sourced from the same-column (collinear, nearest) VLM chip
            conn = ttnn.SocketConnection(
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, c), ttnn.CoreCoord(0, 0)),
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1)),
            )
            socks.append(ttnn.create_socket_pair(vlm, pp[c], ttnn.SocketConfig([conn], mem)))
            out.append(
                ttnn.from_torch(
                    torch.zeros(1, NKV, PREFIX, HD),
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=pp[c],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            )

        # --- eager ---
        for c in (0, 1):
            ttnn.experimental.send_direct_async(src[c], socks[c][0])
            ttnn.experimental.recv_direct_async(out[c], socks[c][1])
        for c in (0, 1):
            ttnn.synchronize_device(pp[c])
            assert _pcc(ttnn.to_torch(out[c]), kv[c]) > 0.99, f"eager stage{c} mismatch"

        # --- traced: send on the VLM trace, recv on each PP trace ---
        tv = ttnn.begin_trace_capture(vlm, cq_id=0)
        for c in (0, 1):
            ttnn.experimental.send_direct_async(src[c], socks[c][0])
        ttnn.end_trace_capture(vlm, tv, cq_id=0)
        tp = []
        for c in (0, 1):
            t = ttnn.begin_trace_capture(pp[c], cq_id=0)
            ttnn.experimental.recv_direct_async(out[c], socks[c][1])
            ttnn.end_trace_capture(pp[c], t, cq_id=0)
            tp.append(t)

        for _ in range(NREPLAY):
            ttnn.execute_trace(vlm, tv, cq_id=0, blocking=False)
            for c in (0, 1):
                ttnn.execute_trace(pp[c], tp[c], cq_id=0, blocking=False)
            for c in (0, 1):
                ttnn.synchronize_device(pp[c])
                assert _pcc(ttnn.to_torch(out[c]), kv[c]) > 0.99, "traced replay mismatch"

        for c in (0, 1):
            ttnn.release_trace(pp[c], tp[c])
        ttnn.release_trace(vlm, tv)
    finally:
        for sm in reversed(subs):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
