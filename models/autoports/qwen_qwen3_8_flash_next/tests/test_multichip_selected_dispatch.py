"""Exact selected-dispatch trace checks: changed routes, zero scores, idle ranks."""

import hashlib
import json
import os
from pathlib import Path
import torch
import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    MeshOps,
    _expert_kernel,
    _EXPERT_DISPATCH_SELECTED,
)


def main():
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=4 * 1024 * 1024)
    trace = None
    records = []
    try:
        ops = MeshOps(mesh)
        rank = ops.shards([torch.tensor([r], dtype=torch.int32) for r in range(4)], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        x = ops.upload(torch.zeros(1, 1, 1, 2560), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        ids = ops.upload(torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        scores = ops.upload(torch.zeros(1, 1, 1, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        counts = ttnn.empty((128,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        capacity = ttnn.empty((1, 1, 128 * 32, 2560), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)

        def run():
            _expert_kernel(
                mesh, [x, ids, scores, rank, counts, capacity], _EXPERT_DISPATCH_SELECTED, [1], workers=32, cb_size=5888
            )

        run()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        run()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        cases = [
            ("rank0", torch.arange(10)),
            ("rank1", 128 + torch.arange(10)),
            ("rank2", 256 + torch.arange(10)),
            ("rank3", 502 + torch.arange(10)),
            ("spread", torch.tensor([0, 127, 128, 255, 256, 383, 384, 511, 80, 300])),
        ]
        for iteration in range(4):
            for mode, route in cases:
                xx = ((torch.arange(2560).reshape(1, 1, 1, 2560) + iteration * 7) % 197 / 197).bfloat16()
                ii = torch.zeros(1, 1, 1, 32, dtype=torch.int32)
                ii[..., :10] = route
                ss = torch.zeros(1, 1, 1, 32, dtype=torch.bfloat16)
                ss[..., :10] = 0.1
                if iteration == 1:
                    ss[..., 7] = 0
                if iteration == 2:
                    ss.zero_()
                for host, device, dtype in [
                    (xx, x, ttnn.bfloat16),
                    (ii, ids, ttnn.uint32),
                    (ss, scores, ttnn.bfloat16),
                ]:
                    ht = ttnn.from_torch(
                        host, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)
                    )
                    ttnn.copy_host_to_device_tensor(ht, device)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                for r in range(4):
                    got = ttnn.to_torch(ttnn.get_device_tensors(counts)[r]).flatten().long()
                    expected = torch.zeros(128, dtype=torch.long)
                    active = []
                    for j, e in enumerate(route.tolist()):
                        if e // 128 == r and ss.flatten()[j] != 0:
                            expected[e % 128] = 1
                            active.append(e % 128 * 32)
                    assert torch.equal(got, expected), (mode, iteration, r, got, expected)
                    if active:
                        payload = ttnn.to_torch(ttnn.get_device_tensors(capacity)[r]).reshape(-1, 2560)
                        assert torch.equal(payload[active], xx.reshape(1, 2560).expand(len(active), -1))
                    records.append(
                        dict(
                            mode=mode,
                            iteration=iteration,
                            rank=r,
                            active=int(got.sum()),
                            exact_counts=True,
                            exact_payload=True,
                        )
                    )
        result = dict(
            status="passed",
            records=records,
            replays=20,
            source_sha256=hashlib.sha256(_EXPERT_DISPATCH_SELECTED.encode()).hexdigest(),
            env={
                k: v
                for k, v in os.environ.items()
                if k.startswith("TT_METAL_TRACE") or k.startswith("TT_METAL_WATCHER")
            },
        )
        root = Path("models/autoports/qwen_qwen3_8_flash_next") / os.environ.get(
            "QWEN38_STAGE_DIR", "doc/optimized_multichip_decoder"
        )
        (root / "selected_dispatch_exact.json").write_text(json.dumps(result, indent=2) + "\n")
        print("SELECTED_DISPATCH_EXACT_PASS", len(records), flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
