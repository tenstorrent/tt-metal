"""Exact device dispatch metadata and selected rows, including idle ranks."""

import argparse, json
from pathlib import Path
import torch, ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import (
    MeshOps,
    _expert_kernel,
    _EXPERT_DISPATCH_PACKED,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="expert_dispatch")
    a = p.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    records = []
    try:
        ops = MeshOps(mesh)
        rank = ops.shards([torch.tensor([r], dtype=torch.int32) for r in range(4)], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        capacity = ttnn.empty((1, 1, 128 * 128, 2560), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        counts = ttnn.empty((128,), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
        for n in (1, 3, 32, 33, 127, 128):
            for mode in ("idle_ranks", "spread"):
                x = (torch.arange(n * 2560).reshape(1, 1, n, 2560) % 197 / 197).bfloat16()
                ids = torch.zeros(1, 1, n, 32, dtype=torch.int32)
                scores = torch.zeros(1, 1, n, 32, dtype=torch.bfloat16)
                for row in range(n):
                    ids[0, 0, row, :10] = (torch.arange(10) + (row * 37 if mode == "spread" else 0)) % 512
                    scores[0, 0, row, :10] = 0.1
                    if row % 3 == 0:
                        scores[0, 0, row, 7] = 0  # Explicit score-flush route.
                xd = ops.upload(x, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                it = ops.upload(ids, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                st = ops.upload(scores, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
                mapping = ttnn.empty((1, 1, n, 32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh)
                _expert_kernel(
                    mesh,
                    [xd, it, st, rank, counts, mapping, capacity],
                    _EXPERT_DISPATCH_PACKED,
                    [n],
                    workers=32,
                    cb_size=5120 + n * 192 + 64,
                )
                ttnn.synchronize_device(mesh)
                for r in range(4):
                    got = ttnn.to_torch(ttnn.get_device_tensors(counts)[r]).flatten().to(torch.int64)
                    mapped = ttnn.to_torch(ttnn.get_device_tensors(mapping)[r]).reshape(n, 32).to(torch.int64)
                    expected = torch.zeros(128, dtype=torch.int64)
                    entries = []
                    for row in range(n):
                        for route in range(10):
                            e = int(ids[0, 0, row, route])
                            score = float(scores[0, 0, row, route])
                            if e // 128 != r or not score:
                                continue
                            local = e % 128
                            offset = int(expected[local])
                            assert mapped[row, route] == offset, (n, mode, r, row, route)
                            entries.append((local * 128 + offset, row))
                            expected[local] += 1
                    assert torch.equal(got, expected), (n, mode, r, got, expected)
                    if entries:
                        payload = ttnn.to_torch(ttnn.get_device_tensors(capacity)[r]).reshape(-1, 2560)
                        assert torch.equal(
                            payload[[i for i, _ in entries]], x.reshape(n, 2560)[[j for _, j in entries]]
                        )
                    records.append(
                        dict(
                            n=n,
                            mode=mode,
                            rank=r,
                            active_experts=int((got > 0).sum()),
                            routed_rows=int(got.sum()),
                            exact_metadata=True,
                            exact_payload=True,
                        )
                    )
                print("DISPATCH_OK", n, mode, flush=True)
        out = Path("models/autoports/qwen_qwen3_8_flash_next/doc/multichip_decoder") / (a.tag + ".json")
        out.write_text(json.dumps(dict(status="passed", records=records), indent=2))
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
