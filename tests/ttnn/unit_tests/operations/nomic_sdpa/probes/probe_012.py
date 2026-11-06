import sys, torch, ttnn

sys.path.insert(0, "tests/ttnn/unit_tests/operations/nomic_sdpa")
from test_nomic_sdpa_segments import _layout, _inputs, _reference, HEADS, HEAD_DIM
from ttnn.operations.nomic_sdpa import nomic_sdpa

device = ttnn.open_device(device_id=0)
try:
    for rows, width in ((8, 512), (4, 1024)):
        seg = _layout(rows, width, seed=rows * width)
        print("layout", rows, width, [[(s, n) for s, n in r] for r in seg][:3], "texts", sum(len(r) for r in seg))
        q, k, v = _inputs(rows, width, seg)
        t = [
            ttnn.from_torch(
                x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            for x in (q, k, v)
        ]
        got = (
            ttnn.to_torch(nomic_sdpa(*t, segments=seg, memory_config=ttnn.L1_MEMORY_CONFIG))
            .double()
            .reshape(rows, width, -1)
        )
        ref = _reference(q, k, v, seg)
        keep = torch.isfinite(ref)
        a, b = got[keep], ref[keep]
        print(
            "rows",
            rows,
            "x",
            width,
            "PCC",
            round(float(torch.corrcoef(torch.stack([a, b]))[0, 1]), 6),
            "relRMS",
            round(float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()), 5),
        )
        # the same inputs through a WRONG reference (whole-row softmax) must disagree: the test discriminates
        full = torch.softmax(q.double() @ k.nan_to_num().double().transpose(-1, -2), -1) @ v.nan_to_num().double()
        full = full.permute(0, 2, 1, 3).reshape(rows, width, -1)
        print(
            "  vs whole-row attention relRMS",
            round(float((a - full[keep]).pow(2).mean().sqrt() / full[keep].pow(2).mean().sqrt()), 4),
        )
finally:
    ttnn.close_device(device)
