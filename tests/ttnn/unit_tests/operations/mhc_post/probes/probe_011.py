import os, torch, ttnn
from eval.golden_tests.mhc_post.conftest import _FabricMeshCache
from eval.golden_tests.mhc_post.feature_spec import _case
from eval.golden_tests.mhc_post.helpers import *
from ttnn.operations.mhc_post import mhc_post

T, C, links = int(os.environ.get("PT", 640)), int(os.environ.get("PC", 1280)), int(os.environ.get("PL", 1))
cache = _FabricMeshCache()
mesh = cache.get((1, 4))
try:
    inputs = _case((1, 1, T, 4 * C))
    BF16 = dict(dtype=ttnn.bfloat16, sublayer_dtype=ttnn.bfloat16)
    cases = make_mesh_case(inputs, 4, mode="collapse", coef_source="packed", seed=7, norm="on", **BF16)
    kw = dict(
        mode="collapse",
        coef_source="packed",
        mesh_device=mesh,
        math_fidelity=ttnn.MathFidelity.HiFi2,
        op=mhc_post,
        **BF16,
    )
    want = call_mhc_post_mesh(cases, **kw)
    fo = feed_buffer(mesh, inputs[1], cluster_axis=1)
    got = call_mhc_post_mesh(
        cases, **kw, feed_output=fo, feed_semaphores=feed_semaphores(mesh), feed_num_links=links, feed_cluster_axis=1
    )
    ys = [ttnn.to_torch(r[3]) for r in chip_results(want)]
    exp = pytorch_feed(ys, (1, 4), 1)
    chips = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(fo)]
    cb = C // 4 // 32
    for i in range(4):
        g = chips[i].reshape(-1, C // 4).float()
        w = exp[i].reshape(-1, C // 4).float()
        gt = g.reshape(g.shape[0] // 32, 32, cb, 32)
        wt = w.reshape(gt.shape)
        bad = (gt != wt).any(3).any(1)
        fill = (gt == 29952.0).all(3).all(1)
        print(
            "chip",
            i,
            "bad",
            int(bad.sum()),
            "unwritten",
            int(fill.sum()),
            "rows",
            bad.any(1).nonzero().flatten().tolist()[:30],
        )
        # where did the expected data go? find bad tile's expected content elsewhere
        idx = bad.nonzero().tolist()[:3]
        for r, c in idx:
            hit = ((gt == wt[r, :, c, :][None, :, None, :]).all(3).all(1)).nonzero().tolist()
            print("  tile", (r, c), "fill" if bool(fill[r, c]) else "garbage", "expected content found at", hit[:4])
finally:
    cache.close()
