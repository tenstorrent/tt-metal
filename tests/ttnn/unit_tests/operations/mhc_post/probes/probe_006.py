import torch, ttnn
from eval.golden_tests.mhc_post.conftest import _FabricMeshCache
from eval.golden_tests.mhc_post.feature_spec import _case
from eval.golden_tests.mhc_post.helpers import *
from ttnn.operations.mhc_post import mhc_post

cache = _FabricMeshCache()
mesh = cache.get((1, 4))
try:
    inputs = _case((1, 1, 640, 4 * 1280))
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
        cases, **kw, feed_output=fo, feed_semaphores=feed_semaphores(mesh), feed_num_links=1, feed_cluster_axis=1
    )
    ys = [ttnn.to_torch(r[3]) for r in chip_results(want)]
    exp = pytorch_feed(ys, (1, 4), 1)
    chips = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(fo)]
    for i in range(4):
        g = chips[i].reshape(-1, 320).float()
        w = exp[i].reshape(-1, 320).float()
        bad = (g != w).reshape(g.shape[0] // 32, 32, 10, 32).any(3).any(1)  # [tile rows, tile cols]
        rows = bad.any(1).nonzero().flatten().tolist()
        print("chip", i, "bad tiles", int(bad.sum()), "tile rows", rows[:40])
        if rows:
            r = rows[0]
            print("tile", r, "cols", bad[r].nonzero().flatten().tolist())
            # is the stale marker there?
            print("tile vals", g[r * 32, :320:32].tolist(), w[r * 32, :320:32].tolist())
finally:
    cache.close()
