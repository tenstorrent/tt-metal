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
    ys = [ttnn.to_torch(r[3]).reshape(640, 1280).float() for r in chip_results(want)]
    yst = [ttnn.to_torch(r[3]).reshape(640, 1280).float() for r in chip_results(got)]
    for i in range(4):
        p = i
        bad = (yst[i] != ys[i]).reshape(20, 32, 40, 32).any(3).any(1)
        own = bad[:, p * 10 : (p + 1) * 10]
        bad[:, p * 10 : (p + 1) * 10] = False
        print(
            "chip",
            i,
            "staging bad tiles (remote blocks)",
            int(bad.sum()),
            "rows",
            bad.any(1).nonzero().flatten().tolist(),
        )
    chips = [ttnn.to_torch(t).reshape(-1, 320).float() for t in ttnn.get_device_tensors(fo)]
    g = chips[0][20 * 32 : 21 * 32]
    print("chip0 D tile row 20 uniq", torch.unique(g).tolist()[:8])
finally:
    cache.close()
