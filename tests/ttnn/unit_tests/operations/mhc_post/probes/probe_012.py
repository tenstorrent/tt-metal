# Refinement 2b: T % 32 != 0 feed at links 1 / 2, multi-image, T < 32 and T = 1 (bit-exact vs the no-feed call)
import torch, ttnn
from eval.golden_tests.mhc_post.conftest import _FabricMeshCache
from eval.golden_tests.mhc_post.feature_spec import _case
from eval.golden_tests.mhc_post.helpers import *
from ttnn.operations.mhc_post import mhc_post

cache = _FabricMeshCache()
mesh = cache.get((1, 4))
BF16 = dict(dtype=ttnn.bfloat16, sublayer_dtype=ttnn.bfloat16)
try:
    for shape, links, norm in [
        ((1, 1, 150, 4 * 5120), 1, "on"),
        ((1, 2, 77, 4 * 640), 2, "on"),
        ((1, 2, 77, 4 * 640), 1, "off"),
        ((3, 5, 4 * 1280), 1, "on"),
        ((2, 1, 4 * 7168), 2, "on"),
        ((1, 1, 1000, 4 * 1792), 1, "on"),
    ]:
        inputs = _case(shape)
        cases = make_mesh_case(inputs, 4, mode="collapse", coef_source="packed", seed=11, norm=norm, **BF16)
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
        try:
            for rep in range(2):
                got = call_mhc_post_mesh(
                    cases,
                    **kw,
                    feed_output=fo,
                    feed_semaphores=feed_semaphores(mesh),
                    feed_num_links=links,
                    feed_cluster_axis=1,
                )
                check_feed_call(got, want, fo, mesh_shape=(1, 4), cluster_axis=1)
            print("case", shape, links, norm, "OK")
        except Exception as e:
            print("case", shape, links, norm, "FAIL", str(e)[:300])
finally:
    cache.close()
