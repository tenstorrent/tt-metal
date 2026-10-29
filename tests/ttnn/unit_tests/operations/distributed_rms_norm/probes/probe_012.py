import ttnn
from eval.golden_tests.distributed_rms_norm.conftest import _FabricMeshCache
from eval.golden_tests.distributed_rms_norm.helpers import run_distributed_rms_norm
import ttnn.operations.distributed_rms_norm.distributed_rms_norm_program_descriptor as pd

cache = _FabricMeshCache()
mesh = cache.get()
orig = pd.local_plan


def lp(*a, **k):
    p = orig(*a, **k)
    print(
        "PROBE plan",
        {
            f: getattr(p, f)
            for f in (
                "regime",
                "num_waves",
                "q_w",
                "owners",
                "q",
                "PW",
                "cap",
                "gamma_mode",
                "gamma_stream",
                "combine_depth",
                "sq_rows",
                "x_depth",
                "out_resident",
                "G_w",
                "rows_max",
                "S",
            )
        },
        "budget",
        a[3],
    )
    return p


pd.local_plan = lp
axes = dict(
    dtype=ttnn.float32,
    layout=ttnn.TILE_LAYOUT,
    fp32_dest_acc_en=True,
    gamma_mode="gamma",
    gamma_dtype=ttnn.float32,
    gamma_layout=ttnn.TILE_LAYOUT,
    memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    cluster_axis=0,
    topology=ttnn.Topology.Linear,
    num_links=1,
)
for i in range(2):
    try:
        run_distributed_rms_norm(((1, 1, 4096, 1024),), mesh_device=mesh, **axes)
        print("PROBE pass")
    except Exception as e:
        print("PROBE fail", str(e)[:200])
