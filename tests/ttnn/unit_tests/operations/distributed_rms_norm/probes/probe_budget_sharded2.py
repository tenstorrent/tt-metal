import torch, ttnn, pytest
from eval.golden_tests.distributed_rms_norm.helpers import create_ttnn_input_tensor, random_stacked
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}]


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
def test_budget(mesh_device):
    m = mesh_device
    orig = pd.footprint

    def fp(R, nw, **kw):
        f = orig(R, nw, **kw)
        if R <= 2:
            print("  R", R, "nw", nw, kw, "->", f)
        return f

    pd.footprint = fp
    for shape, lay in [((1, 1, 128, 4096), ttnn.TensorMemoryLayout.WIDTH_SHARDED)]:
        x = random_stacked(m, shape, torch.bfloat16, seed=1)
        tx = create_ttnn_input_tensor(x, m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_layout=lay)
        print("budget", pd._l1_budget(m))
        try:
            p = pd.make_plan(m, tx, None, 1, ttnn.Topology.Linear, 1)
            print(p.R, p.num_waves)
        except Exception as e:
            print("ERR", e)
