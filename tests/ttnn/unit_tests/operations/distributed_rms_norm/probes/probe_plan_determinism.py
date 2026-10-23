# Is the plan (R, num_waves, layout) identical across back-to-back calls? (trace capture needs the cached program)
import torch, ttnn, pytest
from eval.golden_tests.distributed_rms_norm.helpers import create_ttnn_gamma, create_ttnn_input_tensor, random_stacked
from ttnn.operations.distributed_rms_norm import distributed_rms_norm
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}]


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_det(mesh_device):
    m = mesh_device
    shape = (1, 1, 2048, 1280)
    x = random_stacked(m, shape, torch.bfloat16, seed=0)
    g = random_stacked(m, (1, 1, 1, shape[-1]), torch.bfloat16, seed=1)
    tx = create_ttnn_input_tensor(
        x, m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED
    )
    tg = create_ttnn_gamma(g, m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    orig = pd.make_plan
    base = int(ttnn._ttnn.device.GetAllocatorBaseAddress(m, ttnn.BufferType.L1))

    def spy(*a, **k):
        p = orig(*a, **k)
        print(
            "PLAN R",
            p.R,
            "nw",
            p.num_waves,
            "C",
            p.C,
            "gres",
            p.gamma_resident,
            "L",
            p.L_act,
            "land",
            p.landing_bytes,
            "budget",
            pd._l1_budget(m),
        )
        print("BLOCKS", [(a_, s, al) for a_, s, al in pd._l1_blocks(m)])
        return p

    pd.make_plan = spy
    for _ in range(3):
        out = distributed_rms_norm(tx, cluster_axis=1, gamma=tg, epsilon=1e-5, topology=ttnn.Topology.Ring, num_links=2)
        print("OUT", int(out.buffer_address()) - base, "X", int(tx.buffer_address()) - base)
    pd.make_plan = orig
