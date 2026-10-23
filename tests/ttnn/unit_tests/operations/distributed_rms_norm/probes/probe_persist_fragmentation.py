# Does the persistent landing scratch / global semaphores, created while a big shard is live, cap later CB space?
import torch, ttnn, pytest
from eval.golden_tests.distributed_rms_norm.helpers import create_ttnn_input_tensor, random_stacked
from ttnn.operations.distributed_rms_norm import distributed_rms_norm
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}]


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
def test_frag(mesh_device):
    m = mesh_device
    print("budget empty", pd._l1_budget(m))
    x = random_stacked(m, (1, 1, 32, 4096), torch.float32, seed=1)
    tx = create_ttnn_input_tensor(
        x, m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    )
    out = distributed_rms_norm(tx, cluster_axis=1)
    ttnn.synchronize_device(m)
    for k, e in pd._PERSIST.items():
        print(
            "persist",
            k,
            [int(t.buffer_address()) for t in e["scratch"].values()],
            [int(ttnn.get_global_semaphore_address(s)) for s in e["sems"]],
        )
    print("shard addr", int(tx.buffer_address()), int(out.buffer_address()))
    out.deallocate()
    tx.deallocate()
    print("budget after", pd._l1_budget(m))
