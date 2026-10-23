import torch, ttnn, pytest
from eval.golden_tests.distributed_rms_norm.helpers import create_ttnn_input_tensor, random_stacked
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

FABRIC = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}]


@pytest.mark.parametrize("device_params", FABRIC, indirect=True)
def test_budget(mesh_device):
    m = mesh_device
    print("grid", m.compute_with_storage_grid_size(), "budget empty", pd._l1_budget(m))
    v = ttnn.get_memory_view(m, ttnn.BufferType.L1)
    print(v)
    for shape, lay in [
        ((1, 1, 128, 4096), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ((128, 8192), ttnn.TensorMemoryLayout.WIDTH_SHARDED),
        ((1, 32, 4096), ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    ]:
        x = random_stacked(m, shape, torch.bfloat16, seed=1)
        tx = create_ttnn_input_tensor(x, m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_layout=lay)
        print(shape, lay, tx.memory_config().shard_spec, "budget", pd._l1_budget(m))
        o = ttnn.allocate_tensor_on_device(tx.shape, tx.dtype, tx.layout, m, tx.memory_config())
        print("  after out", pd._l1_budget(m))
        o.deallocate()
        tx.deallocate()
