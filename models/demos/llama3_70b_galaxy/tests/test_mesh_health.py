# Quick health check after a hang: open the 8x4 mesh with fabric, run a small all-gather on each axis, check values.
import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_mesh_health(mesh_device):
    for axis, n in ((1, 4), (0, 8)):
        t = torch.arange(32 * 32 * n, dtype=torch.float32).reshape(1, 1, 32, 32 * n).bfloat16()
        dims = (None, 3) if axis == 1 else (3, None)
        x = ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=(8, 4)),
        )
        y = ttnn.all_gather(x, dim=3, cluster_axis=axis)
        ttnn.synchronize_device(mesh_device)
        out = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
        ok = torch.allclose(out, t.float(), atol=1)
        logger.info(f"HEALTH all_gather axis={axis}: {'OK' if ok else 'MISMATCH'}")
        assert ok
