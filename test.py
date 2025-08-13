import pytest
import torch
import ttnn

# @pytest.mark.parametrize(
#     "device_params",
#     [
#         ({"fabric_config": ttnn.FabricConfig.FABRIC_1D}),
#     ],
#     indirect=True,
# )
@pytest.mark.parametrize("shape", [(2,1,16384, 4)])
def test_dist(shape):
    ttnn.CreateDevice(0, dispatch_core_config=ttnn.DispatchCoreConfig(type=ttnn.DispatchCoreType.ETH, axis=ttnn.DispatchCoreAxis.COL, fabric_config=ttnn.FabricConfig.FABRIC_1D))