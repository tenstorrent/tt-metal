import torch
import pytest
import ttnn
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_typecasting(mesh_device):
    device = mesh_device.get_device(mesh_device.get_device_ids()[0])
    torch_input = torch.randint(0, 32000, (1, 1, 32, 32), dtype=torch.long)
    tt_input = ttnn.from_torch(
        torch_input,
        device=device,
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
    )
    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )
    tt_input = ttnn.typecast(tt_input, dtype=ttnn.uint32, sub_core_grids=sub_core_grids)
    tt_input_to_torch = ttnn.to_torch(tt_input)
    passing, pcc = comp_pcc(tt_input_to_torch, torch_input, 0.99)
    print(passing, pcc)
    assert passing, f"Typecasting failed"
