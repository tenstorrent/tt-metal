"""Error map per (chip, source segment j, 32-row tile of each head): which output regions are wrong."""
import pytest, torch, ttnn
from ttnn.operations.a2a_wkv_b2 import a2a_wkv_b2
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import MESH_SHAPE, _make_inputs, _expected


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_probe_errmap(mesh_device):
    chips, attn, w = _make_inputs(mesh_device, 0)
    exp = _expected(mesh_device, chips, w)
    out = a2a_wkv_b2(attn, w)
    act = [ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)]
    for c, (a, e) in enumerate(zip(act, exp)):
        err = (a - e).abs()  # [1, 24, 640, 128]
        bad = err.reshape(24, 4, 5, 32, 128).amax(dim=(3, 4)) > 0.5  # head, src j, mt
        print(
            f"chip {c}: bad tiles-rows {int(bad.sum())}; per src j {bad.sum(dim=(0, 2)).tolist()}; per mt {bad.sum(dim=(0, 1)).tolist()}; heads {sorted(set(bad.nonzero()[:, 0].tolist()))[:12]}"
        )
