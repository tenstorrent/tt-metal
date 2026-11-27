import pytest, torch, ttnn
from ttnn.operations.a2a_wkv_b2 import a2a_wkv_b2
from tests.ttnn.unit_tests.operations.a2a_wkv_b2.test_a2a_wkv_b2 import MESH_SHAPE, _expected, _make_inputs


def stats(tag, a, e):
    s = (a * e).sum() / (e * e).sum()
    big = e.abs() > 0.25 * e.abs().max()
    nz = e.abs() > 1e-3 * e.abs().max()
    r = (a[nz] / e[nz])[: 1 << 24]
    rb = a[big] / e[big]
    print(
        f"\n{tag}: slope={s.item():.5f} ratio_median_all={r.median().item():.5f} ratio_median_big={rb.median().item():.5f} "
        f"mean_signed_rel_err_big={((a[big]-e[big])/e[big]).mean().item():.5f}"
    )


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_probe(mesh_device):
    chips, attn, w = _make_inputs(mesh_device, 0)
    out = a2a_wkv_b2(attn, w)
    e = torch.stack(_expected(mesh_device, chips, w)).float()
    a = torch.stack([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)])
    stats("device", a, e)
    # host bf8_b quantization of the exact reference
    q = ttnn.to_torch(ttnn.from_torch(e[0:1], dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
    stats("host_bf8_of_ref(chip0)", q, e[0:1])
    qb = e[0:1].to(torch.bfloat16).float()
    stats("bf16_of_ref(chip0)", qb, e[0:1])
    stats("device(chip0)", a[0:1], e[0:1])


@pytest.mark.parametrize("fid", ["LoFi", "HiFi2", "HiFi3", "HiFi4"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_probe_fidelity(mesh_device, fid):
    chips, attn, w = _make_inputs(mesh_device, 0)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=False)
    out = a2a_wkv_b2(attn, w, compute_kernel_config=cfg)
    e = torch.stack(_expected(mesh_device, chips, w)).float()
    a = torch.stack([ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(out)])
    stats(f"device {fid}", a, e)
    from models.common.utility_functions import comp_pcc

    print(f"{fid} pcc={comp_pcc(e, a)[1]:.6f} rel_rms={(torch.sqrt(((a-e)**2).mean())/e.std()).item():.5f}")


@pytest.mark.parametrize("fid", ["LoFi", "HiFi2", "HiFi4"])
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_probe_stock_matmul(mesh_device, fid):
    """Comparator only (probe): stock ttnn.matmul on chip-local data, same dtypes/config."""
    chips, attn, w = _make_inputs(mesh_device, 0)
    a_t = ttnn.from_torch(
        chips[0],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cfg = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(), math_fidelity=getattr(ttnn.MathFidelity, fid), fp32_dest_acc_en=False, packer_l1_acc=False
    )
    y = ttnn.matmul(a_t, w, dtype=ttnn.bfloat8_b, compute_kernel_config=cfg)
    a = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float()
    wd = ttnn.to_torch(ttnn.get_device_tensors(w)[0]).double()
    e = torch.matmul(chips[0].double(), wd).float()
    stats(f"stock matmul {fid}", a, e)
