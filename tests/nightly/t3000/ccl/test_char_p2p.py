"""Characterize the p2p multi-page failure: what exactly triggers it?"""
import pytest, torch, ttnn
from tests.ttnn.utils_for_testing import assert_equal

MESH = (1, 4)

# (shape, layout, mem, label) -- page_size_bytes for RM = last_dim*2 (bf16)
CASES = [
    ((1, 1, 8, 16), ttnn.ROW_MAJOR_LAYOUT, "dram", "8pg_x_32B_DRAM_unaligned"),  # known FAIL
    ((1, 1, 8, 32), ttnn.ROW_MAJOR_LAYOUT, "dram", "8pg_x_64B_DRAM_aligned"),
    ((1, 1, 8, 16), ttnn.ROW_MAJOR_LAYOUT, "l1", "8pg_x_32B_L1_unaligned"),
    ((1, 1, 2, 16), ttnn.ROW_MAJOR_LAYOUT, "dram", "2pg_x_32B_DRAM_unaligned"),
    ((1, 1, 4, 16), ttnn.ROW_MAJOR_LAYOUT, "dram", "4pg_x_32B_DRAM_unaligned"),
    ((1, 1, 16, 16), ttnn.ROW_MAJOR_LAYOUT, "dram", "16pg_x_32B_DRAM_unaligned"),
    ((1, 1, 8, 24), ttnn.ROW_MAJOR_LAYOUT, "dram", "8pg_x_48B_DRAM_unaligned"),
    ((1, 1, 3, 128), ttnn.ROW_MAJOR_LAYOUT, "dram", "3pg_x_256B_DRAM_aligned"),
]


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("shape,layout,mem,label", CASES, ids=[c[3] for c in CASES])
def test_char(mesh_device, shape, layout, mem, label):
    from math import prod

    dtype = torch.bfloat16
    devices = prod(list(mesh_device.shape))
    mds = tuple(s * (devices if i == 0 else 1) for i, s in enumerate(shape))
    t = torch.zeros(mds, dtype=dtype)
    t[0 : shape[0]] = torch.linspace(1, prod(shape), prod(shape)).reshape(shape).to(dtype)
    mc = ttnn.DRAM_MEMORY_CONFIG if mem == "dram" else ttnn.L1_MEMORY_CONFIG
    inp = ttnn.from_torch(
        t, layout=layout, device=mesh_device, memory_config=mc, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
    )
    sent = ttnn.point_to_point(
        inp, ttnn.MeshCoordinate((0, 0)), ttnn.MeshCoordinate((0, 1)), topology=ttnn.Topology.Linear
    )
    got = ttnn.to_torch(sent, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    exp = t[0 : shape[0]]
    act = got[shape[0] : 2 * shape[0]]
    bad = (exp != act).reshape(-1, shape[-1]).any(dim=1).nonzero().flatten().tolist()
    print(f"\n[{label}] page_bytes={shape[-1]*2} num_pages={prod(shape[:-1])} BAD_ROWS={bad}", flush=True)
    assert_equal(exp, act)
