import pytest
import torch

import ttnn


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
def test_probe(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    pos = torch.tensor([9, 130, 255, 3], dtype=torch.int32).reshape(4, 1)
    idx = ttnn.from_torch(pos, device=md, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)
    tab = (torch.arange(512).reshape(512, 1) * torch.ones(1, 32)).float()  # row p = p
    for name, dt, lay in (
        ("bf16 RM", ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
        ("fp32 RM", ttnn.float32, ttnn.ROW_MAJOR_LAYOUT),
        ("uint32 RM", ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
        ("int32 RM", ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
    ):
        try:
            w = ttnn.from_torch(
                tab if dt in (ttnn.bfloat16, ttnn.float32) else tab.to(torch.int32),
                device=md,
                dtype=dt,
                layout=lay,
                mesh_mapper=rep,
            )
            for ol in (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT):
                try:
                    o = ttnn.embedding(idx, w, layout=ol)
                    got = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).reshape(4, -1)[:, 0].tolist()
                    print(f"PROBE {name:10s} out {str(ol):22s} dtype {o.dtype} ->", got, flush=True)
                except Exception as e:
                    print(f"PROBE {name:10s} out {str(ol):22s} FAILED {str(e)[:90]}", flush=True)
        except Exception as e:
            print(f"PROBE {name:10s} upload FAILED {str(e)[:90]}", flush=True)
