import sys

sys.path.insert(0, "/home/iguser/Christy/tt-metal")
import pytest, torch, ttnn
from loguru import logger
from models.common.utility_functions import comp_pcc
import models.experimental.hunyuan_image_3_0.tt.vae.decoder as dec


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("out,r1", [(8, 1), (64, 1), (64, 2), (128, 2)])
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_d2s_chunk_equiv(mesh_device, out, r1):
    b, t, h, w, r2, r3 = 1, 2, 16, 16, 2, 2
    cin = out * r1 * r2 * r3
    torch.manual_seed(0)
    x = torch.randn(b, t, h, w, cin)
    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run():
        o = dec.dcae_depth_to_space_bthwc(x_tt, out_channels=out, r1=r1, r2=r2, r3=r3)
        return ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:b].float()

    saved = dec._D2S_CHUNK_ELEMS
    try:
        dec._D2S_CHUNK_ELEMS = 10**18
        full = run()
        dec._D2S_CHUNK_ELEMS = 1
        chunk = run()
    finally:
        dec._D2S_CHUNK_ELEMS = saved
    p = comp_pcc(full, chunk, 0.999)
    logger.info(
        f"d2s chunk-vs-full out={out} r1={r1}: PCC={p[1]:.6f} full={tuple(full.shape)} chunk={tuple(chunk.shape)}"
    )
    assert p[0], f"out={out} r1={r1} PCC={p[1]}"
