import torch, ttnn
from models.experimental.pi0_5.tt.tile_config import from_torch_pi05


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    sa, sb = a.std(), b.std()
    if sa < 1e-6 or sb < 1e-6:
        return float("nan")
    return (torch.mean((a - a.mean()) * (b - b.mean())) / (sa * sb)).item()


dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
try:
    torch.manual_seed(0)
    for shape in [(1, 8, 16, 256), (1, 1, 16, 256), (1, 1, 1040, 256), (1, 1, 1056, 256), (1, 8, 32, 256)]:
        t = torch.randn(*shape) * 0.1
        td = from_torch_pi05(t, dtype=ttnn.bfloat8_b, device=dev, memory_config=ttnn.L1_MEMORY_CONFIG)
        print(f"{str(shape):20s} input tile {td.get_tile().tile_shape} dtype {td.dtype} layout {td.layout}")
        try:
            o = ttnn.tilize(td, tile=ttnn.Tile((32, 32)), dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG)
            ot = ttnn.to_torch(o)
            print(
                f"{str(shape):20s} in_tile{td.get_tile().tile_shape} -> out{tuple(o.shape)} tile{o.get_tile().tile_shape} dtype {o.dtype} preservePCC {round(pcc(t, ot[:,:,:shape[2],:]),5)}"
            )
            ttnn.deallocate(o)
        except Exception as e:
            print(f"{str(shape):20s} FAILED: {str(e)[:140]}")
        ttnn.deallocate(td)
finally:
    ttnn.close_mesh_device(dev)
