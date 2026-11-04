import torch, ttnn, math
import ttnn.operations.nomic_residual_layer_norm.nomic_residual_layer_norm as m
from ttnn.operations.nomic_residual_layer_norm import nomic_residual_layer_norm, input_memory_config

m.NUM_ROW_BLOCKS = 2
dev = ttnn.open_device(device_id=0)
try:
    for shape in [(8, 1, 512, 768), (4, 1, 512, 768), (8, 1, 500, 768)]:
        for res in (True, False):
            torch.manual_seed(0)
            x = torch.randn(shape).bfloat16()
            r = torch.randn(shape).bfloat16() if res else None
            g = 1 + 0.25 * torch.randn(768)
            b = (0.1 * torch.randn(768)).bfloat16()
            mc = input_memory_config(shape, dev)
            tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
            tr = (
                ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
                if res
                else None
            )
            tg = ttnn.from_torch(g.reshape(1, 1, 1, 768), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
            tb = ttnn.from_torch(b.reshape(1, 1, 1, 768), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            out = ttnn.to_torch(nomic_residual_layer_norm(tx, tg, tb, residual=tr)).double()
            y = x.double() + (r.double() if res else 0)
            ref = torch.nn.functional.layer_norm(y, (768,), g.double(), b.double(), eps=1e-5)
            pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
            rms = ((out - ref) ** 2).mean().sqrt().item() / ref.std().item()
            print(f"PROBE shape={shape} res={res} nb=2 pcc={pcc:.7f} rms={rms:.5f}")
finally:
    ttnn.close_device(dev)
