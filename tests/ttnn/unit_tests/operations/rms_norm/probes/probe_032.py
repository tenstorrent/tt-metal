import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 4096, 256), (1, 1, 8192, 256), (1, 1, 4096, 512)]:
        torch.manual_seed(0)
        x = torch.randn(*shape)
        # check bh selection
        W = shape[-1]
        Wt = (W + 31) // 32
        Ht = 1
        for d in shape[:-1]:
            Ht *= d
        Ht //= 32
        cfg = pd._resolve_compute_config(None)
        bh = pd._regime_a_block_height(Ht, min(Ht, 64), False, 0, Wt, ttnn.bfloat16, True, cfg)
        ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        out = ttnn.to_torch(rms_norm(ti)).float()
        ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
        maxerr = (out - ref).abs().max().item()
        print(f"shape={shape} Ht={Ht} Wt={Wt} bh={bh} PCC={pcc:.6f} maxerr={maxerr:.4f}")
finally:
    ttnn.close_device(dev)
