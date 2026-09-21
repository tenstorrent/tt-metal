import os, torch, ttnn
from pathlib import Path
from tt_lib.utils import tilize_to_list, untilize, pad_weight

dev = ttnn.open_device(device_id=0)
mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
torch.manual_seed(1234)
N,C,H,W = 1,9,384,1024
eps = 1e-2
x = torch.rand((N,C,H,W))*2-0.95
g = torch.rand(1,1,1,W)*2-1
ref = x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)*g.flatten()

for var in ("", "/localdev/dnijemcevic/2026_09_10_port/l1res/variants/chunk8"):
    import importlib, ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
    PD.KERNEL_DIR = Path(var) if var else Path(PD.__file__).parent/"kernels"
    from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn as op
    ttx = ttnn.Tensor(tilize_to_list(x), [N,C,H,W], ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mc)
    ttg = ttnn.Tensor(tilize_to_list(pad_weight(g)), [1,1,32,W], ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mc)
    z = op(ttx, epsilon=eps, weight=ttg, memory_config=mc)
    got = untilize(torch.Tensor(z.cpu().to_torch()).reshape((N,C,H,W)))
    a, b = ref.flatten().double(), got.flatten().double()
    pcc = torch.corrcoef(torch.stack([a,b]))[0,1].item()
    print(f"PCC_RESULT variant={'chunk8' if var else 'baseline'} pcc={pcc:.6f} maxabs={(a-b).abs().max().item():.5f}")
    z.deallocate(); ttx.deallocate(); ttg.deallocate()
ttnn.close_device(dev)
