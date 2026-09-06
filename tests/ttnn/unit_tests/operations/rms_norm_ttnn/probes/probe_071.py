import torch, ttnn
import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn

device = ttnn.open_device(device_id=0)
try:
    shape = (1, 1, 3104, 4064)
    for floor in (1, 2, 4, 8):
        PD.ROW_RESIDENT_MIN_CHUNK_WT = floor
        torch.manual_seed(0)
        t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        lay = ttnn.ROW_MAJOR_LAYOUT
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=lay, device=device)
        g = torch.randn(1, 1, 1, 4064, dtype=torch.float32).to(torch.bfloat16)
        r = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        kw = dict(
            weight=ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=lay, device=device),
            bias=ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=lay, device=device),
            residual_input_tensor=ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=lay, device=device),
        )
        print("CASE floor", floor, flush=True)
        out = rms_norm_ttnn(x, epsilon=1e-12, **kw)
        ttnn.deallocate(out)
finally:
    ttnn.close_device(device)
