"""uint8 investigation: positional encoding, look at the FAILURE STRUCTURE.
value(r,c) = r  (so a permutation shows as row bleed) and separately = c."""
import torch, ttnn
from ttnn.operations import tilize as T
import ttnn.operations.tilize.tilize_program_descriptor as pd

device = ttnn.open_device(device_id=0)
T.SUPPORTED["dtype"] = [ttnn.bfloat16, ttnn.uint8, ttnn.uint16, ttnn.uint32]
T.SUPPORTED["output_dtype"] = [ttnn.bfloat16, ttnn.uint8, ttnn.uint16, ttnn.uint32]

shape = (1, 1, 32, 32)


def run(dt, torch_dt, x):
    tt = ttnn.from_torch(
        x, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    print(f"  in page_size={tt.buffer_page_size()} elem={tt.element_size()}")
    out = T.tilize(tt, dtype=dt)
    print(f"  out page_size={out.buffer_page_size()} elem={out.element_size()}")
    return ttnn.to_torch(out)


# row index encoding
r = torch.arange(32).reshape(32, 1).expand(32, 32).contiguous()
c = torch.arange(32).reshape(1, 32).expand(32, 32).contiguous()

for dt, tdt in ((ttnn.uint8, torch.uint8), (ttnn.uint16, torch.int32)):
    for name, base in (("row=r", r), ("col=c", c), ("lin=r*32+c mod 251", (r * 32 + c) % 251)):
        x = base.to(tdt).reshape(shape)
        print(f"--- {dt} {name}")
        got = run(dt, tdt, x).reshape(32, 32).to(torch.int64)
        exp = x.reshape(32, 32).to(torch.int64)
        print("   exp row0[:16]:", exp[0, :16].tolist())
        print("   got row0[:16]:", got[0, :16].tolist())
        print("   exp col0[:16]:", exp[:16, 0].tolist())
        print("   got col0[:16]:", got[:16, 0].tolist())
        print("   nmismatch:", int((got != exp).sum()), flush=True)
ttnn.close_device(device)
