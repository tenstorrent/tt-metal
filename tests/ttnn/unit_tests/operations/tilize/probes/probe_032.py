"""uint8 stage isolation: where do the bytes die — unpack/math, or pack?
   (a) uint8 in -> bf16 out  : exercises unpack+math with a KNOWN-GOOD packer.
   (b) bf16  in -> uint8 out : exercises the UInt8 packer with a known-good unpack.
   Neither pair is in TARGET; both are pure diagnostics."""
import torch, ttnn
from ttnn.operations import tilize as T

device = ttnn.open_device(device_id=0)
T.SUPPORTED["dtype"] = [ttnn.bfloat16, ttnn.uint8, ttnn.uint16]
T.SUPPORTED["output_dtype"] = [ttnn.bfloat16, ttnn.uint8, ttnn.uint16]

lin = (torch.arange(32).reshape(32, 1) * 32 + torch.arange(32).reshape(1, 32)) % 100


def go(in_dt, out_dt, torch_in_dt):
    x = lin.to(torch_in_dt).reshape(1, 1, 32, 32)
    tt = ttnn.from_torch(
        x, dtype=in_dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = T.tilize(tt, dtype=out_dt)
    got = ttnn.to_torch(out).reshape(32, 32).to(torch.float64)
    exp = lin.to(torch.float64)
    print(f"  {in_dt} -> {out_dt}: nmismatch={int((got!=exp).sum())}/1024")
    print(f"    got row0[:10]={got[0,:10].tolist()}")
    print(f"    exp row0[:10]={exp[0,:10].tolist()}", flush=True)


go(ttnn.uint8, ttnn.bfloat16, torch.uint8)
go(ttnn.bfloat16, ttnn.uint8, torch.bfloat16)
go(ttnn.uint8, ttnn.uint16, torch.uint8)
go(ttnn.uint16, ttnn.uint8, torch.int32)
ttnn.close_device(device)
