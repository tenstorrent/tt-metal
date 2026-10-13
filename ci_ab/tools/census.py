# Writer census for the reduce over W: a reduce with Wt > 1 must not compile the column writer, one with Wt = 1 must.
import glob, os, torch, ttnn

def count():
    return len(glob.glob(os.environ["TT_METAL_CACHE"] + "/**/writer_reduce_w_col_interleaved*/**/*.elf", recursive=True))

dev = ttnn.open_device(device_id=0)
for name, shape, fn, tfn in [("wt8_max", (1, 1, 32, 256), ttnn.max, lambda t: t.max(-1, keepdim=True).values),
                             ("wt1_sum", (1, 1, 64, 8), ttnn.sum, lambda t: t.sum(-1, keepdim=True))]:
    torch.manual_seed(0)
    t = torch.randn(shape).bfloat16()
    y = ttnn.to_torch(fn(ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev), dim=-1, keepdim=True)).float()
    err = (y - tfn(t.float())).abs().max().item()
    print(f"CENSUS {name} column writer ELFs {count()} max abs err {err:.4g}", flush=True)
ttnn.close_device(dev)
