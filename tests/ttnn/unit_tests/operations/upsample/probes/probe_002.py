# Verifier probe: does the stick_stream RM path (validate() bypassed via _dispatch) already serve the
# TARGET - SUPPORTED values float32, non_tile_aligned C and per_axis factors?
import torch, ttnn
from ttnn.operations.upsample.upsample import _dispatch
dev = ttnn.open_device(device_id=0)
cases = [
    ((1,16,16,32), (2,2), ttnn.float32, ttnn.DRAM_MEMORY_CONFIG),
    ((1,40,40,256), (2,2), ttnn.float32, ttnn.DRAM_MEMORY_CONFIG),
    ((1,16,16,3), (2,2), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
    ((2,8,8,17), (2,2), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
    ((1,17,50,17), (2,2), ttnn.float32, ttnn.DRAM_MEMORY_CONFIG),
    ((1,32,32,50), (2,2), ttnn.bfloat16, ttnn.L1_MEMORY_CONFIG),
    ((1,16,32,32), (2,1), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
    ((1,32,16,32), (1,2), ttnn.float32, ttnn.L1_MEMORY_CONFIG),
    ((1,7,9,32), (3,2), ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
    ((2,8,8,50), (2,4), ttnn.float32, ttnn.DRAM_MEMORY_CONFIG),
]
try:
    for shape, (sh, sw), dt, mc in cases:
        torch.manual_seed(0)
        tdt = torch.float32 if dt == ttnn.float32 else torch.bfloat16
        t = torch.randn(*shape, dtype=tdt)
        x = ttnn.from_torch(t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
        try:
            got = ttnn.to_torch(_dispatch(x, sh, sw))
            ref = t.repeat_interleave(sh, dim=1).repeat_interleave(sw, dim=2)
            res = "exact" if torch.equal(got, ref) else f"MISMATCH frac={(got != ref).float().mean().item():.3f}"
        except Exception as e:
            res = f"EXC {type(e).__name__}: {str(e)[:150]}"
        print((shape, (sh, sw), str(dt), "L1" if mc == ttnn.L1_MEMORY_CONFIG else "DRAM", x.buffer_page_size()), res, flush=True)
finally:
    ttnn.close_device(dev)
