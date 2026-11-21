# Verifier probe: are the design's "validate-only" knob-turn values already served by the Phase 0 kernels?
# SUPPORTED is widened in this process only.
import torch, ttnn
import sys, ttnn.operations.slice
S = sys.modules["ttnn.operations.slice.slice"]
S.SUPPORTED.update({"dtype": [ttnn.bfloat16, ttnn.float32, ttnn.uint32], "rank": [2, 3, 4],
                    "sliced_region": ["outer", "h", "w"], "start_alignment": ["tile_aligned", "non_tile_aligned"]})
device = ttnn.open_device(device_id=0)
TL, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
D, L = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
cases = [
  ("rank2_h_tile", (128, 512), (32, 0), (96, 512), ttnn.bfloat16, TL, D),
  ("rank2_h_rm", (17, 64), (3, 0), (17, 64), ttnn.bfloat16, RM, L),
  ("rank4_h_tile", (2, 4, 128, 512), (0, 0, 32, 0), (2, 4, 96, 512), ttnn.bfloat16, TL, D),
  ("rank4_outer_rm", (4, 8, 32, 256), (1, 2, 0, 0), (3, 6, 32, 256), ttnn.bfloat16, RM, D),
  ("tile_w_aligned", (4, 128, 512), (0, 32, 96), (4, 96, 352), ttnn.bfloat16, TL, D),
  ("tile_w_wide", (1, 1, 32, 4096), (0, 0, 0, 0), (1, 1, 32, 2048), ttnn.bfloat16, TL, L),
  ("rm_h_unaligned_start", (1, 96, 128), (0, 16, 0), (1, 80, 128), ttnn.bfloat16, RM, D),
  ("rm_h_unaligned_start_4d", (4, 8, 47, 256), (0, 0, 15, 0), (4, 8, 47, 256), ttnn.bfloat16, RM, L),
  ("fp32_tile", (4, 128, 512), (0, 32, 0), (4, 96, 512), ttnn.float32, TL, D),
  ("fp32_rm_wide", (1, 144, 8400), (0, 64, 0), (1, 144, 8400), ttnn.float32, RM, D),
  ("fp32_rm_small", (4, 128, 47), (0, 64, 0), (4, 100, 47), ttnn.float32, RM, L),
  ("u32_tile", (4, 128, 512), (0, 32, 0), (4, 96, 512), ttnn.uint32, TL, D),
  ("u32_rm", (4, 128, 47), (0, 64, 0), (4, 100, 47), ttnn.uint32, RM, L),
]
ok = 0
for name, shape, s, e, dt, lay, mem in cases:
    try:
        if dt == ttnn.uint32:
            t = torch.randint(-1000, 1000, shape, dtype=torch.int32)
        else:
            t = torch.randn(shape, dtype=torch.float32 if dt == ttnn.float32 else torch.bfloat16)
        x = ttnn.from_torch(t, dtype=dt, layout=lay, device=device, memory_config=mem)
        got = ttnn.to_torch(S.slice(x, list(s), list(e)))
        exp = t[tuple(slice(a, b) for a, b in zip(s, e))]
        exact = list(got.shape) == list(exp.shape) and torch.equal(got.to(exp.dtype), exp)
        ok += exact
        print("CASE", name, "EXACT" if exact else "MISMATCH", flush=True)
    except Exception as ex:
        print("CASE", name, "ERR", type(ex).__name__, str(ex).splitlines()[0][:200], flush=True)
print("SUMMARY", ok, "/", len(cases))
ttnn.close_device(device)
