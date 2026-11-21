import torch, ttnn, traceback
from ttnn.operations.permute import permute
from ttnn.operations.permute.permute_program_artifacts import canonical, select_regime
dev = ttnn.open_device(device_id=0)
cases = [
  ((1,4,16,20), (0,2,1,3), ttnn.ROW_MAJOR_LAYOUT),   # page_gather RM
  ((1,2,64,64), (1,0,2,3), ttnn.TILE_LAYOUT),        # page_gather TILE
  ((1,4,16,400), (0,1,3,2), ttnn.TILE_LAYOUT),       # tile_transpose
  ((1,16,20,20), (0,2,3,1), ttnn.ROW_MAJOR_LAYOUT),  # staged RM
  ((1,4,400,1), (0,3,1,2), ttnn.TILE_LAYOUT),        # staged TILE
]
for shape, dims, layout in cases:
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, layout=layout, device=dev, dtype=ttnn.bfloat16)
    s, d = canonical(shape, dims)
    try:
        y = ttnn.to_torch(permute(t, dims))
        ref = x.permute(dims)
        ok = y.shape == ref.shape and torch.equal(y, ref)
        print("CASE", shape, dims, layout, select_regime(layout, d), "OK" if ok else f"MISMATCH {(y.float()-ref.float()).abs().max() if y.shape==ref.shape else y.shape}")
    except Exception as e:
        print("CASE", shape, dims, layout, "ERR", str(e)[:800])
ttnn.close_device(dev)
