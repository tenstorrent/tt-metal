import torch, ttnn
from ttnn.operations.pad import pad
from ttnn.operations.pad.pad import _run
dev = ttnn.open_device(device_id=0)
try:
    for shape, padding, val in [((1,3,20,20), ((0,0),(0,13),(0,0),(0,0)), 0.0),
                                ((1,3,20,20), ((0,0),(0,13),(0,0),(0,0)), 1.0),
                                ((4,8,47,256), ((0,2),(0,3),(0,1),(0,4)), 1.0),
                                ((1,1,17,50), ((0,0),(0,1),(0,0),(0,1)), 1.0)]:
        torch.manual_seed(0)
        t = torch.randn(shape, dtype=torch.bfloat16)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        y = _run(x, padding, val) if val != 0.0 else pad(x, padding, val)
        got = ttnn.to_torch(y)
        flat = [a for pr in reversed(padding) for a in pr]
        ref = torch.nn.functional.pad(t, flat, value=val)
        print(shape, val, tuple(got.shape), tuple(ref.shape), "EXACT" if torch.equal(got, ref) else f"MISMATCH {(got.float()-ref.float()).abs().max()}  nbad={(got!=ref).sum()}")
finally:
    ttnn.close_device(dev)
