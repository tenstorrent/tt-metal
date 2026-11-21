import torch, ttnn
from ttnn.operations.upsample import upsample
dev = ttnn.open_device(device_id=0)
try:
    for shape, sf in [((1,16,16,32), 2), ((1,7,9,32),(3,3)), ((1,20,20,512),(2,2))]:
        torch.manual_seed(0)
        t = torch.randn(*shape, dtype=torch.bfloat16)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        y = upsample(x, sf)
        s = (sf, sf) if isinstance(sf, int) else sf
        ref = t.repeat_interleave(s[0], dim=1).repeat_interleave(s[1], dim=2)
        got = ttnn.to_torch(y)
        print(shape, sf, list(got.shape), "exact" if torch.equal(got, ref) else f"MISMATCH {(got.float()-ref.float()).abs().max()}", y.memory_config().buffer_type)
finally:
    ttnn.close_device(dev)
