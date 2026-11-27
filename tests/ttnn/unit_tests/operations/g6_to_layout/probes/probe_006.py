import torch, ttnn
from ttnn.operations.g6_to_layout import g6_to_layout

device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1024, 1024, 3), (1, 64, 128, 256), (16, 1, 2560)]:
        torch.manual_seed(0)
        xs = [torch.randn(shape, dtype=torch.bfloat16) for _ in range(3)]
        ins = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for x in xs]
        outs = []
        for t in ins:
            outs.append(g6_to_layout(t, ttnn.ROW_MAJOR_LAYOUT))  # all inputs and outputs stay alive: distinct addresses
        addrs = [(t.buffer_address(), o.buffer_address()) for t, o in zip(ins, outs)]
        ok = all(
            torch.equal(ttnn.to_torch(o).reshape(shape).view(torch.int16), x.view(torch.int16))
            for o, x in zip(outs, xs)
        )
        print("RESULT", shape, "distinct addrs", len(set(addrs)) == 3, "all match", ok)
finally:
    ttnn.close_device(device)
