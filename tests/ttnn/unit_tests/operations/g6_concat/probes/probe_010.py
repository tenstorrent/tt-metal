import torch, ttnn
from ttnn.operations.g6_concat import g6_concat
from eval.golden_tests.g6_concat.helpers import pytorch_g6_concat, place_special_values

dev = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(0)
    xs = place_special_values([torch.randn(1, 6, 31, 64).bfloat16(), torch.randn(1, 6, 31, 192).bfloat16()])
    tt = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for x in xs]
    out = ttnn.to_torch(g6_concat(tt, -1)).view(torch.int16)
    ref = torch.cat(xs, -1).view(torch.int16)
    ora = pytorch_g6_concat(xs, -1).view(torch.int16)
    print("OUT   ", hex(out[0, 0, 0, 0].item() & 0xFFFF), hex(out[0, 0, 0, 64].item() & 0xFFFF))
    print("REF   ", hex(ref[0, 0, 0, 0].item() & 0xFFFF), hex(ref[0, 0, 0, 64].item() & 0xFFFF))
    print("ORACLE", hex(ora[0, 0, 0, 0].item() & 0xFFFF), hex(ora[0, 0, 0, 64].item() & 0xFFFF))
    print("EQ torch.cat bits:", bool(torch.equal(out, ref)), " EQ oracle bits:", bool(torch.equal(out, ora)))
finally:
    ttnn.close_device(dev)
