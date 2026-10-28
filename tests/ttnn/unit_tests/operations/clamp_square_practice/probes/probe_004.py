import ttnn, numpy as np
from ttnn.operations.clamp_square_practice.perf_experiments.bank_affine_split import variant as V

d = ttnn.open_device(device_id=0)
cores = [(x, y) for y in range(10) for x in range(12)]
print("A0", V._anchors(d, ttnn.NOC.NOC_0))
print("A1", V._anchors(d, ttnn.NOC.NOC_1))
C = V.cost_matrix(d, cores)
own = C.argmin(1)
print("own grid")
for y in range(10):
    print("own", "".join(str(own[y * 12 + x]) for x in range(12)))
segs, cnt = V.assign(d, cores, [137] * 64 + [136] * 56, 16384, "near")
for y in range(10):
    print("seg", " ".join("".join(str(b) for b, _, _ in segs[y * 12 + x]).ljust(3) for x in range(12)))
segs, cnt = V.assign(d, cores, [137] * 64 + [136] * 56, 16384, "nearest")
print("load nearest", sorted(set(cnt)))
for y in range(10):
    print("seg", " ".join("".join(str(b) for b, _, _ in segs[y * 12 + x]).ljust(3) for x in range(12)))
ttnn.close_device(d)
