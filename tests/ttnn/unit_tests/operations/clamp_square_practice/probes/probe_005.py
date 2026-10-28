import ttnn, numpy as np
from ttnn.operations.clamp_square_practice.perf_experiments.bank_affine_split import variant as V

d = ttnn.open_device(device_id=0)
cores = [(x, y) for y in range(10) for x in range(12)]
caps = [137] * 64 + [136] * 56
print("G", V.bank_groups(d, "col"), V.bank_groups(d, "pair"))
for m in ["col_near", "col_nearrd", "col_nearest"]:
    segs, cnt = V.assign(d, cores, caps, 16384, m)
    print("M", m)
    for y in range(10):
        print(
            "r",
            " ".join(
                ("".join("W" if G[0] == 0 else "E" for G, _, _ in segs[y * 12 + x]) + str(cnt[y * 12 + x])).rjust(6)
                for x in range(12)
            ),
        )
ttnn.close_device(d)
