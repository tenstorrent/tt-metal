import ttnn, numpy as np
from ttnn.operations.clamp_square_practice.perf_experiments.bank_affine_split import variant as V

d = ttnn.open_device(device_id=0)
cores = [(x, y) for y in range(10) for x in range(12)]
caps = [137] * 64 + [136] * 56
for m in ["col_near"]:
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
    c = V.cost_matrix(d, cores)
    cg = np.stack([c[:, [0, 1, 2, 3]].mean(1), c[:, [4, 5, 6, 7]].mean(1)], 1)
    for lp in (False, True):
        segs, _ = V.assign(d, cores, caps, 16384, m, lp=lp)
        tot = sum(cg[i, 0 if G[0] == 0 else 1] * n for i, s in enumerate(segs) for G, _, n in s)
        print("same cost lp=", lp, tot)
ttnn.close_device(d)
