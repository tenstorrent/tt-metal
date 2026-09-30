import torch, ttnn
from eval.golden_tests.mhc_pre.helpers import make_compute_config
from ttnn.operations.mhc_pre import mhc_pre

device = ttnn.open_device(device_id=0)
try:
    nC = 128
    x = torch.ones((32, nC))
    b = torch.zeros((1, 24))
    ms = list(range(6, 22))
    # column 4+i of W: w[0]=0.5, w[k1]=0.5*2^-m (post columns 4..7 -> 4 m's per call)
    for same_tile in (True, False):
        res = []
        for m0 in range(0, len(ms), 4):
            w = torch.zeros((nC, 24))
            chunk = ms[m0 : m0 + 4]
            for i, m in enumerate(chunk):
                w[0, 4 + i] = 0.5
                w[1 if same_tile else 32, 4 + i] = 0.5 * (1.0 + 0.0) * 2.0**-m * 1.5  # two bits: 1.1b * 2^-m
            dev = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            _, pd, _ = mhc_pre(
                dev(x), dev(w), dev(b), scale=(1.0, 1.0, 1.0), compute_kernel_config=make_compute_config()
            )
            p = ttnn.to_torch(pd).double().reshape(-1, 4)
            r = 1 / (1 + 1e-6) ** 0.5
            z_dev = torch.log((p / 2) / (1 - p / 2))
            mix_dev = (z_dev / r)[0]
            for i, m in enumerate(chunk):
                small = mix_dev[i].item() - 0.5
                res.append(f"m={m}: small/expected={small/(0.5*1.5*2.0**-m):.4f}")
        print("PROBE same_tile" if same_tile else "PROBE other_tile", res)
finally:
    ttnn.close_device(device)
