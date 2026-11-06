import sys, torch, ttnn

sys.path.insert(0, "tests/ttnn/unit_tests/operations/mhc_post")
import test_mhc_post as T
from ttnn.operations.mhc_post import mhc_post

device = ttnn.open_device(device_id=0)
try:
    for x_shape in [(1, 1, 64, 4 * 640), (1, 1, 160, 4 * 5120)]:
        f, x, post, comb, pre, mixes = T._packed(x_shape, ttnn.bfloat16)
        nc = x_shape[-1]
        C = nc // 4
        w = T._weight(nc)
        hc = T._hc_scalars(nc)
        m = T.dev(mixes, device)
        outs = mhc_post(
            T.dev(f, device, ttnn.bfloat16),
            T.dev(x, device, ttnn.bfloat16),
            m,
            m,
            post_col=4,
            comb_col=8,
            stats_weight=T.dev(w.reshape(1, 1, nc, 32), device),
            compute_kernel_config=T.cfg(ttnn.MathFidelity.HiFi2),
            collapse_pre=m,
            hc_scale=hc["scale"],
            hc_base=hc["base"],
            hc_norm_width=hc["norm_width"],
        )
        xo = ttnn.to_torch(outs[0]).double()
        y = ttnn.to_torch(outs[3]).double()
        xs = xo.reshape(*xo.shape[:-1], 4, C)
        yref = (pre.double()[..., :4].unsqueeze(-1) * xs).sum(-2)
        err = (y - yref).abs()
        T_ = x_shape[-2]
        e = err.reshape(T_, C)
        rows = (T_ + 31) // 32
        tile_err = torch.zeros(rows, C // 32)
        for r in range(rows):
            for c in range(C // 32):
                tile_err[r, c] = e[r * 32 : (r + 1) * 32, c * 32 : (c + 1) * 32].max()
        print(x_shape, "max err", err.max().item(), "yref absmax", yref.abs().max().item())
        bad = tile_err > 0.05
        print("bad tiles", bad.sum().item(), "of", bad.numel())
        print(bad.int()[:, :40])
        # check if y equals some single stream
        for i in range(4):
            print(i, (y - xs[..., i, :] * pre.double()[..., i : i + 1]).abs().max().item())
finally:
    ttnn.close_device(device)
