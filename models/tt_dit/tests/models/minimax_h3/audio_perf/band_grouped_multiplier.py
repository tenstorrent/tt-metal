"""Grouped conv with a channel multiplier: in=C, out=2C, groups=C.

Would give both polyphase phases from one pass with no duplicate of x. But
`is_1d_depthwise_conv` requires groups == in == out, so out=2C leaves the depthwise path -- which is
where both the SFPU fp32 tap accumulation and the fused snake live. Measure correctness and speed
against the two depthwise convs it would replace.

Reference layout: group c owns input channel c and output channels 2c, 2c+1.
"""
import statistics
import time

import torch

import ttnn
from models.tt_dit.layers.audio_ops import depthwise_tap_filter


def med(fn, dev, iters=5):
    fn()
    ttnn.synchronize_device(dev)
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        s.append((time.perf_counter() - t0) * 1e3)
        if o is not None:
            ttnn.deallocate(o)
    return statistics.median(s)


d = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
K = 7
try:
    cc = ttnn.init_device_compute_kernel_config(
        d.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    for C, M in [(8, 165606), (32, 41403)]:
        B = 2
        torch.manual_seed(0)
        sub0 = torch.randn(K)
        sub1 = torch.randn(K)
        xt = torch.randn(B, M, C) * 0.3
        x = ttnn.from_torch(xt, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=d)
        cache = {}
        s0 = ttnn.to_torch(
            depthwise_tap_filter(x, sub0.tolist(), 1, mesh_device=d, dtype=ttnn.float32, cache=cache)
        ).float()
        s1 = ttnn.to_torch(
            depthwise_tap_filter(x, sub1.tolist(), 1, mesh_device=d, dtype=ttnn.float32, cache=cache)
        ).float()

        def pair():
            a = depthwise_tap_filter(x, sub0.tolist(), 1, mesh_device=d, dtype=ttnn.float32, cache=cache)
            b = depthwise_tap_filter(x, sub1.tolist(), 1, mesh_device=d, dtype=ttnn.float32, cache=cache)
            ttnn.deallocate(a)
            return b

        t_pair = med(pair, d)

        w = torch.zeros(2 * C, 1, K)
        for c in range(C):
            w[2 * c, 0] = sub0
            w[2 * c + 1, 0] = sub1
        wd = ttnn.from_torch(
            w, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.float32, mesh_mapper=ttnn.ReplicateTensorToMesh(d)
        )
        cfg = ttnn.Conv1dConfig(weights_dtype=ttnn.float32, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)

        def grouped():
            o, _ = ttnn.conv1d(
                input_tensor=ttnn.reshape(x, (B, M, 1, C)),
                weight_tensor=wd,
                device=d,
                in_channels=C,
                out_channels=2 * C,
                batch_size=B,
                input_length=M,
                kernel_size=K,
                stride=1,
                padding=0,
                dilation=1,
                groups=C,
                dtype=ttnn.float32,
                conv_config=cfg,
                compute_config=cc,
                return_output_dim=True,
            )
            return o

        try:
            og = grouped()
            g = ttnn.to_torch(ttnn.to_layout(og, ttnn.ROW_MAJOR_LAYOUT)).float().reshape(-1, 2 * C)
            n = s0.shape[1]
            g = g[: n * B].reshape(B, n, 2 * C) if g.shape[0] >= n * B else g
            e0 = float((g[:, :, 0::2].double() - s0.double()).pow(2).mean().sqrt() / s0.double().std())
            e1 = float((g[:, :, 1::2].double() - s1.double()).pow(2).mean().sqrt() / s1.double().std())
            ttnn.deallocate(og)
            t_g = med(grouped, d)
            print(
                f"C={C:<3} M={M}: pair {t_pair:.2f} ms | grouped {t_g:.2f} ms ({t_pair/t_g:.2f}x) | "
                f"rel_rmse ph0 {e0:.3e} ph1 {e1:.3e}",
                flush=True,
            )
        except Exception as e:
            print(f"C={C:<3} M={M}: GROUPED FAILED -- {type(e).__name__}: {str(e).splitlines()[0][:150]}", flush=True)
        ttnn.deallocate(x)
finally:
    ttnn.close_mesh_device(d)
