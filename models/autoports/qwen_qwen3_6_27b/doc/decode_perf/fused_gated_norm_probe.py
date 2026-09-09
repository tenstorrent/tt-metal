"""Can the fused KDA gated norm serve a swish-gated model?

The op computes ``rms_norm(x) * weight * sigmoid(gate)`` and repacks
``[B*H, T, V] -> [B, T, H*V]``.  Qwen3.5 wants ``rms_norm(x) * weight * silu(z)``
and ``silu(z) = z * sigmoid(z)``, so passing ``gate = z`` and multiplying the
op's output by ``z`` once -- in the packed [B, T, H*V] layout the op already
returns, where z is natively available -- is the same function.

This checks that identity against the shipped six-op sequence.  It is a *negative*
result: the op requires a tile-aligned sequence and decode has T = 1, so it
rejects the call before the identity can be exercised.  Kept so the blockage is
reproducible rather than remembered.
"""

import time

import torch

import ttnn

B, H, D = 32, 12, 128
G = B * H
W = H * D


def timed(fn, device, iters=30):
    fn()
    ttnn.synchronize_device(device)
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(device)
    return 1e6 * (time.perf_counter() - t) / iters


def main():
    d = ttnn.open_device(device_id=0)
    try:
        mc = ttnn.DRAM_MEMORY_CONFIG
        mk = dict(device=d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=mc)
        att_t = torch.randn(B, H, 1, D, dtype=torch.bfloat16) * 0.5
        z_t = torch.randn(1, 1, B, W, dtype=torch.bfloat16) * 0.5
        w_t = torch.randn(1, 1, 1, D, dtype=torch.bfloat16) * 0.2 + 1.0
        att = ttnn.from_torch(att_t, **mk)
        z = ttnn.from_torch(z_t, **mk)
        weight = ttnn.from_torch(w_t, **mk)
        eps = 1e-6

        def shipped():
            out = ttnn.rms_norm(att, epsilon=eps, weight=weight, memory_config=mc)
            zz = ttnn.reshape(z, (B, H, 1, D))
            out = ttnn.multiply(out, ttnn.silu(zz))
            return ttnn.reshape(ttnn.permute(out, (2, 0, 1, 3)), (1, 1, B, W))

        ref = ttnn.to_torch(shipped()).float()
        print("shipped six-op tail:", tuple(ref.shape))

        # The op wants rank-3 operands: input [B*H, T, V], gate [B, T, H*V].
        head_first = ttnn.reshape(att, (G, 1, D))
        gate3 = ttnn.reshape(z, (B, 1, W))
        fused = None
        for wt in (weight, ttnn.reshape(weight, (1, 1, D)), ttnn.reshape(weight, (D,))):
            try:
                fused = ttnn.experimental.kda.sigmoid_gated_rms_norm(
                    head_first, gate3, wt, num_heads=H, epsilon=eps, memory_config=mc
                )
                print("accepted with weight rank", len(wt.shape))
                break
            except Exception as exc:
                reason = str(exc).split("info:")[0].strip().rsplit(":", 1)[-1].strip()
                print(f"  weight rank {len(wt.shape)}: rejected -- {reason}")

        us_shipped = timed(lambda: ttnn.deallocate(shipped()), d)
        print(f"  shipped six-op tail: {us_shipped:7.1f} us")

        if fused is None:
            print(
                "\nBLOCKED, on shape rather than on the swish/sigmoid identity.\n"
                "The op wants input [B*H, T, V], gate [B, T, H*V], weight [V] and a\n"
                "tile-aligned sequence; decode has T = 1, so the last check fails:\n"
                "  attrs.sequence > 0 && attrs.sequence % TILE_HEIGHT == 0\n"
                "Reaching it needs the same user-major packing the fused conv needed, plus\n"
                "a permute of the padded [batch, heads, 1, dim] matmul output into the\n"
                "head-major layout the op expects. See doc/decode_perf."
            )
            return

        got = ttnn.to_torch(ttnn.multiply(fused, gate3, memory_config=mc)).float()
        print("fused + one multiply:", tuple(got.shape))
        a, b = ref.reshape(-1), got.reshape(-1)
        pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        print(
            f"  pcc {pcc:.9f}  max|diff| {float((a - b).abs().max()):.6f}  "
            f"mean|ref| {float(a.abs().mean()):.4f}  identical {bool(torch.equal(a, b))}"
        )
    finally:
        ttnn.close_device(d)


if __name__ == "__main__":
    main()
