import torch, ttnn
from ttnn.operations._op_contract import UnsupportedAxisValue
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention as sdpa

device = ttnn.open_device(device_id=0)


def T(shape, dt=ttnn.bfloat16):
    return ttnn.from_torch(torch.randn(*shape), dtype=dt, layout=ttnn.TILE_LAYOUT, device=device)


def expect_raise(fn, exc, tag):
    try:
        fn()
        print(f"BAD {tag}: no raise")
    except exc:
        print(f"OK {tag}: raised {exc.__name__}")
    except Exception as e:
        print(f"OK {tag}: raised {type(e).__name__} ({e})"[:80])


# rank 3
expect_raise(lambda: sdpa(T([1, 32, 32]), T([1, 1, 32, 32]), T([1, 1, 32, 32])), ValueError, "rank3")
# head_dim mismatch
expect_raise(lambda: sdpa(T([1, 1, 32, 32]), T([1, 1, 32, 64]), T([1, 1, 32, 64])), ValueError, "D_mismatch")
# is_causal + mask
expect_raise(
    lambda: sdpa(T([1, 1, 32, 32]), T([1, 1, 32, 32]), T([1, 1, 32, 32]), attn_mask=T([1, 1, 32, 32]), is_causal=True),
    ValueError,
    "causal+mask",
)
# H_q % H_kv != 0
expect_raise(lambda: sdpa(T([1, 3, 32, 32]), T([1, 2, 32, 32]), T([1, 2, 32, 32])), ValueError, "gqa_ratio")
# unsupported dtype float32 -> UnsupportedAxisValue
expect_raise(
    lambda: sdpa(T([1, 1, 32, 32], ttnn.float32), T([1, 1, 32, 32], ttnn.float32), T([1, 1, 32, 32], ttnn.float32)),
    UnsupportedAxisValue,
    "fp32_unsupported",
)
# is_causal=True (causal mask_mode not in SUPPORTED) -> UnsupportedAxisValue
expect_raise(
    lambda: sdpa(T([1, 1, 32, 32]), T([1, 1, 32, 32]), T([1, 1, 32, 32]), is_causal=True),
    UnsupportedAxisValue,
    "causal_unsupported",
)
ttnn.close_device(device)
