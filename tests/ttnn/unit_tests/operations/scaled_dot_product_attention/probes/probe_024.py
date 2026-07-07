import torch, ttnn

device = ttnn.open_device(device_id=0)

# Reproduce the exact golden test setup
from eval.golden_tests.scaled_dot_product_attention.helpers import run_scaled_dot_product_attention

q_shape = (1, 1, 1024, 64)
k_shape = (1, 1, 1024, 64)
v_shape = (1, 1, 1024, 64)
inputs = (q_shape, k_shape, v_shape)

try:
    run_scaled_dot_product_attention(
        inputs,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mask_mode="none",
        scale_mode="auto",
        fp32_dest_acc_en=True,
        device=device,
    )
    print("PASS")
except Exception as e:
    print(f"FAIL: {e}")

ttnn.close_device(device)
