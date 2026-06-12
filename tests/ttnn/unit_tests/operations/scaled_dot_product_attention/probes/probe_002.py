import torch, ttnn
from eval.golden_tests.scaled_dot_product_attention.helpers import (
    pytorch_scaled_dot_product_attention,
    create_ttnn_input_tensor,
)
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 256, 64), (1, 8, 1024, 128), (1, 1, 8192, 64)]:
        torch.manual_seed(0)
        Q = torch.randn(shape)
        K = torch.randn(shape)
        V = torch.randn(shape)
        exp = pytorch_scaled_dot_product_attention(Q, K, V)
        tq = create_ttnn_input_tensor(Q, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        tk = create_ttnn_input_tensor(K, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        tv = create_ttnn_input_tensor(V, device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        for fid in [ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4]:
            cfg = ttnn.init_device_compute_kernel_config(device.arch(), math_fidelity=fid, fp32_dest_acc_en=True)
            to = scaled_dot_product_attention(tq, tk, tv, compute_kernel_config=cfg)
            r = ttnn.to_torch(to).to(torch.float32)
            rms = ((r - exp.float()).pow(2).mean().sqrt() / exp.float().std()).item()
            print(f"{shape} {fid}: rms={rms:.5f}")
finally:
    ttnn.close_device(device)
