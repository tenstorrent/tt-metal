import math
import torch
import ttnn

device = ttnn.open_device(device_id=0)


def torch_sdpa(q, k, v, scale=None):
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    s = scale if scale is not None else 1.0 / math.sqrt(qf.shape[-1])
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * s
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, vf).to(q.dtype)


try:
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
    from tests.ttnn.utils_for_testing import comp_pcc

    cases = [
        ("W non-aligned, D=50, S=32", (1, 1, 32, 50)),
        ("W non-aligned, D=50, S=64", (1, 1, 64, 50)),
        ("H non-aligned, S=47, D=64", (1, 1, 47, 64)),
        ("H non-aligned, S=33, D=64", (1, 1, 33, 64)),
        ("both non-aligned, S=50, D=50", (1, 1, 50, 50)),
        ("tile aligned, S=32, D=32 (regression)", (1, 1, 32, 32)),
    ]
    for name, shape in cases:
        torch.manual_seed(0)
        Q = torch.randn(*shape, dtype=torch.bfloat16)
        K = torch.randn(*shape, dtype=torch.bfloat16)
        V = torch.randn(*shape, dtype=torch.bfloat16)
        expected = torch_sdpa(Q, K, V)
        try:
            ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            ttnn_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            ttnn_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            ttnn_out = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V)
            torch_out = ttnn.to_torch(ttnn_out)
            diff = (torch_out.float() - expected.float()).abs()
            passed, pcc_val = comp_pcc(torch_out.float(), expected.float())
            print(f"{name}: shape={shape} out={torch_out.shape} max_diff={diff.max().item():.4f} PCC={pcc_val}")
        except Exception as e:
            print(f"{name}: shape={shape} RUNTIME ERROR: {type(e).__name__}: {e}")
finally:
    ttnn.close_device(device)
