import math
import torch
import ttnn

device = ttnn.open_device(device_id=0)


def torch_sdpa(Q, K, V, mask=None):
    qf = Q.to(torch.float32)
    kf = K.to(torch.float32)
    vf = V.to(torch.float32)
    s = 1.0 / math.sqrt(qf.shape[-1])
    scores = torch.matmul(qf, kf.transpose(-2, -1)) * s
    if mask is not None:
        scores = scores + mask.to(torch.float32)
    return torch.matmul(torch.softmax(scores, dim=-1), vf).to(Q.dtype)


try:
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
    from tests.ttnn.utils_for_testing import comp_pcc

    def run(name, B, H, S_q, S_kv, D, mask_shape):
        torch.manual_seed(0)
        Q = torch.randn(B, H, S_q, D, dtype=torch.bfloat16)
        K = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
        V = torch.randn(B, H, S_kv, D, dtype=torch.bfloat16)
        mask = torch.randn(*mask_shape, dtype=torch.bfloat16) if mask_shape else None
        expected = torch_sdpa(Q, K, V, mask=mask)
        try:
            ttnn_Q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            ttnn_K = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            ttnn_V = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            ttnn_mask = (
                ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
                if mask is not None
                else None
            )
            ttnn_out = scaled_dot_product_attention(ttnn_Q, ttnn_K, ttnn_V, attention_mask=ttnn_mask)
            actual = ttnn.to_torch(ttnn_out)
            p, pcc = comp_pcc(actual.float(), expected.float())
            print(f"=== {name}: PCC={pcc:.6f}")
        except Exception as e:
            print(f"=== {name}: ERROR {type(e).__name__}: {e}")

    run("A-broadcast S=47 H=4", 1, 4, 47, 47, 64, (1, 1, 47, 47))
    run("B-perhead S=47 H=4", 1, 4, 47, 47, 64, (1, 4, 47, 47))
    run("C-perhead aligned", 1, 4, 32, 32, 64, (1, 4, 32, 32))
    run("D-perhead Skv=47 only", 1, 4, 32, 47, 64, (1, 4, 32, 47))
    run("E-perhead Sq=47 only Skv=64", 1, 4, 47, 64, 64, (1, 4, 47, 64))
    run("F-broadcast aligned", 1, 4, 32, 32, 64, (1, 1, 32, 32))

finally:
    ttnn.close_device(device)
