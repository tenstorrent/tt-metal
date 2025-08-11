import pytest
import torch

import ttnn


def reference_sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window)
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


@pytest.mark.parametrize(
    "bsz, nh, nkv, dim",
    [
        (1, 64, 8, 64),
    ],
    ids=[
        "gpt20B",
    ],
)
def test_sdpa(device, bsz, nh, nkv, dim):
    dtype = ttnn.bfloat16

    breakpoint()

    q = torch.randn(bsz, nh // nkv, nkv, dim)
    k = torch.randn(bsz, nkv, dim).to(device)
    v = torch.randn(bsz, nkv, dim).to(device)
    s = torch.randn(1, nh, 1, 1).to(device)

    sm_scale = 1.0 / (dim**0.5)

    reference_out = reference_sdpa(q, k, v, s, sm_scale)

    # q_mult = nh // nkv
    # q = torch.randn(bsz, nh // nkv, q_mult, dim).to(device)
    # k = torch.randn(bsz, nkv, dim).to(device)
    # v = torch.randn(bsz, nkv, dim).to(device)
    # sm_scale = 1.0 / (dim ** 0.5)

    # tt_q = ttnn.from_torch(q, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    # tt_k = ttnn.from_torch(k, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
    # tt_v = ttnn.from_torch(v, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)

    # tt_out = tt_sdpa(tt_q, tt_k, tt_v, tt_sink, sm_scale)

    # ref_out = reference_sdpa(q.cpu(), k.cpu(), v.cpu(), s.cpu(), sm_scale)

    # assert comp_allclose_and_pcc(tt_out.to_torch(), ref_out)
