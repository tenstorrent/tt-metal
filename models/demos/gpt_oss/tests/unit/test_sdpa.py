import pytest
import torch
from loguru import logger

import ttnn
from models.utility_functions import comp_pcc

from ...tt.sdpa import sdpa as tt_sdpa


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
    "num_tokens, nh, nkv, dim",
    [
        (128, 64, 8, 64),
        (512, 32, 4, 64),
    ],
    ids=["gpt20B_tp1", "gpt20B_tp2"],
)
@pytest.mark.parametrize("sliding_window", [0, 128], ids=["regular", "sliding_window"])
@pytest.mark.parametrize(
    "num_iters",
    [1, 16],
    ids=["single", "multi"],
)
def test_sdpa(device, num_tokens, nh, nkv, dim, sliding_window, num_iters, reset_seeds):
    dtype = ttnn.bfloat16

    q, k, v = None, None, None
    tt_cache = None
    all_passing = True
    for n in range(num_iters):
        logger.info(f"Running iteration {n} of {num_iters}")
        cur_seq_len = num_tokens + n

        # Torch input
        q = torch.randn(num_tokens, nkv, nh // nkv, dim) if n == 0 else q
        k = torch.randn(num_tokens, nkv, dim) if k is None else k
        v = torch.randn(num_tokens, nkv, dim) if v is None else v
        s = torch.randn(1, nh, 1, 1)
        sm_scale = 1.0 / (dim**0.5)

        if n > 0:
            q = torch.cat([q, torch.randn(1, nkv, nh // nkv, dim)], dim=0)
            k = torch.cat([k, torch.randn(1, nkv, dim)], dim=0)
            v = torch.cat([v, torch.randn(1, nkv, dim)], dim=0)

        mask = torch.triu(torch.full((1, 1, cur_seq_len, cur_seq_len), -float("inf")), diagonal=1)
        if sliding_window > 0:
            mask += torch.tril(torch.full((1, 1, cur_seq_len, cur_seq_len), -float("inf")), diagonal=-sliding_window)

        if n > 0:
            mask = mask[:, :, -1:, :]

        # Torch output
        reference_out = reference_sdpa(q, k, v, s, sm_scale, sliding_window)

        # TT input
        if n == 0:
            q_in = q.view(num_tokens, 1, nh, dim)
            k_in = k
            v_in = v
        else:
            q_in = q[-1:, ...].view(1, 1, nh, dim)
            k_in = k[-1:, ...]
            v_in = v[-1:, ...]

        tt_q = ttnn.from_torch(q_in, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
        tt_k = ttnn.from_torch(k_in, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
        tt_v = ttnn.from_torch(v_in, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
        tt_sink = ttnn.from_torch(s, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
        tt_mask = ttnn.from_torch(mask, device=device, layout=ttnn.TILE_LAYOUT, dtype=dtype)

        # TT output
        tt_out, tt_cache = tt_sdpa(tt_q, tt_k, tt_v, tt_sink, sm_scale=sm_scale, tt_mask=tt_mask, tt_cache=tt_cache)
        tt_out_torch = ttnn.to_torch(tt_out)

        # Only compare the last token
        if n > 0:
            reference_out = reference_out[-1:, ...]

        # Compare outputs
        pcc = 0.99
        passed, pcc_message = comp_pcc(reference_out, tt_out_torch, pcc)
        if not passed:
            logger.error(f"Iteration {n} | Test passed: {passed}, PCC: {pcc_message}")
        all_passing = all_passing and passed

    assert all_passing, "Test failed: Outputs do not match"
