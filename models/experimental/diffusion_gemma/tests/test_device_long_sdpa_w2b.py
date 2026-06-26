# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""W2b long-prompt non-causal SDPA spikes (#47462).

This is the S1/S2 harness from ``DEVICE_LOOP_W2B.md``: canvas queries
``[1, H, 256, DH]`` attend bidirectionally to a long ``[1, Hkv, Sk, DH]``
prefix+canvas K/V rectangle. S1 runs the target maskless non-causal path; S2
keeps an explicit all-zero mask as the A/B control.

The long cases are expensive and intentionally opt-in:

  DG_RUN_DEVICE=1 DG_W2B_SDPA_SWEEP=full pytest .../test_device_long_sdpa_w2b.py
"""

import math
import os

import pytest
import torch

import ttnn
from models.demos.gemma4.tt.attention.prefill import _slice_rope_cache
from models.demos.gemma4.tt.attention.operations import prefill_sdpa_program_config
from tests.ttnn.utils_for_testing import assert_with_pcc

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run W2b SDPA spikes on a Tenstorrent device",
    ),
    pytest.mark.use_module_device,
]

CANVAS_LEN = 256
ORACLE_K_CHUNK = 2048
S_SWEEP = (8192, 32768, 33000, 65536, 131072, 262144)
HEAD_DIM_SWEEP = (256, 512)
SMOKE_CASES = {
    (8192, 256, False),
    (33000, 256, False),
    (8192, 256, True),
}


def _requires_full_sweep():
    return pytest.mark.skipif(
        os.environ.get("DG_W2B_SDPA_SWEEP") != "full",
        reason="set DG_W2B_SDPA_SWEEP=full to run the expensive W2b acceptance sweep",
    )


def _w2b_sweep_params():
    params = []
    for masked in (False, True):
        spike = "s2-masked" if masked else "s1-maskless"
        for head_dim in HEAD_DIM_SWEEP:
            for sk in S_SWEEP:
                marks = () if (sk, head_dim, masked) in SMOKE_CASES else (_requires_full_sweep(),)
                params.append(pytest.param(sk, head_dim, masked, marks=marks, id=f"{spike}-sk{sk}-d{head_dim}"))
    return params


def _torch_online_sdpa(q, k, v, *, k_chunk=ORACLE_K_CHUNK):
    """Memory-bounded fp32 all-attend oracle for very long K sequences."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    repeat = q.shape[1] // k.shape[1]
    running_max = torch.full(q.shape[:-1], -torch.inf, dtype=torch.float32)
    running_sum = torch.zeros(q.shape[:-1], dtype=torch.float32)
    running_out = torch.zeros_like(q, dtype=torch.float32)

    q = q.float()
    for start in range(0, k.shape[-2], k_chunk):
        k_chunk_t = k[:, :, start : start + k_chunk, :].float()
        v_chunk_t = v[:, :, start : start + k_chunk, :].float()
        if repeat != 1:
            k_chunk_t = k_chunk_t.repeat_interleave(repeat, dim=1)
            v_chunk_t = v_chunk_t.repeat_interleave(repeat, dim=1)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k_chunk_t) * scale
        chunk_max = torch.max(scores, dim=-1).values
        new_max = torch.maximum(running_max, chunk_max)
        old_scale = torch.exp(running_max - new_max)
        exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
        chunk_sum = torch.sum(exp_scores, dim=-1)
        chunk_out = torch.einsum("bhqk,bhkd->bhqd", exp_scores, v_chunk_t)

        running_out = running_out * old_scale.unsqueeze(-1) + chunk_out
        running_sum = running_sum * old_scale + chunk_sum
        running_max = new_max

    return running_out / running_sum.unsqueeze(-1)


def _run_long_noncausal_sdpa(device, *, sk, head_dim, masked, pcc=0.99):
    torch.manual_seed(47462 + sk + head_dim + int(masked))
    q = torch.randn(1, 1, CANVAS_LEN, head_dim)
    k = torch.randn(1, 1, sk, head_dim)
    v = torch.randn(1, 1, sk, head_dim)
    golden = _torch_online_sdpa(q, k, v)

    tt_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_mask = None
    if masked:
        mask = torch.zeros(1, 1, CANVAS_LEN, sk, dtype=torch.float32)
        tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tt_out = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        attn_mask=tt_mask,
        is_causal=False,
        program_config=prefill_sdpa_program_config(head_dim, CANVAS_LEN, sk),
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(tt_out)[:, :, :CANVAS_LEN, :]
    assert_with_pcc(golden, out, pcc)

    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)
    tt_out.deallocate(True)
    if tt_mask is not None:
        tt_mask.deallocate(True)


@pytest.mark.parametrize(
    ("sk", "head_dim", "masked"),
    _w2b_sweep_params(),
)
def test_w2b_long_prompt_noncausal_sdpa(device, sk, head_dim, masked):
    _run_long_noncausal_sdpa(device, sk=sk, head_dim=head_dim, masked=masked)


def test_w2b_rope_slice_reaches_256k(device):
    cache_len = 262144
    canvas_len = 256
    prompt_len = cache_len - canvas_len
    cache = ttnn.from_torch(
        torch.zeros(1, 1, cache_len, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    sliced = _slice_rope_cache(cache, prompt_len, canvas_len)
    assert sliced.shape[-2] == canvas_len
    sliced.deallocate(True)
    with pytest.raises(ValueError, match="exceeds cache length"):
        _slice_rope_cache(cache, cache_len, 32)
    cache.deallocate(True)
