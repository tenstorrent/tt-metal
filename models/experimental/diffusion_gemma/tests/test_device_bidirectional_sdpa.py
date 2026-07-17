# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device bidirectional canvas SDPA test (#47462, net-new N1).

Validates the net-new non-causal denoise attention on real Tenstorrent hardware:
canvas queries [b, nh, C, d] attend to the prefix-concat keys
[b, nkv, prompt_len+C, d] through the **baked** canvas mask from
``reference/attention_mask.py`` (full-attn = fully visible; sliding = symmetric
2W+1 window), with ``is_causal=False``. Since ttnn SDPA makes
``sliding_window_size`` and ``attn_mask`` mutually exclusive, the window lives in
the mask. PCC vs a torch SDPA golden using the same mask.

Requires a device (uses the ``device`` fixture); skipped on CPU-only envs.
Run on QB2:
  HF unused — checkpoint-free.
  pytest models/experimental/diffusion_gemma/tests/test_device_bidirectional_sdpa.py
"""

import os

import pytest
import torch

import ttnn
from models.experimental.diffusion_gemma.reference.attention_mask import build_canvas_denoise_mask
from tests.ttnn.utils_for_testing import assert_with_pcc

# Opt-in: these run real kernels on a Tenstorrent device. They need an sfpi
# toolchain matching the LLK source (>= 7.60.0, which adds sfpi::ShiftMode); an
# older sfpi fails dispatch-kernel compile at device open. Gated so the default
# CPU suite never tries to open a device.
pytestmark = [
    pytest.mark.skipif(
        os.environ.get("DG_RUN_DEVICE") != "1",
        reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
    ),
    # Share ONE device across all tests in this module. Repeated per-test
    # CreateDevice/teardown on QB2 (4x Blackhole) can hang an active-erisc core
    # ("Timed out while waiting for active ethernet core ... to become active
    # again"), bricking the board until a reset. Module scope does a single
    # open + single teardown for the whole file.
    pytest.mark.use_module_device,
]

# Device-friendly large-negative (bf16-representable) stand-in for -inf in the mask.
NEG = -1.0e9


def _run_canvas_sdpa(
    device,
    *,
    local_window=False,
    window_half=None,
    prompt_fully_visible=False,
    batch=1,
    num_heads=8,
    num_kv_heads=8,
    prompt_len=256,
    canvas_len=256,
    head_dim=256,
    pcc=0.99,
):
    torch.manual_seed(1234)
    seq_k = prompt_len + canvas_len

    q = torch.randn(batch, num_heads, canvas_len, head_dim)
    k = torch.randn(batch, num_kv_heads, seq_k, head_dim)
    v = torch.randn(batch, num_kv_heads, seq_k, head_dim)

    mask2d = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=local_window,
        window_half=window_half,
        prompt_fully_visible=prompt_fully_visible,
        neg_inf=NEG,
        dtype=torch.float32,
    )  # [canvas_len, seq_k]
    mask = mask2d.view(1, 1, canvas_len, seq_k)  # broadcast over batch + heads

    # torch golden (fp32), expanding KV for GQA if needed
    if num_kv_heads != num_heads:
        k_ref = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_ref = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
    else:
        k_ref, v_ref = k, v
    golden = torch.nn.functional.scaled_dot_product_attention(q, k_ref, v_ref, attn_mask=mask, is_causal=False)

    tt_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
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
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(tt_out)[:, :, :canvas_len, :]
    assert_with_pcc(golden, out, pcc)


def test_canvas_bidirectional(device):
    # CANONICAL denoise geometry: canvas fully sees prompt + canvas (bidirectional)
    # for every layer type — the decoder is fully bidirectional (modeling:1399-1438).
    _run_canvas_sdpa(device)


def test_gqa_bidirectional(device):
    # GQA shape matching the model (16 query / 8 KV heads), canonical full visibility.
    _run_canvas_sdpa(device, num_heads=16, num_kv_heads=8)


# --- NON-canonical: exercise the ttnn SDPA windowed-mask path (NOT the denoise geometry) ---
def test_sdpa_local_window_op(device):
    # op-capability only: symmetric 2W+1 window baked into the mask. The real decoder
    # does NOT window visibility — see build_canvas_denoise_mask docstring.
    _run_canvas_sdpa(device, local_window=True, window_half=64)


def test_sdpa_local_window_prompt_fully_visible_op(device):
    _run_canvas_sdpa(device, local_window=True, window_half=64, prompt_fully_visible=True)
