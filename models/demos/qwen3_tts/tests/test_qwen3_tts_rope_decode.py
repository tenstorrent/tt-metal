# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Accuracy gate for the decode-mode RoPE path in ``rope.apply_rope_qk``.

``apply_rope_qk`` routes single-token Q/K to ``rotary_embedding_llama(is_decode_mode=True)``
because the prefill kernel loops once per head (~2 us/head) while the decode kernel rotates
all heads in one tile (3.4 us). That is a pure dispatch change, so these tests assert the
two kernels agree *bit for bit* — not merely to some PCC threshold — at every head count
the Talker and CodePredictor use, on TP=1 (16 heads / 8 KV) and TP=2 (8 / 4).

They also pin the routing itself, since a silent fallback to the prefill kernel would keep
the numbers perfect while losing the entire speedup.

    pytest models/demos/qwen3_tts/tests/test_qwen3_tts_rope_decode.py -s
"""

import pytest
import torch

import ttnn
from models.demos.qwen3_tts.tt.rope import (
    apply_rope_qk,
    get_decode_transformation_mat,
    get_rope_tensors,
    get_transformation_mat,
)

HEAD_DIM = 128

# (n_q_heads, n_kv_heads) per SKU: TP=1 is the full head count, TP=2 the per-chip slice.
HEAD_COUNTS = [(16, 8), (8, 4)]


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_device(device_id=0, l1_small_size=32768)
    d.enable_program_cache()
    yield d
    ttnn.close_device(d)


def _kcfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _to_dev(t, device, memory_config=ttnn.L1_MEMORY_CONFIG):
    return ttnn.from_torch(t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=memory_config)


@pytest.mark.parametrize("n_q,n_kv", HEAD_COUNTS, ids=[f"heads{q}kv{k}" for q, k in HEAD_COUNTS])
def test_decode_mode_matches_prefill_mode(device, n_q, n_kv, start_pos=128):
    """Decode-mode RoPE must be bit-identical to prefill-mode RoPE on the same input."""
    torch.manual_seed(0)
    q_t = torch.randn(1, n_q, 1, HEAD_DIM, dtype=torch.bfloat16)
    k_t = torch.randn(1, n_kv, 1, HEAD_DIM, dtype=torch.bfloat16)
    cos, sin = get_rope_tensors(device, HEAD_DIM, 1, torch.tensor([start_pos]))
    prefill_mat = get_transformation_mat(HEAD_DIM, device)
    decode_mat = get_decode_transformation_mat(device)
    kc = _kcfg()

    # Reference: force the prefill kernel by withholding decode_trans_mat.
    ref_q, ref_k = apply_rope_qk(
        _to_dev(q_t, device),
        _to_dev(k_t, device),
        cos,
        sin,
        prefill_mat,
        head_dim=HEAD_DIM,
        decode_trans_mat=None,
        compute_kernel_config=kc,
    )
    got_q, got_k = apply_rope_qk(
        _to_dev(q_t, device),
        _to_dev(k_t, device),
        cos,
        sin,
        prefill_mat,
        head_dim=HEAD_DIM,
        decode_trans_mat=decode_mat,
        compute_kernel_config=kc,
    )

    for name, ref, got in (("q", ref_q, got_q), ("k", ref_k, got_k)):
        a = ttnn.to_torch(ref).float()
        b = ttnn.to_torch(got).float()
        assert a.shape == b.shape, f"{name}: shape changed {a.shape} -> {b.shape}"
        max_diff = (a - b).abs().max().item()
        print(
            f"[rope heads={n_q}/{n_kv} {name}] max|prefill - decode| = {max_diff:.4g} "
            f"(|ref|max={a.abs().max().item():.4g})"
        )
        assert max_diff == 0.0, f"{name}: decode-mode RoPE is not bit-identical (max diff {max_diff})"


def test_routing(device, monkeypatch):
    """seq==1 must take the decode kernel; seq>1 and >32 heads must take the prefill one."""
    modes = []
    orig = ttnn.experimental.rotary_embedding_llama

    def spy(*a, **kw):
        modes.append(bool(kw.get("is_decode_mode")))
        return orig(*a, **kw)

    monkeypatch.setattr(ttnn.experimental, "rotary_embedding_llama", spy)

    prefill_mat = get_transformation_mat(HEAD_DIM, device)
    decode_mat = get_decode_transformation_mat(device)
    kc = _kcfg()

    def run(seq, n_heads=16):
        modes.clear()
        q = _to_dev(torch.randn(1, n_heads, seq, HEAD_DIM, dtype=torch.bfloat16), device)
        k = _to_dev(torch.randn(1, n_heads // 2, seq, HEAD_DIM, dtype=torch.bfloat16), device)
        cos, sin = get_rope_tensors(device, HEAD_DIM, seq, torch.arange(seq))
        apply_rope_qk(
            q, k, cos, sin, prefill_mat, head_dim=HEAD_DIM, decode_trans_mat=decode_mat, compute_kernel_config=kc
        )
        return list(modes)

    assert run(1) == [True, True], "seq==1 should use the decode kernel for both Q and K"
    assert run(32) == [False, False], "seq>1 needs a distinct cos/sin row per position"
    # Heads must fit one 32-row tile after the transpose; beyond that, fall back.
    assert run(1, n_heads=64) == [False, False], "n_heads>32 cannot pack into one tile"
    print("[rope routing] seq=1 -> decode, seq=32 -> prefill, heads=64 -> prefill")
