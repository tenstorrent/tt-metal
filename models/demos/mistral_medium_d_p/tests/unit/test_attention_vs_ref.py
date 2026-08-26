# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE: the prefill attention block (dense causal GQA 96/8, full-rotary YaRN) vs the reference.

Block contract (shared with the MLP — see tests/test_factory.py):

    in :  [1, 1, s, 12288]   full emb, replicated across the TP cols  (a post-norm activation)
    out:  [1, 1, s,  3072]   emb/tp, reduce-scattered across TP       (the sharded residual layout)

The reference runs in HF convention (rotate_half + concat cos/sin); the block runs in Meta
interleaved convention with ``convert_hf_qkv_to_meta_format``-swizzled q/k. That pair is proven
equivalent on the host in
``test_checkpoint_ingest.py::test_meta_qkv_swizzle_is_the_inverse_of_hf_rope``, so a failure here is
a device/sharding issue, not a convention one.

**TP=4 is the case that matters:** 24 Q + **2 KV** heads per chip. Two KV heads per chip is the one
thing no other GQA model in the repo does (minimax_m3 is 4/4, gpt_oss 8/8, both landing on 1), so
`1x4` is where a wrong per-device Q|K|V concat or a bad GQA grouping shows up.

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_attention_vs_ref.py -k 1x1
      pytest models/demos/mistral_medium_d_p/tests/unit/test_attention_vs_ref.py -k 1x4   # 4 chips
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_d_p.config import MeshConfig
from models.demos.mistral_medium_d_p.reference.torch_reference import gqa_attention
from models.demos.mistral_medium_d_p.tt.attention import Attention, AttentionConfig, ProgramConfig
from models.demos.mistral_medium_d_p.tt.rope import build_transformation_mat
from models.demos.mistral_medium_d_p.tt.rope_tables import build_hf_cos_sin, build_yarn_cos_sin
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

from ..test_factory import gather_tp_shards, mesh_setup, parametrize_mesh_with_fabric, replicate
from .shapes import EPS, HEAD_DIM, HIDDEN, N_KV, N_Q, YARN, per_chip


def _random_attn_weights(seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "q": torch.randn(N_Q * HEAD_DIM, HIDDEN, generator=g) * 0.02,
        "k": torch.randn(N_KV * HEAD_DIM, HIDDEN, generator=g) * 0.02,
        "v": torch.randn(N_KV * HEAD_DIM, HIDDEN, generator=g) * 0.02,
        "o": torch.randn(HIDDEN, N_Q * HEAD_DIM, generator=g) * 0.02,
    }


def _build_attention(mesh_device, mesh_config, ccl, w, seq_len):
    state = convert_hf_qkv_to_meta_format(
        {
            "q_proj.weight": w["q"],
            "k_proj.weight": w["k"],
            "v_proj.weight": w["v"],
            "o_proj.weight": w["o"],
        },
        HEAD_DIM,
    )
    return Attention(
        mesh_device=mesh_device,
        config=AttentionConfig(
            hidden_size=HIDDEN,
            num_heads=N_Q,
            num_kv_heads=N_KV,
            head_dim=HEAD_DIM,
            max_seq_len=max(seq_len, 128),
            rms_norm_eps=EPS,
        ),
        state_dict=state,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        program_config=ProgramConfig(),
        layer_idx=0,
        transformation_mats={"prefill": build_transformation_mat(mesh_device)},
        weight_dtype=ttnn.bfloat16,
    )


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 4)])
@pytest.mark.parametrize("seq_len", [256, 2048], ids=["s256", "s2k"])
def test_attention_prefill_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Full block vs ``gqa_attention``, with the reduce-scattered output reassembled on the host."""
    torch.manual_seed(0)
    mesh_config, ccl = mesh_setup(mesh_device)
    tp = mesh_config.tp
    w = _random_attn_weights()

    x = torch.randn(1, seq_len, HIDDEN) * 0.1
    cos_hf, sin_hf = build_hf_cos_sin(seq_len, HEAD_DIM, **YARN)
    ref = gqa_attention(x.float(), w, cos_hf, sin_hf, n_q=N_Q, n_kv=N_KV, head_dim=HEAD_DIM)

    attn = _build_attention(mesh_device, mesh_config, ccl, w, seq_len)
    cos_meta, sin_meta = build_yarn_cos_sin(seq_len, HEAD_DIM, **YARN)
    rope_mats = [replicate(cos_meta, mesh_device), replicate(sin_meta, mesh_device)]

    out_tt = attn(replicate(x.reshape(1, 1, seq_len, HIDDEN), mesh_device), rope_mats=rope_mats, kv_cache=None)

    assert out_tt.shape[-1] == HIDDEN // tp, f"expected emb/tp={HIDDEN // tp} per chip, got {out_tt.shape[-1]}"
    out = gather_tp_shards(out_tt, mesh_device).reshape(1, seq_len, HIDDEN)

    passing, pcc = comp_pcc(ref, out, 0.99)
    logger.info(f"attention prefill vs ref (TP={tp}, s={seq_len}): {pcc}")
    assert passing, f"attention PCC fail (TP={tp}, s={seq_len}): {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_fused_qkv_shard_is_per_device_qkv(mesh_device, device_params, reset_seeds):
    """Device *i* must hold ``[Q_i(24 heads) | K_i(2) | V_i(2)]`` = 3584 columns, contiguously.

    A naive ``cat([q, k, v], -1)`` sharded afterwards would give device 0 a slice of Q only.
    Checked on the built weight so a failure names the cause instead of showing up as a low PCC.
    """
    mesh_config, ccl = mesh_setup(mesh_device)
    tp = mesh_config.tp
    pc = per_chip(tp)
    w = _random_attn_weights(seed=11)
    attn = _build_attention(mesh_device, mesh_config, ccl, w, 256)

    state = convert_hf_qkv_to_meta_format(
        {"q_proj.weight": w["q"], "k_proj.weight": w["k"], "v_proj.weight": w["v"]}, HEAD_DIM
    )
    q_t, k_t, v_t = (state[f"{n}_proj.weight"].t() for n in ("q", "k", "v"))  # [H, n*hd]

    per_dev = ttnn.get_device_tensors(attn.weights.wqkv)
    nq_l, nkv_l = pc["n_q"] * HEAD_DIM, pc["n_kv"] * HEAD_DIM
    for dev_idx in range(tp):
        got = ttnn.to_torch(per_dev[dev_idx]).reshape(HIDDEN, pc["qkv"])
        want = torch.cat(
            [
                q_t[:, dev_idx * nq_l : (dev_idx + 1) * nq_l],
                k_t[:, dev_idx * nkv_l : (dev_idx + 1) * nkv_l],
                v_t[:, dev_idx * nkv_l : (dev_idx + 1) * nkv_l],
            ],
            dim=-1,
        )
        passing, pcc = comp_pcc(want, got, 0.999)
        assert passing, (
            f"device {dev_idx}'s fused QKV shard is wrong (pcc {pcc}): must be "
            f"[Q_{dev_idx}({pc['n_q']} heads) | K_{dev_idx}({pc['n_kv']}) | V_{dev_idx}({pc['n_kv']})]"
        )
    logger.info(f"fused QKV interleave correct on all {tp} devices ({pc['qkv']} cols each)")


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_attention_config_guards(mesh_device, device_params, reset_seeds, expect_error):
    """Partial rotary and a non-divisible head count must not be silently accepted."""
    with expect_error(NotImplementedError, "FULL rotary"):
        AttentionConfig(
            hidden_size=HIDDEN,
            num_heads=N_Q,
            num_kv_heads=N_KV,
            head_dim=HEAD_DIM,
            max_seq_len=128,
            rotary_dim=HEAD_DIM // 2,
        )
    with expect_error(ValueError, "divisible"):
        AttentionConfig(hidden_size=HIDDEN, num_heads=N_Q, num_kv_heads=7, head_dim=HEAD_DIM, max_seq_len=128)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
def test_attention_weights_reject_bias(mesh_device, device_params, reset_seeds, expect_error):
    """gpt-oss has attention_bias=True; Mistral does not. A stray bias must fail loud."""
    from models.demos.mistral_medium_d_p.tt.attention.weights import load_attention_weights

    w = _random_attn_weights()
    state = {
        "q_proj.weight": w["q"],
        "q_proj.bias": torch.zeros(N_Q * HEAD_DIM),
        "k_proj.weight": w["k"],
        "v_proj.weight": w["v"],
        "o_proj.weight": w["o"],
    }
    with expect_error(AssertionError, "bias-free"):
        load_attention_weights(
            mesh_device=mesh_device,
            config=AttentionConfig(
                hidden_size=HIDDEN, num_heads=N_Q, num_kv_heads=N_KV, head_dim=HEAD_DIM, max_seq_len=128
            ),
            state_dict=state,
            mesh_config=MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1]),
        )
