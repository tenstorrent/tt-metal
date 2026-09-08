# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3: RoPE vs the reference's own ``apply_rotary_pos_emb``, real weights.

q and k are real: produced by the real ``q_proj``/``k_proj`` and real per-head norms on the
fixture's real embeddings and target taps, in the same order the reference uses (project ->
view heads -> qk-norm -> rope).

The test that matters most is :func:`test_q_uses_the_table_tail`: q must take the **last**
``block`` rows of cos/sin while k takes all of them. Using the leading rows instead is a
silent bug -- shapes match and it is invisible at ctx=0.

Run:
    MESH_DEVICE=T3K pytest models/demos/blackhole/qwen36/tests/dflash/test_rope_tp.py -v -s
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.reference.dflash.dflash import apply_rotary_pos_emb
from models.demos.blackhole.qwen36.tests.dflash.conftest import load_fixture
from models.demos.blackhole.qwen36.tests.test_factory import get_pcc_threshold, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.dflash.rope import DFlashRoPE, build_cos_sin
from models.demos.blackhole.qwen36.tt.dflash.weights import read_state_dict

CTX = 64  # small: rope is position-wise, and this keeps the table upload cheap
BLOCK = 16


def _real_qk(cfg, ctx_len: int, block: int):
    """Real q and k in reference layout: project -> view heads -> qk-norm.

    q comes from the block only; k from ``concat(ctx, block)`` -- the dual-source shape.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

    fx = load_fixture(512)
    noise = fx["noise_embedding"].float()[:, :block]  # [1, block, 5120]
    # Stand in for the encoder's output as the context source, at the right width/scale.
    ctx = fx["target_hidden"].float()[:, :ctx_len, : cfg.hidden_size]

    sd = read_state_dict(
        keys=[
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.q_norm.weight",
            "layers.0.self_attn.k_norm.weight",
        ]
    )

    def norm(x, w):
        m = Qwen3RMSNorm(w.shape[0], eps=cfg.rms_norm_eps)
        with torch.no_grad():
            m.weight.copy_(w)
        return m(x)

    q = (noise @ sd["layers.0.self_attn.q_proj.weight"].float().T).view(1, block, cfg.num_attention_heads, cfg.head_dim)
    q = norm(q, sd["layers.0.self_attn.q_norm.weight"].float()).transpose(1, 2)  # [1, H, block, 128]

    k_all = torch.cat([ctx, noise], dim=1) @ sd["layers.0.self_attn.k_proj.weight"].float().T
    k = k_all.view(1, ctx_len + block, cfg.num_key_value_heads, cfg.head_dim)
    k = norm(k, sd["layers.0.self_attn.k_norm.weight"].float()).transpose(1, 2)  # [1, KV, ctx+block, 128]
    return q, k


def _to_device(x: torch.Tensor, mesh, shard_heads: bool):
    mapper = ttnn.ShardTensorToMesh(mesh, dim=1) if shard_heads else ttnn.ReplicateTensorToMesh(mesh)
    return ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def test_cos_sin_table_matches_hf(drafter_cfg):
    """The table itself, before any device involvement, vs ``Qwen3RotaryEmbedding``."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    from models.demos.blackhole.qwen36.tests.dflash.capture_fixtures import _drafter_hf_config, _resolve

    n = 128
    cos, sin = build_cos_sin(drafter_cfg.head_dim, drafter_cfg.rope_theta, n)

    hf = Qwen3RotaryEmbedding(_drafter_hf_config(_resolve("z-lab/Qwen3.6-27B-DFlash")))
    ref_cos, ref_sin = hf(torch.zeros(1, n, drafter_cfg.head_dim), torch.arange(n)[None])

    torch.testing.assert_close(cos, ref_cos[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(sin, ref_sin[0], rtol=1e-5, atol=1e-5)


@torch.no_grad()
@parametrize_mesh_tp()
def test_rope_tp(mesh_device, reset_seeds, ensure_gc, request, drafter_cfg):
    """Both q and k through the device rope, against the reference's own function."""
    cfg = drafter_cfg
    q, k = _real_qk(cfg, CTX, BLOCK)

    # Oracle: the reference's overridden apply_rotary_pos_emb, which does the tail-slice
    # for q internally. Table spans the whole ctx+block position range.
    cos_t, sin_t = build_cos_sin(cfg.head_dim, cfg.rope_theta, CTX + BLOCK)
    q_exp, k_exp = apply_rotary_pos_emb(q, k, cos_t[None], sin_t[None])

    rope = DFlashRoPE(mesh_device, cfg, max_position=CTX + BLOCK)
    cos_all, sin_all = rope.tables_for(0, CTX + BLOCK)
    cos_tail, sin_tail = rope.tables_for(CTX, BLOCK)  # the q slice: LAST `block` rows

    tt_q = rope.apply(_to_device(q, mesh_device, shard_heads=True), cos_tail, sin_tail)
    tt_k = rope.apply(_to_device(k, mesh_device, shard_heads=True), cos_all, sin_all)

    q_act = ttnn.to_torch(tt_q, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)).float()
    k_act = ttnn.to_torch(tt_k, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=1)).float()

    thr = get_pcc_threshold(request)
    q_pass, q_pcc = comp_pcc(q_exp, q_act, thr)
    k_pass, k_pcc = comp_pcc(k_exp, k_act, thr)
    logger.info(f"rope PCC  q {q_pcc}  k {k_pcc}")
    assert q_pass, f"q PCC {q_pcc}"
    assert k_pass, f"k PCC {k_pcc}"


def test_q_uses_the_table_tail(drafter_cfg):
    """q roped with the table's LEADING rows must NOT match the reference.

    Guards the guard: if the leading and trailing slices happened to agree, the asymmetry in
    :meth:`DFlashRoPE.tables_for` would be untested. Pure torch, no device.
    """
    cfg = drafter_cfg
    q, k = _real_qk(cfg, CTX, BLOCK)
    cos_t, sin_t = build_cos_sin(cfg.head_dim, cfg.rope_theta, CTX + BLOCK)

    q_exp, _ = apply_rotary_pos_emb(q, k, cos_t[None], sin_t[None])

    # What we would get from the wrong (leading-rows) slice.
    head = slice(0, BLOCK)
    q_wrong = q * cos_t[None, None, head] + _rotate_half(q) * sin_t[None, None, head]

    assert not torch.allclose(
        q_exp, q_wrong, rtol=1e-2, atol=1e-2
    ), "leading and trailing table slices agree, so the q/k asymmetry is untested"
    # And the tail slice reproduces the reference exactly.
    tail = slice(CTX, CTX + BLOCK)
    q_right = q * cos_t[None, None, tail] + _rotate_half(q) * sin_t[None, None, tail]
    torch.testing.assert_close(q_exp, q_right, rtol=1e-5, atol=1e-5)


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
