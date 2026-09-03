# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-ATTN` — the full `tt/attention/` prefill block vs a torch reference, `(1,1)` mesh.

Block: QKV projection -> GQA head split (32 Q / 8 KV) -> full llama3-scaled RoPE -> causal SDPA ->
``o_proj``. No biases, no sinks, no sliding window (Llama has none). Reference is HF-convention
``transformers.models.llama.modeling_llama.LlamaAttention`` maths written out in **fp32** on the
same random weights the TT module is built from.

**One frequency set, both conventions.** ``tt/rope.build_meta_cos_sin`` produces the Meta
interleaved pair the device op wants; ``_hf_cos_sin_from_meta`` (imported from
``tests/unit/test_rope_vs_ref.py``) derives the HF ``cat``-of-halves pair from *those same*
frequencies — the structure ``models/demos/gpt_oss_d_p/tests/unit/test_attention_vs_ref.py:83``
uses. A test that built each side's tables independently could not tell a convention bug from a
pass.

**The negative control is what makes the PCC mean anything** (the
``tests/unit/test_rope_vs_ref.py:140`` pattern). Q/K weights must be ``reverse_permute``d into the
Meta layout at load (``DEC-033``); ``test_unswizzled_qk_weights_fail`` builds the module with
``meta_swizzle=False`` and shows the PCC collapses. Without it, a high PCC could be two
symmetrically-wrong sides.

**How the gate is judged.** Absolute threshold ``PCC >= 0.999`` (``03_OUTLINE.md`` §5) **plus** the
gap to the **torch noise floor** (``DEC-032``): the same pipeline with every stored/intermediate
tensor rounded to the dtype the device holds it in but all arithmetic in fp32. The recipe's
Appendix E oracle number (0.9996099) is recorded for context but is **not** a target — the oracle's
reference loads HF weights at the checkpoint's ``torch_dtype: bfloat16``, so it shares the device's
own rounding and its PCC is flattered relative to an fp32 reference.

**Input distribution** (``DEC-026`` / ``R-018``): ``rand(...)*2 - 1``, i.e. uniform on ``[-1, 1)`` —
the oracle's own (``models/tt_transformers/tests/test_attention_prefill.py:161-166``).
**Reference dtype policy:** fp32 weights, fp32 activations, fp32 arithmetic.

Run:
    pytest models/demos/llama32_8b_d_p/tests/unit/test_attention_vs_ref.py -x -q
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.llama32_8b_d_p.tests.test_factory import TestFactory
from models.demos.llama32_8b_d_p.tests.unit.test_mlp_vs_ref import err_ratio, quantize_like_device
from models.demos.llama32_8b_d_p.tests.unit.test_rope_vs_ref import _hf_cos_sin_from_meta, _hf_to_meta_layout
from models.demos.llama32_8b_d_p.tt.attention import Attention, ProgramConfig, attention_config_from_hf
from models.demos.llama32_8b_d_p.tt.attention.operations import (
    apply_qkv_projection,
    apply_rope,
    split_qkv_heads_prefill,
)
from models.demos.llama32_8b_d_p.tt.rope import build_meta_cos_sin, build_transformation_mat

PCC_THRESHOLD = 0.999  # 03_OUTLINE.md §5
ORACLE_PCC = 0.9996099  # tt_transformers test_attention_prefill.py — context only, NOT a target
# Block-level slack over the storage-dtype noise floor. Set from measurement with an
# attribution, not from taste (``DEC-034``): ``test_sdpa_kernel_error_is_the_dominant_term`` shows
# the SDPA kernel alone sits ~70-80x off ITS own bf16-input floor, because a storage-dtype floor
# models tensor rounding and not a kernel's interior. That single term accounts for the whole
# block-level 2.6x (bf8_b) / 5.2x (bf16) measured here, and
# ``test_qkv_and_rope_stage_is_at_the_floor`` pins the stages this package actually owns at ~1x.
MAX_ERR_RATIO = 8.0
# The stages tt/attention/ implements itself (projections + GQA split + RoPE) have no such excuse.
MAX_ERR_RATIO_QKV_STAGE = 3.0

WEIGHT_SCALE = 0.02


# --------------------------------------------------------------------------------------
# Torch reference — HF-convention Llama GQA attention. No bias, no sink, no sliding window.
# --------------------------------------------------------------------------------------
def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _rope_hf(t, cos, sin):
    """``t * cos + rotate_half(t) * sin``; ``cos``/``sin`` are ``[S, head_dim]``."""
    return t * cos + _rotate_half(t) * sin


def _random_attn_state(nq, nkv, head_dim, hidden, *, seed: int) -> dict:
    """HF-layout ``[out, in]`` projection weights. No biases (``attention_bias: false``)."""
    gen = torch.Generator().manual_seed(seed)
    return {
        "q_proj.weight": torch.randn(nq * head_dim, hidden, generator=gen) * WEIGHT_SCALE,
        "k_proj.weight": torch.randn(nkv * head_dim, hidden, generator=gen) * WEIGHT_SCALE,
        "v_proj.weight": torch.randn(nkv * head_dim, hidden, generator=gen) * WEIGHT_SCALE,
        "o_proj.weight": torch.randn(hidden, nq * head_dim, generator=gen) * WEIGHT_SCALE,
    }


def _torch_attention(x, state, cos_hf, sin_hf, cfg, *, quantise=None):
    """``LlamaAttention.forward`` in fp32, HF RoPE convention.

    ``x`` is ``[1, S, hidden]``. ``quantise`` — when given, a ``dtype -> tensor`` rounding callable
    applied at every point the device stores a tensor, which turns this same function into the
    **noise floor** (``DEC-032``): identical arithmetic, identical dtype ladder, fp32 maths.
    """
    nq, nkv, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
    b, s, _ = x.shape
    ident = (lambda t, _dt: t) if quantise is None else quantise

    q = (x @ state["q_proj.weight"].t()).view(b, s, nq, hd).transpose(1, 2)  # [B, NQ, S, HD]
    k = (x @ state["k_proj.weight"].t()).view(b, s, nkv, hd).transpose(1, 2)
    v = (x @ state["v_proj.weight"].t()).view(b, s, nkv, hd).transpose(1, 2)
    # ttnn.linear(dtype=bfloat16) writes the projections back as bf16.
    q, k, v = ident(q, ttnn.bfloat16), ident(k, ttnn.bfloat16), ident(v, ttnn.bfloat16)

    q = ident(_rope_hf(q, cos_hf, sin_hf), ttnn.bfloat16)
    k = ident(_rope_hf(k, cos_hf, sin_hf), ttnn.bfloat16)

    # GQA: torch must repeat; the device must NOT — ttnn.transformer.scaled_dot_product_attention
    # handles the group itself (sdpa_device_operation.cpp:98 asserts only nqh % nkv == 0).
    rep = nq // nkv
    k_rep = k.repeat_interleave(rep, dim=1)
    v_rep = v.repeat_interleave(rep, dim=1)

    scores = (q @ k_rep.transpose(-1, -2)) * cfg.scaling
    # An explicit causal mask, never attention_mask=None: HF's eager path applies only the mask it
    # is handed, so a missing mask yields non-causal attention silently (Appendix F.2).
    scores = scores + torch.triu(torch.full((s, s), float("-inf"), dtype=scores.dtype), diagonal=1)
    out = torch.softmax(scores, dim=-1) @ v_rep
    out = ident(out, ttnn.bfloat16)

    out = out.transpose(1, 2).reshape(b, s, nq * hd)
    # apply_output_projection casts its input to bf8_b before the o_proj matmul (operations.py).
    out = ident(out.reshape(1, 1, s, nq * hd), ttnn.bfloat8_b).reshape(b, s, nq * hd)
    return ident((out @ state["o_proj.weight"].t()).reshape(1, 1, s, -1), ttnn.bfloat16).reshape(b, s, -1)


def _quantise_weights(state, weight_dtype):
    """Round the four projections exactly as the loader stores them: ttnn ``[1, 1, in, out]``."""
    return {
        k: quantize_like_device(w.transpose(-1, -2).unsqueeze(0).unsqueeze(0), weight_dtype)[0, 0].transpose(-1, -2)
        for k, w in state.items()
    }


def _build_attention(mesh_device, objs, cfg, state, *, weight_dtype, meta_swizzle=True, program_config=None):
    return Attention(
        mesh_device,
        cfg,
        state,
        mesh_config=objs["mesh_config"],
        ccl_manager=None,  # TP=1 / SP=1: no collective is entered
        program_config=program_config or ProgramConfig(),
        layer_idx=0,
        transformation_mats={"prefill": build_transformation_mat(mesh_device)},
        weight_dtype=weight_dtype,
        meta_swizzle=meta_swizzle,
    )


def _run_tt_attention(mesh_device, attn, x, cos_meta, sin_meta, **fwd_kwargs):
    def _to_dev(t, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    seq_len = x.shape[1]
    tt_out = attn(
        _to_dev(x.reshape(1, 1, seq_len, -1)),
        rope_mats=[_to_dev(cos_meta), _to_dev(sin_meta)],
        **fwd_kwargs,
    )
    return ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].reshape(1, seq_len, -1)


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("seq_len", [128, 512, 2048], ids=["s128", "s512", "s2048"])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["wbf8_b", "wbf16"])
@torch.no_grad()
def test_attention_vs_ref(mesh_device, seq_len, weight_dtype, reset_seeds):
    """Full prefill attention block on device vs fp32 torch. PCC >= 0.999. See the docstring."""
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    cfg = attention_config_from_hf(hf_config, max_seq_len=max(seq_len, 128))

    assert (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim) == (32, 8, 128), (
        f"expected Llama-3.1-8B GQA 32/8 head_dim 128, got " f"{(cfg.num_heads, cfg.num_kv_heads, cfg.head_dim)}"
    )
    assert cfg.gqa_group_size == 4 and cfg.rotary_dim == cfg.head_dim

    state = _random_attn_state(cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size, seed=0)
    # The oracle's own distribution: uniform [-1, 1) (test_attention_prefill.py:161-166).
    x = torch.rand(1, seq_len, cfg.hidden_size, dtype=torch.float32) * 2 - 1

    # One frequency set -> the Meta pair for the device, the HF pair for the reference.
    cos_meta, sin_meta = build_meta_cos_sin(hf_config, seq_len)
    cos_hf, sin_hf = _hf_cos_sin_from_meta(cos_meta, sin_meta)

    ref = _torch_attention(x, state, cos_hf, sin_hf, cfg)
    floor = _torch_attention(
        quantize_like_device(x.reshape(1, 1, seq_len, -1), ttnn.bfloat16).reshape(1, seq_len, -1),
        _quantise_weights(state, weight_dtype),
        cos_hf,
        sin_hf,
        cfg,
        quantise=lambda t, dt: quantize_like_device(t, dt),
    )

    attn = _build_attention(mesh_device, objs, cfg, state, weight_dtype=weight_dtype)
    out = _run_tt_attention(mesh_device, attn, x, cos_meta, sin_meta)

    assert out.shape == ref.shape == floor.shape == (1, seq_len, cfg.hidden_size)
    passing, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    _, floor_pcc = comp_pcc(ref, floor, 0.0)
    ratio = err_ratio(pcc, floor_pcc)

    logger.info(comp_allclose(ref, out))
    logger.info(
        f"[G-ATTN] seq_len={seq_len} weight_dtype={weight_dtype}: measured PCC = {pcc} | "
        f"torch noise floor = {floor_pcc} | err ratio = {ratio:.2f}x | threshold {PCC_THRESHOLD} | "
        f"oracle {ORACLE_PCC} (context only, DEC-032)"
    )
    assert passing, f"[G-ATTN] seq_len={seq_len} {weight_dtype} below {PCC_THRESHOLD}: {pcc}"
    assert ratio <= MAX_ERR_RATIO, (
        f"[G-ATTN] seq_len={seq_len} {weight_dtype}: PCC {pcc} clears {PCC_THRESHOLD} but sits "
        f"{ratio:.1f}x off the torch noise floor {floor_pcc} — investigate (DEC-032)"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_unswizzled_qk_weights_fail(mesh_device, reset_seeds):
    """Negative control: without the Q/K Meta ``reverse_permute`` the PCC must collapse.

    The device applies the *Meta* rotation (adjacent pairs) while the reference applies the *HF*
    one (element ``i`` against ``i + D/2``). They agree only if the Q/K projection weights are
    ``reverse_permute``d at load (``DEC-033``,
    ``models/tt_transformers/tt/load_checkpoints.py:891``). Building with ``meta_swizzle=False``
    keeps every shape, every dtype and every op identical and changes only that — so a high PCC
    here would mean the positive gate is measuring something other than the RoPE convention.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    cfg = attention_config_from_hf(hf_config, max_seq_len=128)
    seq_len = 128

    state = _random_attn_state(cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size, seed=0)
    x = torch.rand(1, seq_len, cfg.hidden_size, dtype=torch.float32) * 2 - 1
    cos_meta, sin_meta = build_meta_cos_sin(hf_config, seq_len)
    cos_hf, sin_hf = _hf_cos_sin_from_meta(cos_meta, sin_meta)
    ref = _torch_attention(x, state, cos_hf, sin_hf, cfg)

    swizzled = _build_attention(mesh_device, objs, cfg, state, weight_dtype=ttnn.bfloat16)
    raw = _build_attention(mesh_device, objs, cfg, state, weight_dtype=ttnn.bfloat16, meta_swizzle=False)

    _, pcc_ok = comp_pcc(ref, _run_tt_attention(mesh_device, swizzled, x, cos_meta, sin_meta), 0.0)
    _, pcc_bad = comp_pcc(ref, _run_tt_attention(mesh_device, raw, x, cos_meta, sin_meta), 0.0)
    logger.info(f"[G-ATTN] negative control: swizzled PCC = {pcc_ok}, UNswizzled PCC = {pcc_bad}")
    assert float(pcc_bad) < 0.99, (
        f"unswizzled Q/K weights scored {pcc_bad}; the positive gate is not actually testing the "
        f"Meta RoPE convention"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["wbf8_b", "wbf16"])
@torch.no_grad()
def test_fp32_dest_acc_on_the_attention_path(mesh_device, weight_dtype, reset_seeds):
    """A/B ``fp32_dest_acc_en`` across the projections + SDPA + ``o_proj`` (``DEC-031``).

    The template ships ``fp32_dest_acc_en=False`` (``gpt_oss .../config.py:71``); this package
    defaults it to ``True``. The numbers below go into the ``G-ATTN`` detail block, and the assert
    only guards the direction — a precision regression here is otherwise silent.
    """
    from dataclasses import replace

    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    cfg = attention_config_from_hf(hf_config, max_seq_len=512)
    seq_len = 512

    state = _random_attn_state(cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size, seed=0)
    x = torch.rand(1, seq_len, cfg.hidden_size, dtype=torch.float32) * 2 - 1
    cos_meta, sin_meta = build_meta_cos_sin(hf_config, seq_len)
    cos_hf, sin_hf = _hf_cos_sin_from_meta(cos_meta, sin_meta)
    ref = _torch_attention(x, state, cos_hf, sin_hf, cfg)
    _, floor_pcc = comp_pcc(
        ref,
        _torch_attention(
            quantize_like_device(x.reshape(1, 1, seq_len, -1), ttnn.bfloat16).reshape(1, seq_len, -1),
            _quantise_weights(state, weight_dtype),
            cos_hf,
            sin_hf,
            cfg,
            quantise=lambda t, dt: quantize_like_device(t, dt),
        ),
        0.0,
    )

    default_pc = ProgramConfig()
    assert default_pc.fp32_dest_acc_en is True, "DEC-031: the package default is fp32_dest_acc_en=True"
    pccs = {}
    for name, pc in (
        ("fp32_dest_acc=True (package default)", default_pc),
        ("fp32_dest_acc=False (template default)", replace(default_pc, fp32_dest_acc_en=False)),
    ):
        attn = _build_attention(mesh_device, objs, cfg, state, weight_dtype=weight_dtype, program_config=pc)
        _, pcc = comp_pcc(ref, _run_tt_attention(mesh_device, attn, x, cos_meta, sin_meta), 0.0)
        pccs[name] = float(pcc)
        logger.info(
            f"[G-ATTN] compute-kernel A/B ({weight_dtype}, seq {seq_len}): {name}: PCC = {pcc} | "
            f"err ratio = {err_ratio(pcc, floor_pcc):.2f}x (floor {floor_pcc})"
        )

    best = pccs["fp32_dest_acc=True (package default)"]
    for name, pcc in pccs.items():
        assert best >= pcc - 1e-9, (
            f"[G-ATTN] the package default ({best}) is worse than {name} ({pcc}) at {weight_dtype}; "
            f"DEC-031 chose fp32_dest_acc_en=True on measured evidence — re-measure and re-decide"
        )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_sdpa_grid_is_pinned_and_asserted_at_build_time(mesh_device, reset_seeds, expect_error):
    """``DEC-012`` / Appendix F.8: the SDPA grid is a pinned 8x8 field and a bad one fails NOW.

    The dangerous alternative is deriving it from ``compute_with_storage_grid_size()`` — measured
    **(12, 10)** here — which passes every single-card gate (they never enter the ring path) and
    fails only at SP > 1 in P8, because
    ``ring_joint_sdpa_device_operation.cpp:421`` asserts
    ``ccl_core_grid_offset.x >= sdpa_grid.x`` with the offset pinned at ``grid.x - 1``.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    cfg = attention_config_from_hf(hf_config, max_seq_len=128)
    grid = mesh_device.compute_with_storage_grid_size()

    assert ProgramConfig().sdpa_core_grid == (8, 8), "DEC-012: the SDPA grid must stay a pinned 8x8"
    logger.info(
        f"[G-ATTN] device compute grid = ({grid.x}, {grid.y}); pinned SDPA grid = (8, 8); "
        f"ring constraint sdpa.x <= grid.x - 1 = {grid.x - 1}"
    )
    ProgramConfig().assert_sdpa_grid_fits(mesh_device)  # the pinned grid must pass

    # The device-derived grid this recipe explicitly warns against must be refused at BUILD time.
    from dataclasses import replace

    derived = replace(ProgramConfig(), sdpa_core_grid=(grid.x, grid.y))
    with expect_error(AssertionError, "sdpa_core_grid.x"):
        derived.assert_sdpa_grid_fits(mesh_device)
    with expect_error(AssertionError, "sdpa_core_grid.x"):
        _build_attention(
            mesh_device,
            objs,
            cfg,
            _random_attn_state(cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size, seed=0),
            weight_dtype=ttnn.bfloat16,
            program_config=derived,
        )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_cached_len_and_bias_and_partial_rotary_all_fail_loud(mesh_device, reset_seeds, expect_error):
    """The three silent-wrongness paths this module refuses instead of approximating.

    1. ``cached_len > 0`` on a single device: a plain ``is_causal`` SDPA assumes Q row 0 aligns with
       K row 0, so it is off by ``cached_len`` and returns a correctly-shaped wrong answer.
    2. an attention bias in the state dict (``attention_bias: false``) — asserted absent, since
       this module has no bias path at all.
    3. partial rotary (``rotary_dim != head_dim``) — not implemented.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    cfg = attention_config_from_hf(hf_config, max_seq_len=128)
    seq_len = 128
    state = _random_attn_state(cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size, seed=0)

    attn = _build_attention(mesh_device, objs, cfg, state, weight_dtype=ttnn.bfloat16)
    x = torch.rand(1, seq_len, cfg.hidden_size, dtype=torch.float32) * 2 - 1
    cos_meta, sin_meta = build_meta_cos_sin(hf_config, seq_len)
    with expect_error(NotImplementedError, "cached_len>0"):
        _run_tt_attention(mesh_device, attn, x, cos_meta, sin_meta, cached_len=32)

    with_bias = dict(state)
    with_bias["q_proj.bias"] = torch.zeros(cfg.num_heads * cfg.head_dim)
    with expect_error(AssertionError, "attention_bias"):
        _build_attention(mesh_device, objs, cfg, with_bias, weight_dtype=ttnn.bfloat16)

    from dataclasses import replace as dc_replace

    with expect_error(AssertionError, "FULL rotary"):
        dc_replace(cfg, rotary_dim=64)
    logger.info("[G-ATTN] cached_len>0, attention bias and partial rotary all fail loud")


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_sdpa_kernel_error_is_the_dominant_term(mesh_device, reset_seeds):
    """Attribute the block-level gap to the noise floor: it is ``scaled_dot_product_attention``.

    A storage-dtype noise floor (``DEC-032``) models the rounding of tensors, not the *interior* of
    a kernel. This measures ``ttnn.transformer.scaled_dot_product_attention`` on its own — bf16 Q/K/V
    straight in, no projections, no ``o_proj`` — against a torch reference fed the identically
    bf16-rounded Q/K/V with fp32 arithmetic. The resulting ratio is the SDPA kernel's own error in
    units of its own floor, and it is what ``MAX_ERR_RATIO`` above budgets for (``DEC-034``).

    It is also the check that the block-level number is *not* hiding a projection or RoPE bug: if
    this ratio ever drops materially while the block ratio stays high, the extra error is ours.
    """
    objs = TestFactory.setup_test(mesh_device)
    cfg = attention_config_from_hf(objs["hf_config"], max_seq_len=128)
    seq_len, nq, nkv, hd = 128, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim

    q = (torch.rand(1, nq, seq_len, hd) * 2 - 1) * 0.5
    k = (torch.rand(1, nkv, seq_len, hd) * 2 - 1) * 0.5
    v = (torch.rand(1, nkv, seq_len, hd) * 2 - 1) * 0.5

    def _ref(qq, kk, vv):
        rep = nq // nkv
        s = (qq @ kk.repeat_interleave(rep, dim=1).transpose(-1, -2)) * cfg.scaling
        s = s + torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        return torch.softmax(s, dim=-1) @ vv.repeat_interleave(rep, dim=1)

    ref = _ref(q, k, v)
    floor = _ref(*(quantize_like_device(t, ttnn.bfloat16) for t in (q, k, v)))

    def _dev(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    pc = ProgramConfig()
    tt_out = ttnn.transformer.scaled_dot_product_attention(
        _dev(q),
        _dev(k),
        _dev(v),
        is_causal=True,
        scale=cfg.scaling,
        program_config=pc.get_prefill_sdpa_config(mesh_device, seq_len),
        compute_kernel_config=pc.get_compute_kernel_config(mesh_device),
    )
    out = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].float()

    _, pcc = comp_pcc(ref, out, 0.0)
    _, floor_pcc = comp_pcc(ref, floor, 0.0)
    logger.info(
        f"[G-ATTN] SDPA kernel alone (GQA {nq}/{nkv}, head_dim {hd}, seq {seq_len}, no on-chip KV "
        f"repeat): PCC = {pcc} | bf16-input floor = {floor_pcc} | err ratio = "
        f"{err_ratio(pcc, floor_pcc):.1f}x"
    )
    # The op must still be correct in absolute terms; the ratio is diagnostic, not a gate.
    assert float(pcc) > 0.9999, f"the SDPA kernel itself scored {pcc} on bf16 GQA input"


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["wbf8_b", "wbf16"])
@torch.no_grad()
def test_qkv_and_rope_stage_is_at_the_floor(mesh_device, weight_dtype, reset_seeds):
    """The stages this package implements itself must sit AT the noise floor (``DEC-034``).

    Projections -> GQA head split -> full RoPE, stopping before SDPA, so the kernel term measured by
    ``test_sdpa_kernel_error_is_the_dominant_term`` is excluded and nothing can hide behind it.
    Device Q/K come back in the **Meta** layout, so the HF-convention reference is mapped with
    ``_hf_to_meta_layout`` (``tests/unit/test_rope_vs_ref.py:65``) — the same relation the weight
    ``reverse_permute`` encodes. V carries no convention and is compared directly.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    cfg = attention_config_from_hf(hf_config, max_seq_len=512)
    seq_len, nq, nkv, hd = 512, cfg.num_heads, cfg.num_kv_heads, cfg.head_dim

    state = _random_attn_state(nq, nkv, hd, cfg.hidden_size, seed=0)
    x = torch.rand(1, seq_len, cfg.hidden_size, dtype=torch.float32) * 2 - 1
    cos_meta, sin_meta = build_meta_cos_sin(hf_config, seq_len)
    cos_hf, sin_hf = _hf_cos_sin_from_meta(cos_meta, sin_meta)

    def _stage(xx, st, quantise=None):
        """Projections + RoPE only, in the Meta layout the device produces."""
        ident = (lambda t, _dt: t) if quantise is None else quantise
        q = (xx @ st["q_proj.weight"].t()).view(1, seq_len, nq, hd).transpose(1, 2)
        k = (xx @ st["k_proj.weight"].t()).view(1, seq_len, nkv, hd).transpose(1, 2)
        v = (xx @ st["v_proj.weight"].t()).view(1, seq_len, nkv, hd).transpose(1, 2)
        q, k, v = ident(q, ttnn.bfloat16), ident(k, ttnn.bfloat16), ident(v, ttnn.bfloat16)
        q = ident(_hf_to_meta_layout(_rope_hf(q, cos_hf, sin_hf)), ttnn.bfloat16)
        k = ident(_hf_to_meta_layout(_rope_hf(k, cos_hf, sin_hf)), ttnn.bfloat16)
        # V is NOT layout-mapped: only q_proj/k_proj are reverse_permute'd, because only Q and K
        # are rotated. Mapping V here scores PCC ~0.015 — measured, and a useful confirmation
        # that the swizzle really is Q/K-only.
        return q, k, v

    ref = _stage(x, state)
    floor = _stage(
        quantize_like_device(x.reshape(1, 1, seq_len, -1), ttnn.bfloat16).reshape(1, seq_len, -1),
        _quantise_weights(state, weight_dtype),
        quantise=lambda t, dt: quantize_like_device(t, dt),
    )

    attn = _build_attention(mesh_device, objs, cfg, state, weight_dtype=weight_dtype)

    def _to_dev(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    tt_q_flat, tt_kv_flat = apply_qkv_projection(
        _to_dev(x.reshape(1, 1, seq_len, -1)), attn.weights, attn.compute_kernel_config
    )
    tt_q, tt_k, tt_v = split_qkv_heads_prefill(tt_q_flat, tt_kv_flat, nq, nkv)
    trans = attn.transformation_mats["prefill"]
    tt_q = apply_rope(tt_q, [_to_dev(cos_meta), _to_dev(sin_meta)], trans)
    tt_k = apply_rope(tt_k, [_to_dev(cos_meta), _to_dev(sin_meta)], trans)

    def _host(t):
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].float()

    for name, dev_t, ref_t, floor_t in (
        ("Q post-RoPE", tt_q, ref[0], floor[0]),
        ("K post-RoPE", tt_k, ref[1], floor[1]),
        ("V", tt_v, ref[2], floor[2]),
    ):
        got = _host(dev_t)
        assert got.shape == ref_t.shape, f"{name}: device {tuple(got.shape)} vs ref {tuple(ref_t.shape)}"
        _, pcc = comp_pcc(ref_t, got, 0.0)
        _, floor_pcc = comp_pcc(ref_t, floor_t, 0.0)
        ratio = err_ratio(pcc, floor_pcc)
        logger.info(
            f"[G-ATTN] stage ({weight_dtype}): {name}: PCC = {pcc} | floor = {floor_pcc} | " f"err ratio = {ratio:.2f}x"
        )
        assert ratio <= MAX_ERR_RATIO_QKV_STAGE, (
            f"[G-ATTN] {name} at {weight_dtype} sits {ratio:.1f}x off its noise floor {floor_pcc} "
            f"(PCC {pcc}) — this stage is implemented by tt/attention/, so the SDPA kernel term "
            f"cannot explain it (DEC-034)"
        )
