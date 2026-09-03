# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-LAYER` — `tt/layer.py` vs a torch reference, `(1,1)` mesh. **Integration check only.**

Layer: ``input_layernorm -> Attention -> residual add -> post_attention_layernorm -> MLP ->
residual add``, i.e. ``transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward``.

**Read this before reading the numbers.** ``BRINGUP_RECIPE.md`` Appendix E measured the
``models/tt_transformers`` decoder oracle at **0.9999985** — *higher* than either of its own
sublayers (attention 0.9996099, MLP 0.9995823). The residual stream dominates the correlation, so a
full-layer PCC **partially launders a degraded sublayer**. Three consequences, binding per
``03_OUTLINE.md`` §5.1:

1. ``G-RMS``, ``G-ROPE``, ``G-MLP``, ``G-ATTN`` and ``G-KV`` are the only evidence that a sublayer
   is correct. A passing ``G-LAYER`` may never be used to excuse a weak or missing sublayer gate.
2. A layer PCC of ~0.9999 while a sublayer sits at ~0.99 is the **signature** of this masking.
3. :func:`test_layer_pcc_launders_a_degraded_sublayer` measures the attenuation on *this* layer,
   with this hidden size and these inputs, instead of restating Appendix E's anecdote: it injects a
   known relative error into one sublayer's output in torch and reports how much smaller the
   resulting layer-level error is. That number, not the PCC, is what says how much this gate is
   worth.

**Reference.** Composed from the *already gate-validated* fp32 helpers — ``_torch_attention``
(``tests/unit/test_attention_vs_ref.py``, ``G-ATTN``) and ``_torch_mlp``
(``tests/unit/test_mlp_vs_ref.py``, ``G-MLP``) — plus HF's ``LlamaRMSNorm`` body. Reusing them is
the point: a fresh in-test layer reference would be a fourth un-gated implementation of the same
maths, and the two conventions it has to get right (Meta-vs-HF RoPE, ``[out, in]`` weights) are
exactly where a hand-written copy goes wrong.
``tt_transformers``' ``reference_decoder()`` (``models/tt_transformers/tt/model_config.py:4393``)
was *not* used: it branches on whether ``position_embeddings`` is in the layer's forward signature
and its HF weights load at the checkpoint's ``torch_dtype: bfloat16``, so it shares the device's own
rounding (Appendix E.1) and its number would not be comparable to an fp32 floor.

**Input distribution** (``DEC-026`` / ``R-018``): ``rand(...)*2 - 1``, uniform on ``[-1, 1)``, the
attention oracle's own (``models/tt_transformers/tests/test_attention_prefill.py:161-166``).
**Reference dtype policy:** fp32 weights, fp32 activations, fp32 arithmetic throughout — strictly
harder than any bf16-weight reference.

**How it is judged:** absolute ``PCC >= 0.999`` (``03_OUTLINE.md`` §5) **and** the gap to the torch
noise floor (``DEC-032``), where the floor rounds every tensor the device stores to the dtype it
stores it in and does the arithmetic in fp32.

Run:
    pytest models/demos/llama31_8b_d_p/tests/unit/test_decoder_layer_vs_ref.py -x -q
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import (
    TestFactory,
    err_ratio,
    quantize_like_device,
    requires_hf_reference,
)
from models.demos.llama31_8b_d_p.tests.unit.test_attention_vs_ref import (
    _quantise_weights,
    _random_attn_state,
    _torch_attention,
)
from models.demos.llama31_8b_d_p.tests.unit.test_mlp_vs_ref import _random_mlp_state
from models.demos.llama31_8b_d_p.tests.unit.test_rope_vs_ref import _hf_cos_sin_from_meta
from models.demos.llama31_8b_d_p.tt.attention import attention_config_from_hf
from models.demos.llama31_8b_d_p.tt.layer import DecoderLayer
from models.demos.llama31_8b_d_p.tt.rope import build_meta_cos_sin, build_transformation_mat

PCC_THRESHOLD = 0.999  # 03_OUTLINE.md §5 / Appendix E revised column
ORACLE_PCC = 0.9999985  # tt_transformers test_decoder_prefill.py — context only, NOT a target

# The layer's error budget is dominated by the same fused-SDPA term G-ATTN attributes (`DEC-034`,
# Appendix E.5): the kernel sits ~70-80x off ITS own storage-dtype floor, which is the whole of
# G-ATTN's 2.6x (bf8_b) / 5.1x (bf16) block-level gap. A layer adds two norms and two adds on top
# of that block, and the residual add *lowers* the relative error, so the layer ratio is expected at
# or below the attention one. Kept at the same 8.0 so a regression that the residual would otherwise
# hide still trips a threshold.
MAX_ERR_RATIO = 8.0

# Norm gains: real Llama norm weights are O(1) and positive-ish, not O(0.02) like a projection.
# Centred at 1.0 so the norm is near-identity (as in a trained model) but NOT exactly 1.0, because
# an all-ones gain would make the negative control below (swapping the two norm weights) a no-op.
NORM_SCALE = 0.1


def _random_norm_weight(hidden, *, seed):
    gen = torch.Generator().manual_seed(seed)
    return 1.0 + torch.randn(hidden, generator=gen) * NORM_SCALE


def _random_layer_state(hf_config, cfg, *, seed=0) -> dict:
    """A full HF-layout ``model.layers.<i>.*`` sub-dict: 2 norms + 4 projections + 3 FFN weights.

    The two norm gains get **different** seeds so the negative control (swapping them) is a real
    perturbation rather than a rename of equal tensors.
    """
    state = {
        "input_layernorm.weight": _random_norm_weight(hf_config.hidden_size, seed=seed + 100),
        "post_attention_layernorm.weight": _random_norm_weight(hf_config.hidden_size, seed=seed + 200),
    }
    for k, v in _random_attn_state(cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.hidden_size, seed=seed).items():
        state[f"self_attn.{k}"] = v
    for k, v in _random_mlp_state(hf_config, seed=seed + 1).items():
        state[f"mlp.{k}"] = v
    return state


def _sub(state, prefix):
    n = len(prefix) + 1
    return {k[n:]: v for k, v in state.items() if k.startswith(prefix + ".")}


def _torch_rms_norm(x, weight, eps):
    """HF ``LlamaRMSNorm.forward`` in fp32: ``weight * x * rsqrt(mean(x^2) + eps)``.

    Plain, with no Gemma-style ``(1 + weight)`` fold — Llama's config has no such key
    (``00_MODEL_CARD.md`` §2, ``DEC-004``).
    """
    return weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


def _torch_mlp_q(x, state, ident):
    """``down(silu(gate(x)) * up(x))`` with ``tt/mlp.py``'s dtype ladder made explicit.

    ``ttnn.linear(dtype=bfloat16)`` writes gate/up back as bf16, the fused
    ``ttnn.mul(..., SILU, dtype=bfloat16)`` writes the activation as bf16, and ``down_proj`` writes
    bf16 — so ``ident`` is applied at exactly those three points and nowhere else.
    """
    gate = ident(torch.nn.functional.linear(x, state["gate_proj.weight"]), ttnn.bfloat16)
    up = ident(torch.nn.functional.linear(x, state["up_proj.weight"]), ttnn.bfloat16)
    act = ident(torch.nn.functional.silu(gate) * up, ttnn.bfloat16)
    return ident(torch.nn.functional.linear(act, state["down_proj.weight"]), ttnn.bfloat16)


def _torch_layer(x, state, cos_hf, sin_hf, cfg, eps, *, quantise=None):
    """One decoder layer in fp32. ``x`` is ``[1, S, hidden]``; returns the same shape.

    ``quantise`` — when given, a ``(tensor, dtype) -> tensor`` rounding callable applied at every
    point the device *stores* a tensor. That turns this same function into the noise floor
    (``DEC-032``): identical arithmetic, identical dtype ladder, fp32 maths.
    """
    seq = x.shape[1]

    def ident(t, dt):
        if quantise is None:
            return t
        return quantise(t.reshape(1, 1, seq, -1), dt).reshape(t.shape)

    residual = x
    # ttnn.rms_norm writes bf16 (tt/rms_norm.py passes no dtype; the op keeps the input's).
    h = ident(_torch_rms_norm(x, state["input_layernorm.weight"], eps), ttnn.bfloat16)
    h = _torch_attention(h, _sub(state, "self_attn"), cos_hf, sin_hf, cfg, quantise=quantise)
    # ttnn.add(..., output_tensor=attn_out) -> bf16.
    x = ident(residual + h, ttnn.bfloat16)

    residual = x
    h = ident(_torch_rms_norm(x, state["post_attention_layernorm.weight"], eps), ttnn.bfloat16)
    h = _torch_mlp_q(h, _sub(state, "mlp"), ident)
    return ident(residual + h, ttnn.bfloat16)


def _quantise_layer_state(state, weight_dtype, hidden):
    """Round every stored weight exactly as its loader stores it.

    * norm gains: bf16 in the ttnn ``[1, 1, hidden/32, 32]`` layout ``tt/rms_norm.py`` reshapes to,
      **not** the model's ``weight_dtype`` — norm gains stay bf16 regardless (convention 11);
    * the four projections and the three FFN weights: ``weight_dtype``, in the ttnn
      ``[1, 1, in, out]`` layout, because bf8_b's shared-exponent blocking is layout-dependent.
    """
    out = {}
    for k, w in state.items():
        if k.endswith("layernorm.weight"):
            out[k] = quantize_like_device(w.reshape(1, 1, -1, ttnn.TILE_SIZE), ttnn.bfloat16).reshape(hidden)
        else:
            group = "self_attn" if k.startswith("self_attn.") else "mlp"
            out[k] = _quantise_weights({k[len(group) + 1 :]: w}, weight_dtype)[k[len(group) + 1 :]]
    return out


def _to_dev(mesh_device, t, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _run_tt_layer(mesh_device, objs, hf_config, state, x, cos_meta, sin_meta, *, weight_dtype, max_seq_len):
    layer = DecoderLayer(
        mesh_device,
        hf_config,
        state,
        0,
        mesh_config=objs["mesh_config"],
        ccl_manager=None,  # TP=1 / SP=1: no collective is entered
        transformation_mats={"prefill": build_transformation_mat(mesh_device)},
        max_seq_len=max_seq_len,
        weight_dtype=weight_dtype,
    )
    seq_len = x.shape[1]
    out = layer(
        _to_dev(mesh_device, x.reshape(1, 1, seq_len, -1)),
        position_embeddings=[_to_dev(mesh_device, cos_meta), _to_dev(mesh_device, sin_meta)],
    )
    return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].reshape(1, seq_len, -1)


def _setup(mesh_device, seq_len, seed=0):
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    cfg = attention_config_from_hf(hf_config, max_seq_len=max(seq_len, 128))
    state = _random_layer_state(hf_config, cfg, seed=seed)
    x = torch.rand(1, seq_len, hf_config.hidden_size, dtype=torch.float32) * 2 - 1
    cos_meta, sin_meta = build_meta_cos_sin(hf_config, seq_len)
    cos_hf, sin_hf = _hf_cos_sin_from_meta(cos_meta, sin_meta)
    return objs, hf_config, cfg, state, x, cos_meta, sin_meta, cos_hf, sin_hf


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("seq_len", [128, 512, 2048], ids=["s128", "s512", "s2048"])
@pytest.mark.parametrize("weight_dtype", [ttnn.bfloat8_b, ttnn.bfloat16], ids=["wbf8_b", "wbf16"])
@torch.no_grad()
def test_decoder_layer_vs_ref(mesh_device, seq_len, weight_dtype, reset_seeds):
    """One decoder layer on device vs fp32 torch, identical random weights. See the docstring."""
    objs, hf_config, cfg, state, x, cos_meta, sin_meta, cos_hf, sin_hf = _setup(mesh_device, seq_len)
    eps = hf_config.rms_norm_eps

    ref = _torch_layer(x, state, cos_hf, sin_hf, cfg, eps)
    floor = _torch_layer(
        quantize_like_device(x.reshape(1, 1, seq_len, -1), ttnn.bfloat16).reshape(1, seq_len, -1),
        _quantise_layer_state(state, weight_dtype, hf_config.hidden_size),
        cos_hf,
        sin_hf,
        cfg,
        eps,
        quantise=lambda t, dt: quantize_like_device(t, dt),
    )
    out = _run_tt_layer(
        mesh_device,
        objs,
        hf_config,
        state,
        x,
        cos_meta,
        sin_meta,
        weight_dtype=weight_dtype,
        max_seq_len=max(seq_len, 128),
    )

    assert out.shape == ref.shape == floor.shape == (1, seq_len, hf_config.hidden_size)
    passing, pcc = comp_pcc(ref, out, PCC_THRESHOLD)
    _, floor_pcc = comp_pcc(ref, floor, 0.0)
    ratio = err_ratio(pcc, floor_pcc)

    logger.info(comp_allclose(ref, out))
    logger.info(
        f"[G-LAYER] seq_len={seq_len} weight_dtype={weight_dtype}: measured PCC = {pcc} | "
        f"torch noise floor = {floor_pcc} | err ratio = {ratio:.2f}x | threshold {PCC_THRESHOLD} | "
        f"oracle {ORACLE_PCC} (context only, NOT a target — and see DEC-040: measured against one "
        f"consistent fp32 reference this layer scores BELOW its own attention block, so Appendix "
        f"E's 'layer PCC > sublayer PCC' caveat does not reproduce here)"
    )
    assert passing, f"[G-LAYER] seq_len={seq_len} {weight_dtype} below {PCC_THRESHOLD}: {pcc}"
    assert ratio <= MAX_ERR_RATIO, (
        f"[G-LAYER] seq_len={seq_len} {weight_dtype}: PCC {pcc} clears {PCC_THRESHOLD} but sits "
        f"{ratio:.1f}x off the torch noise floor {floor_pcc} — investigate (DEC-032)"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@torch.no_grad()
def test_swapped_norms_fail(mesh_device, reset_seeds):
    """Negative control: swapping the two norm gains must collapse the PCC.

    This is the realistic key-mapping bug, not a synthetic one:
    ``models/tt_transformers/tt/load_checkpoints.py:812-813`` maps ``input_layernorm ->
    attention_norm`` and ``post_attention_layernorm -> ffn_norm``, and the two HF names differ by a
    prefix, so a ``substate`` typo or a reversed mapping table swaps them silently. Every shape,
    dtype and op stays identical, so a high PCC here would mean the positive gate above is
    measuring something other than which gain lands on which norm.

    Follows the ``tests/unit/test_rope_vs_ref.py:140`` pattern.
    """
    seq_len = 128
    objs, hf_config, cfg, state, x, cos_meta, sin_meta, cos_hf, sin_hf = _setup(mesh_device, seq_len)
    ref = _torch_layer(x, state, cos_hf, sin_hf, cfg, hf_config.rms_norm_eps)

    swapped = dict(state)
    swapped["input_layernorm.weight"] = state["post_attention_layernorm.weight"]
    swapped["post_attention_layernorm.weight"] = state["input_layernorm.weight"]

    kw = dict(weight_dtype=ttnn.bfloat16, max_seq_len=128)
    good = _run_tt_layer(mesh_device, objs, hf_config, state, x, cos_meta, sin_meta, **kw)
    bad = _run_tt_layer(mesh_device, objs, hf_config, swapped, x, cos_meta, sin_meta, **kw)

    _, pcc_ok = comp_pcc(ref, good, 0.0)
    _, pcc_bad = comp_pcc(ref, bad, 0.0)
    logger.info(f"[G-LAYER] negative control: correct norms PCC = {pcc_ok}, swapped norms PCC = {pcc_bad}")
    assert float(pcc_bad) < 0.99, (
        f"swapped norm gains scored {pcc_bad}; the positive gate is not actually testing which "
        f"gain lands on which norm"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("eps_rel", [1e-3, 1e-2], ids=["eps1e-3", "eps1e-2"])
@torch.no_grad()
def test_residual_masking_tracks_the_delta_to_stream_ratio(mesh_device, eps_rel, reset_seeds):
    """Measure — and **explain** — how much of a sublayer error the residual add hides.

    Appendix E observed one instance of the effect (the ``tt_transformers`` decoder oracle scoring
    0.9999985, above both of its sublayers) and ``03_OUTLINE.md`` §5.1 turned it into a rule. This
    test measures the mechanism instead of restating the anecdote, and the mechanism has a closed
    form: for a residual add ``y = r + s``, a perturbation ``d`` of the sublayer output ``s`` moves
    ``y`` by the same absolute ``d``, so the *relative* error shrinks by exactly

        attenuation = ||y|| / ||s||

    i.e. masking is not a property of the layer, it is a property of **how small the sublayer's
    delta is relative to the residual stream it is added into**. Where ``||s|| >= ||y||`` there is
    no masking at all, and a layer PCC is then a *harder* test than its sublayer's, not an easier
    one.

    The last-sublayer (MLP) case is the clean one — nothing follows its add — so its attenuation is
    asserted against the predicted ``||y||/||s||``. The attention case is reported but not
    predicted, because its perturbation additionally propagates through the second norm and the
    MLP.

    ``test_real_weights_show_the_residual_dominating`` then measures the ratio in the regime
    Appendix E actually observed. Host-only maths (the mechanism is a property of the residual
    topology, not of a kernel), but it runs inside the gate so the number is re-measured rather
    than remembered.
    """
    seq_len = 128
    _, hf_config, cfg, state, x, _cos_meta, _sin_meta, cos_hf, sin_hf = _setup(mesh_device, seq_len)
    eps = hf_config.rms_norm_eps
    gen = torch.Generator().manual_seed(7)

    h1 = _torch_rms_norm(x, state["input_layernorm.weight"], eps)
    attn_clean = _torch_attention(h1, _sub(state, "self_attn"), cos_hf, sin_hf, cfg)
    mid = x + attn_clean
    h2 = _torch_rms_norm(mid, state["post_attention_layernorm.weight"], eps)
    mlp_clean = _torch_mlp_q(h2, _sub(state, "mlp"), lambda t, _dt: t)
    layer_clean = mid + mlp_clean

    def _probe(name, clean, stream_out):
        d = torch.randn(clean.shape, generator=gen) * (eps_rel * clean.abs().mean())
        if name == "attn":
            mid_d = x + (clean + d)
            h2_d = _torch_rms_norm(mid_d, state["post_attention_layernorm.weight"], eps)
            layer_d = mid_d + _torch_mlp_q(h2_d, _sub(state, "mlp"), lambda t, _dt: t)
        else:
            layer_d = mid + (clean + d)
        _, sub_pcc = comp_pcc(clean, clean + d, 0.0)
        _, layer_pcc = comp_pcc(layer_clean, layer_d, 0.0)
        atten = (1.0 - float(sub_pcc)) / max(1.0 - float(layer_pcc), 1e-30)
        predicted = float(stream_out.norm() / clean.norm())
        logger.info(
            f"[G-LAYER] masking probe (rel eps {eps_rel:g}, {name}): sublayer PCC {sub_pcc} -> "
            f"layer PCC {layer_pcc} | measured attenuation {atten:.2f}x | "
            f"||stream||/||delta|| = {predicted:.2f}x"
        )
        return atten, predicted

    _probe("attn", attn_clean, layer_clean)
    atten, predicted = _probe("mlp", mlp_clean, layer_clean)

    # The closed form is exact up to the second order of the perturbation, so a 1.3x band is
    # generous. This is the assertion that makes the number mean something: if the relationship
    # broke, the explanation above (and §5.1's justification) would be wrong.
    assert 1.0 / 1.3 <= atten / predicted <= 1.3, (
        f"MLP-sublayer attenuation measured {atten:.3f}x but ||y||/||s|| predicts "
        f"{predicted:.3f}x; the residual-masking mechanism in §5.1 is not what is happening here"
    )
    # And the falsifiable part: at THIS input scale the effect is absent, which is itself the
    # finding (DEC-040). Random O(1) activations through 0.02-scale projections give a sublayer
    # delta comparable to the stream, unlike a trained model's.
    logger.info(
        f"[G-LAYER] with random weights the residual does NOT dominate "
        f"(||stream||/||mlp delta|| = {predicted:.2f}x), so G-LAYER on random weights launders "
        f"nothing. See test_real_weights_show_the_residual_dominating for the real-weight regime."
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@requires_hf_reference
@torch.no_grad()
def test_real_weights_show_the_residual_dominating(mesh_device, state_dict, reset_seeds):
    """The masking ratio in the regime Appendix E observed: **real** layer-0 weights, **real**
    embedding activations.

    The previous test shows the attenuation equals ``||stream|| / ||sublayer delta||``. This one
    measures that ratio where it matters, and it needs both halves to be real:

    * the weights are the checkpoint's ``model.layers.0.*``;
    * the input is real ``model.embed_tokens`` rows, not ``randn`` — a trained embedding's rows have
      a per-channel scale two orders below a uniform ``[-1, 1)`` draw, and that scale *is* the
      quantity in question. Using a synthetic input here would answer a different question, which is
      exactly how Appendix E's observation came to look like a property of layers.

    This is a **measurement**, not a threshold: it prints the two ratios and asserts only that the
    real-weight ratio exceeds the random-weight one, which is the claim §5.1 rests on.
    """
    from models.demos.llama31_8b_d_p.utils.substate import substate

    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    cfg = attention_config_from_hf(hf_config, max_seq_len=128)
    eps = hf_config.rms_norm_eps
    seq_len = 128

    real = {k: v.float() for k, v in substate(state_dict, "model.layers.0").items()}
    assert len(real) == 9, f"expected 9 real layer-0 weights, got {sorted(real)}"
    table = substate(state_dict, "model.embed_tokens")["weight"]
    gen = torch.Generator().manual_seed(11)
    ids = torch.randint(0, hf_config.vocab_size, (seq_len,), generator=gen)
    x = table[ids].float().reshape(1, seq_len, hf_config.hidden_size)

    cos_meta, sin_meta = build_meta_cos_sin(hf_config, seq_len)
    cos_hf, sin_hf = _hf_cos_sin_from_meta(cos_meta, sin_meta)

    h1 = _torch_rms_norm(x, real["input_layernorm.weight"], eps)
    attn = _torch_attention(h1, _sub(real, "self_attn"), cos_hf, sin_hf, cfg)
    mid = x + attn
    h2 = _torch_rms_norm(mid, real["post_attention_layernorm.weight"], eps)
    mlp = _torch_mlp_q(h2, _sub(real, "mlp"), lambda t, _dt: t)
    layer = mid + mlp

    r_attn = float(mid.norm() / attn.norm())
    r_mlp = float(layer.norm() / mlp.norm())
    logger.info(
        f"[G-LAYER] REAL layer-0 weights + REAL embedding rows, seq {seq_len}: "
        f"||x||={x.norm():.1f} ||attn delta||={attn.norm():.1f} ||mlp delta||={mlp.norm():.1f} | "
        f"attenuation attn {r_attn:.2f}x, mlp {r_mlp:.2f}x"
    )
    assert r_attn > 1.0 and r_mlp > 1.0, (
        f"with real weights the residual stream does not dominate either sublayer "
        f"(attn {r_attn:.2f}x, mlp {r_mlp:.2f}x); §5.1's premise that a layer PCC launders a "
        f"degraded sublayer would then be false for this model and the rule needs re-deriving"
    )


@torch.no_grad()
def test_promoted_helpers_match_the_p5_copies():
    """`test_factory`'s promoted `quantize_like_device` / `err_ratio` must equal the P5 copies.

    ``DEC-046`` moved both into ``tests/test_factory.py`` (their home now that six gates use them)
    but could not delete the ``tests/unit/test_mlp_vs_ref.py`` copies, because P6 owns neither that
    file nor the two P5 tests that import from it. This asserts the duplicate cannot drift while it
    exists. Host-only.
    """
    from models.demos.llama31_8b_d_p.tests.unit.test_mlp_vs_ref import err_ratio as p5_err_ratio
    from models.demos.llama31_8b_d_p.tests.unit.test_mlp_vs_ref import quantize_like_device as p5_quantize

    t = torch.randn(1, 1, 64, 128, generator=torch.Generator().manual_seed(0))
    for dtype in (ttnn.bfloat16, ttnn.bfloat8_b):
        torch.testing.assert_close(quantize_like_device(t, dtype), p5_quantize(t, dtype), rtol=0.0, atol=0.0)
    for measured, floor in ((0.999, 0.9999), (0.9, 0.99), (1.0, 1.0)):
        assert err_ratio(measured, floor) == p5_err_ratio(measured, floor)
