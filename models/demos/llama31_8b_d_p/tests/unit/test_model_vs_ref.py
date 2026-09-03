# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Gate `G-MODEL` — the full `tt/model.py` stack vs HuggingFace, `(1,1)` mesh. **Integration only.**

``embedding -> DecoderLayer x N -> final RMSNorm -> LMHead`` against
``transformers.LlamaForCausalLM`` loaded in **fp32** from the real Llama-3.1-8B-Instruct checkpoint,
with the *same* weights on both sides.

**Reference — and why HF is the reference here, unlike every other gate in this package.** The
sublayer gates use in-test torch maths because a checkpoint is neither needed nor faster there. A
32-layer stack is different: writing a second full model to compare against would be writing the
bug twice. So the oracle is HF itself, loaded at ``dtype=torch.float32`` with
``attn_implementation="eager"``, and three things are checked *about the oracle* before it is
trusted:

* :func:`test_hf_reference_is_causal` perturbs the last token and asserts every earlier row is
  **bit-identical**. Appendix F.2 warns that HF's eager path applies only the mask it is handed, so
  a reference that silently ran non-causal attention would look exactly like a model bug. Measured
  here: ``max|delta| = 0.0`` on rows ``[:-1]``, so ``create_causal_mask`` does build the mask when
  ``attention_mask=None``. (F.2's warning is real for hand-written ``eager_attention_forward``
  calls; it does **not** apply to ``LlamaModel.forward``.)
* :func:`test_in_test_torch_reference_agrees_with_hf` shows the composed in-test reference
  (``tests/unit/test_decoder_layer_vs_ref.py``'s ``_torch_layer``, i.e. ``G-LAYER``'s and
  ``G-ATTN``'s and ``G-MLP``'s reference maths) reproduces HF on real weights. That both validates
  the sublayer gates' oracle and licenses using the same code to build this gate's noise floor.
* the ``rope_parameters`` HF resolves are printed, so a wrong theta cannot hide (Appendix F.2's
  highest-severity trap).

**How it is judged.** Hidden-state ``PCC >= 0.999`` (``03_OUTLINE.md`` §5) and **top-1 token
agreement = 100%** at the last position, plus the gap to the torch noise floor (``DEC-032``): the
same stack with every device-stored tensor rounded to the dtype the device holds it in and all
arithmetic in fp32.

**This is an integration check and nothing more** (``03_OUTLINE.md`` §5.1). ``G-RMS``, ``G-ROPE``,
``G-MLP``, ``G-ATTN`` and ``G-KV`` are the only evidence that a sublayer is correct. What this gate
adds that they cannot is :func:`test_full_stack_per_layer_pcc_curve`: a **step** in the per-layer
curve localises a single bad layer, which no aggregate number does.

**Input distribution:** real token ids drawn uniformly from the vocabulary with a fixed seed — not
``randn`` activations, because the input to a full model *is* a token id and there is no other
admissible distribution for it. **Reference dtype policy:** fp32 weights, fp32 activations, fp32
arithmetic on both the HF and the floor side.

Run:
    export HF_MODEL=/path/to/Llama-3.1-8B-Instruct
    pytest models/demos/llama31_8b_d_p/tests/unit/test_model_vs_ref.py -x -q
"""

from __future__ import annotations

import datetime
import pathlib

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import (
    TestFactory,
    err_ratio,
    hf_model_path,
    quantize_like_device,
    requires_hf_reference,
)
from models.demos.llama31_8b_d_p.tests.unit.test_decoder_layer_vs_ref import (
    _quantise_layer_state,
    _torch_layer,
    _torch_rms_norm,
)
from models.demos.llama31_8b_d_p.tests.unit.test_rope_vs_ref import _hf_cos_sin_from_meta
from models.demos.llama31_8b_d_p.tt.attention import attention_config_from_hf
from models.demos.llama31_8b_d_p.tt.model import Model
from models.demos.llama31_8b_d_p.tt.rope import build_meta_cos_sin
from models.demos.llama31_8b_d_p.utils.substate import substate

PCC_THRESHOLD = 0.999  # 03_OUTLINE.md §5
# The stack inherits G-LAYER's per-layer gap (1.4-2.9x, itself dominated by the fused SDPA kernel's
# ~71x off ITS own floor — DEC-034 / Appendix E.5) and accumulates it over N layers, so the budget
# is the layer budget with headroom for accumulation rather than a new number. Recorded, not tuned:
# see DEC-047 for the measured values it was set from.
MAX_ERR_RATIO = 8.0
# A *step* in the per-layer error curve is a single-layer bug; smooth growth is accumulation.
# DEC-047 sets this from the measured 32-layer curve.
MAX_LAYER_ERROR_STEP = 4.0
# Layers 0-1 climb off a near-exact baseline (error ~1e-7), where a 10x ratio is noise, not a step.
STEP_CHECK_FROM_LAYER = 3

_RAW_DIR = pathlib.Path(__file__).resolve().parents[2] / "bringup_log" / "raw"


# --------------------------------------------------------------------------------------
# References
# --------------------------------------------------------------------------------------
def _hf_model(num_layers):
    """``LlamaForCausalLM`` in fp32, eager attention, truncated to ``num_layers``.

    ``num_hidden_layers`` is passed straight to ``from_pretrained``, which forwards it to the
    config, so only the first ``num_layers`` layers' tensors are materialised (0.5 s at N=2). The
    unused-key report it prints is expected and is itself useful evidence that the truncation took.
    """
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_path(),
        dtype=torch.float32,
        num_hidden_layers=num_layers,
        attn_implementation="eager",
    )
    model.eval()
    assert model.config.num_hidden_layers == num_layers
    assert model.config._attn_implementation == "eager", (
        f"HF resolved attn_implementation to {model.config._attn_implementation!r}, not 'eager'; "
        f"the mask contract this gate relies on is only checked for the eager path (F.2)"
    )
    logger.info(f"[G-MODEL] HF reference: {num_layers} layers, fp32, eager, rope={model.config.rope_parameters}")
    return model


def _hf_run(model, input_ids):
    """``(last_hidden_state, logits, [per-layer output])`` — per-layer via forward hooks.

    Forward hooks rather than ``output_hidden_states=True``: on transformers 5.12.1 that flag is
    served by a ``@capture_outputs`` decorator (``modeling_llama.py:373``) whose tuple layout (does
    entry ``i`` mean the *input*
    to layer ``i`` or its output? is the last entry pre- or post-final-norm?) is not visible at the
    call site. A hook on ``LlamaDecoderLayer`` returns exactly its output tensor
    (``modeling_llama.py:332``), which is unambiguously what ``tt/layer.py`` returns too.
    """
    captured = {}
    handles = [
        layer.register_forward_hook(lambda _m, _i, out, k=k: captured.__setitem__(k, out.detach().float()))
        for k, layer in enumerate(model.model.layers)
    ]
    try:
        with torch.no_grad():
            last_hidden = model.model(input_ids).last_hidden_state
            logits = model.lm_head(last_hidden)
    finally:
        for h in handles:
            h.remove()
    per_layer = [captured[k] for k in range(len(model.model.layers))]
    return last_hidden, logits, per_layer


def _torch_stack(state_dict, hf_config, cfg, input_ids, num_layers, *, weight_dtype=None):
    """The in-test reference stack. ``weight_dtype=None`` -> pure fp32; else the **noise floor**.

    With a ``weight_dtype`` this is exactly ``DEC-032``'s floor for the whole model: every tensor
    the device stores (the embedding table, both norm gains per layer, the seven projections per
    layer, the final norm gain, the LM head, and every intermediate activation) is rounded to the
    dtype the device holds it in, and all arithmetic is fp32. It reuses ``_torch_layer`` — the same
    function ``G-LAYER`` is measured against, which
    :func:`test_in_test_torch_reference_agrees_with_hf` shows reproduces HF.
    """
    q = (lambda t, dt: quantize_like_device(t, dt)) if weight_dtype is not None else None
    hidden, seq = hf_config.hidden_size, input_ids.shape[-1]

    def ident(t, dt):
        return t if q is None else q(t.reshape(1, 1, seq, -1), dt).reshape(t.shape)

    cos_meta, sin_meta = build_meta_cos_sin(hf_config, seq)
    cos_hf, sin_hf = _hf_cos_sin_from_meta(cos_meta, sin_meta)
    eps = hf_config.rms_norm_eps

    table = substate(state_dict, "model.embed_tokens")["weight"].float()
    if weight_dtype is not None:
        table = quantize_like_device(table.reshape(1, 1, -1, hidden), ttnn.bfloat16).reshape(-1, hidden)
    x = table[input_ids.reshape(-1)].reshape(1, seq, hidden)

    per_layer = []
    for i in range(num_layers):
        layer_state = {k: v.float() for k, v in substate(state_dict, f"model.layers.{i}").items()}
        if weight_dtype is not None:
            layer_state = _quantise_layer_state(layer_state, weight_dtype, hidden)
        x = _torch_layer(x, layer_state, cos_hf, sin_hf, cfg, eps, quantise=q)
        per_layer.append(x)

    norm_w = substate(state_dict, "model.norm")["weight"].float()
    if weight_dtype is not None:
        norm_w = quantize_like_device(norm_w.reshape(1, 1, -1, ttnn.TILE_SIZE), ttnn.bfloat16).reshape(hidden)
    last_hidden = ident(_torch_rms_norm(x, norm_w, eps), ttnn.bfloat16)

    lm_w = substate(state_dict, "lm_head")["weight"].float()
    if weight_dtype is not None:
        lm_w = quantize_like_device(lm_w.transpose(0, 1).unsqueeze(0).unsqueeze(0), weight_dtype)[0, 0].transpose(0, 1)
    logits = torch.nn.functional.linear(last_hidden, lm_w)
    if weight_dtype is not None:
        # LMHead.__call__ writes bf8_b (tt/lm_head.py).
        logits = quantize_like_device(logits.reshape(1, 1, seq, -1), ttnn.bfloat8_b).reshape(logits.shape)
    return last_hidden, logits, per_layer


# --------------------------------------------------------------------------------------
# Device side
# --------------------------------------------------------------------------------------
def _build_tt_model(mesh_device, objs, state_dict, num_layers, seq_len, *, weight_dtype=ttnn.bfloat8_b):
    return Model(
        mesh_device,
        objs["hf_config"],
        state_dict,
        mesh_config=objs["mesh_config"],
        ccl_manager=objs["ccl_manager"],
        max_seq_len=max(seq_len, 128),
        num_layers=num_layers,
        weight_dtype=weight_dtype,
        with_lm_head=True,
    )


def _to_host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _run_tt(tt_model, input_ids, *, skip_lm_head, collect_layers=False, get_last_token=-1, process_last=None):
    """Drive the public prefill surface once.

    Returns ``(host_output, per_layer_host, processed)``. ``process_last`` is the row index handed
    to ``Model.process_output_prefill`` on the **device** tensor — the public path a runtime uses,
    exercised here rather than re-implemented, so a bug in its host-side TP gather is caught by the
    gate instead of by P7.
    """
    tokens_embd, rot_mats, _ = tt_model.prepare_inputs_prefill(input_ids)
    per_layer = []
    cb = (lambda i, h: per_layer.append(_to_host(h))) if collect_layers else None
    out = tt_model.prefill_forward(
        tokens_embd,
        rot_mats_global=rot_mats,
        skip_lm_head=skip_lm_head,
        on_layer_complete=cb,
        get_last_token=get_last_token,
    )
    host = _to_host(out)
    processed = None if process_last is None else tt_model.process_output_prefill(out, process_last)
    out.deallocate(True)
    for t in rot_mats:
        t.deallocate(True)
    return host, per_layer, processed


def _token_ids(vocab_size, seq_len, *, seed=3):
    """Uniform random token ids. The input to a full model IS a token id; there is no other
    admissible distribution, so this cannot be tuned to pass (``E.1``)."""
    return torch.randint(0, vocab_size, (1, seq_len), generator=torch.Generator().manual_seed(seed))


def _top1(logits_bshv, pos):
    return int(torch.argmax(logits_bshv.reshape(-1, logits_bshv.shape[-1])[pos]))


# --------------------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("num_layers", [2, 4], ids=["L2", "L4"])
@pytest.mark.parametrize("seq_len", [128, 512], ids=["s128", "s512"])
@requires_hf_reference
@torch.no_grad()
def test_model_vs_hf_reduced_depth(mesh_device, state_dict, num_layers, seq_len, reset_seeds):
    """Reduced-depth stack vs HF fp32: hidden PCC >= 0.999 and 100% top-1 at the last position."""
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    cfg = attention_config_from_hf(hf_config, max_seq_len=max(seq_len, 128))
    input_ids = _token_ids(hf_config.vocab_size, seq_len)

    ref_hidden, ref_logits, _ = _hf_run(_hf_model(num_layers), input_ids)
    floor_hidden, floor_logits, _ = _torch_stack(
        state_dict, hf_config, cfg, input_ids, num_layers, weight_dtype=ttnn.bfloat8_b
    )

    tt_model = _build_tt_model(mesh_device, objs, state_dict, num_layers, seq_len)
    dev_hidden, _, _ = _run_tt(tt_model, input_ids, skip_lm_head=True)
    last = seq_len - 1
    dev_logits, _, dev_last_logits = _run_tt(tt_model, input_ids, skip_lm_head=False, process_last=last)

    assert dev_hidden.shape == (1, 1, seq_len, hf_config.hidden_size)
    passing, pcc = comp_pcc(ref_hidden, dev_hidden.reshape(1, seq_len, -1), PCC_THRESHOLD)
    _, floor_pcc = comp_pcc(ref_hidden, floor_hidden, 0.0)
    ratio = err_ratio(pcc, floor_pcc)

    ref_top1 = _top1(ref_logits, last)
    dev_top1 = int(torch.argmax(dev_last_logits))
    floor_top1 = _top1(floor_logits, last)
    _, logit_pcc = comp_pcc(ref_logits.reshape(1, seq_len, -1), dev_logits.reshape(1, seq_len, -1), 0.0)

    logger.info(
        f"[G-MODEL] n_layers={num_layers} seq_len={seq_len}: hidden PCC = {pcc} | "
        f"torch noise floor = {floor_pcc} | err ratio = {ratio:.2f}x | threshold {PCC_THRESHOLD} | "
        f"logits PCC = {logit_pcc} | top-1 HF={ref_top1} device={dev_top1} floor={floor_top1}"
    )
    assert passing, f"[G-MODEL] n_layers={num_layers} seq_len={seq_len} hidden PCC {pcc} < {PCC_THRESHOLD}"
    assert dev_top1 == ref_top1, (
        f"[G-MODEL] top-1 disagreement at the last position: HF {ref_top1}, device {dev_top1} "
        f"(the bf8_b noise floor picks {floor_top1}) — 100% agreement is the gate"
    )
    assert ratio <= MAX_ERR_RATIO, (
        f"[G-MODEL] n_layers={num_layers} seq_len={seq_len}: PCC {pcc} clears {PCC_THRESHOLD} but "
        f"sits {ratio:.1f}x off the torch noise floor {floor_pcc} — investigate (DEC-032)"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
@requires_hf_reference
@torch.no_grad()
def test_full_stack_per_layer_pcc_curve(mesh_device, state_dict, seq_len, reset_seeds):
    """All 32 layers vs HF, with the **per-layer** hidden-state PCC curve written to `bringup_log/raw/`.

    This is the one thing an aggregate PCC cannot do. Monotone decay across depth is accumulation
    and is expected; a **step** at one layer is a single-layer bug (a swapped weight, an off-by-one
    layer index, a stale cache entry for one tensor) and must be chased before P7
    (``BRINGUP_RECIPE.md:786-788``). The step criterion is the ratio of consecutive per-layer errors
    ``(1 - pcc_i) / (1 - pcc_{i-1})``, checked from layer ``STEP_CHECK_FROM_LAYER`` onward because
    the first layers climb off a near-exact baseline where a large ratio is noise.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    num_layers = hf_config.num_hidden_layers
    input_ids = _token_ids(hf_config.vocab_size, seq_len)
    cfg = attention_config_from_hf(hf_config, max_seq_len=max(seq_len, 128))

    ref_hidden, ref_logits, ref_layers = _hf_run(_hf_model(num_layers), input_ids)
    tt_model = _build_tt_model(mesh_device, objs, state_dict, num_layers, seq_len)
    dev_hidden, dev_layers, _ = _run_tt(tt_model, input_ids, skip_lm_head=True, collect_layers=True)
    last = seq_len - 1
    _dev_logits, _, dev_last_logits = _run_tt(tt_model, input_ids, skip_lm_head=False, process_last=last)

    assert len(dev_layers) == len(ref_layers) == num_layers

    rows, errs = [], []
    for i, (r, d) in enumerate(zip(ref_layers, dev_layers)):
        _, p = comp_pcc(r, d.reshape(1, seq_len, -1), 0.0)
        e = 1.0 - float(p)
        step = (e / errs[-1]) if errs else float("nan")
        rows.append((i, float(p), e, step))
        errs.append(max(e, 1e-30))

    _, final_pcc = comp_pcc(ref_hidden, dev_hidden.reshape(1, seq_len, -1), 0.0)
    floor_hidden, _floor_logits, _ = _torch_stack(
        state_dict, hf_config, cfg, input_ids, num_layers, weight_dtype=ttnn.bfloat8_b
    )
    _, floor_pcc = comp_pcc(ref_hidden, floor_hidden, 0.0)
    ratio = err_ratio(final_pcc, floor_pcc)

    ref_top1 = _top1(ref_logits, last)
    dev_top1 = int(torch.argmax(dev_last_logits))

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = _RAW_DIR / f"G-MODEL-CURVE_{ts}.log"
    lines = [
        "G-MODEL — per-layer hidden-state PCC curve",
        f"model      : Llama-3.1-8B-Instruct, {num_layers} layers, real weights",
        f"mesh       : {tuple(mesh_device.shape)}, TP={objs['mesh_config'].tp}, SP={objs['mesh_config'].sp}",
        f"seq_len    : {seq_len}; token ids uniform over vocab, seed 3",
        "reference  : transformers LlamaForCausalLM, fp32 weights + fp32 activations, eager attention",
        "device     : activations bf16, projections bf8_b, KV cache unused (no kv_cache passed)",
        "comparison : per-layer forward-hook output vs tt/layer.py output, same layer index",
        "",
        "| layer | PCC | error = 1-PCC | step vs previous |",
        "|---|---|---|---|",
    ]
    lines += [f"| {i} | {p:.7f} | {e:.3e} | {'—' if i == 0 else f'{s:.2f}x'} |" for i, p, e, s in rows]
    lines += [
        "",
        f"final hidden PCC (post final norm) : {float(final_pcc):.7f}",
        f"torch noise floor                  : {float(floor_pcc):.7f}",
        f"err ratio                          : {ratio:.2f}x",
        f"top-1 at position {last}                : HF {ref_top1} / device {dev_top1}",
        f"max step from layer {STEP_CHECK_FROM_LAYER} onward       : "
        f"{max((s for i, _p, _e, s in rows if i >= STEP_CHECK_FROM_LAYER), default=float('nan')):.2f}x "
        f"(threshold {MAX_LAYER_ERROR_STEP}x)",
        "",
    ]
    path.write_text("\n".join(lines))
    for line in lines:
        logger.info(f"[G-MODEL] {line}")

    offenders = [(i, s) for i, _p, _e, s in rows if i >= STEP_CHECK_FROM_LAYER and s > MAX_LAYER_ERROR_STEP]
    assert not offenders, (
        f"[G-MODEL] STEP in the per-layer error curve at {offenders} (threshold "
        f"{MAX_LAYER_ERROR_STEP}x). A step is a single-layer bug, not accumulation — chase it "
        f"before declaring PASS (BRINGUP_RECIPE.md:786-788). Curve: {path}"
    )
    assert float(final_pcc) >= PCC_THRESHOLD, f"[G-MODEL] 32-layer hidden PCC {final_pcc} < {PCC_THRESHOLD}"
    assert dev_top1 == ref_top1, f"[G-MODEL] 32-layer top-1 disagreement: HF {ref_top1}, device {dev_top1}"
    assert ratio <= MAX_ERR_RATIO, (
        f"[G-MODEL] 32-layer PCC {final_pcc} sits {ratio:.1f}x off the torch noise floor "
        f"{floor_pcc} — investigate (DEC-032)"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@requires_hf_reference
@torch.no_grad()
def test_rotated_layer_weights_fail(mesh_device, state_dict, reset_seeds):
    """Negative control: giving layer ``i`` layer ``i+1``'s weights must collapse the PCC.

    The realistic bug, not a synthetic one: ``substate(state_dict, f"model.layers.{i}")`` is built
    from an f-string in a loop (``tt/model.py``), so an off-by-one there — or a per-layer cache
    directory named one index off — silently gives every layer its neighbour's weights. Every shape,
    dtype and op is unchanged, so a high PCC here would mean the positive gates above are not
    actually testing that layer ``i`` got layer ``i``'s weights.

    Follows the ``tests/unit/test_rope_vs_ref.py:140`` pattern.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    num_layers, seq_len = 4, 128
    input_ids = _token_ids(hf_config.vocab_size, seq_len)
    ref_hidden, _, _ = _hf_run(_hf_model(num_layers), input_ids)

    rotated = {k: v for k, v in state_dict.items() if not k.startswith("model.layers.")}
    for i in range(num_layers):
        src = (i + 1) % num_layers
        for k, v in substate(state_dict, f"model.layers.{src}").items():
            rotated[f"model.layers.{i}.{k}"] = v

    good, _, _ = _run_tt(
        _build_tt_model(mesh_device, objs, state_dict, num_layers, seq_len), input_ids, skip_lm_head=True
    )
    bad, _, _ = _run_tt(_build_tt_model(mesh_device, objs, rotated, num_layers, seq_len), input_ids, skip_lm_head=True)

    _, pcc_ok = comp_pcc(ref_hidden, good.reshape(1, seq_len, -1), 0.0)
    _, pcc_bad = comp_pcc(ref_hidden, bad.reshape(1, seq_len, -1), 0.0)
    logger.info(f"[G-MODEL] negative control: correct layer order PCC = {pcc_ok}, rotated = {pcc_bad}")
    assert float(pcc_bad) < 0.99, (
        f"rotating the layer weights scored {pcc_bad}; the positive gates are not testing that "
        f"layer i was built from layer i's sub-dict"
    )


@pytest.mark.parametrize("mesh_device", [TestFactory.SINGLE_CARD_MESH_SHAPE], indirect=True)
@requires_hf_reference
@torch.no_grad()
def test_get_last_token_slice_matches_the_full_sequence(mesh_device, state_dict, reset_seeds):
    """``get_last_token`` must return exactly the corresponding rows of the full-sequence output.

    P7's runtime uses this path (it is the only reason ``prefill_forward`` slices at all), and a
    wrong slice offset produces logits for the wrong token — a bug that looks like a bad model,
    not like an indexing error. Compared bit-exactly: the slice is a copy, not a recomputation.
    """
    objs = TestFactory.setup_test(mesh_device)
    hf_config = objs["hf_config"]
    num_layers, seq_len = 2, 128
    input_ids = _token_ids(hf_config.vocab_size, seq_len)
    tt_model = _build_tt_model(mesh_device, objs, state_dict, num_layers, seq_len)

    full, _, _ = _run_tt(tt_model, input_ids, skip_lm_head=False)
    start = ((seq_len - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    tile, _, tile_last = _run_tt(
        tt_model, input_ids, skip_lm_head=False, get_last_token=start, process_last=(seq_len - 1) % ttnn.TILE_SIZE
    )

    assert tile.shape == (1, 1, ttnn.TILE_SIZE, full.shape[-1]), f"unexpected tile shape {tuple(tile.shape)}"
    torch.testing.assert_close(tile, full[:, :, start : start + ttnn.TILE_SIZE, :], rtol=0.0, atol=0.0)
    # And the row process_output_prefill picks out of the tile is the last token's row of the
    # full-sequence output: the (get_last_token, last_token_idx % 32) pair a runtime must use.
    torch.testing.assert_close(tile_last.reshape(-1), full[0, 0, seq_len - 1, :].reshape(-1), rtol=0.0, atol=0.0)
    logger.info(
        f"[G-MODEL] get_last_token={start} returns rows [{start}, {start + ttnn.TILE_SIZE}) of the "
        f"full-sequence logits bit-exactly, and process_output_prefill(tile, "
        f"{(seq_len - 1) % ttnn.TILE_SIZE}) is the last token's row"
    )


@requires_hf_reference
@torch.no_grad()
def test_hf_reference_is_causal():
    """The HF oracle must be causal. Guards Appendix F.2 before any PCC is believed.

    Changes only the **last** token id and asserts every earlier row of ``last_hidden_state`` is
    bit-identical. A non-causal reference (``attention_mask=None`` reaching a hand-written eager
    path) looks exactly like a model bug, and it would make every number in this file meaningless.
    Host-only.
    """
    model = _hf_model(2)
    ids = _token_ids(model.config.vocab_size, 128)
    other = ids.clone()
    other[0, -1] = (int(other[0, -1]) + 4242) % model.config.vocab_size

    a, _, _ = _hf_run(model, ids)
    b, _, _ = _hf_run(model, other)
    prefix_delta = (a[:, :-1] - b[:, :-1]).abs().max().item()
    last_delta = (a[:, -1] - b[:, -1]).abs().max().item()
    logger.info(
        f"[G-MODEL] HF causality probe: max|delta| on rows [:-1] = {prefix_delta:.3e}, "
        f"on the last row = {last_delta:.3e}"
    )
    assert prefix_delta == 0.0, (
        f"changing the LAST token moved earlier rows by {prefix_delta:.3e}; the HF reference is not "
        f"causal and every PCC in this file is meaningless (Appendix F.2)"
    )
    assert last_delta > 0.0, "changing the last token changed nothing; the probe is not exercising the model"


@requires_hf_reference
@torch.no_grad()
def test_in_test_torch_reference_agrees_with_hf(state_dict):
    """The composed in-test reference (`G-LAYER`'s / `G-ATTN`'s / `G-MLP`'s maths) reproduces HF.

    Two jobs. It validates the sublayer gates' oracle against the real model on real weights — the
    sublayer gates run on random weights, so this is the first time that maths meets the
    checkpoint. And it licenses :func:`_torch_stack` as this gate's **noise floor**: a floor is only
    meaningful if its fp32 limit is the reference.

    Threshold ``PCC >= 0.9999`` and not ``1.0``: the two implementations differ in fp32 summation
    order (HF's fused ``rotate_half`` and ``repeat_interleave`` vs the in-test ``einsum``-free
    form), which is worth ~1e-6, and both sides are genuinely fp32 so nothing dtype-related is being
    hidden. Host-only.
    """
    from models.demos.llama31_8b_d_p.tests.test_factory import llama_config_dims
    from models.demos.llama31_8b_d_p.tt.model_config import llama_hf_config

    hf_config = llama_hf_config(llama_config_dims())
    num_layers, seq_len = 2, 128
    cfg = attention_config_from_hf(hf_config, max_seq_len=seq_len)
    input_ids = _token_ids(hf_config.vocab_size, seq_len)

    ref_hidden, ref_logits, ref_layers = _hf_run(_hf_model(num_layers), input_ids)
    mine_hidden, mine_logits, mine_layers = _torch_stack(state_dict, hf_config, cfg, input_ids, num_layers)

    for i, (r, m) in enumerate(zip(ref_layers, mine_layers)):
        _, p = comp_pcc(r, m, 0.0)
        logger.info(f"[G-MODEL] in-test fp32 reference vs HF, layer {i}: PCC = {p}")
        assert float(p) >= 0.9999, f"the in-test fp32 layer reference disagrees with HF at layer {i}: PCC {p}"
    _, ph = comp_pcc(ref_hidden, mine_hidden, 0.0)
    _, pl = comp_pcc(ref_logits, mine_logits, 0.0)
    logger.info(f"[G-MODEL] in-test fp32 reference vs HF: hidden PCC = {ph}, logits PCC = {pl}")
    assert float(ph) >= 0.9999 and float(pl) >= 0.9999
