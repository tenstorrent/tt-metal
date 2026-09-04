# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""`tt/model.py` vs HF `LlamaForCausalLM` on the **real checkpoint**. Gate: `G-MODEL`.

`embedding -> [DecoderLayer] * n -> final norm -> lm_head`, `(1,1)` mesh, TP=1, no collective and
no KV-cache write (at TP=1 the packed cache refuses the model's 8 local KV heads outright —
`bringup_log/00_MODEL_CARD.md` §4.1; P8's `G-KV-TP8` owns the first real write).

**This is an integration check and it may not stand in for a sublayer gate**
(`BRINGUP_RECIPE.md:1514-1531`). What it adds over `G-LAYER` is the three things only the whole
stack can get wrong: the **embedding→layer→norm→head wiring**, the **depth accumulation** of error,
and — the one thing no PCC can substitute for — whether the model **predicts the same token** as
HuggingFace with real weights.

* **Input distribution:** real Llama-3.1-8B-Instruct token ids. A fixed English prompt is tokenized
  with the checkpoint's own tokenizer and tiled to the sequence length, so the activations are
  in-distribution for the real weights rather than the uniform-random ids a synthetic test would
  use. Recorded because the top-1 half of this gate is only meaningful on inputs the model was
  trained on. The weights are the **real** checkpoint, not random.
* **Reference dtype policy:** HF built **bare from the config** and `.float()`-ed, never through
  `from_pretrained`, which would load at the checkpoint's `torch_dtype` (bf16) and hand back a
  reference that shares the device's own rounding and reports a flattered PCC (recipe §2.1(a),
  `LANDMINES.md` "from_pretrained for a torch reference"). `_attn_implementation = "eager"`, and
  causality is asserted **directly on the device** rather than assumed (recipe P1 trap 3).
* **Computed noise floor:** the same HF model with **every weight replaced by the exact values the
  device holds** — each tensor pushed through the loader's transpose and Q/K Meta swizzle, quantised
  by the package's one quantiser, then mapped back to HF layout — and all remaining math in fp32.
  That is recipe §2.2's floor with nothing internal quantised.
* **Negative control:** the per-layer weights **rotated** (layer `i` gets layer `i+1`'s weights).
  The recipe measured **0.1612**.

**On the error-ratio budget (`R-015`, `DEC-051`).** Appendix A gives `G-MODEL` ">= 0.999, <= 8x the
floor, per-layer step <= 4x from L3, 100% top-1". Every layer of this model contains the fused SDPA
kernel whose slack accounted for the whole of `G-ATTN`'s block gap, so the raw 8x is treated exactly
as `DEC-042`/`DEC-051` treat it: recorded at both dtypes, asserted where §2.3.1 says a raw ratio is
meaningful, and *additionally* gated on the kernel-attributed residual. The **per-layer step** is
the tighter constraint and is asserted as stated — it is a ratio between two *measured* PCCs, so the
fused kernel's fixed slack is in both numerator and denominator and does not distort it.

Run:
    HF_MODEL=... pytest models/demos/llama31_8b_d_p/tests/unit/test_model_vs_ref.py -x -q
    HF_MODEL=... pytest models/demos/llama31_8b_d_p/tests/unit/test_model_vs_ref.py -q -k full_depth
"""

import json
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama31_8b_d_p.tests.test_factory import (
    bundled_config_path,
    err_ratio,
    hf_model_path,
    llama_config_dims,
    quantize_like_device,
    requires_hf_reference,
)
from models.demos.llama31_8b_d_p.tests.unit.test_decoder_layer_vs_ref import (
    _hf_cos_sin,
    _meta_head_index,
    _rms_norm,
    _to_device,
    _torch_layer,
)
from models.demos.llama31_8b_d_p.tt.attention.config import ProgramConfig
from models.demos.llama31_8b_d_p.tt.attention.prefill import run_sdpa
from models.demos.llama31_8b_d_p.tt.config import MeshConfig
from models.demos.llama31_8b_d_p.tt.layer import build_attention_config
from models.demos.llama31_8b_d_p.tt.model import Model
from models.demos.llama31_8b_d_p.tt.model_config import ModelArgs
from models.tt_transformers.tt.load_checkpoints import permute, reverse_permute

WEIGHT_DTYPE = ttnn.bfloat8_b  # `DEC-022`
ACTIVATION_DTYPE = ttnn.bfloat16  # `DEC-022`

PCC_THRESHOLD = 0.999  # `BRINGUP_RECIPE.md:1559-1561` (reduced depth), Appendix A `BRINGUP_RECIPE.md:2074`
MAX_MODEL_ERR_RATIO = 8.0  # `BRINGUP_RECIPE.md:1559-1561`
MAX_LAYER_STEP = 4.0  # `BRINGUP_RECIPE.md:1564`, "from layer 3 onward"
FIRST_GATED_STEP_LAYER = 3  # "from layer 3 onward", 0-based layer index

REDUCED_DEPTHS = [2, 4]
SEQ_LENS = [128, 512]
FULL_DEPTH_SEQ_LEN = 512

# A fixed English prompt: real tokens, so the top-1 comparison is a real next-token prediction
# rather than a coin flip on out-of-distribution input.
PROMPT = (
    "The capital of France is Paris. The capital of Italy is Rome. The capital of Japan is Tokyo. "
    "Large language models are trained on text and predict the next token in a sequence. "
)

_RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "bringup_log", "raw"
)

# Which loader transform each checkpoint key goes through, and at which dtype the device holds it.
# Same table as `test_weight_loading.py`'s `_WEIGHT_PLAN`, expressed per key *pattern* because this
# file walks all 32 layers.
_SWIZZLED = ("self_attn.q_proj.weight", "self_attn.k_proj.weight")
_TRANSPOSED = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "lm_head.weight",
)
_BF16_ON_DEVICE = ("layernorm.weight", "model.norm.weight", "model.embed_tokens.weight")


# ---------------------------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------------------------
def _prompt_tokens(seq_len):
    """The prompt, tokenized with the checkpoint's own tokenizer and tiled to `seq_len`."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path())
    ids = tokenizer(PROMPT, add_special_tokens=True)["input_ids"]
    tiled = (ids * (seq_len // len(ids) + 1))[:seq_len]
    return torch.tensor(tiled, dtype=torch.long).reshape(1, seq_len)


# ---------------------------------------------------------------------------------------------
# the HF oracle, and the floor built from the same class
# ---------------------------------------------------------------------------------------------
def _hf_config(n_layers):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(os.path.dirname(bundled_config_path()))
    cfg._attn_implementation = "eager"
    cfg.num_hidden_layers = n_layers
    return cfg


def _load_checkpoint(n_layers):
    """Only the tensors an `n_layers` model needs."""
    prefixes = ("model.embed_tokens.", "model.norm.", "lm_head.") + tuple(f"model.layers.{i}." for i in range(n_layers))
    return ModelArgs._load_safetensors(hf_model_path(), prefixes=prefixes)


def _device_valued(key, tensor, head_dim):
    """The same tensor holding **exactly the values the device holds**, back in HF layout.

    Push it through the loader's transforms (Q/K Meta swizzle, then the `[out,in] -> [in,out]`
    transpose), quantise with the package's one quantiser in *that* orientation — which matters,
    because `bfloat8_b`'s shared exponent is per 16-element block of the tilized layout — then undo
    the transforms exactly. Inverting a permutation and a transpose is lossless, so what comes back
    is the device's values expressed the way HF wants them.
    """
    dtype = ttnn.bfloat16 if any(p in key for p in _BF16_ON_DEVICE) else WEIGHT_DTYPE
    t = tensor
    swizzle = any(key.endswith(s) for s in _SWIZZLED)
    transpose = any(key.endswith(s) for s in _TRANSPOSED)
    if swizzle:
        t = reverse_permute(t, t.shape[0] // head_dim, t.shape[0], t.shape[1])
    if transpose:
        t = t.transpose(-1, -2)
    if "layernorm.weight" in key or key == "model.norm.weight":
        q = quantize_like_device(t.reshape(1, 1, -1, ttnn.TILE_SIZE), dtype).reshape(-1)
    else:
        q = quantize_like_device(t.unsqueeze(0).unsqueeze(0), dtype)[0, 0]
    if transpose:
        q = q.transpose(-1, -2).contiguous()
    if swizzle:
        q = permute(q, q.shape[0] // head_dim, q.shape[0], q.shape[1])
    assert q.shape == tensor.shape, f"{key}: round trip changed the shape {tuple(tensor.shape)} -> {tuple(q.shape)}"
    return q


def _build_hf_model(n_layers, state_dict, *, quantise=False, rotate_layers=False):
    """An fp32 HF `LlamaForCausalLM`. `quantise` builds the noise floor; `rotate_layers` the control."""
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    cfg = _hf_config(n_layers)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    model = LlamaForCausalLM(cfg).float().eval()

    weights = {}
    for key, tensor in state_dict.items():
        source = key
        if rotate_layers and key.startswith("model.layers."):
            idx = int(key.split(".")[2])
            source = key.replace(f"model.layers.{idx}.", f"model.layers.{(idx + 1) % n_layers}.")
        t = state_dict[source]
        weights[key] = (_device_valued(source, t, head_dim) if quantise else t).float()

    missing, unexpected = model.load_state_dict(weights, strict=False)
    assert not [k for k in missing if "rotary" not in k], f"HF reference is missing weights: {missing}"
    assert not unexpected, f"HF reference rejected keys: {unexpected}"

    if quantise:
        # The RoPE tables are an **input** the device stores, and it stores them in **bf16**
        # (`tt/rope.py::build_prefill_rope` builds bf16 and the helper takes no dtype argument), so
        # §2.2's "round inputs and weights to the device dtype" covers them. HF computes them
        # internally in fp32, which made an earlier version of this floor omit one rounding the
        # device actually pays — worth **45%** of the correct floor error at 32 layers, and it moved
        # the reported full-depth ratio from 1.53x to 2.79x (`DEC-053`, `R-021`). `G-LAYER`'s floor
        # already quantised cos/sin; this makes the two gates agree on what a floor is.
        def _quantise_rope(_module, _inputs, output):
            cos, sin = output
            return (
                quantize_like_device(cos[None], ttnn.bfloat16)[0],
                quantize_like_device(sin[None], ttnn.bfloat16)[0],
            )

        model.model.rotary_emb.register_forward_hook(_quantise_rope)
    return model


@torch.no_grad()
def _run_hf(model, tokens):
    """Run the HF oracle and return `(per_layer_prenorm, post_norm_hidden, last_position_logits)`.

    **Read via explicit forward hooks, not `output_hidden_states`.** On `transformers` 5.12.1 that
    tuple is `(embeddings, L0, ..., L[n-2], POST-FINAL-NORM)`: its **last** element is the
    post-`model.norm` stream, *not* the last layer's output, so the last decoder layer's pre-norm
    output is not in it at all — and `CausalLMOutputWithPast` exposes no `last_hidden_state` to
    disambiguate (`LlamaModel.forward` builds one, `LlamaForCausalLM` drops it:
    `python_env/lib/python3.12/site-packages/transformers/models/llama/modeling_llama.py:421`,
    `:484`). Comparing the device's pre-norm layer output against that last element measures a
    **double norm** and reads as a plausible-but-wrong PCC — it cost this session a debugging pass,
    so the ambiguous route is not used anywhere in this file (`DEC-052`).
    """
    captured = {}
    hooks = [
        layer.register_forward_hook(lambda _m, _i, out, idx=idx: captured.__setitem__(idx, out.detach().float()))
        for idx, layer in enumerate(model.model.layers)
    ]
    hooks.append(
        model.model.norm.register_forward_hook(lambda _m, _i, out: captured.__setitem__("norm", out.detach().float()))
    )
    try:
        out = model(input_ids=tokens, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    n_layers = len(model.model.layers)
    layers = [captured[i] for i in range(n_layers)]
    return layers, captured["norm"], out.logits[0, -1].float()


# ---------------------------------------------------------------------------------------------
# the device model
# ---------------------------------------------------------------------------------------------
def _build_tt_model(mesh_device, hf, state_dict, n_layers, seq_len, *, rotate_layers=False):
    if rotate_layers:
        rotated = {}
        for key, tensor in state_dict.items():
            if key.startswith("model.layers."):
                idx = int(key.split(".")[2])
                rotated[key] = state_dict[key.replace(f"model.layers.{idx}.", f"model.layers.{(idx + 1) % n_layers}.")]
            else:
                rotated[key] = tensor
        state_dict = rotated
    return Model(
        mesh_device,
        hf,
        state_dict,
        mesh_config=MeshConfig(tuple(mesh_device.shape), tp=mesh_device.shape[1]),
        weight_dtype=WEIGHT_DTYPE,
        activation_dtype=ACTIVATION_DTYPE,
        max_seq_len=max(seq_len, ttnn.TILE_SIZE),
        n_layers=n_layers,
        with_lm_head=True,
    )


def _from_device(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def _tt_hidden_states(model, tokens, *, on_layer_output=None):
    """Post-final-norm hidden states, `[1, 1, S, hidden]` — HF's `last_hidden_state` (`DEC-049`)."""
    x, rope_mats, _ = model.prepare_inputs_prefill(tokens)
    out = model.prefill_forward(x, rope_mats, skip_lm_head=True, on_layer_output=on_layer_output)
    hidden = _from_device(out)
    out.deallocate(True)
    return hidden


def _tt_last_logits(model, tokens):
    """The last position's logits, `[vocab]`, through the real `get_last_token` slice path."""
    seq_len = tokens.shape[-1]
    x, rope_mats, _ = model.prepare_inputs_prefill(tokens)
    out = model.prefill_forward(x, rope_mats, get_last_token=seq_len - ttnn.TILE_SIZE)
    row = model.process_output_prefill(out, ttnn.TILE_SIZE - 1)
    out.deallocate(True)
    return row.reshape(-1).float()


# ---------------------------------------------------------------------------------------------
# G-MODEL, reduced depth
# ---------------------------------------------------------------------------------------------
@requires_hf_reference
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("n_layers", REDUCED_DEPTHS, ids=lambda n: f"L{n}")
@pytest.mark.parametrize("seq_len", SEQ_LENS, ids=lambda s: f"s{s}")
def test_model_vs_ref(mesh_device, n_layers, seq_len, reset_seeds):
    """Hidden-state PCC >= 0.999, ratio to the computed floor recorded, and top-1 == HF."""
    hf = llama_config_dims()
    tokens = _prompt_tokens(seq_len)
    state_dict = _load_checkpoint(n_layers)

    ref_model = _build_hf_model(n_layers, state_dict)
    _, ref_post_norm, ref_logits = _run_hf(ref_model, tokens)
    # The device returns the POST-final-norm stream (`DEC-049`), which is what `LlamaModel` calls
    # `last_hidden_state`.
    ref_last_hidden = ref_post_norm.reshape(1, 1, seq_len, -1)
    del ref_model

    floor_model = _build_hf_model(n_layers, state_dict, quantise=True)
    _, floor_post_norm, _ = _run_hf(floor_model, tokens)
    floor_last_hidden = floor_post_norm.reshape(1, 1, seq_len, -1)
    del floor_model

    _, floor = comp_pcc(ref_last_hidden, floor_last_hidden, 0.0)
    floor = float(floor)

    tt_model = _build_tt_model(mesh_device, hf, state_dict, n_layers, seq_len)
    tt_hidden = _tt_hidden_states(tt_model, tokens)
    tt_logits = _tt_last_logits(tt_model, tokens)

    passing, pcc = comp_pcc(ref_last_hidden, tt_hidden, PCC_THRESHOLD)
    pcc = float(pcc)
    ratio = err_ratio(pcc, floor)

    top1_ref = int(ref_logits.argmax())
    top1_dev = int(tt_logits.argmax())
    top5_ref = ref_logits.topk(5).indices.tolist()
    top5_dev = tt_logits.topk(5).indices.tolist()
    gap_ref = (ref_logits.topk(2).values[0] - ref_logits.topk(2).values[1]).item()
    _, logit_pcc = comp_pcc(ref_logits.reshape(1, 1, 1, -1), tt_logits.reshape(1, 1, 1, -1), 0.0)

    logger.info(
        f"[G-MODEL] L{n_layers} s{seq_len}: hidden PCC={pcc:.7f} floor={floor:.7f} ratio={ratio:.2f}x "
        f"(threshold {PCC_THRESHOLD}, raw budget {MAX_MODEL_ERR_RATIO}x)"
    )
    logger.info(
        f"[G-MODEL] L{n_layers} s{seq_len}: last-position logits PCC={float(logit_pcc):.7f}; "
        f"top-1 ref={top1_ref} dev={top1_dev} (ref top-2 logit gap {gap_ref:.4f}); "
        f"top-5 ref={top5_ref} dev={top5_dev}"
    )

    assert passing, f"hidden-state PCC below {PCC_THRESHOLD}: {pcc}"
    assert top1_dev == top1_ref, (
        f"top-1 disagrees with HF at the last position: ref {top1_ref}, device {top1_dev} "
        f"(ref top-5 {top5_ref}, device top-5 {top5_dev}, ref top-2 gap {gap_ref:.4f})"
    )
    assert ratio <= MAX_MODEL_ERR_RATIO, (
        f"the model is {ratio:.2f}x off its floor. Every layer contains the fused SDPA kernel "
        f"(R-015); attribute the gap to it before recording a PASS"
    )


@requires_hf_reference
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_model_is_causal(mesh_device, reset_seeds):
    """**Self-check:** perturb the last token; every earlier row must be unchanged at `max|delta| = 0`.

    Required by the recipe when HF is the oracle (`BRINGUP_RECIPE.md:1568-1571`) and asserted on the
    **device** stack, which is the only place it proves that `is_causal=True` actually reached the
    kernel in all layers.
    """
    hf = llama_config_dims()
    n_layers, seq_len = 2, 128
    tokens = _prompt_tokens(seq_len)
    perturbed = tokens.clone()
    perturbed[0, -1] = (int(tokens[0, -1]) + 1) % hf["vocab_size"]

    tt_model = _build_tt_model(mesh_device, hf, _load_checkpoint(n_layers), n_layers, seq_len)
    base = _tt_hidden_states(tt_model, tokens)
    after = _tt_hidden_states(tt_model, perturbed)

    delta_prefix = (base[:, :, :-1] - after[:, :, :-1]).abs().max().item()
    delta_last = (base[:, :, -1] - after[:, :, -1]).abs().max().item()
    logger.info(f"[G-MODEL] causality: max|delta| rows[:-1]={delta_prefix:.3e}, last row={delta_last:.3e}")
    assert delta_prefix == 0.0, f"changing the last token moved earlier rows by {delta_prefix:.3e} — not causal"
    assert delta_last > 0.0, "changing the last token moved nothing — the probe is blind"


@requires_hf_reference
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_model_rotated_layer_weights_negative_control(mesh_device, reset_seeds):
    """**The negative control:** layer `i` gets layer `i+1`'s weights. The recipe measured 0.1612."""
    hf = llama_config_dims()
    n_layers, seq_len = 4, 128
    tokens = _prompt_tokens(seq_len)
    state_dict = _load_checkpoint(n_layers)

    ref_model = _build_hf_model(n_layers, state_dict)
    _, ref_post_norm, ref_logits = _run_hf(ref_model, tokens)
    ref_last_hidden = ref_post_norm.reshape(1, 1, seq_len, -1)
    del ref_model

    tt_model = _build_tt_model(mesh_device, hf, state_dict, n_layers, seq_len, rotate_layers=True)
    tt_hidden = _tt_hidden_states(tt_model, tokens)
    tt_logits = _tt_last_logits(tt_model, tokens)
    _, pcc = comp_pcc(ref_last_hidden, tt_hidden, 0.0)

    top1_ref, top1_dev = int(ref_logits.argmax()), int(tt_logits.argmax())
    logger.info(
        f"[G-MODEL] control: per-layer weights rotated -> hidden PCC={float(pcc):.5f}; "
        f"top-1 {top1_ref} -> {top1_dev}"
    )
    assert float(pcc) < PCC_THRESHOLD, f"rotating the per-layer weights still scores {float(pcc)}"
    assert top1_dev != top1_ref, "rotating the per-layer weights left top-1 unchanged — the top-1 check is blind"


@requires_hf_reference
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_hf_and_quantised_floor_differ(mesh_device, reset_seeds):
    """**Floor sanity:** the floor model must actually be a floor — i.e. quantised weights must
    change the answer, and change it *less* than the device does.

    Without this, a floor accidentally built from unquantised weights would read as PCC 1.0 and make
    every error ratio infinite (or, worse, make a broken module look fine).
    """
    hf = llama_config_dims()
    n_layers, seq_len = 2, 128
    tokens = _prompt_tokens(seq_len)
    state_dict = _load_checkpoint(n_layers)

    ref_model = _build_hf_model(n_layers, state_dict)
    ref_hidden = _run_hf(ref_model, tokens)[1].reshape(1, 1, seq_len, -1)
    del ref_model
    floor_model = _build_hf_model(n_layers, state_dict, quantise=True)
    floor_hidden = _run_hf(floor_model, tokens)[1].reshape(1, 1, seq_len, -1)
    del floor_model

    _, floor = comp_pcc(ref_hidden, floor_hidden, 0.0)
    logger.info(
        f"[G-MODEL] floor sanity: fp32 vs device-valued weights PCC={float(floor):.7f} "
        f"(1 - floor = {1 - float(floor):.3e}); identical tensors would give exactly 1.0"
    )
    assert float(floor) < 1.0, "the floor model is bit-identical to the reference — the quantiser did nothing"
    assert float(floor) > PCC_THRESHOLD, f"the floor itself is below the gate threshold ({floor})"


# ---------------------------------------------------------------------------------------------
# G-MODEL, full depth: the per-layer PCC curve and its step
# ---------------------------------------------------------------------------------------------
@requires_hf_reference
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_model_full_depth_per_layer_curve(mesh_device, reset_seeds):
    """All 32 layers: the per-layer PCC curve against HF, the same curve against the computed floor,
    the error step, and top-1.

    **What the recipe gates here, precisely.** `BRINGUP_RECIPE.md:1559-1561` attaches
    "hidden-state PCC >= 0.999, <= 8x the floor, top-1 = 100%" to the **reduced layer count** runs;
    `BRINGUP_RECIPE.md:1562-1565` states the full 32-layer run's own gate as "record the per-layer hidden-state PCC
    curve ... and gate the **step** between consecutive layers at **<= 4x** from layer 3 onward".
    Appendix A's one-line row (`BRINGUP_RECIPE.md:2074`) compresses both into one cell, which reads as if the
    absolute threshold also applied at depth 32. This test asserts the phase text's version — the
    step, plus top-1 and the ratio to a **measured** full-depth floor — and records the absolute
    number either way. `DEC-053` carries the reading and the measurement behind it.

    A monotone decay is normal; a **step** at one layer is a bug and must be chased before P7.
    The step is a ratio of two *measured* PCCs, so the fused SDPA kernel's fixed slack is in both
    numerator and denominator and does not distort it (`R-015`).

    The curve is written to `bringup_log/raw/G-MODEL_per_layer_pcc.json`.
    """
    hf = llama_config_dims()
    n_layers = hf["num_hidden_layers"]
    seq_len = FULL_DEPTH_SEQ_LEN
    tokens = _prompt_tokens(seq_len)
    state_dict = _load_checkpoint(n_layers)

    ref_model = _build_hf_model(n_layers, state_dict)
    # `ref_layers[i]` is layer `i`'s output, PRE the final norm — exactly what the device's
    # `on_layer_output` seam hands back. Read by hook, for the reason `_run_hf` documents.
    ref_layers, ref_post_norm, ref_logits = _run_hf(ref_model, tokens)
    ref_layers = [h.reshape(1, 1, seq_len, -1) for h in ref_layers]
    ref_last_hidden = ref_post_norm.reshape(1, 1, seq_len, -1)
    del ref_model

    # The floor at full depth: the same HF class with every weight replaced by the exact values the
    # device holds, all remaining math in fp32 (recipe §2.2). Measured rather than extrapolated from
    # the 2-layer floor, because error accumulation over depth is what is in question.
    floor_model = _build_hf_model(n_layers, state_dict, quantise=True)
    floor_layers, floor_post_norm, _ = _run_hf(floor_model, tokens)
    floor_layers = [h.reshape(1, 1, seq_len, -1) for h in floor_layers]
    floor_last_hidden = floor_post_norm.reshape(1, 1, seq_len, -1)
    del floor_model

    tt_model = _build_tt_model(mesh_device, hf, state_dict, n_layers, seq_len)

    curve = {}

    def _capture(layer_idx, hidden_states):
        _, pcc = comp_pcc(ref_layers[layer_idx], _from_device(hidden_states), 0.0)
        curve[layer_idx] = float(pcc)

    tt_hidden = _tt_hidden_states(tt_model, tokens, on_layer_output=_capture)
    tt_logits = _tt_last_logits(tt_model, tokens)

    floor_curve = {}
    for layer_idx in range(n_layers):
        _, pcc = comp_pcc(ref_layers[layer_idx], floor_layers[layer_idx], 0.0)
        floor_curve[layer_idx] = float(pcc)

    _, final_pcc = comp_pcc(ref_last_hidden, tt_hidden, 0.0)
    _, final_floor = comp_pcc(ref_last_hidden, floor_last_hidden, 0.0)
    final_pcc, final_floor = float(final_pcc), float(final_floor)
    final_ratio = err_ratio(final_pcc, final_floor)

    steps = {}
    for layer_idx in range(1, n_layers):
        prev_err = 1.0 - curve[layer_idx - 1]
        steps[layer_idx] = float("inf") if prev_err <= 0 else (1.0 - curve[layer_idx]) / prev_err

    for layer_idx in range(n_layers):
        step = steps.get(layer_idx)
        logger.info(
            f"[G-MODEL] full-depth curve L{layer_idx:>2}: PCC={curve[layer_idx]:.7f} "
            f"(1-PCC={1 - curve[layer_idx]:.3e})  floor={floor_curve[layer_idx]:.7f} "
            f"ratio={err_ratio(curve[layer_idx], floor_curve[layer_idx]):.2f}x"
            + (f"  step={step:.2f}x" if step is not None else "")
        )

    top1_ref, top1_dev = int(ref_logits.argmax()), int(tt_logits.argmax())
    _, logit_pcc = comp_pcc(ref_logits.reshape(1, 1, 1, -1), tt_logits.reshape(1, 1, 1, -1), 0.0)
    gated = {k: v for k, v in steps.items() if k >= FIRST_GATED_STEP_LAYER}
    worst_layer = max(gated, key=gated.get)
    logger.info(
        f"[G-MODEL] full-depth s{seq_len}: post-norm hidden PCC={final_pcc:.7f} "
        f"floor={final_floor:.7f} ratio={final_ratio:.2f}x (budget {MAX_MODEL_ERR_RATIO}x); "
        f"last pre-norm layer L{n_layers - 1} PCC={curve[n_layers - 1]:.7f} "
        f"floor={floor_curve[n_layers - 1]:.7f} "
        f"ratio={err_ratio(curve[n_layers - 1], floor_curve[n_layers - 1]):.2f}x"
    )
    logger.info(
        f"[G-MODEL] full-depth s{seq_len}: last-position logits PCC={float(logit_pcc):.7f}; "
        f"top-1 ref={top1_ref} dev={top1_dev}; worst gated step L{worst_layer} = "
        f"{gated[worst_layer]:.2f}x (budget {MAX_LAYER_STEP}x); steps L1-L2 (ungated) = "
        f"{steps[1]:.2f}x / {steps[2]:.2f}x"
    )

    os.makedirs(_RAW_DIR, exist_ok=True)
    curve_path = os.path.join(_RAW_DIR, "G-MODEL_per_layer_pcc.json")
    with open(curve_path, "w") as f:
        json.dump(
            {
                "seq_len": seq_len,
                "n_layers": n_layers,
                "weight_dtype": "bfloat8_b",
                "activation_dtype": "bfloat16",
                "reference": "HF LlamaForCausalLM, built bare from config.json and .float()-ed, eager attention",
                "floor": "the same HF class with every weight replaced by the values the device holds, fp32 math",
                "per_layer_pcc": {str(k): v for k, v in sorted(curve.items())},
                "per_layer_floor": {str(k): v for k, v in sorted(floor_curve.items())},
                "per_layer_ratio": {str(k): err_ratio(curve[k], floor_curve[k]) for k in sorted(curve)},
                "per_layer_step": {str(k): v for k, v in sorted(steps.items())},
                "final_post_norm_pcc": final_pcc,
                "final_post_norm_floor": final_floor,
                "final_post_norm_ratio": final_ratio,
                "last_position_logits_pcc": float(logit_pcc),
                "top1_ref": top1_ref,
                "top1_device": top1_dev,
            },
            f,
            indent=2,
        )
    logger.info(f"[G-MODEL] per-layer curve written to {curve_path}")

    assert top1_dev == top1_ref, f"full-depth top-1 disagrees with HF: ref {top1_ref}, device {top1_dev}"
    assert final_ratio <= MAX_MODEL_ERR_RATIO, (
        f"at full depth the model is {final_ratio:.2f}x off its measured floor (PCC {final_pcc:.7f} "
        f"vs floor {final_floor:.7f}) — that is this package's own error, not depth arithmetic"
    )
    for layer_idx, step in gated.items():
        assert step <= MAX_LAYER_STEP, (
            f"the per-layer error step at L{layer_idx} is {step:.2f}x (budget {MAX_LAYER_STEP}x): a "
            f"step in the curve is one sublayer's logic error, not depth accumulation — chase it "
            f"with LLAMA_DELTA_PROBE before P7"
        )


# ---------------------------------------------------------------------------------------------
# HUMAN GATE H5: account for the full-depth error ratio, do not merely note it
# ---------------------------------------------------------------------------------------------
@requires_hf_reference
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_model_full_depth_attribution(mesh_device, reset_seeds):
    """§2.3.1's attributed residual **at model scale**, which is what stays comparable across depths.

    `BRINGUP_RECIPE.md:1559-1564` gates the full-depth run on the per-layer step; the raw ratio at
    32 layers measures **2.79x** against every other gate in this run landing between 1.0x and
    2.4x, and `HUMAN GATE H5` (`BRINGUP_RECIPE.md:194`) says a ratio that clears its budget but sits off the floor
    with no identified cause is a decision, not a clean PASS. §2.3.1 as amended (`:479-487`) asks
    for the attributed residual to be recorded at layer **and** model level for exactly this reason.

    Three chains, all scored against the same fp32 HF reference, all at the gate's own
    configuration (32 layers, seq 512, real prompt tokens, bf8_b weights, bf16 activations):

    * **floor** — this file's own staged layer chain (bit-exact against HF `LlamaDecoderLayer` at
      `G-LAYER`) with every weight replaced by the values the device holds and all math in fp32;
    * **predicted** — the same chain with the **device's real fused SDPA** substituted at every one
      of the 32 layers, so the kernel's error propagates through the rest of each layer and
      accumulates over depth exactly as the arithmetic makes it;
    * **measured** — the device.

    Then `residual = ((1 - measured) - kernel_excess) / (1 - floor)` where
    `kernel_excess = (1 - predicted) - (1 - floor)`.

    The chain is also run with **fp32** cos/sin as well as the bf16 tables the device stores, because
    the gate's HF-built floor computes RoPE internally in fp32 and therefore omits one rounding the
    device pays. That difference is quantified here rather than left as a caveat.
    """
    hf = llama_config_dims()
    n_layers = hf["num_hidden_layers"]
    seq_len = FULL_DEPTH_SEQ_LEN
    head_dim = hf["hidden_size"] // hf["num_attention_heads"]
    tokens = _prompt_tokens(seq_len)
    state_dict = _load_checkpoint(n_layers)

    ref_model = _build_hf_model(n_layers, state_dict)
    _, ref_post_norm, _ = _run_hf(ref_model, tokens)
    ref = ref_post_norm.reshape(1, 1, seq_len, -1)
    del ref_model

    # --- the device measurement, at the same configuration ------------------------------------
    tt_model = _build_tt_model(mesh_device, hf, state_dict, n_layers, seq_len)
    measured_out = _tt_hidden_states(tt_model, tokens)
    _, measured = comp_pcc(ref, measured_out, 0.0)
    measured = float(measured)

    # --- the two host chains, advanced in lockstep so each layer's weights are quantised once ---
    table = _device_valued("model.embed_tokens.weight", state_dict["model.embed_tokens.weight"], head_dim).float()
    x0 = table[tokens[0]].reshape(1, 1, seq_len, -1).contiguous()
    del table
    cos_fp32, sin_fp32 = _hf_cos_sin(hf, seq_len)
    cos_bf16 = quantize_like_device(cos_fp32[None, None], ttnn.bfloat16)[0, 0]
    sin_bf16 = quantize_like_device(sin_fp32[None, None], ttnn.bfloat16)[0, 0]
    src = _meta_head_index(head_dim)

    attention_config = build_attention_config(hf, max_seq_len=seq_len)
    program_config = ProgramConfig()

    def _device_sdpa(q_rot, k_rot, v):
        """The device's fused kernel on this layer's own post-RoPE tensors, in Meta head space."""
        tt_out = run_sdpa(
            _to_device(q_rot[..., src], mesh_device),
            _to_device(k_rot[..., src], mesh_device),
            _to_device(v, mesh_device),
            attention_config,
            program_config,
            mesh_device,
            seq_len,
        )
        out = _from_device(tt_out)
        tt_out.deallocate(True)
        return out

    def _run_chains(cos, sin):
        """Return `(floor_out, predicted_out)` — the same chain with and without the device kernel."""
        x_floor, x_pred = x0.clone(), x0.clone()
        for layer_idx in range(n_layers):
            prefix = f"model.layers.{layer_idx}."
            w = {
                name: _device_valued(prefix + key, state_dict[prefix + key], head_dim).float()
                for name, key in (
                    ("q_proj", "self_attn.q_proj.weight"),
                    ("k_proj", "self_attn.k_proj.weight"),
                    ("v_proj", "self_attn.v_proj.weight"),
                    ("o_proj", "self_attn.o_proj.weight"),
                    ("gate_proj", "mlp.gate_proj.weight"),
                    ("up_proj", "mlp.up_proj.weight"),
                    ("down_proj", "mlp.down_proj.weight"),
                    ("input_layernorm", "input_layernorm.weight"),
                    ("post_attention_layernorm", "post_attention_layernorm.weight"),
                )
            }
            x_floor = _torch_layer(x_floor, w, cos, sin)["out"]
            x_pred = _torch_layer(x_pred, w, cos, sin, sdpa_fn=_device_sdpa)["out"]
            del w
        gain = _device_valued("model.norm.weight", state_dict["model.norm.weight"], head_dim).float()
        return _rms_norm(x_floor, gain), _rms_norm(x_pred, gain)

    floor_bf16_out, pred_bf16_out = _run_chains(cos_bf16, sin_bf16)
    _, floor_bf16 = comp_pcc(ref, floor_bf16_out, 0.0)
    _, pred_bf16 = comp_pcc(ref, pred_bf16_out, 0.0)
    floor_bf16, pred_bf16 = float(floor_bf16), float(pred_bf16)

    floor_fp32_out, _ = _run_chains(cos_fp32, sin_fp32)
    _, floor_fp32 = comp_pcc(ref, floor_fp32_out, 0.0)
    floor_fp32 = float(floor_fp32)

    floor_err = 1.0 - floor_bf16
    kernel_excess = (1.0 - pred_bf16) - floor_err
    residual_ratio = ((1.0 - measured) - kernel_excess) / floor_err
    raw_ratio = err_ratio(measured, floor_bf16)

    logger.info(
        f"[G-MODEL] H5 attribution, 32 layers, s{seq_len}, bf8_b weights / bf16 activations, real "
        f"prompt tokens: measured={measured:.7f} floor(bf16 rope)={floor_bf16:.7f} "
        f"floor(fp32 rope)={floor_fp32:.7f} predicted={pred_bf16:.7f}"
    )
    logger.info(
        f"[G-MODEL] H5 attribution: floor error={floor_err:.3e}; fused-SDPA excess over 32 layers="
        f"{kernel_excess:.3e} ({kernel_excess / floor_err:.2f}x the floor error, "
        f"{100 * kernel_excess / (1 - measured):.1f}% of the total); raw ratio={raw_ratio:.2f}x; "
        f"**SDPA-attributed residual={residual_ratio:.2f}x**"
    )
    logger.info(
        f"[G-MODEL] H5 chain cross-check: the staged chain's floor is {floor_bf16:.7f} with bf16 "
        f"RoPE tables and {floor_fp32:.7f} with fp32 ones; the gate's HF-built floor computes RoPE "
        f"in fp32 internally and measured 0.9994570, so the fp32-table chain is the comparable one"
    )

    if residual_ratio < 1.0:
        # Recipe §2.3: "any ratio below 1.0 is a broken floor, not a kernel beating arithmetic."
        # Here it is the *attribution* that is broken, not the floor: substituting the device kernel
        # into an otherwise-fp32 chain gives predicted={pred} which is WORSE than the device itself,
        # so the subtraction removes more than the kernel really contributes. Recorded, not
        # asserted — the number that gates is the raw ratio against the corrected floor.
        logger.warning(
            f"[G-MODEL] H5: the SDPA-attributed residual is {residual_ratio:.2f}x, i.e. BELOW 1.0. "
            f"The substituted chain scores {pred_bf16:.7f}, worse than the device's {measured:.7f}, "
            f"so §2.3.1's additive attribution OVER-subtracts at 32-layer depth (07_RISKS.md R-021). "
            f"It is a limit of the additive model, not evidence about this package's code."
        )

    assert floor_bf16 < 1.0 and pred_bf16 < 1.0, "a chain returned a perfect score — it is not doing the arithmetic"
    assert kernel_excess > 0, (
        f"substituting the device's fused SDPA made the chain MORE accurate ({kernel_excess:.3e}), "
        f"which means the substitution is not wired to the kernel"
    )
    assert raw_ratio <= MAX_MODEL_ERR_RATIO, (
        f"against a floor that includes every rounding the device pays, the model is {raw_ratio:.2f}x "
        f"off it at 32 layers — that is this package's own error"
    )
    assert abs(floor_fp32 - 0.9994570) < 1e-6, (
        f"the staged chain with fp32 RoPE tables scores {floor_fp32:.7f}, which should reproduce the "
        f"gate's HF-built floor of 0.9994570 — if it does not, the chain is not the same arithmetic "
        f"and none of the attribution above is comparable with the gate"
    )
