#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1: the reference against a REAL checkpoint loaded on CPU — ground truth for the whole model, not
just self-consistency. Pattern: ``minimax_m3/tests/golden_hf_first_token.py``.

What "ground truth" means for THIS model. M3 vendors its own reference, so its equivalent test
checks a hand-written implementation against upstream. Here the reference IS upstream (transformers'
``ministral3``, imported not vendored), so comparing it to ``from_pretrained`` would compare upstream
to itself. The thing that is genuinely ours — and genuinely able to be wrong — is the LOADER: the
wrapper-prefix mapping, the per-tensor fp8 dequantization, the dropped vision keys, the dropped
``input_scale`` keys, and the Meta swizzle. So that is what this checks, on real files:

  1. every weight the production loader produces, against the checkpoint's pre-quantization bf16
     ground truth (``reference_bf16.pt``, written by ``scripts/make_synthetic_checkpoint.py``);
  2. a full reference forward on the LOADED weights against the same forward on the ground-truth
     weights — so a per-tensor scale that is off by a constant shows up as a logits mismatch;
  3. the first token: ``argmax`` of the last position must agree, which is the sign-off the M3 test
     is named for;
  4. where transformers can load the checkpoint itself, our loaded weights against ITS state dict.

Point at any checkpoint directory with ``HF_MODEL``. With the published 128 B checkpoint this is the
real ground-truth test; with a synthetic one (same on-disk format, random weights) it validates
every line of the loader, which is what it is for. It SKIPS, loudly, when neither is available.

Run:
    HF_MODEL=/path/to/checkpoint python -m pytest models/demos/mistral_3_5_d_p/tests/golden_hf_first_token.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from loguru import logger

from models.demos.mistral_3_5_d_p.reference import model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.tt.model_config import ModelArgs

pytestmark = pytest.mark.skipif(
    not (os.getenv("HF_MODEL") and Path(os.getenv("HF_MODEL", "")).is_dir()),
    reason=(
        "set HF_MODEL to a Mistral-Medium-3.5 checkpoint dir. No checkpoint on this host? build a "
        "format-identical synthetic one: python models/demos/mistral_3_5_d_p/scripts/"
        "make_synthetic_checkpoint.py --out /tmp/mistral_synth --layers 2 --hidden 1024 "
        "--intermediate 2048 --vocab 2048"
    ),
)

ISL = 128


@pytest.fixture(scope="module")
def checkpoint():
    return Path(os.environ["HF_MODEL"])


@pytest.fixture(scope="module")
def loaded(checkpoint):
    """The production loader's output: dequantized, prefix-mapped, HF-convention (unswizzled).

    ``convert_to_meta_format=False`` because the comparison here is against HF-convention weights;
    the Meta swizzle is the DEVICE's requirement and is checked separately in
    ``tests/unit/test_fp8_loader.py``.
    """
    return ModelArgs.load_state_dict(checkpoint, convert_to_meta_format=False)


@pytest.fixture(scope="module")
def hf_config(checkpoint):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(str(checkpoint), trust_remote_code=True)
    return getattr(cfg, "text_config", cfg)


@pytest.fixture(scope="module")
def ground_truth(checkpoint):
    path = checkpoint / "reference_bf16.pt"
    if not path.is_file():
        pytest.skip(
            f"{path.name} not present — that file is written by make_synthetic_checkpoint.py and "
            "carries the pre-quantization weights this test compares against. A published "
            "checkpoint has no such file; run the transformers-state-dict test instead."
        )
    return torch.load(path, weights_only=True)


def test_loader_keys_are_the_text_backbone(loaded, hf_config):
    """The loader must emit the text-backbone naming, with no wrapper prefix and no scale keys."""
    assert loaded, "the loader returned an empty state dict"
    for key in loaded:
        assert not key.startswith("model.language_model."), f"wrapper prefix survived on {key}"
        assert not key.startswith("language_model."), f"wrapper prefix survived on {key}"
        assert "vision_tower" not in key and "multi_modal_projector" not in key, f"vision key leaked: {key}"
        assert not key.endswith(".weight_scale"), f"quantization scale leaked into the weights: {key}"
        assert not key.endswith(".input_scale"), f"activation scale leaked into the weights: {key}"

    expected = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(hf_config.num_hidden_layers):
        for suffix in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        ):
            expected.add(f"model.layers.{layer}.{suffix}")
    missing = sorted(expected - set(loaded))
    assert not missing, f"the loader did not produce {len(missing)} expected tensors, e.g. {missing[:5]}"
    logger.info(f"loader produced all {len(expected)} text-backbone tensors and no metadata keys")


def test_loaded_weights_match_pre_quantization_ground_truth(loaded, ground_truth):
    """Every loaded weight against the bf16 it was quantized from.

    fp8 e4m3 keeps 3 mantissa bits, so a per-tensor-scaled round trip lands within a few percent
    RELATIVE error on typical entries; the assertion is on the median relative error, which a wrong
    or missing scale blows past by orders of magnitude (a dropped scale is a factor of ~1/scale).
    """
    worst_name, worst_err = None, 0.0
    for name, want in ground_truth.items():
        got = loaded.get(name)
        assert got is not None, f"the loader dropped {name}"
        assert tuple(got.shape) == tuple(want.shape), f"{name}: shape {tuple(got.shape)} != {tuple(want.shape)}"
        want_f, got_f = want.float(), got.float()
        denominator = want_f.abs().median().clamp(min=1e-8)
        rel = ((got_f - want_f).abs().median() / denominator).item()
        if rel > worst_err:
            worst_name, worst_err = name, rel
        assert rel < 0.1, f"{name}: median relative error {rel:.4f} — a scale is missing or wrong"
    logger.info(f"loader round trip OK; worst median relative error {worst_err:.5f} on {worst_name}")


def test_reference_forward_and_first_token_agree(loaded, ground_truth, hf_config):
    """A full forward on the LOADED weights vs the same forward on the ground-truth weights.

    Per-weight error can be small and still compound; this is the check that the model built from
    the loader's output behaves like the model the checkpoint describes. The first-token argmax is
    the discrete sign-off.
    """
    from models.common.utility_functions import comp_pcc

    torch.manual_seed(0)
    token_ids = torch.randint(0, hf_config.vocab_size, (1, ISL))

    got = reference.model_reference_forward(reference.build_reference_model(hf_config, state_dict=loaded), token_ids)
    want = reference.model_reference_forward(
        reference.build_reference_model(hf_config, state_dict=ground_truth), token_ids
    )

    ok, pcc = comp_pcc(want.logits, got.logits, 0.99)
    logger.info(f"forward on loaded vs ground-truth weights: logits pcc={pcc}")
    assert ok, f"the loaded weights do not reproduce the checkpoint's forward: {pcc}"

    for layer_idx, ((gk, gv), (wk, wv)) in enumerate(zip(got.kv, want.kv)):
        ok_k, pcc_k = comp_pcc(wk, gk, 0.99)
        ok_v, pcc_v = comp_pcc(wv, gv, 0.99)
        assert ok_k and ok_v, f"layer {layer_idx} KV differs: K={pcc_k} V={pcc_v}"

    first_token_got = int(got.logits[0, -1].argmax())
    first_token_want = int(want.logits[0, -1].argmax())
    logger.info(f"first token: loaded={first_token_got} ground_truth={first_token_want}")
    assert first_token_got == first_token_want, f"first-token argmax disagrees: {first_token_got} != {first_token_want}"


def test_against_transformers_own_load(loaded, checkpoint):
    """Our loaded weights against the state dict transformers itself produces, where it can load.

    Two ways this legitimately does not run, and both are skips rather than failures because the
    loader is already covered above:

      * transformers has no model class for this config (``AutoModelForCausalLM`` does not accept the
        ``Mistral3Config`` wrapper, so the concrete wrapper class is used instead);
      * transformers has no fp8 quantization backend on this host, in which case it reads the fp8
        BYTES and casts them to bf16 without applying ``weight_scale`` — the tensors then sit at
        roughly ``1/scale`` times their true magnitude. Comparing against that would fail for a
        reason that has nothing to do with our loader, so it is detected and skipped explicitly.
    """
    try:
        from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

        model = Mistral3ForConditionalGeneration.from_pretrained(
            str(checkpoint), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
        )
    except Exception as e:  # no class for the config, missing backend, OOM, ...
        pytest.skip(f"transformers could not load this checkpoint itself ({type(e).__name__}: {e}); skipping")

    hf_state = model.state_dict()
    del model
    mapped = {}
    for key, value in hf_state.items():
        for prefix in ("model.language_model.", "language_model.model.", "language_model."):
            if key.startswith(prefix):
                key = "model." + key[len(prefix) :]
                break
        mapped[key] = value

    # Did transformers actually dequantize? Compare magnitudes on a projection that IS fp8 in the
    # checkpoint: if it read the raw bytes, its values sit orders of magnitude off ours.
    probe = f"model.layers.0.self_attn.q_proj.weight"
    if probe in mapped and probe in loaded:
        ours = loaded[probe].float().abs().median().clamp(min=1e-12)
        theirs = mapped[probe].float().abs().median().clamp(min=1e-12)
        ratio = float(theirs / ours)
        if not 0.5 < ratio < 2.0:
            pytest.skip(
                f"transformers loaded the fp8 bytes without applying the per-tensor scales "
                f"(median magnitude ratio {ratio:.3g} on {probe}) — no fp8 quantization backend on "
                f"this host, so its state dict is not a valid reference"
            )

    compared = 0
    for name, want in mapped.items():
        if any(marker in name for marker in ("vision_tower", "multi_modal_projector")):
            continue
        got = loaded.get(name)
        if got is None:
            continue  # transformers may expose buffers (rotary inv_freq) the loader has no use for
        denominator = want.float().abs().median().clamp(min=1e-8)
        rel = ((got.float() - want.float()).abs().median() / denominator).item()
        assert rel < 0.1, f"{name}: median relative error {rel:.4f} vs transformers' own load"
        compared += 1
    assert compared > 0, "compared no tensors against transformers' load"
    logger.info(f"agreed with transformers' own load on {compared} tensors")
