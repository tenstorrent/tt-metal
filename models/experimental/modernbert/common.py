# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared config / weight / input helpers for the ModernBERT demo.

Verified against transformers 5.10.2 (the version pinned in
tt_metal/python_env/requirements-dev.txt). Field names differ in other releases:
`global_rope_theta` / `local_rope_theta` do NOT exist here, the thetas live in
the nested `config.rope_parameters` dict.
"""

import os

import torch
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer

MODEL_ID = "answerdotai/ModernBERT-base"
MODEL_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"

# transformers exposes the two rotary thetas keyed by layer type.
FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"


def load_config():
    return AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def load_torch_model(model_location_generator=None, dtype=torch.float32, attn_implementation=None):
    """HuggingFace reference model (ModernBertModel -> last_hidden_state).

    Pass attn_implementation="eager" when comparing in bf16. HF defaults to
    "sdpa", a fused kernel; our reference uses the explicit matmul/softmax/matmul
    form. They agree exactly in fp32 and differ by 0.9979327794 in bf16, which is
    torch's own kernel spread rather than a modelling difference.
    """
    kwargs = {"dtype": dtype}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    if model_location_generator is None or "TT_GH_CI_INFRA" not in os.environ:
        model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION, **kwargs)
    else:
        weights_dir = model_location_generator("modernbert", model_subdir="", download_if_ci_v2=True)
        model = AutoModel.from_pretrained(weights_dir, **kwargs)
    model.eval()
    return model


def load_torch_mlm_model(dtype=torch.float32, attn_implementation=None):
    """HuggingFace ModernBertForMaskedLM (encoder + prediction head + decoder).

    The base checkpoint ships as ModernBertForMaskedLM, so head.dense.weight,
    head.norm.weight, decoder.weight and decoder.bias are present in the file;
    loading it as ModernBertModel simply reports them as unexpected.
    """
    kwargs = {"dtype": dtype}
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForMaskedLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, **kwargs)
    model.eval()
    return model


def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)


def rope_theta(config, layer_type):
    """Rotary theta for a layer type: 160000.0 full, 10000.0 sliding."""
    return config.rope_parameters[layer_type]["rope_theta"]


def sliding_window_half(config):
    """Half-width of the symmetric local attention band: +/-64, total 129.

    Do NOT use HF's `attn.sliding_window`, which holds 65 - an internal
    half-representation, not the band width.
    """
    return config.local_attention // 2


# Long enough, non-repeating text. A repeated short sentence is periodic and the
# period interacts with the 129-wide attention window differently at each
# sequence length, which produces misleading PCC numbers.
SAMPLE_TEXT = """The development of specialized hardware for machine learning has followed a
winding path. Early neural networks ran on general purpose processors, where the dominant
cost was moving data rather than computing on it. Graphics processors changed that calculus
by offering wide parallel arithmetic, though they retained a memory hierarchy designed for
rendering triangles rather than multiplying matrices. More recent accelerators abandon that
inheritance entirely. They place large scratchpad memories adjacent to compute units and
expose the movement of tensors as an explicit part of the programming model. The result is a
machine that rewards careful placement of data and punishes casual copying. Rivers carve
their valleys slowly, and the shape of a watershed records centuries of small decisions made
by water. A meander begins as a trivial asymmetry, perhaps a fallen tree or a patch of harder
rock. Flow accelerates on the outer bank and slows on the inner one, so sediment erodes from
one side and accumulates on the other. Over time the curve deepens until the river doubles
back on itself and cuts a new channel across the neck, abandoning the loop as an oxbow lake.
Farmers who work such floodplains learn to read these scars in the soil, because the
abandoned channels hold water differently and grow different crops. Fermentation is
controlled decay. A baker who leaves flour and water in a warm room is cultivating a
community of wild yeasts and lactic acid bacteria, each consuming sugars and excreting
compounds that the others tolerate or exploit. The sour flavour of a mature starter comes
from acids that also suppress competing organisms, which is why the culture becomes more
stable as it ages. Temperature shifts the balance. Warmer conditions favour bacteria that
produce sharper acids, while cooler ones let yeasts dominate and yield a milder loaf with
more gas. Questions about legal personhood rarely arrive in tidy form. A corporation is not a
person in any biological sense, yet it can own property, enter contracts, sue and be sued.
Courts extended these capacities gradually and for practical reasons, not because anyone
believed a firm possessed an inner life. Difficulty appears when doctrines built for one
purpose are borrowed for another. Rights of conscience presuppose a bearer capable of holding
convictions, and applying them to an entity whose decisions emerge from committees and
fiduciary duties produces conclusions that satisfy neither the letter nor the spirit of the
original rule. Cartographers of the eighteenth century faced a stubborn problem of longitude.
Latitude could be read from the sun or the pole star, but establishing how far east or west a
ship had travelled required knowing the time at a reference meridian, and no pendulum clock
survived the pitching of a deck. The eventual solution was mechanical rather than
astronomical, a sequence of marine chronometers whose escapements tolerated motion and
temperature swings. Astronomers had meanwhile proposed lunar distance tables, which worked but
demanded hours of computation from an exhausted navigator. Both methods persisted side by side
for decades, because redundancy at sea is worth more than elegance. The vocabulary of colour
varies enormously between languages, and the variation is not arbitrary. Communities that
distinguish few basic terms almost always separate dark from light first, then add red, then
green or yellow, then blue. Researchers once read this ordering as evidence that perception
itself differed, but later work suggested the constraint lies in what distinctions prove
useful to name. Dyes and pigments matter here: a culture with access to a stable blue colorant
tends to lexicalise blue earlier than one without. Glassblowers work within a narrow window of
temperature where the material is neither liquid nor solid but something between, stiff enough
to hold a shape yet mobile enough to yield. Skill consists largely in anticipating how quickly
that window closes, and in reheating before it does. An apprentice learns to read the colour of
the glowing mass rather than trust a clock, because the same nominal temperature behaves
differently depending on the thickness of the piece and the draught in the room."""


def build_inputs(seq_len=256, batch_size=1):
    """Realistic tokenized inputs. Never use torch.randint here - random token ids
    produce degenerate attention patterns that can mask real bugs."""
    tok = load_tokenizer()
    ids = tok(SAMPLE_TEXT, return_tensors="pt")["input_ids"]
    if ids.shape[1] < seq_len:
        raise ValueError(f"SAMPLE_TEXT yields only {ids.shape[1]} tokens, need {seq_len}")
    ids = ids[:, :seq_len].repeat(batch_size, 1)
    return ids, torch.ones_like(ids)
