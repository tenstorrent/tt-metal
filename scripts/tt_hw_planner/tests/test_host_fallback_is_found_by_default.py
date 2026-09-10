# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A hot path touching the reference model is a host fallback, whatever the submodule is called.

The check that catches "torch ran on the host while the report calls it a device measurement" was a
list of six submodule names:

    _HF_ALIAS_ROOTS = ("text_decoder", "t2u_model", "vocoder", "speech_encoder",
                       "text_encoder", "lm_head")

which are SeamlessM4T's. Voxtral's are audio_tower and language_model; Gemma's are vision_tower and
language_model. Neither model has ever been checked -- a pipeline calling hf_model.audio_tower(...)
from decode_step passed silently, which is the entire failure the gate exists to prevent.

An allow-list of known-bad names can only produce false NEGATIVES, and every new model is a new one.
Inverted here: any attribute of hf_model reached from a hot function counts, with the metadata
accessors exempt.
"""

import importlib.util as _u
import re
from pathlib import Path

_EMIT = Path(__file__).resolve().parent.parent / "commands" / "emit_e2e.py"


def _mod():
    spec = _u.spec_from_file_location("_emit_e2e_under_test", _EMIT)
    m = _u.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _flagged(m, text: str) -> bool:
    return any(re.search(p, text) for p in m._G1B_HF_FALLBACK)


def test_a_submodule_nobody_listed_is_still_a_fallback():
    m = _mod()
    for call in (
        "self.hf_model.audio_tower(x)",  # voxtral
        "hf_model.language_model.forward(ids)",  # voxtral / gemma
        "self.hf_model.vision_tower(pix)",  # gemma
        "hf_model.some_future_tower(x)",
    ):
        assert _flagged(m, call), call


def test_the_names_that_used_to_be_the_whole_list_still_match():
    m = _mod()
    for call in ("hf_model.text_decoder(x)", "self.hf_model.vocoder(z)", "hf_model.lm_head(h)"):
        assert _flagged(m, call), call


def test_metadata_accessors_are_not_compute():
    """The exemptions carry configuration, not tensors; flagging them would fail every pipeline."""
    m = _mod()
    for call in (
        "n = hf_model.config.num_hidden_layers",
        "d = self.hf_model.device",
        "g = hf_model.generation_config.get_x(1)",
    ):
        assert not _flagged(m, call), call


def test_the_attribute_test_defaults_to_catching():
    m = _mod()
    assert m._is_hf_compute_attr("audio_tower") is True
    assert m._is_hf_compute_attr("anything_at_all") is True
    assert m._is_hf_compute_attr("config") is False
    assert m._is_hf_compute_attr("") is False
    assert m._is_hf_compute_attr("_private") is False


def test_no_model_specific_name_list_remains():
    src = _EMIT.read_text()
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "_HF_ALIAS_ROOTS" not in code, "the SeamlessM4T name list is back"
