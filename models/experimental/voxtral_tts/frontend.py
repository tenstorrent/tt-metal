# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side front end: text + a voice name in, prompt embeddings out.

The serving path's (`demo/`) single import for everything before the device. A façade over
`reference/`'s tokenizer and prompt assembly, not a copy: see VOXTRAL_TTS_BRINGUP.md [ref-02].
"""

from __future__ import annotations

import torch

from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as _pref
from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import TekkenTokenizer


def voices():
    """-> every voice preset the checkpoint ships, sorted."""
    return sorted(TekkenTokenizer().voices)


def prompt_ids(text: str, voice: str):
    """-> the prompt token ids, bit-exact against `mistral_common` (pinned by test_tokenizer_ref)."""
    return TekkenTokenizer().build_prompt(text, voice)


def build_prompt_embeds(text: str, voice: str, backbone_state):
    """text + voice -> inputs_embeds [1, P, 3072] for `TtVoxtralPipeline.generate`. Pass the
    pipeline's `wb` as `backbone_state` to avoid a second copy. see VOXTRAL_TTS_BRINGUP.md [pipe-03]
    """
    ids = torch.tensor(prompt_ids(text, voice), dtype=torch.long)
    return _pref.build_inputs_embeds(ids, _pref.load_voice(voice), backbone_state)
