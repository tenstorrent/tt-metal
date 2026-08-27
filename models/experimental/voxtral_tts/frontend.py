# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side front end: text + a voice name in, prompt embeddings out.

One place for everything that happens before the device, so the serving path (`demo/`) has a single
import and does not reach into `reference/`.

It is a FAÇADE, not a copy. The tokenizer and the prompt assembler still live in `reference/`
because that package is the fp32 oracle the PCC tests gate against, and a second implementation
here is exactly the drift this repo has been removing. The sibling xtts_v2 model went further and
made its front end standalone (it had to -- its tokenizer needed coqui); this one does not need to,
so it does not.
"""

from __future__ import annotations

import torch

from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as _pref
from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import TekkenTokenizer


def voices():
    """-> every voice preset the checkpoint ships (20 of them), sorted."""
    return sorted(TekkenTokenizer().voices)


def prompt_ids(text: str, voice: str):
    """-> the prompt token ids, bit-exact against `mistral_common` (pinned by test_tokenizer_ref)."""
    return TekkenTokenizer().build_prompt(text, voice)


def build_prompt_embeds(text: str, voice: str, backbone_state):
    """text + voice -> inputs_embeds [1, P, 3072], ready for `TtVoxtralPipeline.generate`.

    `backbone_state` is the loaded checkpoint dict -- `TtVoxtralPipeline` already holds one as
    `.wb`, so pass that rather than loading a second copy of ~13 GB.
    """
    ids = torch.tensor(prompt_ids(text, voice), dtype=torch.long)
    return _pref.build_inputs_embeds(ids, _pref.load_voice(voice), backbone_state)
