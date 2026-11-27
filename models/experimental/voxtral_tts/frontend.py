# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side front end: text + a voice name in, prompt embeddings out.

The serving path's single import for everything before the device. A façade over `reference/`'s
tokenizer and prompt assembly, not a copy, so the serving path and the fp32 reference the tests gate
against cannot drift apart. Every function takes an optional `model_dir`; without one it uses the locally available model (see reference/voxtral_paths).
"""

from __future__ import annotations

import functools
import os

import torch

from models.experimental.voxtral_tts.reference import voxtral_pipeline_ref as _pref
from models.experimental.voxtral_tts.reference.voxtral_paths import MODEL_DIR
from models.experimental.voxtral_tts.reference.voxtral_tokenizer_ref import TekkenTokenizer


@functools.lru_cache(maxsize=4)
def _tokenizer(model_dir):
    return TekkenTokenizer(os.path.join(model_dir, "tekken.json"))


def voices(model_dir=None):
    """-> every voice preset the checkpoint ships, sorted."""
    return sorted(_tokenizer(model_dir or MODEL_DIR).voices)


def prompt_ids(text: str, voice: str, model_dir=None):
    """-> the prompt token ids, bit-exact against `mistral_common` (pinned by test_tokenizer_ref)."""
    return _tokenizer(model_dir or MODEL_DIR).build_prompt(text, voice)


def build_prompt_embeds(text: str, voice: str, backbone_state, model_dir=None):
    """text + voice -> inputs_embeds [1, P, 3072] for `TtVoxtralPipeline.generate`. Pass the
    pipeline's `wb` as `backbone_state` so the embedding tables are not loaded a second time.
    """
    model_dir = model_dir or MODEL_DIR
    ids = torch.tensor(prompt_ids(text, voice, model_dir), dtype=torch.long)
    voice_dir = os.path.join(model_dir, "voice_embedding")
    return _pref.build_inputs_embeds(ids, _pref.load_voice(voice, voice_dir), backbone_state)
