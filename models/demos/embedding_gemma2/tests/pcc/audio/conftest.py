# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Audio fixtures. The device (`eg2_device`, module scope) and snapshot come from the model conftest."""

import pytest
import torch

from models.demos.embedding_gemma2.tests.pcc.audio.audio_common import LAYER_CASES, SUBLAYERS, crop_case, load_case
from models.demos.embedding_gemma2.tt.audio.config import AudioConfig
from models.demos.embedding_gemma2.tt.audio.weights import load_audio_config_dicts, load_audio_state_dict


@pytest.fixture(scope="session")
def audio_state_dict(eg2_snapshot):
    return load_audio_state_dict(eg2_snapshot)


@pytest.fixture(scope="session")
def audio_config(eg2_snapshot):
    return AudioConfig.from_hf(*load_audio_config_dicts(eg2_snapshot))


@pytest.fixture(scope="session")
def hf_audio(eg2_snapshot):
    from models.demos.embedding_gemma2.reference.hf_audio_reference import load_reference_audio

    torch.set_num_threads(max(torch.get_num_threads(), 16))
    tower, embed, _ = load_reference_audio(eg2_snapshot)
    return tower, embed


@pytest.fixture(scope="session")
def audio_case(eg2_snapshot):
    """audio_case(name) -> (input_features [B,T,128], input_features_mask [B,T]) from the HF feature extractor.

    name is a case of audio_common.CASES or "crop<N>" (first 4N frames of s30, N tokens)."""

    def get(case):
        if case.startswith("crop"):
            return crop_case(str(eg2_snapshot), int(case[4:]))
        return load_case(str(eg2_snapshot), case)

    return get


@pytest.fixture(scope="session")
def audio_reference_for(hf_audio, audio_case):
    """audio_reference_for(case) -> (features, mask, AudioReferenceOutputs).

    Records every layer's in/out; FFN / attention / lconv in/out for SUBLAYERS on LAYER_CASES and for
    layer 0 on crop cases."""
    from models.demos.embedding_gemma2.reference.hf_audio_reference import run_audio_reference

    tower, embed = hf_audio
    cache = {}

    def get(case):
        if case not in cache:
            if len(cache) >= 3:
                cache.pop(next(iter(cache)))
            feats, mask = audio_case(case)
            sub = SUBLAYERS if case in LAYER_CASES else ((0,) if case.startswith("crop") else ())
            cache[case] = (feats, mask, run_audio_reference(tower, embed, feats, mask, capture_sublayers=sub))
        return cache[case]

    return get


@pytest.fixture(scope="module")
def tt_audio_model(eg2_device, audio_state_dict, audio_config, eg2_cache_dir):
    from models.demos.embedding_gemma2.tt.audio.audio_model import AudioModel

    return AudioModel.from_state_dict(audio_state_dict, audio_config, eg2_device, cache_dir=eg2_cache_dir)


@pytest.fixture(scope="module")
def tt_audio_encoder(eg2_device, tt_audio_model, eg2_snapshot):
    from models.demos.embedding_gemma2.tt.audio.audio_encoder import AudioEncoder

    return AudioEncoder(tt_audio_model, eg2_device, eg2_snapshot)
