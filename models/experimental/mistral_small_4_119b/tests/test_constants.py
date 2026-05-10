# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import models.experimental.mistral_small_4_119b as ms4
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    expert_index_ranges_per_mesh_device,
    text_decoder_layer_state_dict_prefix,
)


def test_hf_model_id():
    assert HF_MODEL_ID == "mistralai/Mistral-Small-4-119B-2603"
    assert ms4.HF_MODEL_ID == HF_MODEL_ID


def test_text_decoder_layer_state_dict_prefix():
    assert text_decoder_layer_state_dict_prefix(0) == "language_model.model.layers.0."
    assert text_decoder_layer_state_dict_prefix(3) == "language_model.model.layers.3."


def test_expert_index_ranges_per_mesh_device_default():
    assert expert_index_ranges_per_mesh_device(128, 4) == [(0, 32), (32, 64), (64, 96), (96, 128)]


def test_architecture_contract():
    assert ms4.EXPECTED_HIDDEN_SIZE == 4096
    assert ms4.EXPECTED_NUM_LAYERS == 36
    assert ms4.EXPECTED_VOCAB_SIZE == 131072
    assert ms4.EXPECTED_RMS_NORM_EPS == 1e-6
    assert ms4.EXPECTED_VISION_HIDDEN_SIZE == 1024
    assert ms4.EXPECTED_NUM_EXPERTS == 128
    assert ms4.EXPECTED_NUM_EXPERTS_PER_TOK == 4
