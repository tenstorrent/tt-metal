# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.functional_whisper.demo.demo import test_demo_for_audio_classification as demo_audio_files
from models.experimental.functional_whisper.demo.demo import (
    test_demo_for_audio_classification_dataset as demo_audio_dataset,
)
import pytest
from models.experimental.functional_whisper.tt import ttnn_functional_whisper, ttnn_optimized_functional_whisper


@pytest.mark.parametrize(
    "input_path",
    (("models/experimental/functional_whisper/demo/dataset/audio_classification"),),
)
@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper, ttnn_functional_whisper),
)
@pytest.mark.parametrize(
    "batch_size",
    ((8),),
)
def test_audio_demo_batch_8(device, reset_seeds, input_path, ttnn_model, batch_size):
    expected_answers = {
        0: "English",
        1: "Estonian",
        2: "French",
        3: "Bengali",
        4: "Bengali",
        5: "Estonian",
        6: "English",
        7: "Indonesian",
    }
    predicted_labels = demo_audio_files(reset_seeds, input_path, ttnn_model, device, batch_size)

    for i in range(batch_size):
        assert expected_answers[i] == predicted_labels[i]


@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper, ttnn_functional_whisper),
)
@pytest.mark.parametrize(
    "batch_size",
    ((8),),
)
@pytest.mark.parametrize(
    "n_iterations",
    ((5),),
)
@pytest.mark.parametrize(
    "accuracy",
    ((0.7),),
)
def test_audio_demo_dataset(device, reset_seeds, ttnn_model, batch_size, n_iterations, accuracy):
    cal_acc = demo_audio_dataset(reset_seeds, ttnn_model, device, batch_size, n_iterations)
    assert cal_acc >= accuracy
