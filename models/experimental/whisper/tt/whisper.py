# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import (
    WhisperProcessor,
    WhisperModel,
    WhisperForAudioClassification,
    WhisperForConditionalGeneration,
    WhisperConfig,
    AutoFeatureExtractor,
    AutoProcessor,
)
from datasets import load_dataset
from loguru import logger
import torch
from transformers import AutoProcessor, WhisperForConditionalGeneration
from datasets import load_dataset


def run_for_conditional_generation():
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features

    generated_ids = model.generate(inputs=input_features)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def run_HF_whisper():
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    model_audio_classification = WhisperForAudioClassification.from_pretrained(
        "sanchit-gandhi/whisper-medium-fleurs-lang-id"
    )

    model.eval()
    config = model.config
    logger.debug(config)

    """
    Example:
    ```python
    >>> import torch
    >>> from transformers import AutoFeatureExtractor, WhisperModel
    >>> from datasets import load_dataset

    >>> model = WhisperModel.from_pretrained("openai/whisper-base")

    >>> feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-base")

    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    >>> input_features = inputs.input_features
    >>> decoder_input_ids = torch.tensor([[1, 1]]) * model.config.decoder_start_token_id

    >>> last_hidden_state = model(input_features, decoder_input_ids=decoder_input_ids).last_hidden_state
    >>> list(last_hidden_state.shape)
    [1, 2, 512]
    ```
    """


def change_model_configuration():
    model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    configuration = model.config
    # Change configuration, optionaly ....
    configuration.encoder_layers = 1
    # Initializing a model (with random weights) from the changed tiny style configuration
    model = WhisperModel(configuration)

    # Accessing the model configuration
    configuration = model.config
    logger.debug(configuration)


def run_HF_whisper_for_audio_classification():
    processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    # Whisper with language modeling head
    # Can be used for automatic speech recognition.
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    logger.debug(model)

    state_dict = model.state_dict()
    logger.debug(state_dict)

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # Process audio data
    inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
    input_features = inputs.input_features
    logger.debug(input_features)

    gen_config = model.generation_config
    logger.debug(gen_config)
    # This model can be used for automatic speech recognition.
    # Here we generate transcription
    # Causal generative model Whisper generates data based on a given context

    generated_ids = model.generate(inputs=input_features)
    logger.debug(generated_ids)

    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logger.debug(transcription)


if __name__ == "__main__":
    torch.manual_seed(1234)
    run_for_conditional_generation()
