# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from datasets import load_dataset
from loguru import logger
from scipy.io import wavfile
import ttnn
from transformers import (
    AutoFeatureExtractor,
    WhisperModel,
    WhisperConfig,
    AutoProcessor,
    WhisperForConditionalGeneration,
)
from models.utility_functions import (
    disable_compilation_reports,
    disable_persistent_kernel_cache,
)
from models.experimental.functional_whisper.tt import ttnn_functional_whisper, ttnn_optimized_functional_whisper
from models.generation_utils import get_logits_processor
from ttnn.model_preprocessing import preprocess_model_parameters

import torch
import os
from os import listdir
from os.path import isfile, join

from transformers import AutoFeatureExtractor, WhisperForAudioClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score


def load_input_paths(folder_path):
    files = [os.path.join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]
    return files


def pad_input_32(tensor, value):
    len = tensor.shape[1]

    if len % 32 == 0:
        return tensor

    padded_len = ((len // 32) + 1) * 32

    pad_tensor = (value * torch.ones(tensor.shape[0], padded_len - len)).to(torch.long)
    tensor = torch.cat([tensor, pad_tensor], dim=1)

    return tensor


def run_generate(
    config,
    input_embeds,
    input_features,
    ttnn_model,
    decoder_hidden_states,
    decoder_attention_mask,
    parameters,
    processor,
    ttnn_linear_weight,
    device,
    generation_config,
):
    input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id

    logits_processor = get_logits_processor(input_ids, config)

    input_ids = pad_input_32(input_ids, config.pad_token_id).to(torch.long)

    decoder_start_values = generation_config.pad_token_id * torch.ones(1, 32).to(torch.long)

    for i in range(32):
        output = ttnn_model.whisper(
            config,
            input_embeds,
            decoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            parameters=parameters,
        )
        output = output @ ttnn_linear_weight

        output = ttnn.from_device(output)

        logits_to_torch = ttnn.to_torch(output)

        next_token_logits = logits_to_torch[:, i, :]

        next_tokens_scores = logits_processor(input_features, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        if (i + 1) % 32 == 0:
            input_ids = torch.cat([input_ids, decoder_start_values], dim=1)

        input_ids[:, i + 1] = next_tokens[:, None]

        decoder_hidden_states, decoder_attention_mask = ttnn_model.preprocess_decoder_inputs(
            config=config, input_ids=input_ids, attention_mask=None, parameters=parameters.decoder, device=device
        )

        if next_tokens == config.eos_token_id:
            break
        logger.info(processor.batch_decode(input_ids, skip_special_tokens=True)[0])

    ttnn_transcription = processor.batch_decode(input_ids, skip_special_tokens=True)[0]

    return ttnn_transcription


def run_demo_functional_whisper_for_audio_classification_inference(
    reset_seeds, input_path, ttnn_model, device, batch_size
):
    feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

    model.eval()
    input_data = load_input_paths(input_path)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )
    if len(input_data) < batch_size:
        assert False, "batch_size exceeds number of audio files available in folder"

    batched_inputs = []
    for i in range(batch_size):
        input_file_path = input_data[i]
        samplerate, data = wavfile.read(input_file_path)

        inputs = feature_extractor(
            data,
            sampling_rate=samplerate,
            return_tensors="pt",
        )

        input_features = inputs.input_features
        if i == 0:
            batched_inputs = input_features
        else:
            batched_inputs = torch.cat((batched_inputs, input_features), dim=0)

    config = model.config
    input_embedding = ttnn_model.preprocess_encoder_inputs(
        input_features=batched_inputs, parameters=parameters.encoder, device=device
    )

    out_logits = ttnn_model.whisper_for_audio_classification(
        config=config,
        inputs_embeds=input_embedding,
        parameters=parameters,
        device=device,
        batch_size=batch_size,
    )

    logits_torch = ttnn.to_torch(out_logits)
    predicted_list = []
    for i in range(batch_size):
        single_logits_torch = logits_torch[i].squeeze(0)
        predicted_class_ids = torch.argmax(single_logits_torch).item()
        predicted_label = model.config.id2label[predicted_class_ids]
        logger.info(f"predicted_label: {predicted_label}")
        predicted_list.append(predicted_label)
    return predicted_list


def run_demo_functional_whisper_for_conditional_generation_inference(input_path, ttnn_model, device, num_inputs):
    torch.manual_seed(0)

    model = WhisperModel.from_pretrained("openai/whisper-tiny.en").to(torch.bfloat16).eval()

    config = WhisperConfig.from_pretrained("openai/whisper-tiny.en")

    processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en", language="English", task="transcribe")
    hf_reference_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    linear_weight = hf_reference_model.proj_out.weight

    linear_weight = hf_reference_model.proj_out.weight
    ttnn_linear_weight = ttnn.from_torch(linear_weight, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    ttnn_linear_weight = ttnn.permute(ttnn_linear_weight, (1, 0))
    ttnn_linear_weight = ttnn.to_layout(ttnn_linear_weight, layout=ttnn.TILE_LAYOUT)

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
    input_data = load_input_paths(input_path)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )

    if len(input_data) < num_inputs:
        assert False, "num_inputs exceeds number of audio files available in folder"
    output_list = {}
    for i in range(num_inputs):
        input_file_path = input_data[i]
        samplerate, data = wavfile.read(input_file_path)
        inputs = feature_extractor(data, sampling_rate=samplerate, return_tensors="pt")
        dtype_to_use = torch.bfloat16
        input_features = inputs.input_features.type(dtype_to_use)

        decoder_input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id
        decoder_input_ids = pad_input_32(decoder_input_ids, config.pad_token_id).to(torch.long)

        attention_mask = None

        (input_embeds, decoder_hidden_states, decoder_attention_mask) = ttnn_model.preprocess_inputs(
            config=config,
            input_features=input_features,
            input_ids=decoder_input_ids,
            attention_mask=attention_mask,
            parameters=parameters,
            device=device,
        )

        generation_config = hf_reference_model.generation_config
        ttnn_output = run_generate(
            config,
            input_embeds,
            input_features,
            ttnn_model,
            decoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            parameters=parameters,
            processor=processor,
            ttnn_linear_weight=ttnn_linear_weight,
            device=device,
            generation_config=generation_config,
        )
        logger.info("Model Output")
        logger.info(ttnn_output)
        output_list[i] = ttnn_output
    for i in range(len(output_list)):
        logger.info(f"output for input {i+1}")
        logger.info(output_list[i])


def run_demo_functional_whisper_for_audio_classification_dataset(
    reset_seeds, ttnn_model, device, batch_size=8, n_iterations=1
):
    feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

    model.eval()
    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    ds_iter = iter(ds)

    reference_labels = []
    predicted_labels = []
    config = model.config
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )

    for _ in range(n_iterations):
        batch_input = []
        # prepare the batched audio inputs
        for bs in range(batch_size):
            sample = next(ds_iter)
            inputs = feature_extractor(
                sample["audio"]["array"],
                sampling_rate=sample["audio"]["sampling_rate"],
                return_tensors="pt",
            )
            input_features = inputs.input_features
            if bs == 0:
                batch_input = input_features
            else:
                batch_input = torch.cat((batch_input, input_features), dim=0)
            reference_labels.append(sample["language"])

        # preprocess the inputs
        input_embedding = ttnn_model.preprocess_encoder_inputs(
            input_features=batch_input, parameters=parameters.encoder, device=device
        )

        # run the model
        out_logits = ttnn_model.whisper_for_audio_classification(
            config=config,
            inputs_embeds=input_embedding,
            parameters=parameters,
            device=device,
            batch_size=batch_size,
        )

        # postprocessing the outputs
        logits_torch = ttnn.to_torch(out_logits)
        for i in range(batch_size):
            single_logits_torch = logits_torch[i].squeeze(0)
            predicted_class_ids = torch.argmax(single_logits_torch).item()
            predicted_label = model.config.id2label[predicted_class_ids]
            predicted_labels.append(predicted_label)

    accuracy = accuracy_score(reference_labels, predicted_labels)
    logger.info(f"reference labels: {reference_labels}")
    logger.info(f"predicted labels: {predicted_labels}")
    logger.info(f"Accuracy: {accuracy}")
    return accuracy


def run_demo_functional_whisper_for_conditional_generation_dataset(ttnn_model, device):
    torch.manual_seed(0)

    model = WhisperModel.from_pretrained("openai/whisper-tiny.en").to(torch.bfloat16).eval()

    config = WhisperConfig.from_pretrained("openai/whisper-tiny.en")

    processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en", language="English", task="transcribe")
    hf_reference_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    linear_weight = hf_reference_model.proj_out.weight

    linear_weight = hf_reference_model.proj_out.weight
    ttnn_linear_weight = ttnn.from_torch(linear_weight, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    ttnn_linear_weight = ttnn.permute(ttnn_linear_weight, (1, 0))
    ttnn_linear_weight = ttnn.to_layout(ttnn_linear_weight, layout=ttnn.TILE_LAYOUT)

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny.en")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], sampling_rate=16000, return_tensors="pt")
    dtype_to_use = torch.bfloat16
    input_features = inputs.input_features.type(dtype_to_use)

    decoder_input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id
    decoder_input_ids = pad_input_32(decoder_input_ids, config.pad_token_id).to(torch.long)

    attention_mask = None

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )

    (input_embeds, decoder_hidden_states, decoder_attention_mask) = ttnn_model.preprocess_inputs(
        config=config,
        input_features=input_features,
        input_ids=decoder_input_ids,
        attention_mask=attention_mask,
        parameters=parameters,
        device=device,
    )

    generation_config = hf_reference_model.generation_config
    ttnn_output = run_generate(
        config,
        input_embeds,
        input_features,
        ttnn_model,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        parameters=parameters,
        processor=processor,
        ttnn_linear_weight=ttnn_linear_weight,
        device=device,
        generation_config=generation_config,
    )
    logger.info("Model Output")
    logger.info(ttnn_output)


@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper, ttnn_functional_whisper),
)
@pytest.mark.parametrize(
    "batch_size",
    ((8),),
)
def test_demo_for_audio_classification(reset_seeds, input_path, ttnn_model, device, batch_size):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    return run_demo_functional_whisper_for_audio_classification_inference(
        reset_seeds, input_path, ttnn_model, device, batch_size
    )


@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper, ttnn_functional_whisper),
)
@pytest.mark.parametrize(
    "num_inputs",
    ((1),),
)
def test_demo_for_conditional_generation(input_path, ttnn_model, device, num_inputs):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    return run_demo_functional_whisper_for_conditional_generation_inference(input_path, ttnn_model, device, num_inputs)


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
def test_demo_for_audio_classification_dataset(reset_seeds, ttnn_model, device, batch_size, n_iterations):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    return run_demo_functional_whisper_for_audio_classification_dataset(
        reset_seeds, ttnn_model, device, batch_size=batch_size, n_iterations=n_iterations
    )


@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_functional_whisper, ttnn_optimized_functional_whisper),
)
def test_demo_for_conditional_generation_dataset(ttnn_model, device):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    return run_demo_functional_whisper_for_conditional_generation_dataset(ttnn_model, device)
