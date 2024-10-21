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
    profiler,
)
from models.demos.whisper.tt import ttnn_functional_whisper, ttnn_optimized_functional_whisper
from models.generation_utils import get_logits_processor, pad_input_32
from ttnn.model_preprocessing import preprocess_model_parameters

import torch
import os
from os import listdir
from os.path import isfile, join

from transformers import AutoFeatureExtractor, WhisperForAudioClassification
from datasets import load_dataset
from torchmetrics.text import WordErrorRate
from sklearn.metrics import accuracy_score


def load_input_paths(folder_path):
    files = [os.path.join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f))]
    return files


def run_generate(
    config,
    input_embeds,
    input_features,
    ttnn_model,
    decoder_hidden_states,
    decoder_attention_mask,
    parameters,
    ttnn_linear_weight,
    device,
    decoder_input_ids,
    generation_config,
    batch_size,
    max_tokens,
    whisper_memory_config,
):
    logits_processor = get_logits_processor(decoder_input_ids, config)
    decoder_start_values = generation_config.pad_token_id * torch.ones(batch_size, input_features.shape[1]).to(
        torch.long
    )
    eos_reached = torch.zeros(batch_size, dtype=torch.bool)

    profiler.start(f"inference_time")
    for i in range(max_tokens):
        ttnn_output = ttnn_model.whisper_for_conditional_generation(
            config=config,
            input_embeds=input_embeds,
            decoder_hidden_states=decoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            parameters=parameters,
            device=device,
            ttnn_linear_weight=ttnn_linear_weight,
            whisper_memory_config=whisper_memory_config,
        )
        ttnn_output = ttnn.from_device(ttnn_output)
        logits_to_torch = ttnn.to_torch(ttnn_output)
        next_token_logits = logits_to_torch[:, i, :]
        next_tokens_scores = logits_processor(input_features, next_token_logits)
        next_tokens = torch.argmax(next_tokens_scores, dim=-1).unsqueeze(0)

        # Check if EOS token is generated for any sample in the batch and
        # Setting subsequent next_tokens to config.pad_token_id if EOS token is reached.
        eos_generated_flags = next_tokens == config.eos_token_id
        eos_reached = eos_reached | eos_generated_flags.squeeze(0)
        next_tokens[:, eos_reached] = config.pad_token_id

        if (i + 1) % 32 == 0:
            decoder_input_ids = torch.cat([decoder_input_ids, decoder_start_values], dim=1)

        decoder_input_ids[:, i + 1] = next_tokens[:, None]
        decoder_hidden_states, decoder_attention_mask = ttnn_model.preprocess_decoder_inputs(
            config=config,
            input_ids=decoder_input_ids,
            attention_mask=None,
            parameters=parameters.decoder,
            device=device,
        )

        if torch.all(next_tokens == config.eos_token_id):
            break

    profiler.end(f"inference_time")
    return decoder_input_ids


def run_demo_functional_whisper_for_audio_classification_inference(
    device, model_name, input_path, ttnn_model, num_inputs, batch_size, whisper_memory_config
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
        batched_inputs = input_features if i == 0 else torch.cat((batched_inputs, input_features), dim=0)

    config = model.config
    input_embedding = ttnn_model.preprocess_encoder_inputs(
        input_features=batched_inputs,
        parameters=parameters.encoder,
        device=device,
        whisper_memory_config=whisper_memory_config,
    )

    out_logits = ttnn_model.whisper_for_audio_classification(
        config=config,
        inputs_embeds=input_embedding,
        parameters=parameters,
        device=device,
        batch_size=batch_size,
        whisper_memory_config=whisper_memory_config,
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


def run_demo_functional_whisper_for_conditional_generation_inference(
    device,
    reset_seeds,
    batch_size,
    model_name,
    input_path,
    ttnn_model,
    max_tokens=32,
    whisper_memory_config=ttnn.L1_MEMORY_CONFIG,
):
    model = WhisperModel.from_pretrained(model_name).eval()
    config = WhisperConfig.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name, language="English", task="transcribe")
    hf_reference_model = WhisperForConditionalGeneration.from_pretrained(model_name)

    linear_weight = hf_reference_model.proj_out.weight
    ttnn_linear_weight = ttnn.from_torch(linear_weight, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    ttnn_linear_weight = ttnn.permute(ttnn_linear_weight, (1, 0))
    ttnn_linear_weight = ttnn.to_layout(ttnn_linear_weight, layout=ttnn.TILE_LAYOUT)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    input_data = load_input_paths(input_path)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )

    if len(input_data) < batch_size:
        assert False, "batch_size exceeds number of audio files available in folder"

    for i in range(batch_size):
        input_file_path = input_data[i]
        samplerate, data = wavfile.read(input_file_path)
        inputs = feature_extractor(data, sampling_rate=samplerate, return_tensors="pt")
        dtype_to_use = torch.bfloat16
        input_features = inputs.input_features.type(dtype_to_use)
        batched_inputs = input_features if i == 0 else torch.cat((batched_inputs, input_features), dim=0)

        decoder_input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id
        decoder_input_ids = pad_input_32(decoder_input_ids, config.pad_token_id).to(torch.long)
        batched_decoder_input_ids = (
            decoder_input_ids if i == 0 else torch.cat((batched_decoder_input_ids, decoder_input_ids), dim=0)
        )

    profiler.start(f"preprocessing_inputs")
    (input_embeds, decoder_hidden_states, decoder_attention_mask) = ttnn_model.preprocess_inputs(
        config=config,
        input_features=batched_inputs,
        input_ids=batched_decoder_input_ids,
        attention_mask=None,
        parameters=parameters,
        device=device,
        whisper_memory_config=whisper_memory_config,
    )
    profiler.end(f"preprocessing_inputs")

    generation_config = hf_reference_model.generation_config
    ttnn_output = run_generate(
        config,
        input_embeds,
        batched_inputs,
        ttnn_model,
        decoder_hidden_states,
        decoder_attention_mask=decoder_attention_mask,
        parameters=parameters,
        ttnn_linear_weight=ttnn_linear_weight,
        device=device,
        decoder_input_ids=batched_decoder_input_ids,
        generation_config=generation_config,
        batch_size=batch_size,
        max_tokens=max_tokens,
        whisper_memory_config=whisper_memory_config,
    )

    profiler.start(f"post_processing_output_to_string")
    ttnn_transcription = processor.batch_decode(ttnn_output, skip_special_tokens=True)
    profiler.end(f"post_processing_output_to_string")

    logger.info("Model Output")
    logger.info(ttnn_transcription)

    measurements = {
        "preprocessing_input": profiler.get("preprocessing_input"),
        "inference_time": profiler.get("inference_time"),
        "post_processing": profiler.get("post_processing_output_to_string"),
    }

    logger.info(f"preprocessing_input: {measurements['preprocessing_input']} s")
    logger.info(f"inference_time: {measurements['inference_time']} s")
    logger.info(f"post_processing : {measurements['post_processing']} s")

    return measurements, ttnn_transcription


def run_demo_functional_whisper_for_audio_classification_dataset(
    device, reset_seeds, model_name, ttnn_model, batch_size, n_iterations, whisper_memory_config
):
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = WhisperForAudioClassification.from_pretrained(model_name)

    model.eval()
    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    sample = iter(ds)

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
        for i in range(batch_size):
            s = next(sample)
            inputs = feature_extractor(s["audio"]["array"], sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.type(torch.bfloat16)
            batch_input = input_features if i == 0 else torch.cat((batch_input, input_features), dim=0)
            reference_labels.append(s["language"])

        input_embedding = ttnn_model.preprocess_encoder_inputs(
            input_features=batch_input,
            parameters=parameters.encoder,
            device=device,
            whisper_memory_config=whisper_memory_config,
        )

        out_logits = ttnn_model.whisper_for_audio_classification(
            config=config,
            inputs_embeds=input_embedding,
            parameters=parameters,
            device=device,
            batch_size=batch_size,
            whisper_memory_config=whisper_memory_config,
        )
        logits_torch = ttnn.to_torch(out_logits)

        for i in range(batch_size):
            single_logits_torch = logits_torch[i].squeeze(0)
            predicted_class_ids = torch.argmax(single_logits_torch).item()
            predicted_label = model.config.id2label[predicted_class_ids]
            predicted_labels.append(predicted_label)

    accuracy = accuracy_score(reference_labels, predicted_labels)
    logger.info(f"Reference labels: {reference_labels}")
    logger.info(f"Predicted labels: {predicted_labels}")
    logger.info(f"Accuracy: {accuracy}")
    return accuracy


def run_demo_functional_whisper_for_conditional_generation_dataset(
    device,
    reset_seeds,
    model_name,
    ttnn_model,
    batch_size=1,
    n_iterations=1,
    max_tokens=32,
    whisper_memory_config=ttnn.L1_MEMORY_CONFIG,
):
    model = WhisperModel.from_pretrained(model_name).eval()
    config = WhisperConfig.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name, language="English", task="transcribe")
    hf_reference_model = WhisperForConditionalGeneration.from_pretrained(model_name)

    linear_weight = hf_reference_model.proj_out.weight
    ttnn_linear_weight = ttnn.from_torch(linear_weight, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    ttnn_linear_weight = ttnn.permute(ttnn_linear_weight, (1, 0))
    ttnn_linear_weight = ttnn.to_layout(ttnn_linear_weight, layout=ttnn.TILE_LAYOUT)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    sample = iter(ds)
    batched_ground_truth_transcriptions = []

    for _ in range(n_iterations):
        for i in range(batch_size):
            s = next(sample)
            inputs = feature_extractor(s["audio"]["array"], sampling_rate=16000, return_tensors="pt")
            ground_truth_transcriptions = s["text"]
            dtype_to_use = torch.bfloat16
            input_features = inputs.input_features.type(dtype_to_use)

            batched_inputs = input_features if i == 0 else torch.cat((batched_inputs, input_features), dim=0)

            decoder_input_ids = torch.tensor([[1, 1]]) * config.decoder_start_token_id
            decoder_input_ids = pad_input_32(decoder_input_ids, config.pad_token_id).to(torch.long)
            batched_decoder_input_ids = (
                decoder_input_ids if i == 0 else torch.cat((batched_decoder_input_ids, decoder_input_ids), dim=0)
            )

            batched_ground_truth_transcriptions.append(ground_truth_transcriptions)

        parameters = preprocess_model_parameters(
            initialize_model=lambda: model.eval(),
            convert_to_ttnn=ttnn_model.convert_to_ttnn,
            custom_preprocessor=ttnn_model.custom_preprocessor,
            device=device,
        )

        (input_embeds, decoder_hidden_states, decoder_attention_mask) = ttnn_model.preprocess_inputs(
            config=config,
            input_features=batched_inputs,
            input_ids=batched_decoder_input_ids,
            attention_mask=None,
            parameters=parameters,
            device=device,
            whisper_memory_config=whisper_memory_config,
        )

        ttnn_output = run_generate(
            config,
            input_embeds,
            batched_inputs,
            ttnn_model,
            decoder_hidden_states,
            decoder_attention_mask=decoder_attention_mask,
            parameters=parameters,
            ttnn_linear_weight=ttnn_linear_weight,
            device=device,
            decoder_input_ids=batched_decoder_input_ids,
            generation_config=hf_reference_model.generation_config,
            batch_size=batch_size,
            max_tokens=max_tokens,
            whisper_memory_config=whisper_memory_config,
        )
        ttnn_transcription = processor.batch_decode(ttnn_output, skip_special_tokens=True)

        logger.info("Model Output")
        logger.info(ttnn_transcription)

        wer = WordErrorRate()
        wer_scores = []
        for transcription, ground_truth in zip(ttnn_transcription, batched_ground_truth_transcriptions):
            transcription = transcription.upper()
            individual_wer_score = wer([transcription], [ground_truth])
            wer_scores.append(individual_wer_score)
            logger.info(f"Individual Sample WER score: {individual_wer_score}")

        average_wer_score = sum(wer_scores) / len(wer_scores)
        logger.info(f"Average WER score: {average_wer_score}")
        accuracy = 1 - average_wer_score
        logger.info(f"Accuracy: {accuracy}")

    return average_wer_score


@pytest.mark.parametrize(
    "model_name, input_loc",
    ((["sanchit-gandhi/whisper-medium-fleurs-lang-id", "models/demos/whisper/demo/dataset/audio_classification"]),),
)
@pytest.mark.parametrize(
    ("ttnn_model", "num_inputs", "batch_size", "WHISPER_MEMORY_CONFIG"),
    ((ttnn_optimized_functional_whisper, 1, 8, ttnn.DRAM_MEMORY_CONFIG),),
)
def test_demo_for_audio_classification(
    device,
    reset_seeds,
    use_program_cache,
    model_name,
    input_loc,
    ttnn_model,
    num_inputs,
    batch_size,
    WHISPER_MEMORY_CONFIG,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    return run_demo_functional_whisper_for_audio_classification_inference(
        device,
        model_name=model_name,
        input_path=input_loc,
        ttnn_model=ttnn_model,
        num_inputs=num_inputs,
        batch_size=batch_size,
        whisper_memory_config=WHISPER_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "model_name, input_loc",
    ((["openai/whisper-tiny.en", "models/demos/whisper/demo/dataset/conditional_generation"]),),
)
@pytest.mark.parametrize(
    ("ttnn_model", "batch_size", "max_tokens", "WHISPER_MEMORY_CONFIG"),
    ((ttnn_optimized_functional_whisper, 8, 32, ttnn.L1_MEMORY_CONFIG),),
)
def test_demo_for_conditional_generation(
    device,
    reset_seeds,
    use_program_cache,
    model_name,
    input_loc,
    ttnn_model,
    batch_size,
    max_tokens,
    WHISPER_MEMORY_CONFIG,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    return run_demo_functional_whisper_for_conditional_generation_inference(
        device,
        reset_seeds,
        batch_size=batch_size,
        model_name=model_name,
        input_path=input_loc,
        ttnn_model=ttnn_model,
        max_tokens=max_tokens,
        whisper_memory_config=WHISPER_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "model_name",
    (["sanchit-gandhi/whisper-medium-fleurs-lang-id"]),
)
@pytest.mark.parametrize(
    ("ttnn_model", "batch_size", "n_iterations", "WHISPER_MEMORY_CONFIG"),
    ((ttnn_optimized_functional_whisper, 8, 1, ttnn.DRAM_MEMORY_CONFIG),),
)
def test_demo_for_audio_classification_dataset(
    device, reset_seeds, use_program_cache, model_name, ttnn_model, batch_size, n_iterations, WHISPER_MEMORY_CONFIG
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    return run_demo_functional_whisper_for_audio_classification_dataset(
        device,
        reset_seeds,
        model_name=model_name,
        ttnn_model=ttnn_model,
        batch_size=batch_size,
        n_iterations=n_iterations,
        whisper_memory_config=WHISPER_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "model_name",
    (["openai/whisper-tiny.en"]),
)
@pytest.mark.parametrize(
    ("ttnn_model", "batch_size", "n_iterations", "max_tokens", "WHISPER_MEMORY_CONFIG"),
    ((ttnn_optimized_functional_whisper, 8, 1, 32, ttnn.L1_MEMORY_CONFIG),),
)
def test_demo_for_conditional_generation_dataset(
    device,
    reset_seeds,
    use_program_cache,
    model_name,
    ttnn_model,
    batch_size,
    n_iterations,
    max_tokens,
    WHISPER_MEMORY_CONFIG,
):
    disable_persistent_kernel_cache()
    disable_compilation_reports()
    return run_demo_functional_whisper_for_conditional_generation_dataset(
        device,
        reset_seeds,
        model_name=model_name,
        ttnn_model=ttnn_model,
        batch_size=batch_size,
        n_iterations=n_iterations,
        max_tokens=max_tokens,
        whisper_memory_config=WHISPER_MEMORY_CONFIG,
    )
