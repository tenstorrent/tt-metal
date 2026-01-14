# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from os import listdir
from os.path import isfile, join
from typing import Optional

import jiwer
import pytest
import torch
from datasets import load_dataset
from evaluate import load
from loguru import logger
from scipy.io import wavfile
from tqdm import tqdm
from transformers import (
    AutoFeatureExtractor,
    AutoProcessor,
    WhisperForAudioClassification,
    WhisperForConditionalGeneration,
)
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.utils.llm_demo_utils import verify_perf
from models.demos.whisper.tt.ttnn_optimized_functional_whisper import (
    WHISPER_BATCH_SIZE,
    WHISPER_L1_SMALL_SIZE,
    WHISPER_TRACE_REGION_SIZE,
    convert_to_ttnn,
    create_custom_mesh_preprocessor,
    encoder,
    init_kv_cache,
    preprocess_encoder_inputs,
)
from models.demos.whisper.tt.whisper_generator import GenerationParams, WhisperGenerator

available_devices = len(ttnn.get_device_ids()) if ttnn.get_device_ids() else 1


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


def repeat_inputs_cyclically(input_data, total_inputs):
    """
    Repeat input data cyclically to match the total number of inputs needed.

    Args:
        input_data: List of input items to repeat
        total_inputs: Total number of inputs needed

    Returns:
        List of input items repeated cyclically to match total_inputs
    """
    if len(input_data) < total_inputs:
        # Repeat inputs cyclically to match total_inputs
        logger.info(
            f"Only {len(input_data)} audio files available, repeating cyclically to match {total_inputs} total inputs"
        )
        original_input_data = input_data.copy()
        while len(input_data) < total_inputs:
            input_data.extend(original_input_data)
        # Trim to exact size needed
        input_data = input_data[:total_inputs]
    return input_data


def load_conditional_generation_ref_model(model_repo, language, task):
    """
    Load Whisper model for conditional generation.

    Args:
        model_repo: HuggingFace model repository ID. Must be one of the supported models.
    """
    allowed_models = ["distil-whisper/distil-large-v3", "openai/whisper-large-v3"]
    if model_repo not in allowed_models:
        raise ValueError(f"Unknown model_repo: {model_repo}. Valid options are {allowed_models}")

    hf_ref_model = WhisperForConditionalGeneration.from_pretrained(model_repo).to(torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(model_repo, language=language, task=task)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_repo)
    config = hf_ref_model.config
    return (
        hf_ref_model,
        config,
        processor,
        feature_extractor,
    )


def init_conditional_generation_tt_model(
    hf_ref_model, config, mesh_device, weights_mesh_mapper, max_batch_size=WHISPER_BATCH_SIZE, max_seq_len=512
):
    model = hf_ref_model.model
    linear_weight = hf_ref_model.proj_out.weight
    ttnn_linear_weight = ttnn.from_torch(
        linear_weight, layout=ttnn.TILE_LAYOUT, device=mesh_device, dtype=ttnn.bfloat16, mesh_mapper=weights_mesh_mapper
    )
    ttnn_linear_weight = ttnn.permute(ttnn_linear_weight, (1, 0))
    ttnn_linear_weight = ttnn.to_layout(ttnn_linear_weight, layout=ttnn.TILE_LAYOUT)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=convert_to_ttnn,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )
    # Note: config.max_length is typically 448 for whisper large models
    kv_cache, cross_attn_cache = init_kv_cache(
        config, mesh_device, max_batch_size, max_seq_len=max_seq_len, weights_mesh_mapper=weights_mesh_mapper
    )

    return parameters, ttnn_linear_weight, kv_cache, cross_attn_cache


def create_functional_whisper_for_conditional_generation_inference_pipeline(
    mesh_device,
    model_repo,
    generation_params: Optional[GenerationParams] = None,
    batch_size_per_device=WHISPER_BATCH_SIZE,
):
    """
    Returns a callable with signature (data, sampling_rate, stream), where data is is a 1D numpy array
    and sampling_rate is an int representing the sampling rate used to acquire data, and stream turns
    signals the callable to return a generator if True, yielding the decoded tokens as they are processed, else
    the callable returns the full decoded output.

    Args:
        mesh_device: The target device
        model_repo: HuggingFace model repository ID. Must be one of the supported models.
        generation_params: Generation parameters for the model. If None, defaults will be used.
    """
    if generation_params is None:
        generation_params = GenerationParams()
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    hf_ref_model, config, processor, feature_extractor = load_conditional_generation_ref_model(
        model_repo, generation_params.language, generation_params.task
    )
    parameters, ttnn_linear_weight, kv_cache, cross_attn_cache = init_conditional_generation_tt_model(
        hf_ref_model, config, mesh_device, weights_mesh_mapper=weights_mesh_mapper, max_batch_size=batch_size_per_device
    )

    # Create WhisperGenerator instance with persistent trace support
    generator = WhisperGenerator(
        config=config,
        mesh_device=mesh_device,
        parameters=parameters,
        processor=processor,
        feature_extractor=feature_extractor,
        ttnn_linear_weight=ttnn_linear_weight,
        generation_config=hf_ref_model.generation_config,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
        kv_cache=kv_cache,
        cross_attn_cache=cross_attn_cache,
        max_batch_size=batch_size_per_device,
    )

    def _model_pipeline(
        current_batch,
        stream=False,
        return_perf_metrics=False,
        generation_params_override: Optional[GenerationParams] = None,
    ):
        # Use override if provided, otherwise use the original generation_params
        params = generation_params_override if generation_params_override is not None else generation_params

        durations = [audio_array.shape[0] / sampling_rate for (sampling_rate, audio_array) in current_batch]
        logger.info(
            f"Running model on batch of {len(current_batch)} samples with durations: {['{:.3f}s'.format(d) for d in durations]}"
        )

        return generator.generate(
            current_batch=current_batch,
            generation_params=params,
            stream_generation=stream,
            return_perf_metrics=return_perf_metrics,
        )

    return _model_pipeline


def run_demo_whisper_for_audio_classification_inference(
    input_path,
    mesh_device,
    num_inputs,
    batch_size_per_device=WHISPER_BATCH_SIZE,
    label=False,
    dataset=None,
):
    torch.manual_seed(1234)
    feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    model.eval()
    config = model.config
    if label:
        assert dataset is not None, "Dataset must be provided when label=True"
        data_iter = iter(dataset)
    else:
        input_data = load_input_paths(input_path)
    inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=convert_to_ttnn,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    total_inputs = num_inputs * batch_size

    if not label:
        input_data = repeat_inputs_cyclically(input_data, total_inputs)

    for i in tqdm(range(0, total_inputs, batch_size), desc="Running Inference"):
        current_batch_size = min(batch_size, total_inputs - i)
        logger.info(f"Running batch {(i // batch_size) + 1} | Batch size: {current_batch_size}")

        all_input_features = []
        true_labels = []

        for j in range(current_batch_size):
            if label:
                sample = next(data_iter)
                audio_array = sample["audio"]["array"]
                sampling_rate = sample["audio"]["sampling_rate"]
                true_labels.append(sample["lang_id"])
            else:
                input_file_path = input_data[i + j]
                logger.info(f"Input path: {input_file_path}")
                sampling_rate, audio_array = wavfile.read(input_file_path)

            # Feature extraction
            inputs = feature_extractor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
            )
            all_input_features.append(inputs.input_features)

        # Combine input features into batch tensor
        input_features = torch.cat(all_input_features, dim=0)  # Shape: [current_batch_size, x, y]
        del all_input_features
        # Encode inputs
        input_embedding = preprocess_encoder_inputs(
            config=config,
            input_features=input_features,
            parameters=parameters.encoder,
            device=mesh_device,
            input_mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
        )

        encoder_outputs = encoder(config=config, inputs_embeds=input_embedding, parameters=parameters.encoder)

        hidden_states = ttnn.matmul(encoder_outputs, parameters.projector.weight)
        hidden_states = ttnn.add(hidden_states, parameters.projector.bias)

        pooled_output = ttnn.mean(hidden_states, dim=-2, keepdim=True)

        logits = ttnn.matmul(pooled_output, parameters.classifier.weight)
        logits = ttnn.add(logits, parameters.classifier.bias)

        # Convert logits to torch
        logits_torch = ttnn.to_torch(logits, mesh_composer=output_mesh_composer)

        # Argmax over class dimension
        predicted_class_ids = torch.argmax(logits_torch.squeeze(1), dim=1)
        predicted_labels = [model.config.id2label[class_id.item()] for class_id in predicted_class_ids]

        for idx, label_str in enumerate(predicted_labels):
            log_msg = f"Sample {i + idx} - Predicted: {label_str}"
            if label:
                true_label_str = model.config.id2label[true_labels[idx]]
                log_msg += f" | True: {true_label_str}"
            logger.info(log_msg)


def run_demo_whisper_for_conditional_generation_inference(
    input_path,
    mesh_device,
    num_inputs,
    model_repo,
    generation_params: Optional[GenerationParams] = None,
    batch_size_per_device=WHISPER_BATCH_SIZE,
    stream=False,
):
    torch.manual_seed(0)
    # instantiate model inference pipeline
    model_pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
        mesh_device,
        model_repo,
        generation_params,
        batch_size_per_device=batch_size_per_device,
    )

    # load data
    input_data = load_input_paths(input_path)

    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    total_inputs = num_inputs * batch_size

    input_data = repeat_inputs_cyclically(input_data, total_inputs)

    total_ttft = 0
    total_decode_throughput = 0
    num_warmup_runs = 1
    for i in tqdm(range(0, total_inputs, batch_size), desc="Running Inference"):
        current_batch_size = min(batch_size, total_inputs - i)
        current_batch = []
        for j in range(current_batch_size):
            input_file_path = input_data[i + j]
            logger.info(f"Input path: {input_file_path}")
            samplerate, data = wavfile.read(input_file_path)
            current_batch.append((samplerate, data))

        # perform model inference
        if stream:
            # Handle streaming mode - iterate over generator
            logger.info(f"Streaming mode enabled for conditional generation inference")
            last_result = None
            for result in model_pipeline(current_batch, stream=True, return_perf_metrics=True):
                last_result = result

            # Extract final metrics from last result
            if last_result is not None:
                ttnn_output, avg_logprob, no_speech_prob, ttft, avg_decode_throughput, is_final = last_result
                print()  # New line after streaming
            else:
                # Fallback if no results
                ttnn_output, avg_logprob, no_speech_prob, ttft, avg_decode_throughput, is_final = (
                    [""] * current_batch_size,
                    None,
                    None,
                    0.0,
                    0.0,
                    False,
                )
        else:
            # Non-streaming mode
            ttnn_output, avg_logprob, no_speech_prob, ttft, avg_decode_throughput = model_pipeline(
                current_batch, stream=False, return_perf_metrics=True
            )

        if i >= num_warmup_runs:  # Exclude first compile run
            total_ttft += ttft
            total_decode_throughput += avg_decode_throughput
        batch_start = i + 1
        batch_end = i + current_batch_size
        logger.info(f"Model Output (Inputs {batch_start}--{batch_end}) Sample: {ttnn_output}")
    avg_ttft = total_ttft / (num_inputs - num_warmup_runs)
    avg_decode_throughput = total_decode_throughput / (num_inputs - num_warmup_runs)
    return avg_ttft, avg_decode_throughput


def run_demo_whisper_for_conditional_generation_dataset(
    mesh_device,
    model_repo,
    generation_params: Optional[GenerationParams] = None,
    batch_size_per_device=WHISPER_BATCH_SIZE,
    stream=False,
):
    torch.manual_seed(0)
    # instantiate model inference pipeline
    model_pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
        mesh_device,
        model_repo,
        generation_params,
        batch_size_per_device=batch_size_per_device,
    )

    # load data
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    # perform model inference
    total_wer = 0
    total_cer = 0
    total_inputs = len(ds)
    for i in tqdm(range(0, total_inputs, batch_size), desc="Running Inference"):
        current_batch_size = min(batch_size, total_inputs - i)
        current_batch = []
        reference_sentences = []
        if current_batch_size < batch_size:
            logger.info(f"Skipping last batch with size {current_batch_size}")
            continue
        for j in range(current_batch_size):
            sample = ds[i + j]
            logger.info(f"Sample ID: {i + j}")
            samplerate = sample["audio"]["sampling_rate"]
            data = sample["audio"]["array"]
            current_batch.append((samplerate, data))
            reference_sentences.append(sample["text"].lower())

        # Perform model inference with optional streaming
        if stream:
            # Handle streaming mode - iterate over generator
            logger.info(f"Streaming mode enabled for dataset evaluation")
            last_result = None
            for result in model_pipeline(current_batch, stream=True, return_perf_metrics=False):
                last_result = result
            # Extract final result
            if last_result is not None:
                ttnn_output, avg_logprob, no_speech_prob, is_final = last_result
            else:
                ttnn_output = [""] * current_batch_size
                avg_logprob = None
                no_speech_prob = None
                is_final = False
        else:
            # Non-streaming mode
            ttnn_output, avg_logprob, no_speech_prob = model_pipeline(
                current_batch,
                stream=False,
                return_perf_metrics=False,
            )
        batch_start = i + 1
        batch_end = i + current_batch_size
        logger.debug(f"Dataset text (Inputs {batch_start}--{batch_end}) Sample: {reference_sentences}")
        logger.debug(f"ttnn Model Output (Inputs {batch_start}--{batch_end}) Sample: {ttnn_output}")
        for j in range(current_batch_size):
            reference = ds[i + j]["text"].lower()
            # Handle both timestamp format (list of segments) and plain text format
            if isinstance(ttnn_output[j], list):
                predicted = " ".join([segment["text"] for segment in ttnn_output[j]]).lower()
            else:
                predicted = ttnn_output[j].lower()
            total_wer += jiwer.wer(reference, predicted)
            total_cer += jiwer.cer(reference, predicted)
    logger.info(f"Average Word Error Rate: {total_wer / len(ds):.4f}")
    logger.info(f"Average Character Error Rate: {total_cer / len(ds):.4f}")


def run_demo_whisper_for_translation_dataset(
    mesh_device,
    model_repo,
    num_inputs,
    generation_params: Optional[GenerationParams] = None,
    batch_size_per_device=WHISPER_BATCH_SIZE,
    stream=False,
):
    torch.manual_seed(0)

    if generation_params is None:
        generation_params = GenerationParams(
            temperatures=(0.0,),
            compression_ratio_threshold=2.4,
            logprob_threshold=-2.0,
            no_speech_threshold=0.6,
            return_timestamps=False,
            language="French",
            task="translate",
        )

    language_code_map = {
        "French": "fr_fr",
        "German": "de_de",
        "Spanish": "es_419",
        "Italian": "it_it",
        "Japanese": "ja_jp",
        "Korean": "ko_kr",
        "Hindi": "hi_in",
        "English": "en_us",
    }

    source_lang_code_full = language_code_map.get(generation_params.language, "fr_fr")  # Default to French
    logger.info(
        f"Setting up translation pipeline: source_language={generation_params.language} -> target_language=English"
    )
    logger.info(f"Using source language code: {source_lang_code_full} with task={generation_params.task}")

    model_pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
        mesh_device,
        model_repo,
        generation_params,
        batch_size_per_device=batch_size_per_device,
    )

    logger.info(f"Loading FLEURS dataset for {generation_params.language} (code: {source_lang_code_full})")

    # Load source language dataset
    ds = load_dataset("google/fleurs", source_lang_code_full, split="validation", streaming=True)

    # Load English dataset for reference translations WITHOUT streaming
    # This ensures we have all IDs available for mapping
    logger.info("Loading English dataset for reference translations (without streaming to build ID map)")
    english_ds = load_dataset("google/fleurs", "en_us", split="validation")

    # Initialize BLEU metric
    bleu = load("bleu")
    total_bleu = 0
    total_samples = 0

    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    total_inputs = num_inputs * batch_size

    logger.info(f"Testing translation from {generation_params.language} to English")
    logger.info(
        f"Processing {total_inputs} samples (batch_size={batch_size}: {batch_size_per_device} per device x {mesh_device.get_num_devices()} devices)"
    )

    # Collect samples from source language
    samples_collected = []
    for i, sample in enumerate(ds):
        if len(samples_collected) >= total_inputs:
            break
        samples_collected.append(sample)

    # Create a dictionary mapping from id to English transcription
    # Using the full English dataset ensures all IDs are available
    logger.info(f"Creating English reference mapping by ID from {len(english_ds)} samples")
    english_map = {item["id"]: item["transcription"] for item in english_ds}

    for i in tqdm(range(0, len(samples_collected), batch_size), desc="Running Translation Test"):
        current_batch_size = min(batch_size, len(samples_collected) - i)

        if current_batch_size < batch_size:
            logger.info(f"Skipping last batch with size {current_batch_size} (expected {batch_size})")
            continue

        current_batch = []
        reference_sentences = []

        for j in range(current_batch_size):
            sample = samples_collected[i + j]
            samplerate = sample["audio"]["sampling_rate"]
            data = sample["audio"]["array"]
            current_batch.append((samplerate, data))

            # Get English translation using ID mapping
            source_text = sample["transcription"]
            english_translation = english_map[sample["id"]]
            reference_sentences.append(english_translation)

            logger.info(f"Sample {i + j + 1}: {generation_params.language} text: {source_text}")
            logger.info(f"Sample {i + j + 1}: English reference: {english_translation}")

        # Perform model inference with optional streaming
        if stream:
            # Handle streaming mode - iterate over generator
            logger.info(f"Streaming mode enabled for translation evaluation")
            last_result = None
            for result in model_pipeline(current_batch, stream=True, return_perf_metrics=False):
                last_result = result
            # Extract final result
            if last_result is not None:
                ttnn_output, avg_logprob, no_speech_prob, is_final = last_result
            else:
                ttnn_output = [""] * current_batch_size
                avg_logprob = None
                no_speech_prob = None
                is_final = False
        else:
            # Non-streaming mode
            ttnn_output, avg_logprob, no_speech_prob = model_pipeline(
                current_batch,
                stream=False,
                return_perf_metrics=False,
            )

        # Process results for each sample in the batch
        for j in range(current_batch_size):
            sample_idx = i + j + 1
            logger.info(f"Sample {sample_idx}: Translated output: {ttnn_output[j]}")

            reference = reference_sentences[j]
            # Handle both timestamp format (list of segments) and plain text format
            if isinstance(ttnn_output[j], list):
                predicted = " ".join([segment["text"] for segment in ttnn_output[j]])
            else:
                predicted = ttnn_output[j]

            # Normalize: lowercase and strip punctuation for better matching
            reference_normalized = reference.lower().strip()
            predicted_normalized = predicted.lower().strip()

            # evaluate library expects predictions as list and references as list of lists
            bleu_result = bleu.compute(
                predictions=[predicted_normalized],
                references=[[reference_normalized]],
                smooth=True,  # Apply smoothing to prevent 0 scores for partial matches
            )
            bleu_score = bleu_result["bleu"] * 100

            total_bleu += bleu_score
            total_samples += 1

            logger.info(f"Sample {sample_idx}: BLEU: {bleu_score:.2f}")

    if total_samples > 0:
        avg_bleu = total_bleu / total_samples
        logger.info(f"Translation Test Results:")
        logger.info(f"Average BLEU Score: {avg_bleu:.2f}")
        logger.info(f"Total samples processed: {total_samples}")
    else:
        logger.warning("No samples were processed for translation test")


@pytest.mark.parametrize(
    "num_inputs,batch_size_per_device",
    [(1, WHISPER_BATCH_SIZE)],
)
@pytest.mark.parametrize(
    "input_path",
    (["models/demos/whisper/demo/dataset/audio_classification"]),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "mesh_device",
    [available_devices]
    if os.getenv("CI") != "true"
    else ([1, available_devices] if available_devices != 1 else [available_devices]),
    indirect=True,
)
# To run the demo with specific device configurations, provide the desired number of devices under the `mesh_device` parameter.
def test_demo_for_audio_classification_inference(
    input_path, mesh_device, num_inputs, batch_size_per_device, is_ci_env, request
):
    if is_ci_env:
        pytest.skip("Skipping test in CI since it provides redundant testing")

    return run_demo_whisper_for_audio_classification_inference(
        input_path,
        mesh_device,
        num_inputs,
        batch_size_per_device,
    )


@pytest.mark.parametrize(
    "num_inputs,batch_size_per_device",
    [(1, WHISPER_BATCH_SIZE)],
)
# To run the demo with specific device configurations, provide the desired number of devices under the `mesh_device` parameter.
@pytest.mark.parametrize(
    "mesh_device",
    [available_devices]
    if os.getenv("CI") != "true"
    else ([1, available_devices] if available_devices != 1 else [available_devices]),
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_demo_for_audio_classification_dataset(
    input_path, mesh_device, num_inputs, batch_size_per_device, is_ci_env, request
):
    if is_ci_env:
        pytest.skip("Skipping test in CI since it provides redundant testing")

    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    return run_demo_whisper_for_audio_classification_inference(
        input_path,
        mesh_device,
        num_inputs,
        batch_size_per_device,
        label=True,
        dataset=ds,
    )


@pytest.mark.parametrize(
    "num_inputs",
    [2],
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    [1, 2],
)
@pytest.mark.parametrize(
    "model_repo",
    ("openai/whisper-large-v3", "distil-whisper/distil-large-v3"),
)
@pytest.mark.parametrize(
    "mesh_device",
    [available_devices]
    if os.getenv("CI") != "true"
    else ([1, available_devices] if available_devices != 1 else [available_devices]),
    indirect=True,
)
@pytest.mark.parametrize(
    "input_path",
    (["models/demos/whisper/demo/dataset/conditional_generation"]),
)
@pytest.mark.parametrize(
    "language",
    ("English",),
)
@pytest.mark.parametrize(
    "task",
    ("transcribe",),
)
@pytest.mark.parametrize(
    "temperatures,compression_ratio_threshold,logprob_threshold,no_speech_threshold,return_timestamps",
    [
        (0.0, None, None, None, False),
        ((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), 2.4, -1.0, 0.6, True),  # generation with generate_kwargs
    ],
)
@pytest.mark.parametrize(
    "stream",
    [False],
)
@pytest.mark.parametrize(
    "prompt",
    [None],
)
# To run the demo with specific device configurations, provide the desired number of devices under the `mesh_device` parameter.
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": WHISPER_L1_SMALL_SIZE, "trace_region_size": WHISPER_TRACE_REGION_SIZE}],
    indirect=True,
)
def test_demo_for_conditional_generation(
    input_path,
    mesh_device,
    num_inputs,
    model_repo,
    language,
    task,
    is_ci_env,
    temperatures,
    compression_ratio_threshold,
    logprob_threshold,
    no_speech_threshold,
    return_timestamps,
    batch_size_per_device,
    stream,
    prompt,
    request,
):
    # Skip test in CI when using generate_kwargs
    if (
        is_ci_env
        and model_repo == "openai/whisper-large-v3"
        and (compression_ratio_threshold is not None or batch_size_per_device == 2)
    ):
        pytest.skip("Skipping test in CI since it provides redundant testing")

    generation_params = GenerationParams(
        temperatures=temperatures,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        return_timestamps=return_timestamps,
        language=language,
        task=task,
        prompt=prompt,
    )
    ttft, decode_throughput = run_demo_whisper_for_conditional_generation_inference(
        input_path,
        mesh_device,
        num_inputs,
        model_repo,
        generation_params,
        batch_size_per_device,
        stream=stream,
    )

    if (
        is_ci_env
        and model_repo == "distil-whisper/distil-large-v3"
        and batch_size_per_device == 1
        and mesh_device.get_num_devices() == available_devices
        and compression_ratio_threshold is None  # Check perf only when generate_kwargs are None
    ):
        metrics_dictionary = {
            2: {"prefill_time_to_token": 0.13, "decode_t/s/u": 124.0},
            8: {"prefill_time_to_token": 0.14, "decode_t/s/u": 105.0},
            32: {"prefill_time_to_token": 0.21, "decode_t/s/u": 80.0},
        }
        if is_blackhole():
            if mesh_device.dram_grid_size().x == 7:  # P100 DRAM grid is 7x1
                expected_perf_metrics = {"prefill_time_to_token": 0.06, "decode_t/s/u": 310.0}
            else:
                expected_perf_metrics = {"prefill_time_to_token": 0.05, "decode_t/s/u": 330.0}
        else:  # wormhole_b0
            expected_perf_metrics = metrics_dictionary[mesh_device.get_num_devices()]
        total_batch = mesh_device.get_num_devices() * batch_size_per_device
        expected_perf_metrics["decode_t/s"] = expected_perf_metrics["decode_t/s/u"] * total_batch
        measurements = {
            "prefill_time_to_token": ttft,
            "decode_t/s": decode_throughput * total_batch,
            "decode_t/s/u": decode_throughput,
        }
        expected_measurements = {
            "prefill_time_to_token": True,
            "decode_t/s": True,
            "decode_t/s/u": True,
        }
        verify_perf(
            measurements, expected_perf_metrics, high_tol_percentage=1.20, expected_measurements=expected_measurements
        )


@pytest.mark.parametrize(
    "model_repo",
    ("openai/whisper-large-v3", "distil-whisper/distil-large-v3"),
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": WHISPER_L1_SMALL_SIZE, "trace_region_size": WHISPER_TRACE_REGION_SIZE}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device",
    [(WHISPER_BATCH_SIZE)],
)
@pytest.mark.parametrize(
    "mesh_device",
    [available_devices]
    if os.getenv("CI") != "true"
    else ([1, available_devices] if available_devices != 1 else [available_devices]),
    indirect=True,
)
@pytest.mark.parametrize(
    "language",
    ("English",),
)
@pytest.mark.parametrize(
    "task",
    ("transcribe",),
)
@pytest.mark.parametrize(
    "temperatures,compression_ratio_threshold,logprob_threshold,no_speech_threshold,return_timestamps",
    [
        (0.0, None, None, None, False),
        ((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), 2.4, -1.0, 0.6, True),  # generation with generate_kwargs
    ],
)
@pytest.mark.parametrize(
    "stream",
    [False],
)
@pytest.mark.parametrize(
    "prompt",
    [
        'Here are several example lines using “Mister”: Good morning. This is Mister John Smith speaking. Mister Smith will join us shortly and Mister Jones is already here. I asked Mister Anderson if Mister Brown could review the file. From here on, whenever the speaker says the name "Mister …", use "Mister" (not "Mr.") in the transcription.'
    ],
)
# To run the demo with specific device configurations, provide the desired number of devices under the `mesh_device` parameter.
def test_demo_for_conditional_generation_dataset(
    mesh_device,
    model_repo,
    language,
    task,
    is_ci_env,
    temperatures,
    compression_ratio_threshold,
    logprob_threshold,
    no_speech_threshold,
    return_timestamps,
    batch_size_per_device,
    stream,
    prompt,
    request,
):
    # Skip test in CI when using generate_kwargs
    if is_ci_env and model_repo == "openai/whisper-large-v3" and compression_ratio_threshold is not None:
        pytest.skip("Skipping test in CI since it provides redundant testing")

    generation_params = GenerationParams(
        temperatures=temperatures,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        return_timestamps=return_timestamps,
        language=language,
        task=task,
        prompt=prompt,
    )
    return run_demo_whisper_for_conditional_generation_dataset(
        mesh_device,
        model_repo,
        generation_params,
        batch_size_per_device,
        stream=stream,
    )


@pytest.mark.parametrize(
    "model_repo",
    ("openai/whisper-large-v3",),
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": WHISPER_L1_SMALL_SIZE, "trace_region_size": WHISPER_TRACE_REGION_SIZE}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [available_devices]
    if os.getenv("CI") != "true"
    else ([1, available_devices] if available_devices != 1 else [available_devices]),
    indirect=True,
)
@pytest.mark.parametrize(
    "source_language",
    ("French",),
)
@pytest.mark.parametrize(
    "num_inputs,batch_size_per_device",
    [(1, WHISPER_BATCH_SIZE)],
)
@pytest.mark.parametrize(
    "temperatures,compression_ratio_threshold,logprob_threshold,no_speech_threshold,return_timestamps",
    [(0.0, None, None, None, False), (0.0, 2.4, -2.0, 0.6, True)],  # Translation needs relaxed thresholds
)
@pytest.mark.parametrize(
    "stream",
    [True],
)
def test_demo_for_translation_dataset(
    mesh_device,
    model_repo,
    source_language,
    is_ci_env,
    num_inputs,
    batch_size_per_device,
    temperatures,
    compression_ratio_threshold,
    logprob_threshold,
    no_speech_threshold,
    return_timestamps,
    stream,
    request,
):
    if is_ci_env:
        pytest.skip("Skipping test in CI since it provides redundant testing")

    generation_params = GenerationParams(
        temperatures=temperatures,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        return_timestamps=return_timestamps,
        language=source_language,
        task="translate",
    )
    return run_demo_whisper_for_translation_dataset(
        mesh_device,
        model_repo,
        num_inputs,
        generation_params,
        batch_size_per_device,
        stream=stream,
    )
