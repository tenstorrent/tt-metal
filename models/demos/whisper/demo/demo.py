# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
from os import listdir
from os.path import isfile, join

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
from models.demos.whisper.tt.ttnn_whisper import WHISPER_L1_SMALL_SIZE, TtnnWhisper

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
    hf_ref_model, config, ttnn_whisper_instance, mesh_device, weights_mesh_mapper, max_batch_size=1, max_seq_len=512
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
        convert_to_ttnn=ttnn_whisper_instance.convert_to_ttnn,
        custom_preprocessor=ttnn_whisper_instance.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )
    # Note: config.max_length is typically 448 for whisper large models
    kv_cache = TtnnWhisper.init_kv_cache(
        config, mesh_device, max_batch_size, max_seq_len=max_seq_len, weights_mesh_mapper=weights_mesh_mapper
    )

    return parameters, ttnn_linear_weight, kv_cache


def run_generate(
    config,
    current_batch,
    feature_extractor,
    ttnn_whisper_instance,
    parameters,
    processor,
    ttnn_linear_weight,
    mesh_device,
    generation_config,
    input_mesh_mapper,
    output_mesh_composer,
    weights_mesh_mapper,
    kv_cache=None,
    stream_generation=False,
    feature_dtype_to_use=torch.bfloat16,
    return_perf_metrics=False,
    temperatures=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    compression_ratio_threshold=2.4,
    logprob_threshold=-2.0,
    no_speech_threshold=0.6,
    return_timestamps=False,
    language="en",
    task="transcribe",
):
    return ttnn_whisper_instance.generate(
        config=config,
        current_batch=current_batch,
        feature_extractor=feature_extractor,
        parameters=parameters,
        processor=processor,
        ttnn_linear_weight=ttnn_linear_weight,
        mesh_device=mesh_device,
        generation_config=generation_config,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
        weights_mesh_mapper=weights_mesh_mapper,
        kv_cache=kv_cache,
        temperatures=temperatures,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        return_timestamps=return_timestamps,
        stream_generation=stream_generation,
        return_perf_metrics=return_perf_metrics,
        language=language,
        task=task,
    )


def create_functional_whisper_for_conditional_generation_inference_pipeline(
    ttnn_whisper_instance,
    mesh_device,
    model_repo,
    language,
    task,
    temperatures,
    compression_ratio_threshold,
    logprob_threshold,
    no_speech_threshold,
    return_timestamps=False,
):
    """
    Returns a callable with signature (data, sampling_rate, stream), where data is is a 1D numpy array
    and sampling_rate is an int representing the sampling rate used to acquire data, and stream turns
    signals the callable to return a generator if True, yielding the decoded tokens as they are processed, else
    the callable returns the full decoded output.

    Args:
        ttnn_whisper_instance: The TtnnWhisper instance
        device: The target device
        model_repo: HuggingFace model repository ID. Must be one of the supported models.
    """
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    hf_ref_model, config, processor, feature_extractor = load_conditional_generation_ref_model(
        model_repo, language, task
    )
    parameters, ttnn_linear_weight, kv_cache = init_conditional_generation_tt_model(
        hf_ref_model, config, ttnn_whisper_instance, mesh_device, weights_mesh_mapper=weights_mesh_mapper
    )

    def _model_pipeline(
        current_batch,
        stream=False,
        return_perf_metrics=False,
        temperatures=temperatures,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        return_timestamps=return_timestamps,
    ):
        durations = [audio_array.shape[0] / sampling_rate for (sampling_rate, audio_array) in current_batch]
        logger.info(
            f"Running model on batch of {len(current_batch)} samples with durations: {['{:.3f}s'.format(d) for d in durations]}"
        )

        return run_generate(
            config,
            current_batch,
            feature_extractor,
            ttnn_whisper_instance,
            parameters=parameters,
            processor=processor,
            ttnn_linear_weight=ttnn_linear_weight,
            mesh_device=mesh_device,
            generation_config=hf_ref_model.generation_config,
            kv_cache=kv_cache,
            stream_generation=stream,
            return_perf_metrics=return_perf_metrics,
            input_mesh_mapper=input_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
            weights_mesh_mapper=weights_mesh_mapper,
            temperatures=temperatures,
            compression_ratio_threshold=compression_ratio_threshold,
            logprob_threshold=logprob_threshold,
            no_speech_threshold=no_speech_threshold,
            return_timestamps=return_timestamps,
            language=language,
            task=task,
        )

    return _model_pipeline


def run_demo_whisper_for_audio_classification_inference(
    input_path,
    ttnn_whisper_instance,
    mesh_device,
    num_inputs,
    batch_size_per_device=1,
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
        convert_to_ttnn=ttnn_whisper_instance.convert_to_ttnn,
        custom_preprocessor=ttnn_whisper_instance.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )
    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    total_inputs = num_inputs * batch_size

    if not label and len(input_data) < total_inputs:
        # Repeat inputs cyclically to match total_inputs
        logger.info(
            f"Only {len(input_data)} audio files available, repeating cyclically to match {total_inputs} total inputs"
        )
        original_input_data = input_data.copy()
        while len(input_data) < total_inputs:
            input_data.extend(original_input_data)
        # Trim to exact size needed
        input_data = input_data[:total_inputs]

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
        input_embedding = ttnn_whisper_instance.preprocess_encoder_inputs(
            config=config,
            input_features=input_features,
            parameters=parameters.encoder,
            device=mesh_device,
            input_mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
        )

        encoder_outputs = ttnn_whisper_instance.encoder(
            config=config, inputs_embeds=input_embedding, parameters=parameters.encoder
        )

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
    ttnn_whisper_instance,
    mesh_device,
    num_inputs,
    model_repo,
    language,
    task,
    temperatures,
    compression_ratio_threshold,
    logprob_threshold,
    no_speech_threshold,
    return_timestamps,
    batch_size_per_device=1,
):
    torch.manual_seed(0)
    # instantiate model inference pipeline
    model_pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
        ttnn_whisper_instance,
        mesh_device,
        model_repo,
        language,
        task,
        temperatures,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        return_timestamps,
    )

    # load data
    input_data = load_input_paths(input_path)

    batch_size = batch_size_per_device * mesh_device.get_num_devices()
    total_inputs = num_inputs * batch_size

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
    ttnn_whisper_instance,
    mesh_device,
    model_repo,
    language,
    task,
    temperatures,
    compression_ratio_threshold,
    logprob_threshold,
    no_speech_threshold,
    return_timestamps,
    batch_size_per_device=1,
):
    torch.manual_seed(0)
    # instantiate model inference pipeline
    model_pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
        ttnn_whisper_instance,
        mesh_device,
        model_repo,
        language,
        task,
        temperatures,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        return_timestamps,
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
    ttnn_whisper_instance,
    mesh_device,
    model_repo,
    source_language,
    num_inputs,
    temperatures,
    compression_ratio_threshold,
    logprob_threshold,
    no_speech_threshold,
    return_timestamps,
    batch_size_per_device=1,
):
    torch.manual_seed(0)

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

    source_lang_code_full = language_code_map.get(source_language, "fr_fr")  # Default to French
    logger.info(f"Setting up translation pipeline: source_language={source_language} -> target_language=English")
    logger.info(f"Using source language code: {source_lang_code_full} with task='translate'")

    model_pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
        ttnn_whisper_instance,
        mesh_device,
        model_repo,
        source_language,
        "translate",
        temperatures,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        return_timestamps,
    )

    logger.info(f"Loading FLEURS dataset for {source_language} (code: {source_lang_code_full})")

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

    logger.info(f"Testing translation from {source_language} to English")
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

            logger.info(f"Sample {i + j + 1}: {source_language} text: {source_text}")
            logger.info(f"Sample {i + j + 1}: English reference: {english_translation}")

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
    "ttnn_whisper_instance",
    (TtnnWhisper(),),
)
@pytest.mark.parametrize(
    "num_inputs,batch_size_per_device",
    [(1, 1)],
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
    input_path, ttnn_whisper_instance, mesh_device, num_inputs, batch_size_per_device, is_ci_env, request
):
    if is_ci_env:
        pytest.skip("Skipping test in CI since it provides redundant testing")

    return run_demo_whisper_for_audio_classification_inference(
        input_path,
        ttnn_whisper_instance,
        mesh_device,
        num_inputs,
        batch_size_per_device,
    )


@pytest.mark.parametrize(
    "ttnn_whisper_instance",
    (TtnnWhisper(),),
)
@pytest.mark.parametrize(
    "num_inputs,batch_size_per_device",
    [(1, 1)],
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
    input_path, ttnn_whisper_instance, mesh_device, num_inputs, batch_size_per_device, is_ci_env, request
):
    if is_ci_env:
        pytest.skip("Skipping test in CI since it provides redundant testing")

    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    return run_demo_whisper_for_audio_classification_inference(
        input_path,
        ttnn_whisper_instance,
        mesh_device,
        num_inputs,
        batch_size_per_device,
        label=True,
        dataset=ds,
    )


@pytest.mark.parametrize(
    "ttnn_whisper_instance",
    (TtnnWhisper(),),
)
@pytest.mark.parametrize(
    "num_inputs,batch_size_per_device",
    [(2, 1)],
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
# To run the demo with specific device configurations, provide the desired number of devices under the `mesh_device` parameter.
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_demo_for_conditional_generation(
    input_path,
    ttnn_whisper_instance,
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
    request,
):
    ttft, decode_throughput = run_demo_whisper_for_conditional_generation_inference(
        input_path,
        ttnn_whisper_instance,
        mesh_device,
        num_inputs,
        model_repo,
        language,
        task,
        temperatures,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        return_timestamps,
        batch_size_per_device,
    )

    if (
        is_ci_env
        and model_repo == "distil-whisper/distil-large-v3"
        and mesh_device.get_num_devices() == available_devices
    ):
        metrics_dictionary = {
            1: {"prefill_time_to_token": 0.24, "decode_t/s/u": 53.2},
            2: {"prefill_time_to_token": 0.27, "decode_t/s/u": 51.1},
            8: {"prefill_time_to_token": 0.28, "decode_t/s/u": 42.1},
            32: {"prefill_time_to_token": 0.35, "decode_t/s/u": 43.1},
        }
        if is_blackhole():
            if mesh_device.dram_grid_size().x == 7:  # P100 DRAM grid is 7x1
                expected_perf_metrics = {"prefill_time_to_token": 0.127, "decode_t/s/u": 87.0}
            else:
                expected_perf_metrics = {"prefill_time_to_token": 0.119, "decode_t/s/u": 94.0}
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
        verify_perf(measurements, expected_perf_metrics, expected_measurements=expected_measurements)


@pytest.mark.parametrize(
    "ttnn_whisper_instance",
    (TtnnWhisper(),),
)
@pytest.mark.parametrize(
    "model_repo",
    ("openai/whisper-large-v3", "distil-whisper/distil-large-v3"),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "batch_size_per_device",
    [(1)],
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
# To run the demo with specific device configurations, provide the desired number of devices under the `mesh_device` parameter.
def test_demo_for_conditional_generation_dataset(
    ttnn_whisper_instance,
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
    request,
):
    if is_ci_env:
        pytest.skip("Skipping test in CI since it provides redundant testing")

    return run_demo_whisper_for_conditional_generation_dataset(
        ttnn_whisper_instance,
        mesh_device,
        model_repo,
        language,
        task,
        temperatures,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        return_timestamps,
        batch_size_per_device,
    )


@pytest.mark.parametrize(
    "ttnn_whisper_instance",
    (TtnnWhisper(),),
)
@pytest.mark.parametrize(
    "model_repo",
    ("openai/whisper-large-v3",),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
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
    [(1, 1)],
)
@pytest.mark.parametrize(
    "temperatures,compression_ratio_threshold,logprob_threshold,no_speech_threshold,return_timestamps",
    [(0.0, None, None, None, False), (0.0, 2.4, -2.0, 0.6, True)],  # Translation needs relaxed thresholds
)
def test_demo_for_translation_dataset(
    ttnn_whisper_instance,
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
    request,
):
    if is_ci_env:
        pytest.skip("Skipping test in CI since it provides redundant testing")

    return run_demo_whisper_for_translation_dataset(
        ttnn_whisper_instance,
        mesh_device,
        model_repo,
        source_language,
        num_inputs,
        temperatures,
        compression_ratio_threshold,
        logprob_threshold,
        no_speech_threshold,
        return_timestamps,
        batch_size_per_device,
    )
