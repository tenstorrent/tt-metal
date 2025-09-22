# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import time
from os import listdir
from os.path import isfile, join

import jiwer
import pytest
import torch
from datasets import load_dataset
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
from models.common.generation_utils import get_logits_processor
from models.common.utility_functions import is_blackhole
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.utils.llm_demo_utils import verify_perf
from models.demos.whisper.tt import ttnn_optimized_functional_whisper
from models.demos.whisper.tt.ttnn_optimized_functional_whisper import WHISPER_L1_SMALL_SIZE, init_kv_cache


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


def load_conditional_generation_ref_model(model_repo):
    """
    Load Whisper model for conditional generation.

    Args:
        model_repo: HuggingFace model repository ID. Must be one of the supported models.
    """
    allowed_models = ["distil-whisper/distil-large-v3", "openai/whisper-large-v3"]
    if model_repo not in allowed_models:
        raise ValueError(f"Unknown model_repo: {model_repo}. Valid options are {allowed_models}")

    hf_ref_model = WhisperForConditionalGeneration.from_pretrained(model_repo).to(torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(model_repo, language="English", task="transcribe")
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_repo)
    config = hf_ref_model.config
    return (
        hf_ref_model,
        config,
        processor,
        feature_extractor,
    )


def init_conditional_generation_tt_model(
    hf_ref_model, config, ttnn_model, mesh_device, weights_mesh_mapper, max_batch_size=1, max_seq_len=512
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
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )
    # Note: config.max_length is typically 448 for whisper large models
    kv_cache = init_kv_cache(
        config, mesh_device, max_batch_size, max_seq_len=max_seq_len, weights_mesh_mapper=weights_mesh_mapper
    )

    return parameters, ttnn_linear_weight, kv_cache


def run_generate(
    config,
    current_batch,
    feature_extractor,
    ttnn_model,
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
    num_devices=2,
):
    all_input_features = []
    start_encode = time.time()
    for sampling_rate, audio_array in current_batch:
        inputs = feature_extractor(
            audio_array,
            sampling_rate=sampling_rate,
            return_tensors="pt",
        )
        all_input_features.append(inputs.input_features)
    input_features = torch.cat(all_input_features, dim=0)  # [B, x, y]
    del all_input_features
    unpadded_batch_size = input_features.shape[0]
    assert unpadded_batch_size == 1 * num_devices, "Only batch size (per device) 1 is supported for inference"
    # Compute embeddings
    input_embeds = ttnn_model.preprocess_encoder_inputs(
        config,
        input_features,
        parameters=parameters.encoder,
        device=mesh_device,
        weights_mesh_mapper=weights_mesh_mapper,
        input_mesh_mapper=input_mesh_mapper,
    )
    # Run encoder
    encoder_hidden_states = ttnn_model.encoder(config, input_embeds, parameters=parameters.encoder)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"Time to encoder states: {(time.time() - start_encode)*1000:.3f}ms")

    # Run decoder

    def _run_generate():
        # Input ids
        input_ids = torch.tensor([[1]]) * config.decoder_start_token_id
        input_ids = input_ids.repeat(input_features.shape[0], 1)
        logits_processor = get_logits_processor(input_ids, config)
        if not kv_cache:
            input_ids = pad_input_32(input_ids, config.pad_token_id).to(torch.long)
            decoder_start_values = generation_config.pad_token_id * torch.ones(1, 32).to(torch.long)
        # Initial decode position
        current_decode_pos = (
            ttnn.from_torch(
                torch.zeros(unpadded_batch_size), device=mesh_device, dtype=ttnn.int32, mesh_mapper=input_mesh_mapper
            )
            if kv_cache
            else None
        )
        MAX_GEN_LEN = config.max_length  # typically 448 for whisper large models
        print_each_iter = False
        output_ids = []
        total_decode_time = 0
        prompt_is_done = [False for _ in range(unpadded_batch_size)]
        for i in tqdm(range(MAX_GEN_LEN), desc="Decode inference iterations"):
            start_iter = time.time()
            decoder_hidden_states, decoder_attention_mask = ttnn_model.preprocess_decoder_inputs(
                config=config,
                input_ids=input_ids,
                attention_mask=None,
                parameters=parameters.decoder,
                device=mesh_device,
                decode_pos=i if kv_cache else None,
                create_attention_mask=(not kv_cache),
                input_mesh_mapper=input_mesh_mapper,
            )

            output = ttnn_model.decoder(
                config,
                decoder_hidden_states,
                decoder_attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                kv_cache=kv_cache,
                current_decode_pos=current_decode_pos,
                parameters=parameters.decoder,
            )

            if not kv_cache:
                # Note: if not using a kv cache, the entire sequence is recomputed at each step
                # Only run the lm head on the last tile to fix bad outputs and reduce redundant computation
                last_tile_start_idx = i // 32 * 32
                output_idx = i % 32
                output = output[:, last_tile_start_idx : last_tile_start_idx + 32, :]
            else:
                output_idx = 0

            output = output @ ttnn_linear_weight
            logits_to_torch = ttnn.to_torch(output, mesh_composer=output_mesh_composer)
            next_token_logits = logits_to_torch[:, output_idx, :]
            next_tokens_scores = logits_processor(input_features, next_token_logits)
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            output_ids.append(next_tokens)

            if i == 0:
                first_token_time = time.time()
                ttft = first_token_time - start_encode

            # Update input_ids and current_decode_pos
            if not kv_cache:
                if (i + 1) % 32 == 0:
                    input_ids = torch.cat([input_ids, decoder_start_values], dim=1)
                input_ids[:, i + 1] = next_tokens[:, None]
            else:
                input_ids = next_tokens[:, None]
                ttnn.plus_one(current_decode_pos)

            total_decode_time += time.time() - start_iter
            avg_decode_throughput = (i + 1) / total_decode_time
            for user_id, user_decode_id in enumerate(next_tokens[:unpadded_batch_size]):
                if user_decode_id == config.eos_token_id:
                    prompt_is_done[user_id] = True
                if prompt_is_done[user_id]:
                    next_tokens[user_id] = config.eos_token_id
            ttnn_transcription = processor.batch_decode(next_tokens.unsqueeze(dim=1), skip_special_tokens=True)
            if print_each_iter:
                logger.info(processor.batch_decode(torch.stack(output_ids, dim=1), skip_special_tokens=True))

            if return_perf_metrics:
                yield ttnn_transcription, ttft, avg_decode_throughput
            else:
                yield ttnn_transcription

            if all(prompt_is_done):
                break
        total_generate_time = time.time() - start_encode
        logger.info(f"Time to first token: {(ttft*1000):.3f}ms")
        logger.info(f"Total decode time: {total_decode_time:.3f}s")
        logger.info(f"Total generate time: {total_generate_time:.3f}s")
        logger.info(f"Average decode throughput (per user): {avg_decode_throughput:.3f} t/s/u")
        logger.info(f"Average decode throughput (total batch): {(avg_decode_throughput * unpadded_batch_size):.3f} t/s")

    # conditionally return generator or full response
    if stream_generation:
        return _run_generate()
    else:
        output = [[] for _ in range(input_features.shape[0])]
        for x in _run_generate():
            if return_perf_metrics:
                out_cur, ttft, avg_decode_throughput = x
            else:
                out_cur = x
            for idx in range(input_features.shape[0]):
                output[idx].append(out_cur[idx])
        output = ["".join(tokens) for tokens in output]
        if return_perf_metrics:
            return output, ttft, avg_decode_throughput
        else:
            return output


def create_functional_whisper_for_conditional_generation_inference_pipeline(
    ttnn_model, mesh_device, model_repo, input_mesh_mapper, output_mesh_composer, weights_mesh_mapper, num_devices=2
):
    """
    Returns a callable with signature (data, sampling_rate, stream), where data is is a 1D numpy array
    and sampling_rate is an int representing the sampling rate used to acquire data, and stream turns
    signals the callable to return a generator if True, yielding the decoded tokens as they are processed, else
    the callable returns the full decoded output.

    Args:
        ttnn_model: The TTNN model
        device: The target device
        model_repo: HuggingFace model repository ID. Must be one of the supported models.
    """
    hf_ref_model, config, processor, feature_extractor = load_conditional_generation_ref_model(model_repo)
    parameters, ttnn_linear_weight, kv_cache = init_conditional_generation_tt_model(
        hf_ref_model, config, ttnn_model, mesh_device, weights_mesh_mapper=weights_mesh_mapper
    )

    def _model_pipeline(
        current_batch,
        input_mesh_mapper,
        output_mesh_composer,
        weights_mesh_mapper,
        stream=False,
        return_perf_metrics=False,
    ):
        durations = [audio_array.shape[0] / sampling_rate for (sampling_rate, audio_array) in current_batch]
        logger.info(
            f"Running model on batch of {len(current_batch)} samples with durations: {['{:.3f}s'.format(d) for d in durations]}"
        )

        return run_generate(
            config,
            current_batch,
            feature_extractor,
            ttnn_model,
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
            num_devices=num_devices,
        )

    return _model_pipeline


def run_demo_whisper_for_audio_classification_inference(
    input_path,
    ttnn_model,
    mesh_device,
    num_inputs,
    num_devices=2,
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
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=mesh_device,
    )
    batch_size = batch_size_per_device * num_devices

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
        input_embedding = ttnn_model.preprocess_encoder_inputs(
            config=config,
            input_features=input_features,
            parameters=parameters.encoder,
            device=mesh_device,
            input_mesh_mapper=inputs_mesh_mapper,
            weights_mesh_mapper=weights_mesh_mapper,
        )

        encoder_outputs = ttnn_model.encoder(
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
    input_path, ttnn_model, mesh_device, num_inputs, model_repo, batch_size_per_device=1, num_devices=2
):
    torch.manual_seed(0)
    # instantiate model inference pipeline
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    model_pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
        ttnn_model,
        mesh_device,
        model_repo,
        weights_mesh_mapper=weights_mesh_mapper,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
        num_devices=num_devices,
    )

    # load data
    input_data = load_input_paths(input_path)

    batch_size = batch_size_per_device * num_devices
    logger.info(f"num_inputs: {num_inputs}, batch_size: {batch_size}")
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
        ttnn_output, ttft, avg_decode_throughput = model_pipeline(
            current_batch,
            stream=False,
            return_perf_metrics=True,
            input_mesh_mapper=input_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
            weights_mesh_mapper=weights_mesh_mapper,
        )
        if i >= num_warmup_runs:  # Exclude first compile run
            total_ttft += ttft
            total_decode_throughput += avg_decode_throughput
        batch_start = i + 1
        batch_end = i + current_batch_size
        logger.debug(f"Model Output (Inputs {batch_start}--{batch_end}) Sample: {ttnn_output}")
    avg_ttft = total_ttft / (num_inputs - num_warmup_runs)
    avg_decode_throughput = total_decode_throughput / (num_inputs - num_warmup_runs)
    return avg_ttft, avg_decode_throughput


def run_demo_whisper_for_conditional_generation_dataset(
    ttnn_model, mesh_device, model_repo, batch_size_per_device=1, num_devices=2
):
    torch.manual_seed(0)
    input_mesh_mapper, weights_mesh_mapper, output_mesh_composer = get_mesh_mappers(mesh_device)
    # instantiate model inference pipeline
    model_pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(
        ttnn_model,
        mesh_device,
        model_repo,
        weights_mesh_mapper=weights_mesh_mapper,
        input_mesh_mapper=input_mesh_mapper,
        output_mesh_composer=output_mesh_composer,
        num_devices=num_devices,
    )

    # load data
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    batch_size = batch_size_per_device * num_devices
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
        ttnn_output = model_pipeline(
            current_batch,
            stream=False,
            return_perf_metrics=False,
            weights_mesh_mapper=weights_mesh_mapper,
            input_mesh_mapper=input_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
        )
        batch_start = i + 1
        batch_end = i + current_batch_size
        logger.debug(f"Dataset text (Inputs {batch_start}--{batch_end}) Sample: {reference_sentences}")
        logger.debug(f"ttnn Model Output (Inputs {batch_start}--{batch_end}) Sample: {ttnn_output}")
        for j in range(current_batch_size):
            reference = ds[i + j]["text"].lower()
            predicted = ttnn_output[j].lower()
            total_wer += jiwer.wer(reference, predicted)
            total_cer += jiwer.cer(reference, predicted)
    logger.info(f"Average Word Error Rate: {total_wer / len(ds):.4f}")
    logger.info(f"Average Character Error Rate: {total_cer / len(ds):.4f}")


@pytest.mark.parametrize(
    "mesh_device",
    ((4, 8),),
    indirect=True,
)
@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper,),
)
@pytest.mark.parametrize(
    "num_inputs,batch_size_per_device",
    [(1, 1)],
)
@pytest.mark.parametrize(
    "input_path",
    (["models/demos/whisper/demo/dataset/audio_classification"]),
)
@pytest.mark.parametrize(
    "run_on_single_card",
    [
        False,
        # True   # Uncomment to run on single_card only
    ],
    ids=lambda val: "run_single_card_only" if val else "run_on_multi_device",
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_demo_for_audio_classification_inference(
    input_path, ttnn_model, mesh_device, num_inputs, batch_size_per_device, is_ci_env, run_on_single_card
):
    if is_ci_env:
        pytest.skip("Skipping test in CI since it provides redundant testing for audio classification inference")

    if run_on_single_card:
        num_devices = 1
    else:
        num_devices = mesh_device.get_num_devices()
    return run_demo_whisper_for_audio_classification_inference(
        input_path,
        ttnn_model,
        mesh_device,
        num_inputs,
        num_devices,
        batch_size_per_device,
    )


@pytest.mark.parametrize(
    "mesh_device",
    ((4, 8),),
    indirect=True,
)
@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper,),
)
@pytest.mark.parametrize(
    "num_inputs,batch_size_per_device",
    [(1, 1)],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "run_on_single_card",
    [
        False,
        # True   # Uncomment to run on single_card only
    ],
    ids=lambda val: "run_single_card_only" if val else "run_on_multi_device",
)
def test_demo_for_audio_classification_dataset(
    input_path, ttnn_model, mesh_device, num_inputs, batch_size_per_device, is_ci_env, run_on_single_card
):
    if is_ci_env:
        pytest.skip("Skipping test in CI since it provides redundant testing")

    if run_on_single_card:
        num_devices = 1
    else:
        num_devices = mesh_device.get_num_devices()
    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    return run_demo_whisper_for_audio_classification_inference(
        input_path,
        ttnn_model,
        mesh_device,
        num_inputs,
        num_devices,
        batch_size_per_device,
        label=True,
        dataset=ds,
    )


@pytest.mark.parametrize(
    "mesh_device",
    ((4, 8),),
    indirect=True,
)
@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper,),
)
@pytest.mark.parametrize(
    "num_inputs,batch_size_per_device",
    [(2, 1)],
)
@pytest.mark.parametrize(
    "model_repo",
    ("distil-whisper/distil-large-v3",),
)
@pytest.mark.parametrize(
    "input_path",
    (["models/demos/whisper/demo/dataset/conditional_generation"]),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "run_on_single_card",
    [
        False,
    ],
    ids=lambda val: "run_single_card_only" if val else "run_on_multi_device",
)
def test_demo_for_conditional_generation(
    input_path, ttnn_model, mesh_device, num_inputs, model_repo, is_ci_env, batch_size_per_device, run_on_single_card
):
    if run_on_single_card:
        num_devices = 1
    else:
        num_devices = mesh_device.get_num_devices()

    ttft, decode_throughput = run_demo_whisper_for_conditional_generation_inference(
        input_path, ttnn_model, mesh_device, num_inputs, model_repo, batch_size_per_device, num_devices=num_devices
    )
    total_batch = num_devices * batch_size_per_device
    if is_ci_env and model_repo == "distil-whisper/distil-large-v3":
        logger.info(f"CI env is True")
        if is_blackhole():
            if mesh_device.dram_grid_size().x == 7:  # P100 DRAM grid is 7x1
                expected_perf_metrics = {"prefill_t/s/u": 4.15, "decode_t/s/u": 90.06}
            else:
                expected_perf_metrics = {"prefill_t/s/u": 4.38, "decode_t/s/u": 96.51}
        else:  # wormhole_b0
            expected_perf_metrics = {"prefill_t/s/u": 3.5, "decode_t/s/u": 41.1}
        expected_perf_metrics["prefill_t/s"] = expected_perf_metrics["prefill_t/s/u"] * total_batch
        expected_perf_metrics["decode_t/s"] = expected_perf_metrics["decode_t/s/u"] * total_batch
        measurements = {
            "prefill_t/s": (1 / ttft) * total_batch,
            "prefill_t/s/u": (1 / ttft),
            "decode_t/s": decode_throughput * total_batch,
            "decode_t/s/u": decode_throughput,
        }
        verify_perf(measurements, expected_perf_metrics)


@pytest.mark.parametrize(
    "mesh_device",
    ((4, 8),),
    indirect=True,
)
@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper,),
)
@pytest.mark.parametrize(
    "model_repo",
    (["distil-whisper/distil-large-v3"]),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
@pytest.mark.parametrize(
    "batch_size_per_device",
    [(1)],
)
@pytest.mark.parametrize(
    "run_on_single_card",
    [
        False,
    ],
    ids=lambda val: "run_single_card_only" if val else "run_on_multi_device",
)
def test_demo_for_conditional_generation_dataset(
    ttnn_model, mesh_device, model_repo, is_ci_env, run_on_single_card, batch_size_per_device
):
    if is_ci_env:
        pytest.skip("Skipping test in CI since it provides redundant testing")

    if run_on_single_card:
        num_devices = 1
    else:
        num_devices = mesh_device.get_num_devices()
    return run_demo_whisper_for_conditional_generation_dataset(
        ttnn_model, mesh_device, model_repo, batch_size_per_device, num_devices
    )
