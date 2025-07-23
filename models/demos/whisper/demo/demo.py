# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

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
from models.demos.utils.llm_demo_utils import verify_perf
from models.demos.whisper.tt import ttnn_optimized_functional_whisper
from models.demos.whisper.tt.ttnn_optimized_functional_whisper import WHISPER_L1_SMALL_SIZE, init_kv_cache
from models.generation_utils import get_logits_processor
from models.utility_functions import is_blackhole


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


def load_conditional_generation_ref_model():
    hf_ref_model = (
        WhisperForConditionalGeneration.from_pretrained("distil-whisper/distil-large-v3").to(torch.bfloat16).eval()
    )
    processor = AutoProcessor.from_pretrained("distil-whisper/distil-large-v3", language="English", task="transcribe")
    feature_extractor = AutoFeatureExtractor.from_pretrained("distil-whisper/distil-large-v3")
    config = hf_ref_model.config
    return (
        hf_ref_model,
        config,
        processor,
        feature_extractor,
    )


def init_conditional_generation_tt_model(hf_ref_model, config, ttnn_model, device, max_batch_size=1, max_seq_len=512):
    model = hf_ref_model.model
    linear_weight = hf_ref_model.proj_out.weight

    ttnn_linear_weight = ttnn.from_torch(linear_weight, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    ttnn_linear_weight = ttnn.permute(ttnn_linear_weight, (1, 0))
    ttnn_linear_weight = ttnn.to_layout(ttnn_linear_weight, layout=ttnn.TILE_LAYOUT)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )

    # Note: config.max_length is 448 for distil-whisper/distil-large-v3
    kv_cache = init_kv_cache(config, device, max_batch_size, max_seq_len=max_seq_len)

    return parameters, ttnn_linear_weight, kv_cache


def run_generate(
    config,
    audio_data,
    sampling_rate,
    feature_extractor,
    ttnn_model,
    parameters,
    processor,
    ttnn_linear_weight,
    device,
    generation_config,
    kv_cache=None,
    stream_generation=False,
    feature_dtype_to_use=torch.bfloat16,
    return_perf_metrics=False,
):
    start_encode = time.time()

    # Compute features
    inputs = feature_extractor(audio_data, sampling_rate=sampling_rate, return_tensors="pt")
    input_features = inputs.input_features.type(feature_dtype_to_use)
    unpadded_batch_size = input_features.shape[0]
    assert unpadded_batch_size == 1, "Only batch size 1 is supported for inference"

    # Compute embeddings
    input_embeds = ttnn_model.preprocess_encoder_inputs(
        config, input_features, parameters=parameters.encoder, device=device
    )

    # Run encoder
    encoder_hidden_states = ttnn_model.encoder(config, input_embeds, parameters=parameters.encoder)
    ttnn.synchronize_device(device)
    logger.info(f"Time to encoder states: {(time.time() - start_encode)*1000:.3f}ms")

    # Run decoder

    def _run_generate():
        # Input ids
        input_ids = torch.tensor([[1]]) * config.decoder_start_token_id
        logits_processor = get_logits_processor(input_ids, config)
        if not kv_cache:
            input_ids = pad_input_32(input_ids, config.pad_token_id).to(torch.long)
            decoder_start_values = generation_config.pad_token_id * torch.ones(1, 32).to(torch.long)

        # Initial decode position
        current_decode_pos = (
            ttnn.from_torch(torch.zeros(unpadded_batch_size), device=device, dtype=ttnn.int32) if kv_cache else None
        )

        MAX_GEN_LEN = config.max_length  # 448 for distil-whisper/distil-large-v3
        print_each_iter = False
        output_ids = []
        total_decode_time = 0
        for i in tqdm(range(MAX_GEN_LEN), desc="Decode inference iterations"):
            start_iter = time.time()

            decoder_hidden_states, decoder_attention_mask = ttnn_model.preprocess_decoder_inputs(
                config=config,
                input_ids=input_ids,
                attention_mask=None,
                parameters=parameters.decoder,
                device=device,
                decode_pos=i if kv_cache else None,
                create_attention_mask=(not kv_cache),
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
            logits_to_torch = ttnn.to_torch(output)
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

            ttnn_transcription = processor.batch_decode(next_tokens.unsqueeze(dim=1), skip_special_tokens=True)[0]
            if print_each_iter:
                logger.info(processor.batch_decode(torch.stack(output_ids, dim=1), skip_special_tokens=True)[0])

            if return_perf_metrics:
                yield ttnn_transcription, ttft, avg_decode_throughput
            else:
                yield ttnn_transcription

            if next_tokens == config.eos_token_id:
                break

        total_generate_time = time.time() - start_encode
        logger.info(f"Time to first token: {(ttft*1000):.3f}ms")
        logger.info(f"Total decode time: {total_decode_time:.3f}s")
        logger.info(f"Total generate time: {total_generate_time:.3f}s")
        logger.info(f"Average decode throughput: {avg_decode_throughput:.3f} t/s/u")

    # conditionally return generator or full response
    if stream_generation:
        return _run_generate()
    else:
        output = []
        for x in _run_generate():
            if return_perf_metrics:
                out_cur, ttft, avg_decode_throughput = x
            else:
                out_cur = x
            output.append(out_cur)
        output = "".join(output)

        if return_perf_metrics:
            return output, ttft, avg_decode_throughput
        else:
            return output


def create_functional_whisper_for_conditional_generation_inference_pipeline(ttnn_model, device):
    """
    Returns a callable with signature (data, sampling_rate, stream), where data is is a 1D numpy array
    and sampling_rate is an int representing the sampling rate used to acquire data, and stream turns
    signals the callable to return a generator if True, yielding the decoded tokens as they are processed, else
    the callable returns the full decoded output.
    """
    hf_ref_model, config, processor, feature_extractor = load_conditional_generation_ref_model()
    parameters, ttnn_linear_weight, kv_cache = init_conditional_generation_tt_model(
        hf_ref_model, config, ttnn_model, device
    )

    def _model_pipeline(data, sampling_rate, stream=False, return_perf_metrics=False):
        logger.info(f"Running model on audio data with duration {data.shape[0]/sampling_rate:.3f}s")

        return run_generate(
            config,
            data,
            sampling_rate,
            feature_extractor,
            ttnn_model,
            parameters=parameters,
            processor=processor,
            ttnn_linear_weight=ttnn_linear_weight,
            device=device,
            generation_config=hf_ref_model.generation_config,
            kv_cache=kv_cache,
            stream_generation=stream,
            return_perf_metrics=return_perf_metrics,
        )

    return _model_pipeline


def run_demo_whisper_for_audio_classification_inference(input_path, ttnn_model, device, num_inputs):
    torch.manual_seed(1234)

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
    if len(input_data) < num_inputs:
        assert False, "num_inputs exceeds number of audio files available in folder"

    for i in range(num_inputs):
        input_file_path = input_data[i]
        samplerate, data = wavfile.read(input_file_path)

        inputs = feature_extractor(
            data,
            sampling_rate=samplerate,
            return_tensors="pt",
        )

        input_features = inputs.input_features

        config = model.config
        input_embedding = ttnn_model.preprocess_encoder_inputs(
            config=config, input_features=input_features, parameters=parameters.encoder, device=device
        )

        encoder_outputs = ttnn_model.encoder(
            config=config, inputs_embeds=input_embedding, parameters=parameters.encoder
        )

        hidden_states = ttnn.matmul(encoder_outputs, parameters.projector.weight)
        hidden_states = ttnn.add(hidden_states, parameters.projector.bias)

        pooled_output = ttnn.mean(hidden_states, dim=-2, keepdim=True)

        logits = ttnn.matmul(pooled_output, parameters.classifier.weight)
        logits = ttnn.add(logits, parameters.classifier.bias)

        logits_torch = ttnn.to_torch(logits)
        predicted_class_ids = torch.argmax(logits_torch).item()
        predicted_label = model.config.id2label[predicted_class_ids]

        logger.info("predicted_label")
        logger.info(predicted_label)


def run_demo_whisper_for_audio_classification_dataset(ttnn_model, device):
    torch.manual_seed(1234)

    feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
    model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

    model.eval()

    ds = load_dataset("google/fleurs", "all", split="validation", streaming=True)
    sample = next(iter(ds))

    inputs = feature_extractor(
        sample["audio"]["array"],
        sampling_rate=sample["audio"]["sampling_rate"],
        return_tensors="pt",
    )

    input_features = inputs.input_features

    logger.debug("Input audio language:")
    logger.debug(sample["language"])

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=ttnn_model.convert_to_ttnn,
        custom_preprocessor=ttnn_model.custom_preprocessor,
        device=device,
    )

    config = model.config
    input_embedding = ttnn_model.preprocess_encoder_inputs(
        config=config, input_features=input_features, parameters=parameters.encoder, device=device
    )

    encoder_outputs = ttnn_model.encoder(config=config, inputs_embeds=input_embedding, parameters=parameters.encoder)

    hidden_states = ttnn.matmul(encoder_outputs, parameters.projector.weight)
    hidden_states = ttnn.add(hidden_states, parameters.projector.bias)

    pooled_output = ttnn.mean(hidden_states, dim=-2, keepdim=True)

    logits = ttnn.matmul(pooled_output, parameters.classifier.weight)
    logits = ttnn.add(logits, parameters.classifier.bias)

    logits_torch = ttnn.to_torch(logits)
    predicted_class_ids = torch.argmax(logits_torch).item()
    predicted_label = model.config.id2label[predicted_class_ids]

    logger.info("predicted_label")
    logger.info(predicted_label)


def run_demo_whisper_for_conditional_generation_inference(input_path, ttnn_model, device, num_inputs):
    torch.manual_seed(0)

    # instantiate model inference pipeline
    model_pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(ttnn_model, device)

    # load data
    input_data = load_input_paths(input_path)

    if len(input_data) < num_inputs:
        assert False, "num_inputs exceeds number of audio files available in folder"

    total_ttft = 0
    total_decode_throughput = 0
    num_warmup_runs = 1
    for i in range(num_inputs):
        input_file_path = input_data[i]
        samplerate, data = wavfile.read(input_file_path)

        # perform model inference
        ttnn_output, ttft, avg_decode_throughput = model_pipeline(
            data, samplerate, stream=False, return_perf_metrics=True
        )
        if i >= num_warmup_runs:  # Exclude first compile run
            total_ttft += ttft
            total_decode_throughput += avg_decode_throughput
        logger.info(f"Model Output (Input {i+1}): {ttnn_output}")

    avg_ttft = total_ttft / (num_inputs - num_warmup_runs)
    avg_decode_throughput = total_decode_throughput / (num_inputs - num_warmup_runs)
    return avg_ttft, avg_decode_throughput


def run_demo_whisper_for_conditional_generation_dataset(ttnn_model, device):
    torch.manual_seed(0)

    # instantiate model inference pipeline
    model_pipeline = create_functional_whisper_for_conditional_generation_inference_pipeline(ttnn_model, device)

    # load data
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

    # perform model inference
    total_wer = 0
    total_cer = 0
    for ds_input in tqdm(ds, desc="Processing dataset inputs"):
        data = ds_input["audio"]["array"]
        sampling_rate = 16000
        ttnn_output = model_pipeline(data, sampling_rate, stream=False)
        logger.info(f"Model output: {ttnn_output}")

        # Compute word and character error rates
        reference = ds_input["text"].lower()
        predicted = ttnn_output.lower()
        total_wer += jiwer.wer(reference, predicted)
        total_cer += jiwer.cer(reference, predicted)

    logger.info(f"Average Word Error Rate: {total_wer / len(ds):.4f}")
    logger.info(f"Average Character Error Rate: {total_cer / len(ds):.4f}")


@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper,),
)
@pytest.mark.parametrize(
    "num_inputs",
    ((1),),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_demo_for_audio_classification(input_path, ttnn_model, device, num_inputs):
    return run_demo_whisper_for_audio_classification_inference(input_path, ttnn_model, device, num_inputs)


@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper,),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_demo_for_audio_classification_dataset(ttnn_model, device, is_ci_env):
    if is_ci_env:
        pytest.skip("Skipping test in CI since it provides redundant testing")
    return run_demo_whisper_for_audio_classification_dataset(ttnn_model, device)


@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper,),
)
@pytest.mark.parametrize(
    "num_inputs",
    (2,),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_demo_for_conditional_generation(input_path, ttnn_model, device, num_inputs, is_ci_env):
    ttft, decode_throughput = run_demo_whisper_for_conditional_generation_inference(
        input_path, ttnn_model, device, num_inputs
    )
    if is_ci_env:
        if is_blackhole():
            if device.dram_grid_size().x == 7:  # P100 DRAM grid is 7x1
                expected_perf_metrics = {"prefill_t/s": 7.67, "decode_t/s/u": 91.0}
            else:
                expected_perf_metrics = {"prefill_t/s": 7.67, "decode_t/s/u": 98.0}
        else:  # wormhole_b0
            expected_perf_metrics = {"prefill_t/s": 3.85, "decode_t/s/u": 51.8}
        expected_perf_metrics["decode_t/s"] = expected_perf_metrics["decode_t/s/u"]  # Only supporting batch 1
        measurements = {"prefill_t/s": 1 / ttft, "decode_t/s": decode_throughput, "decode_t/s/u": decode_throughput}
        verify_perf(measurements, expected_perf_metrics)


@pytest.mark.parametrize(
    "ttnn_model",
    (ttnn_optimized_functional_whisper,),
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": WHISPER_L1_SMALL_SIZE}], indirect=True)
def test_demo_for_conditional_generation_dataset(ttnn_model, device, is_ci_env):
    if is_ci_env:
        pytest.skip("Skipping test in CI since it provides redundant testing")
    return run_demo_whisper_for_conditional_generation_dataset(ttnn_model, device)
