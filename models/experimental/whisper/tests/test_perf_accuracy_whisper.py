# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest

from datasets import load_dataset
import numpy as np
from loguru import logger
from transformers import WhisperModel, AutoFeatureExtractor
from transformers import AutoProcessor, WhisperForConditionalGeneration, WhisperForAudioClassification
from models.experimental.whisper.tt.whisper_for_conditional_generation import (
    TtWhisperForConditionalGeneration,
)
from models.experimental.whisper.tt.whisper_for_audio_classification import TtWhisperForAudioClassification

import tt_lib

from models.experimental.whisper.whisper_utils import run_generate
from models.experimental.whisper.tt.whisper_model import TtWhisperModel
from models.utility_functions import (
    disable_persistent_kernel_cache,
    enable_persistent_kernel_cache,
    Profiler,
    tt2torch_tensor,
    torch2tt_tensor,
)
import evaluate
from models.perf.perf_utils import prep_perf_report


BATCH_SIZE = 1


def run_perf_whisper(expected_inference_time, expected_compile_time, iterations, device):
    profiler = Profiler()
    disable_persistent_kernel_cache()
    comments = "tiny"
    first_key = "first_iter"
    second_key = "second_iter"
    third_key = "conditional_generation_accuracy_iter"
    fourth_key = "audio_classificarion_accuracy_iter"
    cpu_key = "ref_key"

    pytorch_model = WhisperModel.from_pretrained("openai/whisper-tiny.en")
    configuration = pytorch_model.config

    pytorch_model.eval()
    state_dict = pytorch_model.state_dict()

    feature_extractor = AutoFeatureExtractor.from_pretrained("openai/whisper-tiny")
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    inputs = feature_extractor(ds[0]["audio"]["array"], return_tensors="pt")
    # original from HF example should be: seq_len = 3000, when max_source_positions=1500
    input_features = inputs.input_features

    dec_seq_len = 32
    decoder_input_ids = (
        torch.tensor(
            [
                [
                    1,
                ]
                * dec_seq_len
            ]
        )
        * pytorch_model.config.decoder_start_token_id
    )

    with torch.no_grad():
        profiler.start(cpu_key)
        pytorch_output = pytorch_model(input_features=input_features, decoder_input_ids=decoder_input_ids)
        profiler.end(cpu_key)

    tt_whisper = TtWhisperModel(state_dict=state_dict, device=device, config=pytorch_model.config)
    tt_whisper.eval()

    with torch.no_grad():
        input_features = torch2tt_tensor(input_features, device, tt_lib.tensor.Layout.ROW_MAJOR)
        input_features = input_features.to(
            device,
            tt_lib.tensor.MemoryConfig(tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1),
        )
        profiler.start(first_key)
        ttm_output = tt_whisper(input_features=input_features, decoder_input_ids=decoder_input_ids)
        tt_lib.device.Synchronize(device)
        profiler.end(first_key)

        enable_persistent_kernel_cache()

        profiler.start(second_key)
        ttm_output = tt_whisper(input_features=input_features, decoder_input_ids=decoder_input_ids)
        tt_lib.device.Synchronize(device)
        profiler.end(second_key)

        profiler.start(third_key)
        bert_score = evaluate.load("bertscore")

        processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en", language="English", task="transcribe")
        hf_reference_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        hf_reference_model.eval()

        # Setup configs
        model_config = hf_reference_model.config

        # Create tt model
        tt_model = TtWhisperForConditionalGeneration(
            state_dict=hf_reference_model.state_dict(), config=model_config, device=device
        )
        tt_model.eval()

        # Librispeech dataset
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        score = []
        for i in range(iterations):
            output, dataset_text = run_generate(processor, i, hf_reference_model, tt_model, ds, device)
            results = bert_score.compute(predictions=[output], references=[dataset_text], lang="en")
            score.append(results["f1"][0])

        accuracy = sum(score) / iterations
        logger.info("Accuracy for Conditional generation")
        logger.info(accuracy)

        tt_lib.device.Synchronize(device)
        profiler.end(third_key)

        profiler.start(fourth_key)

        feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
        model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

        model.eval()
        state_dict = model.state_dict()

        dataset = load_dataset("google/fleurs", "all", split="validation", streaming=True)
        dataset = dataset.shuffle(seed=10)
        tt_whisper_model = TtWhisperForAudioClassification(state_dict=state_dict, device=device, config=model.config)

        tt_whisper_model.eval()
        predicted_label = []
        golden_labels = []
        dataset_iter = iter(dataset)
        for i in range(iterations):
            sample = next(dataset_iter)
            inputs = feature_extractor(
                sample["audio"]["array"],
                sampling_rate=sample["audio"]["sampling_rate"],
                return_tensors="pt",
            )

            input_features = inputs.input_features

            input_features = torch2tt_tensor(input_features, device, tt_lib.tensor.Layout.ROW_MAJOR)
            ttm_logits = tt_whisper_model(
                input_features=input_features,
            ).logits

            # Convert to Torch
            ttm_logits = tt2torch_tensor(ttm_logits)
            tt_predicted_class_ids = torch.argmax(ttm_logits).item()
            predicted_label.append(tt_predicted_class_ids)
            golden_labels.append(sample["lang_id"])

        predicted_label = np.array(predicted_label)
        golden_labels = np.array(golden_labels)
        accuracy = np.mean(predicted_label == golden_labels)
        logger.info("Accuracy for Audio Classification")
        logger.info(accuracy)

        tt_lib.device.Synchronize(device)
        profiler.end(fourth_key)

    first_iter_time = profiler.get(first_key)
    second_iter_time = profiler.get(second_key)
    third_iter_time = profiler.get(third_key)
    fourth_iter_time = profiler.get(fourth_key)
    cpu_time = profiler.get(cpu_key)
    compile_time = first_iter_time - second_iter_time

    prep_perf_report(
        model_name="whisper",
        batch_size=BATCH_SIZE,
        inference_and_compile_time=first_iter_time,
        inference_time=second_iter_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments=comments,
        inference_time_cpu=cpu_time,
    )

    logger.info(f"whisper tiny inference time: {second_iter_time}")
    logger.info(f"whisper compile time: {compile_time}")
    logger.info(f"whisper conditional generation inference for {iterations} Samples: {third_iter_time}")
    logger.info(f"whisper audio classification inference for {iterations} Samples: {fourth_iter_time}")


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time,iterations",
    (
        (
            4.5,
            25,
            4,
        ),
    ),
)
def test_perf_bare_metal(use_program_cache, expected_inference_time, expected_compile_time, iterations, device):
    run_perf_whisper(expected_inference_time, expected_compile_time, iterations, device)


@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "expected_inference_time, expected_compile_time,iterations",
    (
        (
            4.5,
            27,
            4,
        ),
    ),
)
def test_perf_virtual_machine(use_program_cache, expected_inference_time, expected_compile_time, iterations, device):
    run_perf_whisper(expected_inference_time, expected_compile_time, iterations, device)
