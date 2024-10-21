# Whisper Demo

Demo showcasing Whisper running on Grayskull - e150 and Wormhole - n150, n300 using ttnn.

## Introduction

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a single model to replace many stages of a traditional speech-processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.

## Details

The entry point to whisper model is `whisper` in `models/demos/whisper/tt/ttnn_optimized_functional_whisper.py` for optimized version.. The model picks up certain configs and weights from huggingface pretrained model. We have used openai/whisper-base version from huggingface as our reference.

### Max Tokens: 32

Max Tokens determines the maximum number of input tokens processed by the model in a single pass durig transcription, optimizing performance and compatibility. It's recommended to set the max_tokens to 32

### Batch size: 8

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the batch_size to 8

## How to Run

### Whisper For Audio Classification
Use `pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_audio_classification[models.demos.whisper.tt.ttnn_optimized_functional_whisper-1-8-WHISPER_MEMORY_CONFIG0-sanchit-gandhi/whisper-medium-fleurs-lang-id-models/demos/whisper/demo/dataset/audio_classification]` to run the ttnn optimized functional whisper demo for audio classification.

#### Our another demo is designed to run with `google/fleurs` for Audio classification

Use `pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_audio_classification_dataset` to run audio classification demo with dataset inputs.

### Whisper For Conditional Generation

Use `pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation[models.demos.whisper.tt.ttnn_optimized_functional_whisper-8-32-WHISPER_MEMORY_CONFIG0-openai/whisper-tiny.en-models/demos/whisper/demo/dataset/conditional_generation-device_params0]` to run the ttnn optimized functional whisper demo for conditional generation.

#### Our another demo is designed to run with `hf-internal-testing/librispeech_asr_dummy` for Conditional generation

Use `pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation_dataset` to run conditional generation demo with dataset inputs.


## Inputs

Inputs by default are provided from `dataset/audio_classification` and `dataset/conditional_generation` folder. If you wish to change the inputs, provide a different path to demo.

For demo with dataset, Inputs for Audio classification is taken from `google/fleurs` dataset and Inputs for Conditional generation is taken from `hf-internal-testing/librispeech_asr_dummy` dataset.

### Owner: [kkeerthana0573](https://github.com/kkeerthana0573)
