---

# Functional Whisper Model Demos For Audio Classification and Text Generation

## Introduction

Whisper is a pre-trained model for automatic speech recognition (ASR) and speech translation.The models are trained on either English-only data or multilingual data. The English-only models were trained on the task of speech recognition. The multilingual models were trained on both speech recognition and speech translation tasks.

The demos showcases the Functional Whisper Model for Audio Classification and Text Generation tasks,
`sanchit-gandhi/whisper-medium-fleurs-lang-id` and `openai/whisper-tiny.en` versions Hugging Face are utilized respective tasks.

### Details

The entry point to the Functional Whisper model is the `whisper` function located in `ttnn_optimized_functional_whisper.py`.

## Inputs

Inputs by default are provided from `dataset/audio_classification` and `dataset/conditional_generation` folder. If you wish to change the inputs, provide a different path to demo.

For demo with dataset,Inputs for Audio classification is taken from `google/fleurs` dataset and Inputs for Conditional generation is taken from `hf-internal-testing/librispeech_asr_dummy` dataset.

## Batch size: 8

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It's recommended to set the `batch_size` to 8

## How to run demo for Audio Classification task

To run the demo for audio classification using the Whisper model, follow these instructions:

- Use the following command to run the whisper for audio classification demo with ttnn optimized functional whisper:
  ```
  `pytest --disable-warnings --input-path="models/experimental/functional_whisper/demo/dataset/audio_classification" models/experimental/functional_whisper/demo/demo.py::test_demo_for_audio_classification[8-models.experimental.functional_whisper.tt.ttnn_optimized_functional_whisper]`
  ```

- to run the whisper for audio classification demo with ttnn functional whisper use the following command:
  ```
  pytest --disable-warnings --input-path="models/experimental/functional_whisper/demo/dataset/audio_classification" models/experimental/functional_whisper/demo/demo.py::test_demo_for_audio_classification[8-models.experimental.functional_whisper.tt.ttnn_functional_whisper]
  ```

- our another demo is designed to run with `google/fleurs` dataset for Audio classification, to run the demo for dataset use the command:
  ```
  pytest --disable-warnings models/experimental/functional_whisper/demo/demo.py::test_demo_for_audio_classification_dataset
  ```

## How to run demo for Text Generation task
To run the demo for text generation using the Whisper model, follow these instructions:

- Use the following command to run the whisper for text generation demo with ttnn optimized functional whisper:
  ```
  `pytest --disable-warnings --input-path="models/experimental/functional_whisper/demo/dataset/conditional_generation" models/experimental/functional_whisper/demo/demo.py::test_demo_for_conditional_generation[1-models.experimental.functional_whisper.tt.ttnn_optimized_functional_whisper]`
  ```

- Use the following command to run the whisper for text generation demo with ttnn functional whisper:
  ```
  pytest --disable-warnings --input-path="models/experimental/functional_whisper/demo/dataset/conditional_generation" models/experimental/functional_whisper/demo/demo.py::test_demo_for_conditional_generation[1-models.experimental.functional_whisper.tt.ttnn_functional_whisper]
  ```

- our another demo is designed to run with `hf-internal-testing/librispeech_asr_dummy` for text generation, to run the demo for dataset use the command:
  ```
  pytest --disable-warnings models/experimental/functional_whisper/demo/demo.py::test_demo_for_conditional_generation_dataset
  ```

## Results

The text generation demo presents a comprehensive view of the Whisper model's robustness in audio classification and text generation tasks.

Audio classification predicts the languange of the provided audio sample and dataset demo
also provides the accuracy of the model.
for example `batch_size=8` and `n_iterations=3` gives an accuracy of 0.75

For Text generation, the model predicts transcriptions in the same language as the audio (English).

---
