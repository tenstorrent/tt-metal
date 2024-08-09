# Functional Whisper Model Demos For Audio Classification and Text Generation

## Introduction

Whisper is a pre-trained model for Automatic Speech Recognition (ASR) and Speech Translation. The models are trained on either English-only data or multilingual data. The English-only models were trained on the task of speech recognition. The multilingual models were trained on both Speech Recognition and Speech Translation tasks.

The demos showcases Functional Whisper Model for Audio Classification and Text Generation tasks,
`sanchit-gandhi/whisper-medium-fleurs-lang-id` and `openai/whisper-tiny.en` versions from Hugging Face are utilized for respective tasks.

### Details

The entry point to the Functional Whisper model is the `whisper` function located in `ttnn_optimized_functional_whisper.py`.

## Inputs

Inputs by default are provided from `dataset/audio_classification` and `dataset/conditional_generation` folder. To modify the inputs or specify a different path, adjust the input_path parameter in the command accordingly. It's recommended to avoid direct modifications to the input_data.json file.


For the demos with datasets, Inputs for Audio classification are taken from `google/fleurs` dataset and Inputs for Conditional generation are taken from `hf-internal-testing/librispeech_asr_dummy` dataset.

## Batch size: 8

Batch Size determines the number of input sequences processed simultaneously during training or inference, impacting computational efficiency and memory usage. It is recommended to set the `batch_size` to 8.

## How to run demo for Audio Classification task

To run the demo for audio classification using the Whisper model, follow these instructions:

- Use the following command to run the whisper for audio classification demo with ttnn optimized functional whisper:
  ```
  `pytest --disable-warnings --input-path="models/demos/whisper/demo/dataset/audio_classification" models/demos/whisper/demo/demo.py::test_demo_for_audio_classification[8-models.demos.whisper.tt.ttnn_optimized_functional_whisper]`
  ```

- to run the whisper for audio classification demo with ttnn functional whisper use the following command:
  ```
  pytest --disable-warnings --input-path="models/demos/whisper/demo/dataset/audio_classification" models/demos/whisper/demo/demo.py::test_demo_for_audio_classification[8-8-models.demos.whisper.tt.ttnn_functional_whisper]
  ```

- our another demo is designed to run with `google/fleurs` dataset for Audio classification, to run the demo for dataset use the command:
  ```
  pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_audio_classification_dataset
  ```

## How to run demo for Text Generation task
To run the demo for text generation using the Whisper model, follow these instructions:

- Use the following command to run the whisper for text generation demo with ttnn optimized functional whisper:
  ```
  pytest --disable-warnings --input-path="models/demos/whisper/demo/dataset/conditional_generation" models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation[8-32-models.demos.whisper.tt.ttnn_optimized_functional_whisper]
  ```

- Use the following command to run the whisper for text generation demo with ttnn functional whisper:
  ```
  pytest --disable-warnings --input-path="models/demos/whisper/demo/dataset/conditional_generation" models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation[8-32-models.demos.whisper.tt.ttnn_functional_whisper]
  ```

- Our second demo is designed to run with `hf-internal-testing/librispeech_asr_dummy` dataset for text generation.

- To run the second demo using ttnn optimized functional whisper with dataset inputs for 1 iteration(s), each configured with a batch size of 8 and decoding up to 32 tokens, use the following command :
  ```
  pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation_dataset[8-1-32-models.demos.whisper.tt.ttnn_optimized_functional_whisper]
  ```
- To run the second demo using ttnn functional whisper with dataset inputs for 1 iteration(s), each configured with a batch size of 8 and decoding up to 32 tokens, use the following command:
  ```
  pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation_dataset[8-1-32-models.demos.whisper.tt.ttnn_functional_whisper]
  ```

## Results

The demos presents a comprehensive view of the Whisper model's robustness in audio classification and text generation tasks.

Audio classification predicts the languange of the provided audio sample and the demo using dataset inputs provides the accuracy of the model.
For example, accuracy of 0.75 is observed with `batch_size=8` and `n_iterations=3`

In Text generation, the model predicts transcriptions in the same language as the audio (English).
