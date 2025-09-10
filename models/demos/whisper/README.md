# Whisper

## Platforms:
    Wormhole (n150, n300), Blackhole (p100, p150)

## Introduction

Read more about Whisper at [huggingface.co/distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) & [huggingface.co/openai/whisper-base](https://huggingface.co/openai/whisper-base)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

- Use the following command to run the model:
```sh
pytest --disable-warnings models/demos/whisper/tests/test_whisper_modules.py::test_ttnn_whisper
```

### Conditional Generation Demo

- To run the conditional generation demo with custom inputs:

#### Single Device:

```sh
pytest --disable-warnings --input-path="models/demos/whisper/demo/dataset/conditional_generation" models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation
```

#### Multi Device:
```sh
pytest --disable-warnings --input-path="models/demos/whisper/demo/dataset/conditional_generation" models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation_dp
```

- To run the conditional generation demo with inputs from the `hf-internal-testing/librispeech_asr_dummy` dataset:

#### Single Device:

```sh
pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation_dataset
```

#### Multi Device:

```sh
pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation_dataset_dp
```

### Audio Classification Demo

- To run the audio classification demo with custom inputs:
#### Single Device:

```sh
pytest --disable-warnings --input-path="models/demos/whisper/demo/dataset/audio_classification" models/demos/whisper/demo/demo.py::test_demo_for_audio_classification_inference
```

#### Multi Device:

```sh
pytest --disable-warnings --input-path="models/demos/whisper/demo/dataset/audio_classification" models/demos/whisper/demo/demo.py::test_demo_for_audio_classification_inference_dp
```

- To run the audio classification demo with inputs from the `google/fleurs` dataset:

#### Single Device:

```sh
pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_audio_classification_dataset
```

#### Multi Device:

```sh
pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_audio_classification_dataset_dp
```
