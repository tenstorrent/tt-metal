# Whisper

## Platforms:
    Wormhole (n150, n300), Blackhole (p100, p150)

## Introduction

Read more about Whisper at [huggingface.co/distil-whisper/distil-large-v3](https://huggingface.co/distil-whisper/distil-large-v3) & [huggingface.co/openai/whisper-base](https://huggingface.co/openai/whisper-base)

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## How to Run

### Conditional Generation Demo

- To run the conditional generation demo with custom inputs:

```sh
pytest --disable-warnings --input-path="models/demos/whisper/demo/dataset/conditional_generation" models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation
```

- To run the conditional generation demo with inputs from the `hf-internal-testing/librispeech_asr_dummy` dataset:

```sh
pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_conditional_generation_dataset
```

### Audio Classification Demo

- To run the audio classification demo with custom inputs:

```sh
pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_audio_classification_inference --input-path="models/demos/whisper/demo/dataset/audio_classification"
```

- To run the audio classification demo with inputs from the `google/fleurs` dataset:

```sh
pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_audio_classification_dataset
```

### Translation Demo

- To run the translation demo with inputs from the `google/fleurs` dataset:

```sh
pytest --disable-warnings models/demos/whisper/demo/demo.py::test_demo_for_translation_dataset
```

## Notes:
- By default, demo tests (Conditional Generation, Audio Classification) run on a single device (1) and all available devices.
- To run the demo with specific device configurations, provide the desired number of devices under the `mesh_device` parameter.
