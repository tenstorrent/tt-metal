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

### End to End Model Performance for Whisper Variant - `openai-whisper-base`

#### Single Device:

- End-2-end perf is 35 FPS
```sh
pytest --disable-warnings models/demos/whisper/tests/test_performance.py::test_whisper_v3_e2e_performance[openai-whisper-base]
```

#### Multi Device (DP=2, n300):

- End-2-end perf is 55 FPS
```sh
pytest --disable-warnings models/demos/whisper/tests/test_performance.py::test_whisper_v3_e2e_performance_dp[openai-whisper-base]
```

### End to End Model Performance for Whisper Variant - `distil-whisper/distil-large-v3`

#### Single Device:

- End-2-end perf is 21 FPS
```sh
pytest --disable-warnings models/demos/whisper/tests/test_performance.py::test_whisper_v3_e2e_performance[distil-whisper/distil-large-v3]
```
Note: Reported e2e perf for single device is from N150 execution (that uses tensix dispatch). If you are running for single device on N300, the e2e perf will be slow as N300 uses eth dispatch.

#### Multi Device (DP=2, n300):

- End-2-end perf is 10 FPS
```sh
pytest --disable-warnings models/demos/whisper/tests/test_performance.py::test_whisper_v3_e2e_performance_dp[distil-whisper/distil-large-v3]
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
