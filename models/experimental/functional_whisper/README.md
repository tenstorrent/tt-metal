# Whisper Demo (Wormhole)

## How to Run

Whisper uses 8x8 core grid size, so the following environment variable needs to be set on a N300 setup:

```sh
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

### Conditional Generation

To run the conditional generation demo with custom inputs:
```sh
pytest --disable-warnings --input-path="models/experimental/functional_whisper/demo/dataset/conditional_generation" models/experimental/functional_whisper/demo/demo.py::test_demo_for_conditional_generation
```

To run the conditional generation demo with inputs from the `hf-internal-testing/librispeech_asr_dummy` dataset:
```sh
pytest --disable-warnings models/experimental/functional_whisper/demo/demo.py::test_demo_for_conditional_generation_dataset
```

### Audio Classification

To run the audio classification demo with custom inputs:
```sh
pytest --disable-warnings --input-path="models/experimental/functional_whisper/demo/dataset/audio_classification" models/experimental/functional_whisper/demo/demo.py::test_demo_for_audio_classification
```

To run the audio classification demo with inputs from the `google/fleurs` dataset:
```sh
pytest --disable-warnings models/experimental/functional_whisper/demo/demo.py::test_demo_for_audio_classification_dataset
```
