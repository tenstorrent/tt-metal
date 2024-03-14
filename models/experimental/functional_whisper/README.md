# ttnn_functional_whisper Demo

## How to Run

Use `pytest --disable-warnings --input-path="models/experimental/functional_whisper/demo/dataset/audio_classification" models/experimental/functional_whisper/demo/demo.py::test_demo_for_audio_classification[1-models.experimental.functional_whisper.tt.ttnn_optimized_functional_whisper]` to run the ttnn optimized functional whisper demo for audio classification.

Use `pytest --disable-warnings --input-path="models/experimental/functional_whisper/demo/dataset/audio_classification" models/experimental/functional_whisper/demo/demo.py::test_demo_for_audio_classification[1-models.experimental.functional_whisper.tt.ttnn_functional_whisper]` to run the ttnn functional whisper demo for audio classification.

Use `pytest --disable-warnings --input-path="models/experimental/functional_whisper/demo/dataset/conditional_generation" models/experimental/functional_whisper/demo/demo.py::test_demo_for_conditional_generation[1-models.experimental.functional_whisper.tt.ttnn_optimized_functional_whisper]` to run the ttnn optimized functional whisper demo for conditional generation.

Use `pytest --disable-warnings --input-path="models/experimental/functional_whisper/demo/dataset/conditional_generation" models/experimental/functional_whisper/demo/demo.py::test_demo_for_conditional_generation[1-models.experimental.functional_whisper.tt.ttnn_functional_whisper]` to run the ttnn functional whisper demo for conditional generation.

Our another demo is designed to run with `google/fleurs` for Audio classification and `hf-internal-testing/librispeech_asr_dummy` for Conditional generation

Use `pytest --disable-warnings models/experimental/functional_whisper/demo/demo.py::test_demo_for_audio_classification_dataset` to run audio classification demo with dataset input.

Use `pytest --disable-warnings models/experimental/functional_whisper/demo/demo.py::test_demo_for_conditional_generation_dataset` to run conditional generation demo with dataset input.

## Inputs

Inputs by default are provided from `dataset/audio_classification` and `dataset/conditional_generation` folder. If you wish to change the inputs, provide a different path to demo.

For demo with dataset,Inputs for Audio classification is taken from `google/fleurs` dataset and Inputs for Conditional generation is taken from `hf-internal-testing/librispeech_asr_dummy` dataset.

## Details

The entry point to whisper model is whisper in `models/experimental/functional_whisper/tt/ttnn_optimized_functional_whisper.py` for optimized version.(`models/experimental/functional_whisper/tt/ttnn_functional_whisper.py` for normal version).
