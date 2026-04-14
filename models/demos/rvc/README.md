# RVC Tenstorrent model bring-up

A bring up demo of RVC models in tenstorrent devices

## Prerequisites
- `git lfs` installed and initialized (`git lfs install`)
- Hardware: Tenstorrent N300


## Getting started
Run the following commands from the repository root (`/root/tt-metal`).

Install the package (editable is fine for local work):

```sh
uv pip install -e .
```

Install demo-specific dependencies:

```sh
uv pip install -r ./models/demos/rvc/requirements.txt
```

This repository expects model assets and configs to live outside the package.
Use the helper script to download them via Git LFS (requires `git lfs`):

```sh
chmod +x ./models/demos/rvc/assets-download.sh
./models/demos/rvc/assets-download.sh
```

Then set the required environment variables:

```sh
export RVC_CONFIGS_DIR="$PWD/models/demos/rvc/data/configs"
export RVC_ASSETS_DIR="$PWD/models/demos/rvc/data/assets"
export RVC_TEST_INPUT="$PWD/models/demos/rvc/data/sample-speech.wav"
```

Run inference using the helper script:

```sh
mkdir -p ./models/demos/rvc/data/output
uv run --active models/demos/rvc/scripts/infer_ttnn.py -i ./models/demos/rvc/data/sample-speech.wav -o ./models/demos/rvc/data/output/output_ttnn.wav

# torch version
# uv run --active models/demos/rvc/scripts/infer_torch.py -i ./models/demos/rvc/data/sample-speech.wav -o ./models/demos/rvc/data/output/output_torch.wav

```

Run inference using the Python API:

```python
from models.demos.rvc.torch_impl.vc.pipeline import Pipeline
import soundfile as sf

pipe = Pipeline(if_f0=True, version="v1", num="48k")
audio = pipe.infer("./models/demos/rvc/data/sample-speech.wav", speaker_id=0, f0_method="pm")
sf.write("./models/demos/rvc/data/output/output.wav", audio, pipe.tgt_sr, subtype="PCM_16")
```

Expected layout after download:
- `RVC_CONFIGS_DIR` contains `v1/` and `v2/` config folders plus `hubert_cfg.json`.
- `RVC_ASSETS_DIR` contains `hubert.safetensors` and `pretrained/` weights.


## Development

Running tests
```sh
uv run --active pytest ./models/demos/rvc/tests
```

## Runner

For repeated fixed-shape benchmarking and validation, `rvc` now has a runner surface similar to
`unet_3d`:

- `models.demos.rvc.runner.performant_runner.RVCRunner`
- `models.demos.rvc.runner.performant_runner_infra.RVCTestInfra`

This runner currently assumes a fixed input audio path and fixed inference configuration for the
life of the initialized runner. That matches the constraints needed for later trace-oriented work:
the `rvc` pipeline is more dynamic than `unet_3d`, so trace/CQ optimization will need to target a
stable subgraph or fixed input shape rather than arbitrary audio lengths.

The runner is Tenstorrent-only. Torch reference generation and TT-vs-Torch comparison stay in the
separate eval flow instead of the runtime runner.

## Evaluation

Speaker similarity should live under a dedicated eval surface rather than inside the TT/Torch inference paths.
This demo now exposes:

- module: `models.demos.rvc.evals.speaker_similarity`
- script: `models/demos/rvc/scripts/eval_speaker_similarity.py`
- module: `models.demos.rvc.evals.token_accuracy`
- script: `models/demos/rvc/scripts/eval_token_accuracy.py`
- module: `models.demos.rvc.evals.wer`
- script: `models/demos/rvc/scripts/eval_wer.py`

Recommended backend for the bounty metric:

```sh
pip install transformers
python models/demos/rvc/scripts/eval_speaker_similarity.py \
  --source-audio path/to/original_source.wav \
  --generated-audio path/to/converted_audio.wav \
  --json
```

Notes:
- The default backend uses a Transformers `WavLMForXVector` speaker encoder and reports cosine similarity.
- `speechbrain_ecapa` is also supported, but it depends on `torchaudio` and may require a CPU-compatible install.
- If the model is not already cached locally, backend initialization may need network access to download weights.
- This is intentionally separate from the TTNN inference pipeline so future evals such as WER, mel similarity,
  and frame-level TT-vs-Torch comparisons can live under `models/demos/rvc/evals/`.

Content-preservation WER can be computed from source audio and generated audio:

```sh
python models/demos/rvc/scripts/eval_wer.py \
  --source-audio path/to/original_source.wav \
  --generated-audio path/to/converted_audio.wav \
  --device cpu
```

This runs ASR on both files, normalizes the transcripts, and reports
`WER(source_transcript, generated_transcript)`.

Token-level accuracy against a PyTorch reference can be computed from a Torch-generated
reference audio and a TTNN-generated candidate audio:

```sh
python models/demos/rvc/scripts/eval_token_accuracy.py \
  --reference-audio path/to/output_torch.wav \
  --candidate-audio path/to/output_ttnn.wav \
  --device cpu
```

This runs ASR on both files, normalizes the transcripts, tokenizes them with the same
tokenizer, and reports:

`token_accuracy = 1 - edit_distance(reference_tokens, candidate_tokens) / len(reference_tokens)`

## Attributions
Heavily inspired by RVC (Retrieval-based Voice Conversion) and please see the original repo
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
