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
```

The current demo uses fixed input audio files under:

- `models/demos/rvc/data/speech/sample-speech-0.wav`
- `models/demos/rvc/data/speech/sample-speech-1.wav`
- ...
- `models/demos/rvc/data/speech/sample-speech-7.wav`

Run inference using the helper script:

```sh
mkdir -p ./models/demos/rvc/data/output

# Single-audio run. The pipeline loads
# `models/demos/rvc/data/speech/sample-speech-0.wav`.
uv run --active models/demos/rvc/scripts/infer_ttnn.py -o ./models/demos/rvc/data/output/output_ttnn.wav

# torch version
# uv run --active models/demos/rvc/scripts/infer_torch.py -o ./models/demos/rvc/data/output/output_torch.wav

```

Batch input loading is available with `--batch-run`. This loads the hard-coded files
`sample-speech-0.wav` through `sample-speech-7.wav`, pads them to the longest input,
and runs them as one batch:

```sh
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  --performance-runner \
  --batch-run
```

On multi-device systems, use `--mesh-num-devices` to open a 1xN mesh and shard the
batch dimension across devices. For example, this runs the 8-file batch across 2
devices:

```sh
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  --performance-runner \
  --mesh-num-devices 2 \
  --batch-run
```

For a replicated single input across 2 devices, pass an explicit batch size:

```sh
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  --performance-runner \
  --mesh-num-devices 2 \
  --batch-size 2
```

Run inference using the Python API:

```python
from models.demos.rvc.torch_impl.vc.pipeline import Pipeline
import soundfile as sf

pipe = Pipeline(if_f0=True, version="v1", num="48k")
audio = pipe.infer()
sf.write("./models/demos/rvc/data/output/output.wav", audio, pipe.tgt_sr, subtype="PCM_16")
```

Expected layout after download:
- `RVC_CONFIGS_DIR` contains `v1/` and `v2/` config folders plus `hubert_cfg.json`.
- `RVC_ASSETS_DIR` contains `hubert.safetensors` and `pretrained/` weights.
- `models/demos/rvc/data/speech/` contains the fixed sample speech inputs and
  `sample-speech-transcript.txt`.


## Development

Running tests
```sh
uv run --active pytest ./models/demos/rvc/tests
```

## Runner

For repeated fixed-shape benchmarking and validation, `rvc` now has a runner surface similar to
`unet_3d`:

- `models.demos.rvc.runner.performant_runner.RVCRunner`

This runner currently assumes a fixed input audio path and fixed inference configuration for the
life of the initialized runner. That matches the constraints needed for later trace-oriented work:
the `rvc` pipeline is more dynamic than `unet_3d`, so trace/CQ optimization will need to target a
stable subgraph or fixed input shape rather than arbitrary audio lengths.

The runner supports data-parallel mesh execution when the caller passes a `ttnn.MeshDevice`.
The global batch size must be divisible by `device.get_num_devices()`. Runtime tensors are
split along batch dimension and outputs are concatenated back along batch dimension.

The runner is Tenstorrent-only. Torch reference generation and TT-vs-Torch comparison stay in the
separate eval flow instead of the runtime runner.

## Evaluation

Start with a plain TTNN run before enabling any evaluation flags. This verifies that the
model loads, runs on the Tenstorrent device, and writes converted audio.

```sh
mkdir -p ./models/demos/rvc/data/output
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  -o ./models/demos/rvc/data/output/output_ttnn.wav
```

The expected result is a playable `output_ttnn.wav` file and printed runtime metrics
such as average inference time, output duration, and real-time factor (RTF).

For batch and data-parallel checks, use:

```sh
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  --performance-runner \
  --mesh-num-devices 2 \
  --batch-run
```

This loads all 8 fixed speech samples from `models/demos/rvc/data/speech/`, shards
the batch across 2 devices, and reports the output shape, batch size, input sample
count, average runtime, output duration, and RTF.

After the plain run succeeds, check token-level accuracy against the PyTorch reference and WER

```sh
uv run --active models/demos/rvc/scripts/infer_ttnn.py \
  -o ./models/demos/rvc/data/output/output_ttnn.wav \
  --check-torch-token-accuracy
```

This command runs the TTNN pipeline, generates a PyTorch reference in validation mode,
checks waveform PCC, and reports token-level content accuracy. The token accuracy
target is:

`token_accuracy > 0.95`

## Attributions
Heavily inspired by RVC (Retrieval-based Voice Conversion) and please see the original repo
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
