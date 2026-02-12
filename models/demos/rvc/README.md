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

Run inference using the helper script:

```sh
mkdir -p ./models/demos/rvc/data/output
uv run models/demos/rvc/scripts/infer.py -i ./models/demos/rvc/data/sample-speech.wav -o ./models/demos/rvc/data/output/output.wav
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

If you need the full-featured project (training, CLI, API), use the upstream repository:
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion


## Status
Early and intentionally minimal. Expect missing features and breaking changes while the
interface is refined.

## Attributions
Heavily inspired by RVC (Retrieval-based Voice Conversion) and please see the original repo
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
