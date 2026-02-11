# RVC Nano

A minimal, dependency-light subset of the original RVC (Retrieval-based Voice Conversion) project.
The goal is to keep core voice-conversion functionality while making the codebase easier to
port to other platforms and easier to embed in constrained environments.

## Goals
- Minimal dependencies and small runtime footprint
- Keep core RVC inference functionality
- Clean, portable architecture for non-Python or accelerator-backed ports

## Getting started
Install the package (editable is fine for local work):

```sh
uv pip install -e .
```

This repository expects model assets and configs to live outside the package.
Use the helper script to download them via Git LFS (requires `git lfs`):

```sh
./assets-download.sh
```

Then set the required environment variables:

```sh
export RVC_CONFIGS_DIR="$PWD/configs"
export RVC_ASSETS_DIR="$PWD/assets"
```

Run inference using the helper script:

```sh
uv run scripts/infer.py -i speech-sample-01.wav -o ./output/output.wav
```

Run inference using the Python API:

```python
from rvc.vc.pipeline import Pipeline
import soundfile as sf

pipe = Pipeline(if_f0=True, version="v1", num="48k")
audio = pipe.infer("speech-sample-01.wav", speaker_id=0, f0_method="pm")
sf.write("output.wav", audio, pipe.tgt_sr, subtype="PCM_16")
```

Expected layout after download:
- `RVC_CONFIGS_DIR` contains `v1/` and `v2/` config folders plus `hubert_cfg.json`.
- `RVC_ASSETS_DIR` contains `hubert.safetensors` and `pretrained/` weights.

If you need the full-featured project (training, CLI, API), use the upstream repository:
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion

## Repository structure (high level)
- `rvc/`: core Python package
- `rvc/vc/`: VC pipeline and inference code
- `rvc/synthesizer/`: model definitions
- `rvc/configs/`: config loader

## Status
Early and intentionally minimal. Expect missing features and breaking changes while the
interface is refined.

## Attributions
Heavily inspired by RVC (Retrieval-based Voice Conversion) and please see the original repo
https://github.com/RVC-Project/Retrieval-based-Voice-Conversion

## License
See `LICENSE`.
