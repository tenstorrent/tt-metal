# MiniMax-H3

## Folder structure

Bringup is split by component. Each component has its own folder for team members to slot work into:

```
models/tt_dit/
├── models/
│   ├── MiniMaxH3.md                  # this file
│   ├── transformers/minimax_h3/      # denoising transformer (block, attention, RoPE)
│   ├── vae/minimax_h3/               # video VAE (AutoencoderKLMiniMaxH3)
│   └── audio_vae/minimax_h3/         # audio VAE (AutoencoderKLMiniMaxH3Audio)
└── encoders/qwen3vl/                 # text encoder — already present, shared with other models
```

There is no `__init__.py`: `tt_dit` uses namespace packages, matching `transformers/wan2_2/` and
`transformers/ltx/`. Import as e.g.
`from ....models.transformers.minimax_h3.attention_minimax_h3 import ...`.

Tests follow the layout of the other models, under `models/tt_dit/tests/models/minimax_h3/`.

## Setup

MiniMax-H3 support is not in a released `diffusers` yet. Bringup is pinned to a specific commit of
the `diffusers` main repository, which provides the reference `MiniMaxH3Transformer3DModel`,
`AutoencoderKLMiniMaxH3`, `AutoencoderKLMiniMaxH3Audio` and `MiniMaxH3Scheduler`.

Install it into the environment you run the tests from:

```bash
pip install "diffusers @ git+https://github.com/huggingface/diffusers@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
```

Verify the reference classes resolve:

```bash
python -c "from diffusers import MiniMaxH3Transformer3DModel; print('ok')"
```

### If the environment was created by `uv`

A `uv`-created virtualenv (such as `container_python_env` in the dev container) has no `pip` of its
own. After activating it, a bare `pip install` silently resolves to the system `pip` and installs to
`~/.local`, where the venv will not see it — it reports success and has no effect. Install through
`uv` against the interpreter instead:

```bash
uv pip install --python <venv>/bin/python --no-deps \
  "diffusers @ git+https://github.com/huggingface/diffusers@abc5e9bf71fd38f53cd471bc3acaa84bc5ecbfdc"
```

`--no-deps` keeps the resolver from pulling a newer `numpy` / `Pillow` / `huggingface-hub` into an
environment that `ttnn` was built against. The pinned commit's dependencies are already satisfied by
an environment that had any recent `diffusers` installed. Re-check `import ttnn` after installing.
