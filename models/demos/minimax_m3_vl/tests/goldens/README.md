# MiniMax-M3-VL reference goldens

The `*.safetensors` here are reference activations (per-submodule, tower, and
final vision tokens) captured from the HF `MiniMaxAI/MiniMax-M3` vision tower.
They are **not committed** (large binaries) — regenerate them.

MiniMax-M3-VL's vision model class only exists in transformers >= 5.x, which
conflicts with the repo's pinned transformers. So goldens are generated in an
isolated venv and the main-env ttnn PCC tests compare against them on disk.

## Regenerate

```bash
# one-time: isolated venv with transformers 5.12.x + CPU torch + torchvision
python3 -m venv /localdev/zbaczewski/m3_ref_venv
/localdev/zbaczewski/m3_ref_venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
/localdev/zbaczewski/m3_ref_venv/bin/pip install "transformers==5.12.1" safetensors pillow numpy accelerate

# generate (reads the 2 vision shards; LLM stays on meta)
HF_HOME=/localdev/zbaczewski/hf_cache \
  /localdev/zbaczewski/m3_ref_venv/bin/python models/demos/minimax_m3_vl/tests/gen_goldens.py
```

`manifest.json` records the tensor shapes present in each `<grid>.safetensors`.
