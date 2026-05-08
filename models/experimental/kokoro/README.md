# Kokoro demo

## Demo

From the **tt-metal** repository root (with `PYTHONPATH` set to that root, same as other demos):

```bash
export PYTHONPATH=$(pwd)
python models/experimental/kokoro/demo/reference_demo.py --text "Hello." --output out.wav
```

Install upstream Kokoro for the pipeline: `pip install "kokoro>=0.9.2" soundfile`. Use `--device cpu` or `--device cuda` if you need a specific device.

## Tests

```bash
export PYTHONPATH=$(pwd)
cd models/experimental/kokoro
pytest tests/ --confcutdir=. -v
```

`--confcutdir=.` avoids the repo root `conftest` when optional dependencies are missing. TTNN PCC tests need a full tt-metal runtime and a connected device, or they skip if `ttnn.open_device` is not available.
