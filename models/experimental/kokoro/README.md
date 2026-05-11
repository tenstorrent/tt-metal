# Kokoro (experimental)

Repo-owned **Kokoro-82M** bring-up under `reference/` (HF weights) and **TTNN** ports under `tt/`.

## Full TTNN stack

- **PL-BERT + predictor** on device: `ttnn_kokoro_plbert`, `ttnn_kokoro_predictor`, `ttnn_kokoro_albert`, etc.
- **ISTFTNet vocoder** on device: `KokoroDecoderTt` / `KokoroIstftNetTt` (generator uses device `KokoroTtnnSineGen`).
- **End-to-end module**: `KokoroFullTtnn` in `tt/ttnn_kokoro_full_pipeline.py` composes the above with **no host torch vocoder**; discrete duration/alignment indices still use small CPU tensors (same as the PyTorch reference predictor).

## Demos

```bash
export PYTHONPATH=$(pwd)
# PyTorch reference (CPU/CUDA)
python models/experimental/kokoro/demo/reference_demo.py --text "Hello." --output out.wav
# Full TTNN (Tenstorrent device + `kokoro` for G2P/voice packs only)
python models/experimental/kokoro/demo/ttnn_kokoro_full_demo.py --text "Hello." --voice af_heart --output out_ttnn.wav
```

Install: `pip install "kokoro>=0.9.2" soundfile` and `espeak-ng` on PATH where G2P is needed.

## Tests

From **tt-metal** root:

```bash
export PYTHONPATH=$(pwd)
pytest models/experimental/kokoro/tests/ -v --timeout=600
```

PL-BERT / predictor PCC tests use the `mesh_device` fixture in `tests/conftest.py`. TTNN vocoder PCC tests use the root `device` fixture where applicable.
