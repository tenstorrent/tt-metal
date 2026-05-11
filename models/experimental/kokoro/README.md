# Kokoro (experimental)

Repo-owned **Kokoro-82M** bring-up under `reference/` (HF weights) and **TTNN** ports under `tt/`.

## Full TTNN stack

- **PL-BERT + predictor** on device: `ttnn_kokoro_plbert`, `ttnn_kokoro_predictor`, `ttnn_kokoro_albert`, etc.
- **ISTFTNet vocoder** on device: `KokoroDecoderTt` / `KokoroIstftNetTt` (generator uses device `KokoroTtnnSineGen` by default).
- **End-to-end module**: `KokoroFullTtnn` in `tt/ttnn_kokoro_full_pipeline.py` composes the above with **no host torch vocoder**; discrete duration/alignment indices still use small CPU tensors (same as the PyTorch reference predictor).

## SineGen: device vs PyTorch (PCC / demos)

- Default: **`KokoroTtnnSineGen`** on device inside `SourceModuleHnNSF`.
- Toggle: `use_torch_sinegen=True` on `preprocess_kokoro_generator_parameters` / `preprocess_kokoro_decoder_tt_parameters`, or on **`KokoroFullTtnn`**, or **`--torch-sinegen`** on `demo/ttnn_kokoro_full_demo.py`. Harmonics then use the reference **PyTorch** `SineGen` on CPU (uploaded for the same TTNN `Linear` + `tanh` + STFT stack), which usually **raises waveform PCC** vs PyTorch compared to pure device SineGen. See `tests/test_kokoro_generator_pcc.py::test_kokoro_generator_waveform_pcc_sinegen_modes` and `tests/test_kokoro_pcc_sinegen_comparison_report.py` (run with `pytest -s` for the Markdown table).

## Demos

```bash
export PYTHONPATH=$(pwd)
# PyTorch reference (CPU/CUDA)
python models/experimental/kokoro/demo/reference_demo.py --text "Hello." --output out.wav
# Full TTNN (Tenstorrent device + `kokoro` for G2P/voice packs only)
python models/experimental/kokoro/demo/ttnn_kokoro_full_demo.py --text "Hello." --voice af_heart --output out_ttnn.wav
# Same, but PyTorch SineGen for harmonics (compare audio / PCC vs default)
python models/experimental/kokoro/demo/ttnn_kokoro_full_demo.py --torch-sinegen --output out_ttnn_torchsg.wav
```

Install: `pip install "kokoro>=0.9.2" soundfile` and `espeak-ng` on PATH where G2P is needed.

## Tests

From **tt-metal** root:

```bash
export PYTHONPATH=$(pwd)
pytest models/experimental/kokoro/tests/ -v --timeout=600
```

PL-BERT / predictor PCC tests use the `mesh_device` fixture in `tests/conftest.py`. TTNN vocoder PCC tests use the root `device` fixture where applicable.
