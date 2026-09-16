# Vendored reference code

| file | source | commit / version | edits |
|---|---|---|---|
| modded_dac.py | fishaudio/fish-speech `fish_speech/models/dac/modded_dac.py` | befe4001745417f8c42131739d862b8a6fdbd15a | imports rewired to `reference/dac_nn.py`; `AudioSignal` stubbed (compress/decompress unused) |
| rvq.py | fishaudio/fish-speech `fish_speech/models/dac/rvq.py` | befe400 | import of `dac.nn.quantize` rewired |
| modded_dac_vq.yaml | fishaudio/fish-speech `fish_speech/configs/modded_dac_vq.yaml` | befe400 | none (instantiated without hydra by `codec_config.py`) |
| llama_ref.py | fishaudio/fish-speech `fish_speech/models/text2semantic/llama.py` | befe400 | tokenizer/lora imports replaced (see file header) |
| dac_nn.py | descript-audio-codec `dac/nn/layers.py`, `dac/nn/quantize.py`, `dac/model/base.py` (CodecMixin helpers) | 1.0.0 (MIT) | merged into one module; training/compress paths dropped |

fish-speech is distributed under the Fish Audio Research License (non-commercial); descript-audio-codec under MIT.
