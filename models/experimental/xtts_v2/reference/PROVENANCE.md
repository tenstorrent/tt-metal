# XTTS-v2 reference — provenance

PyTorch "golden" reference for the TTNN port of Coqui XTTS-v2. Every TTNN module
we build will be PCC-validated against these files.

## Source
- Repo: https://github.com/idiap/coqui-ai-TTS (maintained fork of the discontinued `coqui-ai/TTS`)
- Commit: `ca2cf5155bca892ea820ad384400efbfac41b178`
- Fetched: 2026-07-21
- Weights (not vendored): HuggingFace `coqui/XTTS-v2`

## License — read before upstreaming
- **Code** (these files): Mozilla Public License 2.0 (MPL-2.0). File-level copyleft.
  tt-metal is Apache-2.0, so MPL files must keep their license/notices if kept in-tree.
- **Pretrained weights** (`coqui/XTTS-v2`): Coqui Public Model License (CPML), **non-commercial**.
  Fine for research/bring-up; confirm with Tenstorrent legal before any product use.

## Files (original path preserved under TTS/)
| File | Role |
|------|------|
| `TTS/tts/models/xtts.py` | Top-level `Xtts` model — orchestration + inference API |
| `TTS/tts/configs/xtts_config.py` | Hyperparameters / shapes (`XttsArgs`, `XttsAudioConfig`, `XttsConfig`) |
| `TTS/tts/layers/xtts/gpt.py` | `GPT` — autoregressive core (produces audio-code tokens + latents) |
| `TTS/tts/layers/xtts/gpt_inference.py` | `GPT2InferenceModel` — HF GPT-2 wrapper with KV-cache for generation |
| `TTS/tts/layers/xtts/hifigan_decoder.py` | `HifiDecoder` — vocoder + speaker-encoder glue (GPT latents → waveform) |
| `TTS/tts/layers/xtts/perceiver_encoder.py` | `PerceiverResampler` — conditioning resampler |
| `TTS/tts/layers/xtts/tokenizer.py` | `VoiceBpeTokenizer` — multilingual BPE + text cleaning |
| `TTS/tts/layers/xtts/stream_generator.py` | Streaming generation helpers |
| `TTS/tts/layers/xtts/dvae.py` | `DiscreteVAE` — mel↔code tokens (training-time; not on inference path) |
| `TTS/tts/layers/xtts/zh_num2words.py` | Chinese number normalization (tokenizer dep) |
| `TTS/tts/layers/tortoise/autoregressive.py` | `ConditioningEncoder`, `LearnedPositionEmbeddings`, GPT-2 build helpers |
| `TTS/tts/layers/tortoise/arch_utils.py` | `AttentionBlock`, norms, mel-spectrogram helpers |
| `TTS/tts/layers/tortoise/transformer.py` | `GEGLU` + transformer blocks (perceiver dep) |
| `TTS/tts/layers/tortoise/xtransformers.py` | Positional-bias / attention utilities |
| `TTS/encoder/models/resnet.py` | `ResNetSpeakerEncoder` — speaker embedding (voice identity) |
| `TTS/encoder/models/base_encoder.py` | Base class for the speaker encoder |
| `TTS/vocoder/models/hifigan_generator.py` | `HifiganGenerator` — the actual HiFi-GAN (transposed convs + MRF) |
| `TTS/utils/generic_utils.py` | Small helpers (`exists`, `default`, version checks) |

## Not vendored (deliberately) — external glue to resolve next step
These are imported by the files above but NOT copied, because they are framework/
training code, not architecture we intend to port:
- `TTS.tts.models.base_tts.BaseTTS`, `TTS.config.*`, `TTS.tts.configs.shared_configs.*` (Coqpit config framework)
- `TTS.tts.layers.xtts.xtts_manager.SpeakerManager`
- `TTS.encoder.losses.*` (training losses, imported by `base_encoder`)
- `TTS.tts.utils.text.cleaners` + `chinese_mandarin/*` (tokenizer text-normalization leaves)
- `trainer.io.load_fsspec`, `coqpit` (pip packages)

Next step decision: either `pip install coqui-tts` to satisfy these at runtime, or
trim the vendored files to be self-contained (drop training-only code paths).
