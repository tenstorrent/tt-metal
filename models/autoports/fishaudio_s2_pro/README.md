# Fish Audio S2 Pro on Tenstorrent Blackhole

Text-to-speech bring-up of [fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro) (4.5B dual-autoregressive
TTS: a Qwen3-4B-shaped "slow" semantic decoder, a 4-layer "fast" codebook decoder over 10 codebooks, and a 44.1 kHz
DAC codec at 21.5 frames/s) for P150 / P300 / P300x2 (QuietBox 2) / P300x4, packaged for
[tt-model-manager](https://github.com/tenstorrent/tt-model-manager) as a `tt-dit-server` container.

Status (Phase A): slow tower on TTNN via `models/tt_transformers` (tensor parallel over the board's chips), frame
embedding + constrained sampling on host, fast decoder + codec in torch on CPU. Later phases move the fast decoder
(Phase B) and the codec decoder (Phase C) onto the device and optimize (Phase D). The `p300x4` profile is declared
but not validated (the author's box has 4 chips).

## Layout

| path | what |
|---|---|
| `config.py` | model constants + `S2Config.from_snapshot()` |
| `model_params/s2-pro/config.json` | Qwen3-shaped view of the text config so `tt_transformers.ModelArgs` can read it |
| `tt/weights.py` | HF fish checkpoint -> tt_transformers state dict (wqkv split, tied head), fast dict, codebook table, codec state, HF "view" dir |
| `tt/slow_model.py` | `S2SlowTransformer(Transformer)`: frame embedding as the residual, post-norm hidden-state tap |
| `tt/generator.py` | `S2Generator`: prompt -> frames -> codes; `teacher_forced()` for accuracy tests |
| `tt/prompt.py`, `tt/sampling.py` | exact ports of fish-speech's prompt construction and sampler (+RAS) |
| `tt/fast_decoder_torch.py` | torch fast decoder (Phase A runtime, and the Phase B reference) |
| `tt/codec/codec_decoder.py` | `CPUCodec` (torch, vendored DAC) with absolute-position streaming decode |
| `tt/device.py` | mesh open per profile (`FISH_S2_MESH_SHAPE`, `FISH_S2_FABRIC_CONFIG`, `TT_METAL_VISIBLE_DEVICES`) |
| `reference/` | trimmed fish-speech + descript-audio-codec code (see `PROVENANCE.md`) |
| `server/` | FastAPI app: `/v1/tts` (fish-speech compatible), `/v1/audio/speech` (OpenAI), references, health |
| `tests/` | CPU parity tests (weights, prompt, sampler, codec) and device tests (layer, slow model) |
| `demo/demo.py` | CLI synthesis to an artifact dir |
| `tt-model.yaml` | the container manifest (4 profiles) |

## Run

```bash
# one chip (p150 profile) on a multi-chip box
export TT_METAL_VISIBLE_DEVICES=0 FISH_S2_MESH_SHAPE=1x1 HF_HUB_OFFLINE=1
python -m models.autoports.fishaudio_s2_pro.demo.demo --text "Hello from Tenstorrent." --out /tmp/hello --greedy
# server (what tt-model runs inside the container)
python -m uvicorn --host 0.0.0.0 --port 20000 --lifespan on models.autoports.fishaudio_s2_pro.server.app:app
curl -s localhost:20000/v1/tts -H 'Content-Type: application/json' -d '{"text":"Hello.","format":"wav"}' -o hello.wav
```

Tests (device): `FISH_S2_MESH_SHAPE=1x1 TT_METAL_VISIBLE_DEVICES=0 pytest -c /dev/null --rootdir models/autoports/fishaudio_s2_pro --confcutdir models/autoports/fishaudio_s2_pro models/autoports/fishaudio_s2_pro/tests/test_slow_layer.py`
(the `--rootdir/--confcutdir` flags keep tt-metal's root conftest out of the way). CPU parity tests need the
upstream fish-speech package (`~/s2pro-bringup/third_party/fish-speech/.venv`).

## Conventions that matter

- Fish RoPE is the interleaved-pair (Meta) convention: weights are used as-is (no `reverse_permute`, no `use_hf_rope`).
- The LM head is tied to the input embedding; `tok_embeddings.weight` and `output.weight` are the same tensor.
- The fast decoder consumes the POST-final-norm hidden state; `LMHead` frees its input, so the tap clones.
- Codebook 0 is `slow_token - 151678`; the fast head is invoked 10x per frame but only codebooks 1..9 are sampled.
- Small profiles on a bigger box must be a true mesh (`TT_METAL_VISIBLE_DEVICES`), not a submesh of the full parent.

Bring-up evidence and the unattended pipeline live in `~/s2pro-bringup/` (see its `docs/PLAN.md` and `status/STATUS.md`).
