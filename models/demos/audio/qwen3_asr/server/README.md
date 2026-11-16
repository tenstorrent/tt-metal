# Qwen3-ASR TT API server (OpenAI `/v1/audio/transcriptions`)

Serves TT Qwen3-ASR-1.7B on a Blackhole P150 with the **same HTTP contract as the
tt-media-server whisper endpoint**, so the `qwen3-asr-eval` pipeline (and any whisper
client) targets it transparently. Multilingual: omit `language` → auto-detect.

```
POST /v1/audio/transcriptions   (multipart: file=<wav>, model=<str>, language=<optional>)
  -> {"text": ..., "language": ..., "duration": ..., "rtf": ...}
GET  /health -> {"status": "ok"}
```

## Checkpoint setup — the decoder checkpoint must be the EXTRACTED one
The server loads the **text decoder** from `${CKPT}/model.safetensors` with **unprefixed**
keys (`model.embed_tokens.weight`, `model.layers.*`, `model.norm.weight`, `lm_head.weight`).
The raw HF `Qwen/Qwen3-ASR-1.7B` stores these under `thinker.model.*` / `thinker.lm_head`
(the audio encoder is `thinker.audio_tower.*`, loaded separately). So the mounted
`qwen3_asr_text_decoder/` must be the **extracted** checkpoint, NOT the raw HF model.

`entrypoint.sh` auto-extracts it on first run via `reference/extract_text_decoder.py` —
**but only when `$CKPT/model.safetensors` and `config.json` are both absent**. If a *wrong*
`model.safetensors` (e.g. the raw HF model) is already sitting at the mount, extraction is
**skipped** and startup fails with:
```
safetensors_rust.SafetensorError: File does not contain tensor model.embed_tokens.weight
```
Fix: clear the bad checkpoint and let the entrypoint re-extract, or extract manually:
```bash
# in the CPU-reference venv (requirements-reference.txt); needs the base Qwen3-ASR-1.7B in
# the HF cache (QWEN3ASR_SNAP_DIR / QWEN3ASR_SNAP_BASE to override)
rm -f "$CKPT/model.safetensors"          # drop the wrong file
/tmp/qwen3-asr-ref/bin/python models/demos/audio/qwen3_asr/reference/extract_text_decoder.py \
    --ckpt-out "$CKPT" --skip-golden
```
This writes 311 decoder tensors (`thinker.` stripped) + a Qwen3 `config.json` + tokenizer.
Keep the HF cache mounted too — the audio tower is loaded from the raw snapshot.

## Dedicated container (recommended) — boots straight into the API server
```bash
# Build from a clean checkout — the build context is the model dir, so the pinned
# requirement files and the entrypoint all come from the repo (no external context):
docker build -f models/demos/audio/qwen3_asr/server/Dockerfile \
             -t qwen3-asr-server:latest models/demos/audio/qwen3_asr

# launch (chip 3 = fake P150, publishes :8002 to the host):
bash server/run.sh            # -> container `qwen3-asr-api`, READY in ~25s
```

Smoke-test it with any 16 kHz mono wav — `language` is optional (omit it for auto-detect):
```bash
curl http://127.0.0.1:8002/health
# -> {"status":"ok"}

curl -s -X POST http://127.0.0.1:8002/v1/audio/transcriptions \
     -F file=@clip.wav -F model=qwen3-asr
# -> {"text":"...","language":"English","duration":11.0,"rtf":0.081,"segments":1,"model":"qwen3-asr"}

# force a language instead of auto-detecting:
curl -s -X POST http://127.0.0.1:8002/v1/audio/transcriptions \
     -F file=@clip.wav -F model=qwen3-asr -F language=Japanese
```

The image installs the pinned Qwen3-ASR processor from PyPI with `--no-deps`
(`requirements-processor.txt`); it no longer copies a `qwen_asr` tree out of a machine-local
venv. See the top-level README "Install" section for why the two environments are separate.
The entrypoint wires ttnn to the mounted tt-metal (`uv pip install -e /work`, ~9s) then runs
uvicorn. Mounts: tt-metal→/work, extracted checkpoint→/models/qwen3_asr_text_decoder, HF cache.
First request of each new prefill length pays a one-time JIT (~1.5 RTF); later requests ~0.1.
(Server uses host-side argmax greedy decode for robustness across mixed request shapes; a reused
in-server decode trace was prototyped but removed — it destabilized the long-lived server — and
re-landing it stably (RTF back to ~0.05) is a TODO.)

### Dev-container alternative (manual)
`docker exec qwen3asr-dev bash /work/models/demos/audio/qwen3_asr/server/setup_container.sh`
installs the same pinned deps into a stock tt-metal dev container (and fails loudly if the
processor cannot be imported), then launch uvicorn manually. The dedicated image above is
preferred.

## Transparent `--engine` in qwen3-asr-eval
`asr_engines.py` gained a `ttqwen3asr` engine (a `TTWhisperEngine` pointed at :8002,
model `qwen3-asr`) — identical OpenAI client as `ttwhisper`. Use it anywhere whisper is used:
```bash
python stream_linein.py --engine ttqwen3asr --asr-api http://127.0.0.1:8002 ...
```
Verified: EN → English (byte-identical to CPU Qwen3-ASR), JA → Japanese (auto-detected),
RTF ~0.05–0.1. Same model handles both with no language hint, where ttwhisper (English-forced)
cannot transcribe Japanese.
