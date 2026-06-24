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
# in the eval venv (needs the base Qwen3-ASR-1.7B in the HF cache; QWEN3ASR_SNAP_BASE to override)
rm -f /home/ttuser/ttwork/qwen3_asr_text_decoder/model.safetensors   # drop the wrong file
python models/demos/audio/qwen3_asr/reference/extract_text_decoder.py \
    --ckpt-out /home/ttuser/ttwork/qwen3_asr_text_decoder --skip-golden
```
This writes 311 decoder tensors (`thinker.` stripped) + a Qwen3 `config.json` + tokenizer.
Keep the HF cache mounted too — the audio tower is loaded from the raw snapshot.

## Dedicated container (recommended) — boots straight into the API server
```bash
# build the image once (build ctx bakes deps + qwen_asr processor + entrypoint):
docker build -t qwen3-asr-server:latest /home/ttuser/ttwork/qwen3asr-server-build

# launch (chip 3 = fake P150, publishes :8002 to the host):
bash server/run.sh            # -> container `qwen3-asr-api`, READY in ~25s
curl http://127.0.0.1:8002/health
curl -s -X POST http://127.0.0.1:8002/v1/audio/transcriptions -F file=@clip.wav -F model=qwen3-asr
```
The entrypoint wires ttnn to the mounted tt-metal (`uv pip install -e /work`, ~9s) then runs
uvicorn. Mounts: tt-metal→/work, extracted checkpoint→/models/qwen3_asr_text_decoder, HF cache.
First request of each new prefill length pays a one-time JIT (~1.5 RTF); later requests ~0.1.
(No silence warmup — it corrupted the reused KV cache. Server uses `use_trace=False` for robustness;
reusing one decode trace in-server is a TODO that would bring RTF back to ~0.05.)

### Dev-container alternative (manual)
`docker exec qwen3asr-dev bash /work/models/demos/audio/qwen3_asr/server/setup_container.sh` then
launch uvicorn manually; see `setup_container.sh`. The dedicated image above is preferred.

## Transparent `--engine` in qwen3-asr-eval
`asr_engines.py` gained a `ttqwen3asr` engine (a `TTWhisperEngine` pointed at :8002,
model `qwen3-asr`) — identical OpenAI client as `ttwhisper`. Use it anywhere whisper is used:
```bash
python stream_linein.py --engine ttqwen3asr --asr-api http://127.0.0.1:8002 ...
```
Verified: EN → English (byte-identical to CPU Qwen3-ASR), JA → Japanese (auto-detected),
RTF ~0.05–0.1. Same model handles both with no language hint, where ttwhisper (English-forced)
cannot transcribe Japanese.
