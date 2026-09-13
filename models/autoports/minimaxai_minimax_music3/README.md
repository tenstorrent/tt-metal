# MiniMax Music 3 on Tenstorrent Blackhole (single chip)

Bring-up of [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3) (lyrics + caption -> song):
an 8B Qwen3-based global LLM (one decode step per 40 ms frame, classifier-free guidance as a second row), a 4-layer
depth decoder for the 7 residual RVQ codebooks, a 36-layer flow-matching DiT over 200-frame windows and a DAC-style
vocoder, on ONE Blackhole chip (P150; chip 0 of a P300 / QuietBox 2), packaged for
[tt-model-manager](https://github.com/tenstorrent/tt-model-manager) as a `tt-dit-server` container.

| path | what |
|---|---|
| `config.py` | the checkpoint's inference contract (token ids, CFG scales, window/overlap constants) + sub-model configs |
| `reference/` | torch port of the diffusers PR #14456 pipeline (`PROVENANCE.md`): CPU goldens, PCC references, and the CPU stages of serving (condition encoder, vocoder, scheduler loop) |
| `tt/weights.py` | HF view dirs for `tt_transformers.ModelArgs` (incl. the SLICED 16 385-row LM head / embedding), loaders |
| `tt/backbone.py` | `Music3Backbone(Transformer)`: host frame embeddings as the residual, 2 users (cond/uncond), post-norm hidden tap |
| `tt/depth_decoder.py` | TTNN depth decoder: one traced 32-row causal step replayed 7x per frame |
| `tt/dit.py` | TTNN DiT: padded rows, masked SDPA, partial RoPE via the per-tile ROPE op, traced per window length |
| `tt/generator.py` | `Music3Generator`: the whole recipe with the TT modules; `GenStats` |
| `tt/device.py` | true 1x1 mesh (`TT_METAL_VISIBLE_DEVICES=0`), `MUSIC3_MESH_SHAPE` |
| `server/` | FastAPI: `/v1/audio/speech` (SGLang-Omni compatible), `/v1/music/jobs*`, `/v1/health`, `/v1/models` |
| `tests/` | CPU parity (prompt / sampling / modules vs diffusers) and device tests (stages 03-06) |
| `demo/demo.py` | CLI |
| `tt-model.yaml` | the container manifest (profile `p150`) |

## Run

```bash
export TT_METAL_VISIBLE_DEVICES=0 MUSIC3_MESH_SHAPE=1x1 HF_HUB_OFFLINE=1
python -m models.autoports.minimaxai_minimax_music3.demo.demo --caption "acoustic pop, 96 BPM, warm female vocal" \
  --lyrics $'[verse]\nMorning light filtering through the pine\n[chorus]\nSoftly the world begins to breathe' --duration 20 --out /tmp/song
python -m uvicorn --host 0.0.0.0 --port 20000 --lifespan on models.autoports.minimaxai_minimax_music3.server.app:app
```

Device tests: `pytest -c /dev/null --rootdir models/autoports/minimaxai_minimax_music3 --confcutdir models/autoports/minimaxai_minimax_music3 models/autoports/minimaxai_minimax_music3/tests/test_backbone.py`
(env `MUSIC3_SNAPSHOT`, `MUSIC3_GOLDEN_ROOT` from the bring-up work dir `~/music3-bringup`).

## Conventions that matter

- The prompt template, CFG rows, sampling (top-50, CFG 1.5), window (200/100) and overlap (172/86/258) constants are the
  checkpoint's contract; `reference/` is asserted equal to diffusers in stage 01/02.
- The device LM head is sliced to the 16 384 semantic rows + `<|audio_end|>`; the mask makes this exact (stage 02 proof).
- The depth decoder consumes the POST-final-norm LLM hidden state of BOTH rows; `LMHead` frees its input, so the tap clones.
- `torch.Generator` consumption order (c0, c1..c7 per frame, then one `randn` per window) matches diffusers, so a seed
  is a reproducible trajectory given identical logits.
- Single-chip profiles on a bigger box are a true mesh (`TT_METAL_VISIBLE_DEVICES`), never a submesh of the parent.

Bring-up evidence and the unattended pipeline live in `~/music3-bringup/` (`docs/PLAN.md`, `status/STATUS.md`).
