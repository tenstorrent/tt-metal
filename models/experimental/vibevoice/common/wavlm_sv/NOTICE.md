# Vendored WavLM speaker-verification code

This package reproduces the **WavLM-large speaker-verification** model that the
[VibeVoice technical report](https://arxiv.org/abs/2508.19205) uses to compute Speaker
Similarity (SIM) — the UniSpeech `wavlm_large_finetune.pth` (WavLM-large backbone + an
ECAPA-TDNN x-vector head) — loaded with **torch only** (no `fairseq`/`s3prl`/`torchaudio`).

## Files and provenance

| File | Source | Notes |
|------|--------|-------|
| `WavLM.py`, `modules.py` | [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/wavlm) (MIT) | WavLM model definition, unmodified except `from modules` → `from .modules`. |
| `models/ecapa_tdnn.py` | [microsoft/UniSpeech](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) | ECAPA-TDNN x-vector head. Patched: `torchaudio`/`fairseq` (`.utils`) imports made optional; a `"__standalone__"` `config_path` builds the torch-only `StandaloneWavLM` feature extractor instead of the s3prl upstream. |
| `wavlm_standalone.py` | Tenstorrent | Torch-only loader: `StandaloneWavLM` reproduces s3prl's 25 hidden states (input-to-each-layer + final-LN encoder output) and input-waveform normalization; `init_model()` builds ECAPA-TDNN(WavLM-large) and loads the fine-tuned weights. |

## Checkpoint

`wavlm_large_finetune.pth` is auto-downloaded (and HF-cached) from the mirror
[`subatomicseer/wavlm-large-sv-ckpts`](https://huggingface.co/subatomicseer/wavlm-large-sv-ckpts).
The canonical source is microsoft/UniSpeech `downstreams/speaker_verification`.

The WavLM-large architecture config in `wavlm_standalone.py` was validated to match the
checkpoint's backbone keys exactly (488/488 params, no shape mismatches); loading is
`strict=False` only to drop the checkpoint's training-time loss head (`loss_calculator.*`).
