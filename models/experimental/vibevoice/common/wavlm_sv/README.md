# Vendored WavLM speaker-verification code

This package reproduces the **WavLM-large speaker-verification** model that the
[VibeVoice technical report](https://arxiv.org/abs/2508.19205) uses to compute Speaker
Similarity (SIM) — the UniSpeech `wavlm_large_finetune.pth` (WavLM-large backbone + an
ECAPA-TDNN x-vector head) — loaded with **torch only** (no `fairseq`/`s3prl`/`torchaudio`).

The vendored files have been **trimmed to the single inference path this integration uses**
(WavLM-large, eval-only, `config_path="__standalone__"`, `update_extract=False`): dead upstream
code for other configurations was removed. The trim is behavior-preserving — every constructed
parameter is kept, so the checkpoint loads with zero missing keys and the speaker embedding is
**bit-identical** to the untrimmed upstream (verified: max-abs-diff 0.0 across all 25 WavLM hidden
states and the final x-vector). To re-vendor a different configuration, pull the originals from the
upstream sources below.

## Files and provenance

| File | Source | Notes |
|------|--------|-------|
| `WavLM.py`, `modules.py` | [microsoft/unilm](https://github.com/microsoft/unilm/tree/master/wavlm) (MIT) | WavLM model definition. Relative-import fix (`from modules` → `from .modules`) plus a dead-code trim: SSL masking, padding-mask handling, `feature_grad_mult` scaling, encoder layerdrop, the conv2d/custom feature extractors, `Fp32GroupNorm`, `GradMultiply`, `Swish`/`GLU_Linear`, `gelu_accurate`, `quant_noise` (`q_noise==0.0`, inlined), and `MultiheadAttention`'s non-torch fallback + incremental-state helpers were removed. Kept: the `layer_norm` extractor, `layer_norm_first` encoder layers, gelu, gated relative-position attention via `F.multi_head_attention_forward`. |
| `models/ecapa_tdnn.py` | [microsoft/UniSpeech](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) | ECAPA-TDNN x-vector head. Trimmed to the `"__standalone__"` `config_path` that builds the torch-only `StandaloneWavLM` front-end; the `fbank`/`mfcc` (torchaudio), s3prl (`torch.hub`) and fairseq (`UpstreamExpert`) front-ends, the trainable-extractor (`update_extract`) path, and the `__main__` demo were removed. |
| `wavlm_standalone.py` | Tenstorrent | Torch-only loader: `StandaloneWavLM` reproduces s3prl's 25 hidden states (input-to-each-layer + final-LN encoder output) and input-waveform normalization; `init_model()` builds ECAPA-TDNN(WavLM-large) and loads the fine-tuned weights. |

## Checkpoint

`wavlm_large_finetune.pth` is auto-downloaded (and HF-cached) from the mirror
[`subatomicseer/wavlm-large-sv-ckpts`](https://huggingface.co/subatomicseer/wavlm-large-sv-ckpts).
The canonical source is microsoft/UniSpeech `downstreams/speaker_verification`.

The WavLM-large architecture config in `wavlm_standalone.py` was validated to match the
checkpoint's backbone keys exactly (no missing keys, no shape mismatches); loading is
`strict=False` only to drop checkpoint entries this inference-only path does not use: the
training-time loss head (`loss_calculator.*`) and the SSL mask embedding
(`feature_extract.model.mask_emb`, unused once masking is removed).

## Licenses and attribution

This directory vendors third-party code under **more than one license**; the notices below must be
preserved by anyone redistributing it.

- **`WavLM.py`, `modules.py`** — © Microsoft Corporation, from
  [microsoft/unilm](https://github.com/microsoft/unilm/blob/master/LICENSE), **MIT License**. The
  original MIT copyright header is retained at the top of each file. Modified by Tenstorrent
  (relative-import fix + the inference-path trim documented above).

- **`models/ecapa_tdnn.py`** and the **`wavlm_large_finetune.pth` checkpoint** — from
  [microsoft/UniSpeech](https://github.com/microsoft/UniSpeech/blob/main/LICENSE)
  (`downstreams/speaker_verification`), licensed **Creative Commons Attribution-ShareAlike 3.0
  Unported (CC BY-SA 3.0)**, <https://creativecommons.org/licenses/by-sa/3.0/>. UniSpeech in turn
  borrows the ECAPA-TDNN blocks from [lawlict/ECAPA-TDNN](https://github.com/lawlict/ECAPA-TDNN),
  which carries **no license file** (default all-rights-reserved) — the governing terms for our
  copy are therefore UniSpeech's CC BY-SA 3.0.
  - *Attribution*: Microsoft UniSpeech authors (Chen et al., WavLM / UniSpeech-SAT speaker
    verification). Keep this notice and the license link with any redistribution.
  - *Changes made* (CC BY-SA requires indicating modification): `models/ecapa_tdnn.py` was trimmed
    to the standalone WavLM front-end — see its header comment and the provenance table above.
  - ⚠️ *ShareAlike*: CC BY-SA 3.0 requires that adaptations of this file and the checkpoint be
    distributed under the same (or a CC-compatible) license. This is **not** compatible with the
    Apache-2.0 license of the surrounding tt-metal repo. Redistribution / packaging of this model
    should be reviewed against this obligation before shipping.

- **`wavlm_standalone.py` and this README** — © Tenstorrent, Apache-2.0 (repo default), as marked by
  the SPDX header in `wavlm_standalone.py`.

Citing the WavLM paper (linked in `WavLM.py`) and the UniSpeech-SAT speaker-verification work is
customary but is not itself a license requirement.
