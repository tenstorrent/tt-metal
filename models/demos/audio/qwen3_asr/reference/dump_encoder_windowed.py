# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Golden audio-tower outputs for clips of several lengths, with vLLM's windowed attention.

vLLM serves the audio tower with block-diagonal attention over `cu_seqlens` windows of
13 * (n_window_infer // 100) = 104 tokens. HF's sdpa/eager path ignores `cu_seqlens` and
attends over the whole clip, so this script adds the block-diagonal mask HF builds in
`Qwen3ASRAudioEncoder._prepare_attention_mask` to every layer.

For each length it writes `<name>.mel.npy` (128, T) and `<name>.audio_tower.npy`
(feat_out_len(T), 2048). Lengths that are not whole seconds exercise the partial last
chunk, whose padded tokens the model drops.

Run with a venv that has `qwen_asr` (it pins its own transformers):
    python models/demos/audio/qwen3_asr/reference/dump_encoder_windowed.py \
        --wav speech_30s.wav --out /data/qwen3_asr_encoder_golden
"""
import argparse
import os

import numpy as np
import soundfile as sf
import torch

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
SR = 16000
DEFAULT_SECONDS = (0.7, 3.25, 8.0, 9.2, 16.13, 29.37, 30.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True, help="16 kHz mono speech, at least as long as the longest clip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, nargs="+", default=list(DEFAULT_SECONDS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from qwen_asr import Qwen3ASRModel
    from transformers import WhisperFeatureExtractor

    torch.set_num_threads(os.cpu_count())
    wrap = Qwen3ASRModel.from_pretrained(MODEL_ID, dtype=torch.float32, device_map="cpu")
    tower = next(m for m in wrap.model.modules() if m.__class__.__name__ == "Qwen3ASRAudioEncoder")
    tower.eval()

    for layer in tower.layers:
        attn_forward = layer.self_attn.forward

        def windowed(hidden_states, cu_seqlens=None, attention_mask=None, _fwd=attn_forward, **kw):
            mask = torch.full(
                (1, 1, hidden_states.shape[0], hidden_states.shape[0]), torch.finfo(hidden_states.dtype).min
            )
            for a, b in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
                mask[..., a:b, a:b] = 0
            return _fwd(hidden_states, cu_seqlens=cu_seqlens, attention_mask=mask, **kw)

        layer.self_attn.forward = windowed

    wav, sr = sf.read(args.wav, dtype="float32")
    if sr != SR:
        raise ValueError(f"expected {SR} Hz, got {sr}")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    fe = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    for sec in args.seconds:
        clip = wav[: int(round(sec * SR))]
        if len(clip) < int(round(sec * SR)):
            raise ValueError(f"wav shorter than {sec} s")
        feats = fe(clip, sampling_rate=SR, return_attention_mask=True, return_tensors="pt", padding=True)
        flen = int(feats["attention_mask"].sum())
        mel = feats["input_features"][0][:, :flen].float().contiguous()
        with torch.inference_mode():
            out = tower(mel, feature_lens=torch.tensor([flen])).last_hidden_state
        name = f"clip_{sec:g}s".replace(".", "p")
        np.save(os.path.join(args.out, f"{name}.mel.npy"), mel.numpy())
        np.save(os.path.join(args.out, f"{name}.audio_tower.npy"), out.float().numpy())
        print(f"{name}: T={flen} tokens={out.shape[0]}", flush=True)


if __name__ == "__main__":
    main()
