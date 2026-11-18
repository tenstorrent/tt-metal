# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Interleaved prefill lengths in one process: every transcript must stay correct.

One encoder and one decoder serve clips of different real lengths in a mixed order,
with no fixed-length padding. Each transcript is scored against its reference text,
and the first clip is transcribed again at the end and must match its first output.

Needs:
  QWEN3ASR_INTERLEAVE_DIR   dir of 16 kHz mono <name>.wav files, each with a <name>.txt
                            reference transcript. Clips are run in the order of
                            QWEN3ASR_INTERLEAVE_ORDER (comma-separated names), default
                            sorted names.
  QWEN3ASR_SNAP_BASE, QWEN3ASR_TEXT_DECODER / HF_MODEL   as for demo/transcribe.py.

Run on one N150:
  QWEN3ASR_INTERLEAVE_DIR=/data/interleave \
  pytest models/demos/audio/qwen3_asr/tests/test_interleaved_lengths.py -s
"""
import os

import pytest
import torch

import ttnn
from models.demos.audio.qwen3_asr.demo import transcribe as T
from models.tt_transformers.tt.model_config import ModelArgs

DEVICE_PARAMS = {"l1_small_size": 32768, "trace_region_size": 200000000, "num_command_queues": 2}
MAX_WER = float(os.environ.get("QWEN3ASR_INTERLEAVE_MAX_WER", "0.25"))


def _words(s):
    return T.re.sub(r"[^a-z0-9' ]+", " ", s.lower()).split()


def _wer(ref, hyp):
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(len(r), 1)


def _clips():
    root = os.environ.get("QWEN3ASR_INTERLEAVE_DIR")
    if not root or not os.path.isdir(root):
        pytest.skip("set QWEN3ASR_INTERLEAVE_DIR to a dir of <name>.wav + <name>.txt")
    order = os.environ.get("QWEN3ASR_INTERLEAVE_ORDER")
    names = order.split(",") if order else sorted(f[:-4] for f in os.listdir(root) if f.endswith(".wav"))
    clips = []
    for n in names:
        with open(os.path.join(root, f"{n}.txt")) as fh:
            clips.append((n, T.load_wav(os.path.join(root, f"{n}.wav")), fh.read().strip()))
    if len({len(T.pad_to_second(w)) for _, w, _ in clips}) < 2:
        pytest.skip("need clips of at least two different lengths")
    return clips


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_interleaved_prefill_lengths(device):
    clips = _clips()
    snap = T.find_snap()
    ckpt = T.find_text_decoder(snap)
    os.environ["HF_MODEL"] = ckpt

    import json

    from safetensors import safe_open
    from transformers import AutoTokenizer, WhisperFeatureExtractor

    tok = AutoTokenizer.from_pretrained(ckpt)
    with open(os.path.join(snap, "chat_template.json")) as fh:
        chat_template = json.load(fh)["chat_template"]
    fe = WhisperFeatureExtractor.from_pretrained(snap)
    with safe_open(os.path.join(ckpt, "model.safetensors"), "pt") as h:
        embed = h.get_tensor("model.embed_tokens.weight").float()
    enc_params = T.tt_enc.preprocess_weights(T.ref.load_audio_tower_weights(snap_dir=snap, dtype=torch.float32), device)
    args = ModelArgs(device, max_batch_size=1, max_seq_len=2048)
    model = T.Qwen3ASRDecoder(
        args,
        ttnn.bfloat16,
        device,
        args.load_state_dict(),
        args.weight_cache_path(ttnn.bfloat16),
        use_paged_kv_cache=False,
    )

    def transcribe(wav):
        input_ids, mel = T.build_inputs(T.pad_to_second(wav), fe, tok, chat_template)
        audio_embeds = T.tt_enc.encode_mel(mel, enc_params, device).float()
        inp = embed[input_ids].clone()
        mask = input_ids == T.AUDIO_TOKEN_ID
        assert int(mask.sum()) == audio_embeds.shape[0], (int(mask.sum()), audio_embeds.shape)
        inp[mask] = audio_embeds
        ids = model.generate(inp.unsqueeze(0), max_new_tokens=256)
        return T.parse_asr(tok.decode(ids, skip_special_tokens=False))[1], input_ids.shape[0]

    failures, first_text = [], None
    for i, (name, wav, ref_text) in enumerate(clips + clips[:1]):
        text, prefill_len = transcribe(wav)
        wer = _wer(ref_text, text)
        print(f"[{i}] {name} {len(wav) / T.SR:.1f}s S={prefill_len} WER={wer:.3f} :: {text!r}")
        if i == 0:
            first_text = text
        elif i == len(clips) and text != first_text:
            failures.append(f"{name} repeat differs: {first_text!r} -> {text!r}")
        if not text or wer > MAX_WER:
            failures.append(f"[{i}] {name} S={prefill_len} WER={wer:.3f}: {text!r}")
    assert not failures, "\n".join(failures)
