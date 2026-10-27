# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Upstream's own streaming synthesis on fixed tokens: the reference for the streaming gate (stage A, no LLM).

RUN IN THE REFERENCE VENV (see requirements-reference*.txt and scripts/reference_env.py):

    COSYVOICE2_REPO=<upstream checkout> LIBRISPEECH_ROOT=<dir containing LibriSpeech/> \\
        $COSYVOICE2_REF_ENV/bin/python streaming_reference.py --inputs <prepare_inputs.py dir> \\
        --tokens-from <a run dir's results.json, e.g. the demo's> --out-dir <dir>

For each case, the tokens are fixed (the run's `segment_tokens`; one segment per case), and upstream's
`CosyVoice2Model.tts` streaming loop runs over them as if the LLM had already finished (cosyvoice/cli/model.py
:343-374 at 074ca6dc9e80, copied step for step):
- a chunk goes out whenever `hop + pre_lookahead_len` (3) tokens past the offset exist: `token2wav(tokens[: offset +
  hop + 3], offset, stream=True, finalize=False)`;
- the first hop is 25 plus the prompt pad (`ceil(P / 25) * 25 - P`); after each chunk the hop doubles, up to 100;
- then the final chunk: `token2wav(all tokens, offset, finalize=True)`, which leaves `stream` at its default, False.
The hop starts at 25 for every case: upstream's `tts` never resets it (:360), the TT port resets it per utterance
(the notes branch's D3), and the gate compares like with like.

Captured per chunk: the flow's mel of the new frames; each HiFT call's input mel (the 8 cached frames first, when
there are any), its F0 and its sine noise (a fixed draw per call, injected in place of SineGen2's `randn_like`), and
the audio the chunk emits. Also upstream's non-streaming mel of all the tokens (one `finalize=True, streaming=False`
flow call), the control the per-chunk streaming mel must differ from.

Writes `<case_id>.npz` and a `results.json` with the streamed wavs in the schema scripts/eval_wer_sim.py scores.
npz keys: `tokens` [N], `offsets` / `hops` [chunks] (the final chunk's hop is the tokens it takes), `mel_<k>`
[1, frames_k, 80] the chunk's new frames, `hift_mel_<k>`, `hift_f0_<k>`, `hift_noise_<k>`, `speech_<k>` (emitted
audio), `nonstreaming_mel` [1, 2N, 80], `audio` (all chunks' speech, concatenated).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys

import numpy as np
import soundfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reference_env  # noqa: E402

SEED = 1986
PRE_LOOKAHEAD = 3


@contextlib.contextmanager
def injected_sine_noise(torch, noise):
    """SineGen2 draws `torch.randn_like(sine_waves)`, `[1, L, 9]`; hand it `noise` instead. Other draws pass."""
    original = torch.randn_like

    def randn_like(t, *a, **k):
        return noise.to(t.dtype) if tuple(t.shape) == tuple(noise.shape) else original(t, *a, **k)

    torch.randn_like = randn_like
    try:
        yield
    finally:
        torch.randn_like = original


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--tokens-from", required=True, help="a run directory whose results.json has segment_tokens")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--noise-seed", type=int, default=SEED, help="HiFT call n's noise: seed x 1000 + n")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    import torch

    model = reference_env.load_upstream()
    cv = model.model  # CosyVoice2Model
    flow, hift = cv.flow, cv.hift
    with open(os.path.join(args.tokens_from, "results.json")) as fh:
        runs = json.load(fh)["results"]

    # capture the flow's output and HiFT's inputs and outputs, and inject each HiFT call's noise
    flow_out, hift_calls, f0_calls = [], [], []
    flow_inference, hift_inference, f0_forward = flow.inference, hift.inference, hift.f0_predictor.forward

    def recording_flow(*a, **k):
        mel, extra = flow_inference(*a, **k)
        flow_out.append(mel.detach().clone())
        return mel, extra

    def recording_f0(x):
        f0 = f0_forward(x)
        f0_calls.append(f0.detach().clone())
        return f0

    def recording_hift(speech_feat, cache_source=torch.zeros(1, 1, 0), **k):
        n = len(hift_calls)
        frames = speech_feat.shape[2]
        noise = torch.randn(1, frames * 480, 9, generator=torch.Generator().manual_seed(args.noise_seed * 1000 + n))
        with injected_sine_noise(torch, noise):
            speech, source = hift_inference(speech_feat=speech_feat, cache_source=cache_source, **k)
        hift_calls.append({"mel": speech_feat.transpose(1, 2).detach().clone(), "noise": noise, "f0": f0_calls[-1]})
        return speech, source

    flow.inference = recording_flow
    hift.inference = recording_hift
    hift.f0_predictor.forward = recording_f0

    results = []
    for run in runs:
        case_id = run["case_id"]
        assert len(run["segment_tokens"]) == 1, "single-segment cases only"
        tokens = [int(t) for t in run["segment_tokens"][0]]
        d = np.load(os.path.join(args.inputs, f"{case_id}.npz"))
        prompt_token = torch.from_numpy(d["flow_prompt_speech_tokens"].astype(np.int64)).int()
        prompt_feat = torch.from_numpy(d["prompt_feat"])
        embedding = torch.from_numpy(d["flow_embedding"])
        uuid = f"streaming_reference_{case_id}"
        cv.hift_cache_dict[uuid] = None
        cv.token_hop_len = 25  # per utterance, as the TT port does (see the module docstring)
        flow_out.clear()
        hift_calls.clear()
        out = {"tokens": np.array(tokens, dtype=np.int32)}
        offsets, hops, pieces = [], [], []

        def chunk(n_tok, offset, finalize):
            k = len(offsets) - 1  # this chunk's offset is already appended
            before = len(hift_calls)
            with torch.inference_mode():
                kw = dict(finalize=True) if finalize else dict(stream=True, finalize=False)
                speech = cv.token2wav(
                    token=torch.tensor([tokens[:n_tok]], dtype=torch.int32),
                    prompt_token=prompt_token,
                    prompt_feat=prompt_feat,
                    embedding=embedding,
                    token_offset=offset,
                    uuid=uuid,
                    **kw,
                )
            assert len(hift_calls) == before + 1
            call = hift_calls[-1]
            out[f"mel_{k}"] = flow_out[-1][:, :, offset * 2 :].transpose(1, 2).float().numpy()
            out[f"hift_mel_{k}"] = call["mel"].float().numpy()
            out[f"hift_f0_{k}"] = call["f0"].reshape(1, -1).float().numpy()
            out[f"hift_noise_{k}"] = call["noise"].numpy()
            out[f"speech_{k}"] = speech.reshape(-1).float().numpy()
            pieces.append(out[f"speech_{k}"])

        # upstream's tts() streaming loop, with every token already generated
        n_prompt = prompt_token.shape[1]
        prompt_token_pad = int(math.ceil(n_prompt / cv.token_hop_len) * cv.token_hop_len - n_prompt)
        offset = 0
        while True:
            hop = cv.token_hop_len + prompt_token_pad if offset == 0 else cv.token_hop_len
            if len(tokens) - offset < hop + PRE_LOOKAHEAD:
                break
            offsets.append(offset)
            hops.append(hop)
            chunk(offset + hop + PRE_LOOKAHEAD, offset, finalize=False)
            offset += hop
            cv.token_hop_len = min(cv.token_max_hop_len, cv.token_hop_len * cv.stream_scale_factor)
        offsets.append(offset)
        hops.append(len(tokens) - offset)
        chunk(len(tokens), offset, finalize=True)
        cv.hift_cache_dict.pop(uuid)

        # the control: upstream's non-streaming mel of all the tokens
        with torch.inference_mode():
            nonstreaming, _ = flow_inference(
                token=torch.tensor([tokens], dtype=torch.int32),
                token_len=torch.tensor([len(tokens)], dtype=torch.int32),
                prompt_token=prompt_token,
                prompt_token_len=torch.tensor([n_prompt], dtype=torch.int32),
                prompt_feat=prompt_feat,
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32),
                embedding=embedding,
                streaming=False,
                finalize=True,
            )
        out["nonstreaming_mel"] = nonstreaming.transpose(1, 2).float().numpy()
        out["offsets"] = np.array(offsets, dtype=np.int32)
        out["hops"] = np.array(hops, dtype=np.int32)
        audio = np.concatenate(pieces).astype(np.float32)
        out["audio"] = audio
        np.savez(os.path.join(args.out_dir, f"{case_id}.npz"), **out)
        name = f"{case_id}.wav"
        soundfile.write(os.path.join(args.out_dir, name), audio, 24000)
        results.append(
            {k: v for k, v in run.items() if k not in ("wav", "segment_tokens")}
            | {
                "wav": name,
                "audio_s": round(len(audio) / 24000, 3),
                "segment_tokens": [tokens],
                "streaming_chunks": len(offsets),
            }
        )
        print(
            f"  {case_id}: {len(tokens)} tokens, prompt {n_prompt} (pad {prompt_token_pad}); chunks at offsets "
            f"{offsets} with hops {hops}; HiFT frames {[out[f'hift_mel_{k}'].shape[1] for k in range(len(offsets))]}",
            flush=True,
        )
    run = {
        "backend": "pytorch-reference-streaming",
        "device": "cpu",
        "meta": {"tokens_from": os.path.abspath(args.tokens_from), **reference_env.versions()},
        "results": results,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump(run, fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(results)} cases to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
