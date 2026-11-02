# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run upstream CosyVoice2's frontend over the fixed corpus and write the device-side inputs.

RUN IN THE REFERENCE VENV (see requirements-reference*.txt and scripts/reference_env.py):

    COSYVOICE2_REPO=<upstream checkout> LIBRISPEECH_ROOT=<dir containing LibriSpeech/> \\
        $COSYVOICE2_REF_ENV/bin/python prepare_inputs.py --out-dir <dir> [--parity | --extension]

`--extension` writes the token-accuracy extension (`corpus.extension_cases()`) instead, with its own index file, so
both sets can share one directory.

The frontend is not a network this bring-up ports. It is a text normalizer, the Qwen2 text tokenizer, an ONNX
speech tokenizer on a Whisper log-mel, an ONNX speaker encoder (CAM++) on a Kaldi fbank, and a mel filterbank.
So the boundary sits here, as in CosyVoice1: this runs once and writes one flat `.npz` per case. The TTNN side
loads those files (`tt.prompt.PromptContext.from_npz`) without importing cosyvoice, onnxruntime or whisper.

Per case, exactly as `CosyVoice2.inference_zero_shot` builds it:
- `prompt_text = text_normalize(prompt_text, split=False)`;
- `segments = text_normalize(text, split=True)`;
- then `frontend_zero_shot(segment, prompt_text, prompt_wav, 24000, "")`.

The prompt fields do not depend on the segment, so they are stored once. That includes upstream's
"feat = 2 x tokens" alignment of the prompt mel and speech tokens. Every segment and its token ids are stored
too, so the device-side text path can be checked for parity.

Keys (all `np.ndarray`; strings as 0-d unicode arrays, lists as JSON):
    case_json                           the corpus.cases() entry (ids, texts, prompt wav path, set, mode, lang)
    prompt_text_ids          int32 [1, P]
    llm_prompt_speech_tokens int32 [1, S]    (zero_shot: the same as the flow's)
    flow_prompt_speech_tokens int32 [1, S]
    prompt_feat              float32 [1, 2S, 80]
    llm_embedding            float32 [1, 192]
    flow_embedding           float32 [1, 192]
    segments_json            JSON list[str]: upstream's normalized, split segments of `text`
    segment_text_ids_json    JSON list[list[int]]: upstream's token ids for each segment
    meta_json                versions (upstream commit, torch, onnxruntime, ...), corpus version
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus  # noqa: E402
import reference_env  # noqa: E402


def prompt_wav_path(case: dict) -> str:
    base = reference_env.upstream_repo() if case["set"] == "cosyvoice1_parity" else reference_env.librispeech_root()
    return os.path.join(base, case["prompt_wav"])


def build_case(model, case: dict) -> dict:
    fe = model.frontend
    prompt_text = fe.text_normalize(case["prompt_text"], split=False)
    segments = fe.text_normalize(case["text"], split=True)
    wav = prompt_wav_path(case)
    first = fe.frontend_zero_shot(segments[0], prompt_text, wav, model.sample_rate, "")
    seg_ids = [fe.frontend_zero_shot(s, prompt_text, wav, model.sample_rate, "")["text"].tolist()[0] for s in segments]
    assert seg_ids[0] == first["text"].tolist()[0]
    s = int(first["llm_prompt_speech_token_len"][0])
    assert first["prompt_speech_feat"].shape[1] == 2 * s, "upstream's feat == 2 x tokens alignment"

    def arr(t, dtype):
        return t.detach().cpu().numpy().astype(dtype)

    return {
        "case_json": np.array(json.dumps(case, ensure_ascii=False)),
        "prompt_text_ids": arr(first["prompt_text"], np.int32),
        "llm_prompt_speech_tokens": arr(first["llm_prompt_speech_token"], np.int32),
        "flow_prompt_speech_tokens": arr(first["flow_prompt_speech_token"], np.int32),
        "prompt_feat": arr(first["prompt_speech_feat"], np.float32),
        "llm_embedding": arr(first["llm_embedding"], np.float32),
        "flow_embedding": arr(first["flow_embedding"], np.float32),
        "segments_json": np.array(json.dumps(segments, ensure_ascii=False)),
        "segment_text_ids_json": np.array(json.dumps(seg_ids)),
        "prompt_text_normalized": np.array(prompt_text),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--extension", action="store_true", help="the token-accuracy extension instead (corpus.py)")
    ap.add_argument("--parity", action="store_true", help="also write the CosyVoice1-parity case (secondary)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = reference_env.load_upstream()
    meta = {"corpus_version": corpus.CORPUS_VERSION, **reference_env.versions()}
    index = []
    for case in corpus.extension_cases() if args.extension else corpus.cases(include_parity=args.parity):
        fields = build_case(model, case)
        fields["meta_json"] = np.array(json.dumps(meta))
        name = f"{case['case_id']}.npz"
        np.savez(os.path.join(args.out_dir, name), **fields)
        segs = json.loads(str(fields["segments_json"]))
        index.append({"file": name, **{k: case[k] for k in ("case_id", "set", "mode", "lang", "speaker")}})
        print(
            f"  {case['case_id']:<40} prompt {fields['llm_prompt_speech_tokens'].shape[1]:4d} tok, "
            f"{len(segs)} segment(s), text ids {[len(x) for x in json.loads(str(fields['segment_text_ids_json']))]}",
            flush=True,
        )
    index_name = "index_extension.json" if args.extension else "index.json"  # the two sets can share a directory
    with open(os.path.join(args.out_dir, index_name), "w") as fh:
        json.dump({"meta": meta, "cases": index}, fh, indent=2, ensure_ascii=False)
    print(f"wrote {len(index)} cases to {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
