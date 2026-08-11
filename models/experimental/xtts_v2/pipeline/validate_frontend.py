# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Validate the coqui-free front-end (`frontend.py`) against captured phase-A tensors.

Runs in tt-metal's python_env — no coqui, no torchaudio. Checks, per work dir:

  1. conditioning mel  vs work/cond_mel_in.pt    (coqui hooked the conditioning encoder input)
  2. speaker logmel    vs work/speaker_logmel.pt (coqui hooked the instancenorm input)
  3. prompt prefix     vs work/prefix_emb.pt     — token ids are RECOVERED from the capture
     (subtract the learned position embedding per row, exact-match against the text embedding
     table), so no recorded text is needed. Then:
       a. our assemble_prompt(recovered ids, cond from capture) must equal prefix_emb exactly;
       b. tokenizer round-trip: decode(ids) -> encode() must reproduce ids (BPE + [SPACE] +
          language-tag wrapper; cleaners are idempotent on normalized text).

    XTTS_CKPT=/path/to/xtts_ref/model.pth \
      python models/experimental/xtts_v2/pipeline/validate_frontend.py \
        --work /path/to/xtts_pipeline_out/work --ref-wav <ref.wav or ref.pt>
"""

import argparse
import os

import torch

from models.experimental.xtts_v2.frontend import (
    PromptTables,
    XttsTokenizer,
    assemble_prompt,
    conditioning_mels,
    load_reference_audio,
    speaker_logmel,
)
from models.experimental.xtts_v2.reference.xtts_gpt_ref import resolve_ckpt

LATENTS = 32  # conditioning latents rows at the head of the prefix


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


def check(name, ours, ref, pcc_gate=0.9999):
    p = pcc(ours, ref)
    mx = (ours - ref).abs().max().item()
    ok = p >= pcc_gate and ours.shape == ref.shape
    print(
        f"  {'PASS' if ok else 'FAIL'}  {name:22s} shape {tuple(ours.shape)} vs {tuple(ref.shape)}  pcc={p:.7f}  max|d|={mx:.3e}"
    )
    return ok


def recover_token_ids(prefix_emb, tables):
    """prefix_emb [1,P,1024] -> the text token ids coqui embedded (rows LATENTS..P).
    Row i is text_emb[id] + text_pos[i - LATENTS]; subtracting the position embedding leaves
    an exact row of the embedding table (fp32 capture), so nearest-neighbor distance is ~0."""
    text_rows = prefix_emb[0, LATENTS:, :] - tables.text_pos[: prefix_emb.shape[1] - LATENTS]
    d = torch.cdist(text_rows, tables.text_emb)  # [S, vocab]
    ids = d.argmin(dim=1)
    resid = d.gather(1, ids.unsqueeze(1)).max().item()
    return ids, resid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--ref-wav", required=True, help="reference clip (.wav, or the phase-A .pt)")
    ap.add_argument("--sr", type=int, default=22050, help="sample rate if --ref-wav is a raw .pt tensor")
    ap.add_argument("--vocab", default=None, help="vocab.json (default: alongside $XTTS_CKPT)")
    args = ap.parse_args()

    tables = PromptTables()
    # Derive vocab.json from the RESOLVED checkpoint path — the same source PromptTables
    # reads — so $XTTS_CKPT, $XTTS_CKPT_DIR and the HF-hub fallback all just work.
    vocab = args.vocab or os.path.join(os.path.dirname(resolve_ckpt()), "vocab.json")
    if not os.path.isfile(vocab):
        raise SystemExit(f"vocab.json not found at {vocab}; pass --vocab or put it next to the checkpoint")
    tok = XttsTokenizer(vocab)
    audio, sr = load_reference_audio(args.ref_wav, args.sr)
    print(f"ref {tuple(audio.shape)} @ {sr}Hz   work={args.work}")
    ok = True

    # 1. conditioning mel (Block 1 input)
    ref_mel = torch.load(os.path.join(args.work, "cond_mel_in.pt")).float()
    chunks = conditioning_mels(audio, sr, tables.mel_stats)
    print(f"conditioning_mels: {len(chunks)} chunk(s)")
    ok &= check("cond_mel_in", chunks[0], ref_mel)

    # 2. speaker logmel (Block 2 input)
    ref_lm = torch.load(os.path.join(args.work, "speaker_logmel.pt")).float()
    ok &= check("speaker_logmel", speaker_logmel(audio, sr), ref_lm)

    # 3. prompt prefix (Block 3 input)
    ref_prefix = torch.load(os.path.join(args.work, "prefix_emb.pt")).float()
    ids, resid = recover_token_ids(ref_prefix, tables)
    print(f"recovered {ids.numel()} token ids (nn residual {resid:.3e}) -> {ids[:12].tolist()}...")
    body = ids[1:-1]  # strip START_TEXT/STOP_TEXT; assemble_prompt re-adds them
    text = tok.decode(body)
    lang_tag = text[: text.index("]") + 1] if text.startswith("[") else "[en]"
    print(f"decoded text: {text!r}")
    ours_prefix = assemble_prompt(body, ref_prefix[:, :LATENTS, :], tables)
    ok &= check("prefix_emb (assembly)", ours_prefix, ref_prefix, pcc_gate=1.0 - 1e-9)

    # 3b. tokenizer round-trip on the decoded (already-normalized) text. decode() drops
    # [UNK] (id 1) by design, so a capture whose text contained an un-tokenizable char can
    # only round-trip up to the UNK and its BPE neighborhood — flag those, don't fail them.
    # (The authoritative tokenizer check is a direct cross-check against coqui's tokenizer
    # on raw texts, run in the coqui venv.)
    plain = text[len(lang_tag) :] if text.startswith("[") else text
    re_ids = tok.encode(plain, lang=lang_tag.strip("[]"))
    if 1 in body.tolist():
        print(f"  SKIP  tokenizer round-trip   capture contains [UNK] — not round-trippable by design")
    else:
        match = re_ids == body.tolist()
        print(f"  {'PASS' if match else 'FAIL'}  tokenizer round-trip   {len(re_ids)} ids vs {body.numel()}")
        if not match:
            for i, (a, b) in enumerate(zip(re_ids, body.tolist())):
                if a != b:
                    print(f"    first mismatch at {i}: ours={a} ref={b}")
                    break
        ok &= match

    print("ALL PASS" if ok else "FAILURES — see above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
