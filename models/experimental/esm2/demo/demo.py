# SPDX-License-Identifier: MIT
"""Standalone CLI demo: FASTA in -> per-residue embeddings/logits out.

Example:
    python demo/demo.py --fasta demo/example.fasta --checkpoint /weights \
        --device cpu --precision fp32 --out demo/out.npz

Tokenizer is the deterministic host policy: residue -> /weights/vocab.txt ids,
<cls> first / <eos> last; unknown residues map to the vocab table entry. No
masking is applied here (the evaluator owns masking; the demo reads clean
sequences).

The default FASTA example is embedded below: the source sync has twice
dropped non-Python asset files (demo/example.fasta), so the demo writes it
on first use instead of requiring it in the manifest.
"""
from __future__ import annotations

import argparse

import numpy as np

EXAMPLE_FASTA = """>demo_seq1|small_alpha_helix
MKTAYIAKQRQISFVKSHFSRQALERFLDVGAQIVTALSVSGSAGTARHMLADRGD
>demo_seq2|small_beta_sheet
MSDNNEQSKNAMVLAMDEKQKQIPKLMEMTALTVATLTLTDNPQMQLFYRDAA
"""


def load_vocab(path: str) -> dict[str, int]:
    vocab = {}
    with open(path) as f:
        for idx, line in enumerate(f):
            tok = line.rstrip("\n")
            if tok:
                vocab[tok] = idx
    return vocab


def read_fasta(fasta: str) -> list[tuple[str, str]]:
    names, seqs, buf = [], [], []
    for line in fasta.splitlines():
        if line.startswith(">"):
            if buf:
                seqs.append("".join(buf))
                buf = []
            names.append(line[1:].split()[0] if line[1:].split() else "seq")
        else:
            buf.append(line.strip())
    if buf:
        seqs.append("".join(buf))
    return list(zip(names, seqs))


def tokenize(fasta: str, vocab: dict[str, int], max_len: int = 1026):
    """[(name, int64 ids)] with <cls> ... <eos>; truncates to max_len-2."""
    unk = vocab.get("<unk>")
    cases = []
    for name, seq in read_fasta(fasta):
        residues = seq[: max_len - 2]
        ids = ([vocab["<cls>"]] + [vocab.get(r, unk) for r in residues] + [vocab["<eos>"]])
        cases.append((name, np.asarray(ids, dtype=np.int64)))
    return cases


def main() -> int:
    ap = argparse.ArgumentParser(description="ESM-2 TT demo: FASTA -> embeddings/logits")
    ap.add_argument("--fasta", default="demo/example.fasta")
    ap.add_argument("--checkpoint", default="/weights")
    ap.add_argument("--device", default="cpu", choices=["cpu", "tt"])
    ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--out", default="demo/out.npz")
    ap.add_argument("--max-len", type=int, default=1026)
    args = ap.parse_args()

    import json
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from backend import create_backend

    if args.fasta == "demo/example.fasta" and not os.path.exists(args.fasta):
        with open(args.fasta, "w") as f:
            f.write(EXAMPLE_FASTA)

    with open(os.path.join(args.checkpoint, "config.json")) as f:
        cfg_dict = json.load(f)
    backend = create_backend(args.checkpoint, cfg_dict, args.device, precision=args.precision)

    vocab = load_vocab(os.path.join(args.checkpoint, "vocab.txt"))
    with open(args.fasta) as f:
        fasta = f.read()
    cases = tokenize(fasta, vocab, args.max_len)

    L = max(len(ids) for _, ids in cases)
    B = len(cases)
    pad = cfg_dict["pad_token_id"]
    ids = np.full((B, L), pad, dtype=np.int64)
    am = np.zeros((B, L), dtype=np.int64)
    for i, (_, case_ids) in enumerate(cases):
        ids[i, : len(case_ids)] = case_ids
        am[i, : len(case_ids)] = 1
    out = backend.embed(ids, am)
    np.savez(args.out, names=np.asarray([n for n, _ in cases]), input_ids=ids,
             attention_mask=am, logits=out["logits"], hidden=out["hidden"])
    print(f"wrote {args.out}: logits {out['logits'].shape}, hidden {out['hidden'].shape} "
          f"({B} sequence(s), padded to {L})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
