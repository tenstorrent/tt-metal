"""Verify that a tokenized .bin shard matches its source parquet file.

Re-tokenizes documents from ``--parquet`` with the same tokenizer used by
``dataset_preprocessing.py`` (TinyLlama, BOS/EOS-wrapped, no other special
tokens) and compares the resulting uint16 stream against ``--bin`` (loaded as
a ``np.memmap``). Also cross-checks the shard against ``meta.json`` if one is
present alongside the bin.

Defaults verify the validation split:
    /data/awliu/datasets/SlimPajama-6B-tokenized/validation/validation_000000.bin
    /data/awliu/datasets/SlimPajama-6B/data/validation-00000-of-00001-4fb685c22a3f91ef.parquet

Usage:
    python read_bin_tokens.py                 # full verify (re-tokenize all docs)
    python read_bin_tokens.py --max-docs 1000 # quick spot check on first N docs
    python read_bin_tokens.py --preview 64    # also print first N decoded tokens
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

DEFAULT_BIN = Path("/data/awliu/datasets/SlimPajama-6B-tokenized/validation/validation_000000.bin")
DEFAULT_PARQUET = Path("/data/awliu/datasets/SlimPajama-6B/data/" "validation-00000-of-00001-4fb685c22a3f91ef.parquet")
TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DTYPE = np.uint16


def load_meta(bin_path: Path) -> dict | None:
    meta_path = bin_path.parent / "meta.json"
    if not meta_path.exists():
        return None
    with meta_path.open() as f:
        return json.load(f)


def parquet_texts(path: Path, text_field: str = "text", batch_size: int = 1000):
    """Yield document strings from a parquet file in source order."""
    pf = pq.ParquetFile(str(path))
    for record_batch in pf.iter_batches(batch_size=batch_size, columns=[text_field]):
        for text in record_batch.column(text_field).to_pylist():
            if text:
                yield text


def encode_doc(tokenizer, text: str, bos_id: int, eos_id: int) -> np.ndarray:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    out = np.empty(len(ids) + 2, dtype=DTYPE)
    out[0] = bos_id
    out[1:-1] = ids
    out[-1] = eos_id
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--tokenizer", type=str, default=TOKENIZER_NAME)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="Re-tokenize only the first N docs (0 = all). Useful as a fast spot check.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=0,
        help="Print the first N tokens of the bin decoded back to text.",
    )
    parser.add_argument(
        "--first-seqs",
        type=int,
        default=10,
        help="At the end of the run, print the first N BOS..EOS sequences read "
        "directly from the bin (numbered). Set 0 to disable.",
    )
    args = parser.parse_args()

    if not args.bin.is_file():
        raise SystemExit(f"Bin file does not exist: {args.bin}")
    if not args.parquet.is_file():
        raise SystemExit(f"Parquet file does not exist: {args.parquet}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    if bos_id is None or eos_id is None:
        raise SystemExit(f"Tokenizer missing BOS/EOS (bos={bos_id}, eos={eos_id})")
    print(f"  vocab_size={tokenizer.vocab_size}, bos_id={bos_id}, eos_id={eos_id}")

    bin_bytes = args.bin.stat().st_size
    if bin_bytes % np.dtype(DTYPE).itemsize != 0:
        raise SystemExit(f"Bin size {bin_bytes} not a multiple of {np.dtype(DTYPE).itemsize} bytes")
    bin_tokens = np.memmap(args.bin, dtype=DTYPE, mode="r")
    print(f"Bin: {args.bin}\n" f"  size={bin_bytes:,} bytes  tokens={bin_tokens.size:,}  dtype={DTYPE.__name__}")

    meta = load_meta(args.bin)
    if meta is not None:
        shard_record = next((s for s in meta.get("shards", []) if s["file"] == args.bin.name), None)
        print(
            f"meta.json: total_tokens={meta.get('total_tokens'):,} "
            f"num_documents={meta.get('num_documents'):,} "
            f"shards={len(meta.get('shards', []))}"
        )
        if shard_record is not None and shard_record["tokens"] != bin_tokens.size:
            print(
                f"  WARNING: meta says {shard_record['tokens']:,} tokens for "
                f"{args.bin.name}, but bin has {bin_tokens.size:,}"
            )

    if args.preview > 0:
        n = min(args.preview, bin_tokens.size)
        head = bin_tokens[:n].astype(np.int64).tolist()
        print(f"\nFirst {n} tokens (ids): {head}")
        print(f"First {n} tokens (decoded): {tokenizer.decode(head)!r}")

    print(f"\nRe-tokenizing parquet: {args.parquet}")
    offset = 0
    docs_checked = 0
    mismatch = False
    for doc_idx, text in enumerate(parquet_texts(args.parquet, args.text_field)):
        if args.max_docs and doc_idx >= args.max_docs:
            break
        expected = encode_doc(tokenizer, text, bos_id, eos_id)
        end = offset + expected.size
        if end > bin_tokens.size:
            print(
                f"  doc {doc_idx}: bin shorter than expected "
                f"(need {end:,} tokens, have {bin_tokens.size:,}) — "
                f"likely rolled into next shard."
            )
            mismatch = True
            break
        actual = np.asarray(bin_tokens[offset:end])
        if not np.array_equal(actual, expected):
            diff_idx = int(np.argmax(actual != expected))
            print(
                f"  doc {doc_idx}: MISMATCH at bin offset {offset + diff_idx} "
                f"(expected {int(expected[diff_idx])}, got {int(actual[diff_idx])})"
            )
            print(f"    expected[:16]={expected[:16].tolist()}")
            print(f"    actual[:16]  ={actual[:16].tolist()}")
            mismatch = True
            break
        offset = end
        docs_checked += 1
        if docs_checked % 1000 == 0:
            print(f"  ok: {docs_checked:,} docs, {offset:,} tokens consumed")

    print(f"\nChecked {docs_checked:,} docs, consumed {offset:,} / {bin_tokens.size:,} " f"tokens from bin.")

    if mismatch:
        raise SystemExit(1)

    full_pass = args.max_docs == 0
    if full_pass:
        if offset != bin_tokens.size:
            print(
                f"  WARNING: bin has {bin_tokens.size - offset:,} extra trailing tokens "
                f"after consuming all parquet docs. (Could indicate residual content "
                f"or a different shard layout.)"
            )
        if meta is not None and meta.get("num_documents") not in (None, docs_checked):
            print(
                f"  NOTE: meta.json reports {meta['num_documents']:,} docs across all "
                f"shards; this run checked {docs_checked:,} from {args.parquet.name}."
            )
    print("\nVERIFIED: bin contents match the re-tokenized parquet docs.")

    num_seqs = args.first_seqs
    if num_seqs > 0:
        print(f"\n=== First {num_seqs} sequences from {args.bin.name} ===")
        bin_arr = np.asarray(bin_tokens)
        eos_positions = np.flatnonzero(bin_arr == eos_id)
        if eos_positions.size == 0:
            print("  no EOS found in bin (cannot delimit any document).")
        else:
            cursor = 0
            for seq_idx in range(min(num_seqs, eos_positions.size)):
                eos_at = int(eos_positions[seq_idx])
                seq = bin_arr[cursor : eos_at + 1].astype(np.int64).tolist()
                print(f"\n[{seq_idx + 1}/{num_seqs}] offset={cursor:,} " f"eos_at={eos_at:,} length={len(seq):,}")
                print(f"  ids: {seq}")
                print(f"  decoded: {tokenizer.decode(seq)!r}")
                cursor = eos_at + 1


if __name__ == "__main__":
    main()
