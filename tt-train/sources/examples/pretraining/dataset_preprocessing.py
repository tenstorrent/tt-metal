"""Tokenize local SlimPajama-627B parquet shards with the TinyLlama tokenizer.

Reads parquet files from ``--input-dir`` (default
``/data/awliu/datasets/SlimPajama-6B/data``), routes them to splits by filename
prefix (``train-`` / ``validation-`` / ``test-``), tokenizes with the TinyLlama
fast tokenizer, wraps every document with BOS/EOS and writes the concatenated
token stream to memory-mappable ``uint16`` ``.bin`` shards under
``--output-dir`` (default ``/data/awliu/datasets/SlimPajama-6B-tokenized``):

    output-dir/
        train/      train_000000.bin, train_000001.bin, ..., meta.json
        validation/ validation_000000.bin, meta.json
        test/       test_000000.bin, meta.json

This is the standard nanoGPT / TinyLlama-lit-gpt pretraining format. At training
time each shard can be opened with ``np.memmap(path, dtype=np.uint16, mode='r')``.

Usage:
    python dataset_preprocessing.py
    python dataset_preprocessing.py --splits validation
    python dataset_preprocessing.py --num-proc 16 --shard-tokens 2049*16384
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

DEFAULT_INPUT_DIR = "/data/awliu/datasets/SlimPajama-627B_Reupload/data"
DEFAULT_OUTPUT_DIR = "/data/awliu/datasets/SlimPajama-627B-tokenized"
TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SPLITS = ("train", "validation", "test")
DTYPE = np.uint16
SPLIT_PREFIX_RE = re.compile(r"^(train|validation|test)-")


_WORKER_TOKENIZER: AutoTokenizer | None = None
_WORKER_BOS: int = 0
_WORKER_EOS: int = 0


def _worker_init(tokenizer_name: str, bos_id: int, eos_id: int) -> None:
    """Initializer run once per pool worker; loads the fast tokenizer."""
    global _WORKER_TOKENIZER, _WORKER_BOS, _WORKER_EOS
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    _WORKER_BOS = bos_id
    _WORKER_EOS = eos_id


def _worker_tokenize(texts: list[str]) -> tuple[np.ndarray, int]:
    """Encode a batch of documents and return a single concatenated uint16 array.

    Returns the concatenated array and the number of documents in the batch.
    Each document is emitted as ``[BOS, *ids, EOS]`` (no internal special tokens).
    """
    assert _WORKER_TOKENIZER is not None
    if not texts:
        return np.empty(0, dtype=DTYPE), 0
    encodings = _WORKER_TOKENIZER(texts, add_special_tokens=False)
    pieces: list[np.ndarray] = []
    bos = np.array([_WORKER_BOS], dtype=DTYPE)
    eos = np.array([_WORKER_EOS], dtype=DTYPE)
    for ids in encodings["input_ids"]:
        pieces.append(bos)
        pieces.append(np.asarray(ids, dtype=DTYPE))
        pieces.append(eos)
    return np.concatenate(pieces), len(texts)


@dataclass
class ShardInfo:
    file: str
    tokens: int


class ShardWriter:
    """Buffers uint16 tokens and rolls over to a new .bin shard at a fixed size."""

    def __init__(self, out_dir: Path, split: str, shard_tokens: int) -> None:
        self.out_dir = out_dir
        self.split = split
        self.shard_tokens = int(shard_tokens)
        self._buf = np.empty(self.shard_tokens, dtype=DTYPE)
        self._fill = 0
        self._shard_idx = 0
        self.shards: list[ShardInfo] = []
        self.total_tokens = 0

    def _shard_path(self, idx: int) -> Path:
        return self.out_dir / f"{self.split}_{idx:06d}.bin"

    def _flush_full(self) -> None:
        path = self._shard_path(self._shard_idx)
        self._buf.tofile(path)
        self.shards.append(ShardInfo(file=path.name, tokens=self._fill))
        self._shard_idx += 1
        self._fill = 0

    def write(self, tokens: np.ndarray) -> None:
        if tokens.size == 0:
            return
        if tokens.dtype != DTYPE:
            tokens = tokens.astype(DTYPE, copy=False)
        self.total_tokens += int(tokens.size)
        offset = 0
        n = tokens.size
        while offset < n:
            free = self.shard_tokens - self._fill
            take = min(free, n - offset)
            self._buf[self._fill : self._fill + take] = tokens[offset : offset + take]
            self._fill += take
            offset += take
            if self._fill == self.shard_tokens:
                self._flush_full()

    def close(self) -> None:
        if self._fill > 0:
            path = self._shard_path(self._shard_idx)
            self._buf[: self._fill].tofile(path)
            self.shards.append(ShardInfo(file=path.name, tokens=self._fill))
            self._fill = 0
            self._shard_idx += 1


def discover_splits(input_dir: Path) -> dict[str, list[Path]]:
    """Bucket parquet files in ``input_dir`` by split based on filename prefix."""
    buckets: dict[str, list[Path]] = {s: [] for s in SPLITS}
    for path in sorted(input_dir.glob("*.parquet")):
        m = SPLIT_PREFIX_RE.match(path.name)
        if not m:
            continue
        buckets[m.group(1)].append(path)
    return buckets


def iter_text_batches(files: list[Path], text_field: str, batch_size: int) -> Iterator[list[str]]:
    """Yield lists of document strings of length ~batch_size from parquet files."""
    pending: list[str] = []
    for fpath in files:
        pf = pq.ParquetFile(str(fpath))
        for record_batch in pf.iter_batches(batch_size=batch_size, columns=[text_field]):
            col = record_batch.column(text_field).to_pylist()
            for text in col:
                if text:
                    pending.append(text)
                    if len(pending) >= batch_size:
                        yield pending
                        pending = []
    if pending:
        yield pending


def process_split(
    split: str,
    files: list[Path],
    out_dir: Path,
    tokenizer_name: str,
    bos_id: int,
    eos_id: int,
    vocab_size: int,
    text_field: str,
    batch_size: int,
    shard_tokens: int,
    num_proc: int,
) -> dict:
    print(f"\n=== Processing split: {split} ({len(files)} parquet file(s)) ===")
    out_dir.mkdir(parents=True, exist_ok=True)

    writer = ShardWriter(out_dir, split, shard_tokens)
    num_documents = 0
    start = time.time()

    batches = iter_text_batches(files, text_field=text_field, batch_size=batch_size)

    if num_proc <= 1:
        _worker_init(tokenizer_name, bos_id, eos_id)
        for batch in batches:
            tokens, n_docs = _worker_tokenize(batch)
            writer.write(tokens)
            num_documents += n_docs
            _maybe_log(split, num_documents, writer, start)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=num_proc,
            initializer=_worker_init,
            initargs=(tokenizer_name, bos_id, eos_id),
        ) as pool:
            for tokens, n_docs in pool.imap(_worker_tokenize, batches, chunksize=1):
                writer.write(tokens)
                num_documents += n_docs
                _maybe_log(split, num_documents, writer, start)

    writer.close()
    elapsed = time.time() - start
    print(
        f"  [{split}] done: {num_documents:,} docs, {writer.total_tokens:,} tokens, "
        f"{len(writer.shards)} shard(s), {elapsed:.1f}s"
    )

    meta = {
        "split": split,
        "dtype": np.dtype(DTYPE).name,
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "shard_tokens": shard_tokens,
        "total_tokens": writer.total_tokens,
        "num_documents": num_documents,
        "shards": [{"file": s.file, "tokens": s.tokens} for s in writer.shards],
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    return meta


_LAST_LOG_T = [0.0]


def _maybe_log(split: str, num_documents: int, writer: ShardWriter, start: float) -> None:
    now = time.time()
    if now - _LAST_LOG_T[0] < 5.0:
        return
    _LAST_LOG_T[0] = now
    elapsed = max(now - start, 1e-6)
    rate = writer.total_tokens / elapsed
    print(
        f"  [{split}] {num_documents:,} docs, {writer.total_tokens:,} tokens, "
        f"{len(writer.shards)} shard(s) closed, {rate / 1e6:.2f} Mtok/s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
        help="Which splits to process (default: all).",
    )
    parser.add_argument(
        "--shard-tokens",
        type=int,
        default=2049 * 16384,
        help="Tokens per .bin shard (default: 2049*16384 = ~33.6M, matching litgpt/TinyLlama PackedDataset convention).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Documents per tokenizer batch / parquet read batch.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Worker processes for tokenization (1 disables multiprocessing).",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default="text",
        help="Parquet column containing the document text.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=TOKENIZER_NAME,
        help="HF tokenizer to load.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input_dir}")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size
    if bos_id is None or eos_id is None:
        raise SystemExit(f"Tokenizer {args.tokenizer!r} is missing BOS/EOS ids " f"(bos={bos_id}, eos={eos_id}).")
    if vocab_size > np.iinfo(DTYPE).max + 1:
        raise SystemExit(f"Vocab size {vocab_size} does not fit in {np.dtype(DTYPE).name}.")
    print(f"  vocab_size={vocab_size}, bos_id={bos_id}, eos_id={eos_id}")

    buckets = discover_splits(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    totals: dict[str, int] = {}
    for split in args.splits:
        files = buckets.get(split, [])
        if not files:
            print(f"\n=== Skipping split {split!r}: no parquet files found ===")
            totals[split] = 0
            continue
        meta = process_split(
            split=split,
            files=files,
            out_dir=args.output_dir / split,
            tokenizer_name=args.tokenizer,
            bos_id=bos_id,
            eos_id=eos_id,
            vocab_size=vocab_size,
            text_field=args.text_field,
            batch_size=args.batch_size,
            shard_tokens=args.shard_tokens,
            num_proc=args.num_proc,
        )
        totals[split] = meta["total_tokens"]

    print("\n=== Token counts ===")
    for split in args.splits:
        print(f"  {split:<10s}: {totals.get(split, 0):,} tokens")
    print(f"  {'total':<10s}: {sum(totals.values()):,} tokens")


if __name__ == "__main__":
    main()
