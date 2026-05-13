"""Parallel parquet → uint16 .bin tokenizer for SlimPajama / TinyLlama.

Same output layout as ``dataset_preprocessing.py``, but each worker process
owns its own ``ShardWriter`` and writes shards directly to disk. The parent
process only partitions parquet files across workers and merges per-worker
shard lists into a single ``meta.json``. There is no IPC of token arrays —
this removes the single-threaded parent bottleneck (pickle/concatenate/write)
that limits the original script to ~80 ktok/s.

Output:
    output-dir/
        train/      train_w00_s000000.bin, train_w00_s000001.bin, ...,
                    train_w01_s000000.bin, ..., meta.json
        validation/ validation_w00_s000000.bin, ..., meta.json
        test/       test_w00_s000000.bin, ..., meta.json

Shards are NOT globally ordered (training-time shuffle handles that). The
``meta.json`` ``shards`` list is sorted by ``(worker_id, shard_idx)`` so it
is reproducible.

Usage:
    python dataset_preprocessing_parallel.py
    python dataset_preprocessing_parallel.py --splits train --num-proc 48
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

DEFAULT_INPUT_DIR = "/data/awliu/datasets/SlimPajama-627B_Reupload/data"
DEFAULT_OUTPUT_DIR = "/data/awliu/datasets/SlimPajama-627B-tokenized"
TOKENIZER_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
SPLITS = ("train", "validation", "test")
DTYPE = np.uint16
SPLIT_PREFIX_RE = re.compile(r"^(train|validation|test)-")


@dataclass
class ShardInfo:
    file: str
    tokens: int


@dataclass
class WorkerResult:
    worker_id: int
    shards: list[ShardInfo]
    total_tokens: int
    num_documents: int
    elapsed_s: float


@dataclass
class WorkerJob:
    worker_id: int
    split: str
    files: list[str]
    out_dir: str
    tokenizer_name: str
    bos_id: int
    eos_id: int
    text_field: str
    batch_size: int
    shard_tokens: int


class ShardWriter:
    """Buffers uint16 tokens and rolls over to a new .bin shard at a fixed size."""

    def __init__(self, out_dir: Path, split: str, worker_id: int, shard_tokens: int) -> None:
        self.out_dir = out_dir
        self.split = split
        self.worker_id = worker_id
        self.shard_tokens = int(shard_tokens)
        self._buf = np.empty(self.shard_tokens, dtype=DTYPE)
        self._fill = 0
        self._shard_idx = 0
        self.shards: list[ShardInfo] = []
        self.total_tokens = 0

    def _shard_path(self, idx: int) -> Path:
        return self.out_dir / f"{self.split}_w{self.worker_id:02d}_s{idx:06d}.bin"

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
                path = self._shard_path(self._shard_idx)
                self._buf.tofile(path)
                self.shards.append(ShardInfo(file=path.name, tokens=self._fill))
                self._shard_idx += 1
                self._fill = 0

    def close(self) -> None:
        if self._fill > 0:
            path = self._shard_path(self._shard_idx)
            self._buf[: self._fill].tofile(path)
            self.shards.append(ShardInfo(file=path.name, tokens=self._fill))
            self._fill = 0
            self._shard_idx += 1


def _iter_text_batches(files: list[str], text_field: str, batch_size: int) -> Iterator[list[str]]:
    pending: list[str] = []
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for record_batch in pf.iter_batches(batch_size=batch_size, columns=[text_field]):
            for text in record_batch.column(text_field).to_pylist():
                if text:
                    pending.append(text)
                    if len(pending) >= batch_size:
                        yield pending
                        pending = []
    if pending:
        yield pending


def _worker_run(job: WorkerJob) -> WorkerResult:
    """Tokenize this worker's parquet files and write shards directly to disk."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(job.tokenizer_name, use_fast=True)
    bos = np.array([job.bos_id], dtype=DTYPE)
    eos = np.array([job.eos_id], dtype=DTYPE)

    out_dir = Path(job.out_dir)
    writer = ShardWriter(out_dir, job.split, job.worker_id, job.shard_tokens)
    num_documents = 0
    start = time.time()
    last_log = start

    for batch in _iter_text_batches(job.files, job.text_field, job.batch_size):
        encodings = tokenizer(batch, add_special_tokens=False)
        pieces: list[np.ndarray] = []
        for ids in encodings["input_ids"]:
            pieces.append(bos)
            pieces.append(np.asarray(ids, dtype=DTYPE))
            pieces.append(eos)
        writer.write(np.concatenate(pieces))
        num_documents += len(batch)

        now = time.time()
        if now - last_log >= 30.0:
            last_log = now
            elapsed = max(now - start, 1e-6)
            rate = writer.total_tokens / elapsed
            print(
                f"  [w{job.worker_id:02d} {job.split}] {num_documents:,} docs, "
                f"{writer.total_tokens:,} tokens, {len(writer.shards)} shard(s), "
                f"{rate / 1e6:.2f} Mtok/s",
                flush=True,
            )

    writer.close()
    elapsed = time.time() - start
    print(
        f"  [w{job.worker_id:02d} {job.split}] DONE: {num_documents:,} docs, "
        f"{writer.total_tokens:,} tokens, {len(writer.shards)} shard(s), {elapsed:.1f}s",
        flush=True,
    )
    return WorkerResult(
        worker_id=job.worker_id,
        shards=writer.shards,
        total_tokens=writer.total_tokens,
        num_documents=num_documents,
        elapsed_s=elapsed,
    )


def discover_splits(input_dir: Path) -> dict[str, list[Path]]:
    buckets: dict[str, list[Path]] = {s: [] for s in SPLITS}
    for path in sorted(input_dir.glob("*.parquet")):
        m = SPLIT_PREFIX_RE.match(path.name)
        if not m:
            continue
        buckets[m.group(1)].append(path)
    return buckets


def _partition_files(files: list[Path], num_workers: int) -> list[list[str]]:
    """Round-robin partition by sorted filename so size variance averages out."""
    parts: list[list[str]] = [[] for _ in range(num_workers)]
    for i, f in enumerate(files):
        parts[i % num_workers].append(str(f))
    return parts


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
    print(f"\n=== Processing split: {split} ({len(files)} parquet file(s), {num_proc} workers) ===", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_workers = min(num_proc, len(files))
    partitions = _partition_files(files, n_workers)
    jobs = [
        WorkerJob(
            worker_id=wid,
            split=split,
            files=part,
            out_dir=str(out_dir),
            tokenizer_name=tokenizer_name,
            bos_id=bos_id,
            eos_id=eos_id,
            text_field=text_field,
            batch_size=batch_size,
            shard_tokens=shard_tokens,
        )
        for wid, part in enumerate(partitions)
    ]
    for j in jobs:
        print(f"  worker {j.worker_id:02d}: {len(j.files)} parquet file(s)", flush=True)

    start = time.time()
    if n_workers == 1:
        results = [_worker_run(jobs[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = list(pool.imap_unordered(_worker_run, jobs))
    elapsed = time.time() - start

    results.sort(key=lambda r: r.worker_id)
    all_shards: list[ShardInfo] = []
    total_tokens = 0
    num_documents = 0
    for r in results:
        all_shards.extend(r.shards)
        total_tokens += r.total_tokens
        num_documents += r.num_documents

    rate = total_tokens / max(elapsed, 1e-6)
    print(
        f"  [{split}] ALL DONE: {num_documents:,} docs, {total_tokens:,} tokens, "
        f"{len(all_shards)} shard(s), {elapsed:.1f}s wallclock, {rate / 1e6:.2f} Mtok/s aggregate",
        flush=True,
    )

    meta = {
        "split": split,
        "dtype": np.dtype(DTYPE).name,
        "tokenizer": tokenizer_name,
        "vocab_size": vocab_size,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "shard_tokens": shard_tokens,
        "total_tokens": total_tokens,
        "num_documents": num_documents,
        "num_workers": n_workers,
        "shards": [{"file": s.file, "tokens": s.tokens} for s in all_shards],
    }
    with (out_dir / "meta.json").open("w") as f:
        json.dump(meta, f, indent=2)
    return meta


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
        help="Tokens per .bin shard (default: 2049*16384 = ~33.6M).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Documents per tokenizer / parquet read batch.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Worker processes (capped at the number of parquet files in a split).",
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

    print(f"Loading tokenizer: {args.tokenizer}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size
    if bos_id is None or eos_id is None:
        raise SystemExit(f"Tokenizer {args.tokenizer!r} is missing BOS/EOS ids (bos={bos_id}, eos={eos_id}).")
    if vocab_size > np.iinfo(DTYPE).max + 1:
        raise SystemExit(f"Vocab size {vocab_size} does not fit in {np.dtype(DTYPE).name}.")
    print(f"  vocab_size={vocab_size}, bos_id={bos_id}, eos_id={eos_id}", flush=True)

    buckets = discover_splits(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    totals: dict[str, int] = {}
    for split in args.splits:
        files = buckets.get(split, [])
        if not files:
            print(f"\n=== Skipping split {split!r}: no parquet files found ===", flush=True)
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

    print("\n=== Token counts ===", flush=True)
    for split in args.splits:
        print(f"  {split:<10s}: {totals.get(split, 0):,} tokens", flush=True)
    print(f"  {'total':<10s}: {sum(totals.values()):,} tokens", flush=True)


if __name__ == "__main__":
    sys.exit(main())
