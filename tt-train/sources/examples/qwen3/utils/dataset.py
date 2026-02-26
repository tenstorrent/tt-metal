# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
from tqdm import tqdm


class TextDataset:
    """Pre-tokenizes a text corpus into a flat uint32 token array and caches it.

    On the first run, all passages are tokenized and written to a .npy file
    under *cache_dir*.  Subsequent runs with the same dataset + tokenizer load
    the file instantly.  get_batch() samples random windows — no per-iteration
    CPU tokenization overhead.

    Cache key is derived from: tokenizer name/path, number of passages, and
    a content fingerprint (first/last passage prefix), so it invalidates
    automatically when the corpus changes.
    """

    def __init__(self, texts, tokenizer, seq_len, cache_dir=".token_cache", split=""):
        import hashlib

        self.seq_len = seq_len

        # ------------------------------------------------------------------
        # Build a fast cache key: tokenizer identity + corpus fingerprint
        # ------------------------------------------------------------------
        tok_id = getattr(tokenizer, "name_or_path", None) or str(
            getattr(tokenizer, "vocab_size", "")
        )
        sample_front = texts[0][:200] if texts else ""
        sample_back = texts[-1][:200] if texts else ""
        fingerprint = f"{tok_id}|{len(texts)}|{sample_front}|{sample_back}"
        key = hashlib.md5(fingerprint.encode()).hexdigest()
        fname = f"{split}_{key}.npy" if split else f"{key}.npy"
        cache_path = os.path.join(cache_dir, fname)

        # ------------------------------------------------------------------
        # Load from cache or tokenize and save
        # ------------------------------------------------------------------
        if os.path.exists(cache_path):
            print(f"  Loading token cache ({split or 'data'}): {cache_path}")
            self._tokens = np.load(cache_path)
        else:
            from joblib import Parallel, delayed
            import multiprocessing

            n_jobs = multiprocessing.cpu_count()
            print(
                f"  Tokenizing {len(texts):,} passages ({split or 'data'}) "
                f"with {n_jobs} workers..."
            )

            def _encode_batch(batch):
                return [tokenizer.encode(t) for t in batch]

            # Split into per-worker batches for low dispatch overhead
            batch_size = max(1, len(texts) // (n_jobs * 8))
            batches = [
                texts[i : i + batch_size] for i in range(0, len(texts), batch_size)
            ]

            results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_encode_batch)(b)
                for b in tqdm(
                    batches,
                    desc=f"  tokenize/{split or 'data'}",
                    unit="batch",
                    leave=False,
                )
            )

            chunks = [
                np.array(ids, dtype=np.uint32)
                for batch_result in results
                for ids in batch_result
                if ids
            ]
            self._tokens = (
                np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint32)
            )
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_path, self._tokens)
            print(f"  Saved {len(self._tokens):,} tokens → {cache_path}")

        print(f"  {split or 'dataset'}: {len(self._tokens):,} tokens total")

        if len(self._tokens) < seq_len + 1:
            raise ValueError(
                f"Dataset too small after tokenization: "
                f"{len(self._tokens)} tokens < seq_len+1={seq_len + 1}"
            )

    def get_batch(self, batch_size):
        """Sample a random batch of (input, target) pairs.

        Returns (x, y) each shaped [batch_size, seq_len], np.uint32.
        y is x shifted right by one token (next-token prediction).
        """
        max_start = len(self._tokens) - self.seq_len - 1
        starts = np.random.randint(0, max_start, size=batch_size)
        xs = np.stack([self._tokens[i : i + self.seq_len] for i in starts])
        ys = np.stack([self._tokens[i + 1 : i + self.seq_len + 1] for i in starts])
        return xs, ys

    def __len__(self):
        return len(self._tokens)


class SourceCodeDataset:
    """On-the-fly tokenized dataset from source code files.

    Scans a directory tree for C/C++ source files, holds their contents in
    memory, and tokenizes random file concatenations on each get_batch() call.
    No pre-cached token arrays — every batch is freshly assembled.
    """

    SOURCE_EXTENSIONS = frozenset(
        {".cpp", ".hpp", ".c", ".h", ".cc", ".cxx", ".hxx", ".inl"}
    )

    def __init__(self, file_paths, tokenizer, seq_len):
        self.seq_len = seq_len
        self.tokenizer = tokenizer

        self.file_contents = []
        for path in file_paths:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
                if content.strip():
                    self.file_contents.append(content)
            except (OSError, IOError):
                continue

        if not self.file_contents:
            raise ValueError("No readable source files found")

        self._total_chars = sum(len(c) for c in self.file_contents)
        print(
            f"  SourceCodeDataset: {len(self.file_contents)} files, "
            f"~{self._total_chars:,} chars in memory"
        )

    def get_batch(self, batch_size):
        """Assemble a batch by randomly concatenating source files and tokenizing.

        Returns (x, y) each shaped [batch_size, seq_len], np.uint32.
        """
        needed_tokens = batch_size * (self.seq_len + 1) + self.seq_len
        target_chars = needed_tokens * 5  # ~4-5 chars per token for code

        indices = np.random.permutation(len(self.file_contents))
        text_pieces = []
        total_chars = 0
        for idx in indices:
            text_pieces.append(self.file_contents[idx])
            total_chars += len(self.file_contents[idx])
            if total_chars >= target_chars:
                break

        while total_chars < target_chars:
            idx = np.random.randint(len(self.file_contents))
            text_pieces.append(self.file_contents[idx])
            total_chars += len(self.file_contents[idx])

        text = "\n\n".join(text_pieces)
        tokens = np.array(self.tokenizer.encode(text), dtype=np.uint32)

        while len(tokens) < needed_tokens:
            idx = np.random.randint(len(self.file_contents))
            extra = self.tokenizer.encode(self.file_contents[idx])
            tokens = np.concatenate([tokens, np.array(extra, dtype=np.uint32)])

        max_start = len(tokens) - self.seq_len - 1
        starts = np.random.randint(0, max(1, max_start), size=batch_size)
        xs = np.stack([tokens[i : i + self.seq_len] for i in starts])
        ys = np.stack([tokens[i + 1 : i + self.seq_len + 1] for i in starts])
        return xs, ys

    def __len__(self):
        return self._total_chars


def collect_source_files(root_dir, val_fraction=0.1, seed=42):
    """Walk *root_dir* for C/C++ source files and split into train / val lists.

    Returns (train_paths, val_paths).
    """
    exts = SourceCodeDataset.SOURCE_EXTENSIONS
    all_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in exts:
                all_files.append(os.path.join(dirpath, fname))
    all_files.sort()

    if not all_files:
        raise FileNotFoundError(
            f"No source files ({', '.join(sorted(exts))}) found under {root_dir}"
        )

    rng = np.random.RandomState(seed)
    rng.shuffle(all_files)
    split = max(1, int(len(all_files) * val_fraction))
    val_files = all_files[:split]
    train_files = all_files[split:]

    print(
        f"Source code corpus: {len(all_files)} files in {root_dir}\n"
        f"  train: {len(train_files)} files, val: {len(val_files)} files"
    )
    return train_files, val_files


def load_text_datasets(dataset_path):
    """Load raw text passages (no tokenization).

    Supports:
      - "wikitext" / "wikitext-2" → Salesforce/wikitext (wikitext-2-raw-v1)
      - A local .txt file path
      - Any HuggingFace dataset name with a 'text' column

    Returns (train_texts, val_texts) — lists of strings.
    """
    if dataset_path in ("wikitext", "wikitext-2"):
        from datasets import load_dataset

        print("Loading wikitext-2-raw-v1 from HuggingFace...")
        ds = load_dataset(
            "Salesforce/wikitext",
            "wikitext-2-raw-v1",
            ignore_verifications=True,
        )
        train_texts = [t for t in ds["train"]["text"] if t.strip()]
        val_texts = [t for t in ds["validation"]["text"] if t.strip()]
    elif os.path.isfile(dataset_path):
        print(f"Loading text from file: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        split = int(len(lines) * 0.9)
        train_texts = lines[:split]
        val_texts = lines[split:]
    else:
        from datasets import load_dataset

        print(f"Loading dataset '{dataset_path}' from HuggingFace...")
        ds = load_dataset(dataset_path, ignore_verifications=True)
        if "train" in ds:
            train_texts = [str(t) for t in ds["train"]["text"] if str(t).strip()]
        else:
            first_split = list(ds.keys())[0]
            train_texts = [str(t) for t in ds[first_split]["text"] if str(t).strip()]
        if "validation" in ds:
            val_texts = [str(t) for t in ds["validation"]["text"] if str(t).strip()]
        elif "test" in ds:
            val_texts = [str(t) for t in ds["test"]["text"] if str(t).strip()]
        else:
            split = int(len(train_texts) * 0.9)
            val_texts = train_texts[split:]
            train_texts = train_texts[:split]

    print(
        f"Loaded {len(train_texts):,} train passages, "
        f"{len(val_texts):,} val passages"
    )
    return train_texts, val_texts
