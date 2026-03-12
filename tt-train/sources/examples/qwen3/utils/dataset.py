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
