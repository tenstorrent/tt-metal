# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_shakespeare_text():
    ds = load_dataset(
        "text",
        data_files={
            "train": f'{os.environ["TT_METAL_HOME"]}/tt-train/data/shakespeare.txt'
        },
    )
    text = "\n".join(ds["train"]["text"])
    return text


PAD_TOKEN = "<PAD>"
BEGIN_TOKEN = "<BEG>"
END_TOKEN = "<END>"


class CharTokenizer:
    def __init__(
        self,
        text: str,
        add_padding_token: bool = True,
        add_begin_end_tokens: bool = True,
    ):
        # Sorted stable alphabet for reproducibility (matching C++ std::set order)
        chars = sorted(list(set(text)))

        self.stoi = {}
        self.itos = []

        if add_padding_token:
            self.stoi[PAD_TOKEN] = 0
            self.itos.append(PAD_TOKEN)

        for ch in chars:
            if ch in self.stoi:
                continue
            self.stoi[ch] = len(self.itos)
            self.itos.append(ch)

        if add_begin_end_tokens:
            self.stoi[BEGIN_TOKEN] = len(self.itos)
            self.itos.append(BEGIN_TOKEN)
            self.stoi[END_TOKEN] = len(self.itos)
            self.itos.append(END_TOKEN)

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# --- Build tokenizer + dataset splits ---
def prepare_data(yaml_config):
    text = load_shakespeare_text()

    training_config = yaml_config.get("training_config", {})
    use_bpe = training_config.get("tokenizer_type", "char") == "bpe"
    tokenizer_path = training_config.get("tokenizer_path", "")

    if use_bpe:
        assert tokenizer_path, "tokenizer_path is required when use_bpe is true"
        tokenizer_path = os.path.join(
            os.environ["TT_METAL_HOME"], "tt-train", tokenizer_path
        )
        if os.path.isdir(tokenizer_path):
            bpe = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        elif os.path.isfile(tokenizer_path):
            # Load directly from a single tokenizer.json file
            bpe = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        else:
            raise FileNotFoundError(f"Tokenizer path not found: {tokenizer_path}")
        vocab_size = bpe.vocab_size
        encode = lambda t: np.array(bpe.encode(t), dtype=np.uint32)
        decode_fn = lambda ids: bpe.decode(list(ids))
    else:
        ctok = CharTokenizer(text)
        vocab_size = (ctok.vocab_size + 31) // 32 * 32  # pad to multiple of 32
        encode = lambda t: np.array(ctok.encode(t), dtype=np.uint32)
        decode_fn = lambda ids: ctok.decode(list(ids))

    # Encode full corpus, split 90/10
    ids = encode(text)
    n = len(ids)
    n_train = int(n * 0.9)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    return train_ids, val_ids, vocab_size, decode_fn


def get_batch(split_ids: np.ndarray, seq_len: int, batch_size: int):
    n = len(split_ids) - seq_len - 1
    ix = np.random.randint(0, n, size=(batch_size,))
    x = np.stack([split_ids[i : i + seq_len] for i in ix], axis=0)  # [B, T]
    y = np.stack(
        [split_ids[i + 1 : i + seq_len + 1] for i in ix], axis=0
    )  # [B, T] next-token targets
    return x.astype(np.uint32), y.astype(np.uint32)


def build_causal_mask(T: int):
    m = np.tril(np.ones((T, T), dtype=np.float32))
    return m.reshape(1, 1, T, T)
