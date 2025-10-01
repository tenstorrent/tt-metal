import os
import numpy as np
from datasets import load_dataset


def load_shakespeare_text():
    ds = load_dataset(
        "text",
        data_files={"train": f'{os.environ["TT_METAL_HOME"]}/tt-train/data/shakespeare.txt'},
    )
    text = "\n".join(ds["train"]["text"])
    return text


class CharTokenizer:
    def __init__(self, text: str):
        # Sorted stable alphabet for reproducibility
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = chars

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


# --- Build tokenizer + dataset splits ---
def prepare_data():
    text = load_shakespeare_text()

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
    y = np.stack([split_ids[i + 1 : i + seq_len + 1] for i in ix], axis=0)  # [B, T] next-token targets
    return x.astype(np.uint32), y.astype(np.uint32)


def build_causal_mask(T: int):
    m = np.tril(np.ones((T, T), dtype=np.float32))
    return m.reshape(1, 1, T, T)
