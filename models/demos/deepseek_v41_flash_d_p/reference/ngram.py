# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Engram n-gram hashing on the host (port of the checkpoint's ``inference/engram.py``), chunk by chunk.

Token ids -> compressed ids (tokens that normalize alike, e.g. " The" / "the", share one), then each position is
hashed with the max_ngram_size - 1 tokens before it: for n = 2 .. max_ngram_size, the XOR of
compressed_id(t - j) * multiplier[layer][j] over j < n, modulo a prime per (n, head), plus that (n, head)'s offset into
the layer's table. Look-back stops at the start of the sequence (and at a DEAD token, never produced for text):
positions past it read the pad token's compressed id. The result per layer is [S, (max_ngram_size - 1) * n_heads]
table rows; the table lookup itself is ``weights.EngramTable.rows``.

Chunked: the hash of a position needs the compressed ids of the max_ngram_size - 1 positions before it, so the
state carries them (``tail``: the last max_ngram_size - 1 compressed ids, DEAD (-1) where the position does not
exist). Chunked hashing equals one-shot hashing exactly (integer arithmetic).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

DEAD = -1


def find_next_prime(start: int, seen: set[int]) -> int:
    from sympy import isprime

    c = start + 1
    while not isprime(c) or c in seen:
        c += 1
    return c


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Token id -> compressed id (normalize: NFKC, NFD, strip accents, lowercase, collapse whitespace, strip; a token
    that is exactly whitespace keeps one space; a partial UTF-8 byte token keys by its raw form)."""
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def hash_multipliers(layer_ids, max_ngram_size: int, vocab_size: int) -> torch.Tensor:
    """[n_layers, max_ngram_size] odd int64 multipliers from a per-layer numpy RNG (seed 10007 * layer)."""
    bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        g = np.random.default_rng(10007 * layer_id)
        rows.append(torch.tensor(g.integers(low=0, high=bound, size=(max_ngram_size,), dtype=np.int64) * 2 + 1))
    return torch.stack(rows)


def bucket_primes(n_layers: int, max_ngram_size: int, n_heads: int, vocab_size: int) -> torch.Tensor:
    """[n_layers, max_ngram_size - 1, n_heads] distinct primes, drawn in order above vocab_size - 1 per (n, head)."""
    primes, seen = [], set()
    for _ in range(n_layers):
        per_n = []
        for _ in range(max_ngram_size - 1):
            sizes, cur = [], vocab_size - 1
            for _ in range(n_heads):
                cur = find_next_prime(cur, seen)
                seen.add(cur)
                sizes.append(cur)
            per_n.append(sizes)
        primes.append(per_n)
    return torch.tensor(primes, dtype=torch.int64)


@dataclass
class EngramHasher:
    """Hash ids for the engram layers. ``layer_ids`` in config order (the table index of a layer is its position)."""

    layer_ids: tuple[int, ...]
    max_ngram_size: int
    n_heads: int
    token_map: torch.Tensor  # [vocab] int64
    pad_id: int  # compressed id of the pad token
    multipliers: torch.Tensor  # [n_layers, max_ngram_size]
    primes: torch.Tensor  # [n_layers, max_ngram_size - 1, n_heads]
    offsets: torch.Tensor  # [n_layers, (max_ngram_size - 1) * n_heads]

    @classmethod
    def build(cls, args: dict, tokenizer) -> "EngramHasher":
        layer_ids = tuple(args["engram_layer_ids"])
        m, h = args["engram_max_ngram_size"], args["engram_n_heads"]
        token_map, vocab = build_compressed_token_map(tokenizer)
        assert vocab == args["engram_compressed_vocab_size"], (vocab, args["engram_compressed_vocab_size"])
        primes = bucket_primes(len(layer_ids), m, h, args["engram_vocab_size"])
        flat = primes.view(len(layer_ids), -1)
        offsets = torch.cat([torch.zeros(len(layer_ids), 1, dtype=torch.int64), flat[:, :-1].cumsum(-1)], dim=-1)
        return cls(
            layer_ids=layer_ids,
            max_ngram_size=m,
            n_heads=h,
            token_map=torch.tensor(token_map, dtype=torch.int64),
            pad_id=token_map[args.get("engram_pad_id", 2)],
            multipliers=hash_multipliers(layer_ids, m, vocab),
            primes=primes,
            offsets=offsets,
        )

    @property
    def tail_len(self) -> int:
        return self.max_ngram_size - 1

    def new_tail(self) -> torch.Tensor:
        return torch.full((self.tail_len,), DEAD, dtype=torch.int64)

    def hashes(self, tokens: torch.Tensor, layer: int, tail: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """tokens [S] (positions start..start+S-1), tail [max_ngram_size - 1] (compressed ids before start, DEAD where
        none) -> (table rows [S, (max_ngram_size - 1) * n_heads] int64, the new tail)."""
        li = self.layer_ids.index(layer)
        comp = torch.cat([tail, self.token_map[tokens.long()]])  # [tail_len + S]
        s = tokens.shape[0]
        blocked = torch.zeros(s, dtype=torch.bool)
        toks = []
        for shift in range(self.max_ngram_size):
            src = comp[self.tail_len - shift : self.tail_len - shift + s]
            blocked = blocked | (src == DEAD)
            toks.append(torch.where(blocked, self.pad_id, src))
        prods = torch.stack(toks, dim=-1) * self.multipliers[li]  # [S, max_ngram_size]
        rolling, out = prods[:, 0], []
        for i in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, prods[:, i])
            out.append(rolling.unsqueeze(-1) % self.primes[li, i - 1])
        return torch.cat(out, dim=-1) + self.offsets[li], comp[-self.tail_len :].clone()
