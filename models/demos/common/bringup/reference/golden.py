# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Golden trace format, reader, and shared helpers for the reference-side scripts.

One directory per ladder rung, ``<golden_root>/s{seq}_c{chunk}/`` (the prefill server's golden-trace layout):
    manifest.json                         run description, layers, dumped chunks, content_hash
    metadata.json                         token_ids, num_layers, state tensor names, seq_len
    kv_cache/layer_{i}.safetensors        state per layer, key f"{name}_cache_layer_{i}" (key/value for a KV cache)
    chunk_{c:02d}/layer_{i:02d}.safetensors   boundary tensors of layer i in chunk c, keyed by boundary name
    chunk_{c:02d}/model.safetensors       embed, final_norm, top32 ids/values per position, logits of the last 32 rows

``content_hash`` is the sha256 over the sorted (path, file sha256) list of every file except manifest.json.
Tests pin the manifest's hash in their frozen block; a regenerated or edited golden then fails the gate.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from safetensors.torch import load_file

from models.demos.common.bringup.core.freeze import sha256_file
from models.demos.common.bringup.core.spec import CODE_ROOT, Spec

TOPK = 32
LOGITS_TAIL = 32


def rung_dirname(seq: int, chunk: int) -> str:
    return f"s{seq}_c{chunk}"


def rung_dir(spec: Spec, rung: dict) -> Path:
    """A rung with ``golden: <other rung>`` reads the other rung's golden (e.g. last chunk after a golden prefix)."""
    src = spec.rung(rung["golden"]) if rung.get("golden") else rung
    return spec.golden_root / rung_dirname(src["seq"], src["chunk"])


def content_hash(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in d.rglob("*") if p.is_file() and p.name != "manifest.json"):
        h.update(f"{f.relative_to(d)}\0{sha256_file(f)}\n".encode())
    return h.hexdigest()


class Golden:
    def __init__(self, d: Path):
        self.dir = Path(d)
        if not (self.dir / "manifest.json").exists():
            raise FileNotFoundError(f"golden not generated: {self.dir}")
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.meta = json.loads((self.dir / "metadata.json").read_text())
        self.seq, self.chunk = self.manifest["seq"], self.manifest["chunk"]
        self.n_chunks = self.seq // self.chunk
        self._layer = lru_cache(maxsize=8)(self._load_layer)

    @classmethod
    def for_rung(cls, spec: Spec, name: str) -> "Golden":
        return cls(rung_dir(spec, spec.rung(name)))

    @property
    def layers(self) -> list[int]:
        return list(self.manifest["layers"])

    @property
    def dumped_chunks(self) -> list[int]:
        return list(self.manifest.get("dumped_chunks", [self.n_chunks - 1]))

    @property
    def state_names(self) -> list[str]:
        return list(self.meta.get("state_tensors", ["key", "value"]))

    def tokens(self) -> torch.Tensor:
        return torch.tensor(self.meta["token_ids"], dtype=torch.int64)

    def _load_layer(self, c: int, i: int) -> dict:
        return load_file(str(self.dir / f"chunk_{c:02d}" / f"layer_{i:02d}.safetensors"))

    def layer(self, c: int, i: int) -> dict[str, torch.Tensor]:
        return self._layer(c, i)

    def has_layer(self, c: int, i: int) -> bool:
        return (self.dir / f"chunk_{c:02d}" / f"layer_{i:02d}.safetensors").exists()

    def model(self, c: int) -> dict[str, torch.Tensor]:
        return load_file(str(self.dir / f"chunk_{c:02d}" / "model.safetensors"))

    def state(self, i: int) -> dict[str, torch.Tensor]:
        d = load_file(str(self.dir / "kv_cache" / f"layer_{i}.safetensors"))
        return {n: d[f"{n}_cache_layer_{i}"] for n in self.state_names}

    def pinned_hash(self) -> str:
        """The hash recorded at generation (or, for goldens made before hashes, the manifest file's own hash)."""
        return self.manifest.get("content_hash") or sha256_file(self.dir / "manifest.json")

    def verify(self) -> bool:
        return "content_hash" in self.manifest and content_hash(self.dir) == self.manifest["content_hash"]


# ---------------------------------------------------------------- shared helpers for the scripts
def load_spec(path: str | None = None) -> Spec:
    path = path or os.environ.get("BRINGUP_SPEC")
    if not path:
        raise SystemExit("no spec: pass --spec or set BRINGUP_SPEC")
    return Spec.load(path)


def hf_path(spec: Spec) -> str:
    """Local checkpoint dir: paths.hf if set, else $ART/<model>/hf, else the HF hub cache (no download)."""
    if spec.get("paths.hf") or (spec.hf_dir / "config.json").exists():
        return str(spec.hf_dir)
    from huggingface_hub import snapshot_download

    return snapshot_download(spec.data["hf_id"], local_files_only=True)


def tokenizer(spec: Spec):
    hooks = spec.hooks()
    if hasattr(hooks, "tokenizer"):
        return hooks.tokenizer(spec)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_path(spec), trust_remote_code=bool(spec.get("hf.trust_remote_code")))


def text_tokens(spec: Spec, n: int, tok=None) -> torch.Tensor:
    """The first n tokens of the spec's long text (default: A Tale of Two Cities, shipped in the repo), BOS-prefixed.

    ``text.chat_template: true`` puts the text inside one user turn of the tokenizer's chat template (which then
    supplies BOS), for instruction-tuned checkpoints that degenerate on raw text. The prompt is truncated to n tokens
    inside the turn, as a long user message is in serving."""
    src = spec.get("text.source", "models/tt_transformers/tests/tale-of-two-cities.txt.bz2")
    path = Path(src) if Path(src).is_absolute() else CODE_ROOT / src
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rt", encoding="utf-8") as f:
        text = f.read()
    tok = tok or tokenizer(spec)
    if spec.get("text.chat_template"):
        text = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
    else:
        ids = tok(text, add_special_tokens=False)["input_ids"]
    if not spec.get("text.chat_template") and spec.get("text.bos", True) and tok.bos_token_id is not None:
        ids = [tok.bos_token_id] + ids
    if len(ids) < n:
        raise ValueError(f"text has {len(ids)} tokens < {n}")
    return torch.tensor(ids[:n], dtype=torch.int64)


def store_dtype(t: torch.Tensor, keep_fp32: tuple[str, ...], name: str) -> torch.Tensor:
    if not t.is_floating_point():
        return t.contiguous()
    if name.endswith(keep_fp32):
        return t.float().contiguous()
    return t.to(torch.bfloat16).contiguous()
