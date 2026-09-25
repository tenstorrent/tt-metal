# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The canonical prefill input: one fixed token sequence per model, built once at intake and read by every step.

    python -m models.demos.common.bringup.reference.prompt --spec S        # build (or verify) $ART/<model>/input/prompt.json

HF parity, the chunked check, every golden, the device ladder and the contract test all prefill a prefix of the same
tokens. The file records how they were made and a sha256 of the ids; goldens store that hash in their manifest, and
the gate that freezes a test pins the golden, so no step can quietly test prefill on a different input.

spec.yaml ``text``:
    source: models/tt_transformers/tests/tale-of-two-cities.txt.bz2   # long real text (default)
    wrap: raw | user_turn | model_turn     # raw: BOS + text (base models)
                                           # model_turn: chat template with a short user request, then the text as
                                           #   the model's reply (instruction-tuned models predict model turns only)
                                           # user_turn: the text as one user message
    request: "Recite ..."                  # the user message for model_turn
    bos: true                              # raw only
    length: 56320                          # tokens to build (default: the longest ladder rung)

Records (when run as a gate step) prompt_tokens, prompt_hash_ok.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import os
import time
from pathlib import Path

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.core.freeze import sha256_file
from models.demos.common.bringup.core.spec import CODE_ROOT, Spec

DEFAULT_SOURCE = "models/tt_transformers/tests/tale-of-two-cities.txt.bz2"
DEFAULT_REQUEST = "Recite the Project Gutenberg text of A Tale of Two Cities by Charles Dickens, from the start."


def prompt_path(spec: Spec) -> Path:
    return spec.art / "input" / "prompt.json"


def ids_hash(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def wrap_mode(spec: Spec) -> str:
    w = spec.get("text.wrap")
    if w is None:
        w = "user_turn" if spec.get("text.chat_template") else "raw"
    if w not in ("raw", "user_turn", "model_turn"):
        raise ValueError(f"text.wrap {w!r}: use raw, user_turn or model_turn")
    return w


def _source(spec: Spec) -> tuple[Path, str]:
    src = spec.get("text.source", DEFAULT_SOURCE)
    path = Path(src) if Path(src).is_absolute() else CODE_ROOT / src
    opener = bz2.open if path.suffix == ".bz2" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return path, f.read()


def build_ids(spec: Spec, tok, n: int) -> tuple[list[int], dict]:
    path, text = _source(spec)
    mode = wrap_mode(spec)
    info = {
        "wrap": mode,
        "source": str(path.relative_to(CODE_ROOT)) if path.is_relative_to(CODE_ROOT) else str(path),
        "source_sha256": sha256_file(path),
    }
    if mode == "raw":
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if spec.get("text.bos", True) and tok.bos_token_id is not None:
            ids = [tok.bos_token_id] + ids
    elif mode == "user_turn":
        wrapped = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False)
        ids = tok(wrapped, add_special_tokens=False)["input_ids"]
    else:
        request = spec.get("text.request", DEFAULT_REQUEST)
        prefix = tok.apply_chat_template(
            [{"role": "user", "content": request}], add_generation_prompt=True, tokenize=False
        )
        pre = tok(prefix, add_special_tokens=False)["input_ids"]
        ids = pre + tok(text, add_special_tokens=False)["input_ids"]
        info.update(request=request, prefix_text=prefix, prefix_tokens=len(pre))
    if len(ids) < n:
        raise ValueError(f"text gives {len(ids)} tokens < {n}")
    return ids[:n], info


def default_length(spec: Spec) -> int:
    return int(spec.get("text.length") or max(r["seq"] for r in spec.data.get("ladder", [{"seq": 4096}])))


def build(spec: Spec, tok=None, force: bool = False) -> dict:
    """Write the prompt file, or verify an existing one is what the spec builds now. Returns the record."""
    from models.demos.common.bringup.reference.golden import tokenizer

    tok = tok or tokenizer(spec)
    n = default_length(spec)
    ids, info = build_ids(spec, tok, n)
    rec = {
        "model": spec.data.get("hf_id"),
        "n": n,
        "sha256": ids_hash(ids),
        **info,
        "token_ids": ids,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    p = prompt_path(spec)
    if p.exists() and not force:
        old = json.loads(p.read_text())
        if old["sha256"] != rec["sha256"]:
            raise SystemExit(
                f"{p} holds a different prompt (sha {old['sha256'][:12]} != {rec['sha256'][:12]}); "
                "goldens made from it would no longer match. Move it aside to rebuild."
            )
        return old
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    os.replace(tmp, p)
    return rec


def load(spec: Spec) -> dict | None:
    p = prompt_path(spec)
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    if ids_hash(rec["token_ids"]) != rec["sha256"]:
        raise ValueError(f"{p}: token ids do not match their recorded hash")
    return rec


def tokens(spec: Spec, n: int, tok=None) -> torch.Tensor:
    """The first n tokens of the canonical prompt; built in memory if the prompt file does not exist yet."""
    rec = load(spec)
    if rec is None:
        from models.demos.common.bringup.reference.golden import tokenizer

        ids, _ = build_ids(spec, tok or tokenizer(spec), n)
    else:
        ids = rec["token_ids"]
        if len(ids) < n:
            raise ValueError(f"the prompt has {len(ids)} tokens < {n}; raise text.length and rebuild")
    return torch.tensor(ids[:n], dtype=torch.int64)


def main(argv=None):
    from models.demos.common.bringup.reference.golden import load_spec

    ap = argparse.ArgumentParser()
    ap.add_argument("--spec")
    a = ap.parse_args(argv)
    spec = load_spec(a.spec)
    rec = build(spec)
    ok = load(spec) is not None
    print(f"prompt {prompt_path(spec)}: {rec['n']} tokens, wrap {rec['wrap']}, sha {rec['sha256'][:16]}")
    metrics.record("prompt_tokens", rec["n"])
    metrics.record("prompt_hash_ok", int(ok))


if __name__ == "__main__":
    main()
