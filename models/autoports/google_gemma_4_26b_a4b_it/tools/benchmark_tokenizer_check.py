# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Check the pinned native chat template offline; does not start TT or inference."""

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import try_to_load_from_cache
from transformers import AutoTokenizer

MODEL = "google/gemma-4-26B-A4B-it"
REVISION = "4d7ae4984b7db7de8f8457170b3f1a419ee76d52"


def check():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is two plus two?"},
    ]
    results = {}
    for thinking in (False, True):
        kwargs = dict(add_generation_prompt=True, enable_thinking=thinking)
        text = tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
        encoded = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if ids != tokenizer.encode(text, add_special_tokens=False):
            raise ValueError("Native tokenization differs from one render without added special tokens")
        if ids.count(tokenizer.bos_token_id) != 1:
            raise ValueError("Native prompt must contain exactly one BOS")
        if ("<|think|>" in text) != thinking:
            raise ValueError("Native template did not apply the requested thinking policy")
        results[str(thinking).lower()] = {
            "rendered": text,
            "token_ids": ids,
            "bos_count": 1,
            "native_render_matches_tokenization": True,
            "apply_chat_template_return_type": type(encoded).__name__,
        }
    files = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "tokenizer.json",
    ):
        cached = try_to_load_from_cache(MODEL, name, revision=REVISION)
        if not isinstance(cached, str):
            raise ValueError(f"Required pinned tokenizer file is not cached: {name}")
        path = Path(cached)
        files[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "revision": REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "local_files_only": True,
        "source_files": files,
        "template_probe": results,
        "generation_config": json.loads(Path(files["generation_config.json"]["path"]).read_text()),
        "scope": "Offline tokenizer check only; does not establish lm_eval request formatting or live server identity.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(check(), indent=2) + "\n")
    print(f"Pinned native template check passed: {args.output}")
