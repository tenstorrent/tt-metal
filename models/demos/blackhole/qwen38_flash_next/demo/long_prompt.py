# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Build a chat prompt of an exact token length: a book excerpt with a pass-key needle, then a question.

Host-only helpers (standard library plus tokenizer objects passed in); no ttnn import.
"""

import hashlib
import os
import urllib.request
from pathlib import Path

PLACEHOLDER = "<<QWEN38_CONTEXT>>"
_BOOK_TOKENS: dict[str, list[int]] = {}  # tokenized book text, cached per text


def load_context_text(config: dict, cache_dir) -> str:
    """Download the context text once into cache_dir and return it from start_marker on (the story, no header)."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{hashlib.sha256(config['context_url'].encode()).hexdigest()[:16]}.txt"
    if not path.exists():
        with urllib.request.urlopen(config["context_url"], timeout=120) as response:
            data = response.read()
        temp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temp_path.write_bytes(data)
        os.replace(temp_path, path)
    text = path.read_text(encoding="utf-8-sig")
    marker = config["start_marker"]
    position = text.find(marker)
    if position < 0:
        raise ValueError(f"start_marker {marker!r} not found in {path}")
    return text[position:]


def build_prompt_ids(tokenizer, render_chat, config: dict, context_text: str, prompt_len: int, context_start: int = 0):
    """Return (ids, info): exactly prompt_len token ids of chat prefix + book/needle + question + generation prompt.
    The book excerpt starts at token ``context_start`` of the text (0: the story's first token)."""

    rendered = render_chat(PLACEHOLDER + "\n\n" + config["question"])
    assert rendered.count(PLACEHOLDER) == 1, "the placeholder must occur exactly once in the rendered chat"
    prefix_text, suffix_text = rendered.split(PLACEHOLDER)
    prefix = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    suffix = tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    needle = tokenizer(config["needle"], add_special_tokens=False)["input_ids"]
    n_ctx = prompt_len - len(prefix) - len(suffix) - len(needle)
    if n_ctx < 16:
        raise ValueError(f"prompt_len {prompt_len} leaves only {n_ctx} context tokens (need at least 16)")
    if context_text not in _BOOK_TOKENS:
        _BOOK_TOKENS[context_text] = tokenizer(context_text, add_special_tokens=False)["input_ids"]
    book_tokens = _BOOK_TOKENS[context_text]
    if len(book_tokens) < context_start + n_ctx:
        raise ValueError(
            f"prompt_len {prompt_len} needs {n_ctx} context tokens from token {context_start}, "
            f"the text has {len(book_tokens)}"
        )
    ctx = book_tokens[context_start : context_start + n_ctx]
    cut = int(n_ctx * config["needle_depth"])
    ids = prefix + ctx[:cut] + needle + ctx[cut:] + suffix
    assert len(ids) == prompt_len, f"{len(ids)} != {prompt_len}"
    info = {
        "prompt_len": prompt_len,
        "prefix_tokens": len(prefix),
        "suffix_tokens": len(suffix),
        "needle_tokens": len(needle),
        "context_tokens": n_ctx,
        "needle_token_offset": len(prefix) + cut,
        "pass_key": config["pass_key"],
    }
    return ids, info


def user_pass_key(config: dict, user: int) -> str:
    """A distinct 5-digit pass key per user; user 0 keeps the config's key."""

    base = int(config["pass_key"])
    return str(10_000 + (base - 10_000 + 2791 * user) % 90_000)  # 2791 is coprime to 90000: distinct per user


def build_user_prompt_ids(tokenizer, render_chat, config: dict, context_text: str, prompt_len: int, user: int):
    """Prompt of user ``user``: the same length for every user, its own non-overlapping book excerpt (token offset
    ``user * context_tokens``) and its own pass key in the needle.  Returns (ids, info)."""

    key = user_pass_key(config, user)
    user_config = dict(config, pass_key=key, needle=config["needle"].replace(config["pass_key"], key))
    assert key in user_config["needle"]
    _, info0 = build_prompt_ids(tokenizer, render_chat, config, context_text, prompt_len)
    start = user * info0["context_tokens"]  # the needle has the same token count for 5-digit keys of equal length
    ids, info = build_prompt_ids(tokenizer, render_chat, user_config, context_text, prompt_len, context_start=start)
    info["context_start"] = start
    return ids, info
