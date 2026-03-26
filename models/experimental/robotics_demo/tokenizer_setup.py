# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tokenizer setup helper for the robotics demo.

Pre-downloads and caches the Gemma tokenizer so the live demo
never blocks on HuggingFace authentication or network calls.

Run this ONCE during setup:
    python models/experimental/robotics_demo/tokenizer_setup.py

After running, the tokenizer is cached locally and the demo
uses it automatically (no --use-gemma-tokenizer flag needed).
"""

import os
import sys
from pathlib import Path

CACHE_DIR = Path(__file__).parent / ".tokenizer_cache"


def download_gemma_tokenizer(model_name: str = "google/gemma-2b") -> Path:
    """Download and cache the Gemma tokenizer locally."""
    CACHE_DIR.mkdir(exist_ok=True)

    try:
        from transformers import AutoTokenizer
        print(f"Downloading Gemma tokenizer from {model_name}...")
        tok = AutoTokenizer.from_pretrained(model_name, cache_dir=str(CACHE_DIR))
        tok.save_pretrained(str(CACHE_DIR / "gemma_tokenizer"))
        print(f"Tokenizer cached at: {CACHE_DIR / 'gemma_tokenizer'}")
        return CACHE_DIR / "gemma_tokenizer"
    except Exception as e:
        print(f"Could not download Gemma tokenizer: {e}")
        print("The demo will fall back to SimpleRoboticsTokenizer.")
        return None


def get_cached_tokenizer():
    """Load the cached Gemma tokenizer, or return None if not available."""
    cached = CACHE_DIR / "gemma_tokenizer"
    if cached.exists():
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(str(cached))
        except Exception:
            pass
    return None


class DemoTokenizer:
    """
    Auto-selecting tokenizer for customer demos.

    Tries cached Gemma first (best quality, matches PI0 training),
    then falls back to SimpleRoboticsTokenizer (always works).
    """

    def __init__(self):
        self._gemma = get_cached_tokenizer()
        self._fallback = None
        if self._gemma is not None:
            self.name = "Gemma (SentencePiece)"
        else:
            self.name = "SimpleRoboticsTokenizer"

    def encode(self, text: str, max_length: int = 32):
        if self._gemma is not None:
            encoded = self._gemma(
                text, max_length=max_length, padding="max_length",
                truncation=True, return_tensors="pt",
            )
            tokens = encoded["input_ids"][0].tolist()
            mask = encoded["attention_mask"][0].bool().tolist()
            return tokens, mask

        # Fallback: word-based tokenizer
        import torch
        vocab = {
            "<pad>": 0, "<bos>": 2, "<eos>": 3,
            "pick": 100, "place": 101, "grasp": 102, "push": 106,
            "lift": 108, "reach": 105, "move": 104,
            "cube": 200, "block": 201, "object": 202,
            "up": 300, "right": 303, "forward": 304,
            "the": 400, "to": 403, "and": 405,
        }
        tokens = [vocab["<bos>"]]
        for w in text.lower().split():
            tokens.append(vocab.get(w, hash(w) % 255000 + 1000))
        tokens.append(vocab["<eos>"])
        mask = [True] * len(tokens)
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))
            mask += [False] * (max_length - len(mask))
        else:
            tokens, mask = tokens[:max_length], mask[:max_length]
        return tokens, mask


if __name__ == "__main__":
    print("=" * 60)
    print("  Tenstorrent Robotics Demo -- Tokenizer Setup")
    print("=" * 60)
    print()
    print("This script downloads the Gemma tokenizer for offline use.")
    print("You need a HuggingFace account with Gemma access.")
    print()

    hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Tip: Set HUGGING_FACE_HUB_TOKEN or run `huggingface-cli login` first.")
        print()

    path = download_gemma_tokenizer()
    if path:
        print(f"\nSuccess! Tokenizer ready at: {path}")
        print("The demo will use it automatically.")
    else:
        print("\nFailed. The demo will use the fallback tokenizer.")
        print("This still works, but task understanding may be less accurate.")
