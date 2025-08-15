#!/usr/bin/env python3
import argparse
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def _expand(path: str | None) -> str | None:
    if path is None:
        return None
    # Expand ~ and $VARS but keep plain model ids like "gpt2" untouched
    expanded = os.path.expanduser(os.path.expandvars(path))
    return expanded


def load_model(model_ref: str, cache_dir: str | None):
    model_ref = _expand(model_ref)
    if cache_dir:
        cache_dir = _expand(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)

    print(f"Loading model from '{model_ref}' (cache_dir='{cache_dir}')...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_ref, cache_dir=cache_dir)
    model = GPT2LMHeadModel.from_pretrained(model_ref, cache_dir=cache_dir)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Load a GPT-2 model by HF id or local path and print state_dict keys.")
    parser.add_argument(
        "model",
        help="Hugging Face model id (e.g. 'gpt2') OR a local path to a model dir (supports ~ and $VARS).",
    )
    parser.add_argument(
        "-c",
        "--cache-dir",
        default="~/.cache/huggingface/hub",
        help="Cache directory for model files (supports ~ and $VARS). Default: ~/.cache/huggingface",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, cache_dir=args.cache_dir)
    state_dict = model.state_dict()


if __name__ == "__main__":
    main()
