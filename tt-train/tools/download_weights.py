# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)


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
    config = AutoConfig.from_pretrained(model_ref, cache_dir=cache_dir, trust_remote_code=True)

    # Pick the right head automatically (seq2seq vs causal)
    ModelCls = AutoModelForSeq2SeqLM if getattr(config, "is_encoder_decoder", False) else AutoModelForCausalLM
    model = ModelCls.from_pretrained(model_ref, cache_dir=cache_dir, trust_remote_code=True, low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, cache_dir=cache_dir, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Load a GPT-2 model by HF id or local path and print state_dict keys.")
    parser.add_argument(
        "-m",
        "--model",
        default="gpt2",
        help="Hugging Face model id (e.g. 'gpt2') OR a local path to a model dir (supports ~ and $VARS).",
    )
    parser.add_argument(
        "-c",
        "--cache-dir",
        default="~/.cache/huggingface/hub",
        help="Cache directory for model files (supports ~ and $VARS). Default: ~/.cache/huggingface",
    )
    parser.add_argument(
        "--save_safetensors",
        action="store_true",
        help="Save model weights in SafeTensors format.",
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, cache_dir=args.cache_dir)

    if args.save_safetensors:
        raise NotImplementedError("Saving in SafeTensors format is not implemented yet.")
        # model_path = os.path.join(args.cache_dir, args.model)
        # os.makedirs(model_path, exist_ok=True)
        # save_file(model.state_dict(), os.path.join(model_path, "model.safetensors"))
        # print(f"Model weights saved in SafeTensors format at {model_path}/model.safetensors")


if __name__ == "__main__":
    main()
