# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Offline NLLB text translation on a caller-selected TTNN device."""

import argparse
import contextlib
import json
from pathlib import Path
import re
import sys
import os
import time


@contextlib.contextmanager
def official_tokenizer_view(directory):
    """Load only four exact official tokenizer assets; never read model weights.

    HF cache symlinks are supported by validating and copying their target bytes.
    Other model metadata is excluded from the temporary tokenizer-only view.
    """
    import hashlib
    import tempfile

    directory = Path(directory)
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model")
    manifest = json.loads(Path(__file__).with_name("official-assets.json").read_text())
    models = manifest["models"]
    pins = {name: models["600m"]["files"][name] for name in names}
    if any({name: model["files"][name] for name in names} != pins for model in models.values()):
        raise ValueError("Official models no longer share the same tokenizer assets")
    if (directory / "added_tokens.json").exists() or (directory / "added_tokens.json").is_symlink():
        raise ValueError("Unreviewed added_tokens.json is not supported")
    with tempfile.TemporaryDirectory(prefix="nllb-tokenizer-") as temporary:
        view = Path(temporary)
        for name, pin in pins.items():
            path = directory / name
            if not path.is_file() or path.stat().st_size != pin["bytes"]:
                raise ValueError(f"Missing or incompatible official tokenizer asset: {name}")
            with path.open("rb") as stream:
                data = stream.read(pin["bytes"] + 1)
            if len(data) != pin["bytes"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
                raise ValueError(f"Incompatible official tokenizer asset: {name}")
            (view / name).write_bytes(data)
        yield view


def load_text_inputs(checkpoint, config, source_language, target_language, texts, *, tokenizer_directory=None):
    from transformers import AutoTokenizer
    import numpy as np

    if __package__:
        from .nllb_validation import validate_inputs, integer
    else:
        from nllb_validation import validate_inputs, integer
    if not isinstance(texts, (list, tuple)) or not 1 <= len(texts) <= 4:
        raise ValueError("Supply one to four texts")
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("All texts must be strings")
    for language in (source_language, target_language):
        if not isinstance(language, str) or not re.fullmatch(r"[a-z]{3}_[A-Z][a-z]{3}", language):
            raise ValueError(f"Unknown NLLB language: {language}")
    path = Path(checkpoint)
    tokenizer_dir = (
        Path(tokenizer_directory) if tokenizer_directory is not None else (path if path.is_dir() else path.parent)
    )
    if not tokenizer_dir.is_dir():
        raise ValueError(f"Tokenizer directory does not exist: {tokenizer_dir}")
    with official_tokenizer_view(tokenizer_dir) as verified:
        tokenizer = AutoTokenizer.from_pretrained(
            str(verified), use_fast=True, local_files_only=True, trust_remote_code=False, src_lang=source_language
        )
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Tokenizer vocabulary must be a nonempty mapping")
    for value in vocab.values():
        integer(value, "tokenizer token ID", 0, config["vocab_size"] - 1)
    for language in (source_language, target_language):
        if language not in vocab or vocab[language] in (0, 1, 2, tokenizer.unk_token_id):
            raise ValueError(f"Unknown NLLB language: {language}")
    if tokenizer.pad_token_id != 1 or tokenizer.eos_token_id != 2 or max(vocab.values()) >= config["vocab_size"]:
        raise ValueError("Tokenizer and checkpoint vocabulary/special tokens disagree")
    encoded = tokenizer(list(texts), padding=True, truncation=False, return_tensors="np")
    ids = np.asarray(encoded["input_ids"], dtype=np.int64)
    mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
    validate_inputs(ids, mask, config)
    return tokenizer, ids, mask, int(vocab[target_language])


def translate(
    checkpoint,
    config,
    device,
    source_language,
    target_language,
    texts,
    *,
    max_new_tokens,
    precision="bf16",
    tokenizer_directory=None,
):
    """Use a caller-owned open device; return translations and complete token rows.

    Before importing TTNN, set TT_METAL_TRACE_ALLOC_TRACKING=1 and
    TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0. The caller must reserve the
    backend DEVICE_OPTIONS (64 MiB trace region) and enable program cache.
    This call releases its model traces but never configures/closes the device.
    On unresolved cleanup, runtime_setup.retained_owners() retains the model
    and device: do not close/reset/reopen it or claim recovery.
    """
    if __package__:
        from . import runtime_setup
    else:
        import runtime_setup
    runtime_setup.configure_tracking()
    import ttnn

    if __package__:
        from .backend import create_backend
    else:
        from backend import create_backend
    if __package__:
        from .nllb_validation import validate_config, integer
    else:
        from nllb_validation import validate_config, integer
    config = validate_config(config)
    integer(max_new_tokens, "max_new_tokens", 1, min(256, config["max_position_embeddings"] - 1))
    tokenizer, ids, mask, target = load_text_inputs(
        checkpoint, config, source_language, target_language, texts, tokenizer_directory=tokenizer_directory
    )
    with runtime_setup.RuntimeOwner(device) as owner:
        model = owner.bind(create_backend(checkpoint, config, device, precision=precision))
        ttnn.synchronize_device(device)
        started = time.perf_counter()
        tokens = model.generate(ids, mask, target, max_new_tokens)
        ttnn.synchronize_device(device)
        generation_seconds = time.perf_counter() - started
        return {
            "translations": tokenizer.batch_decode(tokens, skip_special_tokens=True),
            "token_ids": tokens.tolist(),
            "precision_policy": model.precision_policy,
            "source_language": source_language,
            "target_language": target_language,
            "generation_seconds": generation_seconds,
        }


def main(argv=None, *, result_stream=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Local .bin file or HF directory")
    parser.add_argument("--config", help="Config JSON; defaults to checkpoint's config.json")
    parser.add_argument("--tokenizer-directory", help="Local tokenizer directory; defaults to checkpoint directory")
    parser.add_argument("--source-language", required=True)
    parser.add_argument("--target-language", required=True)
    parser.add_argument("--device", required=True, type=int, help="Visible TTNN device index")
    parser.add_argument("--max-new-tokens", required=True, type=int, help="1..256, includes language token")
    parser.add_argument("--precision", choices=("bf16", "bfp8_b"), default="bf16")
    parser.add_argument("--text", required=True, action="append", help="Repeat for batches up to four")
    parser.add_argument("--output", help="Optional JSON output file")
    args = parser.parse_args(argv)
    if __package__:
        from . import runtime_setup
    else:
        import runtime_setup
    runtime_setup.configure_tracking()
    import torch

    if __package__:
        from .nllb_validation import validate_config, integer
    else:
        from nllb_validation import validate_config, integer
    checkpoint = Path(args.checkpoint)
    config_path = (
        Path(args.config) if args.config else (checkpoint if checkpoint.is_dir() else checkpoint.parent) / "config.json"
    )
    config = validate_config(json.loads(config_path.read_text()))
    integer(args.device, "device", 0, 2**31 - 1)
    integer(args.max_new_tokens, "max_new_tokens", 1, min(256, config["max_position_embeddings"] - 1))
    # Fail tokenization and host-input errors before opening hardware.
    load_text_inputs(
        checkpoint,
        config,
        args.source_language,
        args.target_language,
        args.text,
        tokenizer_directory=args.tokenizer_directory,
    )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with contextlib.redirect_stdout(sys.stderr):
        with runtime_setup.RuntimeOwner() as owner:
            device = owner.open(args.device)
            result = translate(
                checkpoint,
                config,
                device,
                args.source_language,
                args.target_language,
                args.text,
                max_new_tokens=args.max_new_tokens,
                precision=args.precision,
                tokenizer_directory=args.tokenizer_directory,
            )
    document = json.dumps(result, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(document + "\n", encoding="utf-8")
    print(document, file=result_stream, flush=True)
    return 0


def cli(argv=None):
    """Keep Python/native startup and shutdown diagnostics off the JSON channel."""
    sys.stdout.flush()
    with os.fdopen(os.dup(1), "w", encoding="utf-8") as result_stream:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            return main(argv, result_stream=result_stream)


if __name__ == "__main__":
    raise SystemExit(cli())
