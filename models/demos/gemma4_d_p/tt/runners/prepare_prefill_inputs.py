# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prepare real-text token files and a manifest for the shared prefill producer."""

import argparse
import json
from pathlib import Path

from loguru import logger

from models.demos.gemma4_d_p.tt.runners.adapters.gemma4 import Gemma4PrefillAdapter, Gemma4ServiceConfig

BOOK_IDS = (135, 2600, 1184, 996, 1023, 1399)


def load_prompts(text_paths, num_slots, num_tokens, cache_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(Gemma4PrefillAdapter().hf_model_id)
    prompts = []
    for slot in range(num_slots):
        if text_paths:
            path = text_paths[slot]
        else:
            import requests

            book_id = BOOK_IDS[slot]
            path = cache_dir / f"pg{book_id}.txt"
            if not path.exists():
                url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                cache_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(response.content.decode("utf-8-sig"))
        tokens = tokenizer.encode(path.read_text(), add_special_tokens=True)
        if len(tokens) < num_tokens:
            raise ValueError(f"{path} has {len(tokens)} tokens; need {num_tokens}")
        prompts.append(tokens[:num_tokens])
        logger.info(f"slot={slot} text={path} tokens={num_tokens}")
    return prompts


def write_producer_manifest(output_dir, prompts):
    output_dir = Path(output_dir).resolve()
    slot_paths = []
    for slot, tokens in enumerate(prompts):
        slot_path = output_dir / f"slot_{slot}"
        slot_path.mkdir(parents=True, exist_ok=True)
        (slot_path / "metadata.json").write_text(json.dumps({"token_ids": tokens}) + "\n")
        slot_paths.append(str(slot_path))
    manifest = json.loads(Path(__file__).with_name("manifest.json").read_text())
    manifest["env"]["PREFILL_NUM_USERS"] = str(len(prompts))
    manifest["transport"] = {"connect_timeout_s": 1200}
    manifest["workload"] = {
        "max_requests": len(prompts),
        "interleave": "round_robin",
        "slot_prompts": slot_paths,
        "check_pcc": False,
    }
    manifest_path = output_dir / "producer.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/gemma4_prefill_inputs"))
    parser.add_argument(
        "--slots",
        type=int,
        default=Gemma4ServiceConfig.MAX_USER_SLOTS,
        choices=range(1, Gemma4ServiceConfig.MAX_USER_SLOTS + 1),
    )
    parser.add_argument("--tokens", type=int, default=Gemma4ServiceConfig.MAX_SEQ_LEN)
    parser.add_argument("--text", type=Path, action="append", help="One UTF-8 text file per slot")
    parser.add_argument("--text-cache", type=Path, default=Path("/tmp/gemma4_prefill_text"))
    args = parser.parse_args()
    if not 1 <= args.tokens <= Gemma4ServiceConfig.MAX_SEQ_LEN:
        parser.error(f"--tokens must be between 1 and {Gemma4ServiceConfig.MAX_SEQ_LEN}")
    if args.text and len(args.text) != args.slots:
        parser.error("supply one --text per slot")
    prompts = load_prompts(args.text, args.slots, args.tokens, args.text_cache)
    manifest_path = write_producer_manifest(args.output_dir, prompts)
    logger.info(f"Prepared {len(prompts)} prompts: {manifest_path}")


if __name__ == "__main__":
    main()
