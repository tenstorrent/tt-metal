# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Feed real text to every Gemma4 KV slot and wait for device layer completions."""

import argparse
import json
import os
import struct
import time
from pathlib import Path

import numpy as np
from loguru import logger

from models.demos.gemma4_d_p.tt.runners.adapter import Gemma4PrefillAdapter, Gemma4ServiceConfig

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
    return prompts, tokenizer.pad_token_id


def iter_chunks(prompts, pad_token_id):
    chunk_size = Gemma4ServiceConfig.CHUNK_SIZE
    for start in range(0, max(map(len, prompts)), chunk_size):
        for slot, tokens in enumerate(prompts):
            if start >= len(tokens):
                continue
            end = min(start + chunk_size, len(tokens))
            chunk = tokens[start:end] + [pad_token_id] * (start + chunk_size - end)
            payload = np.asarray(chunk, dtype="<u4").reshape(8, 1, chunk_size // 8)
            yield slot, start, end, payload


def wait_for_layers(channel, timeout_s):
    expected = Gemma4ServiceConfig.NUM_LAYERS
    deadline = time.monotonic() + timeout_s
    completed = 0
    while completed < expected:
        completed += channel.try_consume_all()
        if completed > expected:
            raise RuntimeError(f"Received {completed} layer acknowledgments for a {expected}-layer chunk")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Received {completed}/{expected} layer acknowledgments after {timeout_s}s")
        if completed < expected:
            time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-id", default=os.getenv("PREFILL_H2D_SERVICE_ID", "gemma4_prefill"))
    parser.add_argument(
        "--slots",
        type=int,
        default=int(os.getenv("PREFILL_NUM_USERS", str(Gemma4ServiceConfig.MAX_USERS))),
        choices=range(1, Gemma4ServiceConfig.MAX_USERS + 1),
    )
    parser.add_argument("--tokens", type=int, default=Gemma4ServiceConfig.MAX_SEQ_LEN)
    parser.add_argument(
        "--text", type=Path, action="append", help="One UTF-8 text file per slot; defaults to six Gutenberg books"
    )
    parser.add_argument("--text-cache", type=Path, default=Path("/tmp/gemma4_prefill_text"))
    parser.add_argument(
        "--timeout", type=float, default=1200, help="Connection and per-chunk completion timeout in seconds"
    )
    parser.add_argument("--shutdown", action="store_true", help="Shut down the service after validation")
    parser.add_argument("--results", type=Path, help="Write per-slot token counts and chunk timings as JSON")
    args = parser.parse_args()
    if not 1 <= args.tokens <= Gemma4ServiceConfig.MAX_SEQ_LEN:
        parser.error("--tokens must be between 1 and 262144")
    if args.text and len(args.text) != args.slots:
        parser.error("supply one --text per slot")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    prompts, pad_token_id = load_prompts(args.text, args.slots, args.tokens, args.text_cache)
    import ttnn

    timeout_ms = int(args.timeout * 1000)
    service = ttnn.H2DStreamService.connect(args.service_id, timeout_ms=timeout_ms)
    channel = ttnn.InterProcessCounterChannel.connect(
        f"/tt_prefill_layer_acks_{args.service_id}",
        connect_timeout_ms=timeout_ms,
    )
    if channel.try_consume_all():
        raise RuntimeError("The layer acknowledgment channel contains stale completions")

    chunks = []
    start_time = time.perf_counter()
    for slot, start, end, payload in iter_chunks(prompts, pad_token_id):
        chunk_start = time.perf_counter()
        service.forward_to_tensor_bytes(payload, metadata=struct.pack("<III", slot, start, end))
        wait_for_layers(channel, args.timeout)
        elapsed = time.perf_counter() - chunk_start
        chunks.append(dict(slot=slot, start=start, end=end, seconds=elapsed))
        logger.info(f"slot={slot} [{start},{end}) completed 60 layers in {elapsed:.3f}s")

    elapsed = time.perf_counter() - start_time
    result = dict(tokens_per_slot=[len(prompt) for prompt in prompts], chunks=chunks, seconds=elapsed)
    if args.results:
        args.results.write_text(json.dumps(result, indent=2) + "\n")
    logger.info(f"PASS: {args.slots} slots, {len(chunks)} chunks, {sum(map(len, prompts))} tokens in {elapsed:.1f}s")
    if args.shutdown:
        service.forward_to_tensor_bytes(
            np.zeros((8, 1, Gemma4ServiceConfig.CHUNK_SIZE // 8), dtype="<u4"),
            metadata=struct.pack("<iii", -1, -1, -1),
        )
        service.barrier()
        logger.info("Sent shutdown sentinel")


if __name__ == "__main__":
    main()
