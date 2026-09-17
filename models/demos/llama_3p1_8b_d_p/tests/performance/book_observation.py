# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Exact native final-row assembly and observational book token ranking; no acceptance threshold."""

import hashlib
import heapq
import json
import math
import struct

VOCAB_SIZE = 128256
SHARD_WIDTH = 16032
OBSERVATION_SCOPE = "book_continuation_observation_no_semantic_threshold"
METADATA_KEYS = (
    "book_id",
    "title",
    "expected_next_token_id",
    "expected_next_token",
    "expected_next_word",
    "trailing_prompt_text",
    "actual_following_text",
)


def final_position(tokens):
    if type(tokens) is not int or tokens <= 0 or tokens % 1024:
        raise ValueError("Only complete 1024-token chunks have this final-row mapping")
    position = tokens - 1
    return dict(
        position=position, chunk_start=(position // 1024) * 1024, sp=(position // 256) % 4, local_row=position % 256
    )


def assemble_final_logits(records, tokens, *, shard_width=SHARD_WIDTH):
    expected = final_position(tokens)
    if len(records) != 8 or {r["tp"] for r in records} != set(range(8)):
        raise ValueError("Exactly eight unique TP vocabulary slices are required")
    result = []
    for row in sorted(records, key=lambda r: r["tp"]):
        if any(type(row[key]) is not int or row[key] != expected[key] for key in ("position", "sp", "local_row")):
            raise ValueError("Vocabulary slice comes from the wrong real token row")
        if type(row["tp"]) is not int or len(row["values"]) != shard_width:
            raise ValueError("Vocabulary slice width or TP identity differs")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in row["values"]):
            raise ValueError("Final vocabulary logits must all be finite")
        result.extend(row["values"])
    return result


def rank_logits(values, expected_id, decode):
    if len(values) < 5 or type(expected_id) is not int or not 0 <= expected_id < len(values):
        raise ValueError("Invalid vocabulary or expected token ID")
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
        raise ValueError("Final vocabulary logits must all be finite")
    ids = heapq.nsmallest(5, range(len(values)), key=lambda i: (-values[i], i))
    maximum = values[ids[0]]
    denominator = math.fsum(math.exp(x - maximum) for x in values)
    expected = values[expected_id]
    return dict(
        argmax_token_id=ids[0],
        top5=[
            dict(token_id=i, piece=decode(i), logit=values[i], probability=math.exp(values[i] - maximum) / denominator)
            for i in ids
        ],
        expected_next_token_rank=1
        + sum(x > expected or (x == expected and i < expected_id) for i, x in enumerate(values)),
        expected_next_token_probability=math.exp(expected - maximum) / denominator,
        final_logits_float32_sha256=hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest(),
    )


def validate_prompts(prompts, tokens):
    if len(prompts) != 2 or [p["slot"] for p in prompts] != [0, 1]:
        raise ValueError("Ordered independent slots0/1 are required")
    books = []
    for p in prompts:
        ids, meta = p["token_ids"], p["metadata"]
        if len(ids) != tokens or any(type(i) is not int or not 0 <= i < VOCAB_SIZE for i in ids):
            raise ValueError("Fixture token IDs must have exact context length and vocabulary")
        if any(
            meta.get(key) != value
            for key, value in dict(
                prompt_tokens=tokens, final_prompt_position=tokens - 1, vocab_size=VOCAB_SIZE, bos_count=1
            ).items()
        ):
            raise ValueError("Fixture final-position/length/vocabulary metadata differs")
        digest = hashlib.sha256((json.dumps(ids, separators=(",", ":")) + "\n").encode()).hexdigest()
        if meta.get("token_ids_sha256") != digest:
            raise ValueError("Prompt token IDs differ from their frozen fixture file digest")
        bos = meta["bos_token_id"]
        if type(bos) is not int or ids[0] != bos or ids.count(bos) != 1:
            raise ValueError("Exactly one leading BOS is required")
        if type(meta["expected_next_token_id"]) is not int or not 0 <= meta["expected_next_token_id"] < VOCAB_SIZE:
            raise ValueError("Expected continuation token ID is outside vocabulary")
        if any(
            not isinstance(meta[key], str) for key in METADATA_KEYS if key not in ("book_id", "expected_next_token_id")
        ):
            raise ValueError("Readable book continuation metadata is required")
        books.append(meta["book_id"])
    if books[0] == books[1] or prompts[0]["token_ids"] == prompts[1]["token_ids"]:
        raise ValueError("Two distinct books and prompts are required")


def synthetic_prompts(tokens):
    """Small deterministic metadata for stdlib verifier tests; never used by the device entry."""
    prompts = [
        dict(
            slot=slot,
            token_ids=[128000] + [slot + 10] * (tokens - 1),
            metadata=dict(
                prompt_tokens=tokens,
                final_prompt_position=tokens - 1,
                vocab_size=VOCAB_SIZE,
                bos_token_id=128000,
                bos_count=1,
                book_id=str(1342 if slot == 0 else 1400),
                title=f"Synthetic book {slot}",
                expected_next_token_id=slot + 10,
                expected_next_token=" continuation",
                expected_next_word="continuation",
                trailing_prompt_text="synthetic trailing text",
                actual_following_text=" continuation text",
            ),
        )
        for slot in range(2)
    ]
    for prompt in prompts:
        prompt["metadata"]["token_ids_sha256"] = hashlib.sha256(
            (json.dumps(prompt["token_ids"], separators=(",", ":")) + "\n").encode()
        ).hexdigest()
    return prompts
