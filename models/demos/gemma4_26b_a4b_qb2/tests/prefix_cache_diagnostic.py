# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Remote-only same-physical-cache probe; enabled explicitly by the CI runner."""

import json
import os
from pathlib import Path

import torch

import ttnn


def compare_paths(generator, tokens, table, cache, original, start):
    key = (tuple(tokens.flatten().tolist()), start)
    seen = getattr(generator, "_prefix_diagnostic_seen", set())
    if key in seen:
        return
    seen.add(key)
    generator._prefix_diagnostic_seen = seen
    model = generator.model
    length = tokens.shape[-1]
    replay_start = start or (length - 1) // 32 * 32
    values = {"original": generator._read_logits(original).flatten().clone()}
    # Each arm overwrites the replayed interval causally. Prefix rows before
    # replay_start remain unchanged; stale later rows are masked until replaced.
    for position in range(replay_start, length):
        ids = model.upload(tokens[:, position : position + 1].reshape(1, 1, 1, 1).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        current = model.upload(torch.tensor([[position]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        cache_pos = model.upload(torch.tensor([position], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        decoded = model.decode_forward(
            ids, current_pos=current, cache_pos=cache_pos, page_table=table, kv_cache=cache, batch=1
        )
    values["decode"] = generator._read_logits(decoded).flatten().clone()
    ids = model.upload(tokens[:, replay_start:].reshape(1, 1, 1, -1).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    continued = model.prefill_forward(ids, page_table=table, kv_cache=cache, start_pos=replay_start)
    values["continuation"] = generator._read_logits(continued).flatten().clone()
    references = getattr(generator, "_prefix_diagnostic_references", {})
    if start == 0:
        references[key[0]] = values["decode"].clone()
        generator._prefix_diagnostic_references = references
    elif start == (length - 1) // 32 * 32 and key[0] in references:
        values["cold_cache_decode"] = references[key[0]]
    summary = {"start": start, "replay_start": replay_start, "length": length}
    for name, logits in values.items():
        top = logits.topk(20)
        summary[name] = {"ids": top.indices.tolist(), "logits": top.values.tolist()}
    pairs = [("decode", "continuation"), ("original", "decode")]
    if "cold_cache_decode" in values:
        pairs.append(("original", "cold_cache_decode"))
    for left, right in pairs:
        delta = values[left] - values[right]
        summary[f"{left}_vs_{right}"] = {
            "max_abs": delta.abs().max().item(),
            "rms": delta.square().mean().sqrt().item(),
            "equal": torch.equal(values[left], values[right]),
        }
    output = Path(os.environ["GEMMA4_PREFIX_RESULTS"])
    torch.save(values, output / f"same-cache-length-{length}-start-{start}.pt")
    (output / f"same-cache-length-{length}-start-{start}.json").write_text(json.dumps(summary, indent=2))
    print("PREFIX_CACHE_SAME_CACHE " + json.dumps(summary), flush=True)
    # Restore the original execution path's final KV row before serving returns.
    restore_ids = model.upload(tokens[:, start:].reshape(1, 1, 1, -1).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    model.prefill_forward(restore_ids, page_table=table, kv_cache=cache, start_pos=start)
    assert summary["decode_vs_continuation"]["equal"], "Continuation differs from ordinary decode on identical prefix KV"
    if "cold_cache_decode" in values:
        assert summary["original_vs_cold_cache_decode"]["equal"], "Scheduler reuse differs from cold-cache decode"
