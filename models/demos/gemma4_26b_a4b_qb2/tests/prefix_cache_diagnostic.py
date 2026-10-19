# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Remote-only same-physical-cache probe; enabled explicitly by the CI runner."""

import json
import os
from pathlib import Path

import torch

import ttnn


def compare_paths(generator, tokens, table, cache, original, start):
    model = generator.model
    position = tokens.shape[-1] - 1
    ids = model.upload(tokens[:, -1:].reshape(1, 1, 1, 1).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    current = model.upload(torch.tensor([[position]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    cache_pos = model.upload(torch.tensor([position], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    values = {"original": generator._read_logits(original).flatten().clone()}
    # Replaying the final position only overwrites that position. All causal
    # prefix rows remain unchanged, so both arms consume identical history.
    decoded = model.decode_forward(
        ids, current_pos=current, cache_pos=cache_pos, page_table=table, kv_cache=cache, batch=1
    )
    values["decode"] = generator._read_logits(decoded).flatten().clone()
    continued = model.prefill_forward(ids, page_table=table, kv_cache=cache, start_pos=position)
    values["continuation"] = generator._read_logits(continued).flatten().clone()
    summary = {"start": start, "position": position}
    for name, logits in values.items():
        top = logits.topk(20)
        summary[name] = {"ids": top.indices.tolist(), "logits": top.values.tolist()}
    for left, right in (("decode", "continuation"), ("original", "decode")):
        delta = values[left] - values[right]
        summary[f"{left}_vs_{right}"] = {
            "max_abs": delta.abs().max().item(),
            "rms": delta.square().mean().sqrt().item(),
            "equal": torch.equal(values[left], values[right]),
        }
    output = Path(os.environ["GEMMA4_PREFIX_RESULTS"])
    torch.save(values, output / f"same-cache-start-{start}.pt")
    (output / f"same-cache-start-{start}.json").write_text(json.dumps(summary, indent=2))
    print("PREFIX_CACHE_SAME_CACHE " + json.dumps(summary), flush=True)
    # Restore the original execution path's final KV row before serving returns.
    restore_ids = model.upload(tokens[:, start:].reshape(1, 1, 1, -1).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    model.prefill_forward(restore_ids, page_table=table, kv_cache=cache, start_pos=start)
