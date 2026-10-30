# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""End-of-prefill snapshot: everything decode needs (KV cache, position, rope delta, page table) plus prefill stages."""
from pathlib import Path

import torch

SNAPSHOT_NAME = "prefill_snapshot.pt"


def meta_for(cfg, grid, teacher_tokens):
    """What a snapshot must match to be reused: the run's model/config identity and the HF teacher tokens."""
    return {
        "cache_key": cfg.cache_key(grid),
        "size": cfg.size,
        "kv_blocks": cfg.kv_blocks,
        "teacher_tokens": list(teacher_tokens),
    }


def save(path, meta, kv_cache, decoding_pos, rope_delta, page_table, stages, to_host):
    torch.save(
        {
            "meta": meta,
            "kv": [[to_host(t).to(torch.bfloat16) for t in layer] for layer in kv_cache],
            "decoding_pos": int(decoding_pos),
            "rope_delta": float(rope_delta),
            "page_table": page_table.clone(),
            "stages": {k: v.clone() for k, v in stages.items()},
        },
        path,
    )


def load(path, meta):
    """Load a snapshot (a file or a run folder holding one) and check it was taken for the same run config."""
    path = Path(path)
    if path.is_dir():
        path = path / SNAPSHOT_NAME
    snap = torch.load(path, weights_only=False)
    want, have = meta, snap["meta"]
    diff = {k: (have.get(k), want[k]) for k in want if have.get(k) != want[k]}
    if diff:
        raise ValueError(f"prefill snapshot {path} was taken for a different run (snapshot, now): {diff}")
    return snap


def restore_kv(kv_cache, snap):
    """Write the snapshot's KV cache into the model's (freshly allocated) device KV cache, in place."""
    import ttnn

    for layer, saved in zip(kv_cache, snap["kv"]):
        for dev, host in zip(layer, saved):
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(host, dtype=dev.dtype, layout=dev.layout), dev)
