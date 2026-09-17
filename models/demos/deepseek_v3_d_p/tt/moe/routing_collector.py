"""Retain-only capture of MoE expert selections, for pairing routing with device timings.

Enabled by ``TT_MOE_ROUTING_CAPTURE=<global layer indices>`` (e.g. ``"18,23"``); off otherwise.

The constraint this exists to satisfy: nothing here may synchronize the host inside the timed
region. ``retain`` only keeps a Python reference to the indices tensor the forward has already
moved to DRAM, which stops the allocator reusing that buffer. Every host read happens in
``drain``, after the measurement window has closed.

That is low intrusion, not zero: holding the buffer extends its lifetime and changes the
allocation pattern. Compare a collector-on run against a collector-off one on identical input
before treating either run's Dispatch/Combine numbers as representative.

Eager only. Under trace, Python hooks run at capture time rather than replay, and one retained
address shows only its latest contents -- see ``TtPrefillRuntime._forward_traced``.
"""

import os
from pathlib import Path

import torch
from loguru import logger

import ttnn

_layers = None
_retained = []
_request = (0, 0)
_seq = {}


def target_layers():
    """Global layer indices to capture, parsed once from the environment."""
    global _layers
    if _layers is None:
        raw = os.environ.get("TT_MOE_ROUTING_CAPTURE", "")
        _layers = frozenset(int(tok) for tok in raw.replace(" ", "").split(",") if tok)
        if _layers:
            logger.info(f"[routing_collector] capturing layers {sorted(_layers)}")
    return _layers


def wants(layer_idx):
    return layer_idx in target_layers()


def set_request(request_id, chunk_id):
    """Publish the chunk identity the next forwards belong to. The MoE forward is not given
    request_id, so it reads it from here -- the same indirection the traced path uses for
    _trace_request_id."""
    global _request
    _request = (request_id, chunk_id)


def retain(layer_idx, indices, *, actual_start, actual_isl, num_routed_experts, num_experts_per_tok):
    """Keep ``indices`` alive. No copy, no conversion, no host read: this runs inside the
    region being timed."""
    request_id, chunk_id = _request
    # Occurrence index for this layer, recorded rather than inferred: compile()'s WARMUP pass and
    # chunk 0 both arrive as (request_id=0, actual_start=0), so the metadata alone cannot order them.
    _seq[layer_idx] = _seq.get(layer_idx, -1) + 1
    _retained.append(
        dict(
            layer=layer_idx,
            seq=_seq[layer_idx],
            request_id=request_id,
            chunk_id=chunk_id,
            actual_start=actual_start,
            actual_isl=actual_isl,
            num_routed_experts=num_routed_experts,
            num_experts_per_tok=num_experts_per_tok,
            tensor=indices,
        )
    )


def _validate(ids, entry):
    """Structural checks on one shard. Returns a list of complaints, empty when sound."""
    problems = []
    n_experts = entry["num_routed_experts"]
    topk = entry["num_experts_per_tok"]

    # Expert `n_experts` is the padding sentinel (ExpertMapping.create_dispatch_table maps that
    # column to -1); expert n_experts-1 is a real expert. Counting them together corrupts every
    # per-expert total, so they are separated here rather than by a range check alone.
    real = ids[ids != n_experts]
    if real.numel() and (real.min() < 0 or real.max() >= n_experts):
        problems.append(f"expert id out of range [0,{n_experts}): min={int(real.min())} max={int(real.max())}")

    if ids.numel() % topk:
        problems.append(f"{ids.numel()} assignments is not a multiple of top-k={topk}")
    else:
        rows = ids.view(-1, topk)
        distinct = torch.tensor([len(set(r.tolist())) for r in rows])
        bad = int((distinct != topk).sum())
        if bad:
            problems.append(f"{bad} of {rows.shape[0]} tokens do not select {topk} distinct experts")
    return problems


def drain(mesh_device, out_path):
    """Synchronize, export every retained selection, release the references.

    Call only after the whole request has finished on every rank and the measurement boundary
    has closed -- not per forward.
    """
    if not _retained:
        return None

    ttnn.synchronize_device(mesh_device)
    records, problems = [], []
    for entry in _retained:
        shards = ttnn.get_device_tensors(entry["tensor"])
        for shard_idx, shard in enumerate(shards):
            ids = ttnn.to_torch(shard).to(torch.int32).flatten()
            problems += [f"L{entry['layer']} shard {shard_idx}: {p}" for p in _validate(ids, entry)]
            records.append(
                {k: v for k, v in entry.items() if k != "tensor"} | dict(shard_index=shard_idx, expert_ids=ids)
            )

    _retained.clear()

    for problem in problems:
        logger.error(f"[routing_collector] {problem}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mesh_device_ids = [d.id() for d in mesh_device.get_devices()]
    except Exception:  # best effort; shard_index is the authoritative identity
        mesh_device_ids = None
    torch.save(dict(records=records, problems=problems, mesh_device_ids=mesh_device_ids), out_path)
    logger.info(f"[routing_collector] wrote {len(records)} shard records to {out_path}")
    return out_path
