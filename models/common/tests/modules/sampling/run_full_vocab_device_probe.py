"""Bounded silicon diagnostic for the staged unrestricted categorical path.

This is not a qualification test and does not enable a runtime capability.
It reads back only the B selected token IDs and B validity flags; vocabulary
logits, probabilities, CDFs, and random variates remain on device.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import time

import torch
import ttnn

from models.common.sampling.full_vocab_device import sample_unrestricted_top_p_one


IMAGE = "sha256:25353ce73feef741e5ecea3c6b00e5ee3aca5376aa22df7bc4c68944cf26315d"
TT_METAL = "fbfae6720eaeb4da84ba871f085b307c60aafb8c"


def read_replicated(tensor):
    shards = ttnn.get_device_tensors(tensor)
    host = [ttnn.to_torch(shard).reshape(-1) for shard in shards]
    assert host and all(torch.equal(host[0], other) for other in host[1:])
    return host[0]


def release(result, *, protect):
    protected = {id(tensor) for tensor in protect}
    seen = set()
    for tensor in reversed(result.owned_tensors):
        if id(tensor) in seen or id(tensor) in protected:
            continue
        seen.add(id(tensor))
        try:
            if tensor.is_allocated():
                ttnn.deallocate(tensor)
        except RuntimeError:
            # A public no-op/view may share a buffer already released through
            # another tensor object. The bounded diagnostic closes the device
            # after the cases and never reuses these Python objects.
            pass


def upload(mesh, tensor):
    return ttnn.from_torch(
        tensor,
        device=mesh,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def run_once(mesh, logits_host, seeds, *, vocab_size):
    batch = logits_host.shape[2]
    logits = upload(mesh, logits_host)
    inverse_temperature = upload(mesh, torch.ones(1, 1, batch, 1))
    row_scratch = [upload(mesh, torch.zeros(1, 1, 1, 1)) for _ in range(batch)]
    start = time.monotonic()
    result = sample_unrestricted_top_p_one(
        logits,
        inverse_temperature=inverse_temperature,
        row_scratch=row_scratch,
        seed_values=seeds,
        active_rows=[True] * batch,
        vocab_size=vocab_size,
        ops=ttnn,
    )
    ttnn.synchronize_device(mesh)
    elapsed = time.monotonic() - start
    tokens = read_replicated(result.token_ids).long()[:batch]
    valid = read_replicated(result.valid_distribution).bool()[:batch]
    release(result, protect=(logits, inverse_temperature, *row_scratch))
    for tensor in (logits, inverse_temperature, *row_scratch):
        if tensor.is_allocated():
            ttnn.deallocate(tensor)
    return tokens, valid, elapsed


def run(mesh):
    records = []

    # Small deterministic replay and loose distribution sanity. Four equal
    # categories must all be reachable; this is not a statistical qualifier.
    small = torch.full((1, 1, 4, 64), -torch.inf, dtype=torch.float32)
    small[..., :4] = 0.0
    first, valid, elapsed = run_once(mesh, small, [101, 202, 303, 404], vocab_size=64)
    replay, replay_valid, replay_elapsed = run_once(mesh, small, [101, 202, 303, 404], vocab_size=64)
    assert torch.equal(first, replay) and valid.all() and replay_valid.all()
    counts = torch.zeros(4, dtype=torch.int64)
    for iteration in range(16):
        tokens, flags, _ = run_once(
            mesh, small, [1000 + 4 * iteration + lane for lane in range(4)], vocab_size=64
        )
        assert flags.all() and ((0 <= tokens) & (tokens < 4)).all()
        counts += torch.bincount(tokens, minlength=4)
    assert bool((counts > 0).all()), counts.tolist()
    records.append(
        {
            "case": "small_equal4",
            "batch": 4,
            "width": 64,
            "replay_equal": True,
            "counts_64_draws": counts.tolist(),
            "first_seconds": elapsed,
            "replay_seconds": replay_elapsed,
        }
    )

    # GPT-shaped B32 tensor with the real logical vocabulary and an explicit
    # -inf tile tail. Input is synthetic; no model weights are loaded.
    batch, vocab, width = 32, 201088, 262144
    column = torch.arange(width, dtype=torch.float32)
    rows = []
    for lane in range(batch):
        row = torch.sin(column[:vocab] * (0.00031 + lane * 0.000001))
        row = torch.cat((row, torch.full((width - vocab,), -torch.inf)))
        rows.append(row)
    gpt_logits = torch.stack(rows).reshape(1, 1, batch, width)
    seeds = [17001 + lane * 97 for lane in range(batch)]
    tokens, valid, elapsed = run_once(mesh, gpt_logits, seeds, vocab_size=vocab)
    assert valid.all() and ((0 <= tokens) & (tokens < vocab)).all()
    replay, replay_valid, replay_elapsed = run_once(mesh, gpt_logits, seeds, vocab_size=vocab)
    assert replay_valid.all() and torch.equal(tokens, replay)
    records.append(
        {
            "case": "gpt_b32_v201088_padded262144",
            "batch": batch,
            "vocab": vocab,
            "width": width,
            "replay_equal": True,
            "tokens": tokens.tolist(),
            "first_seconds": elapsed,
            "replay_seconds": replay_elapsed,
        }
    )
    return records


if __name__ == "__main__":
    source = Path(inspect.getfile(sample_unrestricted_top_p_one))
    this_file = Path(__file__)
    print(
        json.dumps(
            {
                "stage": "manifest",
                "schema": "ttq.diagnostic.full_vocab_public_ops/v1",
                "qualification_claim": False,
                "image": IMAGE,
                "tt_metal_commit": TT_METAL,
                "source_path": str(source),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "probe_sha256": hashlib.sha256(this_file.read_bytes()).hexdigest(),
                "torch": torch.__version__,
            }
        ),
        flush=True,
    )
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4), l1_small_size=16384, trace_region_size=0
    )
    try:
        records = run(mesh)
        print(
            json.dumps(
                {
                    "stage": "summary",
                    "schema": "ttq.diagnostic.full_vocab_public_ops/v1",
                    "qualification_claim": False,
                    "status": "pass",
                    "records": records,
                }
            ),
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
