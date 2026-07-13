# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bit-identity A/B for the chunked ragged prefill (>4096) extension.

Extends the ragged-vs-dense proof past the single-call ceiling: for each long prompt length it runs
the shared DENSE 128-expert prefill and the CHUNKED ragged prefill over identical token IDs, and
requires the final logits AND the whole KV cache to be elementwise identical (max_abs == 0). This is
the device half of the ``DG_PREFILL_RAGGED_LONG`` extension; the host half (dispatch/gating and the
chunk-loop plumbing) is covered device-free in ``tests/test_prefill_moe.py``.

Baseline  = shared dense prefill  (DG_PREFILL_MOE_RAGGED=0).
Candidate = chunked ragged prefill (DG_PREFILL_MOE_RAGGED=1, DG_PREFILL_RAGGED_LONG=1).

Lengths default to 4096 (single-chunk fast path), 6144 (chunk + 2048 tail) and 8192 (two even
chunks) so the seam and the tail are both exercised. ``DG_PREFILL_MOE_TUNED`` is left at its default
for both sides (the tuned geometry is orthogonal and separately verified bit-identical).

*** DEVICE-OWNERSHIP: run only when QB2 is free. Opens its own (1,4) mesh, so it cannot share the
device with a running vLLM server. ***

Example:
    source /home/zni/venvs/tt-diffusion-gemma/bin/activate
    export TT_METAL_HOME=/home/zni/tt-metal PYTHONPATH=/home/zni/tt-metal
    DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
      python models/experimental/diffusion_gemma/doc/optimize_perf/verify_chunked_ragged_prefill.py \
      --seq-lens 4096,6144,8192 \
      --output-json models/experimental/diffusion_gemma/doc/optimize_perf/chunked_ragged_prefill_bitident.json
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import torch

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt.generate import (
    _pad_prompt_tokens_for_prefill,
    embed_host_tokens,
)
from models.experimental.diffusion_gemma.tt.prefill_moe import (
    RAGGED_FLAG,
    RAGGED_LONG_FLAG,
    _find_supported_experts,
    ragged_long_prefill_enabled,
    ragged_prefill_moe_enabled,
    use_tuned_prefill_moe,
)
from models.experimental.diffusion_gemma.tt.sparse_moe import ragged_prefill_chunk_size

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")


def _to_host(tensor):
    if tensor.device().get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()
    return ttnn.to_torch(tensor).float()


@contextmanager
def _env(**overrides):
    """Set/restore env vars; a value of None deletes the variable for the duration."""
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _selector(mode):
    """``dense`` = shared 128-expert prefill; ``chunked`` = chunked ragged prefill."""
    if mode == "dense":
        return _env(**{RAGGED_FLAG: "0", RAGGED_LONG_FLAG: "0"})
    if mode == "chunked":
        return _env(**{RAGGED_FLAG: "1", RAGGED_LONG_FLAG: "1"})
    raise ValueError(f"unknown selector mode: {mode}")


def _tensor_shards(tensor):
    if tensor.device().get_num_devices() > 1:
        return ttnn.get_device_tensors(tensor)
    return [tensor]


def _forward(model, tokens, *, mode):
    prefill_tokens = _pad_prompt_tokens_for_prefill(tokens)
    with _selector(mode):
        embeds = embed_host_tokens(model, prefill_tokens)
        ttnn.synchronize_device(model.mesh_device)
        start = time.perf_counter()
        with use_tuned_prefill_moe(model):
            logits = model(
                embeds,
                is_decode=False,
                input_ids_torch=prefill_tokens,
                get_last_token=((tokens.shape[1] - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE,
            )
        ttnn.synchronize_device(model.mesh_device)
        elapsed = time.perf_counter() - start
    host = _to_host(logits)
    logits.deallocate(True)
    return elapsed, host


def _snapshot_kv(model, seq_len):
    snapshots = []
    seen = set()
    for cache in model.tt_kv_cache:
        for tensor in cache:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            host_shards = [ttnn.to_torch(shard)[..., :seq_len, :].clone() for shard in _tensor_shards(tensor)]
            snapshots.append((tensor, host_shards))
    return snapshots


def _compare_kv(snapshots, seq_len):
    exact = True
    max_abs = 0.0
    for tensor, reference_shards in snapshots:
        candidate_shards = _tensor_shards(tensor)
        if len(candidate_shards) != len(reference_shards):
            return False, float("inf")
        for candidate, reference in zip(candidate_shards, reference_shards):
            candidate_host = ttnn.to_torch(candidate)[..., :seq_len, :]
            exact = exact and torch.equal(reference, candidate_host)
            max_abs = max(max_abs, float((reference.float() - candidate_host.float()).abs().max()))
    return exact, max_abs


def _run_length(model, seq_len):
    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, model.hf_config.vocab_size, (1, seq_len), dtype=torch.long, generator=generator)

    dense_s, dense_logits = _forward(model, tokens, mode="dense")
    dense_kv = _snapshot_kv(model, seq_len)
    chunked_s, chunked_logits = _forward(model, tokens, mode="chunked")

    logits_exact = torch.equal(dense_logits, chunked_logits)
    logits_max_abs = float((dense_logits - chunked_logits).abs().max())
    kv_exact, kv_max_abs = _compare_kv(dense_kv, seq_len)

    chunk = ragged_prefill_chunk_size()
    num_chunks = (seq_len + chunk - 1) // chunk
    print(
        f"RESULT_CHUNKED_RAGGED seq_len={seq_len} chunk={chunk} num_chunks={num_chunks} "
        f"dense_s={dense_s:.4f} chunked_s={chunked_s:.4f} speedup={dense_s / chunked_s:.2f} "
        f"logits_exact={int(logits_exact)} logits_max_abs={logits_max_abs:.6g} "
        f"kv_exact={int(kv_exact)} kv_max_abs={kv_max_abs:.6g}",
        flush=True,
    )
    return {
        "seq_len": seq_len,
        "chunk_size": chunk,
        "num_chunks": num_chunks,
        "dense_prefill_s": dense_s,
        "chunked_prefill_s": chunked_s,
        "speedup": dense_s / chunked_s,
        "logits_exact": logits_exact,
        "logits_max_abs": logits_max_abs,
        "kv_cache_exact_all_layers_all_devices": kv_exact,
        "kv_cache_max_abs": kv_max_abs,
    }


def run(*, seq_lens, num_layers, max_seq_len, output_json):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    rows = []
    try:
        model_inputs = build_tt_model_from_checkpoint_dir(
            mesh,
            CKPT,
            max_batch_size=1,
            max_seq_len=max(max_seq_len, max(seq_lens)),
            num_layers=num_layers,
            create_kv_cache=True,
        )
        model = model_inputs.tt_model
        if _find_supported_experts(model) is None:
            raise RuntimeError("model did not satisfy the ragged-prefill support guard")
        # These flags default ON/OFF respectively; assert the run actually exercises the toggle.
        assert ragged_prefill_moe_enabled(), "ragged prefill must be enabled for the candidate"
        with _selector("chunked"):
            assert ragged_long_prefill_enabled(), "chunked candidate must have long prefill enabled"

        for seq_len in seq_lens:
            rows.append(_run_length(model, seq_len))

        failures = [r for r in rows if not (r["logits_exact"] and r["kv_cache_exact_all_layers_all_devices"])]
        result = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "command": shlex.join(sys.argv),
            "checkpoint": CKPT,
            "hardware": {
                "architecture": "Blackhole",
                "system": "P150x4 QB2",
                "mesh_shape": [1, 4],
                "tensor_parallel": 4,
            },
            "model": {"name": "DiffusionGemma 26B-A4B-it", "layers": num_layers},
            "baseline": "shared dense 128-expert prefill (DG_PREFILL_MOE_RAGGED=0)",
            "candidate": "chunked ragged prefill (DG_PREFILL_MOE_RAGGED=1, DG_PREFILL_RAGGED_LONG=1)",
            "rows": rows,
            "all_bit_identical": not failures,
        }
        if output_json:
            path = Path(output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
        if failures:
            raise AssertionError(
                "chunked ragged prefill differs from dense: "
                + ", ".join(
                    f"seq_len={r['seq_len']} logits_max_abs={r['logits_max_abs']} kv_max_abs={r['kv_cache_max_abs']}"
                    for r in failures
                )
            )
        print(f"RESULT_CHUNKED_RAGGED_ALL_BIT_IDENTICAL seq_lens={seq_lens}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _parse_lens(value):
    lens = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lens or any(v <= 0 or v % ttnn.TILE_SIZE != 0 for v in lens):
        raise argparse.ArgumentTypeError(f"seq lens must be positive multiples of {ttnn.TILE_SIZE}")
    return lens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-lens", type=_parse_lens, default=[4096, 6144, 8192])
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    run(
        seq_lens=args.seq_lens,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
