# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end causal-prefill A/B for the exact dense-MoE geometry.

Builds one DiffusionGemma model, warms both stock and tuned program configs, then
alternates them over identical token IDs. The measured span is the synchronized
model forward (embedding and host readback excluded). Final logits must be
elementwise identical.

*** DEVICE-OWNERSHIP: run only when QB2 is free. ***
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
    prefill_prompt_tokens,
)
from models.experimental.diffusion_gemma.tt.prefill_moe import (
    FLAG,
    _find_supported_experts,
    tuned_prefill_moe_enabled,
    use_tuned_prefill_moe,
)

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")


def _to_host(tensor):
    if tensor.device().get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).float()
    return ttnn.to_torch(tensor).float()


@contextmanager
def _selector(value: bool | None):
    original = os.environ.get(FLAG)
    if value is None:
        os.environ.pop(FLAG, None)
    else:
        os.environ[FLAG] = "1" if value else "0"
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = original


def _forward(model, tokens, *, tuned):
    prefill_tokens = _pad_prompt_tokens_for_prefill(tokens)
    with _selector(tuned):
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


def _production_prefill(model, tokens, *, tuned):
    with _selector(tuned):
        ttnn.synchronize_device(model.mesh_device)
        start = time.perf_counter()
        result = prefill_prompt_tokens(model, tokens)
        ttnn.synchronize_device(model.mesh_device)
        return time.perf_counter() - start, result


def _tensor_shards(tensor):
    if tensor.device().get_num_devices() > 1:
        return ttnn.get_device_tensors(tensor)
    return [tensor]


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


def _mean(values):
    return sum(values) / len(values)


def run(*, seq_len, non_aligned_len, num_layers, max_seq_len, iters, output_json):
    initial_environment = {
        name: os.environ.get(name, "<unset>")
        for name in (FLAG, "TT_METAL_WATCHER", "TT_METAL_WATCHER_DISABLE_ETH", "TT_METAL_WATCHER_APPEND")
    }
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
    try:
        model_inputs = build_tt_model_from_checkpoint_dir(
            mesh,
            CKPT,
            max_batch_size=1,
            max_seq_len=max(max_seq_len, seq_len),
            num_layers=num_layers,
            create_kv_cache=True,
        )
        model = model_inputs.tt_model
        supported_experts = _find_supported_experts(model)
        if supported_experts is None:
            raise RuntimeError("production model did not satisfy the exact tuned-prefill support guard")
        generator = torch.Generator().manual_seed(0)
        tokens = torch.randint(
            0,
            model.hf_config.vocab_size,
            (1, seq_len),
            dtype=torch.long,
            generator=generator,
        )

        _forward(model, tokens, tuned=False)
        _forward(model, tokens, tuned=True)

        baseline_times = []
        tuned_times = []
        baseline_host = None
        tuned_host = None
        for _ in range(iters):
            baseline_s, baseline_host = _forward(model, tokens, tuned=False)
            baseline_kv = _snapshot_kv(model, seq_len)
            tuned_s, tuned_host = _forward(model, tokens, tuned=True)
            baseline_times.append(baseline_s)
            tuned_times.append(tuned_s)

        baseline_s = _mean(baseline_times)
        tuned_s = _mean(tuned_times)
        logits_exact = torch.equal(baseline_host, tuned_host)
        logits_max_abs = float((baseline_host - tuned_host).abs().max())
        kv_exact, kv_max_abs = _compare_kv(baseline_kv, seq_len)

        non_aligned_tokens = tokens[:, :non_aligned_len]
        _, non_aligned_baseline = _forward(model, non_aligned_tokens, tuned=False)
        _, non_aligned_tuned = _forward(model, non_aligned_tokens, tuned=True)
        non_aligned_exact = torch.equal(non_aligned_baseline, non_aligned_tuned)
        non_aligned_max_abs = float((non_aligned_baseline - non_aligned_tuned).abs().max())

        _production_prefill(model, tokens, tuned=False)
        _production_prefill(model, tokens, tuned=None)
        production_baseline_times = []
        production_default_times = []
        for _ in range(iters):
            baseline_prod_s, baseline_result = _production_prefill(model, tokens, tuned=False)
            with _selector(None):
                default_enabled = tuned_prefill_moe_enabled()
            default_prod_s, default_result = _production_prefill(model, tokens, tuned=None)
            production_baseline_times.append(baseline_prod_s)
            production_default_times.append(default_prod_s)

        production_baseline_s = _mean(production_baseline_times)
        production_default_s = _mean(production_default_times)
        print(
            f"RESULT_PREFILL_E2E seq_len={seq_len} layers={num_layers} "
            f"baseline_s={baseline_s:.6f} tuned_s={tuned_s:.6f} speedup={baseline_s/tuned_s:.3f} "
            f"logits_exact={int(logits_exact)} logits_max_abs={logits_max_abs:.6g} "
            f"kv_exact={int(kv_exact)} kv_max_abs={kv_max_abs:.6g}",
            flush=True,
        )
        print(
            f"RESULT_PREFILL_PRODUCTION seq_len={seq_len} layers={num_layers} "
            f"flag_off_s={production_baseline_s:.6f} unset_default_s={production_default_s:.6f} "
            f"speedup={production_baseline_s/production_default_s:.3f} "
            f"default_enabled={int(default_enabled)}",
            flush=True,
        )
        print(
            f"RESULT_PREFILL_NONALIGNED raw_len={non_aligned_len} padded_len={tokens.shape[1]} "
            f"exact={int(non_aligned_exact)} max_abs={non_aligned_max_abs:.6g}",
            flush=True,
        )
        if not logits_exact or not kv_exact or not non_aligned_exact:
            raise AssertionError(
                "tuned prefill differs from baseline: "
                f"logits_max_abs={logits_max_abs}, kv_max_abs={kv_max_abs}, "
                f"non_aligned_max_abs={non_aligned_max_abs}"
            )
        if baseline_result != default_result or not default_enabled:
            raise AssertionError("production prefill did not reproduce the unset default-on selector")

        result = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "command": shlex.join(sys.argv),
            "checkpoint": CKPT,
            "initial_environment": initial_environment,
            "hardware": {
                "architecture": "Blackhole",
                "system": "P150x4 QB2",
                "mesh_shape": [1, 4],
                "tensor_parallel": 4,
                "compute_grid": [11, 10],
            },
            "model": {
                "name": "DiffusionGemma 26B-A4B-it",
                "layers": num_layers,
                "expert_layers_guarded": len(supported_experts),
                "expert_weight_dtype": "bfloat16",
            },
            "model_forward": {
                "scope": "synchronized model forward; host embedding and logits readback excluded",
                "raw_prompt_tokens": seq_len,
                "padded_prompt_tokens": seq_len,
                "iterations": iters,
                "baseline_times_s": baseline_times,
                "tuned_times_s": tuned_times,
                "baseline_mean_s": baseline_s,
                "tuned_mean_s": tuned_s,
                "speedup": baseline_s / tuned_s,
                "logits_exact": logits_exact,
                "logits_max_abs": logits_max_abs,
                "kv_cache_exact_all_layers_all_devices": kv_exact,
                "kv_cache_max_abs": kv_max_abs,
            },
            "production_prefill": {
                "scope": "prefill_prompt_tokens; host embedding included, host logits readback absent",
                "flag_off_times_s": production_baseline_times,
                "unset_default_times_s": production_default_times,
                "flag_off_mean_s": production_baseline_s,
                "unset_default_mean_s": production_default_s,
                "speedup": production_baseline_s / production_default_s,
                "unset_default_enabled": default_enabled,
                "flag_off_result": list(baseline_result),
                "unset_default_result": list(default_result),
            },
            "non_aligned_correctness": {
                "raw_prompt_tokens": non_aligned_len,
                "padded_prompt_tokens": seq_len,
                "logits_exact": non_aligned_exact,
                "logits_max_abs": non_aligned_max_abs,
            },
        }
        if output_json:
            path = Path(output_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--non-aligned-len", type=int, default=1001)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if not 0 < args.non_aligned_len <= args.seq_len:
        parser.error("--non-aligned-len must be in [1, --seq-len]")
    run(
        seq_len=args.seq_len,
        non_aligned_len=args.non_aligned_len,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        iters=args.iters,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
