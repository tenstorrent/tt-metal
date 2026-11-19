# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Address-sensitive local TP4 convolution/recurrence checks; no model weights needed.

Run as a module with --output PATH. The default covers the batch/core/padding
boundaries of compact decode. Device execution belongs to the stage's serialized
hardware runner; this module performs no device reset.
"""

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder, _local12_recurrent

CHANNELS = 2560
HEADS = 12
HEAD_DIM = 128


def tensor_hash(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def read_shards(value):
    return [ttnn.to_torch(shard).contiguous() for shard in ttnn.get_device_tensors(value)]


def upload_shards(mesh, values, *, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    assert len(values) == 4
    return ttnn.from_torch(
        torch.cat(values, dim=-1).contiguous(),
        device=mesh,
        layout=layout,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )


def compare_users(actual, expected, *, label, checks, exact=True):
    """Persist every user/rank result before raising; aggregate PCC cannot hide swaps."""
    assert len(actual) == len(expected) == 4
    result = {"label": label, "require_exact": exact, "per_rank_user": []}
    checks.append(result)
    passed = True
    for rank, (got, ref) in enumerate(zip(actual, expected)):
        assert got.shape == ref.shape, (label, rank, got.shape, ref.shape)
        rank_results = []
        result["per_rank_user"].append(rank_results)
        for user in range(got.shape[0]):
            left, right = got[user].float().flatten(), ref[user].float().flatten()
            finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
            equal = finite and torch.equal(left, right)
            if equal:
                correlation = 1.0
            elif finite and left.std() > 0 and right.std() > 0:
                correlation = float(torch.corrcoef(torch.stack((left.double(), right.double())))[0, 1])
            else:
                correlation = None
            error = float((left - right).abs().max()) if finite else None
            rank_results.append({"user": user, "exact": equal, "pcc": correlation, "max_abs": error})
            passed = passed and finite and (equal if exact else correlation is not None and correlation >= 0.99999)
    result["pass"] = passed
    assert passed, f"{label}: per-user check failed; inspect JSON artifact"


def run_batch(mesh, batch, report, save):
    # A minimal object invokes the production method, not a duplicate convolution.
    model = MultichipDecoder.__new__(MultichipDecoder)
    model.mesh = mesh
    generators = [torch.Generator().manual_seed(17001 + 101 * batch + rank) for rank in range(4)]
    users = torch.arange(batch).reshape(batch, 1, 1).float()
    slots = torch.arange(3).reshape(1, 3, 1).float()
    channels = (torch.arange(CHANNELS) % 257).reshape(1, 1, CHANNELS).float() / 2048
    history_host = [
        (0.03125 * users + 0.125 * slots + channels + 0.0625 * rank).to(torch.bfloat16) for rank in range(4)
    ]
    taps = [
        [
            (torch.randn(1, 1, CHANNELS, generator=generators[rank]) * 0.0625 + (tap + 1) * 0.03125).to(torch.bfloat16)
            for rank in range(4)
        ]
        for tap in range(4)
    ]
    model.taps_rm = [upload_shards(mesh, values, layout=ttnn.ROW_MAJOR_LAYOUT) for values in taps]
    permutation = torch.arange(batch - 1, -1, -1)
    inverse = torch.argsort(permutation)
    histories = {
        "compact": upload_shards(mesh, history_host, layout=ttnn.ROW_MAJOR_LAYOUT),
        "expanded": upload_shards(mesh, history_host, layout=ttnn.ROW_MAJOR_LAYOUT),
        "permuted": upload_shards(mesh, [value[permutation] for value in history_host], layout=ttnn.ROW_MAJOR_LAYOUT),
    }
    state_host = [
        (torch.randn(batch, HEADS, HEAD_DIM, HEAD_DIM, generator=g) * 0.015625).to(torch.bfloat16) for g in generators
    ]
    states = {
        "packed": upload_shards(mesh, [value.reshape(batch * HEADS, HEAD_DIM, HEAD_DIM) for value in state_host]),
        "separate": upload_shards(mesh, [value.reshape(batch * HEADS, HEAD_DIM, HEAD_DIM) for value in state_host]),
        "permuted": upload_shards(
            mesh, [value[permutation].reshape(batch * HEADS, HEAD_DIM, HEAD_DIM) for value in state_host]
        ),
    }
    item = {
        "batch": batch,
        "permutation": permutation.tolist(),
        "initial_history_sha256": [tensor_hash(value) for value in history_host],
        "initial_state_sha256": [tensor_hash(value) for value in state_host],
        "tap_sha256": [[tensor_hash(value) for value in tap] for tap in taps],
        "steps": [],
    }
    report["batches"].append(item)
    save()
    for step in range(4):
        evidence = {"step": step, "checks": []}
        item["steps"].append(evidence)
        checks = evidence["checks"]
        print(f"COMPACTION batch={batch} step={step} begin", flush=True)
        token_host = [
            (
                torch.randn(batch, 1, CHANNELS, generator=g) * 0.125
                + 0.03125 * users
                + channels
                + step * 0.0625
                + rank * 0.03125
            ).to(torch.bfloat16)
            for rank, g in enumerate(generators)
        ]
        evidence["token_sha256"] = [tensor_hash(value) for value in token_host]
        compact_input = upload_shards(mesh, [value.reshape(1, batch, CHANNELS) for value in token_host])
        expanded_input = upload_shards(mesh, token_host)
        permuted_input = upload_shards(mesh, [value[permutation].reshape(1, batch, CHANNELS) for value in token_host])
        conv = model._local_batched_conv(compact_input, histories["compact"])
        conv_expanded = model._local_batched_conv(expanded_input, histories["expanded"])
        conv_permuted = model._local_batched_conv(permuted_input, histories["permuted"])
        assert conv.layout == ttnn.ROW_MAJOR_LAYOUT
        assert tuple(conv.shape) == tuple(conv.padded_shape) == (batch, 1, CHANNELS)
        history_host = [torch.cat((old[:, 1:3], token), dim=1) for old, token in zip(history_host, token_host)]
        for name, history in histories.items():
            actual = read_shards(history)
            if name == "permuted":
                actual = [value[inverse] for value in actual]
            compare_users(actual, history_host, label=f"history_shift_{name}", checks=checks)
        conv_host = read_shards(conv)
        compare_users(read_shards(conv_expanded), conv_host, label="compact_vs_expanded_convolution", checks=checks)
        compare_users(
            [value[inverse] for value in read_shards(conv_permuted)],
            conv_host,
            label="permuted_convolution",
            checks=checks,
        )

        # Build gates in compact TILE [1,B,12], then use the production reshape.
        # The reader must see B distinct padded tiles, one per user after reshape.
        heads = torch.arange(HEADS).reshape(1, 1, HEADS).float()
        gates_host = [(-0.02 - users * 0.001 - heads * 0.002 - rank * 0.003 - step * 0.004) for rank in range(4)]
        beta_host = [
            (0.125 + users * 0.005 + heads * 0.015 + rank * 0.007 + step * 0.01).to(torch.bfloat16) for rank in range(4)
        ]
        evidence["gate_sha256"] = [tensor_hash(value) for value in gates_host]
        evidence["beta_sha256"] = [tensor_hash(value) for value in beta_host]

        def gates(values, dtype, permute=False):
            ordered = [value[permutation] if permute else value for value in values]
            uploaded = upload_shards(mesh, [value.reshape(1, batch, HEADS) for value in ordered], dtype=dtype)
            result = ttnn.reshape(uploaded, (batch, 1, HEADS))
            assert tuple(result.padded_shape) == (batch, 32, 32), tuple(result.padded_shape)
            compare_users(read_shards(result), ordered, label=f"gate_reshape_{dtype}_{permute}", checks=checks)
            return result

        g = gates(gates_host, ttnn.float32)
        beta = gates(beta_host, ttnn.bfloat16)
        g_permuted = gates(gates_host, ttnn.float32, True)
        beta_permuted = gates(beta_host, ttnn.bfloat16, True)
        packed_output = _local12_recurrent(conv, conv, conv, g, beta, states["packed"], packed_qkv=True)
        # Independent row-major uploads avoid letting the same device slice bug
        # contaminate both sides of the packed-vs-separate address comparison.
        q = upload_shards(mesh, [value[:, :, :512] for value in conv_host], layout=ttnn.ROW_MAJOR_LAYOUT)
        k = upload_shards(mesh, [value[:, :, 512:1024] for value in conv_host], layout=ttnn.ROW_MAJOR_LAYOUT)
        v = upload_shards(mesh, [value[:, :, 1024:] for value in conv_host], layout=ttnn.ROW_MAJOR_LAYOUT)
        separate_output = _local12_recurrent(q, k, v, g, beta, states["separate"])
        permuted_output = _local12_recurrent(
            conv_permuted, conv_permuted, conv_permuted, g_permuted, beta_permuted, states["permuted"], packed_qkv=True
        )
        packed_host = [value.reshape(batch, HEADS, HEAD_DIM) for value in read_shards(packed_output)]
        compare_users(
            [value.reshape(batch, HEADS, HEAD_DIM) for value in read_shards(separate_output)],
            packed_host,
            label="packed_vs_separate_recurrence_output",
            checks=checks,
        )
        compare_users(
            [value.reshape(batch, HEADS, HEAD_DIM)[inverse] for value in read_shards(permuted_output)],
            packed_host,
            label="permuted_recurrence_output",
            checks=checks,
        )
        packed_state = [value.reshape(batch, HEADS, HEAD_DIM, HEAD_DIM) for value in read_shards(states["packed"])]
        for name in ("separate", "permuted"):
            actual = [value.reshape(batch, HEADS, HEAD_DIM, HEAD_DIM) for value in read_shards(states[name])]
            if name == "permuted":
                actual = [value[inverse] for value in actual]
            compare_users(actual, packed_state, label=f"recurrent_state_{name}", checks=checks)
        evidence["output_sha256"] = [tensor_hash(value) for value in packed_host]
        evidence["state_sha256"] = [tensor_hash(value) for value in packed_state]
        evidence["pass"] = True
        save()
        print(f"COMPACTION batch={batch} step={step} pass ({len(checks)} checks)", flush=True)
    ttnn.synchronize_device(mesh)
    item["pass"] = True
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[2, 3, 9, 10, 17, 31, 32])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert all(1 <= batch <= 32 for batch in args.batches)
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__).resolve(), root / "tt/multichip_decoder.py", root / "tt/optimized_decoder.py"]
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(
            [sys.executable, "-m", "models.autoports.qwen_qwen3_6_27b.tests.check_multichip_compaction", *sys.argv[1:]]
        ),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "mesh": [1, 4],
        "fabric": "FABRIC_1D",
        "inputs": "deterministic synthetic; rank/user/channel/history/step differentiated",
        "steps_per_batch": 4,
        "strict_exact": True,
        "environment": {
            key: os.environ.get(key)
            for key in ("TT_METAL_WATCHER", "TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_DEVICE_PROFILER")
        },
        "batches": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))
        grid = mesh.compute_with_storage_grid_size()
        report["worker_grid"] = [grid.x, grid.y]
        for batch in args.batches:
            run_batch(mesh, batch, report, save)
        report["status"] = "pass"
    except BaseException:
        report["status"] = "fail"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
    print(json.dumps({"status": report["status"], "batches": args.batches, "artifact": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
