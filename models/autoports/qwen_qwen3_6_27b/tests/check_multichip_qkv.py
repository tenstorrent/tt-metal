# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact local Q6/KV1 staging, padding, ownership and cached-program checks."""

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
from models.autoports.qwen_qwen3_6_27b.tests.check_multichip_compaction import compare_users, tensor_hash, upload_shards
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder, _local_qkv_split


def run_batch(mesh, batch, report, save):
    model = MultichipDecoder.__new__(MultichipDecoder)
    model.mesh = mesh
    memory = model._compact_height_mem(model._compact_cores(batch), 32, 256)
    owners = [[user % 8, user // 8] for user in range(batch)]
    generator = torch.Generator().manual_seed(43600 + batch)
    # Use BF16 bit patterns: distinct 256-channel head signatures and user/rank
    # signatures, spanning both signs and finite magnitudes below one. This also keeps
    # failure diagnostics finite when a corrupt copy swaps signs or channels.
    index = torch.arange(batch * 2048).reshape(1, 1, batch, 2048)
    original = []
    for rank in range(4):
        bits = (index * 109 + rank * 2003 + 129) % 0x3F00 + 0x0080
        signs = ((index // 127 + rank) % 2) * 0x8000
        original.append((bits + signs).to(torch.int16).view(torch.bfloat16))
    permutation = torch.randperm(batch, generator=generator)
    if batch > 1 and torch.equal(permutation, torch.arange(batch)):
        permutation = torch.arange(batch - 1, -1, -1)
    changed = [-value.flip(-1) for value in original]
    variants = [
        ("original", original),
        ("permuted", [value[:, :, permutation] for value in original]),
        ("changed", changed),
        ("original_again", original),
    ]
    persistent = upload_shards(mesh, original)
    persistent_address = persistent.buffer_address()
    item = {"batch": batch, "owners": owners, "permutation": permutation.tolist(), "cases": []}
    report["batches"].append(item)
    # Retain outputs so cached-program validation also exercises fresh output
    # addresses instead of depending on allocator reuse at the same address.
    retained_outputs = []
    for case_index, (name, values) in enumerate(variants):
        print(f"QKV batch={batch} case={name} begin", flush=True)
        source = upload_shards(mesh, values)
        ttnn.copy(source, persistent)
        assert persistent.buffer_address() == persistent_address
        before = mesh.num_program_cache_entries()
        outputs = _local_qkv_split(persistent, memory_config=memory)
        ttnn.synchronize_device(mesh)
        after = mesh.num_program_cache_entries()
        assert after > 0, "program cache must be enabled to validate cached runtime arguments"
        retained_outputs.append(outputs)
        case = {
            "name": name,
            "input_sha256": [tensor_hash(value) for value in values],
            "input_address_reused": True,
            "program_cache_before": before,
            "program_cache_after": after,
            "program_cache_reused": case_index > 0 and before == after,
            "checks": [],
            "outputs": {},
        }
        item["cases"].append(case)
        if case_index:
            assert before == after, f"QKV cache miss at B={batch} {name}: {before}->{after}"
        for output_name, output, heads, start in zip(("q", "k", "v"), outputs, (6, 1, 1), (0, 1536, 1792)):
            assert output.dtype == ttnn.bfloat16 and output.layout == ttnn.TILE_LAYOUT
            assert tuple(output.shape) == (1, batch, heads, 256)
            assert tuple(output.padded_shape) == (1, batch, 32, 256)
            assert output.memory_config() == memory
            expected = [value[..., start : start + heads * 256].reshape(batch, heads, 256) for value in values]
            actual, padded = [], []
            for shard in ttnn.get_device_tensors(output):
                host = shard.cpu()
                actual.append(host.to_torch().reshape(batch, heads, 256))
                # This checkout exposes a documented host-only padded conversion
                # (pytensor.cpp). No logical-size expansion or device view tricks.
                padded.append(host.to_torch_with_padded_shape().reshape(batch, 32, 256))
            compare_users(actual, expected, label=f"{output_name}_logical", checks=case["checks"])
            compare_users(
                [value[:, :heads] for value in padded],
                expected,
                label=f"{output_name}_padded_live_rows",
                checks=case["checks"],
            )
            padding = [value[:, heads:] for value in padded]
            compare_users(
                padding,
                [torch.zeros_like(value) for value in padding],
                label=f"{output_name}_zero_padding",
                checks=case["checks"],
            )
            case["outputs"][output_name] = {
                "logical_shape": list(output.shape),
                "padded_shape": list(output.padded_shape),
                "shard_shape": [32, 256],
                "owner_grid_exact": True,
                "logical_sha256": [tensor_hash(value) for value in actual],
                "padded_sha256": [tensor_hash(value) for value in padded],
            }
        case["pass"] = True
        save()
        print(f"QKV batch={batch} case={name} pass", flush=True)
    item["pass"] = True
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 3, 9, 17, 31, 32])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert all(1 <= batch <= 32 for batch in args.batches)
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    files = [
        Path(__file__).resolve(),
        root / "tests/check_multichip_compaction.py",
        root / "tt/multichip_decoder.py",
        root / "tt/optimized_decoder.py",
    ]
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(
            [sys.executable, "-m", "models.autoports.qwen_qwen3_6_27b.tests.check_multichip_qkv", *sys.argv[1:]]
        ),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "mesh": [1, 4],
        "strict_exact": True,
        "padding_read_method": "device shard.cpu().to_torch_with_padded_shape(); supported host API, no altered logical shapes",
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
        mesh.enable_program_cache()
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
    print(json.dumps({"status": report["status"], "artifact": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
