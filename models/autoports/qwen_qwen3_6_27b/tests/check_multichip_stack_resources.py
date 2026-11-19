# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""64 independent decoder resource owners, shared weights, serialized warmed trace.

This is a resource/isolation test, not full-model inference. Every layer consumes
one shared prepared residual; numerical layer-to-layer handoff has its own test.
Run B1 and B32 in separate processes. Persistent storage and all program variants
are prepared before the main capture; there are no allocation acknowledgments.
"""

import argparse
import copy
import hashlib
import json
import os
import shlex
import subprocess
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import REVISION, load_layer
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import MultichipDecoder, _BlackholeCCLManager


def digest(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def state_tensors(layer, state):
    names = ("conv", "recurrent") if layer.kind == "linear_attention" else ("key", "value")
    return {name: getattr(state, name) for name in names}


def memory_sample(mesh, label):
    entry = {"label": label, "program_cache_entries": mesh.num_program_cache_entries(), "memory": {}}
    for name in ("L1", "L1_SMALL", "DRAM", "TRACE"):
        view = ttnn.get_memory_view(mesh, getattr(ttnn.BufferType, name))
        fields = (
            "num_banks",
            "total_bytes_per_bank",
            "total_bytes_allocated_per_bank",
            "total_bytes_free_per_bank",
            "largest_contiguous_bytes_free_per_bank",
        )
        item = {field: int(getattr(view, field)) for field in fields}
        item["allocated_bytes_per_device"] = item["num_banks"] * item["total_bytes_allocated_per_bank"]
        entry["memory"][name] = item
    return entry


def tracker_positive_control(mesh):
    """Separate tiny trace, fully torn down before any decoder resources exist."""
    source = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    target = ttnn.clone(source)
    ttnn.copy(source, target)
    ttnn.synchronize_device(mesh)
    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
    ttnn.copy(source, target)
    ttnn.end_trace_capture(mesh, trace, cq_id=0)
    intruder = ttnn.empty(
        (1, 1, 32, 32),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    try:
        try:
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        except RuntimeError as error:
            assert "device buffer(s) still alive before trace replay" in str(error), str(error)
            result = {"pass": True, "expected_rejection": str(error).splitlines()[0]}
        else:
            raise AssertionError("allocation tracker did not reject a deliberately live post-capture allocation")
    finally:
        ttnn.deallocate(intruder)
        ttnn.release_trace(mesh, trace)
        ttnn.deallocate(source)
        ttnn.deallocate(target)
    return result


def seed_state(layer, state, index):
    """Different initial state per owner catches cross-layer state substitutions."""
    for name, tensor in state_tensors(layer, state).items():
        shape = tuple(tensor.shape)
        host = torch.zeros(shape, dtype=torch.float32)
        level = (index + 1) / 8192
        if name == "recurrent":
            diagonal = torch.arange(128)
            host[..., diagonal, diagonal] = level
            host[:, :, 0, 1] = torch.arange(shape[0]).reshape(-1, 1) / 4096
        else:
            host += level
            host += (torch.arange(shape[-1]) % 17) / 8192
        initial = layer._upload(host, tensor.dtype, tensor.layout)
        ttnn.copy(initial, tensor)
        ttnn.deallocate(initial)


def snapshot(layers, states, outputs):
    """Full logical output/state hashes, all ranks; host-only diagnostic samples.

    Download one tensor at a time to bound host memory. There are no device
    slices, new program signatures or persistent device allocations here.
    """
    records = []
    for index, (layer, state, output) in enumerate(zip(layers, states, outputs)):
        item = {"layer": index, "kind": layer.kind, "tensors": {}}
        for name, tensor in {"output": output, **state_tensors(layer, state)}.items():
            shards = []
            for rank, shard in enumerate(ttnn.get_device_tensors(tensor)):
                value = ttnn.to_torch(shard).contiguous()
                assert torch.isfinite(value).all(), (index, name, rank, "nonfinite")
                flat = value.reshape(-1)
                sample_indices = sorted(set([0, 1, flat.numel() // 2, flat.numel() - 2, flat.numel() - 1]))
                shards.append(
                    {
                        "rank": rank,
                        "logical_shape": list(value.shape),
                        "sha256": digest(value),
                        "sample_indices": sample_indices,
                        "samples": flat[sample_indices].float().tolist(),
                    }
                )
            item["tensors"][name] = shards
        records.append(item)
    return records


def require_exact(actual, expected, checks):
    assert len(actual) == len(expected) == 64
    for got, ref in zip(actual, expected):
        assert got["layer"] == ref["layer"] and got["kind"] == ref["kind"]
        for name in ref["tensors"]:
            for shard, wanted in zip(got["tensors"][name], ref["tensors"][name]):
                equal = shard["logical_shape"] == wanted["logical_shape"] and shard["sha256"] == wanted["sha256"]
                check = {"layer": got["layer"], "tensor": name, "rank": shard["rank"], "exact": equal}
                checks.append(check)
                assert equal, check


def run(args, mesh, report, save):
    root = Path(__file__).resolve().parents[1]
    cfg = Qwen3_5TextConfig(**json.loads((root / "doc/hf_config.json").read_text())["text_config"])
    assert len(cfg.layer_types) == 64
    assert Counter(cfg.layer_types) == {"linear_attention": 48, "full_attention": 16}
    assert args.prefill_length <= cfg.max_position_embeddings
    policy = json.loads(args.policy_file.read_text()) if args.policy_file else None
    per_kind_policy = policy is not None and set(policy) == {"linear_attention", "full_attention"}
    if per_kind_policy:
        assert all(isinstance(value, dict) for value in policy.values())
    prototypes = {}
    for index in (0, 3):
        layer = MultichipDecoder.from_state_dict(
            load_layer(index),
            hf_config=cfg,
            layer_idx=index,
            mesh_device=mesh,
            policy=policy[cfg.layer_types[index]] if per_kind_policy else policy,
        )
        prototypes[layer.kind] = layer
    report["memory_samples"].append(memory_sample(mesh, "two_representative_layers"))
    layers, used, owners = [], set(), []
    exchange_core = ttnn._ttnn.operations.experimental.kda.qkv_causal_conv1d_silu_exchange_core(
        mesh.compute_with_storage_grid_size()
    )
    exchange_grid = ttnn.CoreRangeSet({ttnn.CoreRange(exchange_core, exchange_core)})
    for index, kind in enumerate(cfg.layer_types):
        original = prototypes[kind]
        if kind not in used:
            layer = original
            used.add(kind)
        else:
            layer = copy.copy(original)
            layer.layer_idx = index
            from ..tt.fused_decode_norm import install as install_decode_norm

            install_decode_norm(layer, layer_idx=index)
            layer._ccl_buffers = {}
            layer._prefill_batch_indices = {}
            layer.ccl = _BlackholeCCLManager(mesh, original.ccl.num_links, original.ccl.topology)
            if kind == "linear_attention":
                layer.conv_semaphores = [ttnn.create_global_semaphore(mesh, exchange_grid, 0) for _ in range(3)]
                layer.exchange = layer._upload(torch.zeros(1, 12, 2560), layout=ttnn.ROW_MAJOR_LAYOUT)
        assert layer.weights is original.weights and layer.row_parallel_weights is original.row_parallel_weights
        semaphores = layer.ccl.rs_ping_pong_semaphores + layer.ccl.ag_ping_pong_semaphores + layer.ccl.barrier_semaphore
        record = {
            "layer": index,
            "kind": kind,
            "ccl_semaphores": [int(ttnn.get_global_semaphore_address(s)) for s in semaphores],
        }
        if kind == "linear_attention":
            record["conv_semaphores"] = [int(ttnn.get_global_semaphore_address(s)) for s in layer.conv_semaphores]
            record["exchange_address"] = int(layer.exchange.buffer_address())
        record["fused_decode_norm"] = layer._fused_decode_norm.ownership() if layer._fused_decode_norm else None
        owners.append(record)
        layers.append(layer)
    all_sems = [
        address for owner in owners for key in ("ccl_semaphores", "conv_semaphores") for address in owner.get(key, [])
    ]
    assert len(all_sems) == len(set(all_sems)) == 912
    norm_sems = [a for o in owners if o["fused_decode_norm"] for a in o["fused_decode_norm"]["semaphores"]]
    assert len(all_sems + norm_sems) == len(set(all_sems + norm_sems))
    report["resource_counts"]["fused_decode_norm_owners"] = sum(o["fused_decode_norm"] is not None for o in owners)
    report["resource_counts"]["fused_decode_norm_semaphores"] = len(norm_sems)
    assert len({o["exchange_address"] for o in owners if "exchange_address" in o}) == 48
    report["owners"] = owners
    report["selected_policies"] = {kind: dict(layer.policy) for kind, layer in prototypes.items()}
    report["memory_samples"].append(memory_sample(mesh, "all64_communication_owners"))
    save()

    generator = torch.Generator().manual_seed(31861 + args.batch)
    page_table = torch.randperm(args.batch * 2, generator=generator).reshape(args.batch, 2)
    prefill_pages = (args.prefill_length + 31) // 32
    prefill_table = torch.arange(args.prefill_batch * prefill_pages, dtype=torch.int32).roll(1)
    prefill_table = prefill_table.reshape(args.prefill_batch, prefill_pages)
    states, prefill_states, saved, outputs = [], [], [], []
    x = torch.randn(args.batch, 1, 5120, generator=generator, dtype=torch.bfloat16) * 0.1
    alternate = torch.randn(args.batch, 1, 5120, generator=generator, dtype=torch.bfloat16) * 0.1
    prefill_host = (
        torch.randn(args.prefill_batch, args.prefill_length, 5120, generator=generator, dtype=torch.bfloat16) * 0.1
    )
    positions = (torch.arange(args.batch, dtype=torch.int32) % 7 + 1).reshape(args.batch, 1)
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    upload = layers[0]._upload
    stable_x = layers[0].upload_decode_residual(x)
    stable_pos = upload(positions.flatten(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    stable_rope = tuple(upload(t.unsqueeze(1)) for t in rotary(x, positions.long()))
    variants = []
    for name, host, pos in (("original", x, positions), ("changed", alternate, positions + 1)):
        variants.append(
            {
                "name": name,
                "x": layers[0].upload_decode_residual(host),
                "position": upload(pos.flatten(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
                "rope": tuple(upload(t.unsqueeze(1)) for t in rotary(host, pos.long())),
            }
        )
    prefill_x = layers[0].upload_residual(prefill_host)
    prefill_positions = torch.arange(args.prefill_length).reshape(1, -1).expand(args.prefill_batch, -1)
    prefill_rope = tuple(upload(t.unsqueeze(1)) for t in rotary(prefill_host, prefill_positions))
    report["input_sha256"] = {"original": digest(x), "changed": digest(alternate), "prefill": digest(prefill_host)}
    report["page_table"] = page_table.tolist()
    report["prefill_page_table"] = prefill_table.tolist()
    report["prefill_page_size"] = 32
    report["prefill_input_local_shape"] = list(prefill_x.shape)
    report["prefill_rope_shapes"] = [list(t.shape) for t in prefill_rope]
    report["positions"] = {"original": positions.flatten().tolist(), "changed": (positions + 1).flatten().tolist()}
    for index, layer in enumerate(layers):
        linear = layer.kind == "linear_attention"
        state = layer.create_recurrent_state(args.batch) if linear else layer.create_paged_state(page_table)
        seed_state(layer, state, index)
        states.append(state)
        saved.append({name: ttnn.clone(tensor) for name, tensor in state_tensors(layer, state).items()})
        prefill_state = (
            layer.create_recurrent_state(args.prefill_batch) if linear else layer.create_paged_state(prefill_table)
        )
        prefill_states.append(prefill_state)
        owners[index]["prefill_state_shapes"] = {
            name: list(t.shape) for name, t in state_tensors(layer, prefill_state).items()
        }
        if not linear:
            owners[index]["prefill_page_table_shape"] = list(prefill_state.page_table.shape)
        outputs.append(ttnn.clone(stable_x))
        owners[index]["state_addresses"] = {
            name: int(t.buffer_address()) for name, t in state_tensors(layer, state).items()
        }
        if not linear:
            owners[index]["page_table_address"] = int(state.page_table.buffer_address())
    state_addresses = [address for owner in owners for address in owner["state_addresses"].values()]
    assert len(state_addresses) == len(set(state_addresses)) == 128
    assert len({o["page_table_address"] for o in owners if "page_table_address" in o}) == 16
    report["memory_samples"].append(memory_sample(mesh, "all_persistent_buffers_ready"))
    save()

    def restore():
        for layer, state, initial in zip(layers, states, saved):
            if layer._fused_decode_norm:
                assert layer._fused_decode_norm.next_phase == 0, "Incomplete native norm phase pair"
            for name, tensor in state_tensors(layer, state).items():
                ttnn.copy(initial[name], tensor)

    def refresh(variant):
        ttnn.copy(variant["x"], stable_x)
        ttnn.copy(variant["position"], stable_pos)
        for src, dst in zip(variant["rope"], stable_rope):
            ttnn.copy(src, dst)

    def sequence():
        for layer, state, target in zip(layers, states, outputs):
            kwargs = {} if layer.kind == "linear_attention" else {"current_pos": stable_pos, "rope": stable_rope}
            value = layer.decode_forward(stable_x, state=state, **kwargs)
            assert tuple(value.shape) == (1, args.batch, 1280)
            ttnn.copy(value, target)
            del value

    print(f"RESOURCE prefill batch={args.prefill_batch} length={args.prefill_length} warm all64 begin", flush=True)
    for index, (layer, state) in enumerate(zip(layers, prefill_states)):
        out = layer.prefill_forward(
            prefill_x, state=state, rope=prefill_rope if layer.kind == "full_attention" else None
        )
        assert tuple(out.shape) == (args.prefill_batch, args.prefill_length, 1280)
        del out
        if (index + 1) % 8 == 0:
            print(f"RESOURCE prefill warm {index + 1}/64", flush=True)
            report["warm_cache_growth"].append(
                {"phase": "prefill", "layers": index + 1, "entries": mesh.num_program_cache_entries()}
            )
    ttnn.synchronize_device(mesh)
    report["memory_samples"].append(memory_sample(mesh, "all64_prefill_warmed"))
    references = {}
    for variant in variants:
        restore()
        refresh(variant)
        print(f"RESOURCE eager batch={args.batch} variant={variant['name']} begin", flush=True)
        sequence()
        ttnn.synchronize_device(mesh)
        references[variant["name"]] = snapshot(layers, states, outputs)
        report["eager_reference"][variant["name"]] = references[variant["name"]]
        report["memory_samples"].append(memory_sample(mesh, "eager_" + variant["name"]))
        save()
    # Each owner prepares its own scratch before capture. Reusing mutable
    # buffers across independent layers would make trace lifetimes ambiguous.
    ccl_addresses = []
    for layer, owner in zip(layers, owners):
        owner["persistent_ccl_addresses"] = [
            int(t.buffer_address()) for buffers in layer._ccl_buffers.values() for t in buffers if t is not None
        ]
        ccl_addresses.extend(owner["persistent_ccl_addresses"])
    assert len(ccl_addresses) == len(set(ccl_addresses))
    report["persistent_ccl_ownership_unique"] = True
    norm_addresses = []
    for layer, owner in zip(layers, owners):
        if layer._fused_decode_norm:
            current = layer._fused_decode_norm.ownership()
            assert current["persistent_addresses"] == owner["fused_decode_norm"]["persistent_addresses"]
            assert current["semaphores"] == owner["fused_decode_norm"]["semaphores"]
            assert args.batch in current["warmed_batches"]
            owner["fused_decode_norm"] = current
            norm_addresses.extend(current["persistent_addresses"])
    live_addresses = ccl_addresses + state_addresses + norm_addresses
    assert len(live_addresses) == len(set(live_addresses))
    report["fused_decode_norm_ownership_unique"] = True
    index_addresses = []
    for layer, owner in zip(layers, owners):
        owner["prefill_batch_index_addresses"] = [
            int(t.buffer_address()) for t in layer._prefill_batch_indices.values()
        ]
        index_addresses.extend(owner["prefill_batch_index_addresses"])
    assert len(index_addresses) == len(set(index_addresses))
    report["prefill_batch_index_ownership_unique"] = True
    report["changed_input_updates_every_output_and_state"] = []
    for index in range(64):
        for name, original_shards in references["original"][index]["tensors"].items():
            for before, after in zip(original_shards, references["changed"][index]["tensors"][name]):
                changed = before["sha256"] != after["sha256"]
                check = {"layer": index, "tensor": name, "rank": before["rank"], "changed": changed}
                report["changed_input_updates_every_output_and_state"].append(check)
                assert changed, check
    # All copy/reset variants have executed too. Restore before the sole main
    # capture, then forbid cache misses for capture and every replay helper.
    restore()
    refresh(variants[0])
    ttnn.synchronize_device(mesh)
    warmed_entries = mesh.num_program_cache_entries()
    mesh.set_program_cache_misses_allowed(False)
    trace = None
    try:
        report["memory_samples"].append(memory_sample(mesh, "before_main_capture"))
        print(f"RESOURCE capture batch={args.batch} all64 begin", flush=True)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        sequence()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        ttnn.synchronize_device(mesh)
        assert mesh.num_program_cache_entries() == warmed_entries
        report["trace_id"] = str(trace)
        report["memory_samples"].append(memory_sample(mesh, "after_main_capture"))
        trace_view = report["memory_samples"][-1]["memory"]["TRACE"]
        report["main_trace_allocated_bytes_per_device"] = trace_view["allocated_bytes_per_device"]
        save()
        for replay_index in range(args.replays):
            variant = variants[replay_index % 2]
            restore()
            refresh(variant)
            print(f"RESOURCE replay={replay_index} variant={variant['name']} begin", flush=True)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            got = snapshot(layers, states, outputs)
            replay = {
                "index": replay_index,
                "variant": variant["name"],
                "trace_id": str(trace),
                "checks": [],
                "observed": got,
            }
            report["replays"].append(replay)
            require_exact(got, references[variant["name"]], replay["checks"])
            assert mesh.num_program_cache_entries() == warmed_entries
            replay["pass"] = True
            save()
        report["memory_samples"].append(memory_sample(mesh, "after_replays"))
        report["main_captures"] = 1
        report["main_recaptures"] = 0
        report["late_program_cache_growth"] = mesh.num_program_cache_entries() - warmed_entries
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        mesh.set_program_cache_misses_allowed(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, choices=(1, 32), required=True)
    parser.add_argument("--prefill-batch", type=int, choices=range(1, 33), default=1)
    parser.add_argument("--prefill-length", type=int, default=65)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy-file", type=Path)
    parser.add_argument("--trace-mib", type=int, default=32)
    parser.add_argument("--replays", type=int, default=3)
    args = parser.parse_args()
    assert args.replays >= 3 and args.trace_mib > 0 and args.prefill_length > 0
    assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1", "set TT_METAL_TRACE_ALLOC_TRACKING=1 before startup"
    assert (
        os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    ), "program-cache allocations must be tracked"
    torch.set_num_threads(8)
    repo = Path(__file__).resolve().parents[4]
    sources = [
        Path(__file__).resolve(),
        repo / "models/autoports/qwen_qwen3_6_27b/tt/multichip_decoder.py",
        repo / "models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py",
        repo / "models/demos/gpt_oss/tt/ccl.py",
        repo / "models/autoports/qwen_qwen3_6_27b/tt/fused_decode_norm.py",
        repo / "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_program_factory.cpp",
        repo / "ttnn/cpp/ttnn/operations/experimental/ccl/rms_allgather/device/rms_allgather_device_operation.cpp",
        repo
        / "ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/dit_fused_distributed_rmsnorm_program_factory.cpp",
        repo
        / "ttnn/cpp/ttnn/operations/experimental/ccl/matmul_reduce_scatter_prefill/device/matmul_reduce_scatter_prefill_program_factory.cpp",
        repo
        / "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_program_factory.cpp",
    ]
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(
            [
                sys.executable,
                "-m",
                "models.autoports.qwen_qwen3_6_27b.tests.check_multichip_stack_resources",
                *sys.argv[1:],
            ]
        ),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources
        },
        "hf_revision": REVISION,
        "mesh": [1, 4],
        "batch": args.batch,
        "prefill_batch": args.prefill_batch,
        "prefill_length": args.prefill_length,
        "prefill_shape": [args.prefill_batch, args.prefill_length, 5120],
        "cq_id": 0,
        "trace_region_bytes": args.trace_mib * 1024**2,
        "resource_counts": {
            "layers": 64,
            "ccl_managers": 64,
            "ccl_semaphores": 768,
            "conv_semaphores": 144,
            "conv_exchange_buffers": 48,
        },
        "scope": "resource/isolation test; representative layer0/layer3 weights shared, same prepared input at every layer; not full-model numerical inference or full-stack weight capacity",
        "oracle": "exact SHA256 of every logical output and complete mutable state tensor, every layer and mesh rank",
        "all_main_persistent_allocations_before_capture": True,
        "allocation_acknowledgments": 0,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
            )
        },
        "memory_samples": [],
        "warm_cache_growth": [],
        "eager_reference": {},
        "replays": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_mib * 1024**2)
        mesh.enable_program_cache()
        report["tracker_positive_control"] = tracker_positive_control(mesh)
        report["memory_samples"].append(memory_sample(mesh, "before_layers"))
        save()
        run(args, mesh, report, save)
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
