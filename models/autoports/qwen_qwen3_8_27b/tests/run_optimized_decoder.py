# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-shape HF parity and traced decode runner; conversions are test boundaries."""

import argparse
import copy
import json
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.reference import (
    load_config,
    load_layer_weights,
    make_reference,
    synthetic_layer_weights,
)
from models.autoports.qwen_qwen3_8_27b.tt.fused_decoder import FusedDecoder
from models.autoports.qwen_qwen3_8_27b.tt.optimized_decoder import OptimizedDecoder


class NoTorch(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"Torch operation in measured forward: {func}")


@contextmanager
def device_only():
    def forbidden(*args, **kwargs):
        raise AssertionError("Host conversion in measured forward")

    with (
        NoTorch(),
        patch.object(ttnn, "from_torch", forbidden),
        patch.object(ttnn, "to_torch", forbidden),
        patch.object(ttnn, "as_tensor", forbidden),
    ):
        yield


def pcc(a, b):
    assert tuple(a.shape) == tuple(b.shape), (a.shape, b.shape)
    if a.numel() > 2_000_000:
        # Bounded host comparison memory at advertised context.
        a, b = a.flatten(), b.flatten()
        sums = torch.zeros(5, dtype=torch.float64)
        for start in range(0, a.numel(), 1_000_000):
            aa, bb = a[start : start + 1_000_000].double(), b[start : start + 1_000_000].double()
            assert torch.isfinite(aa).all() and torch.isfinite(bb).all()
            sums += torch.stack((aa.sum(), bb.sum(), (aa * aa).sum(), (bb * bb).sum(), (aa * bb).sum()))
        sa, sb, saa, sbb, sab = sums
        n = a.numel()
        return ((sab - sa * sb / n) / torch.sqrt((saa - sa * sa / n) * (sbb - sb * sb / n))).item()
    a, b = a.float().flatten(), b.float().flatten()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    return torch.corrcoef(torch.stack((a, b)))[0, 1].item()


def run_case(args, config, reference, rope, mesh, decoder):
    torch.manual_seed(123 + args.length)
    cache = DynamicCache(config=config)
    x = (torch.randn(args.batch, args.length + 1, config.hidden_size) * 0.1).bfloat16()
    if args.activations:
        recorded = torch.load(args.activations / f"layer{args.layer}.pt", weights_only=True)
        assert recorded.shape[1] >= args.length + 2
        x = recorded[torch.arange(args.batch) % recorded.shape[0], : args.length + 1].clone()
    position_ids = torch.arange(args.length + 1).unsqueeze(0).expand(args.batch, -1)
    cos, sin = rope(x, position_ids)
    with torch.no_grad():
        expected_prefill = torch.empty_like(x[:, :-1])
        # Preserve the exact HF layer and all target shapes; bound only the
        # reference's temporary attention mask and activations.
        for start in range(0, args.length, 1024):
            end = min(start + 1024, args.length)
            mask = None
            if reference.layer_type == "full_attention":
                future = torch.arange(end)[None, :] > torch.arange(start, end)[:, None]
                mask = torch.zeros(end - start, end, dtype=torch.bfloat16).masked_fill(future, float("-inf"))[
                    None, None
                ]
            expected_prefill[:, start:end] = reference(
                x[:, start:end],
                position_embeddings=(cos[:, start:end], sin[:, start:end]),
                attention_mask=mask,
                past_key_values=cache,
            )
            if args.length > 4096 and (end % 16384 == 0 or end == args.length):
                print("HF_PREFIX", end, flush=True)
        if args.length < config.max_position_embeddings:
            replay_cache = copy.deepcopy(cache)
            prefix_cache = copy.deepcopy(cache)
            expected_decode = reference(
                x[:, -1:], position_embeddings=(cos[:, -1:], sin[:, -1:]), past_key_values=cache
            )
    print("HF_READY", reference.layer_type, args.length, flush=True)
    trace = None
    try:

        def upload(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        pages_per_user = (args.length + 4 + 31) // 32
        owned_pages = args.batch * pages_per_user
        num_pages = owned_pages + 3
        permutation = torch.randperm(num_pages)
        table = permutation[:owned_pages].reshape(args.batch, pages_per_user).int()
        page_table = upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        state = decoder.allocate_state(batch_size=args.batch, num_pages=num_pages)
        device_x = upload(x[:, :-1])
        device_cos, device_sin = upload(cos[:, :-1]), upload(sin[:, :-1])
        initial_state = {name: ttnn.clone(tensor) for name, tensor in vars(state).items() if tensor is not None}
        print("TT_PREFILL_BEGIN", flush=True)

        def prefill():
            with device_only():
                return decoder.prefill_forward(
                    device_x, state=state, page_table=page_table, cos=device_cos, sin=device_sin
                )

        timings = {}
        comparisons = {}

        def compare(mode, tensor):
            if args.compare_dir is None:
                return
            args.compare_dir.mkdir(parents=True, exist_ok=True)
            path = args.compare_dir / f"layer{args.layer}_b{args.batch}_s{args.length}_{mode}.pt"
            host = ttnn.to_torch(tensor)
            if args.baseline:
                torch.save(host, path)
            else:
                comparisons[mode] = pcc(torch.load(path, weights_only=True), host)
                assert comparisons[mode] >= 0.995, comparisons

        actual_prefill = prefill()
        assert tuple(actual_prefill.shape) == (args.batch, args.length, config.hidden_size), actual_prefill.shape
        compare("prefill", actual_prefill)
        if args.benchmark:
            samples = []
            for _ in range(5):
                for name, snapshot in initial_state.items():
                    ttnn.copy(snapshot, getattr(state, name))
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                measured = prefill()
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - begin) * 1000)
                ttnn.deallocate(measured)
            timings["warmed_prefill_ms"] = samples
        if getattr(args, "trace_prefill", False):
            for name, snapshot in initial_state.items():
                ttnn.copy(snapshot, getattr(state, name))
            prefill_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced_prefill = prefill()
            ttnn.end_trace_capture(mesh, prefill_trace, cq_id=0)
            samples = []
            try:
                for _ in range(5):
                    for name, snapshot in initial_state.items():
                        ttnn.copy(snapshot, getattr(state, name))
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter()
                    ttnn.execute_trace(mesh, prefill_trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples.append((time.perf_counter() - begin) * 1000)
                    assert pcc(expected_prefill, ttnn.to_torch(traced_prefill)) >= 0.995
                timings["traced_prefill_ms"] = samples
            finally:
                ttnn.release_trace(mesh, prefill_trace)
                ttnn.deallocate(traced_prefill)
        prefill_host = ttnn.to_torch(actual_prefill)
        prefill_per_user = [pcc(expected_prefill[i], prefill_host[i]) for i in range(args.batch)]
        assert min(prefill_per_user) >= 0.995, prefill_per_user
        prefill_pcc = pcc(expected_prefill, prefill_host)
        print("PREFILL_PCC", prefill_pcc, flush=True)
        assert prefill_pcc >= 0.995
        continuation_pcc = None
        if args.continuation and args.length > 1:
            split = min(33, args.length // 2)
            continued = decoder.allocate_state(batch_size=args.batch, num_pages=num_pages)
            first = decoder.prefill_forward(
                upload(x[:, :split]),
                state=continued,
                page_table=page_table,
                cos=upload(cos[:, :split]),
                sin=upload(sin[:, :split]),
            )
            positions = torch.arange(split, args.length, dtype=torch.int32)[:, None].expand(-1, args.batch)
            inputs = upload(x[:, split:-1])
            cc, ss = upload(cos[:, split:-1]), upload(sin[:, split:-1])
            pp = upload(positions, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            with device_only():
                tail = decoder.prefill_forward(
                    inputs, state=continued, start_pos=split, page_table=page_table, cos=cc, sin=ss, positions=pp
                )
            continuation_pcc = pcc(expected_prefill, torch.cat((ttnn.to_torch(first), ttnn.to_torch(tail)), dim=1))
            print("CONTINUATION_PCC", continuation_pcc, flush=True)
            assert continuation_pcc >= 0.995
            state = continued
        if args.profile:
            from tracy import signpost

            for name, snapshot in initial_state.items():
                ttnn.copy(snapshot, getattr(state, name))
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL")
            profile_start = time.perf_counter()
            warmed_prefill = prefill()
            ttnn.synchronize_device(mesh)
            timings["profiled_prefill_ms"] = [(time.perf_counter() - profile_start) * 1000]
            signpost("PERF_PREFILL_END")
            assert pcc(expected_prefill, ttnn.to_torch(warmed_prefill)) >= 0.995
            ttnn.deallocate(warmed_prefill)
        if state.key is not None:
            for tensor in (state.key, state.value):
                unused = ttnn.to_torch(tensor)[permutation[owned_pages:]]
                assert torch.count_nonzero(unused) == 0
        if args.length == config.max_position_embeddings:
            metrics = {
                "implementation": type(decoder).__name__,
                "policy": getattr(decoder, "policy", {}),
                "activations": str(args.activations) if args.activations else "synthetic",
                "timings": timings,
                "equivalence_pcc": comparisons,
                "layer": args.layer,
                "kind": reference.layer_type,
                "weights": "synthetic_from_real_stats" if args.synthetic_stats else "real",
                "length": args.length,
                "batch": args.batch,
                "prefill_pcc": prefill_pcc,
                "runtime_fallback_audit": "passed",
                "decode": "Covered separately at context max_position_embeddings; no out-of-contract extra token.",
            }
            print(json.dumps(metrics), flush=True)
            return metrics
        snapshots = {name: ttnn.clone(tensor) for name, tensor in vars(state).items() if tensor is not None}

        def restore():
            for name, snapshot in snapshots.items():
                ttnn.copy(snapshot, getattr(state, name))

        decode_x = upload(x[:, -1:])
        decode_cos, decode_sin = upload(cos[:, -1:]), upload(sin[:, -1:])
        current_pos = upload(
            torch.full((args.batch,), args.length, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT
        )

        def decode():
            with device_only():
                return decoder.decode_forward(
                    decode_x,
                    state=state,
                    page_table=page_table,
                    current_pos=current_pos,
                    cos=decode_cos,
                    sin=decode_sin,
                )

        print("TT_DECODE_WARM", flush=True)
        warm = decode()
        assert tuple(warm.shape) == (args.batch, 1, config.hidden_size), warm.shape
        ttnn.synchronize_device(mesh)
        eager_pcc = pcc(expected_decode, ttnn.to_torch(warm))
        print("EAGER_DECODE_PCC", eager_pcc, flush=True)
        ttnn.deallocate(warm)
        restore()
        print("TT_TRACE_CAPTURE", flush=True)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        actual_decode = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        restore()
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        compare("decode", actual_decode)
        assert tuple(actual_decode.shape) == (args.batch, 1, config.hidden_size), actual_decode.shape
        result = ttnn.to_torch(actual_decode)
        decode_per_user = [pcc(expected_decode[i], result[i]) for i in range(args.batch)]
        assert min(decode_per_user) >= 0.995, decode_per_user
        decode_pcc = pcc(expected_decode, result)
        print("TRACED_DECODE_PCC", decode_pcc, flush=True)
        assert decode_pcc >= 0.995
        restore()
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        repeated = ttnn.to_torch(actual_decode)
        assert torch.equal(result, repeated)
        if args.profile:
            restore()
            ttnn.synchronize_device(mesh)
            signpost("PERF_DECODE")
            profile_start = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            timings["profiled_decode_ms"] = [(time.perf_counter() - profile_start) * 1000]
            signpost("PERF_DECODE_END")

        if args.benchmark:
            samples = []
            for _ in range(30):
                restore()
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=getattr(args, "blocking_trace", False))
                if not getattr(args, "blocking_trace", False):
                    ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - begin) * 1000)
                assert torch.equal(result, ttnn.to_torch(actual_decode))
            timings["traced_decode_ms"] = samples

        # Refresh exactly the buffers baked into the trace; use a different
        # token and, for full attention, different per-user absolute positions.
        sequential_pcc = []
        if args.activations:
            restore()
            steps = min(4, recorded.shape[1] - args.length, config.max_position_embeddings - args.length)
            for step in range(steps):
                value = recorded[
                    torch.arange(args.batch) % recorded.shape[0], args.length + step : args.length + step + 1
                ].clone()
                position = torch.full((args.batch,), args.length + step, dtype=torch.int32)
                cc, ss = rope(value, position[:, None])
                for destination, source in ((decode_x, value), (decode_cos, cc), (decode_sin, ss)):
                    ttnn.copy_host_to_device_tensor(
                        ttnn.from_torch(source.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), destination
                    )
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(position, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT), current_pos
                )
                with torch.no_grad():
                    expected_step = reference(value, position_embeddings=(cc, ss), past_key_values=replay_cache)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                sequential_pcc.append(pcc(expected_step, ttnn.to_torch(actual_decode)))
                assert sequential_pcc[-1] >= 0.995, sequential_pcc
            print("SEQUENTIAL_TRACE_PCC", sequential_pcc, flush=True)

        changed_x = (torch.randn(args.batch, 1, config.hidden_size) * 0.1).bfloat16()
        if args.activations:
            changed_x = recorded[
                torch.arange(args.batch) % recorded.shape[0], args.length + 1 : args.length + 2
            ].clone()
        changed_pos = torch.tensor([args.length - user % min(args.length, 4) for user in range(args.batch)])
        changed_cos, changed_sin = rope(changed_x, changed_pos[:, None])
        with torch.no_grad():
            if reference.layer_type == "full_attention":
                expected_changed = []
                for user in range(args.batch):
                    control_cache = DynamicCache(config=config)
                    original = prefix_cache.layers[args.layer]
                    pos = changed_pos[user].item()
                    control_cache.update(
                        original.keys[user : user + 1, :, :pos].clone(),
                        original.values[user : user + 1, :, :pos].clone(),
                        args.layer,
                    )
                    expected_changed.append(
                        reference(
                            changed_x[user : user + 1],
                            position_embeddings=(changed_cos[user : user + 1], changed_sin[user : user + 1]),
                            past_key_values=control_cache,
                        )
                    )
                expected_changed = torch.cat(expected_changed)
            else:
                expected_changed = reference(
                    changed_x, position_embeddings=(changed_cos, changed_sin), past_key_values=prefix_cache
                )

        def refresh(tensor, value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            host = ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout)
            ttnn.copy_host_to_device_tensor(host, tensor)

        restore()
        refresh(decode_x, changed_x)
        refresh(decode_cos, changed_cos)
        refresh(decode_sin, changed_sin)
        refresh(current_pos, changed_pos.int(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        changed_result = ttnn.to_torch(actual_decode)
        changed_pcc = pcc(expected_changed, changed_result)
        print("CHANGED_INPUT_TRACE_PCC", changed_pcc, flush=True)
        assert changed_pcc >= 0.995
        assert not torch.equal(result, changed_result)
        swapped_pcc = None
        if state.key is not None and args.batch > 1:
            restore()
            refresh(page_table, table.flip(0), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            refresh(decode_x, changed_x.flip(0))
            refresh(decode_cos, changed_cos.flip(0))
            refresh(decode_sin, changed_sin.flip(0))
            refresh(current_pos, changed_pos.flip(0).int(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            swapped_pcc = pcc(expected_changed.flip(0), ttnn.to_torch(actual_decode))
            assert swapped_pcc >= 0.995
        metrics = {
            "implementation": type(decoder).__name__,
            "policy": getattr(decoder, "policy", {}),
            "activations": str(args.activations) if args.activations else "synthetic",
            "timings": timings,
            "equivalence_pcc": comparisons,
            "layer": args.layer,
            "kind": reference.layer_type,
            "weights": "synthetic_from_real_stats" if args.synthetic_stats else "real",
            "length": args.length,
            "batch": args.batch,
            "prefill_pcc": prefill_pcc,
            "traced_decode_pcc": decode_pcc,
            "trace_wait": "blocking" if getattr(args, "blocking_trace", False) else "nonblocking_then_synchronize",
            "prefill_per_user_pcc": prefill_per_user,
            "decode_per_user_pcc": decode_per_user,
            "repeat_bitwise_equal": True,
            "changed_input_trace_pcc": changed_pcc,
            "sequential_trace_pcc": sequential_pcc,
            "runtime_fallback_audit": "passed",
            "unused_pages_unchanged": True if state.key is not None else None,
            "refreshed_page_table_pcc": swapped_pcc,
            "continuation_pcc": continuation_pcc,
        }
        print(json.dumps(metrics), flush=True)
        return metrics
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)


def run(args):
    torch.set_num_threads(4)
    if getattr(args, "policy_file", None):
        args.policy = args.policy_file.read_text()
    config = load_config(args.snapshot)
    weights = (
        synthetic_layer_weights(args.synthetic_stats, config.layer_types[args.layer])
        if args.synthetic_stats
        else load_layer_weights(args.snapshot, args.layer)
    )
    reference = make_reference(config, args.layer, weights)
    rope = Qwen3_5TextRotaryEmbedding(config)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        decoder_cls = FusedDecoder if args.baseline else OptimizedDecoder
        decoder = decoder_cls.from_state_dict(
            weights,
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh,
            **({} if args.baseline else {"policy": json.loads(args.policy)}),
        )
        memory = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        setup_memory = {
            "banks": memory.num_banks,
            "bytes_per_bank": memory.total_bytes_per_bank,
            "allocated_bytes_per_bank": memory.total_bytes_allocated_per_bank,
        }
        results = []
        lengths = [int(value) for value in args.lengths.split(",")] if args.lengths else [args.length]
        for length in lengths:
            args.length = length
            results.append(run_case(args, config, reference, rope, mesh, decoder))
            results[-1]["setup_dram_memory"] = setup_memory
            args.output.write_text(json.dumps(results, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="{}")
    parser.add_argument("--policy-file", type=Path)
    parser.add_argument("--activations", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--compare-dir", type=Path)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(0, 3), required=True)
    parser.add_argument("--length", type=int, default=32)
    parser.add_argument("--lengths", help="Comma-separated increasing lengths and short reuse, on one loaded instance")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace-prefill", action="store_true")
    parser.add_argument("--blocking-trace", action="store_true")
    parser.add_argument("--continuation", action="store_true")
    parser.add_argument("--synthetic-stats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
