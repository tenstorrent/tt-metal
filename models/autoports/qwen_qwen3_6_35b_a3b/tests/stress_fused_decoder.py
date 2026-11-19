# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stateful real-weight traced decoder endurance and continuation regression."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, hf_forward, layer, pcc, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only
from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import FusedDecoder


def run(args):
    tracking = not getattr(args, "clean_performance", False)
    assert (os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == "1") == tracking
    if not tracking:
        assert os.environ.get("TT_METAL_WATCHER", "0") == "0"
        assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") == "0"
    assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
    record(args.output)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    c = config()
    assert args.prompt_length > 0 and args.decode_length > 0
    assert args.prompt_length + args.decode_length + args.continuation_length <= c.max_position_embeddings
    sd = weights(args.layer)
    hf = layer(sd, args.layer)
    prefix = torch.randn(1, args.prompt_length, 2048, dtype=torch.bfloat16) * 0.05
    tokens = torch.randn(1, args.decode_length, 2048, dtype=torch.bfloat16) * 0.05
    continuation = torch.randn(1, args.continuation_length, 2048, dtype=torch.bfloat16) * 0.05
    activation_file = getattr(args, "activation_file", None)
    if activation_file:
        payload = torch.load(activation_file, weights_only=True)
        assert payload["metadata"]["checkpoint_revision"] == "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
        inputs = payload["layers"][str(args.layer)]["x"]
        total = args.prompt_length + args.decode_length + args.continuation_length
        assert inputs.shape == (1, total, 2048)
        prefix = inputs[:, : args.prompt_length].contiguous()
        tokens = inputs[:, args.prompt_length : args.prompt_length + args.decode_length].contiguous()
        continuation = inputs[:, -args.continuation_length :].contiguous()
    continuation_start = args.prompt_length + args.decode_length
    # Optional immutable CPU-oracle reuse for isolated device A/B experiments.
    # Any harness/reference/HF source or sequence-parameter change invalidates it.
    import inspect

    from models.autoports.qwen_qwen3_6_35b_a3b.tests import reference

    oracle_key = dict(
        parameters={
            k: getattr(args, k) for k in ("layer", "seed", "prompt_length", "decode_length", "continuation_length")
        },
        source_hashes={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(reference.__file__),
                Path(inspect.getfile(type(hf))),
                Path(inspect.getfile(DynamicCache)),
            )
        },
        revision=reference.REVISION,
        torch_version=torch.__version__,
    )
    oracle_path = Path(args.oracle_cache) if args.oracle_cache else None
    cached = torch.load(oracle_path, weights_only=False) if oracle_path and oracle_path.exists() else None
    if cached is not None:
        assert cached["key"] == oracle_key, "CPU oracle source/parameter fingerprint mismatch"
        assert all(
            torch.equal(cached[k], v)
            for k, v in (("prefix", prefix), ("tokens", tokens), ("continuation", continuation))
        )
        ref_p, refs, ref_continuation = (cached[k] for k in ("ref_p", "refs", "ref_continuation"))
        print("HF_STRESS_ORACLE_REUSED", str(oracle_path), flush=True)
    else:
        cache = DynamicCache(config=c)
        ref_p = hf_forward(hf, prefix, cache, 0)
        refs = []
        for i in range(args.decode_length):
            refs.append(hf_forward(hf, tokens[:, i : i + 1], cache, args.prompt_length + i).clone())
            if args.decode_length >= 4096 and (i + 1) % 2048 == 0:
                print("HF_STRESS_REFERENCE_PROGRESS", args.layer, i + 1, args.decode_length, flush=True)
        ref_continuation = hf_forward(hf, continuation, cache, continuation_start)
        if oracle_path:
            oracle_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                dict(
                    key=oracle_key,
                    prefix=prefix,
                    tokens=tokens,
                    continuation=continuation,
                    ref_p=ref_p,
                    refs=refs,
                    ref_continuation=ref_continuation,
                ),
                oracle_path,
            )
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        dec = FusedDecoder.from_state_dict(
            sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh, **json.loads(getattr(args, "build_kwargs", "{}"))
        )
        state = (
            dec.new_recurrent_state()
            if dec.kind == "linear_attention"
            else dec.new_kv_state(context_length=args.prompt_length + args.decode_length + args.continuation_length)
        )
        keys = ("conv", "recurrent") if dec.kind == "linear_attention" else ("k", "v")

        def upload(t):
            return ttnn.from_torch(t, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        tp, td = upload(prefix), upload(tokens[:, :1])
        plan = dec.prepare_prefill(seq_len=args.prompt_length)
        pos = dec.prepare_positions(start_pos=args.prompt_length, seq_len=1) if dec.kind == "full_attention" else None
        with device_only():
            prefill = dec.prefill_forward(tp, state=state, plan=plan)
        prefill_pcc = pcc(ref_p, ttnn.to_torch(prefill).reshape(prefix.shape))
        assert prefill_pcc >= 0.995
        del prefill
        saved = {k: ttnn.clone(state[k]) for k in keys}

        def restore():
            for k in keys:
                ttnn.copy(saved[k], state[k])

        restore()
        with device_only():
            warm = dec.decode_forward(td, state=state, positions=pos)
        del warm
        restore()
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            out = dec.decode_forward(td, state=state, positions=pos)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        from ttnn.tools import trace_allocation_tracker

        trace_allocation_tracker.acknowledge_corruptible(out)

        def refresh(dest, t, dtype, layout):
            host = ttnn.from_torch(t, dtype=dtype, layout=layout)
            ttnn.copy_host_to_device_tensor(host, dest)

        first = []
        rows = []
        trace_times = []
        for repeat in range(args.repeats):
            restore()
            for i in range(args.decode_length):
                refresh(td, tokens[:, i : i + 1], ttnn.bfloat16, ttnn.TILE_LAYOUT)
                if pos is not None:
                    inv = 1 / (c.rope_parameters["rope_theta"] ** (torch.arange(0, 64, 2).float() / 64))
                    a = (args.prompt_length + i) * inv
                    a = torch.cat([a, a]).reshape(1, 1, 1, 64)
                    refresh(pos["cos"], a.cos(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                    refresh(pos["sin"], a.sin(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
                    refresh(
                        pos["current_pos"],
                        torch.tensor([args.prompt_length + i], dtype=torch.int32),
                        ttnn.int32,
                        ttnn.ROW_MAJOR_LAYOUT,
                    )
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                trace_times.append((time.perf_counter() - begin) * 1000)
                got = ttnn.to_torch(out).reshape(1, 1, 2048).clone()
                score = pcc(refs[i], got)
                contribution = pcc(refs[i] - tokens[:, i : i + 1], got - tokens[:, i : i + 1])
                if repeat == 0:
                    first.append(got)
                else:
                    assert torch.equal(first[i], got), (repeat, i, "nondeterminism")
                rows.append(
                    dict(repeat=repeat, position=args.prompt_length + i, pcc=score, contribution_pcc=contribution)
                )
                if args.decode_length >= 4096 and (i + 1) % 2048 == 0:
                    print("TT_STRESS_PROGRESS", args.layer, repeat, i + 1, args.decode_length, flush=True)
            print("STRESS_REPEAT_COMPLETE", args.layer, repeat, min(r["pcc"] for r in rows), flush=True)
        ttnn.release_trace(mesh, trace)
        continuation_plan = dec.prepare_prefill(seq_len=args.continuation_length, start_pos=continuation_start)
        continuation_input = upload(continuation)
        with device_only():
            continued = dec.prefill_forward(continuation_input, state=state, plan=continuation_plan)
        got_continuation = ttnn.to_torch(continued).reshape(continuation.shape)
        continuation_rows = [
            dict(
                position=continuation_start + i,
                pcc=pcc(ref_continuation[:, i : i + 1], got_continuation[:, i : i + 1]),
                contribution_pcc=pcc(
                    ref_continuation[:, i : i + 1] - continuation[:, i : i + 1],
                    got_continuation[:, i : i + 1] - continuation[:, i : i + 1],
                ),
            )
            for i in range(args.continuation_length)
        ]
        row_diagnostics = [r for r in rows + continuation_rows if min(r["pcc"], r["contribution_pcc"]) < 0.995]
        whole_scores = dict(
            decode_whole_pcc=pcc(torch.cat(refs, dim=1), torch.cat(first, dim=1)),
            continuation_whole_pcc=pcc(ref_continuation, got_continuation),
            decode_contribution_whole_pcc=pcc(torch.cat(refs, dim=1) - tokens, torch.cat(first, dim=1) - tokens),
            continuation_contribution_whole_pcc=pcc(ref_continuation - continuation, got_continuation - continuation),
        )
        whole_output_gate = getattr(args, "whole_output_gate", False)
        failures = (
            [dict(metric=k, pcc=v) for k, v in whole_scores.items() if v < 0.995]
            if whole_output_gate
            else row_diagnostics
        )
        Path(args.output).write_text(
            json.dumps(
                dict(
                    gate_pass=not failures,
                    failed_rows=failures,
                    decoder=type(dec).__name__,
                    layer=args.layer,
                    prefill_pcc=prefill_pcc,
                    activation_source=activation_file or "seeded random BF16 std0.05",
                    precision_policy=getattr(dec, "precision_policy", None),
                    **whole_scores,
                    gate_scope="whole-output .995; per-row diagnostic" if whole_output_gate else "strict per-row .995",
                    row_diagnostics=row_diagnostics,
                    prompt_length=args.prompt_length,
                    seed=args.seed,
                    decode_length=args.decode_length,
                    repeats=args.repeats,
                    captures=1,
                    repeated_bit_equal=True,
                    tracking=tracking,
                    decode_trace_samples_ms=trace_times,
                    decode_trace_median_ms=statistics.median(trace_times),
                    timing_boundary="blocking traced decoder only; input transfers, HF, readback, PCC and restore excluded",
                    program_cache_included=True,
                    cpu_oracle_key=oracle_key,
                    cpu_oracle_cache=str(oracle_path) if oracle_path else None,
                    acknowledged_buffer="output only; copied to CPU before replay, no aliases or other live traces",
                    results=rows,
                    continuation_length=args.continuation_length,
                    continuation_results=continuation_rows,
                ),
                indent=2,
            )
            + "\n"
        )
        if failures:
            torch.save(
                dict(
                    prefix=prefix,
                    tokens=tokens,
                    reference=refs,
                    outputs=first,
                    continuation=continuation,
                    continuation_reference=ref_continuation,
                    continuation_output=got_continuation,
                ),
                str(args.output) + ".tensors.pt",
            )
        assert not failures, failures[:4]
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=913)
    p.add_argument("--prompt-length", type=int, default=128)
    p.add_argument("--decode-length", type=int, default=128)
    p.add_argument("--continuation-length", type=int, default=33)
    p.add_argument("--output", required=True)
    p.add_argument("--oracle-cache")
    run(p.parse_args())
