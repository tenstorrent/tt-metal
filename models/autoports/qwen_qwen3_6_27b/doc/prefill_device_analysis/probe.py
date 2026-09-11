# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Real-weight prefill A/B probe; experimental patches never change serving defaults.

Run without Tracy for latency, with Tracy for one reduced-stack device window.
The reduced stack contains layers 0 and 3: layer 0 sees real embeddings, while
layer 3 sees layer 0's output (not the full stack's layer-3 activation).
"""

import argparse
import contextlib
import inspect
import json
import math
import os
import statistics
import textwrap
import time
from pathlib import Path

os.environ.setdefault("QWEN36_PREFILL_RECURRENCE", "eager")

import torch
from tracy import signpost

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt import functional_decoder as fd
from models.autoports.qwen_qwen3_6_27b.tt import multichip_decoder as md
from models.autoports.qwen_qwen3_6_27b.tt.generator import build_generator


def patch_function(module, name, replacements):
    original = getattr(module, name)
    source = textwrap.dedent(inspect.getsource(original))
    for old, new in replacements:
        if old not in source and new in source:
            continue  # The production path may already contain an accepted repair.
        if old not in source:
            raise ValueError(f"experimental patch no longer matches {name}: {old}")
        source = source.replace(old, new)
    namespace = dict(original.__globals__)
    exec(compile(source, f"<prefill experiment {name}>", "exec"), namespace)
    setattr(module, name, namespace[name])


def configure(candidate):
    os.environ["QWEN36_PREFILL_SCAN"] = "hillis" if candidate.startswith("hillis") else "sequential"
    if candidate in ("outer", "outer_hoist"):
        changes = [
            ("_scan_matmul(ttnn.transpose(k_t, -2, -1), delta)", "ttnn.multiply(ttnn.transpose(k_t, -2, -1), delta)")
        ]
        if candidate == "outer_hoist":
            changes += [
                ("outputs = []", "outputs = []\nkey_transposed = ttnn.transpose(key, -2, -1)"),
                ("ttnn.transpose(k_t, -2, -1)", "key_transposed[:, step : step + 1]"),
            ]
        patch_function(md, "_sequential_recurrence", changes)
    if candidate == "single_silu":
        patch_function(
            md.MultichipDecoder,
            "_tp_linear",
            [
                (
                    "if not decode and fused_activation == ttnn.UnaryOpType.SILU:",
                    'if not decode and "program_config" not in kwargs and fused_activation == ttnn.UnaryOpType.SILU:',
                )
            ],
        )
    if candidate.startswith("kda_scan"):
        kda_recurrence.fp32 = candidate.endswith("fp32")
        md._sequential_recurrence = kda_recurrence
    if candidate in ("mac", "traced_keepalive_mac"):
        patch_function(
            md,
            "_sequential_recurrence",
            [
                (
                    "update = _scan_matmul(ttnn.transpose(k_t, -2, -1), delta)",
                    "next_state = ttnn.mac(ttnn.transpose(k_t, -2, -1), delta, decayed)",
                ),
                ("state = ttnn.add(decayed, update)", "state = next_state"),
                ("ttnn.deallocate(update)", ""),
            ],
        )
    if candidate in ("ccl_bfp8", "ccl_direct"):
        changes = [
            (
                'weight_name == "mlp_down_decode" else "token_mixer"',
                'weight_name in ("mlp_down_decode", "mlp_down_prefill") else "token_mixer"',
            )
        ]
        if candidate == "ccl_direct":
            changes.append(
                (
                    "dtype=self.policy.activation_residual_dtype,",
                    "dtype=ttnn.bfloat8_b if row else self.policy.activation_residual_dtype,",
                )
            )
        patch_function(md.MultichipDecoder, "_tp_linear", changes)
    if candidate == "packed":
        source = textwrap.dedent(inspect.getsource(md.MultichipDecoder._linear_attention_prefill_chunk_impl))
        old = source[source.index("    mixed = self._tp_linear(") : source.index("    mixed = ttnn.permute(mixed")]
        new = """    packed = self._tp_linear(
        hidden_states, "packed_prefill", k=5120, n=4160, decode=False,
        compute_kernel_config=self.linear_input_compute_kernel_config)
    mixed = packed[..., :2560]
    z = packed[..., 2560:4096]
    beta = packed[..., 4096:4108]
    decay = packed[..., 4128:4140]

"""
        patch_function(md.MultichipDecoder, "_linear_attention_prefill_chunk_impl", [(old, new)])
    if candidate == "reuse12":
        os.environ["QWEN36_SCAN_MATMUL_GRID"] = "3x4"
        patch_function(
            fd,
            "_scan_matmul",
            [("min(int(a.shape[-1]), int(a.shape[-2]), int(b.shape[-1])) >= ttnn.TILE_SIZE", "True")],
        )
        md._scan_matmul = fd._scan_matmul


def kda_recurrence(query, key, value, beta, decay, *, initial_state, groups, sequence, value_dim, batch, value_heads):
    """Eight token affine groups fit 12 local heads * 8 <= 110 workers.

    An experimental algebraic replacement, not a validated serving policy.
    FP32 scan entries are an op requirement, so include their cast cost.
    """
    if batch != 1:
        raise ValueError("This experimental KDA mapping is batch 1 only")
    if kda_recurrence.fp32:
        key, value, beta, decay = [ttnn.typecast(t, ttnn.float32) for t in (key, value, beta, decay)]
    key_t = ttnn.transpose(key, -2, -1)
    # Avoid host upload in the measured path: the wrapper installs this at setup.
    identity = kda_recurrence.identity
    transform = ttnn.multiply(decay, ttnn.subtract(identity, ttnn.multiply(beta, ttnn.multiply(key_t, key))))
    bias = ttnn.multiply(beta, ttnn.multiply(key_t, value))
    state = ttnn.typecast(ttnn.reshape(initial_state, (groups, value_dim, value_dim)), ttnn.float32)
    outputs = []
    for start in range(0, sequence, 8):
        length = min(8, sequence - start)
        a = ttnn.reshape(transform[:, start : start + length], (groups * length, value_dim, value_dim))
        b = ttnn.reshape(bias[:, start : start + length], (groups * length, value_dim, value_dim))
        entries = ttnn.experimental.kda.affine_exclusive_scan(a, b, state, length)
        if kda_recurrence.fp32:
            product = ttnn.matmul(
                a,
                entries,
                dtype=ttnn.float32,
                compute_kernel_config=kda_recurrence.compute,
                program_config=ttnn.MatmulMultiCoreReuseProgramConfig(
                    compute_with_storage_grid_size=(8, 10),
                    in0_block_w=4,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=4,
                    per_core_N=4,
                ),
            )
        else:
            product = fd._scan_matmul(a, entries)
        inclusive = ttnn.add(product, b)
        states = ttnn.reshape(inclusive, (groups, length, value_dim, value_dim))
        state = ttnn.typecast(ttnn.reshape(states[:, -1:], (groups, value_dim, value_dim)), ttnn.float32)
        attended = fd._scan_matmul(query[:, start : start + length], ttnn.typecast(states, ttnn.bfloat16))
        outputs.append(attended)
    output = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=1)
    return output, ttnn.reshape(state, (batch, value_heads, value_dim, value_dim))


def host_tensor(value):
    return ttnn.to_torch(ttnn.get_device_tensors(value)[0]).float().clone()


def comparison(reference, actual):
    a, b = reference.double().reshape(-1), actual.double().reshape(-1)
    return {
        "pcc": float(torch.corrcoef(torch.stack((a, b)))[0, 1]) if a.std() and b.std() else None,
        "max_abs": float((a - b).abs().max()),
        "relative_l2": float((a - b).norm() / a.norm().clamp_min(1e-12)),
        "finite": bool(b.isfinite().all()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        default="baseline",
        choices=(
            "baseline",
            "runtime",
            "outer",
            "outer_hoist",
            "hillis",
            "kda_scan",
            "kda_scan_fp32",
            "mac",
            "single_silu",
            "packed",
            "ccl_bfp8",
            "ccl_direct",
            "reuse12",
            "chunk128",
            "traced",
            "traced_keepalive",
            "traced_keepalive_mac",
        ),
    )
    parser.add_argument("--program-block-limit", type=int)
    parser.add_argument("--compact-native-output", action="store_true")
    parser.add_argument("--prefill-l1", action="store_true")
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1, help="Allocated slots; one is active")
    parser.add_argument("--active-slot", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--save", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--trace-recurrence", action="store_true")
    parser.add_argument("--oracle", action="store_true", help="Check the first real recurrence chunk against FP64")
    parser.add_argument("--check-silu", action="store_true")
    args = parser.parse_args()
    if args.full and os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        raise ValueError("Full-stack device profiling is intentionally prohibited")
    configure(args.candidate)
    if args.prefill_l1:
        patch_function(
            md.MultichipDecoder,
            "_tp_linear",
            [('if self._multichip_candidate == "multichip_prefill_l1":', "if True: # Isolated prefill L1 experiment")],
        )
    if args.program_block_limit:
        original_program = md._prefill_program

        def block_program(**kwargs):
            kwargs["in0_block_w_limit"] = args.program_block_limit
            return original_program(**kwargs)

        md._prefill_program = block_program
    if args.compact_native_output:
        patch_function(
            md.MultichipDecoder,
            "_linear_attention_native_prefill",
            [
                (
                    "attended = ttnn.reshape(attended, (self.batch * 12, sequence, 1, 128))",
                    "pass # Native head-major output already has the tail's element order",
                )
            ],
        )
    if args.candidate == "chunk128" and fd.LINEAR_PREFILL_CHUNK_SIZE != 128:
        raise ValueError("chunk128 requires QWEN36_LINEAR_PREFILL_CHUNK_SIZE=128 before Python import")
    torch.set_num_threads(8)
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(md.TARGET_FABRIC)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 4),
        trace_region_size=(
            32_000_000
            if args.trace_recurrence or args.candidate.startswith("traced") or args.candidate == "runtime"
            else 0
        ),
    )
    captures, layer_samples = {}, {}
    recurrence_result = {}
    oracle_result = {}
    silu_result = {}
    recurrence_traces = {}
    allow_trace = not args.candidate.startswith("traced_keepalive")
    if args.candidate.startswith("traced"):
        recurrence = md._sequential_recurrence

        def cached_recurrence(*values, **kwargs):
            if not allow_trace:
                return recurrence(*values, **kwargs)
            sources = (*values, kwargs["initial_state"])
            key = tuple((tuple(t.shape), str(t.dtype)) for t in sources)
            if key not in recurrence_traces:
                buffers = [ttnn.clone(t) for t in sources]
                for source, destination in zip(sources, buffers):
                    ttnn.copy(source, destination)
                call_kwargs = {**kwargs, "initial_state": buffers[-1]}
                warm = recurrence(*buffers[:-1], **call_kwargs)
                ttnn.synchronize_device(mesh)
                for value in warm:
                    ttnn.deallocate(value)
                from models.autoports.qwen_qwen3_6_27b.doc.prefill_device_analysis.trace_keepalive import (
                    preserve_trace_tensors,
                )

                manager = (
                    preserve_trace_tensors()
                    if args.candidate.startswith("traced_keepalive")
                    else contextlib.nullcontext([])
                )
                with manager as keepalive:
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    outputs = recurrence(*buffers[:-1], **call_kwargs)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                recurrence_traces[key] = (buffers, outputs, trace, keepalive)
            buffers, outputs, trace, _ = recurrence_traces[key]
            for source, destination in zip(sources, buffers):
                ttnn.copy(source, destination)
            if os.environ.get("PREFILL_TRACE_DEBUG"):
                print(
                    "TRACE_INPUTS",
                    json.dumps([comparison(host_tensor(a), host_tensor(b)) for a, b in zip(sources, buffers)]),
                    flush=True,
                )
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            if os.environ.get("PREFILL_TRACE_DEBUG"):
                ttnn.synchronize_device(mesh)
                actual = [host_tensor(t) for t in outputs]
                expected = recurrence(*values, **kwargs)
                print(
                    "TRACE_DEBUG",
                    json.dumps([comparison(host_tensor(a), b) for a, b in zip(expected, actual)]),
                    flush=True,
                )
            # The existing caller owns/deallocates returned tensors. Do not let
            # that ownership escape into persistent trace buffers.
            return tuple(ttnn.clone(t) for t in outputs)

        md._sequential_recurrence = cached_recurrence
    if args.trace_recurrence:
        recurrence = md._sequential_recurrence

        def compare_trace(*values, **kwargs):
            if recurrence_result:
                return recurrence(*values, **kwargs)
            ttnn.synchronize_device(mesh)
            started = time.perf_counter()
            result = recurrence(*values, **kwargs)
            ttnn.synchronize_device(mesh)
            eager_ms = (time.perf_counter() - started) * 1000
            reference = [host_tensor(v) for v in result]
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            traced_result = recurrence(*values, **kwargs)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            samples = []
            for _ in range(7):
                started = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - started) * 1000)
            recurrence_result.update(
                {
                    "sequence": kwargs["sequence"],
                    "groups": kwargs["groups"],
                    "eager_ms": eager_ms,
                    "trace_ms": samples,
                    "trace_median_ms": statistics.median(samples),
                    "output_comparison": [comparison(a, host_tensor(b)) for a, b in zip(reference, traced_result)],
                }
            )
            ttnn.release_trace(mesh, trace)
            return result

        md._sequential_recurrence = compare_trace
    if args.oracle:
        original_recurrence = md._sequential_recurrence

        def with_oracle(*values, **kwargs):
            if oracle_result:
                return original_recurrence(*values, **kwargs)
            q, k, v, beta, decay = [host_tensor(t).double() for t in values]
            state = host_tensor(kwargs["initial_state"]).double().reshape(kwargs["groups"], 1, 128, 128)
            expected = []
            for step in range(kwargs["sequence"]):
                key = k[:, step : step + 1]
                state = state * decay[:, step : step + 1]
                delta = (v[:, step : step + 1] - key @ state) * beta[:, step : step + 1]
                state = state + key.transpose(-2, -1) @ delta
                expected.append(q[:, step : step + 1] @ state)
            actual = original_recurrence(*values, **kwargs)
            oracle_result.update(
                {
                    "sequence": kwargs["sequence"],
                    "groups": kwargs["groups"],
                    "output": comparison(torch.cat(expected, dim=1), host_tensor(actual[0])),
                    "state": comparison(state, host_tensor(actual[1])),
                    "reference": "FP64 recurrence using the exact TT preprocessed Q/K/V/beta/decay and entry cache",
                }
            )
            return actual

        md._sequential_recurrence = with_oracle
    measuring = False
    try:
        gen = build_generator(
            model_dir=Path(__file__).parents[2],
            mesh_device=mesh,
            batch=args.batch,
            max_context=max(512, args.sequence),
            **({} if args.full else {"layer_indices": [0, 3]}),
        )
        if args.candidate.startswith("kda_scan"):
            kda_recurrence.identity = ttnn.from_torch(
                torch.eye(128).reshape(1, 1, 128, 128),
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                dtype=ttnn.float32 if kda_recurrence.fp32 else ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            kda_recurrence.compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=True,
                math_approx_mode=False,
                packer_l1_acc=True,
            )
        for layer in gen.model.layers:
            if args.candidate in ("ccl_bfp8", "ccl_direct"):
                layer.ccl_token_mixer_dtype = layer.ccl_mlp_dtype = ttnn.bfloat8_b
            if args.candidate == "packed" and layer.layer_kind == "linear_attention":
                layer.weights["packed_prefill"] = ttnn.to_memory_config(
                    layer.weights["linear_packed_decode"], ttnn.DRAM_MEMORY_CONFIG
                )
            if args.check_silu:
                original_linear = layer._tp_linear

                def check_silu(hidden, weight_name, *, _original=original_linear, _index=layer.layer_idx, **kwargs):
                    result = _original(hidden, weight_name, **kwargs)
                    if weight_name == "mlp_gate_prefill" and str(_index) not in silu_result:
                        raw = _original(hidden, weight_name, **{**kwargs, "fused_activation": None})
                        raw_host = host_tensor(raw)
                        expected = torch.nn.functional.silu(raw_host)
                        actual = host_tensor(result)
                        silu_result[str(_index)] = {
                            "vs_single_silu": comparison(expected, actual),
                            "vs_double_silu": comparison(torch.nn.functional.silu(expected), actual),
                            "reference": "Torch SiLU on the identical TT LoFi/BFP4 gate projection with activation disabled",
                        }
                    return result

                layer._tp_linear = check_silu
        prompt = "Explain how to implement a stable merge sort in Python, including its time and memory complexity. "
        rendered = gen.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt * (args.sequence // 12 + 2)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = gen.tokenizer.encode(rendered, add_special_tokens=False)[: args.sequence]
        assert len(ids) == args.sequence
        if not 0 <= args.active_slot < args.batch:
            raise ValueError("active slot is outside allocated batch")
        tokens = torch.zeros(args.batch, args.sequence, dtype=torch.long)
        tokens[args.active_slot] = torch.tensor(ids, dtype=torch.long)
        prompt_lens = [args.sequence if i == args.active_slot else 0 for i in range(args.batch)]
        for layer in gen.model.layers:
            original = layer.prefill_forward
            index = layer.layer_idx

            def timed(*, _original=original, _index=index, **kwargs):
                if measuring:
                    ttnn.synchronize_device(mesh)
                    signpost(f"LAYER_{_index}")
                    started = time.perf_counter()
                value = _original(**kwargs)
                if measuring:
                    ttnn.synchronize_device(mesh)
                    layer_samples.setdefault(str(_index), []).append((time.perf_counter() - started) * 1000)
                    signpost(f"LAYER_{_index}_END")
                if not measuring and _index in (0, 3):
                    captures[f"layer_{_index}"] = host_tensor(value)
                return value

            layer.prefill_forward = timed

        samples = []
        warmups = 2 if args.candidate.startswith("traced_keepalive") else 1
        for iteration in range(args.iterations + warmups):
            gen.reset()
            ttnn.synchronize_device(mesh)
            measuring = iteration >= warmups
            if measuring:
                signpost("PERF_PREFILL")
            started = time.perf_counter()
            logits = gen.prefill_forward(
                tokens, page_table=gen._page_table, kv_cache=gen.kv_cache, prompt_lens=prompt_lens
            )
            ttnn.synchronize_device(mesh)
            elapsed = (time.perf_counter() - started) * 1000
            if measuring:
                signpost("PERF_PREFILL_END")
                samples.append(elapsed)
            else:
                captures["logits"] = logits.float().clone()
                for layer in gen.model.layers:
                    if layer.layer_idx in (0, 3):
                        for name, value in layer.caches.items():
                            if name in ("conv", "recurrent", "key", "value"):
                                captures[f"cache_{layer.layer_idx}_{name}"] = host_tensor(value)
            print(f"PREFILL_ITER {iteration} {elapsed:.3f} ms", flush=True)
            allow_trace = True
        result = {
            "candidate": args.candidate,
            "experiment_overrides": {
                "program_block_limit": args.program_block_limit,
                "compact_native_output": args.compact_native_output,
                "prefill_l1": args.prefill_l1,
                "tail_fusion": os.environ.get("TAIL_FUSION"),
            },
            "all_checked_tensors_finite": all(bool(torch.isfinite(v).all()) for v in captures.values()),
            "sequence": args.sequence,
            "batch": args.batch,
            "active_slot": args.active_slot,
            "linear_chunk_size": fd.LINEAR_PREFILL_CHUNK_SIZE,
            "layer_indices": [layer.layer_idx for layer in gen.model.layers],
            "snapshot": str(fd.default_snapshot()),
            "precision": gen.model.precision_summary(),
            "samples_ms": samples,
            "median_ms": statistics.median(samples),
            "layer_samples_ms": layer_samples,
            "layer_median_ms": {k: statistics.median(v) for k, v in layer_samples.items()},
            "measurement": "warmed eager generator prefill with synchronized per-layer boundaries",
            "recurrence_trace_probe": recurrence_result,
            "recurrence_trace_count": len(recurrence_traces),
            "trace_retained_tensors": sum(len(item[3]) for item in recurrence_traces.values()),
            "recurrence_oracle": oracle_result,
            "silu_oracle": silu_result,
        }
        retained_buffers = {}
        for buffers, outputs, _, keepalive in recurrence_traces.values():
            for value in [*buffers, *outputs, *keepalive]:
                if value.is_allocated():
                    key = (str(value.memory_config().buffer_type), value.buffer_address())
                    pages = math.prod(value.padded_shape) // 1024
                    retained_buffers[key] = max(retained_buffers.get(key, 0), pages * value.buffer_page_size())
        result["trace_tensor_packed_bytes_per_device"] = sum(retained_buffers.values())
        result["trace_unique_buffers"] = len(retained_buffers)
        if args.reference:
            reference = torch.load(args.reference, weights_only=True)
            result["comparison"] = {k: comparison(reference[k], v) for k, v in captures.items()}
            result["logits_top1_equal"] = bool(
                torch.equal(reference["logits"].argmax(-1), captures["logits"].argmax(-1))
            )
        if args.save:
            torch.save(captures, args.save)
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(result, indent=2) + "\n")
        print("PREFILL_RESULT", json.dumps({k: v for k, v in result.items() if k != "precision"}), flush=True)
        if not result["all_checked_tensors_finite"]:
            raise AssertionError("Non-finite checked layer/logit/cache tensors; candidate is invalid")
    finally:
        for _, _, trace, _ in recurrence_traces.values():
            ttnn.release_trace(mesh, trace)
        if "gen" in locals():
            gen.teardown()
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
