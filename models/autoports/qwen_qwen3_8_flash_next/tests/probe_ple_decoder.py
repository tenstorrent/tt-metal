# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Complete real-weight layer-1 PLE decoder probe with resident packed lookup.

HF reads only selected rows of its exact real checkpoint table, while TT keeps
all 128 shards resident in the selected packed format. TT hashes tokens and performs lookup
inside every prefill/decode pass. No token-derived embedding is supplied to TT.
The four persistent request states are snapshotted only at the explicit test
boundary, then restored before warmup/capture validation and repeat replay.
"""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_quantization import SelectedRows
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import PackedEmbedding


def streaming_pcc(expected, actual, *, residual=None, threshold=0.995, chunk_rows=32, worst_count=10):
    """FP64 PCC with bounded scratch; every logical row participates in the gate.

    Global centered moments are merged using the parallel-variance formula,
    avoiding cancellation in sum(x*x)-sum(x)**2/n. Constant equal rows score1;
    unequal constants score0. Nonfinite data is an explicit failed row and is
    serialized as null, never nonstandard JSON NaN/Infinity.
    """
    width = expected.shape[-1]
    target = expected.reshape(-1, width)
    observed = actual.reshape_as(target)
    residual_rows = residual.reshape_as(target) if residual is not None else None
    total = 0
    mean_x = mean_y = sum_xx = sum_yy = sum_xy = 0.0
    minimum = 1.0
    failed = nonfinite = 0
    worst = []
    max_error = 0.0
    for start in range(0, len(target), chunk_rows):
        x = target[start : start + chunk_rows].double()
        y = observed[start : start + chunk_rows].double()
        if residual_rows is not None:
            r = residual_rows[start : start + chunk_rows].double()
            x = x - r
            y = y - r
        finite = torch.isfinite(x).all(-1) & torch.isfinite(y).all(-1)
        centered_x = x - x.mean(-1, keepdim=True)
        centered_y = y - y.mean(-1, keepdim=True)
        denominator = (centered_x.square().sum(-1) * centered_y.square().sum(-1)).sqrt()
        values = (centered_x * centered_y).sum(-1) / denominator
        constants = denominator == 0
        values = torch.where(constants, (x == y).all(-1).double(), values).clamp(-1, 1)
        for offset, (value, is_finite) in enumerate(zip(values.tolist(), finite.tolist())):
            valid = is_finite and math.isfinite(value)
            score = value if valid else None
            nonfinite += not valid
            failed += not valid or value < threshold
            if valid:
                minimum = min(minimum, value)
            worst.append({"row": start + offset, "pcc": score})
        worst.sort(key=lambda row: row["pcc"] if row["pcc"] is not None else -math.inf)
        del worst[worst_count:]
        if not bool(finite.all()):
            continue
        max_error = max(max_error, (x - y).abs().max().item())
        count = x.numel()
        block_x, block_y = x.mean().item(), y.mean().item()
        dx, dy = block_x - mean_x, block_y - mean_y
        factor = total * count / (total + count)
        xc, yc = x - block_x, y - block_y
        sum_xx += xc.square().sum().item() + dx * dx * factor
        sum_yy += yc.square().sum().item() + dy * dy * factor
        sum_xy += (xc * yc).sum().item() + dx * dy * factor
        mean_x += dx * count / (total + count)
        mean_y += dy * count / (total + count)
        total += count
    if nonfinite:
        global_pcc = None
    elif sum_xx == 0 or sum_yy == 0:
        global_pcc = 1.0 if max_error == 0 else 0.0
    else:
        global_pcc = max(-1.0, min(1.0, sum_xy / math.sqrt(sum_xx * sum_yy)))
    return {
        "global_pcc": global_pcc,
        "minimum_row_pcc": minimum if not nonfinite else None,
        "failed_row_count": failed,
        "nonfinite_row_count": nonfinite,
        "rows": len(target),
        "worst_rows": worst,
        "max_abs_error": max_error if not nonfinite else None,
        "passed": failed == 0 and global_pcc is not None and global_pcc >= threshold,
    }


def metrics(expected, actual, residual):
    actual = actual.reshape(expected.shape)
    whole = streaming_pcc(expected, actual)
    contribution = streaming_pcc(expected, actual, residual=residual)
    return {
        "pcc": whole["global_pcc"],
        "contribution_pcc": contribution["global_pcc"],
        "finite": whole["nonfinite_row_count"] == 0 and contribution["nonfinite_row_count"] == 0,
        "max_abs_error": whole["max_abs_error"],
        "whole": whole,
        "contribution": contribution,
        "passed": whole["passed"] and contribution["passed"],
    }


def load_reference(snapshot, config):
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "model.language_model.layers.1."
    keys = [key for key in index if key.startswith(prefix) and ".ngram_embedding.shard_" not in key]
    weights = {}
    for filename in sorted({index[key] for key in keys}):
        with safe_open(snapshot / filename, framework="pt") as f:
            for key in keys:
                if index[key] == filename:
                    weights[key.removeprefix(prefix)] = f.get_tensor(key)
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, 1)
    result = reference.load_state_dict(weights, strict=False, assign=True)
    expected_missing = ["ple.ple_embedding.ngram_embedding.weight"]
    if result.missing_keys != expected_missing or result.unexpected_keys:
        raise ValueError(f"Unexpected HF weight contract: {result}")
    reference.ple.ple_embedding.ngram_embedding = SelectedRows(
        snapshot, index, prefix + "ple.ple_embedding.ngram_embedding.", "absmax"
    )
    reference.eval()
    return reference, weights


def token_rows(tokens):
    """Explicit public input encoding, not token-derived model computation."""
    result = torch.zeros(1, tokens.shape[1], 16, dtype=torch.int64)
    result[..., 0] = tokens
    result[..., 1] = 1
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/ttuser/qwen38-flash/packed_ple_nf4_absmax"))
    parser.add_argument("--seq-len", type=int, default=33)
    parser.add_argument("--decode-steps", type=int, choices=(0, 1, 3), default=3)
    parser.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b"), default="bfloat16")
    parser.add_argument(
        "--prepare-programs",
        action="store_true",
        help="Warm exact small prefill/tail/decode signatures before allocating full prompt, restoring all request states",
    )
    parser.add_argument(
        "--prepare-all-tails",
        action="store_true",
        help="Prepare physical prefill32 then1..31 and decode1 before full prompt allocation; implies --prepare-programs and requires seq-len>=32",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device-output-cache",
        type=Path,
        help="Optional diagnostic export of existing host readbacks/fixture/source identity after all runtime passes; no metric or gate changes",
    )
    args = parser.parse_args()
    if args.prepare_all_tails:
        args.prepare_programs = True
        if args.seq_len < 32:
            parser.error("--prepare-all-tails requires --seq-len >=32")
    if args.seq_len < 1:
        parser.error("--seq-len must be positive")
    torch.set_num_threads(16)
    torch.manual_seed(1731)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    if args.seq_len + args.decode_steps > config.max_position_embeddings:
        parser.error("Prefill plus requested decode steps exceeds the advertised context")
    reference, weights = load_reference(args.snapshot, config)
    prompt = (torch.randn(1, args.seq_len, 10240) * 0.02).bfloat16()
    # Always consume the original three decode RNG draws and ID extent before
    # slicing requested work, preserving the existing CPU fixture semantics.
    decode_inputs = [(torch.randn(1, 1, 10240) * 0.02).bfloat16() for _ in range(3)]
    ids = torch.randint(0, config.vocab_size, (1, args.seq_len + 3))
    eos = config.eos_token_id[0] if isinstance(config.eos_token_id, list) else config.eos_token_id
    # Resume across an EOS boundary, and introduce another EOS during decode.
    ids[0, max(args.seq_len - 2, 0)] = eos
    ids[0, args.seq_len + 1] = eos
    decode_inputs = decode_inputs[: args.decode_steps]
    cache = DynamicCache(config=config)
    with torch.inference_mode():
        expected_prefill = reference(
            prompt, position_embeddings=(None, None), past_key_values=cache, ple_input_ids=ids[:, : args.seq_len]
        )
        expected_decode = [
            reference(
                x,
                position_embeddings=(None, None),
                past_key_values=cache,
                ple_input_ids=ids[:, args.seq_len + i : args.seq_len + i + 1],
            )
            for i, x in enumerate(decode_inputs)
        ]
    print("PLE_DECODER_HF_REFERENCE_READY", flush=True)
    report = {
        "scope": "complete real-weight layer1 decoder, device hash and fully resident packed PLE lookup",
        "snapshot": str(args.snapshot),
        "cache_dir": str(args.cache_dir),
        "layer_idx": 1,
        "seq_len": args.seq_len,
        "decode_steps": args.decode_steps,
        "final_logical_context": args.seq_len + args.decode_steps,
        "decode_positions": list(range(args.seq_len, args.seq_len + args.decode_steps)),
        "weights": "real",
        "decode": [],
        "runtime_boundary": "device hidden states and raw token IDs; no prepared embeddings",
        "threshold": 0.995,
        "expert_dtype": args.expert_dtype,
        "prepare_programs": args.prepare_programs,
        "prepare_all_tails": args.prepare_all_tails,
        "metric_contract": "FP64 global and every-row whole/contribution PCC>=0.995; scratch32rows; worst10rows",
        "runtime_audit": {"prefill": False, "repeated_prefill": False, "decode_warm_capture_replay": None},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace_id = None
    try:
        packed = PackedEmbedding.from_checkpoint(args.snapshot, cache_dir=args.cache_dir, mesh_device=mesh)
        model = FunctionalDecoder.from_state_dict(
            weights,
            hf_config=config,
            layer_idx=1,
            mesh_device=mesh,
            packed_embedding=packed,
            expert_dtype=getattr(ttnn, args.expert_dtype),
        )
        buckets = (
            tuple(range(1, 33)) if args.prepare_all_tails else tuple(sorted({1, 7, 31, 32, args.seq_len % 32 or 32}))
        )
        report["hash_lengths"] = list(buckets)
        state = model.create_state(hash_lengths=buckets)

        def host(value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout)

        def device(value, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.to_device(host(value, dtype=dtype, layout=layout), mesh)

        state_names = ("conv", "recurrent", "ple_conv", "token_history")

        def snapshot_state():
            return {
                name: host(
                    ttnn.to_torch(getattr(state, name)),
                    dtype=getattr(state, name).dtype,
                    layout=getattr(state, name).layout,
                )
                for name in state_names
            }

        def restore_state(snapshot):
            for name in state_names:
                ttnn.copy_host_to_device_tensor(snapshot[name], getattr(state, name))

        initial_state = snapshot_state()

        if args.prepare_programs:

            def memory():
                view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
                return {
                    "banks": view.num_banks,
                    "free_bytes": view.num_banks * view.total_bytes_free_per_bank,
                    "largest_contiguous_bytes_per_bank": view.largest_contiguous_bytes_free_per_bank,
                }

            preparation = {
                "scope": "Setup-only program preparation before resident full-prompt allocation; full public prefill, decode and repeated-prefill gates remain unchanged",
                "before": memory(),
                "shape_profile": "all32 physical prefill lengths"
                if args.prepare_all_tails
                else "requested prefill/tail lengths",
                "signatures": [],
                "runtime_audit": True,
                "state_restored": False,
            }
            report["program_preparation"] = preparation

            def warm_signature(phase, start, count):
                warm_hidden = warm_ids = warm_output = None
                try:
                    restore_state(initial_state)
                    warm_hidden = device(prompt[:, start : start + count].unsqueeze(0))
                    warm_ids = device(
                        token_rows(ids[:, start : start + count]), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                    with runtime_audit():
                        operation = model.prefill_forward if phase == "prefill" else model.decode_forward
                        warm_output = operation(warm_hidden, state=state, token_ids=warm_ids)
                    ttnn.synchronize_device(mesh)
                    preparation["signatures"].append(
                        {
                            "phase": phase,
                            "tokens": count,
                            "prompt_start": start,
                            "hidden_shape": [1, 1, count, 10240],
                            "token_shape": [1, count, 16],
                        }
                    )
                finally:
                    ttnn.synchronize_device(mesh)
                    for tensor in (warm_output, warm_ids, warm_hidden):
                        if tensor is not None:
                            ttnn.deallocate(tensor)
                    restore_state(initial_state)
                    ttnn.synchronize_device(mesh)

            # Reuse existing real prompt data; do not consume any RNG draws.
            warm_signature("prefill", 0, min(args.seq_len, 32))
            tail = args.seq_len % 32
            if args.prepare_all_tails:
                for count in range(1, 32):
                    warm_signature("prefill", 0, count)
            elif args.seq_len > 32 and tail:
                warm_signature("prefill", args.seq_len - tail, tail)
            warm_signature("decode", 0, 1)
            preparation["state_restored"] = all(
                torch.equal(ttnn.to_torch(getattr(state, name)), ttnn.to_torch(initial_state[name]))
                for name in state_names
            )
            assert preparation["state_restored"], "Program preparation must restore every request state"
            preparation["after"] = memory()
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print("PLE_PROGRAM_PREPARATION", json.dumps(preparation), flush=True)

        prompt_device = device(prompt.unsqueeze(0))
        prompt_ids_device = device(token_rows(ids[:, : args.seq_len]), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        decode_device = device(decode_inputs[0].unsqueeze(0)) if args.decode_steps else None
        decode_ids_device = (
            device(token_rows(ids[:, args.seq_len : args.seq_len + 1]), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            if args.decode_steps
            else None
        )
        refresh_hidden = [host(x.unsqueeze(0)) for x in decode_inputs]
        refresh_tokens = [
            host(
                token_rows(ids[:, args.seq_len + i : args.seq_len + i + 1]),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            for i in range(args.decode_steps)
        ]
        with runtime_audit():
            prefill = model.prefill_forward(prompt_device, state=state, token_ids=prompt_ids_device)
        report["runtime_audit"]["prefill"] = True
        ttnn.synchronize_device(mesh)
        actual_prefill = ttnn.to_torch(prefill)
        report["prefill"] = metrics(expected_prefill, actual_prefill, prompt)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("PLE_PREFILL", json.dumps(report["prefill"]), flush=True)
        ttnn.deallocate(prefill)
        if args.decode_steps:
            prefix_state = snapshot_state()
            # Compile the exact decode signature before capture, restoring every
            # request state, including both lexical-token history and PLE conv tail.
            restore_state(prefix_state)
            with runtime_audit():
                warmed = model.decode_forward(decode_device, state=state, token_ids=decode_ids_device)
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(warmed)
            restore_state(prefix_state)
            trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
            with runtime_audit():
                output = model.decode_forward(decode_device, state=state, token_ids=decode_ids_device)
            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
            # Only this exact output is exempt. Its only consumer is synchronous
            # readback before replay; no input/state/program cache is acknowledged.
            trace_allocation_tracker.acknowledge_corruptible(output)
            replay_outputs = []
            replay_states = []
            for repetition in range(2):
                # Capture itself mutates all four states: restore again after it.
                restore_state(prefix_state)
                round_outputs = []
                for step in range(args.decode_steps):
                    ttnn.copy_host_to_device_tensor(refresh_hidden[step], decode_device)
                    ttnn.copy_host_to_device_tensor(refresh_tokens[step], decode_ids_device)
                    with runtime_audit():
                        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                    actual = ttnn.to_torch(output).reshape(expected_decode[step].shape)
                    round_outputs.append(actual.clone())
                    if repetition == 0:
                        report["decode"].append(
                            dict(step=step, **metrics(expected_decode[step], actual, decode_inputs[step]))
                        )
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print("PLE_DECODE", json.dumps(report["decode"][-1]), flush=True)
                replay_outputs.append(round_outputs)
                replay_states.append({name: ttnn.to_torch(getattr(state, name)).clone() for name in state_names})
            report["trace_id"] = str(trace_id)
            report["captures"] = 1
            report["replays"] = 2 * args.decode_steps
            report["runtime_audit"]["decode_warm_capture_replay"] = True
            report["reset_replay_equal"] = all(torch.equal(a, b) for a, b in zip(*replay_outputs))
            report["reset_state_equal"] = {
                name: torch.equal(replay_states[0][name], replay_states[1][name]) for name in state_names
            }
            ttnn.release_trace(mesh, trace_id)
            trace_id = None
            # Readbacks above own the evidence. Release the unused trace output
            # before the repeated capacity-bound prefill, matching its first pass.
            ttnn.deallocate(output)
        else:
            report["captures"] = 0
            report["replays"] = 0
            report["reset_replay_equal"] = None
            report["reset_state_equal"] = {}
        # This is a new request on the same instance, not prefix continuation.
        restore_state(initial_state)
        with runtime_audit():
            repeated = model.prefill_forward(prompt_device, state=state, token_ids=prompt_ids_device)
        report["runtime_audit"]["repeated_prefill"] = True
        report["prefill_reset_equal"] = torch.equal(actual_prefill, ttnn.to_torch(repeated))
        all_metrics = [report["prefill"], *report["decode"]]
        report["passed"] = (
            all(row["passed"] for row in all_metrics)
            and (not args.decode_steps or report["reset_replay_equal"])
            and all(report["reset_state_equal"].values())
            and report["prefill_reset_equal"]
        )
        if args.device_output_cache is not None:
            # All measured/audited calls are complete. Reuse existing host
            # readbacks; this export cannot alter the runtime allocation path.
            runtime_root = Path(__file__).parents[1] / "tt"
            identity = {
                "seed": 1731,
                "seq_len": args.seq_len,
                "decode_steps": args.decode_steps,
                "expert_dtype": args.expert_dtype,
                "snapshot": str(args.snapshot),
                "config_sha256": hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256((args.cache_dir / "manifest.json").read_bytes()).hexdigest(),
                "hf_source_sha256": hashlib.sha256(
                    Path(inspect.getfile(Qwen4ExpTextDecoderLayer)).read_bytes()
                ).hexdigest(),
                "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "runtime_sources": {
                    str(path.relative_to(runtime_root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(runtime_root.rglob("*"))
                    if path.suffix in (".py", ".cpp", ".hpp")
                },
            }
            args.device_output_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "identity": identity,
                    "prompt": prompt,
                    "ids": ids,
                    "decode_inputs": decode_inputs,
                    "expected_prefill": expected_prefill,
                    "actual_prefill": actual_prefill,
                    "expected_decode": expected_decode,
                    "actual_decode": replay_outputs[0] if args.decode_steps else [],
                    "report": report,
                },
                args.device_output_cache,
            )
            report["device_output_cache"] = {
                "path": str(args.device_output_cache),
                "identity": identity,
                "scope": "Existing host readbacks exported after all measured calls; literal failure retained",
            }
    except Exception as exc:
        report["error"] = str(exc).split("backtrace:")[0]
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)
    print(json.dumps(report), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
