# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic standalone TT logits with the saved HTTP request's host sampler.

Run under an outer 1800-second timeout. Atomic per-token snapshots preserve the
last completed token and pending operation if a device call hangs. This is an
explicit optional host-sampling control, never a production/performance path.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / "readiness_vllm/sampling_controls/http_bad_words_card_1024.json"
PRECISION = ROOT / "doc/datatype_sweep/selected_precision_config.json"


def source_identity():
    paths = [
        *sorted((ROOT / "tt").glob("*.py")),
        PRECISION,
        Path(__file__),
        ROOT / "tests/run_vllm_reasoning_budget.py",
    ]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def compare_tokens(expected, actual):
    prefix = 0
    for left, right in zip(expected, actual):
        if left != right:
            break
        prefix += 1
    exact = expected == actual
    return dict(
        status="EXACT" if exact else "DIFFERENT",
        exact_prefix_tokens=prefix,
        expected_tokens=len(expected),
        actual_tokens=len(actual),
        first_divergence=(
            None
            if exact
            else dict(
                index=prefix,
                http_token=expected[prefix] if prefix < len(expected) else None,
                native_token=actual[prefix] if prefix < len(actual) else None,
            )
        ),
    )


def boundary_snapshot(scores, history, seed, index, reference):
    """Compact candidates for an identical-prefix HF numerical discriminator."""
    import torch
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

    raw = scores.float().clone()
    values = raw.clone()
    values[:, list(set(history))] -= reference["presence_penalty"]
    bans = reference["bad_words_token_ids"]
    assert all(len(tokens) == 1 for tokens in bans)
    values[:, [tokens[0] for tokens in bans]] = -float("inf")
    values /= reference["generation"]["temperature"]
    processed = apply_top_k_top_p(
        values,
        torch.tensor([reference["generation"]["top_k"]], dtype=torch.int32),
        torch.tensor([reference["generation"]["top_p"]]),
    )
    kept = torch.nonzero(torch.isfinite(processed[0])).flatten().tolist()
    raw_ids = sorted(set(torch.topk(raw[0], 128).indices.tolist()) | set(kept))
    return dict(
        seed=seed,
        index=index,
        history=list(history),
        raw_top_logit_values={str(token): float(raw[0, token]) for token in raw_ids},
        processed_top_logit_values={str(token): float(processed[0, token]) for token in kept},
    )


def collect(args, reference, result, save):
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    # The shared CPU sampler does not need automatic vLLM platform plugins.
    os.environ["VLLM_PLUGINS"] = ""
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_vllm_reasoning_budget import shared_host_processor
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel

    torch.set_num_threads(8)
    prompt = reference["prompt_tokens"]
    generation = reference["generation"]
    mesh = generator = None
    try:
        result["pending_operation"] = "model_setup"
        save()
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=805306368)
        model = QwenModel(mesh, checkpoint=args.checkpoint, max_seq_len=2048, precision_config=PRECISION)
        assert list(model.layer_indices) == list(range(40)), "Control requires all 40 layers"
        generator = QwenGenerator(model, max_batch_size=32, decode_buckets=(2, 32))
        generation_config_path = model.checkpoint.path / "generation_config.json"
        generation_config = json.loads(generation_config_path.read_text())
        eos = generation_config.get("eos_token_id", model.config.eos_token_id)
        eos_ids = set(eos if isinstance(eos, list) else [eos])
        eos_ids.discard(None)
        assert eos_ids, "Checkpoint must declare its generation EOS tokens"
        assert all(0 <= token < model.vocab_size for token in prompt)
        result.update(
            checkpoint=str(model.checkpoint.path),
            generation_config=generation_config,
            generation_config_sha256=hashlib.sha256(generation_config_path.read_bytes()).hexdigest(),
            eos_token_ids=sorted(eos_ids),
            precision=model.precision_config,
            layers=list(model.layer_indices),
            vocabulary_size=model.vocab_size,
        )
        width = next(states for states in generator.cache if "page_table" in states[0])[0]["page_table"].shape[-1]
        # Only slot 0 is live; row 1 is an inactive physical bucket placeholder.
        pages = torch.arange(width, dtype=torch.int32).repeat(2, 1)
        result["page_table"] = pages.tolist()

        def prefill():
            logits = generator.prefill_forward(
                torch.tensor([prompt], dtype=torch.int64),
                prompt_lens=[len(prompt)],
                start_pos=[0],
                slots=(0,),
                page_table=pages[:1],
            )
            assert logits.shape == (1, 1, model.vocab_size)
            return logits[:, 0].clone()

        # Match the existing logit diagnostic: warm host prefill before capture.
        result["pending_operation"] = "warm_prefill_and_prepare"
        save()
        prefill()
        generator.prepare()
        traces = dict(generator.traces)
        assert set(traces) == {"model_2", "model_fixed", "sampling"}
        assert generator.counters["captures"] == 3
        result["trace_ids"] = {key: str(value) for key, value in traces.items()}
        records = {row["seed"]: row for row in reference["outputs"]}

        for seed in args.seeds:
            expected_choice = records[seed]["response"]["choices"][0]
            expected = expected_choice["token_ids"]
            row = dict(
                seed=seed,
                collection_status="RUNNING",
                comparison_status="PENDING",
                tokens=[],
                boundaries=[],
                text="",
                http_tokens=expected,
                http_finish_reason=expected_choice["finish_reason"],
                http_response_id=records[seed]["response"]["id"],
            )
            result["outputs"].append(row)
            started = time.monotonic()
            result["pending_operation"] = f"seed={seed}:reset_and_prefill"
            save()
            generator.reset()
            scores = prefill()
            processor = shared_host_processor(
                prompt,
                seed,
                generation["temperature"],
                generation["top_p"],
                reference["presence_penalty"],
                reference["bad_words_token_ids"],
            )
            sampler_path = Path(sys.modules["vllm_tt_plugin.host_sampler"].__file__)
            result["host_sampler_source"] = dict(
                path=str(sampler_path), sha256=hashlib.sha256(sampler_path.read_bytes()).hexdigest()
            )
            before = dict(generator.counters)
            replays_before = dict(generator.trace_replays)
            for step in range(reference["max_tokens"]):
                assert scores.shape == (1, model.vocab_size) and torch.isfinite(scores).all()
                history = torch.tensor([prompt + row["tokens"]], dtype=torch.int64)
                boundary = (
                    boundary_snapshot(scores, row["tokens"], seed, step, reference) if step in (0, 16, 44) else None
                )
                with torch.inference_mode():
                    selected = processor(history, scores)
                token = int(torch.argmax(selected, dim=-1).item())
                if boundary is not None:
                    boundary["recorded_token"] = token
                    row["boundaries"].append(boundary)
                row["tokens"].append(token)
                row["text"] = generator.tokenizer.decode(row["tokens"], skip_special_tokens=False)
                row["elapsed_s"] = time.monotonic() - started
                row["counters"] = dict(generator.counters)
                row["trace_replays"] = dict(generator.trace_replays)
                finished = token in eos_ids or step + 1 == reference["max_tokens"]
                result["pending_operation"] = f"seed={seed}:" + (
                    "compare" if finished else f"decode_after_token={step}"
                )
                save()
                if step == 0 or (step + 1) % 128 == 0 or finished:
                    print(f"NATIVE_HOST seed={seed} tokens={step + 1} eos={token in eos_ids}", flush=True)
                if finished:
                    row["finish_reason"] = "stop" if token in eos_ids else "length"
                    break
                tokens = torch.tensor([[token], [0]], dtype=torch.int32)
                handle = generator.decode_forward_v1(
                    tokens,
                    [len(prompt) + step, -1],
                    kv_cache=generator.cache,
                    page_table=pages,
                    sampling_params=None,
                    reload_inputs=True,
                    read_from_device=False,
                    enable_trace=True,
                )
                returned, events = generator.read_decode_output(handle, async_read=True)
                assert returned is handle and events
                output = generator.process_decode_output_host(returned, is_tokens=False)
                assert output.shape == (2, 1, model.vocab_size)
                scores = output[:1, 0].clone()
                assert generator.counters["sampling_replays"] == before["sampling_replays"]
                assert generator.traces == traces and generator.counters["captures"] == 3
                assert generator.counters["releases"] == 0
            decode_count = len(row["tokens"]) - 1
            assert generator.trace_replays.get("model_2", 0) - replays_before.get("model_2", 0) == decode_count
            assert generator.counters["model_replays"] - before["model_replays"] == decode_count
            row["comparison"] = compare_tokens(expected, row["tokens"])
            row["comparison_status"] = row["comparison"]["status"]
            row["collection_status"] = "PASS"
            row["counter_delta"] = {key: value - before[key] for key, value in generator.counters.items()}
            save()
            print(f"NATIVE_HOST seed={seed} comparison={row['comparison']}", flush=True)
        result["counters_before_shutdown"] = dict(generator.counters)
    finally:
        result["pending_operation"] = "shutdown"
        save()
        try:
            if generator is not None:
                generator.close()
                result["counters_after_shutdown"] = dict(generator.counters)
                assert not generator.traces and generator.counters["captures"] == generator.counters["releases"]
        finally:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Use a fresh artifact path"
    reference = json.loads(args.reference.read_text())
    assert reference["status"] == "PASS" and reference["backend"] == "http"
    assert reference["case"] == "bad_words" and reference["generation"]["top_k"] == 20
    assert reference["structured_outputs"] is None
    assert len(reference["prompt_tokens"]) + reference["max_tokens"] <= 2048
    records = {row["seed"]: row for row in reference["outputs"]}
    assert len(args.seeds) == len(set(args.seeds)) and all(seed in records for seed in args.seeds)
    for seed in args.seeds:
        body = records[seed]["response"]
        assert body["prompt_token_ids"] == reference["prompt_tokens"]
        assert len(body["choices"]) == 1 and body["choices"][0]["token_ids"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        collection_status="RUNNING",
        comparison_status="PENDING",
        diagnostic_only=True,
        performance_measurement=False,
        backend="native_tt_shared_host_sampler",
        command=[sys.executable, *sys.argv],
        required_outer_timeout_s=1800,
        reference=str(args.reference.resolve()),
        source_sha256=source_identity(),
        request={key: value for key, value in reference.items() if key not in ("outputs", "status", "rubric")},
        seeds=args.seeds,
        mesh=[1, 4],
        layers=list(range(40)),
        capacity=32,
        decode_buckets=[2, 32],
        active_slots=[0],
        standalone_context=2048,
        trace_region_size=805306368,
        limitations=[
            "Standalone cache allocation and request scheduling differ from the HTTP server.",
            "Host sampling is diagnostic-only; elapsed times are not performance measurements.",
            "Collection success is separate from token agreement and does not establish model quality.",
            "Source hashes identify this checkout; saved HTTP provenance must be checked separately.",
        ],
        outputs=[],
    )

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    try:
        collect(args, reference, result, save)
        assert source_identity() == result["source_sha256"], "Source changed during control"
        result["collection_status"] = "PASS"
        result["comparison_status"] = (
            "EXACT" if all(row["comparison_status"] == "EXACT" for row in result["outputs"]) else "DIFFERENT"
        )
        result["pending_operation"] = None
    except BaseException as error:
        result.update(collection_status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        save()
    print("NATIVE_HOST_CONTROL", result["collection_status"], result["comparison_status"], args.output, flush=True)
    # Finish and persist every requested seed before reporting a comparison failure.
    if result["comparison_status"] != "EXACT":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
