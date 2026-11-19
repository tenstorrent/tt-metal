# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""All-layer logit diagnostic: standalone baseline, then attach to real vLLM.

Raw repeated-token fixtures stress serving mechanics, not qualitative quality.
Host argmax/full-logit reads here are explicit compatibility diagnostics only.
The HTTP logprobs requests deliberately use the plugin's host-logit route.
Neither mode changes the canonical on-device sampler or measures performance.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
FIXTURES = {f"raw{length}": [9707 + variant] * length for variant, length in enumerate((129, 130, 131))}
TOP_N = 20
ATOL = 1e-5


def source_identity():
    paths = [*sorted((ROOT / "tt").glob("*.py")), ROOT / "doc/datatype_sweep/selected_precision_config.json"]
    identity = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    native = ROOT.parents[2] / "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_fused_swiglu"
    identity.update(
        {
            "repo:" + str(path.relative_to(ROOT.parents[2])): hashlib.sha256(path.read_bytes()).hexdigest()
            for family in (native, native.parent / "post_combine_reduce")
            for path in sorted(family.rglob("*"))
            if path.suffix in (".cpp", ".hpp")
        }
    )
    return identity


def compact_logits(logits):
    import torch

    assert logits.ndim == 1 and torch.isfinite(logits).all(), "Expected finite full-vocabulary logits"
    scores = torch.log_softmax(logits.float(), dim=-1)
    top_scores, top_ids = torch.topk(scores, TOP_N)
    cutoff = float(top_scores[-1])
    # A tie crossing rank20 permits different tied IDs in a correct top20 list.
    # Retain every boundary tie so this never relaxes score or higher-rank checks.
    eligible = torch.nonzero(scores >= cutoff).reshape(-1).tolist()
    greedy = int(torch.argmax(logits))
    return dict(
        token_id=greedy,
        token_logprob=float(scores[greedy]),
        top_logprobs={str(token): float(score) for token, score in zip(top_ids.tolist(), top_scores.tolist())},
        eligible_top_logprobs={str(token): float(scores[token]) for token in eligible},
        cutoff_logprob=cutoff,
        logits_sha256=hashlib.sha256(logits.float().contiguous().numpy().tobytes()).hexdigest(),
        vocabulary_size=logits.numel(),
    )


def token_id(value):
    if not isinstance(value, str) or not value.startswith("token_id:"):
        raise AssertionError(f"Expected unambiguous token_id:NNN key, got {value!r}")
    suffix = value.removeprefix("token_id:")
    if not suffix.isdigit():
        raise AssertionError(f"Malformed token id {value!r}")
    return int(suffix)


def response_steps(body):
    choices = body.get("choices", [])
    assert len(choices) == 1, "Expected exactly one completion"
    choice = choices[0]
    logprobs = choice.get("logprobs") or {}
    tokens, values, tops = (logprobs.get(key) for key in ("tokens", "token_logprobs", "top_logprobs"))
    assert all(
        isinstance(value, list) and len(value) == 2 for value in (tokens, values, tops)
    ), "Missing two-token logprobs"
    steps = []
    for token, value, top in zip(tokens, values, tops):
        assert isinstance(top, dict) and len(top) == TOP_N, "Expected exactly20 top-logprob entries"
        assert value is not None and math.isfinite(value), "Invalid emitted-token logprob"
        parsed = {str(token_id(key)): score for key, score in top.items()}
        assert len(parsed) == TOP_N and all(math.isfinite(score) for score in parsed.values())
        steps.append(dict(token_id=token_id(token), token_logprob=value, top_logprobs=parsed))
    if choice.get("token_ids") is not None:
        assert choice["token_ids"] == [step["token_id"] for step in steps], "Token fields disagree"
    assert choice.get("finish_reason") == "length", choice.get("finish_reason")
    return steps


def compare_steps(expected, actual, label):
    assert len(expected) == len(actual) == 2, label
    maximum = 0.0
    for position, (left, right) in enumerate(zip(expected, actual)):
        prefix = f"{label}.token{position}"
        assert left["token_id"] == right["token_id"], (prefix, left["token_id"], right["token_id"])
        delta = abs(left["token_logprob"] - right["token_logprob"])
        assert delta <= ATOL, (prefix, "emitted logprob", delta)
        maximum = max(maximum, delta)
        eligible = left.get("eligible_top_logprobs", left["top_logprobs"])
        actual_ids = set(right["top_logprobs"])
        assert actual_ids <= set(eligible), (prefix, "unexpected top20 IDs", sorted(actual_ids - set(eligible)))
        if "cutoff_logprob" in left:
            required = {key for key, value in eligible.items() if value > left["cutoff_logprob"]}
            assert required <= actual_ids, (prefix, "higher-ranked token omitted", sorted(required - actual_ids))
        else:
            assert set(eligible) == actual_ids, (prefix, "top20 IDs changed between HTTP requests")
        for key, value in right["top_logprobs"].items():
            delta = abs(eligible[key] - value)
            assert math.isfinite(delta) and delta <= ATOL, (prefix, key, delta)
            maximum = max(maximum, delta)
    return maximum


def standalone(args, result, save):
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    assert os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] == "1"
    assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenGenerator
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenModel

    torch.set_num_threads(8)
    mesh = generator = None
    full_reference = {}
    external_reference = None
    if args.compare_full_logits:
        external_reference = torch.load(args.compare_full_logits, map_location="cpu", weights_only=True)
        assert set(external_reference) == set(FIXTURES)
        result["full_logits_reference_sha256"] = hashlib.sha256(args.compare_full_logits.read_bytes()).hexdigest()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        model = QwenModel(
            mesh,
            checkpoint=args.checkpoint,
            max_seq_len=1024,
            suppress_inactive_moe_routes=args.route_mask,
            specialize_solo_slot0_attention=args.solo_attention,
            **({"fuse_hidden_row_mask": True} if args.hidden_row_mask else {}),
            **({"packed_postreduce_weights": True} if args.packed_postreduce_weights else {}),
            **({"native_qk_reader": True} if args.native_qk_reader else {}),
            **({"prune_postreduce_padding": True} if args.prune_postreduce_padding else {}),
            **({"fuse_recurrent_state_commit": True} if args.conditional_state_commit else {}),
            **({"fuse_moe_route_plan": True} if args.route_plan else {}),
            **({"ep_weight_nd_sharding": True} if args.ep_weight_nd_sharding else {}),
        )
        assert list(model.layer_indices) == list(range(40)), "This control requires all40 layers"
        assert all(0 <= token < model.vocab_size for tokens in FIXTURES.values() for token in tokens)
        buckets = (2, 4, 32) if args.four_row_bucket else (2, 32)
        generator = QwenGenerator(model, max_batch_size=32, decode_buckets=buckets)
        result["decode_buckets"] = list(buckets)
        result.update(
            precision=model.precision_config, layers=list(model.layer_indices), vocabulary_size=model.vocab_size
        )
        width = next(group for group in generator.cache if "page_table" in group[0])[0]["page_table"].shape[-1]
        pages = torch.zeros((32, width), dtype=torch.int32)
        for row in range(3):
            pages[row] = torch.arange(width).roll(-row * (width // 3))
        assert math.ceil(132 / 32) <= width // 3

        def prefill(names):
            lengths = [len(FIXTURES[name]) for name in names]
            tokens = torch.zeros((len(names), max(lengths)), dtype=torch.int64)
            for row, name in enumerate(names):
                tokens[row, : lengths[row]] = torch.tensor(FIXTURES[name])
            logits = generator.prefill_forward(
                tokens,
                prompt_lens=lengths,
                start_pos=[0] * len(names),
                slots=tuple(range(len(names))),
                page_table=pages[: len(names)],
            )
            assert logits.shape == (len(names), 1, model.vocab_size)
            return logits[:, 0].clone(), lengths

        # Warm all diagnostic read/prefill forms before retaining any trace.
        prefill(list(FIXTURES))
        generator.prepare()
        traces = dict(generator.traces)
        expected_model_keys = {generator._bucket_trace_key(bucket) for bucket in buckets}
        if generator.solo_slot0_attention:
            expected_model_keys.add("model_solo0")
        expected_capture_count = len(expected_model_keys) + 2
        assert set(traces) == expected_model_keys | {"sampling", "sampling_64"}
        assert generator.counters["captures"] == expected_capture_count
        result["trace_ids"] = {key: str(value) for key, value in traces.items()}
        result["baseline"] = {}
        result["cases"] = []

        def run(names, buckets, label):
            generator.decode_buckets = buckets
            generator.reset()
            first, lengths = prefill(names)
            wire_rows = next(bucket for bucket in buckets if bucket >= len(names))
            tokens = torch.zeros((wire_rows, 1), dtype=torch.int32)
            # Diagnostic caller-owned host sampling, never used for performance.
            tokens[: len(names), 0] = torch.argmax(first, dim=-1).to(torch.int32)
            positions = lengths + [-1] * (wire_rows - len(names))
            before = dict(generator.counters)
            replays_before = dict(generator.trace_replays)
            key = generator.model_trace_key(tuple(range(len(names))))
            handle = generator.decode_forward_v1(
                tokens,
                positions,
                kv_cache=generator.cache,
                page_table=pages[:wire_rows],
                sampling_params=None,
                reload_inputs=True,
                read_from_device=False,
                enable_trace=True,
            )
            returned, events = generator.read_decode_output(handle, async_read=True)
            assert returned is handle and events
            output = generator.process_decode_output_host(returned, is_tokens=False)
            assert output.shape == (wire_rows, 1, model.vocab_size)
            second = output[: len(names), 0].clone()
            assert generator.counters["sampling_replays"] == before["sampling_replays"]
            assert generator.trace_replays.get(key, 0) == replays_before.get(key, 0) + 1
            assert generator.traces == traces and generator.counters["captures"] == expected_capture_count
            assert generator.counters["releases"] == 0
            row = dict(label=label, fixtures=names, buckets=list(buckets), wire_rows=wire_rows, trace=key, exact=True)
            result["cases"].append(row)
            for index, name in enumerate(names):
                pair = (first[index], second[index])
                if external_reference is not None:
                    for step, (expected, actual) in enumerate(zip(external_reference[name], pair)):
                        assert torch.equal(expected, actual), (
                            "external_full_logits",
                            label,
                            name,
                            step,
                            int((expected != actual).sum()),
                            float((expected - actual).abs().max()),
                        )
                if name not in full_reference:
                    full_reference[name] = tuple(value.clone() for value in pair)
                    result["baseline"][name] = [compact_logits(value) for value in pair]
                else:
                    for step, (expected, actual) in enumerate(zip(full_reference[name], pair)):
                        if not torch.equal(expected, actual):
                            row["exact"] = False
                            row["failure"] = dict(
                                fixture=name,
                                step=step,
                                max_abs=float((expected - actual).abs().max()),
                                differing=int((expected != actual).sum()),
                            )
                            save()
                            raise AssertionError(
                                f"Full-vocabulary logits differ: {label}/{name}/step{step}: {row['failure']}"
                            )
            save()
            print("LOGIT_CASE_PASS", label, flush=True)

        for name in FIXTURES:
            run([name], buckets, name + "/B2")
            if args.four_row_bucket:
                run([name], (4, 32), name + "/forced_B4")
            run([name], (32,), name + "/forced_B32")
        for names in (list(FIXTURES), list(reversed(FIXTURES))):
            run(names, buckets, "reordered/" + ",".join(names))
        generator.decode_buckets = buckets
        if args.save_full_logits:
            assert not args.save_full_logits.exists(), "Preserve the existing full-logit reference"
            args.save_full_logits.parent.mkdir(parents=True, exist_ok=True)
            torch.save(full_reference, args.save_full_logits)
            result["full_logits_saved_sha256"] = hashlib.sha256(args.save_full_logits.read_bytes()).hexdigest()
        if args.serial_output_tokens:
            result["serial_device_tokens"] = {}
            for name, tokens in FIXTURES.items():
                result["serial_device_tokens"][name] = generator.generate(
                    tokens,
                    args.serial_output_tokens,
                    temperature=0.0,
                    top_k=1,
                    top_p=1.0,
                    seed=42,
                    stop_on_eos=False,
                )
                assert len(result["serial_device_tokens"][name]) == args.serial_output_tokens
                save()
        result.update(trace_replays=dict(generator.trace_replays), counters_before_shutdown=dict(generator.counters))
    finally:
        try:
            if generator is not None:
                generator.close()  # Also discards an output lease after any read/assertion failure.
                result["counters_after_shutdown"] = dict(generator.counters)
                assert not generator.traces and generator.counters["releases"] == generator.counters["captures"]
        finally:
            if mesh is not None:
                ttnn.close_mesh_device(mesh)


def http(args, result, save):
    import requests

    assert args.baseline is not None, "HTTP mode requires --baseline"
    baseline = json.loads(args.baseline.read_text())
    assert baseline["status"] == "PASS" and baseline["mode"] == "standalone"
    assert baseline["fixtures"] == FIXTURES and baseline["layers"] == list(range(40))
    assert baseline["source_sha256"] == result["source_sha256"], "Local source changed since standalone baseline"
    result["baseline_source"] = str(args.baseline)
    result["baseline_sha256"] = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    result["server_url"] = args.url
    result["server_log"] = None if args.server_log is None else str(args.server_log)
    result["requests"] = []
    canonical = {}

    def request(name):
        payload = dict(
            model=MODEL,
            prompt=FIXTURES[name],
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            seed=42,
            max_tokens=2,
            ignore_eos=True,
            logprobs=TOP_N,
            return_tokens_as_token_ids=True,
            return_token_ids=True,
            stream=False,
        )
        started = time.monotonic()
        response = requests.post(args.url.rstrip("/") + "/v1/completions", json=payload, timeout=args.timeout)
        try:
            body = response.json()
        except ValueError:
            body = {"unparsed": response.text[:2000]}
        return dict(
            fixture=name,
            payload=payload,
            started=started,
            elapsed_s=time.monotonic() - started,
            status=response.status_code,
            response=body,
        )

    def accept(record, phase):
        record["phase"] = phase
        result["requests"].append(record)
        save()
        assert record["status"] == 200, (phase, record["fixture"], record["response"])
        name = record["fixture"]
        steps = response_steps(record["response"])
        record["max_abs_vs_standalone"] = compare_steps(baseline["baseline"][name], steps, phase + "/" + name)
        if name in canonical:
            record["max_abs_vs_first_http"] = compare_steps(canonical[name], steps, phase + "/repeat/" + name)
        else:
            canonical[name] = steps
        record["passed"] = True
        save()

    for repeat in range(2):
        for name in FIXTURES:
            accept(request(name), f"solo{repeat}")
    for order, names in enumerate((list(FIXTURES), list(reversed(FIXTURES)))):
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            for record in pool.map(request, names):
                accept(record, f"concurrent_order{order}")
    assert len(result["requests"]) == 12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("standalone", "http"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--trace-region-size", type=int, default=805306368)
    parser.add_argument("--serial-output-tokens", type=int, choices=(0, 96), default=0)
    parser.add_argument("--four-row-bucket", action="store_true")
    parser.add_argument("--route-mask", action="store_true")
    parser.add_argument("--route-plan", action="store_true")
    parser.add_argument("--solo-attention", action="store_true")
    parser.add_argument("--packed-postreduce-weights", action="store_true")
    parser.add_argument("--native-qk-reader", action="store_true")
    parser.add_argument("--prune-postreduce-padding", action="store_true")
    parser.add_argument("--hidden-row-mask", action="store_true")
    parser.add_argument("--conditional-state-commit", action="store_true")
    parser.add_argument("--ep-weight-nd-sharding", action="store_true")
    parser.add_argument("--save-full-logits", type=Path)
    parser.add_argument("--compare-full-logits", type=Path)
    args = parser.parse_args()
    assert not args.route_plan or args.route_mask
    assert not args.output.exists(), "Use a fresh artifact path"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="RUNNING",
        mode=args.mode,
        command=sys.argv,
        model=MODEL,
        revision=REVISION,
        source_sha256=source_identity(),
        fixtures=FIXTURES,
        capacity=32,
        standalone_context=1024,
        trace_region_size=args.trace_region_size,
        atol=ATOL,
        rtol=0,
        top_logprobs=TOP_N,
        serial_output_tokens=args.serial_output_tokens,
        route_mask=args.route_mask,
        route_plan=args.route_plan,
        solo_attention=args.solo_attention,
        hidden_row_mask=args.hidden_row_mask,
        conditional_state_commit=args.conditional_state_commit,
        ep_weight_nd_sharding=args.ep_weight_nd_sharding,
        full_logits_reference=None if args.compare_full_logits is None else str(args.compare_full_logits),
        full_logits_saved=None if args.save_full_logits is None else str(args.save_full_logits),
        limitations=[
            "Raw repeated-token stress fixtures: no qualitative verdict or model-accuracy claim.",
            "Full-vocabulary exactness applies to the recorded standalone all40 buckets and row reorder comparisons.",
            "HTTP exposes only top20 and emitted-token logprobs; it cannot establish full-vocabulary equality.",
            "HTTP logprobs20 deliberately invokes optional plugin host sampling; this is not the performance route.",
            "HTTP concurrency is requested, not guaranteed scheduler co-batching; correlate audited server replay events.",
            "Standalone cache/context1024 differs from the served allocation; cross-path comparison is tested, not assumed.",
            "Optional serial96 uses canonical on-device feedback but does not replace a nonasync vLLM server control.",
            "Source hashes describe this checkout; match the server registration/startup provenance separately.",
        ],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    try:
        (standalone if args.mode == "standalone" else http)(args, result, save)
        result["status"] = "PASS"
    except BaseException as error:
        result.update(status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        save()
    print("VLLM_LOGIT_DETERMINISM_PASS", args.output, flush=True)


if __name__ == "__main__":
    main()
