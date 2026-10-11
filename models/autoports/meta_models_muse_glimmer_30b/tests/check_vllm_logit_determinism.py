# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-token served/standalone logit determinism diagnostics, never performance.

collect: existing HTTP server, exact saved chat IDs, sequential repeats and
reordered prompt arrays. Requires explicit server host-logprob compatibility.
control: run only after that server closes; full selected TT model, batch4,
standalone-owned cache, sample=False, no autoregressive host-token feedback.

Default tolerance (atol=2e-5, rtol=1e-6) allows FP32 log-softmax reduction/JSON
rounding, not BF16-scale model-logit drift. Top20 membership and emitted tokens
must be identical across repeated HTTP requests. Standalone comparisons allow
ties exactly at the top20 cutoff, while checking every served logprob value.
"""

import argparse
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROMPTS = ROOT / "readiness_vllm/qualitative_prompt_format.json"
SELECTED = ROOT / "doc/datatype_sweep/selected_precision_config.json"
ORDERS = [(0,), (1,), (2,), (0,), (1,), (2,), (0, 1, 2, 0), (1, 2, 0, 0), (2, 0, 0, 1), (0, 0, 1, 2)]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def close_number(actual, expected, args):
    if not (math.isfinite(actual) and math.isfinite(expected)):
        raise AssertionError(f"Nonfinite logprob: {actual}, {expected}")
    error = abs(actual - expected)
    if error > args.atol + args.rtol * abs(expected):
        raise AssertionError(f"Logprob drift: {actual} vs {expected}; abs={error}")
    return error


def token_id(token):
    prefix = "token_id:"
    if not isinstance(token, str) or not token.startswith(prefix):
        raise AssertionError(f"Expected unambiguous token_id:N logprob key, got {token!r}")
    return int(token[len(prefix) :])


def normalize_choice(choice):
    value = choice.get("logprobs")
    assert value and len(value["tokens"]) == 1, "Expected exactly one generated token with logprobs"
    assert len(value["token_logprobs"]) == len(value["top_logprobs"]) == 1
    top = {str(token_id(key)): float(logprob) for key, logprob in value["top_logprobs"][0].items()}
    assert len(top) == 20, f"Expected top20 map, got {len(top)} entries"
    chosen = token_id(value["tokens"][0])
    assert str(chosen) in top, "Greedy token absent from reported top20"
    assert all(math.isfinite(logprob) for logprob in top.values())
    return dict(token=chosen, token_logprob=float(value["token_logprobs"][0]), top20=top)


def compare_maps(actual, baseline, args):
    assert actual["token"] == baseline["token"], "Emitted token differs across repeated/batched prompts"
    assert actual["top20"].keys() == baseline["top20"].keys(), "Top20 membership differs"
    errors = [close_number(actual["top20"][key], baseline["top20"][key], args) for key in baseline["top20"]]
    errors.append(close_number(actual["token_logprob"], baseline["token_logprob"], args))
    return dict(max_abs=max(errors), mean_abs=sum(errors) / len(errors), token=actual["token"])


def collect(args, report):
    metadata = json.loads(args.prompts.read_text())
    assert metadata["chat_template_present"] and metadata["mode"] == "chat"
    prompts = metadata["prompts"][:3]
    assert len(prompts) == 3 and all(prompt["token_ids"] for prompt in prompts)
    report.update(
        prompt_format_path=str(args.prompts),
        prompt_format_sha256=sha256(args.prompts),
        prompt_format_metadata={key: value for key, value in metadata.items() if key != "prompts"},
        prompts=prompts,
        server_url=args.server_url,
        server_max_num_seqs=args.server_max_num_seqs,
        server_mode="explicit host-logprob diagnostic; not the on-device performance path",
        cross_batch_evidence="HTTP prompt-array indices; internal scheduler slots are not inferred",
        requests=[],
        comparisons=[],
    )
    endpoint = args.server_url.rstrip("/")
    endpoint += "/completions" if endpoint.endswith("/v1") else "/v1/completions"
    baseline = {}
    for request_index, order in enumerate(ORDERS):
        payload = dict(
            model=args.model,
            prompt=[prompts[index]["token_ids"] for index in order],
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            logprobs=20,
            return_tokens_as_token_ids=True,
            seed=117,
            ignore_eos=True,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
        )
        started = time.monotonic()
        submitted_at = time.time()
        request = urllib.request.Request(
            endpoint, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                raw = json.load(response)
        except urllib.error.HTTPError as error:
            report["requests"].append(
                dict(
                    request_index=request_index,
                    order=order,
                    payload=payload,
                    http_status=error.code,
                    response_body=error.read().decode(errors="replace"),
                )
            )
            save(args.output, report)
            raise
        record = dict(
            request_index=request_index,
            order=order,
            payload=payload,
            raw_response=raw,
            submitted_unix_seconds=submitted_at,
            completed_unix_seconds=time.time(),
            elapsed_seconds=time.monotonic() - started,
            outputs=[],
        )
        report["requests"].append(record)
        # Persist the untouched response before validating a failing field.
        save(args.output, report)
        choices = raw.get("choices", [])
        assert len(choices) == len(order), f"Missing completion choices for request {request_index}"
        by_index = {choice["index"]: choice for choice in choices}
        assert set(by_index) == set(range(len(order))), "Missing/duplicate prompt-array indices"
        usage = raw.get("usage", {})
        assert usage.get("completion_tokens") == len(order), "Request did not emit exactly one token per prompt"
        assert usage.get("prompt_tokens") == sum(
            len(prompts[index]["token_ids"]) for index in order
        ), "Prompt IDs were truncated or augmented"
        for row, prompt in enumerate(order):
            normalized = normalize_choice(by_index[row])
            close_number(normalized["token_logprob"], normalized["top20"][str(normalized["token"])], args)
            record["outputs"].append(dict(prompt_index=prompt, array_index=row, **normalized))
            if prompt in baseline:
                report["comparisons"].append(
                    dict(
                        request_index=request_index,
                        prompt_index=prompt,
                        array_index=row,
                        **compare_maps(normalized, baseline[prompt], args),
                    )
                )
            else:
                baseline[prompt] = normalized
        save(args.output, report)
        print(f"COLLECT request={request_index} order={order} passed", flush=True)
    report["baseline"] = baseline


def control(args, report):
    served = json.loads(args.served.read_text())
    assert served["status"] == "pass" and served["mode"] == "collect", "Require passing served collection"
    assert served["selected_precision_sha256"] == sha256(SELECTED), "Selected precision changed since collection"
    assert served["prompt_format_sha256"] == sha256(args.prompts), "Saved prompt format changed since collection"
    prompts = served["prompts"]
    report.update(
        served_path=str(args.served),
        served_sha256=sha256(args.served),
        prompts=prompts,
        standalone_batch_size=4,
        server_max_num_seqs=served["server_max_num_seqs"],
        standalone_cache_ownership=True,
        comparison="FP32 log_softmax of standalone last-token raw logits",
        physical_batch_tiles=dict(standalone=32, server=32 * ((served["server_max_num_seqs"] + 31) // 32)),
        comparisons=[],
        raw_logits=[],
    )
    # No TTNN or vLLM import occurs in collect mode. The owner serializes this
    # full-model control after stopping the existing server.
    import torch

    torch.set_num_threads(16)
    import ttnn
    from models.autoports.meta_models_muse_glimmer_30b.tt.generator import build_generator

    selected = json.loads(SELECTED.read_text())
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    generator = None
    baseline = {}
    try:
        generator = build_generator(ROOT, mesh, batch_size=4, precision_config=SELECTED)
        assert generator.model.layer_indices == list(range(52)), "Control must run all real layers"
        assert generator.model.precision_config == selected
        report["loaded_precision"] = generator.model.precision_config
        report["revision"] = selected["revision"]
        report["layers"] = generator.model.layer_indices
        report["torch_threads"] = torch.get_num_threads()
        # Pages are standalone-owned, distinct across slots, and never retired.
        max_length = max(len(prompt["token_ids"]) for prompt in prompts)
        pages_per_slot = (max_length + 31) // 32
        table = torch.zeros((4, 4096), dtype=torch.int32)
        for slot in range(4):
            table[slot, :pages_per_slot] = 1 + slot * pages_per_slot + torch.arange(pages_per_slot)
        report["active_page_table"] = table[:, :pages_per_slot].tolist()
        for request in served["requests"]:
            order = request["order"]
            generator.reset(clear_cache=True)
            lengths = [len(prompts[index]["token_ids"]) for index in order]
            tokens = torch.zeros((len(order), max(lengths)), dtype=torch.int32)
            for row, index in enumerate(order):
                tokens[row, : lengths[row]] = torch.tensor(prompts[index]["token_ids"], dtype=torch.int32)
            raw = generator.prefill_forward(
                tokens,
                page_table=table,
                kv_cache=generator.kv_cache,
                prompt_lens=lengths,
                slots=list(range(len(order))),
                sample=False,
            )
            raw = raw.reshape(4, -1)[: len(order), : generator.model.vocab_size].contiguous().clone()
            assert torch.isfinite(raw).all(), "Nonfinite standalone logits"
            path = args.output.with_name(f"{args.output.stem}_request{request['request_index']:02d}_logits.pt")
            torch.save(
                dict(
                    raw_logits=raw,
                    order=order,
                    lengths=lengths,
                    prompt_token_ids=[prompts[index]["token_ids"] for index in order],
                ),
                path,
            )
            report["raw_logits"].append(str(path))
            save(args.output, report)
            normalized = raw.float().log_softmax(-1)
            for row, output in enumerate(request["outputs"]):
                prompt = order[row]
                assert prompt == output["prompt_index"] and row == output["array_index"]
                values = normalized[row]
                errors = [
                    close_number(float(values[int(token)]), logprob, args) for token, logprob in output["top20"].items()
                ]
                chosen = output["token"]
                errors.append(close_number(float(values[chosen]), output["token_logprob"], args))
                assert raw[row, chosen] == raw[row].max(), "Served greedy token is not standalone top1"
                cutoff = float(values.topk(20).values[-1])
                assert all(
                    float(values[int(token)]) >= cutoff for token in output["top20"]
                ), "Served top20 contains a token below standalone rank20"
                row_report = dict(
                    request_index=request["request_index"],
                    prompt_index=prompt,
                    standalone_slot=row,
                    server_array_index=row,
                    token=chosen,
                    max_logprob_abs=max(errors),
                    mean_logprob_abs=sum(errors) / len(errors),
                    top20_cutoff=cutoff,
                    top20_cutoff_ties=int((values == cutoff).sum()),
                )
                if prompt in baseline:
                    reference = baseline[prompt]
                    torch.testing.assert_close(raw[row].float(), reference.float(), atol=0, rtol=0)
                    row_report["standalone_repeat_max_logit_abs"] = float(
                        (raw[row].float() - reference.float()).abs().max()
                    )
                else:
                    baseline[prompt] = raw[row].clone()
                report["comparisons"].append(row_report)
            save(args.output, report)
            print(f"CONTROL request={request['request_index']} order={order} passed", flush=True)
        report["counters"] = dict(generator.counters)
    finally:
        try:
            if generator is not None:
                generator.close()
        finally:
            ttnn.close_mesh_device(mesh)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "control"))
    parser.add_argument("--server-url", default="http://127.0.0.1:8000")
    parser.add_argument("--server-max-num-seqs", type=int, default=32)
    parser.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--served", type=Path, default=ROOT / "readiness_vllm/logit_determinism_collect.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=1e-6)
    args = parser.parse_args()
    args.output = args.output or ROOT / f"readiness_vllm/logit_determinism_{args.mode}.json"
    report = dict(
        status="running",
        mode=args.mode,
        diagnostic_only=True,
        max_tokens=1,
        tolerance=dict(
            atol=args.atol,
            rtol=args.rtol,
            rationale="FP32 log-softmax reduction/JSON rounding; no BF16-scale drift allowance",
        ),
        selected_precision_path=str(SELECTED),
        selected_precision_sha256=sha256(SELECTED),
        selected_precision=json.loads(SELECTED.read_text()),
    )
    try:
        (collect if args.mode == "collect" else control)(args, report)
        report["status"] = "pass"
    except BaseException as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save(args.output, report)


if __name__ == "__main__":
    main()
