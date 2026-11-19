# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU HF incremental replay at compact native-TT sampling boundaries.

Use an outer timeout of 600 seconds. Collection success is not an accuracy gate.
The native JSON is read once; a running artifact contributes only snapshots that
were complete at that read. No full vocabulary tensor is written to disk.
"""

import argparse
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
DEFAULT_REFERENCE = ROOT / "readiness_vllm/sampling_controls/http_bad_words_card_1024.json"
DEFAULT_HF = ROOT / "readiness_vllm/sampling_controls/hf_bad_words_card_1024.json"


def sampling_ops():
    # Disable vLLM's optional platform/plugin discovery for this CPU diagnostic.
    os.environ["VLLM_PLUGINS"] = ""
    import torch
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p, sample_with_exponential_noise

    return torch, apply_top_k_top_p, sample_with_exponential_noise


def process_logits(raw, history, request):
    torch, apply_top_k_top_p, _ = sampling_ops()
    values = raw.float().clone()
    if values.ndim != 2 or values.shape[0] != 1 or not torch.isfinite(values).all():
        raise ValueError("Expected one finite CPU full-vocabulary logit row")
    if values.device.type != "cpu":
        raise ValueError("Boundary comparison is CPU-only")
    if any(len(ids) != 1 for ids in request["bad_words_token_ids"]):
        raise ValueError("This diagnostic only implements singleton bad-word masks")
    if history:
        values[:, sorted(set(history))] -= request["presence_penalty"]
    values[:, [ids[0] for ids in request["bad_words_token_ids"]]] = -float("inf")
    generation = request["generation"]
    values /= generation["temperature"]
    return apply_top_k_top_p(
        values,
        torch.tensor([generation["top_k"]], dtype=torch.int32),
        torch.tensor([generation["top_p"]], dtype=torch.float32),
    )


def distribution_summary(values, noise):
    torch, _, sample_with_exponential_noise = sampling_ops()
    support = torch.nonzero(torch.isfinite(values[0])).flatten().tolist()
    if not support:
        raise ValueError("A boundary has empty processed support")
    probs = values.softmax(dim=-1, dtype=torch.float32)
    winner = int(sample_with_exponential_noise(probs.clone(), noise.clone()).item())
    # Exact division scores determine the winner. Logit-log(noise) margins are
    # also useful because they remove the arbitrary softmax normalizer/offset.
    division = probs / noise
    ranks = torch.topk(division[0], min(2, len(support))).indices.tolist()
    log_race = values.double() - noise.double().log()
    runner_up = ranks[1] if len(ranks) > 1 else None
    return dict(
        winner=winner,
        support=sorted(support),
        winner_probability=float(probs[0, winner]),
        winner_noise=float(noise[0, winner]),
        winner_division_score=float(division[0, winner]),
        runner_up=runner_up,
        division_margin=(float(division[0, winner] - division[0, runner_up]) if runner_up is not None else None),
        log_race_margin=(float(log_race[0, winner] - log_race[0, runner_up]) if runner_up is not None else None),
    )


def compare_boundary(raw, snapshot, noise, request):
    torch, _, _ = sampling_ops()
    vocab = raw.shape[-1]
    history = snapshot["history"]
    hf_values = process_logits(raw, history, request)
    tt_raw = {int(token): float(value) for token, value in snapshot["raw_top_logit_values"].items()}
    tt_processed = {int(token): float(value) for token, value in snapshot["processed_top_logit_values"].items()}
    if not tt_processed or not set(tt_processed).issubset(tt_raw):
        raise ValueError("Native snapshot must include raw values for every processed-support token")
    if any(token < 0 or token >= vocab for token in set(tt_raw) | set(tt_processed)):
        raise ValueError("Native snapshot has an out-of-vocabulary candidate")
    if any(not math.isfinite(value) for value in [*tt_raw.values(), *tt_processed.values()]):
        raise ValueError("Native compact maps must contain finite values only")
    native_values = torch.full_like(raw, -float("inf"), dtype=torch.float32)
    for token, value in tt_processed.items():
        native_values[0, token] = value
    hf = distribution_summary(hf_values, noise)
    native = distribution_summary(native_values, noise)
    native_matches = native["winner"] == snapshot["recorded_token"]
    # Check retained values against their raw logits and the declared arithmetic.
    native_pre = torch.tensor([tt_raw[token] for token in tt_processed], dtype=torch.float32)
    seen = set(history)
    native_pre -= torch.tensor([request["presence_penalty"] if token in seen else 0.0 for token in tt_processed])
    native_pre /= request["generation"]["temperature"]
    native_retained = torch.tensor(list(tt_processed.values()), dtype=torch.float32)
    retained_match = torch.equal(native_pre, native_retained)
    bans = {ids[0] for ids in request["bad_words_token_ids"]}
    native_has_banned_support = bool(bans & set(tt_processed))
    candidates = sorted(set(hf["support"]) | set(native["support"]))
    table = []
    for token in candidates:
        table.append(
            dict(
                token=token,
                seen_in_history=token in seen,
                hf_raw=float(raw[0, token]),
                native_raw=tt_raw.get(token),
                raw_delta_hf_minus_native=(float(raw[0, token]) - tt_raw[token] if token in tt_raw else None),
                hf_processed=(float(hf_values[0, token]) if torch.isfinite(hf_values[0, token]) else None),
                native_processed=tt_processed.get(token),
                exponential_noise=float(noise[0, token]),
            )
        )
    shared_ids = sorted(tt_raw)
    delta = raw[0, shared_ids].double() - torch.tensor([tt_raw[token] for token in shared_ids], dtype=torch.float64)
    deltas = dict(
        candidate_count=len(shared_ids),
        mean_hf_minus_native=float(delta.mean()),
        mean_absolute_difference=float(delta.abs().mean()),
        maximum_absolute_difference=float(delta.abs().max()),
        centered_rmse=float(((delta - delta.mean()).square().mean()).sqrt()),
        scope="Only native compact raw candidates; not a full-vocabulary error statistic",
    )
    explains = native_matches and retained_match and not native_has_banned_support and hf["winner"] != native["winner"]
    return dict(
        seed=snapshot["seed"],
        index=snapshot["index"],
        history=history,
        hf=hf,
        native_replay=native,
        recorded_native_token=snapshot["recorded_token"],
        recorded_token_matches_native_replay=native_matches,
        retained_native_values_match_declared_penalty_temperature=retained_match,
        native_support_contains_banned_token=native_has_banned_support,
        support_overlap=sorted(set(hf["support"]) & set(native["support"])),
        hf_only_support=sorted(set(hf["support"]) - set(native["support"])),
        native_only_support=sorted(set(native["support"]) - set(hf["support"])),
        raw_candidate_differences=deltas,
        candidates=table,
        conditional_distribution_difference_explains_draw=explains,
        interpretation=(
            "Captured distributions produce different winners with the same reconstructed noise; native winner matches recording. Conditional on complete native support and stated RNG history, these distributions suffice to explain this draw."
            if explains
            else "Native recording or retained arithmetic is inconsistent with replay; do not attribute the boundary to HF/TT model arithmetic."
            if not native_matches or not retained_match or native_has_banned_support
            else "Both distributions produce the same winner at this boundary. This is neither full-logit equivalence nor evidence that unobserved boundaries agree."
        ),
    )


def validated_snapshots(native, request):
    for key in ("model", "revision", "prompt_tokens", "generation", "presence_penalty", "bad_words_token_ids"):
        if native.get("request", {}).get(key) != request[key]:
            raise ValueError(f"Native/reference request mismatch: {key}")
    if request["model"] != MODEL or request["revision"] != REVISION:
        raise ValueError("Boundary control requires the pinned Qwen model and revision")
    if request.get("chat_template_kwargs", {}).get("enable_thinking") is False:
        raise ValueError("Boundary control requires thinking enabled")
    if request["generation"] != dict(temperature=1.0, top_p=0.95, top_k=20) or request["presence_penalty"] != 1.5:
        raise ValueError("Boundary control is scoped to the thinking model-card profile")
    grouped = {}
    for row in native["outputs"]:
        snapshots = sorted(row.get("boundaries", []), key=lambda entry: entry["index"])
        if not snapshots:
            continue
        seed = row["seed"]
        if seed in grouped:
            raise ValueError(f"Duplicate native seed: {seed}")
        if len({entry["index"] for entry in snapshots}) != len(snapshots):
            raise ValueError(f"Duplicate boundary index for seed {seed}")
        longest = snapshots[-1]["history"]
        for entry in snapshots:
            index = entry["index"]
            if index < 0 or entry["seed"] != seed or len(entry["history"]) != index:
                raise ValueError("Native boundary seed/history/index mismatch")
            if entry["history"] != longest[:index] or entry["history"] != row["tokens"][:index]:
                raise ValueError("Native boundaries do not share the recorded incremental trajectory")
            if index >= len(row["tokens"]) or entry["recorded_token"] != row["tokens"][index]:
                raise ValueError("Native boundary's recorded draw does not match its row")
        grouped[seed] = snapshots
    if not grouped:
        raise ValueError("Native artifact has no completed boundary snapshots")
    return grouped


def collect(args, request, native, grouped, result, save):
    torch, _, _ = sampling_ops()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    rendered = tokenizer.apply_chat_template(
        request["messages"], tokenize=False, add_generation_prompt=True, **request["chat_template_kwargs"]
    )
    prompt = tokenizer.encode(rendered, add_special_tokens=False)
    if prompt != request["prompt_tokens"]:
        raise ValueError("Pinned tokenizer/template does not reproduce reference prompt tokens")
    result["pending_operation"] = "load_native_hf_bfloat16"
    save()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True, dtype=torch.bfloat16
    ).eval()
    if model.device.type != "cpu" or model.dtype != torch.bfloat16:
        raise ValueError("Expected native HF model on CPU in BF16")
    result.update(
        resolved_model_dtype=str(model.dtype), model_class=type(model).__name__, torch_version=torch.__version__
    )
    http_rows = {row["seed"]: row["response"]["choices"][0]["token_ids"] for row in request["outputs"]}
    hf_rows = {}
    if DEFAULT_HF.exists():
        prior = json.loads(DEFAULT_HF.read_text())
        if all(
            prior.get(key) == request.get(key)
            for key in ("model", "revision", "prompt_tokens", "generation", "presence_penalty", "bad_words_token_ids")
        ):
            hf_rows = {row["seed"]: row["tokens"] for row in prior["outputs"]}
            result["prior_native_hf_control"] = str(DEFAULT_HF)
    with torch.inference_mode():
        for seed, snapshots in grouped.items():
            started = time.monotonic()
            by_index = {entry["index"]: entry for entry in snapshots}
            history = snapshots[-1]["history"]
            generator = torch.Generator(device="cpu").manual_seed(seed)
            past = None
            for index in range(max(by_index) + 1):
                result["pending_operation"] = f"seed={seed}:hf_incremental_index={index}"
                save()
                # Always prefill only the original prompt, then one native token
                # per forward. This preserves the HF recurrent/cache regime.
                inputs = prompt if index == 0 else [history[index - 1]]
                output = model(
                    input_ids=torch.tensor([inputs], dtype=torch.int64),
                    past_key_values=past,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                past = output.past_key_values
                if past is None:
                    raise RuntimeError("HF did not return its incremental cache")
                raw = output.logits[:, -1, :].float().clone()
                del output
                if native.get("vocabulary_size", raw.shape[-1]) != raw.shape[-1]:
                    raise ValueError("Native and HF vocabulary widths differ")
                # Advance exactly once per generated-token index, even between
                # sampled snapshots. Never sample from a compressed vocabulary.
                noise = torch.empty_like(raw).exponential_(generator=generator)
                if index in by_index:
                    comparison = compare_boundary(raw, by_index[index], noise, request)
                    comparison["http_history_matches"] = history[:index] == http_rows.get(seed, [])[:index]
                    comparison["http_token_at_index"] = (
                        http_rows.get(seed, [])[index] if len(http_rows.get(seed, [])) > index else None
                    )
                    comparison["prior_hf_history_matches"] = (
                        history[:index] == hf_rows[seed][:index] if seed in hf_rows else None
                    )
                    comparison["prior_hf_token_at_index"] = (
                        hf_rows[seed][index] if seed in hf_rows and len(hf_rows[seed]) > index else None
                    )
                    comparison["native_vs_prior_hf_first_divergence_at_boundary"] = (
                        comparison["prior_hf_history_matches"] is True
                        and comparison["prior_hf_token_at_index"] is not None
                        and comparison["recorded_native_token"] != comparison["prior_hf_token_at_index"]
                    )
                    comparison["hf_replay_matches_prior_hf_draw_on_shared_history"] = (
                        comparison["hf"]["winner"] == comparison["prior_hf_token_at_index"]
                        if comparison["prior_hf_history_matches"] is True
                        and comparison["prior_hf_token_at_index"] is not None
                        else None
                    )
                    comparison["elapsed_s"] = time.monotonic() - started
                    for candidate in comparison["candidates"]:
                        candidate["text"] = tokenizer.decode([candidate["token"]], skip_special_tokens=False)
                    result["boundaries"].append(comparison)
                    save()
                    print(
                        f"HF_BOUNDARY seed={seed} index={index} hf={comparison['hf']['winner']} native={comparison['native_replay']['winner']} recorded_match={comparison['recorded_token_matches_native_replay']}",
                        flush=True,
                    )
            del past


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a fresh boundary-control output artifact")
    request = json.loads(args.reference.read_text())
    native = json.loads(args.native.read_text())
    grouped = validated_snapshots(native, request)
    result = dict(
        collection_status="RUNNING",
        diagnostic_only=True,
        performance_measurement=False,
        command=[sys.executable, *sys.argv],
        required_outer_timeout_s=600,
        native_artifact=str(args.native.resolve()),
        native_collection_status_at_read=native.get("collection_status"),
        reference=str(args.reference.resolve()),
        model=MODEL,
        revision=REVISION,
        request={key: value for key, value in request.items() if key not in ("outputs", "status", "rubric")},
        requested_boundaries={
            str(seed): [entry["index"] for entry in snapshots] for seed, snapshots in grouped.items()
        },
        pending_operation="setup",
        boundaries=[],
        limitations=[
            "Native artifact is read once; later snapshots and unfinished seeds are not included.",
            "Native processed map is assumed to contain every finite post-filter token, as supplied by boundary_snapshot; omitted raw logits cannot independently verify full support construction.",
            "RNG is reconstructed from seed and draw index, not captured from live serving; agreement is conditional on one full-vocabulary FP32 exponential draw per preceding generated token.",
            "Native standalone TT snapshots do not directly establish HTTP serving sampler/history/cache state.",
            "Only original prompt prefill followed by one-token cached HF forwards are used; native TT history is teacher-forced even after it differs from free-running HF.",
            "Candidate error statistics are partial; sampled-token agreement is not full-logit agreement or a quality pass.",
            "The helper performs no device access and writes no full-vocabulary tensors or artifact hashes.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        temporary.replace(args.output)

    save()
    try:
        collect(args, request, native, grouped, result, save)
        result["collection_status"] = "PASS"
        result["pending_operation"] = None
        result["summary"] = dict(
            boundaries=len(result["boundaries"]),
            native_recording_replay_mismatches=sum(
                not row["recorded_token_matches_native_replay"] for row in result["boundaries"]
            ),
            different_hf_native_winners=sum(
                row["hf"]["winner"] != row["native_replay"]["winner"] for row in result["boundaries"]
            ),
            conditionally_explained_different_draws=sum(
                row["conditional_distribution_difference_explains_draw"] for row in result["boundaries"]
            ),
            scope="Diagnostic collection only; no model-quality or serving-readiness verdict",
        )
    except BaseException as error:
        result.update(collection_status="FAIL", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        save()
    print("HF_BOUNDARY_CONTROL", result["collection_status"], args.output, flush=True)


if __name__ == "__main__":
    main()
