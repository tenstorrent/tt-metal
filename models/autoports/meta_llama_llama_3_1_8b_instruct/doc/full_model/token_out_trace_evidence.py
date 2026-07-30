# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Generate token-out trace evidence for the Llama 3.1 8B full-model stage."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import ttnn
from transformers import AutoTokenizer

from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.generator import build_generator
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device
from models.common.sampling import SamplingParams, format_sampling_params


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MODEL_DIR = Path("models/autoports/meta_llama_llama_3_1_8b_instruct")
DEFAULT_PROMPT_FILE = Path("models/common/readiness_check/autoregressive_prompt.txt")


def _read_first_scalar(tt_tensor: ttnn.Tensor) -> int:
    device_tensor = ttnn.get_device_tensors(tt_tensor)[0]
    return int(ttnn.to_torch(device_tensor).reshape(-1)[0].item())


def _counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if isinstance(after_value, bool):
            delta[key] = after_value
        elif isinstance(after_value, (int, float)) and isinstance(before_value, (int, float)):
            delta[key] = after_value - before_value
        elif key == "notes":
            delta[key] = after_value[len(before_value or []) :]
    return delta


def _sampling_all_gather_axis(tt_sampling) -> int:
    return int(getattr(tt_sampling, "sampling_all_gather_axis", getattr(tt_sampling, "_sampling_all_gather_axis", 1)))


def _decode_step_record(
    generator,
    *,
    label: str,
    token_id: int,
    current_pos: int,
    page_table: torch.Tensor,
    token_from_host: bool,
) -> tuple[int, dict[str, Any]]:
    before = generator.trace_counters()
    sampled = generator._decode_trace_sample(
        token_id,
        current_pos,
        page_table=page_table,
        enable_trace=True,
        token_from_host=token_from_host,
        refresh_sampled_hidden=True,
    )
    ttnn.synchronize_device(generator.mesh_device)
    after = generator.trace_counters()

    persistent_token = _read_first_scalar(generator._decode_trace.tokens)
    persistent_current_pos = _read_first_scalar(generator._decode_trace.current_pos)
    expected_current_pos = current_pos + 1
    record = {
        "label": label,
        "input_token": int(token_id),
        "current_pos_before": int(current_pos),
        "sampled_token": int(sampled),
        "persistent_token_after_sampling": persistent_token,
        "persistent_current_pos_after_replay": persistent_current_pos,
        "expected_current_pos_after_replay": expected_current_pos,
        "token_feedback_matches": persistent_token == int(sampled),
        "current_position_matches": persistent_current_pos == expected_current_pos,
        "counter_delta": _counter_delta(before, after),
        "counters_after": after,
    }
    return int(sampled), record


def _build_prompt(prompt_file: Path) -> tuple[list[int], str]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True)
    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    return tokenizer.encode(prompt_text, add_special_tokens=True), prompt_text


def _run_token_out(generator, tokenizer, prompt_ids: list[int], prompt_text: str, max_new_tokens: int) -> dict[str, Any]:
    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    generator.reset()

    print(f"Running token-out {max_new_tokens}-token decode from prompt_len={len(prompt_ids)}")
    prefill_start = time.perf_counter()
    logits = generator.prefill_forward(
        prompt,
        page_table=generator.page_table,
        kv_cache=generator.kv_cache,
        prompt_lens=[len(prompt_ids)],
        return_all_logits=False,
    )
    ttnn.synchronize_device(generator.mesh_device)
    ttft_ms = (time.perf_counter() - prefill_start) * 1000.0

    first_token = int(torch.argmax(logits.reshape(-1)).item())
    generated = [first_token]
    feed_token = first_token
    token_from_host = True
    current_pos = len(prompt_ids)
    step_ms: list[float] = []

    decode_start = time.perf_counter()
    for _ in range(1, max_new_tokens):
        step_start = time.perf_counter()
        feed_token = generator._decode_trace_sample(
            int(feed_token),
            current_pos,
            page_table=generator.page_table,
            enable_trace=True,
            token_from_host=token_from_host,
            refresh_sampled_hidden=True,
        )
        ttnn.synchronize_device(generator.mesh_device)
        step_ms.append((time.perf_counter() - step_start) * 1000.0)
        generated.append(int(feed_token))
        current_pos += 1
        token_from_host = False
    decode_elapsed_s = time.perf_counter() - decode_start

    steady_step_ms = step_ms[1:]
    steady_elapsed_s = sum(steady_step_ms) / 1000.0
    token_out = {
        "prefill_argmax_first_token": first_token,
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        "ttft_ms": ttft_ms,
        "decode_elapsed_s_including_first_trace_capture": decode_elapsed_s,
        "decode_tokens_after_prefill": max(0, max_new_tokens - 1),
        "decode_t_s_u_including_first_trace_capture": (max_new_tokens - 1) / decode_elapsed_s
        if max_new_tokens > 1 and decode_elapsed_s > 0
        else 0.0,
        "steady_replay_decode_elapsed_s_excluding_first_trace_capture": steady_elapsed_s,
        "steady_replay_tokens": len(steady_step_ms),
        "steady_replay_t_s_u_excluding_first_trace_capture": len(steady_step_ms) / steady_elapsed_s
        if steady_elapsed_s > 0
        else 0.0,
        "first_decode_step_ms_with_model_and_sampling_trace_capture": step_ms[0] if step_ms else 0.0,
        "median_replay_step_ms_after_capture": statistics.median(steady_step_ms) if steady_step_ms else 0.0,
        "counters": generator.trace_counters(),
        "sampling_path": {
            "force_argmax": bool(generator.sampling.tt_sampling.force_argmax_sampling),
            "max_top_k": int(generator.sampling.tt_sampling.max_top_k),
            "num_gather_links": int(generator.sampling.tt_sampling.num_gather_links),
            "sampling_dp": int(generator.sampling.tt_sampling._sampling_dp),
            "sampling_all_gather_axis": _sampling_all_gather_axis(generator.sampling.tt_sampling),
        },
    }
    print(
        "Token-out pass complete: TTFT "
        f"{ttft_ms:.2f} ms, decode {token_out['decode_t_s_u_including_first_trace_capture']:.2f} t/s/u, "
        f"replay {token_out['steady_replay_t_s_u_excluding_first_trace_capture']:.2f} t/s/u"
    )
    del prompt_text
    return token_out


def _run_token_out_no_readback(generator, prompt_ids: list[int], max_new_tokens: int) -> dict[str, Any]:
    prompt = torch.tensor([prompt_ids], dtype=torch.long)
    generator.reset()

    print(f"Running no-readback token-out replay from prompt_len={len(prompt_ids)}, gen_len={max_new_tokens}")
    prefill_start = time.perf_counter()
    logits = generator.prefill_forward(
        prompt,
        page_table=generator.page_table,
        kv_cache=generator.kv_cache,
        prompt_lens=[len(prompt_ids)],
        return_all_logits=False,
    )
    ttnn.synchronize_device(generator.mesh_device)
    ttft_ms = (time.perf_counter() - prefill_start) * 1000.0

    first_token = int(torch.argmax(logits.reshape(-1)).item())
    current_pos = len(prompt_ids)
    first_decode_start = time.perf_counter()
    generator._decode_trace_sample(
        first_token,
        current_pos,
        page_table=generator.page_table,
        enable_trace=True,
        token_from_host=True,
        refresh_sampled_hidden=True,
        readback=False,
    )
    ttnn.synchronize_device(generator.mesh_device)
    first_decode_ms = (time.perf_counter() - first_decode_start) * 1000.0

    replay_tokens = max(0, max_new_tokens - 2)
    replay_start = time.perf_counter()
    for replay_idx in range(replay_tokens):
        generator._decode_trace_sample(
            0,
            current_pos + 1 + replay_idx,
            page_table=generator.page_table,
            enable_trace=True,
            token_from_host=False,
            refresh_sampled_hidden=True,
            readback=False,
        )
    if replay_tokens:
        ttnn.synchronize_device(generator.mesh_device)
    replay_elapsed_s = time.perf_counter() - replay_start

    counters = generator.trace_counters()
    result = {
        "prefill_argmax_first_token": first_token,
        "ttft_ms": ttft_ms,
        "first_decode_step_ms_with_model_and_sampling_trace_capture": first_decode_ms,
        "steady_replay_tokens": replay_tokens,
        "steady_replay_elapsed_s": replay_elapsed_s,
        "steady_replay_t_s_u": replay_tokens / replay_elapsed_s if replay_elapsed_s > 0 else 0.0,
        "steady_replay_ms_per_token": (replay_elapsed_s * 1000.0 / replay_tokens) if replay_tokens else 0.0,
        "counters": counters,
        "sampling_path": {
            "force_argmax": bool(generator.sampling.tt_sampling.force_argmax_sampling),
            "max_top_k": int(generator.sampling.tt_sampling.max_top_k),
            "num_gather_links": int(generator.sampling.tt_sampling.num_gather_links),
            "sampling_dp": int(generator.sampling.tt_sampling._sampling_dp),
            "sampling_all_gather_axis": _sampling_all_gather_axis(generator.sampling.tt_sampling),
        },
        "host_boundary_audit": {
            "sampled_token_readbacks": int(counters["sampled_token_readbacks"]),
            "token_input_host_copies": int(counters["token_input_host_copies"]),
            "position_host_copies": int(counters["position_host_copies"]),
            "rope_index_host_copies": int(counters["rope_index_host_copies"]),
            "page_table_host_copies": int(counters["page_table_host_copies"]),
            "position_device_increments": int(counters["position_device_increments"]),
            "rope_index_device_increments": int(counters["rope_index_device_increments"]),
        },
    }
    print(
        "No-readback token-out pass complete: TTFT "
        f"{ttft_ms:.2f} ms, replay {result['steady_replay_t_s_u']:.2f} t/s/u, "
        f"readbacks {counters['sampled_token_readbacks']}"
    )
    return result


def _run_trace_feedback_probe(generator, first_token: int, prompt_len: int) -> dict[str, Any]:
    print("Running focused reset/page-table probe on existing trace and existing prefilled KV state")
    generator.reset(keep_decode_trace=True)
    steps = []

    page_table = generator.page_table
    token1, record1 = _decode_step_record(
        generator,
        label="reused_trace_reset_all_first_decode",
        token_id=first_token,
        current_pos=prompt_len,
        page_table=page_table,
        token_from_host=True,
    )
    steps.append(record1)

    token2, record2 = _decode_step_record(
        generator,
        label="unchanged_page_table_device_feedback",
        token_id=token1,
        current_pos=prompt_len + 1,
        page_table=page_table,
        token_from_host=False,
    )
    steps.append(record2)

    changed_page_table = page_table.clone()
    changed_page_table[0, -1] = int(changed_page_table[0, -1].item()) + 1
    _token3, record3 = _decode_step_record(
        generator,
        label="changed_page_table_device_feedback",
        token_id=token2,
        current_pos=prompt_len + 2,
        page_table=changed_page_table,
        token_from_host=False,
    )
    steps.append(record3)

    assertions = {
        "all_sampled_tokens_written_to_persistent_decode_input": all(
            step["token_feedback_matches"] for step in steps
        ),
        "all_current_positions_incremented_on_device": all(step["current_position_matches"] for step in steps),
        "unchanged_page_table_not_recopied": steps[1]["counter_delta"].get("page_table_host_copies") == 0,
        "changed_page_table_copied_once": steps[2]["counter_delta"].get("page_table_host_copies") == 1,
        "changed_page_table_refresh_marked_changed_only": steps[2]["counter_delta"].get(
            "last_page_table_refresh_changed_only"
        )
        is True,
        "reused_trace_reset_all_copied_position_once": steps[0]["counter_delta"].get("position_host_copies") == 1,
        "unchanged_page_table_kept_token_input_on_device": steps[1]["counter_delta"].get("token_input_host_copies")
        == 0,
    }
    return {
        "uses_existing_prefilled_kv_state_after_reset": True,
        "prefill_argmax_first_token": int(first_token),
        "steps": steps,
        "assertions": assertions,
    }


def _run_topk_topp_smoke(generator, current_pos: int) -> dict[str, Any]:
    print("Running top-k/top-p capable split-sampler smoke")
    generator.sampling.reset_trace()
    generator.sampling.reset_sampling_params(
        format_sampling_params(
            SamplingParams(temperature=0.7, top_k=8, top_p=0.9, seed=12345),
            generator.sampling.tt_sampling.max_batch_size,
        )
    )
    page_table = generator._prev_page_table.clone() if generator._prev_page_table is not None else generator.page_table
    before = generator.trace_counters()
    sampled = generator._decode_trace_sample(
        _read_first_scalar(generator._decode_trace.tokens),
        current_pos,
        page_table=page_table,
        enable_trace=True,
        token_from_host=False,
        refresh_sampled_hidden=True,
    )
    ttnn.synchronize_device(generator.mesh_device)
    after = generator.trace_counters()
    persistent_token = _read_first_scalar(generator._decode_trace.tokens)
    return {
        "sampling_params": {
            "temperature": 0.7,
            "top_k": 8,
            "top_p": 0.9,
            "seed": 12345,
        },
        "force_argmax": bool(generator.sampling.tt_sampling.force_argmax_sampling),
        "sampled_token": int(sampled),
        "persistent_token_after_sampling": persistent_token,
        "token_feedback_matches": persistent_token == int(sampled),
        "counter_delta": _counter_delta(before, after),
        "counters_after": after,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--hf-model", default=MODEL_ID)
    parser.add_argument("--mesh-device", default="T3K")
    parser.add_argument("--fabric-config", default="FABRIC_1D_RING")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument(
        "--include-readback-token-out",
        action="store_true",
        help="Reserved for a separate mesh-session wrapper; the default optimized evidence is no-readback token-out.",
    )
    parser.add_argument(
        "--token-out-only",
        action="store_true",
        help=(
            "Run only the optimized no-readback token-out path. This is intended for watcher/runtime-integrity "
            "runs where the trace feedback and top-k/top-p probes already have separate evidence."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_MODEL_DIR / "doc" / "full_model" / "token_out_trace_evidence.json",
    )
    args = parser.parse_args()
    if args.hf_model != MODEL_ID:
        raise ValueError(f"This harness is pinned to {MODEL_ID}, got {args.hf_model}")

    prompt_ids, prompt_text = _build_prompt(args.prompt_file)

    print(f"Opening {args.mesh_device} {args.fabric_config} mesh")
    mesh_device = open_readiness_mesh_device(args.mesh_device, args.fabric_config)
    generator = None
    try:
        print("Building full generator")
        generator = build_generator(model_dir=args.model_dir, mesh_device=mesh_device)
        token_out_no_readback = _run_token_out_no_readback(generator, prompt_ids, args.max_new_tokens)
        token_out = None
        if args.include_readback_token_out:
            raise RuntimeError(
                "Running readback and no-readback token-out back-to-back in one mesh session can leave "
                "trace-owned buffers active for the second prefill. Use a separate process for qualitative "
                "readback evidence; the optimized token-out benchmark is intentionally no-readback only."
            )
        feedback_probe = None
        topk_topp_smoke = None
        if not args.token_out_only:
            feedback_probe = _run_trace_feedback_probe(
                generator,
                token_out_no_readback["prefill_argmax_first_token"],
                len(prompt_ids),
            )
            topk_topp_smoke = _run_topk_topp_smoke(generator, len(prompt_ids) + 3)

        trace_probe_passed = True if args.token_out_only else all(feedback_probe["assertions"].values())
        topk_topp_smoke_passed = (
            True
            if args.token_out_only
            else topk_topp_smoke["token_feedback_matches"] and not topk_topp_smoke["force_argmax"]
        )
        greedy_uses_split_sampling = not token_out_no_readback["sampling_path"]["force_argmax"]
        no_readback_audit_passed = (
            token_out_no_readback["host_boundary_audit"]["sampled_token_readbacks"] == 0
            and token_out_no_readback["host_boundary_audit"]["token_input_host_copies"] == 1
            and token_out_no_readback["host_boundary_audit"]["position_host_copies"] == 1
            and token_out_no_readback["host_boundary_audit"]["rope_index_host_copies"] == 1
            and token_out_no_readback["host_boundary_audit"]["page_table_host_copies"] == 1
            and not token_out_no_readback["sampling_path"]["force_argmax"]
        )
        status = (
            "pass"
            if trace_probe_passed and topk_topp_smoke_passed and greedy_uses_split_sampling and no_readback_audit_passed
            else "fail"
        )
        detailed_status = {
            "token_out_only": bool(args.token_out_only),
            "all_trace_feedback_assertions_passed": None
            if args.token_out_only
            else all(feedback_probe["assertions"].values()),
            "topk_topp_smoke_passed": None if args.token_out_only else topk_topp_smoke_passed,
            "greedy_force_argmax": token_out_no_readback["sampling_path"]["force_argmax"],
            "greedy_uses_split_sampling": greedy_uses_split_sampling,
            "no_readback_audit_passed": no_readback_audit_passed,
        }
        result = {
            "model_dir": str(args.model_dir),
            "hf_model_id": MODEL_ID,
            "mesh_device": args.mesh_device,
            "fabric_config": args.fabric_config,
            "max_new_tokens": args.max_new_tokens,
            "artifacts": {
                "json": str(args.output_json),
                "stdout": str(args.output_json.with_name(args.output_json.stem + "_stdout.txt")),
            },
            "prompt": {
                "file": str(args.prompt_file),
                "num_tokens": len(prompt_ids),
                "text": prompt_text,
            },
            "token_out": token_out,
            "token_out_no_readback": token_out_no_readback,
            "trace_feedback_probe": feedback_probe,
            "topk_topp_smoke": topk_topp_smoke,
            "status": status,
            "detailed_status": detailed_status,
            "summary": {
                "token_out_ttft_ms": None,
                "token_out_decode_t_s_u_including_first_trace_capture": None,
                "token_out_steady_replay_t_s_u_excluding_first_trace_capture": None,
                "token_out_no_readback_ttft_ms": token_out_no_readback["ttft_ms"],
                "token_out_no_readback_steady_replay_t_s_u": token_out_no_readback["steady_replay_t_s_u"],
                "token_out_no_readback_steady_replay_ms_per_token": token_out_no_readback[
                    "steady_replay_ms_per_token"
                ],
                "trace_probe_passed": trace_probe_passed,
                "topk_topp_smoke_passed": topk_topp_smoke_passed,
                "no_readback_audit_passed": no_readback_audit_passed,
            },
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh_device, args.fabric_config)


if __name__ == "__main__":
    main()
