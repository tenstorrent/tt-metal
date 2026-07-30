# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused vLLM adapter async/stale-input contract probe.

This is not a serving benchmark. It instantiates the same adapter class used by
the TT vLLM plugin, allocates a vLLM-owned KV cache, and drives
``decode_forward(read_from_device=False)`` followed by
``read_decode_output(async_read=True)`` and ``process_decode_output_host``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import ttnn
from transformers import AutoConfig, AutoTokenizer

from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.generator_vllm import (
    Llama31_8B_InstructForCausalLM,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.model import MODEL_ID
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device
from models.common.sampling import SamplingParams


DEFAULT_MODEL_DIR = Path("models/autoports/meta_llama_llama_3_1_8b_instruct")
DEFAULT_PROMPT_FILE = DEFAULT_MODEL_DIR / "doc" / "optimized_full_model" / "prompt_128.txt"
DEFAULT_OUTPUT_JSON = DEFAULT_MODEL_DIR / "doc" / "optimized_vllm" / "vllm_async_contract_probe.json"


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


def _first_token(value: Any) -> int:
    tokens = value[0] if isinstance(value, tuple) else value
    if isinstance(tokens, torch.Tensor):
        return int(tokens.reshape(-1)[0].item())
    raise TypeError(f"Expected torch token tensor, got {type(tokens)!r}")


def _is_device_decode_output(value: Any) -> bool:
    if isinstance(value, ttnn.Tensor):
        return True
    if isinstance(value, tuple):
        return any(isinstance(item, ttnn.Tensor) for item in value)
    return False


def _read_async(adapter: Llama31_8B_InstructForCausalLM, tt_out: Any) -> tuple[Any, int]:
    read_value, events = adapter.read_decode_output(tt_out, async_read=True)
    for event in events:
        ttnn.event_synchronize(event)
    return adapter.process_decode_output_host(read_value, is_tokens=True), len(events)


def _decode_step(
    adapter: Llama31_8B_InstructForCausalLM,
    *,
    label: str,
    token_id: int,
    current_pos: int,
    page_table: torch.Tensor,
    kv_cache: Any,
    sampling_params: SamplingParams,
    reset_batch: bool = False,
) -> tuple[int, dict[str, Any]]:
    before = adapter.trace_counters()
    tt_out = adapter.decode_forward(
        torch.tensor([[int(token_id)]], dtype=torch.int64),
        torch.tensor([int(current_pos)], dtype=torch.int32),
        page_table,
        kv_cache,
        enable_trace=True,
        read_from_device=False,
        sampling_params=sampling_params,
        reset_batch=reset_batch,
    )
    returned_device_tensors = _is_device_decode_output(tt_out)
    host_out, read_event_count = _read_async(adapter, tt_out)
    ttnn.synchronize_device(adapter.mesh_device)
    after = adapter.trace_counters()
    sampled = _first_token(host_out)
    record = {
        "label": label,
        "input_token": int(token_id),
        "current_pos": int(current_pos),
        "sampled_token": sampled,
        "decode_forward_returned_device_tensors": returned_device_tensors,
        "async_read_event_count": read_event_count,
        "counter_delta": _counter_delta(before, after),
        "counters_after": after,
    }
    return sampled, record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--hf-model", default=MODEL_ID)
    parser.add_argument("--mesh-device", default="T3K")
    parser.add_argument("--fabric-config", default="FABRIC_1D_RING")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args()

    if args.hf_model != MODEL_ID:
        raise ValueError(f"This probe is pinned to {MODEL_ID}, got {args.hf_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, local_files_only=True)
    prompt_text = args.prompt_file.read_text(encoding="utf-8").strip()
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)
    if len(prompt_ids) > args.max_model_len:
        raise ValueError(f"Prompt has {len(prompt_ids)} tokens, beyond max_model_len={args.max_model_len}")

    mesh_device = open_readiness_mesh_device(args.mesh_device, args.fabric_config)
    adapter: Llama31_8B_InstructForCausalLM | None = None
    try:
        hf_config = AutoConfig.from_pretrained(args.hf_model, local_files_only=True)
        adapter = Llama31_8B_InstructForCausalLM.initialize_vllm_model(
            hf_config=hf_config,
            mesh_device=mesh_device,
            max_batch_size=1,
            max_seq_len=args.max_model_len,
        )
        num_blocks = max(1, (args.max_model_len + args.block_size - 1) // args.block_size)
        local_kv_heads = adapter.model.hf_config.num_key_value_heads // mesh_device.get_num_devices()
        kv_cache_shape = (num_blocks, local_kv_heads, args.block_size, adapter.model.hf_config.head_dim)
        kv_cache = adapter.allocate_kv_cache(kv_cache_shape, torch.bfloat16, adapter.model.n_layers)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        sampling_params = SamplingParams(
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            enable_log_probs=False,
            num_logprobs=0,
        )

        prompt = torch.tensor([prompt_ids], dtype=torch.int64)
        prefill_out = adapter.prefill_forward(
            prompt,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=[len(prompt_ids)],
            sampling_params=sampling_params,
        )
        first_token = _first_token(prefill_out)
        current_pos = len(prompt_ids)

        steps: list[dict[str, Any]] = []
        token, record = _decode_step(
            adapter,
            label="reset_batch_first_decode",
            token_id=first_token,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            reset_batch=True,
        )
        steps.append(record)
        current_pos += 1

        token, record = _decode_step(
            adapter,
            label="unchanged_page_table_device_feedback",
            token_id=token,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )
        steps.append(record)
        current_pos += 1

        changed_host_token = (token + 17) % adapter.model.vocab_size
        token, record = _decode_step(
            adapter,
            label="changed_host_token_refresh",
            token_id=changed_host_token,
            current_pos=current_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )
        steps.append(record)
        current_pos += 1

        jumped_pos = current_pos + 4
        token, record = _decode_step(
            adapter,
            label="changed_current_position_refresh",
            token_id=token,
            current_pos=jumped_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )
        steps.append(record)
        current_pos = jumped_pos + 1

        changed_page_table = page_table.clone()
        changed_page_table[0, -1] = (int(changed_page_table[0, -1].item()) + 1) % num_blocks
        _token, record = _decode_step(
            adapter,
            label="changed_page_table_refresh",
            token_id=token,
            current_pos=current_pos,
            page_table=changed_page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )
        steps.append(record)

        by_label = {step["label"]: step for step in steps}
        assertions = {
            "capabilities_advertise_async_without_overlap": adapter.model_capabilities.get("supports_async_decode")
            is True
            and adapter.model_capabilities.get("tt_async_decode_allows_overlap") is False,
            "all_decode_forward_returns_device_tensors": all(
                step["decode_forward_returned_device_tensors"] for step in steps
            ),
            "all_async_reads_record_events": all(step["async_read_event_count"] >= 1 for step in steps),
            "all_model_replays_nonblocking": all(
                step["counter_delta"].get("model_trace_nonblocking_replays") == 1 for step in steps
            ),
            "unchanged_page_table_not_recopied": by_label["unchanged_page_table_device_feedback"][
                "counter_delta"
            ].get("page_table_host_copies")
            == 0,
            "unchanged_token_feedback_kept_on_device": by_label["unchanged_page_table_device_feedback"][
                "counter_delta"
            ].get("token_input_host_copies")
            == 0,
            "changed_host_token_copied_once": by_label["changed_host_token_refresh"]["counter_delta"].get(
                "token_input_host_copies"
            )
            == 1,
            "changed_host_token_did_not_recopy_position_or_page_table": by_label["changed_host_token_refresh"][
                "counter_delta"
            ].get("position_host_copies")
            == 0
            and by_label["changed_host_token_refresh"]["counter_delta"].get("page_table_host_copies") == 0,
            "changed_current_position_reset_inputs": by_label["changed_current_position_refresh"][
                "counter_delta"
            ].get("position_host_copies")
            == 1
            and by_label["changed_current_position_refresh"]["counter_delta"].get("rope_index_host_copies") == 1,
            "changed_page_table_copied_once": by_label["changed_page_table_refresh"]["counter_delta"].get(
                "page_table_host_copies"
            )
            == 1,
            "changed_page_table_marked_changed_only": by_label["changed_page_table_refresh"]["counter_delta"].get(
                "last_page_table_refresh_changed_only"
            )
            is True,
            "greedy_uses_split_sampling": adapter.trace_counters().get("sampling_force_argmax") is False,
        }
        result = {
            "model_dir": str(args.model_dir),
            "hf_model_id": args.hf_model,
            "mesh_device": args.mesh_device,
            "fabric_config": args.fabric_config,
            "max_model_len": args.max_model_len,
            "block_size": args.block_size,
            "kv_cache_shape": list(kv_cache_shape),
            "prompt_tokens": len(prompt_ids),
            "prefill_first_sampled_token": first_token,
            "steps": steps,
            "assertions": assertions,
            "status": "pass" if all(assertions.values()) else "fail",
            "final_counters": adapter.trace_counters(),
            "model_capabilities": adapter.model_capabilities,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "assertions": assertions}, indent=2))
    finally:
        if adapter is not None:
            adapter.teardown()
        close_readiness_mesh_device(mesh_device, args.fabric_config)


if __name__ == "__main__":
    main()
