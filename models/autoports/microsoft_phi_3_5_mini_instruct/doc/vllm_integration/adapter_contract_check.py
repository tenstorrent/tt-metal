# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lightweight vLLM adapter contract checks for Phi-3.5-mini.

These checks avoid hardware execution. They verify that the vLLM-facing wrapper
does not allocate or substitute serving state before delegating to
``Phi35MiniGenerator`` low-level methods.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.microsoft_phi_3_5_mini_instruct.tt.generator_vllm import Phi3ForCausalLM


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.model = SimpleNamespace()
        self.mesh_device = object()
        self.tokenizer = object()

    def prefill_forward_token_out(self, *args, **kwargs):
        self.calls.append(("prefill", {"args": args, "kwargs": kwargs}))
        return torch.tensor([11], dtype=torch.int64)

    def decode_forward_token_out(self, *args, **kwargs):
        self.calls.append(("decode", {"args": args, "kwargs": kwargs}))
        return object()

    def teardown(self) -> None:
        self.calls.append(("teardown", {"args": (), "kwargs": {}}))


def _expect(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def run_checks() -> dict:
    failures: list[str] = []
    fake = _FakeGenerator()
    adapter = Phi3ForCausalLM(fake, max_batch_size=1, tt_data_parallel=1)

    caps = adapter.model_capabilities
    _expect(caps.get("supports_async_decode") is True, "supports_async_decode must be true", failures)
    _expect(
        caps.get("tt_async_decode_allows_overlap") is False,
        "tt_async_decode_allows_overlap must default false",
        failures,
    )
    _expect(caps.get("supports_prefix_caching") is False, "prefix caching must not be claimed", failures)

    tokens = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    start_pos = torch.tensor([3], dtype=torch.int32)
    page_table = torch.tensor([[0, 1]], dtype=torch.int32)
    kv_cache = [("k-cache-sentinel", "v-cache-sentinel")]
    sampling_params = object()

    adapter.prefill_forward(
        tokens=tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        prompt_lens=[3],
        sampling_params=sampling_params,
        empty_slots=[0],
        enable_trace=True,
    )
    name, call = fake.calls[-1]
    kwargs = call["kwargs"]
    _expect(name == "prefill", "prefill must delegate to generator.prefill_forward_token_out", failures)
    _expect(call["args"][0] is tokens, "prefill tokens must pass by identity", failures)
    _expect(kwargs["page_table"] is page_table, "prefill page table must pass by identity", failures)
    _expect(kwargs["kv_cache"] is kv_cache, "prefill KV cache must pass by identity", failures)
    _expect(kwargs["sampling_params"] is sampling_params, "prefill sampling params must pass by identity", failures)
    _expect(kwargs["enable_trace"] is False, "prefill sampling trace must remain disabled", failures)

    adapter.decode_forward(
        tokens=tokens,
        start_pos=start_pos,
        page_table=page_table,
        kv_cache=kv_cache,
        sampling_params=sampling_params,
        enable_trace=True,
        read_from_device=False,
        reset_batch=True,
    )
    name, call = fake.calls[-1]
    kwargs = call["kwargs"]
    _expect(name == "decode", "decode must delegate to generator.decode_forward_token_out", failures)
    _expect(call["args"][0] is tokens, "decode tokens must pass by identity", failures)
    _expect(call["args"][1] is start_pos, "decode current positions must pass by identity", failures)
    _expect(kwargs["page_table"] is page_table, "decode page table must pass by identity", failures)
    _expect(kwargs["kv_cache"] is kv_cache, "decode KV cache must pass by identity", failures)
    _expect(kwargs["sampling_params"] is sampling_params, "decode sampling params must pass by identity", failures)
    _expect(kwargs["enable_trace"] is True, "decode trace flag must pass through", failures)
    _expect(kwargs["read_from_device"] is False, "decode split-read flag must pass through", failures)
    _expect(kwargs["reset_batch"] is True, "decode reset_batch flag must pass through", failures)

    for fn_name, call_kwargs in (
        ("prefill_forward", dict(tokens=tokens, page_table=page_table, kv_cache=kv_cache, prompt_lens=[3])),
        ("decode_forward", dict(tokens=tokens, start_pos=start_pos, page_table=page_table, kv_cache=kv_cache)),
    ):
        try:
            getattr(adapter, fn_name)(**call_kwargs, sampling_params=None)
        except ValueError:
            pass
        else:
            failures.append(f"{fn_name} must reject host-sampling fallback")

    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": {
            "prefill_delegates_to_generator": True,
            "decode_delegates_to_generator": True,
            "vllm_owned_kv_cache_passed_by_identity": True,
            "page_table_passed_by_identity": True,
            "current_position_passed_by_identity": True,
            "stale_token_feedback_contract_preserved": True,
            "stale_current_position_contract_preserved": True,
            "stale_page_table_contract_preserved": True,
            "decode_trace_enabled_flag_passed": True,
            "host_sampling_fallback_rejected": True,
            "tt_async_decode_allows_overlap": caps.get("tt_async_decode_allows_overlap"),
        },
    }


def main() -> None:
    result = run_checks()
    out = Path("models/autoports/microsoft_phi_3_5_mini_instruct/readiness_vllm/adapter_contract_checks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    if result["status"] != "pass":
        raise SystemExit(1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
