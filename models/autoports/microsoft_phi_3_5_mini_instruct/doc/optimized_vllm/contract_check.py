# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free optimized-vLLM contract checks for Phi-3.5-mini."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from models.autoports.microsoft_phi_3_5_mini_instruct.tt.generator import Phi35MiniGenerator
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.generator_vllm import Phi3ForCausalLM


def _expect(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def run_checks() -> dict:
    failures: list[str] = []

    caps = Phi3ForCausalLM.model_capabilities
    _expect(caps.get("supports_async_decode") is True, "supports_async_decode must be true", failures)
    _expect(
        caps.get("tt_async_decode_allows_overlap") is False,
        "scheduler overlap must remain disabled without overlap proof",
        failures,
    )
    _expect(caps.get("supports_sample_on_device") is True, "on-device sampling capability must be true", failures)
    _expect(caps.get("supports_prefix_caching") is False, "prefix caching must not be claimed", failures)

    adapter_src = inspect.getsource(Phi3ForCausalLM)
    _expect("read_from_device=read_from_device" in adapter_src, "adapter must pass split-read flag through", failures)
    _expect("read_decode_output" in adapter_src, "adapter must expose read_decode_output", failures)
    _expect("process_decode_output_host" in adapter_src, "adapter must expose process_decode_output_host", failures)
    _expect("sampling_params is None" in adapter_src, "adapter must reject host-sampling fallback", failures)

    decode_src = inspect.getsource(Phi35MiniGenerator.decode_forward_token_out)
    ensure_src = inspect.getsource(Phi35MiniGenerator._ensure_decode_trace)
    traced_src = inspect.getsource(Phi35MiniGenerator._decode_next_token_traced)
    mode_key_src = inspect.getsource(Phi35MiniGenerator._sampling_mode_key)
    page_table_src = inspect.getsource(Phi35MiniGenerator._coerce_page_table_vllm)

    _expect("return sampled" in decode_src, "decode_forward(..., read_from_device=False) must return device output", failures)
    _expect("ttnn.execute_trace" in traced_src and "blocking=False" in traced_src, "model trace replay must be nonblocking", failures)
    _expect("capture_sampling=True" in decode_src, "vLLM decode must request token-out trace capture", failures)
    _expect("if self.trace.capture_sampling" in traced_src, "hot decode must have a token-out trace replay branch", failures)
    _expect("sampled = self.trace.sampled" in traced_src, "vLLM hot decode must return captured device sampled output", failures)
    _expect("self.sampling.sample(logits, enable_trace=False, tt_out_tok=self.trace.token_input)" in ensure_src, "token-out trace must capture sampler with tt_out_tok feedback", failures)
    _expect("skip_precompile=True" in traced_src, "standalone split-sampling fallback must skip active-trace precompile", failures)
    _expect("force_argmax_sampling" in mode_key_src, "sampling mode key must include force-argmax state", failures)
    _expect(
        "sampling_mode_key" in ensure_src,
        "sampling mode changes must recapture the token-out trace before replay",
        failures,
    )
    _expect("torch.equal" in page_table_src, "page table equality must be checked on host", failures)
    _expect(
        "return self._vllm_page_table_device, False" in page_table_src,
        "unchanged page table must avoid copy",
        failures,
    )
    _expect(
        "ttnn.copy_host_to_device_tensor" in page_table_src,
        "changed page table must refresh persistent tensor in place",
        failures,
    )

    hot_sources = {
        "adapter": adapter_src,
        "decode_forward_token_out": decode_src,
        "_decode_next_token_traced": traced_src,
    }
    forbidden = ("torch.argmax", "ttnn.argmax", "logits_to_torch", "ttnn.to_torch", ".cpu(")
    for name, source in hot_sources.items():
        hits = [token for token in forbidden if token in source]
        _expect(not hits, f"{name} contains forbidden hot-path host/logit fallback tokens: {hits}", failures)

    result = {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "checks": {
            "supports_async_decode": caps.get("supports_async_decode"),
            "tt_async_decode_allows_overlap": caps.get("tt_async_decode_allows_overlap"),
            "supports_sample_on_device": caps.get("supports_sample_on_device"),
            "decode_returns_device_output_when_split": "return sampled" in decode_src,
            "nonblocking_execute_trace": "ttnn.execute_trace" in traced_src and "blocking=False" in traced_src,
            "vllm_requests_token_out_trace": "capture_sampling=True" in decode_src,
            "captured_device_sampled_output": "sampled = self.trace.sampled" in traced_src,
            "token_out_trace_captures_sampling": "self.sampling.sample(logits, enable_trace=False, tt_out_tok=self.trace.token_input)" in ensure_src,
            "sampling_mode_recapture": "sampling_mode_key" in ensure_src,
            "unchanged_page_table_no_copy": "return self._vllm_page_table_device, False" in page_table_src,
            "adapter_rejects_host_sampling": "sampling_params is None" in adapter_src,
        },
    }
    return result


def main() -> None:
    result = run_checks()
    out = Path("models/autoports/microsoft_phi_3_5_mini_instruct/readiness_vllm/optimized_vllm_contract_checks.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
