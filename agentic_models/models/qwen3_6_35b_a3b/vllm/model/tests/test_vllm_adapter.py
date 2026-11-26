# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of the vLLM adapter (``tt/generator_vllm.py``) against vllm-tt-plugin f6fe76b. No device is opened.

* the capabilities the plugin reads, for the bare import and for the served profile (QWEN_FAST=1 QWEN_SPEC=3
  QWEN_PREFIX_CACHE=1): device top-k 32, plugin patch P1 (``preserve_prefill_state_slots``), none of the
  P2-P4 capabilities (this model keeps vLLM's mamba "align" mode, plugin patch P5);
* every hook the plugin calls accepts the keywords the plugin passes;
* the config-time hooks (KV spec, pool size, spec plan) and the device-free rejections;
* the vendored edits: the ``ttnn.bringup`` MoE forks, ``exact_state=True`` on the GDN ops, and the 32-wide sampler.

  agentic-models test --model qwen3_6_35b_a3b --phase vllm -k vllm_adapter
"""

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PACKAGE = "agentic_models.models.qwen3_6_35b_a3b.vllm"
TT = Path(__file__).resolve().parents[2] / "tt"
SERVED_ENV = {"QWEN_FAST": "1", "QWEN_SPEC": "3", "QWEN_PREFIX_CACHE": "1"}
#: Capabilities of plugin patches P2-P4 (other models). Declaring one would change this model's validated plugin path.
OTHER_PATCH_CAPABILITIES = (
    "supports_prefill_sampling_mask",
    "supports_original_prompt_lens",
    "prefix_cache_recurrent_state_in_model",
)


def adapter_class():
    from agentic_models.models.qwen3_6_35b_a3b.vllm.tt.generator_vllm import QwenForCausalLM

    return QwenForCausalLM


def capabilities(env):
    """model_capabilities is computed from the environment at class definition: import in a fresh process."""
    code = (
        f"import json; from {PACKAGE}.tt.generator_vllm import QwenForCausalLM as C; "
        "print('CAPS=' + json.dumps(C.model_capabilities))"
    )
    clean = {k: v for k, v in os.environ.items() if not k.startswith("QWEN_")}
    out = subprocess.run(
        [sys.executable, "-c", code], env={**clean, **env}, capture_output=True, text=True, check=True
    ).stdout
    return json.loads(next(line for line in out.splitlines() if line.startswith("CAPS="))[5:])


def test_capabilities_bare_import():
    caps = capabilities({})
    assert caps["preserve_prefill_state_slots"] is True
    assert caps["max_device_top_k"] == 32
    assert caps["output_tokens_per_step"] == 1
    assert caps["supports_chunked_prefill"] and caps["supports_sample_on_device"] and caps["supports_async_decode"]
    assert not caps["supports_prefix_caching"] and not caps["supports_spec_decode"]
    assert not caps["supports_device_penalties"]
    assert not any(key in caps for key in OTHER_PATCH_CAPABILITIES)


def test_capabilities_served_profile():
    caps = capabilities(SERVED_ENV)
    assert caps["preserve_prefill_state_slots"] is True
    assert caps["max_device_top_k"] == 32
    assert caps["supports_device_penalties"] is True
    assert caps["supports_prefix_caching"] is True
    assert caps["supports_spec_decode"] and caps["supports_async_spec_decode"] and caps["spec_ids_are_sampled"]
    assert caps["spec_requirements"] == ["device_propose"] and caps["spec_hidden_handoff"] == ["on_device"]
    assert not any(key in caps for key in OTHER_PATCH_CAPABILITIES)
    assert not caps["supports_device_grammar"] and not caps["supports_compact_host_logits"]


def test_sampling_limits_are_32_wide():
    from agentic_models.models.qwen3_6_35b_a3b.vllm.tt.serving_capabilities import QwenServingCapabilities as S

    assert S.serving_capabilities["max_device_top_k"] == 32
    # The sampler is built with max_top_k 32 and no 64-wide draw (ttnn.sampling at the pin takes k <= 32).
    calls = [
        n
        for n in ast.walk(ast.parse((TT / "generator.py").read_text()))
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "SimpleNamespace"
    ]
    keywords = {k.arg: k.value for call in calls for k in call.keywords}
    assert ast.literal_eval(keywords["max_top_k"]) == 32
    assert "device_sampling_draw_width" not in keywords
    # The served (fast) generator captures only the 32-wide sampling traces, and serving requires only those.
    cls = next(
        n
        for n in ast.parse((TT / "fast_generator.py").read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "FastQwenGenerator"
    )
    fast = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "prepare")
    captured = {n.value for n in ast.walk(fast) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    assert {"sampling", "sampling_pen"} <= captured and not any(s.startswith("sampling_64") for s in captured)
    assert '"sampling_64"}' not in (TT / "serving.py").read_text()


def gdn_and_moe_calls():
    calls = {}
    for name in ("fast_gdn.py", "multichip_decoder.py"):
        for node in ast.walk(ast.parse((TT / name).read_text())):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                calls.setdefault(node.func.attr, []).append((ast.unparse(node.func.value), node))
    return calls


def test_moe_calls_use_the_bringup_forks():
    calls = gdn_and_moe_calls()
    for op in ("moe_fused_swiglu", "post_combine_reduce"):
        assert [owner for owner, _ in calls[op]] == ["ttnn.bringup"], op


def test_gdn_calls_opt_into_exact_state():
    calls = gdn_and_moe_calls()
    found = calls["fused_recurrent_gated_delta_rule"] + calls["chunk_gated_delta_rule"]
    assert len(found) == 3
    for owner, call in found:
        assert owner == "ttnn.transformer"
        assert {k.arg: ast.literal_eval(k.value) for k in call.keywords if k.arg == "exact_state"} == {
            "exact_state": True
        }


# Keyword sets the plugin (f6fe76b) passes; file:line of the call site.
PLUGIN_CALLS = {
    # loader.py:38
    "initialize_vllm_model": (
        (None, None, 32),
        dict(max_seq_len=262144, tt_data_parallel=1, optimizations="performance"),
    ),
    # worker.py:760
    "get_max_tokens_all_users": (
        (),
        dict(model_name="m", num_devices=4, tt_data_parallel=1, max_model_len=262144, max_num_seqs=32),
    ),
    # model_runner.py:4910-4946
    "warmup_model_prefill": ((), dict(kv_cache=None, can_sample_on_device=True, enable_trace=False)),
    "warmup_model_decode": (
        (),
        dict(kv_cache=None, max_batch_size=32, num_blocks=8192, can_sample_on_device=True, enable_trace=True),
    ),
    # model_runner.py:797
    "allocate_kv_cache_per_layer": (([],), {}),
    # model_runner.py:1866
    "propose_draft_tokens": ((3, None, None, None), dict(hidden=None)),
    # async_decode.py:1539/1675, 1783
    "read_decode_output": ((None,), dict(async_read=True)),
    "process_decode_output_host": ((None,), dict(is_tokens=True)),
    "release_request": ((0,), {}),
    "release_persistent_capture": ((), {}),
}


@pytest.mark.parametrize("hook", sorted(PLUGIN_CALLS))
def test_hooks_accept_the_plugin_call(hook):
    cls = adapter_class()
    args, kwargs = PLUGIN_CALLS[hook]
    method = getattr(cls, hook)
    bound_self = () if isinstance(inspect.getattr_static(cls, hook), (staticmethod, classmethod)) else (object(),)
    inspect.signature(method).bind(*bound_self, *args, **kwargs)


class FakeGenerator:
    def __init__(self):
        self.cache = object()
        self.prepared = False
        self.prepare_calls = 0
        self.traces, self.counters, self.trace_replays, self.sampling_trace_replays = {}, {}, {}, {}

    def prepare(self):
        self.prepare_calls += 1
        self.prepared = True


def bare_adapter(generator=None):
    adapter = object.__new__(adapter_class())
    adapter.generator = generator
    adapter.decode_route_counts = {"device": 0, "host_logits": 0}
    return adapter


def test_warmup_prepares_once_on_the_vllm_allocation(capsys):
    g = FakeGenerator()
    adapter = bare_adapter(g)
    adapter.warmup_model_prefill(kv_cache=g.cache, can_sample_on_device=True, enable_trace=False)
    adapter.warmup_model_decode(
        kv_cache=g.cache, max_batch_size=32, num_blocks=8192, can_sample_on_device=True, enable_trace=True
    )
    assert g.prepare_calls == 1
    with pytest.raises(ValueError, match="trace_mode=decode_only"):
        adapter.warmup_model_prefill(kv_cache=g.cache, can_sample_on_device=True, enable_trace=True)
    with pytest.raises(ValueError, match="exact vLLM allocation"):
        adapter.warmup_model_decode(
            kv_cache=object(), max_batch_size=32, num_blocks=1, can_sample_on_device=True, enable_trace=True
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(max_batch_size=32, tt_data_parallel=2), "one 1x4 mesh"),
        (dict(max_batch_size=32, optimizations="accuracy"), "datatype-sweep"),
        (dict(max_batch_size=33), "1 through 32"),
    ],
)
def test_initialize_rejects_unsupported_layouts_before_touching_the_device(kwargs, match):
    with pytest.raises(ValueError, match=match):
        adapter_class().initialize_vllm_model(None, None, max_seq_len=1024, **kwargs)


def test_kv_cache_spec_covers_the_full_attention_layers():
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    layer_types = (["linear_attention"] * 3 + ["full_attention"]) * 10
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(layer_types=layer_types, num_key_value_heads=2, head_dim=256)
        )
    )
    spec = adapter_class().get_kv_cache_spec(config)
    assert list(spec) == [f"model.layers.{i}.self_attn" for i in range(10)]
    for value in spec.values():
        assert isinstance(value, FullAttentionSpec)
        assert (value.block_size, value.num_kv_heads, value.head_size, value.dtype) == (32, 2, 256, torch.bfloat16)


def test_kv_pool_follows_the_environment(monkeypatch):
    _, kwargs = PLUGIN_CALLS["get_max_tokens_all_users"]
    monkeypatch.delenv("QWEN_KV_POOL_TOKENS", raising=False)
    assert adapter_class().get_max_tokens_all_users(**kwargs) == 262144
    monkeypatch.setenv("QWEN_KV_POOL_TOKENS", "655360")
    assert adapter_class().get_max_tokens_all_users(**kwargs) == 655360


def test_spec_plan(monkeypatch):
    from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

    cls = adapter_class()
    monkeypatch.delenv("QWEN_SPEC", raising=False)
    assert isinstance(cls.spec_plan(None, 32, 3), SpecReject)
    monkeypatch.setenv("QWEN_SPEC", "3")
    plan = cls.spec_plan(None, 32, 3)
    assert isinstance(plan, SpecPlan)
    assert plan.effective_k == 3 and plan.accept_modes == ("argmax_ids",) and plan.supports_narrow_decode
    reject = cls.spec_plan(None, 32, 2)
    assert isinstance(reject, SpecReject) and reject.supported_k == (3,)


def test_plugin_logprob_sentinel_is_normalized():
    from vllm_tt_plugin.model_input import TTSamplingParams

    params = TTSamplingParams(temperature=[1.0, 0.0], top_k=[20, 1], top_p=[0.95, 1.0], num_logprobs=[-2, 3])
    kwargs = {"sampling_params": params}
    adapter_class()._translate_sampling(kwargs)
    assert kwargs["sampling_params"].num_logprobs == [0, 3]
    assert kwargs["sampling_params"].top_k == [20, 1]


def test_prefill_rejects_multimodal_inputs():
    adapter = bare_adapter(generator=None)
    with pytest.raises(ValueError, match="pixel_values"):
        adapter.prefill_forward(
            torch.zeros(1, 32, dtype=torch.int64), pixel_values=[torch.zeros(1)], prompt_lens=[32], start_pos=[0]
        )


class FakeSpecGenerator:
    """Records the ordinary decode the verify fallback runs; spec_verify must not be reached."""

    def __init__(self):
        self.spec = SimpleNamespace(U=8)
        self.calls = []

    def decode_forward_v1(self, tokens, start_pos, **kwargs):
        self.calls.append((tokens.clone(), start_pos.clone(), dict(kwargs), dict(self._spec_pending_accepted)))
        return (tokens.reshape(-1) + 1).to(torch.int64).reshape(-1, 1)

    def spec_verify(self, *args):
        raise AssertionError("the verify graph serves slots < 8 only")


def test_verify_with_live_rows_past_the_verify_graph_decodes_narrow():
    """Plugin f6fe76b keeps proposals across a prefill step: a verify can arrive with live rows in slots >= 8."""
    from vllm_tt_plugin.spec_decode import accept_greedy_drafts

    g = FakeSpecGenerator()
    adapter = bare_adapter(g)
    B, K = 11, 3
    tokens = torch.arange(B * (1 + K), dtype=torch.int64).reshape(B, 1 + K)
    start_pos = torch.arange(B).unsqueeze(1) * 100 + torch.arange(1 + K)
    accepted = torch.ones(B, dtype=torch.int32)
    accepted[0] = 3
    num_valid = torch.zeros(B, dtype=torch.int32)
    num_valid[:4] = K
    out = adapter.decode_forward(
        tokens, start_pos, spec_mode="argmax_ids", num_valid_drafts=num_valid, accepted_counts=accepted,
        draft_token_ids=tokens[:, 1:], read_from_device=False,
    )
    assert len(g.calls) == 1
    col0, pos0, kwargs, pending = g.calls[0]
    assert torch.equal(col0, tokens[:, :1]) and torch.equal(pos0, start_pos[:, 0])
    assert kwargs["read_from_device"] is True and pending[0] == 3
    assert g._spec_pending_accepted is None and adapter._spec_last == ("ordinary", None)
    assert out.spec_mode == "argmax_ids" and out.argmax_ids.shape == (B, 1)
    # the walk reads the narrow answer as one committed token per row, whatever drafts were scheduled
    committed, counts = accept_greedy_drafts(out.argmax_ids, tokens[:, 1:][:, :0].to(torch.int32), num_valid)
    assert torch.equal(counts, torch.ones(B, dtype=torch.int32))
    assert torch.equal(committed, (tokens[:, :1] + 1).to(torch.int32))
