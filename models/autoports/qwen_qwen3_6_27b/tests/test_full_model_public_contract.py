import inspect
import json
from pathlib import Path

import ttnn
from models.autoports.qwen_qwen3_6_27b.tt.generator import Qwen36Generator
from models.autoports.qwen_qwen3_6_27b.tt.model import Qwen36Model
from models.autoports.qwen_qwen3_6_27b.tt.precision_config import load_precision_config, safe_baseline_config

ROOT = Path("models/autoports/qwen_qwen3_6_27b")


def test_precision_config_is_strict_and_resolves_runtime_policy():
    config = load_precision_config(safe_baseline_config())
    full = config.policy_for(0, "full_attention")
    linear = config.policy_for(1, "linear_attention")
    assert full.attention_weight_dtype == ttnn.bfloat16
    assert full.cache_dtype == ttnn.bfloat8_b
    assert full.qkv_fidelity == ttnn.MathFidelity.HiFi2
    assert full.mlp_fidelity == ttnn.MathFidelity.LoFi
    assert linear.linear_input_weight_dtype == ttnn.bfloat4_b
    assert linear.linear_recurrent_state_dtype == ttnn.bfloat8_b
    assert config.ccl_dtype("token_mixer") == ttnn.bfloat16


def test_precision_config_rejects_ignored_sampling_assumption():
    raw = safe_baseline_config()
    raw["logits_sampling"]["sampled_token_dtype"] = "BF16"
    try:
        load_precision_config(raw)
    except ValueError as error:
        assert "UINT32" in str(error)
    else:
        raise AssertionError("an unsupported sampler token dtype must not be silently ignored")


def test_context_contract_matches_public_prefill_limit():
    contract = json.loads((ROOT / "doc/context_contract.json").read_text())
    assert contract["current_supported_context"] == Qwen36Generator.MAX_PREFILL_TOKENS
    assert contract["full_model"]["batch_1_supported_context"] == Qwen36Generator.MAX_PREFILL_TOKENS
    assert contract["full_model"]["batch_1_decode_cache_context"] == 262144


def test_public_split_sampling_boundary_is_explicit():
    setup = inspect.signature(Qwen36Generator.setup_token_out_decode)
    assert {"tokens", "positions", "page_table", "kv_cache", "active_mask", "sampling_params"} <= set(setup.parameters)
    step = inspect.signature(Qwen36Generator.token_out_decode_step)
    assert {"page_table", "readback"} <= set(step.parameters)


def test_prefill_request_state_cleanup_clears_decoder_aliases():
    class Layer:
        pass

    layer = Layer()
    for name in (
        "_sequence_masks",
        "_conv_state_selector_chunks",
        "_sequence_mask",
        "_conv_state_selectors",
        "_cache_page_table",
    ):
        setattr(layer, name, object())
    model = Qwen36Model.__new__(Qwen36Model)
    model.layers = [layer]
    model.clear_prefill_request_state()
    assert all(value is None for name, value in vars(layer).items() if name.startswith("_"))


def test_trace_release_deallocates_only_owned_persistent_inputs(monkeypatch):
    calls = []

    class Sampling:
        def reset_trace(self):
            calls.append(("sampling_reset",))

    generator = Qwen36Generator.__new__(Qwen36Generator)
    generator.mesh_device = object()
    generator.sampling = Sampling()
    generator._decode_trace_id = "decode"
    generator._compat_trace_id = "compat"
    shared_token = object()
    generator._trace_token = shared_token
    generator._trace_position = object()
    generator._trace_active_mask = object()
    generator._trace_active_state_mask = object()
    # Exercise the double-free guard even though current production captures
    # allocate compatibility inputs independently.
    generator._compat_token = shared_token
    generator._compat_position = object()
    caller_page_table = object()
    trace_logits = object()
    compat_logits = object()
    generator._trace_page_table = caller_page_table
    generator._trace_logits = trace_logits
    generator._trace_sampled = object()
    generator._compat_logits = compat_logits
    generator._trace_cache_backups = None

    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: calls.append(("sync", mesh)))
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, trace_id: calls.append(("release", trace_id)))
    monkeypatch.setattr(ttnn, "deallocate", lambda tensor: calls.append(("deallocate", tensor)))

    generator._release_traces()

    assert calls[:4] == [
        ("sync", generator.mesh_device),
        ("release", "decode"),
        ("release", "compat"),
        ("sampling_reset",),
    ]
    deallocated = [call[1] for call in calls if call[0] == "deallocate"]
    assert len(deallocated) == 5
    assert len({id(tensor) for tensor in deallocated}) == 5
    assert shared_token in deallocated
    assert caller_page_table not in deallocated
    assert trace_logits not in deallocated
    assert compat_logits not in deallocated
    assert all(
        getattr(generator, name) is None
        for name in (
            "_trace_token",
            "_trace_position",
            "_trace_active_mask",
            "_trace_active_state_mask",
            "_trace_page_table",
            "_trace_logits",
            "_trace_sampled",
            "_compat_token",
            "_compat_position",
            "_compat_logits",
        )
    )
