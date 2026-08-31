# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.autoports.google_gemma_4_31b.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_31b.tt.generator_vllm import Gemma4ForCausalLM, _page_table_prefix
from models.autoports.google_gemma_4_31b.tt.model import Gemma4FullModel
from models.common.readiness_check import run_vllm_server

ROOT = Path(__file__).resolve().parents[4]


class _SamplingParams:
    temperature = [0.0, 0.0]
    top_k = [262_144, 262_144]


class _FakeGenerator:
    def __init__(self):
        self.prepare_calls = []
        self.next_calls = []
        self.decode_calls = []
        self.release_calls = 0
        self._sampling_trace_id = None
        self.trace_lifecycle_counters = {"release_calls": 0, "release_synchronizations": 0}

    def _release_all_decode_traces(self):
        self.release_calls += 1
        self.trace_lifecycle_counters["release_calls"] += 1

    def prepare_token_out_decode(self, **kwargs):
        self.prepare_calls.append(kwargs)
        return ("prepared-device-token", None)

    def decode_next_token_traced(self, **kwargs):
        self.next_calls.append(kwargs)
        return ("next-device-token", None)

    def decode_forward(self, tokens, start_pos, **kwargs):
        self.decode_calls.append((tokens, start_pos, kwargs))
        return "host-logits"


def _adapter(*, trace_id=7):
    adapter = object.__new__(Gemma4ForCausalLM)
    adapter.generator = _FakeGenerator()
    adapter.model = SimpleNamespace(trace_state=SimpleNamespace(trace_id=trace_id))
    adapter._decode_cache_identity = ((11, 12),)
    adapter._decode_sampling_key = ("greedy", 1)
    adapter._decode_active_batch_size = 1
    adapter.host_sampling_compat = True
    adapter._page_tables_to_tt = lambda page_tables_per_layer, page_table, rows=None: (["tt-page-table"], [3])
    return adapter


def test_capabilities_require_async_traced_greedy_and_disable_prefix_cache():
    capabilities = Gemma4ForCausalLM.model_capabilities
    assert capabilities["supports_async_decode"] is True
    assert capabilities["supports_async_decode_overlap"] is True
    assert capabilities["supports_prefix_caching"] is False
    assert capabilities["supports_sample_on_device"] is True
    assert capabilities["sample_on_device_policy"] == "greedy_only"


def test_vllm_protocol_constructor_and_full_context_hybrid_pool_budget():
    assert any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in inspect.signature(Gemma4ForCausalLM).parameters.values()
    )
    base_tokens = Gemma4ForCausalLM.get_max_tokens_all_users(
        model_name="google/gemma-4-31B",
        num_devices=4,
        tt_data_parallel=1,
        max_model_len=119_552,
        max_num_seqs=32,
    )
    generic_worker_blocks = 32 + (1024 * 32 * 8) // 64
    final_blocks = base_tokens // 64 + generic_worker_blocks
    assert final_blocks == 5 * (119_552 // 64 + 1) + 119_552 // 128


def test_proven_steady_async_decode_ignores_all_stale_host_trace_inputs():
    adapter = _adapter()
    kv_cache = [[object(), object()]]
    adapter._decode_cache_identity = tuple(tuple(id(tensor) for tensor in pair) for pair in kv_cache)

    output = adapter.decode_forward(
        tokens=torch.tensor([[999], [888]], dtype=torch.int32),
        start_pos=torch.tensor([101, -1], dtype=torch.int32),
        page_table=torch.zeros((2, 4), dtype=torch.int32),
        kv_cache=kv_cache,
        sampling_params=_SamplingParams(),
        reset_batch=False,
        enable_trace=True,
        read_from_device=False,
        reuse_device_decode_inputs=True,
    )

    assert output == ("next-device-token", None)
    assert adapter.generator.prepare_calls == []
    assert adapter.generator.next_calls == [{}]


def test_unproven_steady_decode_still_checks_scheduler_page_tables():
    adapter = _adapter()
    kv_cache = [[object(), object()]]
    adapter._decode_cache_identity = tuple(tuple(id(tensor) for tensor in pair) for pair in kv_cache)

    output = adapter.decode_forward(
        tokens=torch.tensor([[17], [0]], dtype=torch.int32),
        start_pos=torch.tensor([102, -1], dtype=torch.int32),
        page_table=torch.zeros((2, 4), dtype=torch.int32),
        kv_cache=kv_cache,
        sampling_params=_SamplingParams(),
        reset_batch=False,
        enable_trace=True,
        read_from_device=False,
    )

    assert output == ("next-device-token", None)
    assert adapter.generator.next_calls == [
        {
            "page_table": ["tt-page-table"],
            "kv_cache": kv_cache,
            "page_table_generations": [3],
        }
    ]


def test_reset_decode_passes_vllm_cache_and_fixed_slot_state_to_canonical_prepare():
    adapter = _adapter()
    kv_cache = [[object(), object()]]
    output = adapter.decode_forward(
        tokens=torch.tensor([[42], [0]], dtype=torch.int32),
        start_pos=torch.tensor([149, -1], dtype=torch.int32),
        page_table=torch.zeros((2, 4), dtype=torch.int32),
        kv_cache=kv_cache,
        sampling_params=_SamplingParams(),
        reset_batch=True,
        enable_trace=True,
        read_from_device=False,
    )

    assert output == ("prepared-device-token", None)
    call = adapter.generator.prepare_calls[0]
    assert call["kv_cache"] is kv_cache
    assert call["page_table"] == ["tt-page-table"]
    assert call["page_table_generations"] == [3]
    assert call["active_batch_size"] == 1
    assert call["first_input_tokens"].tolist() == [42]
    assert call["start_positions"].tolist() == [149]
    assert call["prompt_lengths"] == [149]
    assert call["pad_to_max_batch"] is False
    assert call["top_k"] == 1
    assert call["top_p"] == 0.0
    assert call["temperature"] == 1.0


def test_dynamic_decode_batch_change_recaptures_only_the_active_prefix():
    adapter = _adapter()
    adapter._decode_sampling_key = ("greedy", 1)
    kv_cache = [[object(), object()]]
    adapter._decode_cache_identity = tuple(tuple(id(tensor) for tensor in pair) for pair in kv_cache)

    output = adapter.decode_forward(
        tokens=torch.tensor([[11], [22], [0], [0]], dtype=torch.int32),
        start_pos=torch.tensor([100, 80, -1, -1], dtype=torch.int32),
        page_table=torch.zeros((4, 4), dtype=torch.int32),
        kv_cache=kv_cache,
        sampling_params=_SamplingParams(),
        reset_batch=False,
        enable_trace=True,
        read_from_device=False,
    )

    assert output == ("prepared-device-token", None)
    assert adapter.generator.release_calls == 1
    call = adapter.generator.prepare_calls[0]
    assert call["first_input_tokens"].tolist() == [11, 22]
    assert call["start_positions"].tolist() == [100, 80]
    assert call["active_batch_size"] == 2
    assert adapter._decode_sampling_key == ("greedy", 2)


def test_dynamic_decode_rejects_non_prefix_active_slots():
    adapter = _adapter()
    kv_cache = [[object(), object()]]
    try:
        adapter.decode_forward(
            tokens=torch.tensor([[11], [0], [22]], dtype=torch.int32),
            start_pos=torch.tensor([100, -1, 80], dtype=torch.int32),
            page_table=torch.zeros((3, 4), dtype=torch.int32),
            kv_cache=kv_cache,
            sampling_params=_SamplingParams(),
            enable_trace=True,
            read_from_device=False,
        )
    except ValueError as exc:
        assert "contiguous prefix" in str(exc)
    else:
        raise AssertionError("dynamic decode accepted a hole in the active request prefix")


def test_host_sampling_decode_restores_page_tables_after_trace_release():
    adapter = _adapter()
    kv_cache = [[object(), object()]]

    output = adapter.decode_forward(
        tokens=torch.tensor([[11], [22]], dtype=torch.int32),
        start_pos=torch.tensor([100, 80], dtype=torch.int32),
        page_table=torch.zeros((2, 4), dtype=torch.int32),
        kv_cache=kv_cache,
        sampling_params=None,
        enable_trace=True,
        read_from_device=False,
    )

    assert output == "host-logits"
    assert adapter.generator.release_calls == 1
    tokens, positions, kwargs = adapter.generator.decode_calls[0]
    assert tokens.tolist() == [[11], [22]]
    assert positions.tolist() == [100, 80]
    assert kwargs["page_table"] == ["tt-page-table"]
    assert kwargs["kv_cache"] is kv_cache
    assert kwargs["enable_trace"] is False
    assert kwargs["return_device_logits"] is True


def test_page_table_prefix_is_a_zero_copy_host_view():
    source = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    prefix = _page_table_prefix(source, rows=2)
    assert prefix.shape == (2, 6)
    assert prefix.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert torch.equal(prefix, source[:2])


def test_generator_serving_mode_has_no_hidden_standalone_cache_fallback():
    generator = object.__new__(Gemma4Generator)
    generator._owns_standalone_cache = False
    generator.kv_cache = None
    generator.page_tables = None
    try:
        generator._resolve_cache_pair(None, None)
    except ValueError as exc:
        assert "caller-owned" in str(exc)
    else:
        raise AssertionError("serving mode unexpectedly accepted missing external cache handles")


def test_adapter_source_delegates_token_feedback_without_host_loop():
    source = inspect.getsource(Gemma4ForCausalLM.decode_forward)
    assert "prepare_token_out_decode" in source
    assert "decode_next_token_traced" in source
    assert "read_sampled_token" not in source
    assert "write_teacher_forced_token" not in source
    assert "argmax" not in source.lower()


def test_prefill_and_host_decode_release_prior_trace_before_page_table_conversion():
    source = inspect.getsource(Gemma4ForCausalLM)
    prefill = source[source.index("    def prefill_forward(") : source.index("    def decode_forward(")]
    decode = source[source.index("    def decode_forward(") : source.index("    def read_decode_output(")]
    assert prefill.index("self._release_decode_state()") < prefill.index("self._page_tables_to_tt(")
    assert decode.index("self._release_decode_state()") < decode.index("self._page_tables_to_tt(")


def test_prefill_keeps_live_traces_for_warm_shapes():
    source = inspect.getsource(Gemma4ForCausalLM)
    prefill = source[source.index("    def prefill_forward(") : source.index("    def decode_forward(")]
    # A first-of-its-shape prefill releases the decode traces (its program
    # and shape-cache allocations persist while a trace owns allocator
    # addresses); a warm-shape prefill keeps them live so the next decode
    # refreshes and replays instead of re-capturing.
    assert prefill.index(
        "if not self._keep_decode_traces or prefill_shape not in self._warm_prefill_shapes:"
    ) < prefill.index("self._release_decode_state()")
    assert "self._warm_prefill_shapes.add(prefill_shape)" in prefill
    assert "release_decode_traces=False" in prefill
    generator_prefill = inspect.getsource(Gemma4Generator.prefill_forward)
    assert "release_decode_traces" in generator_prefill
    assert inspect.signature(Gemma4Generator.prefill_forward).parameters["release_decode_traces"].default is True


def test_decode_reset_batch_reuses_matching_live_traces():
    source = inspect.getsource(Gemma4ForCausalLM.decode_forward)
    # reset_batch alone (same batch shape, sampling key, and KV cache) must
    # not release the live traces: prepare_token_out_decode refreshes the
    # trace-bound inputs in place and replays without a capture.
    reuse = source[source.index("steady_trace_reuse = (") : source.index("if must_prepare and not steady_trace_reuse:")]
    assert "not cache_changed" in reuse
    assert "self._decode_sampling_key == sampling_key" in reuse
    assert "self.model.trace_state.trace_id is not None" in reuse
    assert "self.generator._sampling_trace_id is not None" in reuse


def test_page_table_conversion_releases_traces_before_reallocation():
    source = inspect.getsource(Gemma4ForCausalLM._page_tables_to_tt)
    assert "_page_table_state_would_reallocate" in source
    assert source.index("self._release_decode_state()") < source.index("_update_page_table_state(")


def test_dynamic_prepare_releases_trace_before_batch_shaped_page_table_conversion():
    source = inspect.getsource(Gemma4ForCausalLM.decode_forward)
    start = source.index("        must_prepare = (")
    end = source.index("            output = self.generator.prepare_token_out_decode")
    prepare = source[start:end]
    assert prepare.index("self._release_decode_state()") < prepare.index("self._page_tables_to_tt(")
    assert "rows=active_batch_size" in prepare
    assert "decode trace prepare: active_batch=%d" in source
    assert "decode traces ready: active_batch=%d" in source
    release_source = inspect.getsource(Gemma4ForCausalLM._release_decode_state)
    assert "decode traces released after CQ synchronization" in release_source


def test_adapter_lifecycle_uses_vllm_configured_logger():
    module_source = inspect.getsource(inspect.getmodule(Gemma4ForCausalLM))
    assert "from vllm.logger import init_logger" in module_source
    assert 'logger = init_logger(f"vllm.{__name__}")' in module_source
    assert Gemma4ForCausalLM.decode_forward.__globals__["logger"].name.startswith("vllm.")


def test_generator_dynamic_batch_is_explicit_and_trace_release_is_synchronized():
    signature = inspect.signature(Gemma4Generator.prepare_token_out_decode)
    assert signature.parameters["pad_to_max_batch"].default is True
    prepare_source = inspect.getsource(Gemma4Generator.prepare_token_out_decode)
    release_source = inspect.getsource(Gemma4Generator._release_all_decode_traces)
    assert "if pad_to_max_batch and tokens.shape[0] < self.max_batch_size" in prepare_source
    assert "int(tokens.shape[0])" in prepare_source
    assert release_source.index("ttnn.synchronize_device") < release_source.index("ttnn.release_trace")


def test_split_sampler_is_prepared_and_captured_before_first_nonblocking_replay():
    prepare_source = inspect.getsource(Gemma4Generator.prepare_token_out_decode)
    initial_prepare = prepare_source[prepare_source.index("self._prewarm_split_sampling_workloads(") :]
    assert initial_prepare.index("self._prewarm_split_sampling_workloads(") < initial_prepare.index(
        "self.model.capture_decode_trace("
    )
    assert initial_prepare.index("self.model.capture_decode_trace(") < initial_prepare.index(
        "self._capture_sampling_trace("
    )
    assert initial_prepare.index("self._capture_sampling_trace(") < initial_prepare.index(
        "self.model.execute_decode_trace()"
    )
    assert initial_prepare.index("self.model.execute_decode_trace()") < initial_prepare.index(
        "self._execute_sampling_trace()"
    )
    assert "return output" in initial_prepare

    capture_source = inspect.getsource(Gemma4Generator._capture_sampling_trace)
    before_capture = capture_source[: capture_source.index("ttnn.begin_trace_capture")]
    assert "greedy_tp4_sampler.decode_forward" not in before_capture
    assert "force_argmax_sampler.decode_forward" not in before_capture
    assert "sampler.decode_forward" not in before_capture
    assert "ttnn.synchronize_device" in before_capture
    assert capture_source.count("self.greedy_tp4_sampler.decode_forward") == 1
    assert capture_source.count("self.force_argmax_sampler.decode_forward") == 1
    assert capture_source.count("self.sampler.decode_forward") == 1

    initialize_source = inspect.getsource(Gemma4FullModel.initialize_trace_state)
    assert initialize_source.index("stable_tables[identity] = ttnn.clone(table)") < initialize_source.index(
        "ttnn.copy(source, target)"
    )
    assert initialize_source.index("ttnn.copy(source, target)") < initialize_source.index(
        "self.trace_state.page_table_identities ="
    )
    assert "warmed_pairs" in initialize_source


def test_hma_decode_preserves_shared_storage_and_uses_geometry_aware_update():
    adapter_source = inspect.getsource(Gemma4ForCausalLM.allocate_kv_cache_per_layer)
    decoder_source = (ROOT / "models/autoports/google_gemma_4_31b/tt/multichip_decoder.py").read_text()
    assert "ttnn.reshape" not in adapter_source
    assert "cache_geometry_matches" in decoder_source
    assert "config.cache_position_modulo is None and cache_geometry_matches" in decoder_source


def test_bundle_registers_the_autoport_with_unpatched_vllm():
    """Selection must not depend on a patched tenstorrent/vllm.

    The autoport is registered from this checkout's own plugin bundle, which the
    TT plugin picks up from EXTRA_MODELS_DIR ahead of its built-in map. That
    replaces an earlier local patch to the plugin's platform.py adding a
    TT_GEMMA4_TEXT_VER selector: the patch was never pushed, is absent from
    tenstorrent/vllm, and cannot be pushed from this account, so asserting
    against plugin source made this contract depend on an unpushable change.

    Without a registration the checkpoint's Gemma4ForConditionalGeneration
    resolves to models/demos/gemma4 and a server serves the wrong implementation
    while appearing healthy, which Stage 11 rules invalid even when run.py exits 0.
    """
    bundle = ROOT / "models/autoports/vllm_bundles/gemma4_31b_autoport/vllm_metadata.json"
    assert bundle.is_file(), f"missing plugin bundle: {bundle}"
    metadata = json.loads(bundle.read_text())
    # TTPlatform.check_and_update_config prepends "TT" to each declared
    # architecture and requires a matching registration, so the bundle declares
    # the bare HF arch and the plugin derives TTGemma4ForConditionalGeneration.
    assert metadata["arch"] == "Gemma4ForConditionalGeneration"
    assert metadata["main_class"] == ("models.autoports.google_gemma_4_31b.tt.generator_vllm:Gemma4ForCausalLM")


@pytest.mark.xfail(
    strict=False,
    reason=(
        "Stage 10's plugin-side async-decode work was never pushed and is absent "
        "from both tenstorrent/vllm dev and tenstorrent/vllm-tt-plugin main (the "
        "repo tt-shield migrated to, re-checked 2026-08-17): model_runner.py has "
        "no greedy_only sampling policy and async_decode.py has no "
        "reuse_device_decode_inputs, page_tables_changed or "
        "scheduler_updates_page_tables handling. Not a "
        "correctness risk on greedy traffic, because the adapter's decode_forward "
        "declares reuse_device_decode_inputs=False and so takes the conservative "
        "path when the plugin never passes it; it costs the steady async fast "
        "path. Rebuild those hooks and this test should XPASS."
    ),
)
def test_plugin_honors_greedy_only_and_async_decode_reuse():
    plugin = ROOT / "vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin"
    if not plugin.is_dir():
        pytest.skip(f"no nested vLLM plugin checkout at {plugin}")
    runner_source = (plugin / "model_runner.py").read_text()
    async_source = (plugin / "async_decode.py").read_text()
    assert 'sampling_policy == "greedy_only"' in runner_source
    assert 'kwargs["reuse_device_decode_inputs"]' in async_source
    assert "if model_input.page_tables_changed" in async_source
    assert "scheduler_updates_page_tables(scheduler_output)" in async_source


def test_max_context_gate_rejects_http_200_error_payload(monkeypatch, tmp_path, expect_error):
    tokenizer = SimpleNamespace(bos_token_id=2, encode=lambda *_args, **_kwargs: [9])
    monkeypatch.setattr(
        run_vllm_server.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: tokenizer,
    )
    response = SimpleNamespace(
        status_code=200,
        content=b"{}",
        json=lambda: {"error": {"message": "engine failed", "code": 500}},
    )
    monkeypatch.setattr(run_vllm_server.requests, "post", lambda *_args, **_kwargs: response)
    artifact = tmp_path / "max_context.json"

    with expect_error(RuntimeError, "error payload"):
        run_vllm_server._run_max_context_prompt_check(
            server_url="http://localhost:8000",
            hf_model="local-model",
            max_model_len=128,
            output_file=artifact,
        )

    assert '"error"' in artifact.read_text()


def test_prefill_releases_both_reshaped_and_source_normalized_inputs():
    source = inspect.getsource(Gemma4ForCausalLM)
    decoder_source = (ROOT / "models/autoports/google_gemma_4_31b/tt/multichip_decoder.py").read_text()
    assert "source_hidden_states=normed" in decoder_source
    assert "source_hidden_states.is_allocated()" in decoder_source
    assert "if is_decode:\n            normed.deallocate(True)" in decoder_source
    assert "source_hidden_states" not in source


def test_chunked_prefill_releases_full_mlp_norm_before_output_concat():
    decoder_source = (ROOT / "models/autoports/google_gemma_4_31b/tt/multichip_decoder.py").read_text()
    chunked_mlp = decoder_source[
        decoder_source.index("        if not is_decode and normed.shape[-2] > MLP_CHUNK:") : decoder_source.index(
            "        hidden_states = self.layer.post_feedforward_layernorm.forward(mlp_output)"
        )
    ]
    assert chunked_mlp.index("normed.deallocate(True)") < chunked_mlp.index("ttnn.concat(outputs, dim=2)")


def test_full_attention_streams_row_major_chunks_before_one_tile_conversion():
    decoder_source = (ROOT / "models/autoports/google_gemma_4_31b/tt/multichip_decoder.py").read_text()
    stream = decoder_source[
        decoder_source.index("    def _chunked_full_attention_concatenated(") : decoder_source.index(
            "    def _forward_device("
        )
    ]
    assert "layout=ttnn.ROW_MAJOR_LAYOUT" in stream
    assert "chunk_rm = ttnn.to_layout(chunk, ttnn.ROW_MAJOR_LAYOUT)" in stream
    assert "slice_write(\n                chunk_rm,\n                output_rm" in stream
    assert stream.index("q.deallocate(True)") < stream.index("ttnn.to_layout(output_rm, ttnn.TILE_LAYOUT)")


def test_sliding_attention_streams_chunks_and_releases_sources_before_tile_conversion():
    decoder_source = (ROOT / "models/autoports/google_gemma_4_31b/tt/multichip_decoder.py").read_text()
    prefill = decoder_source[
        decoder_source.index("    def _prefill_attention_tp(") : decoder_source.index(
            "    def _chunked_full_attention_concatenated("
        )
    ]
    stream = decoder_source[
        decoder_source.index("    def _chunked_sliding_attention_concatenated(") : decoder_source.index(
            "    def _forward_device("
        )
    ]
    assert "concatenated = self._chunked_sliding_attention_concatenated(" in prefill
    assert "chunked_prefill_sdpa_sliding" not in prefill
    assert "layout=ttnn.ROW_MAJOR_LAYOUT" in stream
    assert "slice_write(\n                chunk_rm,\n                output_rm" in stream
    final_tile = stream.index("ttnn.to_layout(output_rm, ttnn.TILE_LAYOUT)")
    for source in ("q.deallocate(True)", "k.deallocate(True)", "v.deallocate(True)"):
        assert stream.index(source) < final_tile


def test_long_attention_projects_and_reduces_chunks_into_one_tiled_output():
    decoder_source = (ROOT / "models/autoports/google_gemma_4_31b/tt/multichip_decoder.py").read_text()
    prefill = decoder_source[
        decoder_source.index("    def _prefill_attention_tp(") : decoder_source.index(
            "    def _chunked_full_attention_concatenated("
        )
    ]
    stream = decoder_source[
        decoder_source.index("    def _chunked_attention_output_projection(") : decoder_source.index(
            "    def _chunked_full_attention_concatenated("
        )
    ]
    assert "self._chunked_attention_output_projection(concatenated, weights.o_proj, residual=residual)" in prefill
    assert "layout=ttnn.TILE_LAYOUT" in stream
    assert "strategy=ttnn.ShardStrategy.BLOCK" in stream
    assert "padded_chunk_rows = math.ceil(chunk_rows / ttnn.TILE_SIZE) * ttnn.TILE_SIZE" in stream
    assert "write_chunk = ttnn.to_memory_config(chunk_output, shard_memory)" in stream
    assert "slice_write(\n                write_chunk,\n                output" in stream
    assert "ttnn.to_layout(output" not in stream


def test_global_prefill_reads_hma_cache_through_zero_copy_layer_geometry():
    decoder_source = (ROOT / "models/autoports/google_gemma_4_31b/tt/multichip_decoder.py").read_text()
    prefill = decoder_source[
        decoder_source.index("    def _prefill_attention_tp(") : decoder_source.index(
            "    def _chunked_attention_output_projection("
        )
    ]
    view = decoder_source[
        decoder_source.index("    def _paged_cache_read_view(") : decoder_source.index(
            "    def _chunked_attention_output_projection("
        )
    ]
    assert "read_k_cache = self._paged_cache_read_view(k_cache, local_kv_heads, config.head_dim)" in prefill
    assert "read_v_cache = self._paged_cache_read_view(v_cache, local_kv_heads, config.head_dim)" in prefill
    assert "block_size = effective_block_size(cache, head_dim, num_kv_heads)" in view
    assert "cache.dtype != ttnn.bfloat8_b" in view
    assert "cache.layout != ttnn.TILE_LAYOUT" in view
    assert "block_size % ttnn.TILE_SIZE or head_dim % ttnn.TILE_SIZE" in view
    assert "math.prod(desired_shape) != math.prod(cache.padded_shape)" in view
    assert "return ttnn.experimental.view(cache, desired_shape)" in view
    assert "ttnn.reshape" not in view
    assert ".deallocate(True)" not in view


def test_long_single_user_attention_fuses_rowwise_norm_and_residual_by_chunk():
    decoder_source = (ROOT / "models/autoports/google_gemma_4_31b/tt/multichip_decoder.py").read_text()
    stream = decoder_source[
        decoder_source.index("    def _chunked_attention_output_projection(") : decoder_source.index(
            "    def _chunked_full_attention_concatenated("
        )
    ]
    forward = decoder_source[decoder_source.index("    def _forward_device(") : decoder_source.index("\n\n__all__")]
    assert "normalized = self.layer.post_attention_layernorm.forward(reduced)" in stream
    assert "residual_chunk = ttnn.slice(residual" in stream
    assert "chunk_output = ttnn.add(residual_chunk, normalized)" in stream
    assert "residual=residual if batch_size == 1 else None" in forward
    assert "if attention_residual_fused:" in forward


def test_long_single_user_mlp_streams_the_complete_residual_branch():
    decoder_source = (ROOT / "models/autoports/google_gemma_4_31b/tt/multichip_decoder.py").read_text()
    stream = decoder_source[
        decoder_source.index("    def _chunked_mlp_residual(") : decoder_source.index("    def _forward_device(")
    ]
    forward = decoder_source[decoder_source.index("    def _forward_device(") : decoder_source.index("\n\n__all__")]
    assert "normed = self.layer.pre_feedforward_layernorm.forward(residual_chunk)" in stream
    assert "mlp_output = self.layer.shared_mlp(normed)" in stream
    assert "post_norm = self.layer.post_feedforward_layernorm.forward(mlp_output)" in stream
    assert "combined = ttnn.add(residual_chunk, post_norm)" in stream
    assert stream.count("residual_chunk = ttnn.slice(residual") == 2
    assert "if self.layer.layer_scalar != 1.0:" in stream
    assert "slice_write(\n                write_chunk,\n                output" in stream
    assert "return self._chunked_mlp_residual(hidden_states)" in forward
