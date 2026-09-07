import ast
import tempfile
import inspect
import json
from pathlib import Path

from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import LINEAR_PREFILL_CHUNK_SIZE
from models.autoports.qwen_qwen3_6_27b.tt.generator_vllm import Qwen36ForCausalLM
from models.autoports.qwen_qwen3_6_27b.tt.model import _streaming_prefill_chunk_size
from models.common.sampling import SamplingParams


def test_vllm_capabilities_and_context_pool():
    caps = Qwen36ForCausalLM.model_capabilities
    assert caps == {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_async_decode_overlap": True,
        "supports_sample_on_device": True,
        "max_device_top_k": 32,
    }
    pool = Qwen36ForCausalLM.get_max_tokens_all_users(max_model_len=262144, max_num_seqs=32)
    page_size = 800
    assert pool == 1_726_400
    assert (pool + 32 * page_size) // page_size == 2_190
    assert pool >= 262_144
    measured_largest_free_block = 62_914_560
    concat_bytes_per_bank_at_64 = 805_306_368 // 8
    assert LINEAR_PREFILL_CHUNK_SIZE == 32
    assert concat_bytes_per_bank_at_64 // 2 == 50_331_648
    assert concat_bytes_per_bank_at_64 // 2 < measured_largest_free_block


def test_streaming_prefill_chunk_bounds_embedding_memory_across_the_batch():
    """Peak prefill embedding memory must not scale with the serving batch.

    Streaming prefill embeds ``batch`` rows per chunk while only one sequence is
    actually being prefilled, so an unbounded chunk makes a 32-slot server pay
    32x the memory of a 1-slot one. At chunk 32768 / hidden 5120 / bf16 that is
    a 10.7 GB allocation and prefill dies on the first request past 32768
    tokens. Batch 1 must be untouched; every larger batch must cost no more.
    """
    hidden, dtype_bytes = 5120, 2
    baseline = 1 * _streaming_prefill_chunk_size(32768, 64, 1) * hidden * dtype_bytes
    assert baseline == 335544320
    for batch in (1, 2, 4, 8, 16, 32):
        chunk = _streaming_prefill_chunk_size(32768, 64, batch)
        assert batch * chunk * hidden * dtype_bytes <= baseline, (batch, chunk)
        assert chunk % 64 == 0 and chunk >= 64
    # batch 1 keeps the exact chunk it had before the bound existed
    assert _streaming_prefill_chunk_size(32768, 64, 1) == 32768
    # an odd page size still aligns, and still shrinks with batch. lcm(800, 32)
    # is 800, so the batch-32 budget of 1024 floors to one 800-token quantum.
    assert _streaming_prefill_chunk_size(32768, 800, 1) == 32000
    assert _streaming_prefill_chunk_size(32768, 800, 32) == 800
    # a batch large enough to undercut the quantum falls back to one quantum
    assert _streaming_prefill_chunk_size(32768, 64, 100000) == 64


def test_streaming_prefill_chunks_follow_effective_kv_page_boundaries():
    assert _streaming_prefill_chunk_size(32768, 64) == 32768
    assert _streaming_prefill_chunk_size(32768, 800) == 32000
    assert _streaming_prefill_chunk_size(32768, 1024) == 32768

    boundaries = {
        64: (32767, 32768, 32769),
        800: (
            31999,
            32000,
            32001,
            32767,
            32768,
            32769,
            32780,
            33599,
            33600,
            33601,
            63999,
            64000,
            64001,
        ),
        1024: (32767, 32768, 32769),
    }
    for page_size, sequences in boundaries.items():
        chunk = _streaming_prefill_chunk_size(32768, page_size)
        assert 0 < chunk <= 32768
        assert chunk % page_size == 0
        assert chunk % LINEAR_PREFILL_CHUNK_SIZE == 0
        for sequence in sequences:
            starts = range(0, sequence, chunk)
            assert all(start % page_size == 0 for start in starts)
            assert all(start % LINEAR_PREFILL_CHUNK_SIZE == 0 for start in starts)


def test_decode_delegates_to_canonical_token_out_path():
    source = inspect.getsource(Qwen36ForCausalLM.decode_forward)
    assert "setup_token_out_decode" in source
    assert "token_out_decode_step" in source
    forbidden = ("argmax", "topk", "to_torch", "_to_host_logits")
    assert all(name not in source for name in forbidden)


def test_host_sampling_compatibility_preserves_slot_remap():
    source = inspect.getsource(Qwen36ForCausalLM.decode_forward)
    host_branch = source.split("if sampling_params is None:", 1)[1].split("gen.sampling.apply_decode_state", 1)[0]
    assert "if slot_remap is not None:" in host_branch
    assert "gen.remap_decode_slots(remap)" in host_branch


def test_linear_scan_does_not_hold_both_concat_workspaces():
    source = (Path(__file__).parents[1] / "tt" / "multichip_decoder.py").read_text()
    transform_concat = source.index("previous_transform = ttnn.concat")
    transform_deallocate = source.index("ttnn.deallocate(previous_transform)", transform_concat)
    bias_concat = source.index("previous_bias = ttnn.concat", transform_concat)
    assert transform_concat < transform_deallocate < bias_concat
    assert "output_tensor=scan_scratch" not in source
    assert "linear_scan_scratch" not in source


def test_linear_prefill_metadata_uses_scan_chunk_size():
    tt_dir = Path(__file__).parents[1] / "tt"
    generator_source = (tt_dir / "generator.py").read_text()
    model_source = (tt_dir / "model.py").read_text()
    decoder_source = (tt_dir / "multichip_decoder.py").read_text()
    assert "range(0, physical_len, LINEAR_PREFILL_CHUNK_SIZE)" in generator_source
    assert "min(LINEAR_PREFILL_CHUNK_SIZE, physical_len - start)" in generator_source
    assert "start // LINEAR_PREFILL_CHUNK_SIZE" in model_source
    assert "math.ceil(end / LINEAR_PREFILL_CHUNK_SIZE)" in model_source
    assert "(batch, chunk_len + 4)" in generator_source
    assert "(1, self.batch, 1, sequence + 4)" in decoder_source


def test_adapter_has_no_independent_sampling_implementation():
    tree = ast.parse(inspect.getsource(Qwen36ForCausalLM))
    method_names = {
        node.name for node in tree.body[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "sample" not in method_names
    assert "sampling" not in method_names


def test_plugin_registration_targets_autoport():
    """The bundle must claim this architecture for the autoport, not the demo.

    This used to read a patched `platform.py` out of a sibling vLLM checkout.
    The plugin is now a standalone repo installed as a package, so there is no
    such path, and the registration moved into an in-repo bundle that the plugin
    discovers through EXTRA_MODELS_DIR. Assert against the bundle, which is the
    artifact this repo actually ships.
    """
    bundle = Path(__file__).resolve().parents[2] / "vllm_bundles" / "qwen36_autoport" / "vllm_metadata.json"
    metadata = json.loads(bundle.read_text())
    assert metadata["arch"] == "Qwen3_5ForConditionalGeneration"
    assert metadata["main_class"] == "models.autoports.qwen_qwen3_6_27b.tt.generator_vllm:Qwen36ForCausalLM"


def test_compact_sampling_params_scatter_to_persistent_slots():
    params = SamplingParams(
        temperature=[0.5, 2.0],
        top_k=[7, 19],
        top_p=[0.8, 0.6],
        presence_penalty=[0.25, 1.5],
        frequency_penalty=[0.5, 1.25],
        repetition_penalty=[1.1, 2.0],
        seed=[42, 999],
    )
    formatted = Qwen36ForCausalLM._format_slot_sampling(params, [5, 2])
    assert formatted.top_k[5] == 7
    assert formatted.top_k[2] == 19
    assert formatted.seed[:2] == [42, 999]
    assert formatted.presence_penalty[5] == 0.25
    assert formatted.frequency_penalty[2] == 1.25
    assert formatted.repetition_penalty[5] == 1.1


def test_sampling_contract_key_changes_only_for_device_sampler_state():
    baseline = SamplingParams(temperature=[1.0], top_k=[1], top_p=[0.0], seed=[42])
    next_seed = SamplingParams(temperature=[1.0], top_k=[1], top_p=[0.0], seed=[999])
    changed_top_k = SamplingParams(temperature=[1.0], top_k=[7], top_p=[0.0], seed=[999])
    assert Qwen36ForCausalLM._sampling_key(baseline) == Qwen36ForCausalLM._sampling_key(next_seed)
    assert Qwen36ForCausalLM._sampling_key(baseline) != Qwen36ForCausalLM._sampling_key(changed_top_k)


def test_decode_skips_unchanged_sampler_parameter_refresh():
    source = inspect.getsource(Qwen36ForCausalLM.decode_forward)
    assert "sampling_changed = sampling_key != self._sampling_contract_key" in source
    assert "refresh_sampling_params=sampling_changed" in source


def test_default_snapshot_prefers_host_staged_weights():
    """The autoport must accept weights the way tt-inference-server stages them.

    run_vllm_api_server downloads with snapshot_download(local_dir=...) into a
    flat CACHE_ROOT/weights/<name> directory and sets MODEL_WEIGHTS_DIR. That
    path carries no revision, so a resolver that insists on the pinned revision
    in the HF hub layout fails *after* the weights are successfully on disk --
    which is exactly how CI run 34116873055 died.
    """
    import os
    from pathlib import Path
    from models.autoports.qwen_qwen3_6_27b.tt import functional_decoder as fd

    saved = {k: os.environ.get(k) for k in ("MODEL_WEIGHTS_DIR", "CACHE_ROOT")}
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / "staged"
        staged.mkdir()
        try:
            # an incomplete directory must NOT be trusted: it looks non-empty
            # but would fail later at load
            os.environ["MODEL_WEIGHTS_DIR"] = str(staged)
            os.environ.pop("CACHE_ROOT", None)
            assert fd.default_snapshot() != staged

            (staged / "model.safetensors.index.json").write_text("{}")
            assert fd.default_snapshot() == staged

            # CACHE_ROOT/weights/<name> is the fallback when the variable did
            # not reach this process
            os.environ.pop("MODEL_WEIGHTS_DIR")
            root = Path(tmp) / "cache_root"
            weights = root / "weights" / fd.MODEL_ID.split("/")[-1]
            weights.mkdir(parents=True)
            os.environ["CACHE_ROOT"] = str(root)
            assert fd.default_snapshot() != weights
            (weights / "model.safetensors.index.json").write_text("{}")
            assert fd.default_snapshot() == weights
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
