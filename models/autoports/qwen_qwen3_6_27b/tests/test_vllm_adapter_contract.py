import ast
import inspect
from pathlib import Path

from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import LINEAR_PREFILL_CHUNK_SIZE
from models.autoports.qwen_qwen3_6_27b.tt.generator_vllm import Qwen36ForCausalLM
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
    platform = Path("../vllm/plugins/vllm-tt-plugin/src/vllm_tt_plugin/platform.py").read_text()
    assert "models.autoports.qwen_qwen3_6_27b.tt.generator_vllm:Qwen36ForCausalLM" in platform


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
