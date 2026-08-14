import ast
import inspect
from pathlib import Path

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
    assert pool == 1_726_400
    assert (pool + 32 * 800) // 800 == 2_190
    assert pool >= 262_144


def test_decode_delegates_to_canonical_token_out_path():
    source = inspect.getsource(Qwen36ForCausalLM.decode_forward)
    assert "setup_token_out_decode" in source
    assert "token_out_decode_step" in source
    forbidden = ("argmax", "topk", "to_torch", "_to_host_logits")
    assert all(name not in source for name in forbidden)


def test_adapter_has_no_independent_sampling_implementation():
    tree = ast.parse(inspect.getsource(Qwen36ForCausalLM))
    method_names = {node.name for node in tree.body[0].body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
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
