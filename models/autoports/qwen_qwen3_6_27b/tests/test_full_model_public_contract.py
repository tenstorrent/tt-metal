import inspect
import json
from pathlib import Path

from models.autoports.qwen_qwen3_6_27b.tt.generator import Qwen36Generator
from models.autoports.qwen_qwen3_6_27b.tt.model import Qwen36Model


ROOT = Path("models/autoports/qwen_qwen3_6_27b")


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
        "_sequence_masks", "_conv_state_selector_chunks", "_sequence_mask",
        "_conv_state_selectors", "_cache_page_table",
    ):
        setattr(layer, name, object())
    model = Qwen36Model.__new__(Qwen36Model)
    model.layers = [layer]
    model.clear_prefill_request_state()
    assert all(value is None for name, value in vars(layer).items() if name.startswith("_"))
