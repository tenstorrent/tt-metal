# Transformers 5.x compatibility shims for HF custom-code models loaded by the
# tt_symbiote tests (e.g. Ling-mini-2.0's modeling_bailing_moe_v2.py from the
# HF Hub). The same shims live in
# models/experimental/tt_symbiote/vllm/generator_vllm_ling.py and run when the
# vLLM adapter is imported by the inference server. Pytests bypass that
# adapter, so the shims must be applied here before any HF dynamic-import
# chain fires (i.e. before AutoModel.from_pretrained / trust_remote_code).
#
# (1) is_torch_fx_available was removed in transformers 5.x. Returning False
#     skips the torch.fx wrapping (a tracing-only optimisation we don't use).
# (2) ROPE_INIT_FUNCTIONS['default'] was dropped in transformers 5.x. HF custom
#     code sets rope_type='default' when rope_scaling is absent; we reinstate
#     the key with the original plain inv-freq formula. The HF rotary embedding
#     is replaced by TTNNBailingRotaryEmbedding at runtime, so this only needs
#     to survive HF __init__.
import torch
import transformers.utils.import_utils as _tui

if not hasattr(_tui, "is_torch_fx_available"):
    _tui.is_torch_fx_available = lambda: False

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE_INIT

if "default" not in _ROPE_INIT:

    def _default_rope_init(config, device=None, seq_len=None, **kwargs):
        base = getattr(config, "rope_theta", 10000.0)
        partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim))
        return inv_freq, 1.0  # attention_factor unused for default RoPE

    _ROPE_INIT["default"] = _default_rope_init

import pytest
from models.experimental.tt_symbiote.modules.moe import Glm4MoeConfig


@pytest.fixture
def default_glm_config():
    """Default GLM configuration for testing."""
    return Glm4MoeConfig(
        hidden_size=2048,
        intermediate_size=10240,
        moe_intermediate_size=1536,
        num_local_experts=64,
        num_experts_per_tok=4,
        n_shared_experts=1,
        routed_scaling_factor=1.8,
        n_group=1,
        topk_group=1,
        norm_topk_prob=True,
    )
