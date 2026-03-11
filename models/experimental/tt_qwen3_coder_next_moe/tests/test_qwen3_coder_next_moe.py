"""
Experimental tests for non-tt_symbiote Qwen3-Coder-Next MoE (TtQwen3CoderNextMoELayer).

This is a copy of the original tt_transformers test, but imports the implementation
from models.experimental.tt_qwen3_coder_next_moe instead of the shim.
"""

import pytest

try:
    from transformers import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock
except ImportError:
    pytest.skip(
        "Qwen3-Next requires transformers>=4.56.2. Install with: pip install 'transformers>=4.56.2'",
        allow_module_level=True,
    )

import torch
import ttnn

from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.common import Mode
from models.experimental.tt_qwen3_coder_next_moe import Qwen3CoderNextMoEConfig, TtQwen3CoderNextMoELayer


def _make_qwen3_moe_state_dict_and_ref(
    prefix: str,
    hidden_size: int = 512,
    num_experts: int = 8,
    moe_intermediate_size: int = 256,
    shared_intermediate: int = 256,
):
    """Build state_dict (prefixed) and reference Qwen3NextSparseMoeBlock."""
    config = Qwen3NextConfig(
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_intermediate,
        num_experts=num_experts,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        norm_topk_prob=True,
        decoder_sparse_step=1,
    )
    ref = Qwen3NextSparseMoeBlock(config)
    ref.eval()
    sd = ref.state_dict()
    state_dict = {f"{prefix}{k}": v.clone() for k, v in sd.items()}
    return state_dict, config, ref


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{}], indirect=True)
def test_qwen3_coder_next_moe_vs_torch(mesh_device, reset_seeds, device_params):
    """Compare TtQwen3CoderNextMoELayer output to Qwen3NextSparseMoeBlock (synthetic weights)."""
    prefix = "model.layers.0.mlp."
    hidden_size = 512
    num_experts = 8
    moe_intermediate_size = 256
    shared_intermediate = 256
    num_experts_per_tok = 2

    state_dict, hf_config, ref = _make_qwen3_moe_state_dict_and_ref(
        prefix,
        hidden_size=hidden_size,
        num_experts=num_experts,
        moe_intermediate_size=moe_intermediate_size,
        shared_intermediate=shared_intermediate,
    )
    config = Qwen3CoderNextMoEConfig(
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        shared_expert_intermediate_size=shared_intermediate,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
    )

    tt_moe = TtQwen3CoderNextMoELayer(
        mesh_device=mesh_device,
        state_dict=state_dict,
        state_dict_prefix=prefix,
        config=config,
        layer_num=0,
        dtype=ttnn.bfloat16,
        tt_ccl=None,
        dummy_weights=False,
        weight_cache_path=None,
    )

    batch_seq = 4
    torch.manual_seed(123)
    pt_input = torch.randn(1, batch_seq, hidden_size, dtype=torch.bfloat16) * 0.02
    # Ref model weights are float32; run ref in float32 to avoid m1/m2 dtype mismatch (BFloat16 != float)
    ref_out = ref(pt_input.float())
    if isinstance(ref_out, tuple):
        ref_out = ref_out[0]

    tt_input = ttnn.from_torch(
        pt_input.reshape(1, 1, batch_seq, hidden_size),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_out = tt_moe(tt_input, mode=Mode.DECODE)
    tt_out_t = ttnn.to_torch(tt_out)
    if isinstance(tt_out_t, (list, tuple)):
        tt_out_t = tt_out_t[0]
    tt_out_t = torch.Tensor(tt_out_t).reshape(1, batch_seq, hidden_size).float()

    pcc_threshold = 0.98  # MoE mixes device bfloat16 + CPU torch; 0.99 used for pure-device modules
    passing, actual_pcc = comp_pcc(ref_out, tt_out_t, pcc_threshold)
    print(f"Qwen3 Coder Next MoE PCC: {actual_pcc} (threshold {pcc_threshold})")
    assert passing, f"Qwen3 Coder Next MoE PCC below {pcc_threshold}: {actual_pcc}"
