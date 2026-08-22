# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import ttnn

from models.experimental.deepseek_v4_flash.tt.attention import DeepSeekV4Attention
from models.experimental.deepseek_v4_flash.tt.hyperconnection import DeepSeekV4HyperConnection
from models.experimental.deepseek_v4_flash.tt.l1_weights import build_l1_weight_tensor, shard_layout
from models.experimental.deepseek_v4_flash.tt.moe import DeepSeekV4MLP, DeepSeekV4TopKRouter
from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_two_resident_decoder_layers_have_distinct_packed_regions():
    layout = shard_layout(("heavily_compressed_attention", "compressed_sparse_attention"))
    for name in ("q_a_proj", "o_b_proj", "shared_down_proj", "router_gate", "attn_hc.fn", "ffn_hc.fn"):
        first = layout.region(0, name)
        second = layout.region(1, name)
        assert first.tile_offset != second.tile_offset
        assert second.tile_offset >= first.tile_offset + first.num_tiles


def _config():
    return SimpleNamespace(
        layer_types=["compressed_sparse_attention"],
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        o_groups=8,
        o_lora_rank=1024,
        rms_norm_eps=1.0e-6,
        hidden_size=4096,
        compress_rates={"compressed_sparse_attention": 128},
    )


def _weights():
    return {
        "q_a_proj.weight": torch.randn(1024, 4096, dtype=torch.bfloat16),
        "q_b_proj.weight": torch.randn(32768, 1024, dtype=torch.bfloat16),
        "kv_proj.weight": torch.randn(512, 4096, dtype=torch.bfloat16),
        "o_a_proj.weight": torch.randn(8 * 1024, 4096, dtype=torch.bfloat16),
        "o_b_proj.weight": torch.randn(4096, 8192, dtype=torch.bfloat16),
        "q_a_norm.weight": torch.randn(1024, dtype=torch.bfloat16),
        "kv_norm.weight": torch.randn(512, dtype=torch.bfloat16),
        "sinks": torch.randn(64, dtype=torch.bfloat16),
        "compressor.kv_proj.weight": torch.randn(1024, 4096, dtype=torch.bfloat16),
        "compressor.gate_proj.weight": torch.randn(1024, 4096, dtype=torch.bfloat16),
        "compressor.kv_norm.weight": torch.randn(512, dtype=torch.bfloat16),
        "compressor.position_bias": torch.randn(128, 1024, dtype=torch.bfloat16),
    }


def test_attention_galaxy32_packed_l1_projections(device):
    """Galaxy32 attention uses one BF4 L1 tensor for all seven projection matmuls."""
    torch.manual_seed(0)
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y < 120:
        pytest.skip("Galaxy32 packed attention requires a 120-core Blackhole chip")

    weights = _weights()
    system_config = load_system_config(profile="galaxy32")
    names = frozenset(
        (
            "q_a_proj",
            "q_b_proj",
            "kv_proj",
            "compressor.gate_proj",
            "compressor.kv_proj",
            "o_a_proj",
            "o_b_proj",
        )
    )
    packed_tensor, layout = build_l1_weight_tensor(
        [{name: weights[f"{name}.weight"] for name in names}],
        device,
        layer_types=("compressed_sparse_attention",),
        weight_names_by_layer=(names,),
    )
    attention = DeepSeekV4Attention(
        _config(),
        0,
        weights,
        device,
        weight_dtype=ttnn.bfloat4_b,
        use_prefetcher=False,
        system_config=system_config,
        packed_weights=(packed_tensor, layout, 0),
    )

    assert attention.use_packed_l1_weights
    packed_tensor, layout, slot = attention.packed_weights
    assert slot == 0
    assert packed_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    assert packed_tensor.memory_config().shard_spec.grid.num_cores() == 120
    assert layout.shard_tiles == 1152

    cases = (
        ("q_a_proj", attention.q_a_proj, weights["q_a_proj.weight"], (1, 1, 1, 4096)),
        ("q_b_proj", attention.q_b_proj, weights["q_b_proj.weight"], (1, 1, 1, 1024)),
        ("kv_proj", attention.kv_proj, weights["kv_proj.weight"], (1, 1, 1, 4096)),
        (
            "compressor.kv_proj",
            attention.compressor.kv_proj,
            weights["compressor.kv_proj.weight"],
            (1, 1, 1, 4096),
        ),
        (
            "compressor.gate_proj",
            attention.compressor.gate_proj,
            weights["compressor.gate_proj.weight"],
            (1, 1, 1, 4096),
        ),
        ("o_b_proj", attention.o_b_proj, weights["o_b_proj.weight"], (1, 1, 1, 8192)),
    )
    for name, projection, weight, shape in cases:
        assert projection.packed_weight_spec.k_blocks == (layout.region(0, name).spec.k_blocks or 1)
        assert projection.partial_width_sharded == (projection.packed_weight_spec.k_blocks > 1)
        x = torch.randn(shape, dtype=torch.bfloat16)
        ref = x.float() @ weight.float().transpose(-2, -1)
        x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = projection(x_tt)
        assert_with_pcc(ref, ttnn.to_torch(out).float(), 0.99)
        assert projection.packed_weight_spec.tile_offset == layout.region(0, name).tile_offset

    grouped_weight = weights["o_a_proj.weight"].reshape(8, 1024, 4096).transpose(1, 2).contiguous()
    grouped_x = torch.randn(1, 8, 1, 4096, dtype=torch.bfloat16)
    grouped_ref = torch.matmul(grouped_x.reshape(8, 1, 4096).float(), grouped_weight.float())
    grouped_tt = ttnn.from_torch(grouped_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    grouped_out = attention.o_a_proj(grouped_tt)
    assert_with_pcc(grouped_ref, ttnn.to_torch(grouped_out).float().reshape(8, 1, 1024), 0.99)
    assert attention.o_a_proj.packed_weight_spec.tile_offset == layout.region(0, "o_a_proj").tile_offset


def test_decoder_remaining_modules_use_one_packed_tensor(device):
    """Shared expert, learned router and both HC projections share the decoder pack."""
    grid = device.compute_with_storage_grid_size()
    if grid.x * grid.y < 120:
        pytest.skip("Galaxy32 packed decoder weights require a 120-core Blackhole chip")
    torch.manual_seed(1)
    cfg = SimpleNamespace(
        hidden_size=4096,
        moe_intermediate_size=2048,
        num_local_experts=256,
        num_experts_per_tok=8,
        routed_scaling_factor=1.0,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1.0e-6,
        rms_norm_eps=1.0e-6,
    )
    checkpoint = {
        "shared_experts.gate_proj.weight": torch.randn(2048, 4096, dtype=torch.bfloat16),
        "shared_experts.up_proj.weight": torch.randn(2048, 4096, dtype=torch.bfloat16),
        "shared_experts.down_proj.weight": torch.randn(4096, 2048, dtype=torch.bfloat16),
        "gate.weight": torch.randn(256, 4096, dtype=torch.bfloat16),
        "gate.e_score_correction_bias": torch.zeros(256, dtype=torch.bfloat16),
        "attn.fn": torch.randn(24, 16384, dtype=torch.bfloat16),
        "attn.base": torch.zeros(24, dtype=torch.bfloat16),
        "attn.scale": torch.ones(3, dtype=torch.bfloat16),
        "ffn.fn": torch.randn(24, 16384, dtype=torch.bfloat16),
        "ffn.base": torch.zeros(24, dtype=torch.bfloat16),
        "ffn.scale": torch.ones(3, dtype=torch.bfloat16),
    }
    placement_weights = {
        "shared_gate_proj": checkpoint["shared_experts.gate_proj.weight"],
        "shared_up_proj": checkpoint["shared_experts.up_proj.weight"],
        "shared_down_proj": checkpoint["shared_experts.down_proj.weight"],
        "router_gate": checkpoint["gate.weight"],
        "attn_hc.fn": checkpoint["attn.fn"],
        "ffn_hc.fn": checkpoint["ffn.fn"],
    }
    names = frozenset(placement_weights)
    tensor, layout = build_l1_weight_tensor(
        [placement_weights],
        device,
        layer_types=("compressed_sparse_attention",),
        weight_names_by_layer=(names,),
    )
    packed = (tensor, layout, 0)
    mlp = DeepSeekV4MLP(
        checkpoint,
        "shared_experts",
        device,
        config=cfg,
        weight_dtype=ttnn.bfloat4_b,
        packed_weights=packed,
    )
    router = DeepSeekV4TopKRouter(cfg, checkpoint, device, packed_weights=packed)
    attn_hc = DeepSeekV4HyperConnection(
        cfg,
        {"fn": checkpoint["attn.fn"], "base": checkpoint["attn.base"], "scale": checkpoint["attn.scale"]},
        device,
        packed_weights=packed,
        packed_name="attn_hc.fn",
    )
    ffn_hc = DeepSeekV4HyperConnection(
        cfg,
        {"fn": checkpoint["ffn.fn"], "base": checkpoint["ffn.base"], "scale": checkpoint["ffn.scale"]},
        device,
        packed_weights=packed,
        packed_name="ffn_hc.fn",
    )

    x = torch.randn(1, 1, 1, 4096, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gate_ref = x.float() @ checkpoint["gate.weight"].float().T
    assert_with_pcc(gate_ref, ttnn.to_torch(router.gate(x_tt)).float(), 0.99)
    gate = torch.nn.functional.silu(x.float() @ checkpoint["shared_experts.gate_proj.weight"].float().T)
    up = x.float() @ checkpoint["shared_experts.up_proj.weight"].float().T
    mlp_ref = (gate * up) @ checkpoint["shared_experts.down_proj.weight"].float().T
    assert_with_pcc(mlp_ref, ttnn.to_torch(mlp(x_tt)).float(), 0.97)

    streams = torch.randn(1, 1, 4, 4096, dtype=torch.bfloat16)
    flat = streams.reshape(1, 1, 1, 16384)
    rms = flat.float() * torch.rsqrt(flat.float().pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)
    decode_tile = ttnn.Tile((1, 32))
    for module, name in ((attn_hc, "attn"), (ffn_hc, "ffn")):
        fn_ref = rms @ checkpoint[f"{name}.fn"].float().T
        fn_actual = ttnn.to_torch(
            module.fn(
                ttnn.from_torch(
                    rms.to(torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    tile=decode_tile,
                    memory_config=module.fn.get_input_memory_config(1, 16384, 1),
                )
            )
        ).float()[..., :24]
        assert_with_pcc(fn_ref, fn_actual, 0.99)
