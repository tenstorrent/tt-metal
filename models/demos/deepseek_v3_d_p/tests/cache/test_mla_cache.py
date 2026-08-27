# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import shutil
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import profiler
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import fabric2d_device_params
from models.demos.deepseek_v3_d_p.tt.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker, report_and_clear
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache
from tests.ttnn.utils_for_testing import comp_pcc

CACHE_DIR = Path("/tmp/DS_PREFILL_mla")


@pytest.fixture(autouse=True)
def cleanup_cache():
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    yield
    report_and_clear()


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            (2, 2),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 2), topology="mesh-2x2"),
            id="fabric2d-2x2",
        ),
        # Blackhole forms whole-box meshes only, so the 2x2 case above never runs on an 8-chip
        # loudbox -- it is a 4-device QuietBox / Wormhole shape. 2x4 makes this test actually
        # executable there, which is where the Kimi weight caches are exercised.
        pytest.param(
            (2, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(2, 4), topology="mesh-2x4"),
            id="fabric2d-2x4",
        ),
        # Blackhole accepts 32-device meshes only, so neither shape above runs on the galaxy and every
        # row there skips -- returning rc=0, which reads as coverage in a diff. This row is the one
        # that executes on Blackhole, and so the only one that covers mistral_small_4 at all.
        pytest.param(
            (8, 4),
            fabric2d_device_params(),
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="fabric2d-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
# "k3" (not "kimi_k3") and "mistral4" (not "mistral_small_4") because pytest -k is substring-based.
# deepseek_v3_d_p is the historical default of the `variant` fixture, so keeping it first preserves
# this test's original coverage.
# mistral_small_4 resolves through the adapter's hand-built `mistral4_hf_config` rather than
# AutoConfig, so it needs no checkpoint here — only the MLA dims the state_dict below is built from.
@pytest.mark.parametrize(
    "variant",
    ["deepseek_v3_d_p", "kimi_k3", "mistral_small_4"],
    indirect=True,
    ids=["dsv3", "k3", "mistral4"],
)
def test_mla_weights_cold_warm_cache(mesh_device, device_params, config_only, variant):
    """Test: weights → cold cache → warm cache produce identical outputs.

    The kimi_k3 case additionally covers the ``g_proj`` output-gate weight: that it is written to and
    read back from the cache, and that ``check_cache_complete``'s gate flag is not merely cosmetic
    (a non-gated cache must fail the gated check, or a K3 layer could load a cache with no gate in
    it and silently run ungated)."""
    config = config_only
    layer_idx = 0
    seq_len = 1024
    sp_axis = 0
    tp_axis = 1

    # Set max_seq_len on config (required by MLA)
    config.max_seq_len = seq_len

    mesh_shape = list(mesh_device.shape)

    # Create random weights matching MLA architecture
    torch.manual_seed(42)
    std = config.initializer_range

    state_dict = {
        "q_a_proj.weight": (torch.randn(config.q_lora_rank, config.hidden_size) * std).to(torch.bfloat16),
        "q_a_layernorm.weight": torch.ones(config.q_lora_rank, dtype=torch.bfloat16),
        "q_b_proj.weight": (
            torch.randn(
                config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim),
                config.q_lora_rank,
            )
            * std
        ).to(torch.bfloat16),
        "kv_a_proj_with_mqa.weight": (
            torch.randn(
                config.kv_lora_rank + config.qk_rope_head_dim,
                config.hidden_size,
            )
            * std
        ).to(torch.bfloat16),
        "kv_a_layernorm.weight": torch.ones(config.kv_lora_rank, dtype=torch.bfloat16),
        "kv_b_proj.weight": (
            torch.randn(
                config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            )
            * std
        ).to(torch.bfloat16),
        "o_proj.weight": (
            torch.randn(
                config.hidden_size,
                config.num_attention_heads * config.v_head_dim,
            )
            * std
        ).to(torch.bfloat16),
    }
    # Kimi-K3 output gate. Appended so the manual_seed(42) draw order above is unchanged for the
    # non-gated variants.
    has_output_gate = bool(getattr(config, "mla_use_output_gate", False))
    if has_output_gate:
        state_dict["g_proj.weight"] = (
            torch.randn(
                config.num_attention_heads * config.v_head_dim,
                config.hidden_size,
            )
            * std
        ).to(torch.bfloat16)

    # Create input (full tensor - mesh_mapper will shard automatically)
    # Following pattern from test_mla.py: create full tensor, let TTNN shard it
    x = torch.randn(1, 1, seq_len, config.hidden_size, dtype=torch.float32)
    x_tt = ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(2, 3),  # Shard dim 2 (seq_len) on SP, dim 3 (hidden_size) on TP
            mesh_shape=mesh_device.shape,
        ),
    )

    # Create RoPE tensors
    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope_setup.get_rope_tensors(seq_len)

    # Initialize KVPE cache (required by MLA forward)
    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    # Helper to convert TP-sharded output to torch
    def to_torch_concat(tt_tensor):
        """Convert TP-sharded 4D tensor to torch."""
        return ttnn.to_torch(
            tt_tensor,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=(2, 3), mesh_shape=mesh_device.shape  # Concat SP and TP dims
            ),
        )

    # === Path 1: From Weights ===
    mla_from_weights = ttMLA(
        config,
        state_dict,
        mesh_device,
        layer_idx=layer_idx,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        weight_cache_path=None,
    )
    output1_tt = mla_from_weights.forward(x_tt, rope_tensors, tt_kvpe_cache)
    output1 = to_torch_concat(output1_tt)

    # === Path 2: Cold Cache ===
    init_checker(CACHE_DIR)
    assert not ttMLA.check_cache_complete(
        CACHE_DIR, f"layer_{layer_idx}.mla", has_output_gate=has_output_gate
    ), "Cache should be empty before build"

    profiler.clear()
    profiler.start("build_cache")
    ttMLA.build_ttnn_cache(
        state_dict,
        CACHE_DIR,
        mesh_device,
        config,
        layer_idx,
        seq_len,
        sp_axis,
        tp_axis,
    )
    profiler.end("build_cache")

    init_checker(CACHE_DIR)
    assert ttMLA.check_cache_complete(
        CACHE_DIR, f"layer_{layer_idx}.mla", has_output_gate=has_output_gate
    ), "Cache should be complete after build"
    if has_output_gate:
        # A gated cache must satisfy the non-gated check too (g_proj is additive, so the
        # non-gated name set is a strict subset)...
        assert ttMLA.check_cache_complete(
            CACHE_DIR, f"layer_{layer_idx}.mla", has_output_gate=False
        ), "gated cache should also satisfy the non-gated check"
    else:
        # ...and conversely a non-gated cache must NOT pass the gated check, or a K3 layer could
        # silently load a cache with no g_proj in it.
        assert not ttMLA.check_cache_complete(
            CACHE_DIR, f"layer_{layer_idx}.mla", has_output_gate=True
        ), "non-gated cache must not satisfy the gated check"

    profiler.start("cold_load")
    mla_cold = ttMLA(
        config,
        {},
        mesh_device,  # Empty state_dict loads from cache
        layer_idx=layer_idx,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        weight_cache_path=CACHE_DIR,
    )
    profiler.end("cold_load")
    output2_tt = mla_cold.forward(x_tt, rope_tensors, tt_kvpe_cache)
    output2 = to_torch_concat(output2_tt)

    # === Path 3: Warm Cache ===
    profiler.start("warm_load")
    mla_warm = ttMLA(
        config,
        {},
        mesh_device,
        layer_idx=layer_idx,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        weight_cache_path=CACHE_DIR,
    )
    profiler.end("warm_load")
    output3_tt = mla_warm.forward(x_tt, rope_tensors, tt_kvpe_cache)
    output3 = to_torch_concat(output3_tt)

    # === Validation ===
    passed_cold, pcc_cold = comp_pcc(output1, output2)
    passed_warm, pcc_warm = comp_pcc(output1, output3)

    logger.info(f"MLA Cache Test:")
    logger.info(f"  Weights vs Cold Cache PCC: {pcc_cold}")
    logger.info(f"  Weights vs Warm Cache PCC: {pcc_warm}")
    logger.info(f"  build_cache: {profiler.get('build_cache')*1000:.1f} ms")
    logger.info(f"  cold_load:   {profiler.get('cold_load')*1000:.1f} ms")
    logger.info(f"  warm_load:   {profiler.get('warm_load')*1000:.1f} ms")

    assert passed_cold, f"Cold cache mismatch: PCC={pcc_cold}"
    assert passed_warm, f"Warm cache mismatch: PCC={pcc_warm}"
