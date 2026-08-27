# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only contracts for the Galaxy Llama-3.3-70B reconstruction."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.common.models.galaxy.plans import build_galaxy_decode_collectives
from models.common.models.galaxy.recipes import resolve_galaxy_decode_placements
from models.common.models.llama33_70b_galaxy import hf_adaptor
from models.common.models.llama33_70b_galaxy import model as galaxy_model
from models.common.models.llama33_70b_galaxy import weight_utils
from models.common.models.llama33_70b_galaxy.model import (
    LLAMA33_70B_GALAXY_ACCURACY,
    LLAMA33_70B_GALAXY_PERFORMANCE,
    LLAMA33_70B_PREFETCHED_WEIGHT_NAMES,
    Llama33_70BGalaxyLayerWeights,
    Llama33_70BGalaxyModelParameters,
    Llama33_70BGalaxyWeights,
    build_llama33_70b_galaxy_lazy_weights,
    build_llama33_70b_galaxy_model,
    default_paged_attention_config,
    parameters_from_hf_config,
    validate_llama33_70b_checkpoint,
)
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm import rmsnorm_2d
from models.common.modules.rmsnorm.rmsnorm_2d import RMSNorm2DResidualPolicy, _resolve_2d_config


def _hf_config(**overrides):
    values = dict(
        num_hidden_layers=80,
        hidden_size=8192,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=28672,
        vocab_size=128256,
        head_dim=128,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        rope_scaling={"factor": 8.0, "original_max_position_embeddings": 8192},
        max_position_embeddings=131072,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _linear(weight, bias=None):
    return SimpleNamespace(weight=weight, bias=bias)


def _self_attn(*, dim=8, n_heads=4, n_kv_heads=2, head_dim=4):
    def rows(count):
        return torch.arange(count * dim, dtype=torch.float32).reshape(count, dim)

    return SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_size=dim,
        ),
        q_proj=_linear(rows(n_heads * head_dim)),
        k_proj=_linear(rows(n_kv_heads * head_dim)),
        v_proj=_linear(rows(n_kv_heads * head_dim)),
        o_proj=_linear(torch.arange(dim * n_heads * head_dim, dtype=torch.float32).reshape(dim, n_heads * head_dim)),
    )


# ---------------------------------------------------------------------------
# Checkpoint contract
# ---------------------------------------------------------------------------


def test_checkpoint_contract_accepts_the_exact_product():
    validate_llama33_70b_checkpoint(_hf_config(), n_layers=1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_size": 4096}, "Unexpected Llama-3.3-70B geometry"),
        ({"num_hidden_layers": 32}, "Unexpected Llama-3.3-70B geometry"),
        ({"vocab_size": 32000}, "Unexpected Llama-3.3-70B geometry"),
        ({"head_dim": 64}, "requires head_dim 128"),
        ({"attention_bias": True}, "bias-free"),
        ({"tie_word_embeddings": True}, "untied LM head"),
    ],
)
def test_checkpoint_contract_fails_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_llama33_70b_checkpoint(_hf_config(**overrides))


def test_checkpoint_contract_rejects_an_out_of_range_layer_subset():
    with pytest.raises(ValueError, match=r"n_layers must be in \[1, 80\]"):
        validate_llama33_70b_checkpoint(_hf_config(), n_layers=81)


# ---------------------------------------------------------------------------
# Parameters and geometry
# ---------------------------------------------------------------------------


def test_parameters_resolve_the_galaxy_geometry():
    params = parameters_from_hf_config(_hf_config(), max_seq_len=2048, prefill_sequence_lengths=(128, 2048))
    geometry = params.geometry()

    assert (params.dim, params.n_heads, params.n_kv_heads, params.head_dim) == (8192, 64, 8, 128)
    assert (params.hidden_dim, params.vocab_size, params.n_layers) == (28672, 128256, 80)
    assert (params.rope_theta, params.rope_scaling_factor, params.original_context_len) == (500000.0, 8.0, 8192)
    # 129024, not 128256. Llama's vocabulary is already a multiple of the
    # 8-shard tile, so the old minimal rule left it unpadded - and 16032 columns
    # per device is 501 tiles, a width no usable core count divides, which hung
    # the decode LM head's column all-reduce forever (D-B19). The padding is
    # masked to -inf by LMHead2D, and it is what makes that mask load-bearing for
    # Llama for the first time. See `galaxy_padded_vocab_size`.
    assert params.padded_vocab_size == 129024
    assert params.padded_vocab_size % (8 * 32 * 24) == 0
    assert (geometry.local_dim, geometry.local_hidden_dim) == (2048, 3584)
    assert geometry.attention_dim == geometry.dim
    assert geometry.prefill_sequence_lengths == (128, 2048)


def test_parameters_support_a_one_layer_model():
    params = parameters_from_hf_config(_hf_config(), n_layers=1)

    assert params.n_layers == 1
    assert params.with_layers(3).n_layers == 3
    assert params.geometry().dim == 8192


def test_parameters_reject_an_impossible_layer_count():
    with pytest.raises(ValueError, match=r"n_layers must be in \[1, 80\]"):
        Llama33_70BGalaxyModelParameters(n_layers=0)


def test_rope_table_covers_twice_the_served_context():
    assert Llama33_70BGalaxyModelParameters(max_seq_len=2048).rope_table_len() == 8192
    assert Llama33_70BGalaxyModelParameters(max_seq_len=8192).rope_table_len() == 16384


def test_default_paged_geometry_covers_the_physical_batch():
    params = Llama33_70BGalaxyModelParameters(max_seq_len=2048)
    paged = default_paged_attention_config(params)

    assert paged.block_size == 32
    assert paged.max_num_blocks == (2048 // 32) * 32


def test_precision_recipes_differ_only_in_the_feed_forward_projection():
    assert LLAMA33_70B_GALAXY_ACCURACY.mlp_w1_w3_dtype == ttnn.bfloat8_b
    assert LLAMA33_70B_GALAXY_PERFORMANCE.mlp_w1_w3_dtype == ttnn.bfloat4_b
    assert LLAMA33_70B_GALAXY_PERFORMANCE.wqkv_dtype == LLAMA33_70B_GALAXY_ACCURACY.wqkv_dtype
    assert LLAMA33_70B_GALAXY_PERFORMANCE.kv_cache_dtype == LLAMA33_70B_GALAXY_ACCURACY.kv_cache_dtype


# ---------------------------------------------------------------------------
# Provider conversion
# ---------------------------------------------------------------------------


def test_reverse_permute_interleaves_each_head_pair():
    n_heads, head_dim, dim = 3, 4, 2
    source = torch.arange(n_heads * head_dim, dtype=torch.float32).reshape(-1, 1).repeat(1, dim)

    permuted = weight_utils.reverse_permute(source, n_heads, n_heads * head_dim, dim)

    expected = [head * head_dim + offset for head in range(n_heads) for offset in (0, 2, 1, 3)]
    assert permuted[:, 0].tolist() == [float(value) for value in expected]


def test_fused_qkv_packs_each_mesh_row_contiguously():
    dim, rows = 4, 2
    wq = torch.arange(dim * 8, dtype=torch.float32).reshape(dim, 8)
    wk = torch.arange(dim * 4, dtype=torch.float32).reshape(dim, 4) + 100
    wv = torch.arange(dim * 4, dtype=torch.float32).reshape(dim, 4) + 200

    fused = weight_utils.fuse_qkv_by_mesh_row(wq, wk, wv, rows=rows)

    assert fused.shape == (dim, 16)
    assert torch.equal(fused[:, 0:4], wq[:, 0:4])
    assert torch.equal(fused[:, 4:6], wk[:, 0:2])
    assert torch.equal(fused[:, 6:8], wv[:, 0:2])
    assert torch.equal(fused[:, 8:12], wq[:, 4:8])
    assert torch.equal(fused[:, 12:14], wk[:, 2:4])
    assert torch.equal(fused[:, 14:16], wv[:, 2:4])


def test_fused_qkv_rejects_widths_that_cannot_shard_over_rows():
    with pytest.raises(ValueError, match="must shard over 8 mesh rows"):
        weight_utils.fuse_qkv_by_mesh_row(torch.zeros(4, 12), torch.zeros(4, 8), torch.zeros(4, 8))


def test_attention_conversion_returns_tt_layout_shapes():
    attn = _self_attn()

    wqkv, wo = weight_utils.attention_weights_from_hf_layer(attn, rows=2)

    assert wqkv.shape == (8, 4 * 4 + 2 * 4 + 2 * 4)
    assert wo.shape == (4 * 4, 8)
    assert torch.equal(wo, attn.o_proj.weight.T)


def test_mlp_conversion_transposes_every_projection():
    mlp = SimpleNamespace(
        gate_proj=_linear(torch.arange(12, dtype=torch.float32).reshape(3, 4)),
        up_proj=_linear(torch.arange(12, dtype=torch.float32).reshape(3, 4) + 100),
        down_proj=_linear(torch.arange(12, dtype=torch.float32).reshape(4, 3)),
    )

    w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(mlp)

    assert (w1.shape, w2.shape, w3.shape) == ((4, 3), (3, 4), (4, 3))
    assert torch.equal(w1, mlp.gate_proj.weight.T)
    assert torch.equal(w2, mlp.down_proj.weight.T)
    assert torch.equal(w3, mlp.up_proj.weight.T)


def test_embedding_table_is_vocab_major_and_two_dimensional():
    embed = SimpleNamespace(weight=torch.arange(24, dtype=torch.float32).reshape(6, 4))

    table = weight_utils.embedding_table_torch(embed)

    assert table.shape == (6, 4)
    assert table.dtype == torch.bfloat16


def test_lm_head_weight_is_transposed_and_zero_padded():
    lm_head = SimpleNamespace(weight=torch.arange(24, dtype=torch.float32).reshape(6, 4))

    weight = weight_utils.lm_head_weight_torch(lm_head, dim=4, vocab_size=6, padded_vocab_size=8)

    assert weight.shape == (4, 8)
    assert torch.equal(weight[:, :6], lm_head.weight.to(torch.bfloat16).T)
    assert torch.all(weight[:, 6:] == 0)
    with pytest.raises(ValueError, match="LM-head weight must have shape"):
        weight_utils.lm_head_weight_torch(lm_head, dim=8, vocab_size=6, padded_vocab_size=8)


def test_rope_tables_come_from_the_provider_module():
    head_dim, table_len = 4, 6

    def rotary(x, position_ids):
        angles = position_ids.float().T.repeat(1, head_dim)
        return angles.cos().unsqueeze(0), angles.sin().unsqueeze(0)

    cos, sin = weight_utils.build_rope_cos_sin_torch(rotary, table_len, head_dim)

    assert cos.shape == (1, 1, table_len, head_dim)
    assert sin.shape == (1, 1, table_len, head_dim)
    assert cos.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Assembly contracts
# ---------------------------------------------------------------------------


def _weights(layers: int) -> Llama33_70BGalaxyWeights:
    layer = Llama33_70BGalaxyLayerWeights(
        wqkv=torch.zeros(1),
        wo=torch.zeros(1),
        w1=torch.zeros(1),
        w2=torch.zeros(1),
        w3=torch.zeros(1),
        attention_norm=torch.zeros(1),
        ff_norm=torch.zeros(1),
    )
    return Llama33_70BGalaxyWeights(
        embedding=torch.zeros(1),
        rope_cos=torch.zeros(1),
        rope_sin=torch.zeros(1),
        layers=tuple(layer for _ in range(layers)),
        final_norm=torch.zeros(1),
        lm_head=torch.zeros(1),
    )


def _mesh():
    mesh = MagicMock(spec=ttnn.MeshDevice)
    mesh.shape = (8, 4)
    mesh.get_num_devices.return_value = 32
    mesh.arch.return_value = ttnn.device.Arch.WORMHOLE_B0
    mesh.dram_grid_size.return_value = SimpleNamespace(x=12, y=1)
    return mesh


def _shaped_weights(params: Llama33_70BGalaxyModelParameters, *, layers: int) -> Llama33_70BGalaxyWeights:
    """Correctly shaped host tensors with no data, for placement contracts."""

    def zeros(*shape: int):
        return torch.zeros(shape, dtype=torch.bfloat16)

    layer = Llama33_70BGalaxyLayerWeights(
        wqkv=zeros(params.dim, params.head_dim * (params.n_heads + 2 * params.n_kv_heads)),
        wo=zeros(params.n_heads * params.head_dim, params.dim),
        w1=zeros(params.dim, params.hidden_dim),
        w2=zeros(params.hidden_dim, params.dim),
        w3=zeros(params.dim, params.hidden_dim),
        attention_norm=zeros(params.dim),
        ff_norm=zeros(params.dim),
    )
    return Llama33_70BGalaxyWeights(
        embedding=zeros(params.vocab_size, params.dim),
        rope_cos=zeros(1, 1, params.rope_table_len(), params.head_dim),
        rope_sin=zeros(1, 1, params.rope_table_len(), params.head_dim),
        layers=tuple(layer for _ in range(layers)),
        final_norm=zeros(params.dim),
        lm_head=zeros(params.dim, params.padded_vocab_size),
    )


def test_model_assembly_rejects_a_layer_count_mismatch():
    params = Llama33_70BGalaxyModelParameters(n_layers=2)

    with pytest.raises(ValueError, match="expected 2 layer weight sets, got 1"):
        build_llama33_70b_galaxy_model(object(), params=params, weights=_weights(1))


def test_runtime_config_reports_the_galaxy_batched_prefill_policy():
    runtime = hf_adaptor.Llama33_70BGalaxyRuntimeConfig(
        model_name="Llama-3.3-70B-Instruct",
        model_cache_path=None,
        max_context_len=131072,
        max_seq_len=2048,
        max_prefill_chunk_size=2048,
        trace_prefill_supported_seq_lens=(128,),
        n_layers=1,
        n_kv_heads=8,
        head_dim=128,
    )

    assert runtime.max_batch_size == 32
    assert runtime.max_prefill_batch_size == 32
    assert runtime.minimum_active_prefill_rows == 16
    assert runtime.allow_cached_prefix_batching is False
    assert runtime.can_enable_trace(128) is True
    assert runtime.can_enable_trace(128, num_cached_tokens=8) is False
    assert runtime.can_enable_trace(256) is False


def test_prefetch_registration_is_ordered_per_layer():
    mesh = _mesh()
    params = Llama33_70BGalaxyModelParameters(n_layers=2)
    lazy = build_llama33_70b_galaxy_lazy_weights(
        mesh_device=mesh,
        geometry=params.geometry(),
        precision=LLAMA33_70B_GALAXY_ACCURACY,
        weights=_shaped_weights(params, layers=2),
    )
    registration = lazy.prefetch_registration()

    assert LLAMA33_70B_PREFETCHED_WEIGHT_NAMES == ("wqkv", "wo", "w1", "w3", "w2")
    assert [name for name, _ in registration] == [
        f"layer[{index}].{name}" for index in range(2) for name in LLAMA33_70B_PREFETCHED_WEIGHT_NAMES
    ]
    # Decode weights carry the DRAM ring placement; prefill stays interleaved.
    first = lazy.layers[0]
    assert registration[0][1] is first.wqkv
    assert first.wqkv.memory_config != ttnn.DRAM_MEMORY_CONFIG
    assert first.prefill_wqkv.memory_config == ttnn.DRAM_MEMORY_CONFIG
    assert first.wqkv.dtype == first.prefill_wqkv.dtype
    assert first.wqkv.mesh_mapper_config is first.prefill_wqkv.mesh_mapper_config


def test_batched_and_chunked_lengths_resolve_one_recipe_family_each():
    """Every prefill shape the model may be asked for is resolved up front.

    ``Attention2D`` looks its recipe up by identity and fails closed on a miss,
    so a shape that is not registered here can never reach the hot path.
    """

    from models.common.models.galaxy.recipes import resolve_galaxy_prefill_placements
    from models.common.modules.attention.attention_2d import (
        PrefillAttentionMode,
        PrefillCollectiveMode,
        PrefillRecipeIdentity,
        PrefillRowMode,
    )

    params = Llama33_70BGalaxyModelParameters(
        prefill_sequence_lengths=(128, 512),
        batched_prefill_sequence_lengths=(128,),
        chunked_prefill_sequence_lengths=(512,),
    )
    geometry = params.geometry()
    prefill = resolve_galaxy_prefill_placements(geometry, _mesh())
    recipes = galaxy_model._attention_sequence_configs(
        geometry, LLAMA33_70B_GALAXY_ACCURACY, prefill, params.chunked_prefill_sequence_lengths
    )

    def identity(length, row_mode, attention_mode):
        return PrefillRecipeIdentity(length, row_mode, PrefillCollectiveMode.REGULAR, attention_mode)

    assert set(recipes) == {
        identity(128, PrefillRowMode.SINGLE_ROW, PrefillAttentionMode.REGULAR),
        identity(512, PrefillRowMode.SINGLE_ROW, PrefillAttentionMode.REGULAR),
        identity(512, PrefillRowMode.SINGLE_ROW, PrefillAttentionMode.PREFIX_CHUNKED),
        identity(128, PrefillRowMode.CONCAT_32, PrefillAttentionMode.REGULAR),
    }
    chunked = recipes[identity(512, PrefillRowMode.SINGLE_ROW, PrefillAttentionMode.PREFIX_CHUNKED)]
    plain = recipes[identity(512, PrefillRowMode.SINGLE_ROW, PrefillAttentionMode.REGULAR)]
    # Chunked SDPA reads the paged cache, so its chunks come from the page-table
    # alignment rather than the request length.
    assert chunked.sdpa_program_config is prefill.chunked_sdpa_program_config
    assert chunked.sdpa_program_config is not plain.sdpa_program_config
    assert chunked.qkv_program_config is plain.qkv_program_config


def test_batched_prefill_is_off_by_default():
    params = Llama33_70BGalaxyModelParameters()

    assert params.batched_prefill_sequence_lengths == ()
    assert params.chunked_prefill_sequence_lengths == ()
    assert params.geometry().batched_prefill_sequence_lengths == ()


def test_a_chunked_length_without_its_plain_recipe_fails_closed():
    with pytest.raises(ValueError, match="chunked prefill lengths must also be plain prefill lengths"):
        Llama33_70BGalaxyModelParameters(prefill_sequence_lengths=(128,), chunked_prefill_sequence_lengths=(512,))


def test_package_owns_its_graph_and_imports_no_model_named_implementation():
    for module in (galaxy_model, hf_adaptor, weight_utils):
        source = inspect.getsource(module)
        assert "models.demos" not in source
        assert "models.tt_transformers" not in source
        assert "models.common.models.llama33_70b." not in source
        assert "models.common.models.qwen3_32b" not in source
        # The graph is package-owned: only topology-neutral Galaxy machinery is
        # borrowed, never the shared dense-transformer composition.
        assert "galaxy.dense_transformer" not in source


# ---------------------------------------------------------------------------
# Fused-norm statistics placement (Milestone A defect D1)
# ---------------------------------------------------------------------------


def _norm_ccl(mesh):
    def context(mode):
        return SimpleNamespace(
            mesh_device=mesh,
            mode=mode,
            worker_sub_device_id=f"{{mode}}-worker",
            resources=lambda *_args, **_kwargs: None,
            next_semaphore_handles=lambda *_args, **_kwargs: None,
            next_barrier_semaphore_handle=lambda *_args, **_kwargs: None,
        )

    return SimpleNamespace(context=context, mesh_device=mesh)


def test_distributed_norm_resolves_its_statistics_onto_the_decode_input_origin(monkeypatch):
    """The stats shard must sit on the first core of the norm input shard grid.

    `fused_rms_minimal` builds its stats circular buffer on that core and binds
    it to the stats tensor's L1 address, so any other placement reduces
    unrelated L1 - Milestone A defect D1, which `RMSNorm2D` now rejects outright
    via `_require_fused_stats_placement`. This model must therefore not name a
    stats core of its own; it has to let the module resolve one. The guard is
    here because a model that disagreed would fail on device at the first fused
    decode norm of every layer, and nothing else catches it on host.
    """

    monkeypatch.setattr(rmsnorm_2d, "resolve_lazy_weight", lambda weight, **_: weight)
    monkeypatch.setattr(ttnn, "ShardTensor2dMesh", lambda *_args, **_kwargs: "shard-2d-mapper")
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda *_args, **_kwargs: "replicate-mapper")
    mesh = _mesh()
    params = Llama33_70BGalaxyModelParameters(n_layers=1)
    geometry = params.geometry()
    decode_placements = resolve_galaxy_decode_placements(geometry, mesh)
    ccl = _norm_ccl(mesh)

    config = galaxy_model._norm_config(
        LazyWeight(source=torch.zeros(params.dim, dtype=torch.bfloat16), device=mesh),
        mesh_device=mesh,
        geometry=geometry,
        precision=LLAMA33_70B_GALAXY_ACCURACY,
        resources=SimpleNamespace(ccl=ccl),
        prefetch_contexts=(None, None),
        decode_placements=decode_placements,
        eps=params.rms_norm_eps,
        residual_policy=RMSNorm2DResidualPolicy.FUSED_DECODE,
    )

    assert config.decode_stats_memcfg is None, "the model must not pin a stats core; RMSNorm2D owns it"

    resolved = _resolve_2d_config(config)
    input_origin = resolved.decode_input_memcfg.shard_spec.grid.bounding_box().start
    stats_origin = resolved.decode_stats_memcfg.shard_spec.grid.bounding_box().start

    assert (stats_origin.x, stats_origin.y) == (input_origin.x, input_origin.y)

    # And the persistent buffer the decode plan allocates for that collective is
    # the tensor `_require_fused_stats_placement` actually inspects, so it has to
    # land on the same core.
    stats_plan = next(
        plan
        for plan in build_galaxy_decode_collectives(mesh, geometry, decode_placements)
        if plan.key.operation == "all_gather" and tuple(plan.key.geometry) == (1, 1, geometry.max_batch_size, 32)
    )
    buffer_origin = stats_plan.persistent_output_specs[0].memory_config.shard_spec.grid.bounding_box().start
    assert (buffer_origin.x, buffer_origin.y) == (input_origin.x, input_origin.y)
