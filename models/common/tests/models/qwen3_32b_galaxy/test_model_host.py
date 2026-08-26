# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only contracts for the Galaxy Qwen3-32B reconstruction."""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import ttnn
from models.common.models.galaxy.plans import build_galaxy_decode_collectives
from models.common.models.galaxy.recipes import resolve_galaxy_decode_placements
from models.common.models.qwen3_32b_galaxy import hf_adaptor
from models.common.models.qwen3_32b_galaxy import model as galaxy_model
from models.common.models.qwen3_32b_galaxy import weight_utils
from models.common.models.qwen3_32b_galaxy.model import (
    QWEN3_32B_GALAXY_ACCURACY,
    QWEN3_32B_GALAXY_PERFORMANCE,
    QWEN3_32B_PREFETCHED_WEIGHT_NAMES,
    Qwen3_32BGalaxyLayerWeights,
    Qwen3_32BGalaxyModelParameters,
    Qwen3_32BGalaxyWeights,
    build_qwen3_32b_galaxy_lazy_weights,
    build_qwen3_32b_galaxy_model,
    default_paged_attention_config,
    parameters_from_hf_config,
    validate_qwen3_32b_checkpoint,
)
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm import rmsnorm_2d
from models.common.modules.rmsnorm.rmsnorm_2d import RMSNorm2DGeometry, RMSNorm2DResidualPolicy, _resolve_2d_config


def _hf_config(**overrides):
    values = dict(
        num_hidden_layers=64,
        hidden_size=5120,
        num_attention_heads=64,
        num_key_value_heads=8,
        intermediate_size=25600,
        vocab_size=151936,
        head_dim=128,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        max_position_embeddings=40960,
        attention_bias=False,
        tie_word_embeddings=False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _linear(weight, bias=None):
    return SimpleNamespace(weight=weight, bias=bias)


def _self_attn(*, dim=8, n_heads=4, n_kv_heads=2, head_dim=4, qk_norm=True, bias=False):
    def rows(count):
        return torch.arange(count * dim, dtype=torch.float32).reshape(count, dim)

    def norm(offset):
        return SimpleNamespace(weight=torch.arange(head_dim, dtype=torch.float32) + offset)

    return SimpleNamespace(
        config=SimpleNamespace(
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads,
            head_dim=head_dim,
            hidden_size=dim,
        ),
        head_dim=head_dim,
        q_proj=_linear(
            rows(n_heads * head_dim), torch.arange(n_heads * head_dim, dtype=torch.float32) if bias else None
        ),
        k_proj=_linear(
            rows(n_kv_heads * head_dim), torch.arange(n_kv_heads * head_dim, dtype=torch.float32) if bias else None
        ),
        v_proj=_linear(
            rows(n_kv_heads * head_dim),
            torch.arange(n_kv_heads * head_dim, dtype=torch.float32) + 500 if bias else None,
        ),
        o_proj=_linear(torch.arange(dim * n_heads * head_dim, dtype=torch.float32).reshape(dim, n_heads * head_dim)),
        q_norm=norm(0) if qk_norm else None,
        k_norm=norm(50) if qk_norm else None,
    )


# ---------------------------------------------------------------------------
# Checkpoint contract
# ---------------------------------------------------------------------------


def test_checkpoint_contract_accepts_the_exact_product():
    validate_qwen3_32b_checkpoint(_hf_config(), n_layers=1)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"hidden_size": 4096}, "Unexpected Qwen3-32B geometry"),
        ({"num_hidden_layers": 32}, "Unexpected Qwen3-32B geometry"),
        ({"head_dim": 80}, "Unexpected Qwen3-32B geometry"),
        ({"vocab_size": 152064}, "Unexpected Qwen3-32B geometry"),
        ({"attention_bias": True}, "bias-free"),
        ({"tie_word_embeddings": True}, "untied LM head"),
    ],
)
def test_checkpoint_contract_fails_closed(overrides, message):
    with pytest.raises(ValueError, match=message):
        validate_qwen3_32b_checkpoint(_hf_config(**overrides))


# ---------------------------------------------------------------------------
# Parameters and geometry
# ---------------------------------------------------------------------------


def test_parameters_resolve_the_decoupled_head_geometry():
    params = parameters_from_hf_config(_hf_config(), max_seq_len=2048, prefill_sequence_lengths=(128, 2048))
    geometry = params.geometry()

    assert (params.dim, params.n_heads, params.n_kv_heads, params.head_dim) == (5120, 64, 8, 128)
    assert (params.hidden_dim, params.vocab_size, params.n_layers) == (25600, 151936, 64)
    assert params.rope_theta == 1000000.0
    # Qwen3 decouples head_dim from the hidden size: attention projects to 8192
    # and the output projection reduces back to 5120.
    assert params.attention_dim == 8192
    assert geometry.attention_dim == 8192
    assert geometry.local_attention_dim == 1024
    assert geometry.local_dim == 1280
    assert params.padded_vocab_size == 152064
    assert geometry.local_padded_vocab_size == 152064 // 8


def test_parameters_support_a_one_layer_model():
    params = parameters_from_hf_config(_hf_config(), n_layers=1)

    assert params.n_layers == 1
    assert params.qk_norm is True
    assert params.with_layers(2).n_layers == 2


def test_parameters_reject_an_impossible_layer_count():
    with pytest.raises(ValueError, match=r"n_layers must be in \[1, 64\]"):
        Qwen3_32BGalaxyModelParameters(n_layers=65)


def test_default_paged_geometry_covers_the_physical_batch():
    paged = default_paged_attention_config(Qwen3_32BGalaxyModelParameters(max_seq_len=2048))

    assert (paged.block_size, paged.max_num_blocks) == (32, (2048 // 32) * 32)


def test_accuracy_recipe_keeps_the_narrow_mlp_in_bfloat16():
    assert QWEN3_32B_GALAXY_ACCURACY.mlp_w1_w3_dtype == ttnn.bfloat16
    assert QWEN3_32B_GALAXY_ACCURACY.mlp_w2_dtype == ttnn.bfloat16
    assert QWEN3_32B_GALAXY_PERFORMANCE.mlp_w1_w3_dtype == ttnn.bfloat8_b
    assert QWEN3_32B_GALAXY_ACCURACY.kv_cache_dtype == ttnn.bfloat8_b
    assert QWEN3_32B_GALAXY_PERFORMANCE.wqkv_dtype == QWEN3_32B_GALAXY_ACCURACY.wqkv_dtype


# ---------------------------------------------------------------------------
# Provider conversion
# ---------------------------------------------------------------------------


def test_head_local_vectors_use_the_same_interleave_as_the_projections():
    source = torch.arange(8, dtype=torch.float32)

    interleaved = weight_utils.reverse_permute_1d(source)

    assert interleaved.tolist() == [0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]
    with pytest.raises(ValueError, match="width must be even"):
        weight_utils.reverse_permute_1d(torch.arange(3, dtype=torch.float32))


def test_attention_conversion_returns_qk_norms_and_no_bias():
    attn = _self_attn()

    wqkv, wo, q_norm, k_norm, wqkv_bias = weight_utils.attention_weights_from_hf_layer(attn, rows=2)

    assert wqkv.shape == (8, 4 * 4 + 2 * 4 + 2 * 4)
    assert wo.shape == (4 * 4, 8)
    assert q_norm is not None and q_norm.shape == (4,)
    assert k_norm is not None and k_norm.shape == (4,)
    assert torch.equal(q_norm, weight_utils.reverse_permute_1d(attn.q_norm.weight))
    assert wqkv_bias is None


def test_attention_conversion_packs_a_present_bias_row_major():
    attn = _self_attn(bias=True)

    _, _, _, _, wqkv_bias = weight_utils.attention_weights_from_hf_layer(attn, rows=2)

    assert wqkv_bias is not None
    assert wqkv_bias.shape == (4 * 4 + 2 * 4 + 2 * 4,)
    # Mesh row 0 owns the first half of Q, then of K, then of V. The Q and K
    # halves are interleaved per head; V is not.
    assert wqkv_bias[8:12].tolist() == [0.0, 2.0, 1.0, 3.0]
    assert wqkv_bias[12:16].tolist() == [500.0, 501.0, 502.0, 503.0]


def test_attention_conversion_reports_missing_qk_norms():
    attn = _self_attn(qk_norm=False)

    _, _, q_norm, k_norm, _ = weight_utils.attention_weights_from_hf_layer(attn, rows=2)

    assert (q_norm, k_norm) == (None, None)


def test_lm_head_weight_pads_qwen_vocabulary_with_zero_columns():
    lm_head = SimpleNamespace(weight=torch.arange(24, dtype=torch.float32).reshape(6, 4))

    weight = weight_utils.lm_head_weight_torch(lm_head, dim=4, vocab_size=6, padded_vocab_size=8)

    assert weight.shape == (4, 8)
    assert torch.all(weight[:, 6:] == 0)


# ---------------------------------------------------------------------------
# Assembly contracts
# ---------------------------------------------------------------------------


def _weights(layers: int, *, qk_norm: bool = True) -> Qwen3_32BGalaxyWeights:
    layer = Qwen3_32BGalaxyLayerWeights(
        wqkv=torch.zeros(1),
        wo=torch.zeros(1),
        w1=torch.zeros(1),
        w2=torch.zeros(1),
        w3=torch.zeros(1),
        attention_norm=torch.zeros(1),
        ff_norm=torch.zeros(1),
        q_norm=torch.zeros(1) if qk_norm else None,
        k_norm=torch.zeros(1) if qk_norm else None,
    )
    return Qwen3_32BGalaxyWeights(
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


def _shaped_weights(params: Qwen3_32BGalaxyModelParameters, *, layers: int) -> Qwen3_32BGalaxyWeights:
    """Correctly shaped host tensors with no data, for placement contracts."""

    def zeros(*shape: int):
        return torch.zeros(shape, dtype=torch.bfloat16)

    layer = Qwen3_32BGalaxyLayerWeights(
        wqkv=zeros(params.dim, params.head_dim * (params.n_heads + 2 * params.n_kv_heads)),
        # Qwen3 reduces n_heads * head_dim back to dim, so WO is not square.
        wo=zeros(params.attention_dim, params.dim),
        w1=zeros(params.dim, params.hidden_dim),
        w2=zeros(params.hidden_dim, params.dim),
        w3=zeros(params.dim, params.hidden_dim),
        attention_norm=zeros(params.dim),
        ff_norm=zeros(params.dim),
        q_norm=zeros(params.head_dim),
        k_norm=zeros(params.head_dim),
    )
    return Qwen3_32BGalaxyWeights(
        embedding=zeros(params.vocab_size, params.dim),
        rope_cos=zeros(1, 1, params.rope_table_len(), params.head_dim),
        rope_sin=zeros(1, 1, params.rope_table_len(), params.head_dim),
        layers=tuple(layer for _ in range(layers)),
        final_norm=zeros(params.dim),
        lm_head=zeros(params.dim, params.padded_vocab_size),
    )


def test_model_assembly_requires_per_head_qk_norm_weights():
    params = Qwen3_32BGalaxyModelParameters(n_layers=1)

    with pytest.raises(ValueError, match="per-head Q and K normalization"):
        build_qwen3_32b_galaxy_model(object(), params=params, weights=_weights(1, qk_norm=False))


def test_model_assembly_rejects_a_layer_count_mismatch():
    params = Qwen3_32BGalaxyModelParameters(n_layers=2)

    with pytest.raises(ValueError, match="expected 2 layer weight sets, got 1"):
        build_qwen3_32b_galaxy_model(object(), params=params, weights=_weights(1))


def test_runtime_config_reports_the_galaxy_batched_prefill_policy():
    runtime = hf_adaptor.Qwen3_32BGalaxyRuntimeConfig(
        model_name="Qwen3-32B",
        model_cache_path=None,
        max_context_len=40960,
        max_seq_len=2048,
        max_prefill_chunk_size=2048,
        trace_prefill_supported_seq_lens=(128,),
        n_layers=1,
        n_kv_heads=8,
        head_dim=128,
    )

    assert runtime.max_batch_size == 32
    assert runtime.minimum_active_prefill_rows == 16
    assert runtime.allow_cached_prefix_batching is False
    assert runtime.can_enable_trace(128) is True
    assert runtime.can_enable_trace(2048) is False


def test_prefetch_registration_is_ordered_per_layer():
    mesh = _mesh()
    params = Qwen3_32BGalaxyModelParameters(n_layers=2)
    lazy = build_qwen3_32b_galaxy_lazy_weights(
        mesh_device=mesh,
        geometry=params.geometry(),
        precision=QWEN3_32B_GALAXY_ACCURACY,
        weights=_shaped_weights(params, layers=2),
    )
    registration = lazy.prefetch_registration()

    assert QWEN3_32B_PREFETCHED_WEIGHT_NAMES == ("wqkv", "wo", "w1", "w3", "w2")
    assert [name for name, _ in registration] == [
        f"layer[{index}].{name}" for index in range(2) for name in QWEN3_32B_PREFETCHED_WEIGHT_NAMES
    ]
    # Decode weights carry the DRAM ring placement; prefill stays interleaved.
    first = lazy.layers[0]
    assert registration[0][1] is first.wqkv
    assert first.wqkv.memory_config != ttnn.DRAM_MEMORY_CONFIG
    assert first.prefill_wqkv.memory_config == ttnn.DRAM_MEMORY_CONFIG
    assert first.wqkv.dtype == first.prefill_wqkv.dtype
    assert first.wqkv.mesh_mapper_config is first.prefill_wqkv.mesh_mapper_config
    # The per-head Q/K norms are layer state, never prefetched ring operands.
    registered = {id(weight) for _, weight in registration}
    assert id(first.q_norm) not in registered
    assert id(first.k_norm) not in registered


def test_qk_norms_resolve_to_head_local_geometry():
    mesh = _mesh()
    params = Qwen3_32BGalaxyModelParameters(n_layers=1)
    lazy = build_qwen3_32b_galaxy_lazy_weights(
        mesh_device=mesh,
        geometry=params.geometry(),
        precision=QWEN3_32B_GALAXY_ACCURACY,
        weights=_shaped_weights(params, layers=1),
    )
    layer = lazy.layers[0]

    for weight in (layer.q_norm, layer.k_norm):
        config = galaxy_model._head_local_norm_config(
            weight, mesh_device=mesh, precision=QWEN3_32B_GALAXY_ACCURACY, eps=params.rms_norm_eps
        )

        # Attention2D rejects any other geometry, and RMSNorm2D derives the
        # normalized width from the weight itself.
        assert config.geometry is RMSNorm2DGeometry.HEAD_LOCAL
        assert config.weight.source.numel() == params.head_dim
        assert tuple(config.weight.source.shape)[-1] == params.head_dim
        assert config.eps == params.rms_norm_eps
        # Head-local normalization issues no collective, so it borrows no CCL.
        assert config.tt_ccl is None
        assert config.decode_input_memcfg == ttnn.DRAM_MEMORY_CONFIG
        assert config.decode_output_memcfg == ttnn.DRAM_MEMORY_CONFIG
        assert config.prefill_input_memcfg == ttnn.DRAM_MEMORY_CONFIG
        assert config.prefill_output_memcfg == ttnn.DRAM_MEMORY_CONFIG


def test_lazy_weights_reject_a_fused_qkv_bias():
    mesh = _mesh()
    params = Qwen3_32BGalaxyModelParameters(n_layers=1)
    weights = _shaped_weights(params, layers=1)
    qkv_size = params.head_dim * (params.n_heads + 2 * params.n_kv_heads)
    biased = replace(
        weights,
        layers=(replace(weights.layers[0], wqkv_bias=torch.zeros(qkv_size, dtype=torch.bfloat16)),),
    )

    with pytest.raises(ValueError, match="fused QKV bias"):
        build_qwen3_32b_galaxy_lazy_weights(
            mesh_device=mesh,
            geometry=params.geometry(),
            precision=QWEN3_32B_GALAXY_ACCURACY,
            weights=biased,
        )


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

    params = Qwen3_32BGalaxyModelParameters(
        prefill_sequence_lengths=(128, 512),
        batched_prefill_sequence_lengths=(128,),
        chunked_prefill_sequence_lengths=(512,),
    )
    geometry = params.geometry()
    prefill = resolve_galaxy_prefill_placements(geometry, _mesh())
    recipes = galaxy_model._attention_sequence_configs(
        geometry, QWEN3_32B_GALAXY_ACCURACY, prefill, params.chunked_prefill_sequence_lengths
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
    params = Qwen3_32BGalaxyModelParameters()

    assert params.batched_prefill_sequence_lengths == ()
    assert params.chunked_prefill_sequence_lengths == ()
    assert params.geometry().batched_prefill_sequence_lengths == ()


def test_a_chunked_length_without_its_plain_recipe_fails_closed():
    with pytest.raises(ValueError, match="chunked prefill lengths must also be plain prefill lengths"):
        Qwen3_32BGalaxyModelParameters(prefill_sequence_lengths=(128,), chunked_prefill_sequence_lengths=(512,))


def test_package_owns_its_graph_and_imports_no_model_named_implementation():
    for module in (galaxy_model, hf_adaptor, weight_utils):
        source = inspect.getsource(module)
        assert "models.demos" not in source
        assert "models.tt_transformers" not in source
        assert "models.common.models.qwen3_32b." not in source
        assert "models.common.models.llama33_70b" not in source
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
    params = Qwen3_32BGalaxyModelParameters(n_layers=1)
    geometry = params.geometry()
    decode_placements = resolve_galaxy_decode_placements(geometry, mesh)
    ccl = _norm_ccl(mesh)

    config = galaxy_model._norm_config(
        LazyWeight(source=torch.zeros(params.dim, dtype=torch.bfloat16), device=mesh),
        mesh_device=mesh,
        geometry=geometry,
        precision=QWEN3_32B_GALAXY_ACCURACY,
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
