# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Device-free contracts and an explicit P150x2 gate for the TT DFlash core."""

from __future__ import annotations

import inspect
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import (
    close_mesh,
    compose_replicated,
    open_mesh,
    resolve_profile,
)
from models.autoports.poolside_laguna_xs_2_1.tt.dflash_reference import (
    DEFAULT_DFLASH_SNAPSHOT,
    DFlashTargetAuxCapture,
    LagunaDFlashCheckpoint,
    LagunaDFlashConfig,
    apply_neox_rope,
    build_proposal_block,
    evaluate_dflash_draft_argmax_accuracy,
    expected_checkpoint_shapes,
)
from models.autoports.poolside_laguna_xs_2_1.tt.dflash_serving import DFlashServedController, DFlashServingEnvelope
from models.autoports.poolside_laguna_xs_2_1.tt.dflash_tt import (
    DFlashTTCore,
    DFlashTTProposalCache,
    build_dflash_decoder_config,
    build_dflash_rope_tables,
    dflash_bf16_policy,
    dflash_layer_checkpoint_names,
    dflash_shared_checkpoint_names,
    map_dflash_layer_state_dict,
    map_dflash_shared_state_dict,
)
from models.autoports.poolside_laguna_xs_2_1.tt.model import LagunaModel, load_top_level_tensors
from models.autoports.poolside_laguna_xs_2_1.tt.multichip_decoder import _cache_layer_identity
from models.autoports.poolside_laguna_xs_2_1.tt.optimized_decoder import LayerConfig


def _published_config() -> LagunaDFlashConfig:
    config = LagunaDFlashConfig(
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=5,
        num_attention_heads=64,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=100352,
        draft_vocab_size=100352,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        rope_theta=500_000.0,
        sliding_window=512,
        hidden_act="silu",
        attention_bias=False,
        gating="per-head",
        num_experts=0,
        architectures=("DFlashLagunaForCausalLM",),
        torch_dtype="bfloat16",
        layer_types=("sliding_attention",) * 5,
        aux_hidden_state_layer_ids=(2, 14, 26, 34, 40),
        target_layer_ids=(1, 13, 25, 33, 39),
        block_size=16,
        mask_token_id=12,
        causal=True,
    )
    config.validate()
    return config


def _meta_checkpoint(config: LagunaDFlashConfig) -> dict[str, torch.Tensor]:
    return {
        name: torch.empty(shape, dtype=torch.bfloat16, device="meta")
        for name, shape in expected_checkpoint_shapes(config).items()
    }


def test_corrected_decoder_config_is_dense_full_rotary_swa():
    source = _published_config()
    config = build_dflash_decoder_config(source)

    assert config.layer_types == ("sliding_attention",) * 5
    assert config.mlp_only_layers == tuple(range(5))
    assert config.num_experts == 0
    assert config.rope_theta == 500_000.0
    for branch in ("full_attention", "sliding_attention"):
        assert config.rope_parameters[branch]["rope_theta"] == 500_000.0
        assert config.rope_parameters[branch]["partial_rotary_factor"] == 1.0

    for layer_idx in range(5):
        layer = LayerConfig.from_hf(config, layer_idx)
        assert layer.is_sliding and layer.sliding_window == 512
        assert not layer.is_moe and layer.intermediate == 8192
        assert (layer.num_heads, layer.num_kv_heads, layer.head_dim) == (64, 8, 128)
        assert layer.rotary_dim == layer.head_dim == 128


def test_full_rope_tables_match_reference_neox_rotation(expect_error):
    config = _published_config()
    cos, sin = build_dflash_rope_tables(config, 19)
    positions = torch.tensor([0, 7, 18])
    x = torch.linspace(-1.0, 1.0, 3 * 2 * config.head_dim).reshape(3, 2, config.head_dim)

    left, right = x.split(config.head_dim // 2, dim=-1)
    c = cos[positions, : config.head_dim // 2].unsqueeze(1)
    s = sin[positions, : config.head_dim // 2].unsqueeze(1)
    table_result = torch.cat((left * c - right * s, right * c + left * s), dim=-1)
    reference = apply_neox_rope(x, positions, theta=config.rope_theta)
    torch.testing.assert_close(table_result, reference, rtol=1e-6, atol=1e-6)

    with expect_error(ValueError, "max_seq_len"):
        build_dflash_rope_tables(config, 0)
    with expect_error(ValueError, "max_seq_len"):
        build_dflash_rope_tables(config, config.max_position_embeddings + 1)


def test_strict_layer_mapping_splits_fused_qkv_rows():
    config = _published_config()
    state = _meta_checkpoint(config)
    prefix = "layers.0."
    # A narrow expanded row-marker tensor proves Q/K/V ordering without allocating
    # the checkpoint's full 40 MiB fused matrix in this CPU-only test.
    markers = torch.empty((config.fused_qkv_size, 1), dtype=torch.bfloat16)
    markers[: config.q_size] = 1
    markers[config.q_size : config.q_size + config.kv_size] = 2
    markers[config.q_size + config.kv_size :] = 3
    state[prefix + "self_attn.qkv_proj.weight"] = markers.expand(-1, config.hidden_size)

    mapped = map_dflash_layer_state_dict(state, config, 0)
    assert set(mapped) == {
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.g_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    assert mapped["self_attn.q_proj.weight"].shape == (8192, 2048)
    assert mapped["self_attn.k_proj.weight"].shape == (1024, 2048)
    assert mapped["self_attn.v_proj.weight"].shape == (1024, 2048)
    assert mapped["self_attn.q_proj.weight"][0, 0].item() == 1
    assert mapped["self_attn.k_proj.weight"][0, 0].item() == 2
    assert mapped["self_attn.v_proj.weight"][0, 0].item() == 3
    assert len(dflash_layer_checkpoint_names(config, 0)) == 10


@pytest.mark.parametrize("fault", ["missing", "unexpected", "shape", "dtype"])
def test_strict_layer_mapping_rejects_checkpoint_drift(fault, expect_error):
    config = _published_config()
    state = _meta_checkpoint(config)
    key = "layers.2.self_attn.q_norm.weight"
    if fault == "missing":
        del state[key]
    elif fault == "unexpected":
        state["layers.2.self_attn.bias"] = torch.empty((1,), dtype=torch.bfloat16, device="meta")
    elif fault == "shape":
        state[key] = torch.empty((127,), dtype=torch.bfloat16, device="meta")
    else:
        state[key] = torch.empty((128,), dtype=torch.float32, device="meta")
    match = {"shape": "shape_mismatch", "dtype": "non_bf16"}.get(fault, fault)
    with expect_error(ValueError, match):
        map_dflash_layer_state_dict(state, config, 2)


def test_strict_shared_mapping_loads_only_draft_owned_weights(expect_error):
    config = _published_config()
    state = _meta_checkpoint(config)
    shared = map_dflash_shared_state_dict(state, config)
    assert tuple(shared) == dflash_shared_checkpoint_names(config)
    assert tuple(shared) == (
        "aux_hidden_norms.0.weight",
        "aux_hidden_norms.1.weight",
        "aux_hidden_norms.2.weight",
        "aux_hidden_norms.3.weight",
        "aux_hidden_norms.4.weight",
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
    )
    assert shared["fc.weight"].shape == (2048, 10240)

    state["lm_head.weight"] = torch.empty((100352, 2048), dtype=torch.bfloat16, device="meta")
    with expect_error(ValueError, "unexpected"):
        map_dflash_shared_state_dict(state, config)


def test_cache_namespace_is_disjoint_and_default_is_unchanged(expect_error):
    assert _cache_layer_identity(0) == 0
    assert _cache_layer_identity(0, "dflash") == "dflash_L0"
    assert _cache_layer_identity(0, "dflash") != _cache_layer_identity(0)
    with expect_error(ValueError, "cache_namespace"):
        _cache_layer_identity(0, "../escape")


def test_core_is_default_off_and_requires_bounded_rope_allocation(expect_error):
    class UntouchedMesh:
        def __getattr__(self, name):  # pragma: no cover - any access is a test failure
            raise AssertionError(f"default-off gate touched mesh attribute {name}")

    with expect_error(RuntimeError, "enable_experimental=True"):
        DFlashTTCore.from_checkpoint(UntouchedMesh())
    with expect_error(ValueError, "max_seq_len"):
        DFlashTTCore.from_checkpoint(UntouchedMesh(), enable_experimental=True)


def test_bf16_policy_does_not_quantize_core_weights_or_cache():
    policy = dflash_bf16_policy()
    for field in policy._DTYPE_FIELDS:
        assert getattr(policy, field) == ttnn.bfloat16
    for field in policy._FID_FIELDS:
        assert getattr(policy, field) == "HiFi4"

    # Shared fusion and the draft final norm are outside MultichipDecoder, so
    # lock their explicit use of the same HiFi4/fp32-destination kernel too.
    combine_source = inspect.getsource(DFlashTTCore.combine_aux_hidden_states)
    final_norm_source = inspect.getsource(DFlashTTCore.apply_final_norm)
    assert "_ck_hifi4" in combine_source and "compute_kernel_config=precision_ck" in combine_source
    assert "_ck_hifi4" in final_norm_source and "compute_kernel_config=" in final_norm_source


def test_target_capture_paths_are_explicit_separate_and_default_off(monkeypatch, expect_error):
    """Capture the exact post-layer IDs without adding work to normal forwards."""

    assert "dflash" not in inspect.getsource(LagunaModel.prefill_layers).lower()
    assert "dflash" not in inspect.getsource(LagunaModel.decode_layers).lower()

    class FakeLayer:
        PIPE_CHUNK = 8192

        def __init__(self, index):
            self.layer_idx = index
            self.cfg = SimpleNamespace(attention_type="sliding_attention")

        def prefill_forward(self, hidden, *args, **kwargs):
            return hidden + (self.layer_idx + 1)

        def decode_forward(self, hidden, *args, **kwargs):
            return hidden + (self.layer_idx + 1)

    model = object.__new__(LagunaModel)
    model.layers = [FakeLayer(index) for index in range(40)]
    model.cfg = SimpleNamespace(hidden=2048, max_position_embeddings=262144)
    model._build_prefill_rope = lambda *args, **kwargs: {"sliding_attention": (object(), object())}
    model._build_decode_rope = lambda *args, **kwargs: {"sliding_attention": (object(), object())}
    monkeypatch.setattr(
        ttnn,
        "slice",
        lambda value, starts, ends: value[starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]],
    )
    monkeypatch.setattr(ttnn, "concat", lambda values, dim: torch.cat(tuple(values), dim=dim))
    monkeypatch.setattr(ttnn, "reshape", lambda value, shape: value.reshape(shape))
    hidden = torch.zeros((1, 4, 2048))

    with expect_error(RuntimeError, "default-off"):
        model.prefill_layers_with_dflash_aux(hidden, [None] * 40, object())
    final, capture = model.prefill_layers_with_dflash_aux(
        hidden,
        [None] * 40,
        object(),
        start_pos=100,
        valid_seq_len=3,
        enable_experimental=True,
    )
    assert final[0, 0, 0].item() == sum(range(1, 41))
    assert (capture.start_position, capture.row_count, capture.end_position) == (100, 3, 102)
    capture.validate(_published_config())
    slices = capture.hidden_states.reshape(1, 3, 5, 2048)
    cumulative = [sum(range(1, layer + 2)) for layer in (1, 13, 25, 33, 39)]
    assert slices[0, 0, :, 0].tolist() == cumulative

    with expect_error(RuntimeError, "default-off"):
        model.decode_layers_with_dflash_aux(
            torch.zeros((1, 1, 1, 2048)), object(), object(), object(), [None] * 40, absolute_position=103
        )
    _, decode_capture = model.decode_layers_with_dflash_aux(
        torch.zeros((1, 1, 1, 2048)),
        object(),
        object(),
        object(),
        [None] * 40,
        absolute_position=103,
        enable_experimental=True,
    )
    decode_capture.validate(_published_config())
    assert (decode_capture.start_position, decode_capture.row_count) == (103, 1)

    verify_hidden = torch.zeros((1, 1, 3, 2048))
    with expect_error(ValueError, "sequential_kv_write=True"):
        model.decode_layers_with_dflash_aux(
            verify_hidden,
            object(),
            object(),
            object(),
            [None] * 40,
            absolute_position=104,
            enable_experimental=True,
        )
    _, verify_capture = model.decode_layers_with_dflash_aux(
        verify_hidden,
        object(),
        object(),
        object(),
        [None] * 40,
        absolute_position=104,
        sequential_kv_write=True,
        enable_experimental=True,
    )
    verify_capture.validate(_published_config())
    assert (verify_capture.start_position, verify_capture.row_count, verify_capture.end_position) == (104, 3, 106)
    assert verify_capture.hidden_states.shape == (1, 3, 5 * 2048)


def test_one_round_driver_resets_context_and_carries_only_query(monkeypatch, expect_error):
    config = _published_config()
    mesh = object()
    context_rows = 3
    h = config.hidden_size
    flattened = torch.arange(context_rows * 5 * h, dtype=torch.float32).reshape(1, context_rows, 5 * h)
    capture = DFlashTargetAuxCapture(flattened, start_position=100, row_count=context_rows)

    class FakeLayer:
        def __init__(self, index):
            self.index = index
            self.context_seen = []

        def _rope_prefill(self, start, seq, sin=False):
            assert (start, seq) == (100, 32)
            return ("sin" if sin else "cos", start, seq)

        def prefill_forward(self, value, kv, page_table, **kwargs):
            self.context_seen.append(value[:, :context_rows].clone())
            result = value.clone()
            result[:, :context_rows] += 1000 + self.index
            result[:, context_rows:] += self.index + 1
            return result

    core = object.__new__(DFlashTTCore)
    core.config = config
    core.layers = {index: FakeLayer(index) for index in range(5)}
    core.mesh_device = mesh
    core.max_seq_len = 1024
    fused_context = flattened.reshape(1, context_rows, 5, h).mean(dim=2)
    core.combine_aux_hidden_states = lambda value: fused_context.clone()
    core.apply_final_norm = lambda value: value * 2

    cache = object.__new__(DFlashTTProposalCache)
    cache.core = core
    cache.block_size = 32
    cache.capacity = 544
    cache.kv_cache = {index: object() for index in range(5)}
    cache.page_tables = {index: object() for index in range(5)}
    cache._request_id = "request-1"
    cache._closed = False
    cache._context = capture.hidden_states
    cache._context_owned = False
    cache._context_start = capture.start_position
    cache._context_rows = capture.row_count

    class TargetOwner:
        device = mesh
        cfg = SimpleNamespace(hidden=h, vocab=config.vocab_size)

        def __init__(self):
            self.raw_projection_calls = 0

        def embed_prefill(self, ids):
            return ids.to(torch.float32).unsqueeze(-1).expand(*ids.shape, h) / 100

        def lm_head_shards_dflash(self, hidden, *, enable_experimental=False):
            assert enable_experimental
            self.raw_projection_calls += 1
            return hidden[..., :7]

    target = TargetOwner()
    monkeypatch.setattr(ttnn, "from_torch", lambda value, **kwargs: value)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda value: object())
    monkeypatch.setattr(ttnn, "concat", lambda values, dim: torch.cat(tuple(values), dim=dim))
    monkeypatch.setattr(
        ttnn,
        "slice",
        lambda value, starts, ends: value[starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]],
    )

    with expect_error(RuntimeError, "default-off"):
        core.proposal_round(cache, target_model=target, bonus_token_id=7)
    result = core.proposal_round(
        cache,
        target_model=target,
        bonus_token_id=7,
        enable_experimental=True,
    )
    assert result.block.input_ids.tolist() == [7] + [12] * 15
    assert result.logits_shards.shape == (1, 15, 7)
    assert target.raw_projection_calls == 1
    for layer in core.layers.values():
        torch.testing.assert_close(layer.context_seen[0], fused_context)
    # The semantic query accumulates +1,+2,+3,+4,+5, then the draft norm doubles it.
    expected = (result.block.input_ids[1:16].to(torch.float32).unsqueeze(-1).expand(-1, h) / 100 + 15) * 2
    torch.testing.assert_close(result.sampled_hidden_states[0], expected)


def test_proposal_cache_lifetime_and_contiguous_511_row_retention(monkeypatch, expect_error):
    config = _published_config()
    cache = object.__new__(DFlashTTProposalCache)
    cache.core = SimpleNamespace(config=config)
    cache.max_context_rows = 511
    cache._request_id = None
    cache._context = None
    cache._context_owned = False
    cache._context_start = None
    cache._context_rows = 0
    cache._closed = False
    cache.kv_cache = {}
    cache.page_tables = {}
    monkeypatch.setattr(ttnn, "concat", lambda values, dim: torch.cat(tuple(values), dim=dim))
    monkeypatch.setattr(
        ttnn,
        "slice",
        lambda value, starts, ends: value[starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2]],
    )

    width = 5 * config.hidden_size
    prefill = DFlashTargetAuxCapture(torch.zeros((1, 511, width)), start_position=10, row_count=511)
    decode = DFlashTargetAuxCapture(torch.ones((1, 1, width)), start_position=521, row_count=1)
    with expect_error(RuntimeError, "begin_request"):
        cache.update_target_capture(prefill)
    cache.begin_request("r")
    cache.update_target_capture(prefill)
    cache.update_target_capture(decode)
    retained = cache.target_capture()
    assert (retained.start_position, retained.row_count, retained.end_position) == (11, 511, 521)
    assert bool((retained.hidden_states[:, -1] == 1).all())
    with expect_error(ValueError, "not adjacent"):
        cache.update_target_capture(DFlashTargetAuxCapture(torch.zeros((1, 1, width)), start_position=700, row_count=1))
    with expect_error(RuntimeError, "cache owns"):
        cache.end_request("wrong")
    cache.end_request("r")
    with expect_error(RuntimeError, "no active target context"):
        cache.target_capture()
    cache.close()
    assert cache.closed
    with expect_error(RuntimeError, "closed"):
        cache.begin_request("next")


_RUN_HW = os.environ.get("TT_LAGUNA_RUN_DFLASH_TT_HW", "0").strip().lower() in {"1", "true", "yes"}
_SNAPSHOT = Path(os.environ.get("LAGUNA_DFLASH_SNAPSHOT", DEFAULT_DFLASH_SNAPSHOT))
_HAS_CHECKPOINT = (_SNAPSHOT / "config.json").is_file() and (_SNAPSHOT / "model.safetensors").is_file()


@pytest.mark.skipif(not _RUN_HW, reason="set TT_LAGUNA_RUN_DFLASH_TT_HW=1 for the isolated P150x2 gate")
@pytest.mark.skipif(not _HAS_CHECKPOINT, reason="published Laguna DFlash checkpoint is unavailable")
@torch.inference_mode()
def test_one_layer_d2_bf16_prefill_pcc_chips_2_3():
    """Qualify shared aux fusion plus draft layer 0 against the CPU reference."""

    if os.environ.get("TT_VISIBLE_DEVICES") != "2,3":
        pytest.fail("DFlash hardware proof is pinned to TT_VISIBLE_DEVICES=2,3")
    profile = resolve_profile("p150x2", trace_region_size=200_000_000)
    mesh = open_mesh(ttnn, profile)
    try:
        core = DFlashTTCore.from_checkpoint(
            mesh,
            snapshot=_SNAPSHOT,
            layer_indices=(0,),
            max_seq_len=64,
            policy=dflash_bf16_policy(),
            enable_experimental=True,
        )
        layer = core.layers[0]
        assert layer.D == 2
        assert (layer.cfg.num_heads, layer.cfg.num_kv_heads) == (32, 4)
        assert layer.cfg.is_sliding and layer.cfg.rotary_dim == 128

        reference = LagunaDFlashCheckpoint(_SNAPSHOT).load_reference(layer_indices=(0,))
        h = core.config.hidden_size
        context_tokens = query_tokens = 16
        aux = (((torch.arange(context_tokens * 5 * h) % 97) - 48).float() / 2400).reshape(context_tokens, 5, h)
        query = (((torch.arange(query_tokens * h) % 83) - 41).float() / 2100).reshape(query_tokens, h)
        aux = aux.to(torch.bfloat16)
        query = query.to(torch.bfloat16)
        context_positions = torch.arange(context_tokens)
        query_positions = torch.arange(context_tokens, context_tokens + query_tokens)

        expected_context = reference.combine_aux_hidden_states(aux)
        context_kv = reference.precompute_context_kv(expected_context, context_positions)
        expected_query = reference.forward_query_embeddings(query, query_positions, context_kv)

        replicate = ttnn.ReplicateTensorToMesh(mesh)
        aux_tt = ttnn.from_torch(
            aux.reshape(1, context_tokens, 5 * h),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=replicate,
        )
        query_tt = ttnn.from_torch(
            query.reshape(1, query_tokens, h),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=replicate,
        )
        context_tt = core.combine_aux_hidden_states(aux_tt)
        context_got = compose_replicated(ttnn, context_tt, mesh, profile).reshape(context_tokens, h)

        x_tt = ttnn.concat((context_tt, query_tt), dim=1)
        kv = layer.alloc_kv_cache(max_users=1, max_seq_len=64, block_size=32, dtype=ttnn.bfloat16)
        page_table = layer.make_page_table(1, kv["blocks_per_user"])
        hidden_tt = layer.prefill_forward(x_tt, kv, page_table, user_id=0, start_pos=0)
        normalized_tt = core.apply_final_norm(hidden_tt)
        got = compose_replicated(ttnn, normalized_tt, mesh, profile).reshape(32, h)[context_tokens:]

        def pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
            return float(torch.corrcoef(torch.stack((actual.float().flatten(), expected.float().flatten())))[0, 1])

        context_pcc = pcc(context_got, expected_context)
        query_pcc = pcc(got, expected_query)
        print(f"DFLASH_TT_PCC aux={context_pcc:.8f} query={query_pcc:.8f}")
        assert context_pcc >= 0.995, f"DFlash aux fusion PCC {context_pcc:.6f} < 0.995"
        assert query_pcc >= 0.995, f"DFlash layer-0 prefill PCC {query_pcc:.6f} < 0.995"
    finally:
        close_mesh(ttnn, mesh)


@pytest.mark.skipif(not _RUN_HW, reason="set TT_LAGUNA_RUN_DFLASH_TT_HW=1 for the isolated P150x2 gate")
@pytest.mark.skipif(not _HAS_CHECKPOINT, reason="published Laguna DFlash checkpoint is unavailable")
@torch.inference_mode()
def test_full_five_layer_one_round_pcc_and_warm_latency_chips_2_3():
    """Qualify the exact five-layer driver and raw target-owned projection."""

    if os.environ.get("TT_VISIBLE_DEVICES") != "2,3":
        pytest.fail("DFlash hardware proof is pinned to TT_VISIBLE_DEVICES=2,3")
    profile = resolve_profile("p150x2", trace_region_size=200_000_000)
    mesh = open_mesh(ttnn, profile)
    proposal_cache = None
    try:
        core = DFlashTTCore.from_checkpoint(
            mesh,
            snapshot=_SNAPSHOT,
            max_seq_len=64,
            policy=dflash_bf16_policy(),
            enable_experimental=True,
        )
        assert tuple(core.layers) == (0, 1, 2, 3, 4)
        reference = LagunaDFlashCheckpoint(_SNAPSHOT).load_reference()
        config = core.config
        h = config.hidden_size
        context_rows = 16
        aux = (
            (((torch.arange(context_rows * 5 * h) % 97) - 48).float() / 2400)
            .reshape(context_rows, 5, h)
            .to(torch.bfloat16)
        )
        context_positions = torch.arange(context_rows)
        block = build_proposal_block(config, bonus_token_id=37, last_valid_position=context_rows - 1)

        # The draft checkpoint intentionally owns neither tensor.  Load the
        # target's real BF16 tables and keep ownership on this target facade.
        target_top = load_top_level_tensors(["model.embed_tokens.weight", "lm_head.weight"])
        target_embedding = target_top["model.embed_tokens.weight"].to(torch.bfloat16)
        target_lm_head = target_top["lm_head.weight"].to(torch.bfloat16)
        expected_context = reference.combine_aux_hidden_states(aux)
        expected_kv = reference.precompute_context_kv(expected_context, context_positions)
        expected_query = reference.forward_query_embeddings(
            reference.embed_input_ids(block.input_ids, target_embedding),
            block.positions,
            expected_kv,
        )
        expected_logits = reference.compute_logits(expected_query[block.sample_indices], target_lm_head)

        replicate = ttnn.ReplicateTensorToMesh(mesh)
        embed_w = ttnn.from_torch(
            target_embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        lm_head_w = ttnn.from_torch(
            target_lm_head.t().contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=1),
        )

        class TargetOwner:
            device = mesh
            cfg = SimpleNamespace(hidden=h, vocab=config.vocab_size)

            def embed_prefill(self, token_ids):
                embedded = ttnn.embedding(token_ids, embed_w, layout=ttnn.TILE_LAYOUT)
                return ttnn.reshape(embedded, (1, token_ids.shape[-1], h))

            def lm_head_shards_dflash(self, hidden, *, enable_experimental=False):
                assert enable_experimental
                return ttnn.linear(hidden, lm_head_w, compute_kernel_config=core.layers[0]._ck_hifi4)

        aux_tt = ttnn.from_torch(
            aux.reshape(1, context_rows, 5 * h),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        proposal_cache = core.allocate_proposal_cache(enable_experimental=True)
        proposal_cache.begin_request("hardware-gate")
        proposal_cache.update_target_capture(DFlashTargetAuxCapture(aux_tt, start_position=0, row_count=context_rows))
        target = TargetOwner()
        aux_roundtrip = compose_replicated(ttnn, aux_tt, mesh, profile).reshape(context_rows, 5, h)
        context_tt = core.combine_aux_hidden_states(aux_tt)
        context_got = compose_replicated(ttnn, context_tt, mesh, profile).reshape(context_rows, h)
        memory = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        memory_used = int(memory.total_bytes_allocated_per_bank) * int(memory.num_banks)
        memory_total = int(memory.total_bytes_per_bank) * int(memory.num_banks)

        # Cold compile, then three stable-shape warm rounds.
        result = core.proposal_round(
            proposal_cache,
            target_model=target,
            bonus_token_id=37,
            enable_experimental=True,
        )
        ttnn.synchronize_device(mesh)
        cache_entries = int(mesh.num_program_cache_entries())
        warm_seconds = []
        for _ in range(3):
            started = time.perf_counter()
            result = core.proposal_round(
                proposal_cache,
                target_model=target,
                bonus_token_id=37,
                enable_experimental=True,
            )
            ttnn.synchronize_device(mesh)
            warm_seconds.append(time.perf_counter() - started)
            assert int(mesh.num_program_cache_entries()) == cache_entries

        got_hidden = compose_replicated(
            ttnn,
            result.sampled_hidden_states,
            mesh,
            profile,
        ).reshape(15, h)
        got_logits = ttnn.to_torch(
            result.logits_shards,
            mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1),
        ).reshape(
            15, -1
        )[:, : config.vocab_size]

        def pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
            return float(torch.corrcoef(torch.stack((actual.float().flatten(), expected.float().flatten())))[0, 1])

        hidden_pcc = pcc(got_hidden, expected_query[1:16])
        logits_pcc = pcc(got_logits, expected_logits)
        aux_pcc = pcc(aux_roundtrip, aux)
        context_pcc = pcc(context_got, expected_context)
        draft_accuracy = evaluate_dflash_draft_argmax_accuracy(got_logits, expected_logits)
        top1_matches = int((got_logits.argmax(dim=-1) == expected_logits.argmax(dim=-1)).sum())
        got_top1 = got_logits.argmax(dim=-1)
        expected_top1 = expected_logits.argmax(dim=-1)
        median = sorted(warm_seconds)[1]
        print(
            "DFLASH_TT_FULL5 "
            f"aux_pcc={aux_pcc:.8f} context_pcc={context_pcc:.8f} "
            f"hidden_pcc={hidden_pcc:.8f} logits_pcc={logits_pcc:.8f} "
            f"top1={top1_matches}/15 warm_s={warm_seconds} median_s={median:.6f} "
            f"tied_rows={list(draft_accuracy.tied_rows)} "
            f"program_cache_entries={cache_entries} "
            f"dram_used_mib={memory_used / 2**20:.1f} dram_total_mib={memory_total / 2**20:.1f} "
            f"got_top1={got_top1.tolist()} expected_top1={expected_top1.tolist()}"
        )
        assert aux_pcc == 1.0, f"DFlash auxiliary transfer PCC {aux_pcc:.6f} != 1"
        assert context_pcc >= 0.999, f"DFlash fused context PCC {context_pcc:.6f} < 0.999"
        assert hidden_pcc >= 0.995, f"full-five DFlash hidden PCC {hidden_pcc:.6f} < 0.995"
        assert logits_pcc >= 0.995, f"full-five DFlash logit PCC {logits_pcc:.6f} < 0.995"
        assert not draft_accuracy.tied_rows, "deterministic exact gate unexpectedly contains a reference tie"
        assert draft_accuracy.literal_exact and draft_accuracy.passed
        assert top1_matches == 15, f"full-five DFlash target top-1 matches {top1_matches}/15"
    finally:
        if proposal_cache is not None:
            proposal_cache.close()
        close_mesh(ttnn, mesh)


_RUN_SERVING_HW = os.environ.get("TT_LAGUNA_RUN_DFLASH_SERVING_HW", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}


@pytest.mark.skipif(
    not _RUN_SERVING_HW,
    reason="set TT_LAGUNA_RUN_DFLASH_SERVING_HW=1 for the bounded P150x2 served-controller gate",
)
@pytest.mark.skipif(not _HAS_CHECKPOINT, reason="published Laguna DFlash checkpoint is unavailable")
@torch.inference_mode()
def test_served_controller_full_target_accuracy_and_warm_latency_chips_2_3():
    """Gate full target capture/verify, draft tie contract, and fallback.

    This is a correctness/diagnostic gate, not a performance qualification:
    the first measured real-context round failed the baseline TPOT comparison.
    """

    if os.environ.get("TT_VISIBLE_DEVICES") != "2,3":
        pytest.fail("DFlash served-controller proof is pinned to TT_VISIBLE_DEVICES=2,3")
    profile = resolve_profile("p150x2", trace_region_size=400_000_000)
    mesh = open_mesh(ttnn, profile)
    proposal_cache = None
    controller = None
    try:
        # The target uses its selected production precision except for a BF16 LM
        # head, matching the official DFlash CPU projection contract.
        target = LagunaModel.from_pretrained(
            mesh,
            max_seq_len=128,
            lm_head_dtype=ttnn.bfloat16,
        )
        assert len(target.layers) == 40 and target.D == 2
        core = DFlashTTCore.from_checkpoint(
            mesh,
            snapshot=_SNAPSHOT,
            max_seq_len=128,
            policy=dflash_bf16_policy(),
            enable_experimental=True,
        )
        reference = LagunaDFlashCheckpoint(_SNAPSHOT).load_reference()
        config = core.config
        h = config.hidden_size
        replicate = ttnn.ReplicateTensorToMesh(mesh)

        def device(value, dtype):
            return ttnn.from_torch(
                value,
                dtype=dtype,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=replicate,
            )

        def pcc(actual: torch.Tensor, expected: torch.Tensor) -> float:
            return float(torch.corrcoef(torch.stack((actual.float().flatten(), expected.float().flatten())))[0, 1])

        def host_aux(capture):
            return compose_replicated(ttnn, capture.hidden_states, mesh, profile).reshape(capture.row_count, 5, h)

        def page_table(rows, blocks_per_user):
            host = torch.arange(blocks_per_user, dtype=torch.int32).reshape(1, -1).repeat(rows, 1)
            return device(host, ttnn.int32)

        target_caches = [target.alloc_kv_cache(max_users=1, max_seq_len=128, block_size=64) for _ in range(2)]
        blocks = int(target_caches[0][0]["blocks_per_user"])
        assert blocks == 2
        pt1 = page_table(1, blocks)

        prompt_rows = 32
        prompt_tokens = ((torch.arange(prompt_rows, dtype=torch.int64) * 37 + 11) % config.vocab_size).reshape(1, -1)
        prompt_tt = device(prompt_tokens.to(torch.int32), ttnn.uint32)

        def prefill(cache, *, capture_aux):
            embedded = target.embed_prefill(prompt_tt)
            if capture_aux:
                return target.prefill_layers_with_dflash_aux(
                    embedded,
                    cache,
                    pt1,
                    start_pos=0,
                    valid_seq_len=prompt_rows,
                    enable_experimental=True,
                )
            return target.prefill_layers(embedded, cache, pt1, start_pos=0), None

        hidden_a, initial_capture = prefill(target_caches[0], capture_aux=True)
        hidden_b, _ = prefill(target_caches[1], capture_aux=False)
        last_a = ttnn.slice(hidden_a, [0, prompt_rows - 1, 0], [1, prompt_rows, h])
        last_b = ttnn.slice(hidden_b, [0, prompt_rows - 1, 0], [1, prompt_rows, h])
        prompt_logits_a = target.logits_to_host(target.lm_head_shards_prefill(last_a)).reshape(-1)
        prompt_logits_b = target.logits_to_host(target.lm_head_shards_prefill(last_b)).reshape(-1)
        assert torch.argmax(prompt_logits_a).item() == torch.argmax(prompt_logits_b).item()
        known_bonus = int(torch.argmax(prompt_logits_a))

        # First qualify the target-generated auxiliary state through the exact
        # published five-layer CPU reference and target-owned raw LM head.
        initial_aux = host_aux(initial_capture).to(torch.bfloat16)
        positions = torch.arange(prompt_rows)
        target_top = load_top_level_tensors(["model.embed_tokens.weight", "lm_head.weight"])
        target_embedding = target_top["model.embed_tokens.weight"].to(torch.bfloat16)
        target_lm_head = target_top["lm_head.weight"].to(torch.bfloat16)
        block = build_proposal_block(
            config,
            bonus_token_id=known_bonus,
            last_valid_position=prompt_rows - 1,
        )
        expected_context = reference.combine_aux_hidden_states(initial_aux)
        expected_context_kv = reference.precompute_context_kv(expected_context, positions)
        expected_hidden, expected_layer_outputs = reference.forward_query_embeddings_with_layer_outputs(
            reference.embed_input_ids(block.input_ids, target_embedding),
            block.positions,
            expected_context_kv,
        )
        expected_draft_logits = reference.compute_logits(
            expected_hidden[block.sample_indices],
            target_lm_head,
        )

        proposal_cache = core.allocate_proposal_cache(enable_experimental=True)
        proposal_cache.begin_request("cpu-contract")
        proposal_cache.update_target_capture(initial_capture)
        # Capture the five materialized draft-layer outputs only for this
        # diagnostic gate.  Production proposal calls retain the original
        # methods and allocate no trace state.
        stage_tensors = {}
        original_prefill_methods = {}
        for layer_index, layer in core.layers.items():
            original_prefill_methods[layer_index] = layer.prefill_forward

            def traced_prefill(*args, _layer_index=layer_index, _original=layer.prefill_forward, **kwargs):
                output = _original(*args, **kwargs)
                stage_tensors[_layer_index] = output
                return output

            layer.prefill_forward = traced_prefill
        try:
            proposal = core.proposal_round(
                proposal_cache,
                target_model=target,
                bonus_token_id=known_bonus,
                enable_experimental=True,
            )
        finally:
            for layer_index, original in original_prefill_methods.items():
                core.layers[layer_index].prefill_forward = original
        got_hidden = compose_replicated(
            ttnn,
            proposal.sampled_hidden_states,
            mesh,
            profile,
        ).reshape(15, h)
        got_draft_logits = target.logits_to_host(proposal.logits_shards).reshape(15, config.vocab_size)
        got_layer_outputs = []
        for layer_index in range(config.num_hidden_layers):
            query_stage = ttnn.slice(
                stage_tensors[layer_index],
                [0, prompt_rows, 0],
                [1, prompt_rows + 16, h],
            )
            got_layer_outputs.append(compose_replicated(ttnn, query_stage, mesh, profile).reshape(16, h))
        draft_ids = torch.argmax(got_draft_logits, dim=-1).to(torch.int32)
        expected_draft_ids = torch.argmax(expected_draft_logits, dim=-1).to(torch.int32)
        draft_accuracy = evaluate_dflash_draft_argmax_accuracy(got_draft_logits, expected_draft_logits)
        context_tt = core.combine_aux_hidden_states(initial_capture.hidden_states)
        got_context = compose_replicated(ttnn, context_tt, mesh, profile).reshape(prompt_rows, h)
        proposal_cache.end_request("cpu-contract")

        # Compare one 16-row eager target verify to 16 authoritative B=1
        # target steps on an independently prefilled KV cache.
        verify_tokens = torch.tensor([known_bonus, *draft_ids.tolist()], dtype=torch.int64)
        verify_positions = torch.arange(prompt_rows, prompt_rows + 16, dtype=torch.int32)

        def target_verify(cache, token_ids, absolute_positions):
            token_ids = torch.as_tensor(token_ids, dtype=torch.int64).reshape(-1)
            absolute_positions = torch.as_tensor(absolute_positions, dtype=torch.int32).reshape(-1)
            rows = int(token_ids.numel())
            tok = device(token_ids.reshape(1, rows).to(torch.int32), ttnn.uint32)
            cur = device(absolute_positions, ttnn.int32)
            ridx = device(absolute_positions.reshape(1, rows), ttnn.uint32)
            hidden = target.embed_decode(tok)
            hidden, capture = target.decode_layers_with_dflash_aux(
                hidden,
                cur,
                ridx,
                page_table(rows, blocks),
                cache,
                absolute_position=int(absolute_positions[0]),
                sequential_kv_write=True,
                enable_experimental=True,
            )
            logits = target.logits_to_host(target.lm_head_shards_decode(hidden)).reshape(rows, config.vocab_size)
            return logits, capture

        batched_logits, batched_capture = target_verify(
            target_caches[0],
            verify_tokens,
            verify_positions,
        )
        sequential_logits = []
        sequential_aux = []
        for token, position in zip(verify_tokens, verify_positions):
            logits, capture = target_verify(target_caches[1], [int(token)], [int(position)])
            sequential_logits.append(logits[0])
            sequential_aux.append(host_aux(capture)[0])
        sequential_logits = torch.stack(sequential_logits)
        sequential_aux = torch.stack(sequential_aux)
        batched_aux = host_aux(batched_capture)
        target_batched_ids = torch.argmax(batched_logits, dim=-1).to(torch.int32)
        target_sequential_ids = torch.argmax(sequential_logits, dim=-1).to(torch.int32)
        accepted, expected_committed = DFlashServedController._accept_greedy(
            draft_ids.tolist(),
            target_sequential_ids.tolist(),
        )

        active_target_cache = [target_caches[0]]

        def verify_callback(tokens, absolute_positions, **kwargs):
            logits, capture = target_verify(active_target_cache[0], tokens, absolute_positions)
            return torch.argmax(logits, dim=-1).to(torch.int32).tolist(), capture

        def draft_argmax(round_result):
            logits = target.logits_to_host(round_result.logits_shards).reshape(15, config.vocab_size)
            return torch.argmax(logits, dim=-1).to(torch.int32).tolist()

        controller = DFlashServedController(
            core=core,
            proposal_cache=proposal_cache,
            target_model=target,
            verify_greedy=verify_callback,
            draft_argmax=draft_argmax,
            envelope=DFlashServingEnvelope(enabled=True),
        )

        def controller_round(request_id):
            controller.begin_request(request_id, initial_capture)
            started = time.perf_counter()
            first = controller.serve_token(known_bonus=known_bonus, position=prompt_rows)
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - started
            return first, elapsed

        cold_first, _ = controller_round("served-cold")
        committed = [cold_first]
        current = cold_first
        cursor = prompt_rows + 1
        while controller.pending_tokens:
            current = controller.serve_token(known_bonus=current, position=cursor)
            committed.append(current)
            cursor += 1
        assert committed == expected_committed
        controller.end_request("served-cold")
        cache_entries = int(mesh.num_program_cache_entries())
        warm_seconds = []
        for iteration in range(3):
            first, elapsed = controller_round(f"served-warm-{iteration}")
            warm_seconds.append(elapsed)
            assert first == expected_committed[0]
            assert int(mesh.num_program_cache_entries()) == cache_entries
            controller.end_request(f"served-warm-{iteration}")

        # Hardware-check the block-tail one-row path at residue 49. Refill both
        # target caches with the same 49-token real prefix in a 64-row bucket.
        padded_prompt = torch.zeros((1, 64), dtype=torch.int64)
        padded_prompt[:, :49] = ((torch.arange(49) * 37 + 11) % config.vocab_size).reshape(1, -1)
        padded_tt = device(padded_prompt.to(torch.int32), ttnn.uint32)

        def prefill_49(cache, capture_aux):
            embedded = target.embed_prefill(padded_tt)
            if capture_aux:
                return target.prefill_layers_with_dflash_aux(
                    embedded,
                    cache,
                    pt1,
                    start_pos=0,
                    valid_seq_len=49,
                    enable_experimental=True,
                )
            return target.prefill_layers(embedded, cache, pt1, start_pos=0), None

        hidden49_a, capture49 = prefill_49(target_caches[0], True)
        hidden49_b, _ = prefill_49(target_caches[1], False)
        logits49_a = target.logits_to_host(
            target.lm_head_shards_prefill(ttnn.slice(hidden49_a, [0, 48, 0], [1, 49, h]))
        ).reshape(-1)
        logits49_b = target.logits_to_host(
            target.lm_head_shards_prefill(ttnn.slice(hidden49_b, [0, 48, 0], [1, 49, h]))
        ).reshape(-1)
        fallback_known = int(torch.argmax(logits49_a))
        assert fallback_known == int(torch.argmax(logits49_b))
        fallback_logits_expected, fallback_capture_expected = target_verify(
            target_caches[1],
            [fallback_known],
            [49],
        )
        fallback_expected = int(torch.argmax(fallback_logits_expected))
        fallback_aux_expected = host_aux(fallback_capture_expected)
        active_target_cache[0] = target_caches[0]

        controller.begin_request("fallback-cold", capture49)
        fallback_got = controller.serve_target_token(known_bonus=fallback_known, position=49)
        ttnn.synchronize_device(mesh)
        fallback_aux_got = host_aux(controller.cache.target_capture())[-1:]
        assert fallback_got == fallback_expected
        controller.end_request("fallback-cold")
        fallback_cache_entries = int(mesh.num_program_cache_entries())
        fallback_warm_seconds = []
        for iteration in range(3):
            controller.begin_request(f"fallback-warm-{iteration}", capture49)
            started = time.perf_counter()
            got = controller.serve_target_token(known_bonus=fallback_known, position=49)
            ttnn.synchronize_device(mesh)
            fallback_warm_seconds.append(time.perf_counter() - started)
            assert got == fallback_expected
            assert int(mesh.num_program_cache_entries()) == fallback_cache_entries
            controller.end_request(f"fallback-warm-{iteration}")

        memory = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        memory_used = int(memory.total_bytes_allocated_per_bank) * int(memory.num_banks)
        memory_total = int(memory.total_bytes_per_bank) * int(memory.num_banks)
        context_pcc = pcc(got_context, expected_context)
        draft_hidden_pcc = pcc(got_hidden, expected_hidden[1:16])
        draft_logits_pcc = pcc(got_draft_logits, expected_draft_logits)
        verify_aux_pcc = pcc(batched_aux, sequential_aux)
        verify_logits_pcc = pcc(batched_logits, sequential_logits)
        fallback_aux_pcc = pcc(fallback_aux_got, fallback_aux_expected)
        stage_pcc = [pcc(actual, expected) for actual, expected in zip(got_layer_outputs, expected_layer_outputs)]
        stage_max_abs = [
            float((actual.float() - expected.float()).abs().max())
            for actual, expected in zip(got_layer_outputs, expected_layer_outputs)
        ]
        # The prior run disagreed at sampled row 11, which is query row 12
        # because query row zero is the unsampled anchor.
        diagnostic_sample_row = 11
        diagnostic_query_row = diagnostic_sample_row + 1
        stage_row_pcc = [
            pcc(actual[diagnostic_query_row], expected[diagnostic_query_row])
            for actual, expected in zip(got_layer_outputs, expected_layer_outputs)
        ]
        stage_row_max_abs = [
            float((actual[diagnostic_query_row].float() - expected[diagnostic_query_row].float()).abs().max())
            for actual, expected in zip(got_layer_outputs, expected_layer_outputs)
        ]
        logit_row_pcc = pcc(
            got_draft_logits[diagnostic_sample_row],
            expected_draft_logits[diagnostic_sample_row],
        )
        logit_row_max_abs = float(
            (got_draft_logits[diagnostic_sample_row].float() - expected_draft_logits[diagnostic_sample_row].float())
            .abs()
            .max()
        )
        cpu_top2_values, cpu_top2_ids = torch.topk(expected_draft_logits[diagnostic_sample_row].float(), k=2)
        tt_top2_values, tt_top2_ids = torch.topk(got_draft_logits[diagnostic_sample_row].float(), k=2)
        cpu_top2_margin = float(cpu_top2_values[0] - cpu_top2_values[1])
        tt_top2_margin = float(tt_top2_values[0] - tt_top2_values[1])
        exactness_diagnostics = (
            f"TT={draft_ids.tolist()} CPU={expected_draft_ids.tolist()} "
            f"stage_pcc={stage_pcc} stage_max_abs={stage_max_abs} "
            f"row11_stage_pcc={stage_row_pcc} row11_stage_max_abs={stage_row_max_abs} "
            f"row11_logit_pcc={logit_row_pcc:.8f} row11_logit_max_abs={logit_row_max_abs:.8f} "
            f"row11_cpu_top2={list(zip(cpu_top2_ids.tolist(), cpu_top2_values.tolist()))} "
            f"row11_cpu_margin={cpu_top2_margin:.8f} "
            f"row11_tt_top2={list(zip(tt_top2_ids.tolist(), tt_top2_values.tolist()))} "
            f"row11_tt_margin={tt_top2_margin:.8f}"
        )
        warm_median = sorted(warm_seconds)[1]
        fallback_median = sorted(fallback_warm_seconds)[1]
        print(
            "DFLASH_SERVED_FULL_TARGET "
            f"context_pcc={context_pcc:.8f} draft_hidden_pcc={draft_hidden_pcc:.8f} "
            f"draft_logits_pcc={draft_logits_pcc:.8f} "
            f"draft_argmax_equal={bool(torch.equal(draft_ids, expected_draft_ids))} "
            f"draft_accuracy_contract={draft_accuracy.passed} tied_rows={list(draft_accuracy.tied_rows)} "
            f"verify_aux_pcc={verify_aux_pcc:.8f} verify_logits_pcc={verify_logits_pcc:.8f} "
            f"verify_argmax_equal={bool(torch.equal(target_batched_ids, target_sequential_ids))} "
            f"accepted={accepted} committed={expected_committed} "
            f"warm_s={warm_seconds} median_s={warm_median:.6f} "
            f"fallback_aux_pcc={fallback_aux_pcc:.8f} fallback_token={fallback_got} "
            f"fallback_warm_s={fallback_warm_seconds} fallback_median_s={fallback_median:.6f} "
            f"program_cache_entries={cache_entries} fallback_program_cache_entries={fallback_cache_entries} "
            f"dram_used_mib={memory_used / 2**20:.1f} dram_total_mib={memory_total / 2**20:.1f} "
            f"draft_ids={draft_ids.tolist()} target_ids={target_batched_ids.tolist()} "
            f"stage_pcc={stage_pcc} stage_max_abs={stage_max_abs} "
            f"row11_stage_pcc={stage_row_pcc} row11_stage_max_abs={stage_row_max_abs} "
            f"row11_logit_pcc={logit_row_pcc:.8f} row11_logit_max_abs={logit_row_max_abs:.8f} "
            f"row11_cpu_top2_ids={cpu_top2_ids.tolist()} row11_cpu_top2_values={cpu_top2_values.tolist()} "
            f"row11_cpu_margin={cpu_top2_margin:.8f} "
            f"row11_tt_top2_ids={tt_top2_ids.tolist()} row11_tt_top2_values={tt_top2_values.tolist()} "
            f"row11_tt_margin={tt_top2_margin:.8f}"
        )
        assert context_pcc >= 0.999
        assert draft_hidden_pcc >= 0.995
        assert draft_logits_pcc >= 0.995
        assert draft_accuracy.passed, exactness_diagnostics
        assert verify_aux_pcc >= 0.995
        assert verify_logits_pcc >= 0.995
        assert torch.equal(target_batched_ids, target_sequential_ids)
        assert committed == expected_committed
        assert fallback_aux_pcc >= 0.995
        assert fallback_got == fallback_expected
    finally:
        if controller is not None:
            controller.close()
        elif proposal_cache is not None:
            proposal_cache.close()
        close_mesh(ttnn, mesh)
