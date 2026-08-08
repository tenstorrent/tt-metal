# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma canvas attention: mask geometry, KV phases, partial merge, device SDPA (#47462)."""

import inspect
import math
import os

import pytest
import torch

from models.experimental.diffusion_gemma.kv_phase import KVCachePhase, KVPhaseMapping, coerce_kv_cache_phase
from models.experimental.diffusion_gemma.reference.attention_mask import build_canvas_denoise_mask, canvas_positions

# Device-friendly large-negative (bf16-representable) stand-in for -inf in the mask.
NEG = -1.0e9

# The on-device tests are opt-in: they run real kernels on a Tenstorrent device and need an
# sfpi toolchain matching the LLK source (>= 7.60.0, which adds sfpi::ShiftMode); an older
# sfpi fails dispatch-kernel compile at device open. ``use_module_device`` shares ONE device:
# repeated per-test CreateDevice/teardown on QB2 (4x Blackhole) can hang an active-erisc core
# ("Timed out while waiting for active ethernet core ... to become active again"), bricking the
# board until a reset.
_requires_device = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run on a Tenstorrent device (needs sfpi >= 7.60.0)",
)
_requires_device_w2b = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run W2b SDPA spikes on a Tenstorrent device",
)
_requires_device_integration = pytest.mark.skipif(
    os.environ.get("DG_RUN_DEVICE") != "1",
    reason="set DG_RUN_DEVICE=1 to run QB2 bidirectional attention integration tests",
)

# The ttnn / Gemma4 / DiffusionGemma stack is imported behind the same gate as the tests that
# need it. Every device test here is skipped without DG_RUN_DEVICE=1, while the host-only mask,
# KV-phase and merge-math tests must stay collectable (and passing) when that stack cannot be
# imported — otherwise one import failure downstream costs this whole module instead of the
# device tests alone.
if os.environ.get("DG_RUN_DEVICE") == "1":
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm,
        Gemma4TextDecoderLayer,
        Gemma4TextRotaryEmbedding,
        Gemma4TextScaledWordEmbedding,
        apply_rotary_pos_emb,
    )

    import ttnn
    from models.common.utility_functions import comp_pcc
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tests.test_factory import (
        find_layer_idx,
        num_layers_for_full_attention_group,
        parametrize_mesh_with_fabric,
    )
    from models.demos.gemma4.tests.unit.test_model import (
        _create_hf_model,
        _create_hf_text_config,
        _hf_model_state_to_tt_state,
    )
    from models.demos.gemma4.tt.ccl import CCLManager
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
    from models.experimental.diffusion_gemma.config import DiffusionConfig
    from models.experimental.diffusion_gemma.reference.denoise_loop import denoise_block as ref_denoise_block
    from models.experimental.diffusion_gemma.reference.self_conditioning import SelfConditioning
    from models.experimental.diffusion_gemma.tests.trajectory_pcc import compare_trajectories
    from models.experimental.diffusion_gemma.tt.attention_merge import merge_attention_partials
    from models.experimental.diffusion_gemma.tt.denoise_forward import (
        DenoiseLogitsAdapter,
        denoise_attention_forward,
        denoise_hidden_forward,
        denoise_logits_from_tokens,
        embed_canvas_tokens,
        read_prompt_kv_cache_slice,
    )
    from models.experimental.diffusion_gemma.tt.denoise_loop import denoise_block
    from models.experimental.diffusion_gemma.tt.diffusion_attention import (
        _denoise_sdpa_program_config,
        _manual_gqa_attention,
        _slice_rope_cache,
    )
    from models.experimental.diffusion_gemma.tt.model import DiffusionGemma4Model
    from models.experimental.diffusion_gemma.tt.self_conditioning import TtSelfConditioning
    from tests.ttnn.utils_for_testing import assert_with_pcc


def _mesh_1x4(func):
    """(1, 4) mesh + FABRIC_1D parametrization, bound only when device runs are enabled.

    ``parametrize_mesh_with_fabric`` enumerates the system's devices while the decorator is
    applied, so binding it unconditionally would make collection of the host-only tests depend
    on device discovery.
    """
    if os.environ.get("DG_RUN_DEVICE") != "1":
        return func
    return parametrize_mesh_with_fabric([(1, 4)])(func)


# --- canvas denoise mask geometry -----------------------------------------------------------


def _attend(mask):
    """Boolean attend matrix from an additive (0 / -inf) mask."""
    return mask == 0


def test_canvas_positions_offset_by_prompt_len():
    pos = canvas_positions(prompt_len=100, canvas_len=8)
    assert torch.equal(pos, torch.arange(100, 108))


def test_denoise_mask_is_fully_bidirectional_by_default():
    mask = build_canvas_denoise_mask(prompt_len=20, canvas_len=8)
    assert mask.shape == (8, 28)
    assert torch.all(mask == 0)


def test_full_attention_layer_type_is_fully_bidirectional():
    mask = build_canvas_denoise_mask(
        prompt_len=20,
        canvas_len=8,
        layer_type="full_attention",
        sliding_window=4,
    )
    assert torch.all(mask == 0)


def test_sliding_attention_layer_type_windows_prompt_tail():
    """Sliding denoise visibility = last (W-1) COMMITTED positions + the WHOLE canvas.

    Updated 2026-07-24 (#51080). This test previously asserted a per-(q,k)
    ``abs(q_abs - k_abs) <= sliding_window`` staircase, which HF does not implement — it even
    asserted that a canvas column was masked, while HF pads the canvas region with
    unconditional True. HF's window is a cache-retention effect on the committed prefix only;
    see tests/test_reference.py for the pinned reference behaviour.
    """
    prompt_len, canvas_len, sliding_window = 10, 6, 4
    attend = _attend(
        build_canvas_denoise_mask(
            prompt_len,
            canvas_len,
            layer_type="sliding_attention",
            sliding_window=sliding_window,
        )
    )

    keep_from = prompt_len - (sliding_window - 1)  # 10 - 3 = 7
    # Evicted committed positions are hidden; retained ones are attended.
    assert not attend[0, keep_from - 1]
    assert attend[0, keep_from]
    assert attend[0, prompt_len - 1]
    # The canvas is ALWAYS fully visible, for every canvas query row.
    for row in range(canvas_len):
        assert attend[row, prompt_len:].all()
    # No query dependence: every canvas row sees exactly the same key set.
    for row in range(1, canvas_len):
        assert (attend[row] == attend[0]).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        pytest.param(
            {"prompt_len": 10, "canvas_len": 6, "layer_type": "sliding_attention"},
            "sliding_window must be positive",
            id="sliding-attention-without-window",
        ),
        pytest.param(
            {"prompt_len": 4, "canvas_len": 4, "local_window": True},
            "window_half is required",
            id="local-window-without-window-half",
        ),
    ],
)
def test_canvas_denoise_mask_rejects_incomplete_window_spec(expect_error, kwargs, message):
    with expect_error(ValueError, match=message):
        build_canvas_denoise_mask(**kwargs)


# --- non-canonical local_window mask bake (ttnn SDPA windowed-mask path only) ----------------


def test_local_window_is_symmetric_and_centered():
    prompt_len, canvas_len, w = 10, 12, 3
    mask = build_canvas_denoise_mask(prompt_len, canvas_len, local_window=True, window_half=w)
    attend = _attend(mask)
    assert mask.shape == (canvas_len, prompt_len + canvas_len)

    q_abs = canvas_positions(prompt_len, canvas_len)
    for i in range(canvas_len):
        keys = attend[i].nonzero(as_tuple=True)[0]
        lo, hi = int(keys.min()), int(keys.max())
        assert lo == max(0, int(q_abs[i]) - w)
        assert hi == min(prompt_len + canvas_len - 1, int(q_abs[i]) + w)
        assert bool(attend[i, int(q_abs[i])])  # attends to itself


def test_local_window_covers_prompt_tail_for_early_canvas():
    prompt_len, canvas_len, w = 10, 12, 3
    attend = _attend(build_canvas_denoise_mask(prompt_len, canvas_len, local_window=True, window_half=w))
    assert attend[0, 7] and attend[0, 8] and attend[0, 9]
    assert not attend[0, 6]
    deep = w + 1
    assert not attend[deep, prompt_len - 1]


def test_local_window_prompt_fully_visible_variant():
    prompt_len, canvas_len, w = 10, 12, 2
    attend = _attend(
        build_canvas_denoise_mask(prompt_len, canvas_len, local_window=True, window_half=w, prompt_fully_visible=True)
    )
    assert torch.all(attend[:, :prompt_len])
    deep = canvas_len - 1
    assert not attend[deep, prompt_len + 0]
    assert attend[deep, prompt_len + deep]
    assert attend[deep, prompt_len + deep - w]


def test_local_window_inclusive_vs_exclusive_boundary():
    prompt_len, canvas_len, w = 5, 6, 2
    inc = _attend(build_canvas_denoise_mask(prompt_len, canvas_len, local_window=True, window_half=w, inclusive=True))
    exc = _attend(build_canvas_denoise_mask(prompt_len, canvas_len, local_window=True, window_half=w, inclusive=False))
    assert exc.sum() < inc.sum()
    assert torch.all(inc | ~exc)


def test_additive_mask_values_are_zero_or_neg_inf():
    mask = build_canvas_denoise_mask(8, 8, local_window=True, window_half=2)
    vals = torch.unique(mask)
    assert torch.all((vals == 0) | torch.isinf(vals) & (vals < 0))


# --- KV cache phase coercion ----------------------------------------------------------------


def test_kv_phase_defaults_preserve_gemma4_write_paths():
    assert coerce_kv_cache_phase(None, is_decode=False) == KVCachePhase.PREFILL_WRITE
    assert coerce_kv_cache_phase(None, is_decode=True) == KVCachePhase.COMMIT_APPEND


def test_kv_phase_accepts_explicit_readonly_value():
    assert coerce_kv_cache_phase("denoise_readonly", is_decode=False) == KVCachePhase.DENOISE_READONLY


@pytest.mark.parametrize(
    ("value", "is_decode", "message"),
    [
        pytest.param(
            KVCachePhase.DENOISE_READONLY,
            True,
            "DENOISE_READONLY is a prefill-only KV phase",
            id="enum-denoise-readonly-in-decode",
        ),
        pytest.param(
            "denoise_readonly",
            True,
            "DENOISE_READONLY is a prefill-only KV phase",
            id="str-denoise-readonly-in-decode",
        ),
        pytest.param(
            KVCachePhase.PREFILL_WRITE,
            True,
            "PREFILL_WRITE is a prefill-only KV phase",
            id="enum-prefill-write-in-decode",
        ),
        pytest.param(
            "prefill_write",
            True,
            "PREFILL_WRITE is a prefill-only KV phase",
            id="str-prefill-write-in-decode",
        ),
        pytest.param(
            KVCachePhase.COMMIT_APPEND,
            False,
            "COMMIT_APPEND is a decode-only KV phase",
            id="enum-commit-append-in-prefill",
        ),
        pytest.param(
            "commit_append",
            False,
            "COMMIT_APPEND is a decode-only KV phase",
            id="str-commit-append-in-prefill",
        ),
    ],
)
def test_kv_phase_rejects_mode_mismatches(value, is_decode, message, expect_error):
    with expect_error(ValueError, match=message):
        coerce_kv_cache_phase(value, is_decode=is_decode)


def test_denoise_knobs_are_isolated_from_shared_gemma4_attention():
    """The bidirectional/prefix-KV/RoPE-offset knobs live ONLY in DiffusionGemma.

    The shared Gemma4 attention stack must stay on its stock causal signature so
    the backbone is untouched; the denoise-specific knobs belong to the
    diffusion-local ``denoise_attention`` helper instead.
    """
    # Imported here rather than at module scope: this is the only host test that reaches into
    # the ttnn-backed attention stack, and it should not be able to break the rest of the file.
    from models.demos.gemma4.tt.attention import Gemma4Attention
    from models.demos.gemma4.tt.attention.prefill import prefill_forward
    from models.demos.gemma4.tt.model import Gemma4Model
    from models.experimental.diffusion_gemma.tt.diffusion_attention import denoise_attention

    diffusion_params = {"attn_mask", "kv_hidden_states", "prefix_kv", "q_rope_offset", "kv_phase"}
    for fn in (Gemma4Model.__call__, Gemma4Attention.__call__, prefill_forward):
        shared = set(inspect.signature(fn).parameters) & diffusion_params
        assert not shared, f"{fn.__qualname__} leaked diffusion-only kwargs: {shared}"

    denoise_sig = inspect.signature(denoise_attention).parameters
    for name in ("attn_mask", "kv_hidden_states", "prefix_kv", "q_rope_offset"):
        assert name in denoise_sig


# --- KV phase logical-to-physical mapping (reference spec) -----------------------------------


def test_commit_positions_append_after_prompt():
    mapping = KVPhaseMapping(prompt_len=32, canvas_len=8, sliding_window=16)

    assert mapping.commit_positions == tuple(range(32, 40))
    assert mapping.canvas_scratch_positions == tuple(range(8))


def test_sliding_frozen_cache_keeps_only_live_window():
    mapping = KVPhaseMapping(prompt_len=20, canvas_len=4, sliding_window=8)

    assert mapping.sliding_frozen_positions == tuple(range(12, 20))
    assert mapping.sliding_frozen_slots == (4, 5, 6, 7, 0, 1, 2, 3)


def test_sliding_commit_slots_wrap_after_window_boundary():
    mapping = KVPhaseMapping(prompt_len=14, canvas_len=6, sliding_window=16)

    assert mapping.commit_positions == (14, 15, 16, 17, 18, 19)
    assert mapping.sliding_commit_slots == (14, 15, 0, 1, 2, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt_len": -1},
        {"prompt_len": 0, "canvas_len": 0},
        {"prompt_len": 0, "sliding_window": 0},
    ],
)
def test_mapping_rejects_invalid_dimensions(kwargs, expect_error):
    with expect_error(ValueError):
        KVPhaseMapping(**kwargs)


# --- flash-attention partial merge: merge math (design task T7) ------------------------------


def _group_softmax(scores_g, values_g):
    """Softmax-normalized output + flash log-sum-exp for one key group.

    Args:
        scores_g: ``[H, C, Kg]`` fp32 attention scores for this group.
        values_g: ``[H, Kg, vhd]`` fp32 values for this group.

    Returns:
        ``(out, lse)`` with ``out`` ``[H, C, vhd]`` = softmax(scores_g) @ values_g
        and ``lse`` ``[H, C, 1]`` = logsumexp(scores_g) — the exact statistic the
        ``return_lse=True`` SDPA kernel emitted (``m + log(l)``) before that extension
        was reverted.
    """
    probs = torch.softmax(scores_g, dim=-1)
    out = probs @ values_g
    lse = torch.logsumexp(scores_g, dim=-1, keepdim=True)
    return out, lse


def _torch_merge(out_a, lse_a, out_b, lse_b):
    """Torch mirror of ``merge_attention_partials`` (the exact merge formula)."""
    m = torch.maximum(lse_a, lse_b)
    wa = torch.exp(lse_a - m)
    wb = torch.exp(lse_b - m)
    denom = wa + wb
    return (out_a * wa + out_b * wb) / denom


def _build_reference(*, num_heads, canvas, prefix_len, canvas_keys, vhd, seed):
    """Build a random two-group attention problem and its exact merged golden.

    Returns ``(out_a, lse_a, out_b, lse_b, full_out)`` all fp32 torch tensors:
    per-group partials for the merge inputs plus ``full_out`` ``[H, C, vhd]`` =
    the single full-softmax attention over ``concat(group_a, group_b)``.
    """
    torch.manual_seed(seed)
    total_keys = prefix_len + canvas_keys
    scores = torch.randn(num_heads, canvas, total_keys, dtype=torch.float32)
    values = torch.randn(num_heads, total_keys, vhd, dtype=torch.float32)

    # Ground truth: one softmax over all keys.
    full_out = torch.softmax(scores, dim=-1) @ values

    # Split keys into group A (prefix) and group B (canvas), each self-normalized.
    out_a, lse_a = _group_softmax(scores[..., :prefix_len], values[:, :prefix_len, :])
    out_b, lse_b = _group_softmax(scores[..., prefix_len:], values[:, prefix_len:, :])
    return out_a, lse_a, out_b, lse_b, full_out


def test_merge_formula_matches_full_softmax():
    """Hostless check of the merge MATH: two self-normalized groups == one full softmax.

    The ``return_lse`` SDPA producer was reverted out of ``ttnn/cpp/`` on 2026-07-30, so this
    test is now the only thing exercising the merge — it builds its LSE inputs from torch and
    never needed the kernel, which is exactly why it still passes.
    """
    out_a, lse_a, out_b, lse_b, full_out = _build_reference(
        num_heads=4, canvas=16, prefix_len=48, canvas_keys=16, vhd=32, seed=47470
    )
    merged = _torch_merge(out_a, lse_a, out_b, lse_b)
    torch.testing.assert_close(merged, full_out, atol=1e-5, rtol=1e-5)


# --- real Gemma4 masked non-causal denoise integration (QB2 mesh) ----------------------------
# These ``mesh_device`` nodes MUST stay ahead of every ``device``-fixture test below:
# ``use_module_device`` holds one single-chip device open until module teardown, and fabric
# expects every device in the system to be openable when the (1, 4) mesh comes up.


def _build_tt_model(mesh_device, hf_model, hf_text_config, *, num_layers, max_seq_len):
    model_args = Gemma4ModelArgs.from_hf_config(hf_text_config)
    model_args._hf_text_config = hf_text_config
    tp = mesh_device.shape[1] if hasattr(mesh_device, "shape") else 1
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    return DiffusionGemma4Model(
        mesh_device=mesh_device,
        hf_config=model_args,
        state_dict=_hf_model_state_to_tt_state(hf_model),
        ccl_manager=CCLManager(mesh_device, num_links=1) if tp > 1 else None,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=max_seq_len,
        max_local_batch_size=1,
        num_layers=num_layers,
        create_kv_cache=True,
    )


def _mesh_mapper(mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None


def _to_torch(tt_tensor, mesh_device):
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    return ttnn.to_torch(ttnn.get_device_tensors(tt_tensor)[0]) if is_mesh else ttnn.to_torch(tt_tensor)


def _to_device(mesh_device, value):
    return ttnn.from_torch(
        value,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=_mesh_mapper(mesh_device),
    )


def _to_device_tokens(mesh_device, value):
    return ttnn.from_torch(
        value.to(torch.int32),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=_mesh_mapper(mesh_device),
    )


def _to_device_canvas_ids(mesh_device, value):
    return ttnn.from_torch(
        value.view(value.shape[0], 1, value.shape[1], 1).to(torch.int32),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint32,
        mesh_mapper=_mesh_mapper(mesh_device),
    )


def _torch_attention_reference(hf_model, hf_text_config, layer_idx, canvas_hidden, kv_hidden, mask):
    layer_type = hf_text_config.layer_types[layer_idx]
    attn = hf_model.layers[layer_idx].self_attn
    head_dim = attn.head_dim
    q_shape = (*canvas_hidden.shape[:-1], -1, head_dim)
    kv_shape = (*kv_hidden.shape[:-1], -1, head_dim)

    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    pos_ids = torch.arange(kv_hidden.shape[1]).unsqueeze(0)
    cos, sin = rope(kv_hidden, pos_ids, layer_type=layer_type)
    q_cos = cos[:, -canvas_hidden.shape[1] :, :]
    q_sin = sin[:, -canvas_hidden.shape[1] :, :]

    query = attn.q_norm(attn.q_proj(canvas_hidden).view(q_shape))
    query = apply_rotary_pos_emb(query, q_cos, q_sin, unsqueeze_dim=2).transpose(1, 2)

    key_linear = attn.k_proj(kv_hidden).view(kv_shape)
    value_linear = attn.v_proj(kv_hidden).view(kv_shape) if attn.v_proj is not None else key_linear
    key = attn.k_norm(key_linear)
    key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    value = attn.v_norm(value_linear).transpose(1, 2)

    if attn.num_key_value_groups != 1:
        key = key.repeat_interleave(attn.num_key_value_groups, dim=1)
        value = value.repeat_interleave(attn.num_key_value_groups, dim=1)
    out = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, is_causal=False, scale=1.0
    )
    out = out.transpose(1, 2).reshape(canvas_hidden.shape[0], canvas_hidden.shape[1], -1)
    return attn.o_proj(out)


def _torch_denoise_hidden_reference(hf_model, canvas_hidden, prompt_kv_hidden_by_layer, mask):
    hidden = canvas_hidden
    for layer_idx, layer in enumerate(hf_model.layers):
        residual = hidden
        normed = layer.input_layernorm(hidden)
        kv_hidden = torch.cat([prompt_kv_hidden_by_layer[layer_idx], normed], dim=1)
        hidden = _torch_attention_reference(hf_model, hf_model.config, layer_idx, normed, kv_hidden, mask)
        hidden = layer.post_attention_layernorm(hidden)
        hidden = residual + hidden

        residual = hidden
        hidden = layer.pre_feedforward_layernorm(hidden)
        hidden = layer.mlp(hidden)
        if layer.enable_moe_block:
            hidden_1 = layer.post_feedforward_layernorm_1(hidden)
            hidden_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = layer.router(hidden_flat)
            hidden_2 = layer.pre_feedforward_layernorm_2(hidden_flat)
            hidden_2 = layer.experts(hidden_2, top_k_index, top_k_weights)
            hidden_2 = hidden_2.reshape(residual.shape)
            hidden_2 = layer.post_feedforward_layernorm_2(hidden_2)
            hidden = hidden_1 + hidden_2
        hidden = layer.post_feedforward_layernorm(hidden)
        hidden = residual + hidden
        hidden = hidden * layer.layer_scalar

    return hf_model.norm(hidden)


def _torch_denoise_logits_reference(hf_model, canvas_hidden, prompt_kv_hidden_by_layer, mask):
    hidden = _torch_denoise_hidden_reference(hf_model, canvas_hidden, prompt_kv_hidden_by_layer, mask)
    logits = hf_model.lm_head(hidden)
    cap = hf_model.config.final_logit_softcapping
    if cap and cap > 0:
        logits = torch.tanh(logits / cap) * cap
    return logits


# NOTE: removed test_real_attention_prefill_accepts_all_attend_noncausal_mask --
# it exercised a full-model all-attend mask threaded through the SHARED Gemma4
# prefill op, a capability intentionally removed from the backbone. Bidirectional
# denoise now lives in the diffusion-local denoise_attention helper, validated by
# test_real_attention_denoise_mask_covers_prompt_prefix_for_layer_type below.


@_requires_device_integration
@pytest.mark.use_module_device
@_mesh_1x4
@pytest.mark.parametrize("layer_type", ["sliding_attention", "full_attention"])
def test_real_attention_denoise_mask_covers_prompt_prefix_for_layer_type(mesh_device, layer_type, reset_seeds):
    torch.manual_seed(5)
    prompt_len = 64
    canvas_len = 256
    total_len = prompt_len + canvas_len

    base_config = _create_hf_text_config(vocab_size=256, num_layers=1)
    num_layers = 1 if layer_type == "sliding_attention" else num_layers_for_full_attention_group(base_config)
    hf_text_config = _create_hf_text_config(vocab_size=256, num_layers=num_layers)
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    tt_model = _build_tt_model(mesh_device, hf_model, hf_text_config, num_layers=num_layers, max_seq_len=total_len)
    layer_idx = find_layer_idx(hf_text_config, layer_type)

    prompt_hidden = torch.randn(1, prompt_len, hf_text_config.hidden_size)
    canvas_hidden = torch.randn(1, canvas_len, hf_text_config.hidden_size)
    kv_hidden = torch.cat([prompt_hidden, canvas_hidden], dim=1)
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=False,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, total_len)
    with torch.no_grad():
        golden = _torch_attention_reference(hf_model, hf_text_config, layer_idx, canvas_hidden, kv_hidden, mask)

    tt_canvas_hidden = _to_device(mesh_device, canvas_hidden.unsqueeze(0))
    tt_prompt_hidden = _to_device(mesh_device, prompt_hidden.unsqueeze(0))
    tt_prompt_out = tt_model.layers[layer_idx].self_attn(
        tt_prompt_hidden,
        rope_mats=tt_model._get_rope_mats(layer_idx, seq_len=prompt_len),
        is_decode=False,
        keep_kv=True,
    )
    tt_prompt_out.deallocate(True)
    tt_prompt_kv = tt_model.layers[layer_idx].self_attn._last_kv
    tt_out = denoise_attention_forward(
        tt_model,
        layer_idx=layer_idx,
        prompt_kv=tt_prompt_kv,
        canvas_hidden=tt_canvas_hidden,
    )
    out = _to_torch(tt_out, mesh_device).squeeze(0)
    tt_prompt_kv[0].deallocate(True)
    tt_prompt_kv[1].deallocate(True)

    passing, message = assert_with_pcc(golden.float(), out.float(), 0.99)
    assert passing, message


@_requires_device_integration
@pytest.mark.use_module_device
@_mesh_1x4
def test_denoise_logits_forward_returns_full_canvas_logits(mesh_device, reset_seeds):
    torch.manual_seed(6)
    prompt_len = 64
    canvas_len = 256
    total_len = prompt_len + canvas_len
    vocab_size = 256

    hf_text_config = _create_hf_text_config(vocab_size=vocab_size, num_layers=1)
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    tt_model = _build_tt_model(mesh_device, hf_model, hf_text_config, num_layers=1, max_seq_len=total_len)

    canvas_tokens = torch.randint(0, vocab_size, (1, canvas_len), dtype=torch.long)
    prev_logits = torch.randn(1, canvas_len, vocab_size)
    self_conditioning_ref = SelfConditioning(
        hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
        activation=hf_text_config.hidden_activation,
    ).eval()
    self_conditioning_state = {
        "pre_norm.weight": self_conditioning_ref.pre_norm.weight.data.clone(),
        "gate_proj.weight": self_conditioning_ref.gate_proj.weight.data.clone(),
        "up_proj.weight": self_conditioning_ref.up_proj.weight.data.clone(),
        "down_proj.weight": self_conditioning_ref.down_proj.weight.data.clone(),
    }
    with torch.no_grad():
        canvas_hidden = hf_model.embed_tokens(canvas_tokens)
        conditioned_canvas_hidden = self_conditioning_ref.condition(
            canvas_hidden,
            prev_logits,
            hf_model.embed_tokens.weight,
        )
    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
    with torch.no_grad():
        prompt_hidden = hf_model.embed_tokens(prompt_tokens)
        prompt_kv_hidden = hf_model.layers[0].input_layernorm(prompt_hidden)
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=False,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, total_len)
    with torch.no_grad():
        golden = _torch_denoise_logits_reference(hf_model, conditioned_canvas_hidden, [prompt_kv_hidden], mask)
        golden_hidden = _torch_denoise_hidden_reference(
            hf_model,
            conditioned_canvas_hidden,
            [prompt_kv_hidden],
            mask,
        )

    tt_canvas_tokens = _to_device_tokens(mesh_device, canvas_tokens)
    tt_prompt_tokens = _to_device_tokens(mesh_device, prompt_tokens)
    tt_prompt_hidden = embed_canvas_tokens(tt_model, tt_prompt_tokens)
    tt_prompt_logits = tt_model(
        tt_prompt_hidden,
        is_decode=False,
        input_ids_torch=prompt_tokens,
    )
    tt_prompt_logits.deallocate(True)
    tt_prompt_kv_by_layer = [read_prompt_kv_cache_slice(tt_model.tt_kv_cache[0], prompt_len=prompt_len)]
    tt_prev_logits = _to_device(mesh_device, prev_logits.unsqueeze(0))
    tt_self_conditioning_embedding = _to_device(
        mesh_device,
        hf_model.embed_tokens.weight.detach().unsqueeze(0).unsqueeze(0),
    )
    self_conditioning = TtSelfConditioning(
        mesh_device,
        self_conditioning_state,
        hidden_size=hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
    )
    tt_logits = denoise_logits_from_tokens(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        canvas_tokens=tt_canvas_tokens,
        self_conditioning=self_conditioning,
        prev_logits=tt_prev_logits,
        self_conditioning_embedding_weight=tt_self_conditioning_embedding,
    )
    tt_canvas_hidden = embed_canvas_tokens(tt_model, tt_canvas_tokens)
    conditioned = self_conditioning.condition(
        tt_canvas_hidden,
        tt_prev_logits,
        tt_self_conditioning_embedding,
    )
    tt_canvas_hidden.deallocate(True)
    tt_hidden = denoise_hidden_forward(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        canvas_hidden=conditioned,
    )
    logits = _to_torch(tt_logits, mesh_device).squeeze(0)
    hidden = _to_torch(tt_hidden, mesh_device).squeeze(0)
    for tt_k, tt_v in tt_prompt_kv_by_layer:
        tt_k.deallocate(True)
        tt_v.deallocate(True)

    # Full logits include the shared bf16 MoE/lm_head/softcap path; this branch's
    # known full-model ceiling is below the attention-only 0.99 acceptance.
    _, hidden_pcc = comp_pcc(golden_hidden.float(), hidden.float(), pcc=0.0)
    print(
        "\n[denoise logits drift] "
        f"hidden_pcc={hidden_pcc:.5f} logits_argmax_agreement="
        f"{float((golden.argmax(dim=-1) == logits.argmax(dim=-1)).float().mean()):.4f}"
    )
    passing, message = assert_with_pcc(golden.float(), logits.float(), 0.98)
    assert passing, message


@_requires_device_integration
@pytest.mark.use_module_device
@_mesh_1x4
def test_denoise_logits_adapter_threads_prev_logits_for_self_conditioning(mesh_device, reset_seeds):
    """Device-vs-device wiring equivalence; HF-golden logits tests own numerical correctness."""
    torch.manual_seed(7)
    prompt_len = 64
    canvas_len = 256
    total_len = prompt_len + canvas_len
    vocab_size = 256

    hf_text_config = _create_hf_text_config(vocab_size=vocab_size, num_layers=1)
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    tt_model = _build_tt_model(mesh_device, hf_model, hf_text_config, num_layers=1, max_seq_len=total_len)

    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
    canvas_tokens_step0 = torch.randint(0, vocab_size, (1, canvas_len), dtype=torch.long)
    canvas_tokens_step1 = torch.randint(0, vocab_size, (1, canvas_len), dtype=torch.long)
    tt_prompt_tokens = _to_device_tokens(mesh_device, prompt_tokens)
    tt_canvas_tokens_step0 = _to_device_tokens(mesh_device, canvas_tokens_step0)
    tt_canvas_tokens_step1 = _to_device_tokens(mesh_device, canvas_tokens_step1)

    tt_prompt_hidden = embed_canvas_tokens(tt_model, tt_prompt_tokens)
    tt_prompt_logits = tt_model(
        tt_prompt_hidden,
        is_decode=False,
        input_ids_torch=prompt_tokens,
    )
    tt_prompt_logits.deallocate(True)
    tt_prompt_kv_by_layer = [read_prompt_kv_cache_slice(tt_model.tt_kv_cache[0], prompt_len=prompt_len)]

    self_conditioning_ref = SelfConditioning(
        hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
        activation=hf_text_config.hidden_activation,
    ).eval()
    self_conditioning_state = {
        "pre_norm.weight": self_conditioning_ref.pre_norm.weight.data.clone(),
        "gate_proj.weight": self_conditioning_ref.gate_proj.weight.data.clone(),
        "up_proj.weight": self_conditioning_ref.up_proj.weight.data.clone(),
        "down_proj.weight": self_conditioning_ref.down_proj.weight.data.clone(),
    }
    self_conditioning = TtSelfConditioning(
        mesh_device,
        self_conditioning_state,
        hidden_size=hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
    )
    tt_self_conditioning_embedding = _to_device(
        mesh_device,
        hf_model.embed_tokens.weight.detach().unsqueeze(0).unsqueeze(0),
    )

    adapter = DenoiseLogitsAdapter(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        self_conditioning=self_conditioning,
        self_conditioning_embedding_weight=tt_self_conditioning_embedding,
    )
    step0_logits = adapter(tt_canvas_tokens_step0, 0)
    expected_step1_logits = denoise_logits_from_tokens(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        canvas_tokens=tt_canvas_tokens_step1,
        self_conditioning=self_conditioning,
        prev_logits=step0_logits,
        self_conditioning_embedding_weight=tt_self_conditioning_embedding,
    )
    step1_logits = adapter(tt_canvas_tokens_step1, 1)

    expected = _to_torch(expected_step1_logits, mesh_device).squeeze(0)
    actual = _to_torch(step1_logits, mesh_device).squeeze(0)
    adapter.reset()
    expected_step1_logits.deallocate(True)
    for tt_k, tt_v in tt_prompt_kv_by_layer:
        tt_k.deallocate(True)
        tt_v.deallocate(True)

    passing, message = assert_with_pcc(expected.float(), actual.float(), 0.999)
    assert passing, message


@_requires_device_integration
@pytest.mark.use_module_device
@_mesh_1x4
@pytest.mark.parametrize("enable_moe", [True, False], ids=["moe", "dense"])
def test_denoise_controller_real_logits_records_decision_flips(mesh_device, reset_seeds, enable_moe):
    torch.manual_seed(8)
    prompt_len = 64
    canvas_len = 256
    total_len = prompt_len + canvas_len
    vocab_size = 256
    max_steps = 2

    hf_text_config = _create_hf_text_config(vocab_size=vocab_size, num_layers=1)
    hf_text_config.enable_moe_block = enable_moe
    if getattr(hf_text_config, "enable_moe_block", False):
        hf_text_config.num_experts = 4
        hf_text_config.top_k_experts = 2
    hf_model = _create_hf_model(hf_text_config)
    tt_model = _build_tt_model(mesh_device, hf_model, hf_text_config, num_layers=1, max_seq_len=total_len)

    prompt_tokens = torch.randint(0, vocab_size, (1, prompt_len), dtype=torch.long)
    init_canvas = torch.randint(0, vocab_size, (1, canvas_len), dtype=torch.long)
    gumbel_noise = [torch.zeros(1, canvas_len, vocab_size) for _ in range(max_steps)]
    noise_tokens = [torch.randint(0, vocab_size, (1, canvas_len), dtype=torch.long) for _ in range(max_steps)]
    cfg = DiffusionConfig(
        max_denoise_steps=max_steps,
        entropy_stop_threshold=-1.0,
        stable_steps_to_halt=1,
        entropy_budget=0.1,
    )

    self_conditioning_ref = SelfConditioning(
        hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
        activation=hf_text_config.hidden_activation,
    ).eval()
    self_conditioning_state = {
        "pre_norm.weight": self_conditioning_ref.pre_norm.weight.data.clone(),
        "gate_proj.weight": self_conditioning_ref.gate_proj.weight.data.clone(),
        "up_proj.weight": self_conditioning_ref.up_proj.weight.data.clone(),
        "down_proj.weight": self_conditioning_ref.down_proj.weight.data.clone(),
    }
    mask = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=False,
        neg_inf=NEG,
        dtype=torch.float32,
    ).view(1, 1, canvas_len, total_len)
    with torch.no_grad():
        prompt_hidden = hf_model.embed_tokens(prompt_tokens)
        prompt_kv_hidden = hf_model.layers[0].input_layernorm(prompt_hidden)
    ref_logits_by_step = []

    class TorchLogitsAdapter:
        def __init__(self):
            self.prev_logits = None

        def __call__(self, canvas, step):
            with torch.no_grad():
                canvas_hidden = hf_model.embed_tokens(canvas)
                conditioned = self_conditioning_ref.condition(
                    canvas_hidden,
                    self.prev_logits,
                    hf_model.embed_tokens.weight,
                    enabled=self.prev_logits is not None,
                )
                logits = _torch_denoise_logits_reference(hf_model, conditioned, [prompt_kv_hidden], mask)
                self.prev_logits = logits
                ref_logits_by_step.append(logits)
                return logits

    ref = ref_denoise_block(
        TorchLogitsAdapter(),
        init_canvas,
        cfg,
        vocab_size,
        gumbel_noise_fn=lambda step: gumbel_noise[step],
        noise_tokens_fn=lambda step: noise_tokens[step],
    )

    tt_prompt_tokens = _to_device_tokens(mesh_device, prompt_tokens)
    tt_prompt_hidden = embed_canvas_tokens(tt_model, tt_prompt_tokens)
    tt_prompt_logits = tt_model(
        tt_prompt_hidden,
        is_decode=False,
        input_ids_torch=prompt_tokens,
    )
    tt_prompt_logits.deallocate(True)
    tt_prompt_kv_by_layer = [read_prompt_kv_cache_slice(tt_model.tt_kv_cache[0], prompt_len=prompt_len)]
    self_conditioning = TtSelfConditioning(
        mesh_device,
        self_conditioning_state,
        hidden_size=hf_text_config.hidden_size,
        intermediate_size=hf_text_config.intermediate_size,
        eps=hf_text_config.rms_norm_eps,
    )
    tt_self_conditioning_embedding = _to_device(
        mesh_device,
        hf_model.embed_tokens.weight.detach().unsqueeze(0).unsqueeze(0),
    )
    tt_adapter_base = DenoiseLogitsAdapter(
        tt_model,
        prompt_hidden_by_layer=tt_prompt_kv_by_layer,
        self_conditioning=self_conditioning,
        self_conditioning_embedding_weight=tt_self_conditioning_embedding,
    )
    tt_logits_by_step = []

    def tt_adapter(canvas_tokens, step):
        logits = tt_adapter_base(canvas_tokens, step)
        tt_logits_by_step.append(_to_torch(logits, mesh_device).squeeze(0).float())
        return logits

    tt = denoise_block(
        tt_adapter,
        _to_device_canvas_ids(mesh_device, init_canvas),
        cfg,
        gumbel_noise_fn=lambda step: _to_device(mesh_device, gumbel_noise[step].unsqueeze(0)),
        noise_tokens_fn=lambda step: _to_device_canvas_ids(mesh_device, noise_tokens[step]),
    )

    comparison = compare_trajectories(
        ref,
        tt,
        min_argmax_agreement=0.10,
        min_sampled_agreement=0.10,
        min_accept_iou=0.0,
        min_canvas_agreement=0.98,
        min_per_step_entropy_pcc=0.60,
        max_entropy_abs_err_threshold=0.50,
        committed_match_threshold=0.10,
        entropy_pcc_threshold=0.99,
    )
    accept_flips = [int((ra.accept_mask != rb.accept_mask).sum()) for ra, rb in zip(ref.per_step, tt.per_step)]
    argmax_flips = [int((ra.argmax != rb.argmax).sum()) for ra, rb in zip(ref.per_step, tt.per_step)]
    canvas_flips = [int((ra.canvas != rb.canvas).sum()) for ra, rb in zip(ref.per_step, tt.per_step)]
    logits_pcc = [
        float(comp_pcc(ref_logits_by_step[i].float(), tt_logits_by_step[i].float(), pcc=0.0)[1])
        for i in range(max_steps)
    ]
    logits_mean_abs = [
        float((ref_logits_by_step[i].float() - tt_logits_by_step[i].float()).abs().mean()) for i in range(max_steps)
    ]
    logits_max_abs = [
        float((ref_logits_by_step[i].float() - tt_logits_by_step[i].float()).abs().max()) for i in range(max_steps)
    ]
    ref_top2_margin_mean = [
        float(torch.topk(ref_logits_by_step[i].float(), k=2, dim=-1).values.diff(dim=-1).abs().mean())
        for i in range(max_steps)
    ]
    logits_argmax_agreement = [
        float((ref_logits_by_step[i].argmax(dim=-1) == tt_logits_by_step[i].argmax(dim=-1)).float().mean())
        for i in range(max_steps)
    ]
    logits_top8_contains_ref_argmax = [
        float(
            (tt_logits_by_step[i].topk(k=8, dim=-1).indices == ref_logits_by_step[i].argmax(dim=-1, keepdim=True))
            .any(dim=-1)
            .float()
            .mean()
        )
        for i in range(max_steps)
    ]
    print(
        "\n[real-logits trajectory] "
        f"mode={'moe' if enable_moe else 'dense'} "
        f"accept_flips={accept_flips} argmax_flips={argmax_flips} canvas_flips={canvas_flips} "
        f"entropy_pcc={comparison.per_step_entropy_pcc} "
        f"logits_pcc={logits_pcc} logits_argmax_agreement={logits_argmax_agreement} "
        f"logits_top8_contains_ref_argmax={logits_top8_contains_ref_argmax} "
        f"logits_mean_abs={logits_mean_abs} logits_max_abs={logits_max_abs} "
        f"ref_top2_margin_mean={ref_top2_margin_mean}"
    )

    tt_adapter_base.reset()
    for tt_k, tt_v in tt_prompt_kv_by_layer:
        tt_k.deallocate(True)
        tt_v.deallocate(True)

    assert comparison.steps_match and comparison.halted_match
    assert comparison.passed, comparison
    assert ref.num_steps == tt.num_steps == max_steps
    assert not ref.halted and not tt.halted
    assert len(accept_flips) == max_steps
    # This remains a diagnostic for the known bf16 decision-bar blocker, but it
    # should still fail loudly if the real-logits path stops resembling torch.
    min_logits_pcc = 0.96 if enable_moe else 0.975
    max_total_accept_flips = 2 if enable_moe else 4
    assert min(logits_pcc) >= min_logits_pcc
    assert min(logits_top8_contains_ref_argmax) >= 0.80
    assert sum(accept_flips) <= max_total_accept_flips


# --- flash-attention partial merge on device (design task T7) --------------------------------


@_requires_device
@pytest.mark.use_module_device
def test_device_merge_partials_matches_torch(device):
    """Real ttnn merge vs the torch golden at a bf16-appropriate PCC.

    bf16 partial outputs (activation dtype) + fp32 lse, so the merged result
    carries only bf16 rescale drift; PCC 0.99 mirrors the sibling on-device SDPA
    tests. Bitwise agreement is NOT expected (gated on decision-agreement per the
    design doc)."""
    out_a, lse_a, out_b, lse_b, full_out = _build_reference(
        num_heads=8, canvas=64, prefix_len=256, canvas_keys=64, vhd=128, seed=47471
    )

    # Merge inputs: [1, H, C, vhd] bf16 outputs + [1, H, C, 1] fp32 lse.
    tt_out_a = ttnn.from_torch(out_a.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_out_b = ttnn.from_torch(out_b.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_lse_a = ttnn.from_torch(lse_a.unsqueeze(0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    tt_lse_b = ttnn.from_torch(lse_b.unsqueeze(0), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    tt_merged = merge_attention_partials(tt_out_a, tt_lse_a, tt_out_b, tt_lse_b)
    merged = ttnn.to_torch(tt_merged)[0]  # drop the batch dim -> [H, C, vhd]

    assert_with_pcc(full_out, merged, 0.99)
    tt_merged.deallocate(True)


# --- on-device bidirectional canvas SDPA (#47462, net-new N1) --------------------------------
# Canvas queries [b, nh, C, d] attend to the prefix-concat keys [b, nkv, prompt_len+C, d]
# through the BAKED canvas mask from ``reference/attention_mask.py``, with ``is_causal=False``.
# ttnn SDPA makes ``sliding_window_size`` and ``attn_mask`` mutually exclusive, so any window
# has to live in the mask. PCC vs a torch SDPA golden using the same mask. Uses the ``device``
# fixture, so CPU-only envs skip. HF unused — checkpoint-free:
#   DG_RUN_DEVICE=1 pytest models/experimental/diffusion_gemma/tests/test_attention.py \
#     -k "canvas_bidirectional or gqa_bidirectional or sdpa_local_window or staged_gqa"


def _run_canvas_sdpa(
    device,
    *,
    local_window=False,
    window_half=None,
    batch=1,
    num_heads=8,
    num_kv_heads=8,
    prompt_len=256,
    canvas_len=256,
    head_dim=256,
    pcc=0.99,
):
    torch.manual_seed(1234)
    seq_k = prompt_len + canvas_len

    q = torch.randn(batch, num_heads, canvas_len, head_dim)
    k = torch.randn(batch, num_kv_heads, seq_k, head_dim)
    v = torch.randn(batch, num_kv_heads, seq_k, head_dim)

    mask2d = build_canvas_denoise_mask(
        prompt_len,
        canvas_len,
        local_window=local_window,
        window_half=window_half,
        neg_inf=NEG,
        dtype=torch.float32,
    )  # [canvas_len, seq_k]
    mask = mask2d.view(1, 1, canvas_len, seq_k)  # broadcast over batch + heads

    # torch golden (fp32), expanding KV for GQA if needed
    if num_kv_heads != num_heads:
        k_ref = k.repeat_interleave(num_heads // num_kv_heads, dim=1)
        v_ref = v.repeat_interleave(num_heads // num_kv_heads, dim=1)
    else:
        k_ref, v_ref = k, v
    golden = torch.nn.functional.scaled_dot_product_attention(q, k_ref, v_ref, attn_mask=mask, is_causal=False)

    tt_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    tt_out = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        attn_mask=tt_mask,
        is_causal=False,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(tt_out)[:, :, :canvas_len, :]
    assert_with_pcc(golden, out, pcc)


@_requires_device
@pytest.mark.use_module_device
def test_canvas_bidirectional(device):
    # CANONICAL denoise geometry: canvas fully sees prompt + canvas (bidirectional)
    # for every layer type — the decoder is fully bidirectional (modeling:1399-1438).
    _run_canvas_sdpa(device)


@_requires_device
@pytest.mark.use_module_device
def test_gqa_bidirectional(device):
    # GQA shape matching the model (16 query / 8 KV heads), canonical full visibility.
    _run_canvas_sdpa(device, num_heads=16, num_kv_heads=8)


@_requires_device
@pytest.mark.use_module_device
def test_staged_gqa_fallback_matches_torch(device):
    """Validate the maskless fallback math that runs after the QB2 SDPA L1 clash."""
    torch.manual_seed(47464)
    q = torch.randn(1, 4, 32, 32)
    k = torch.randn(1, 2, 64, 32)
    v = torch.randn(1, 2, 64, 32)
    golden = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(2, dim=1),
        v.repeat_interleave(2, dim=1),
        is_causal=False,
        scale=1.0,
    )

    tt_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = _manual_gqa_attention(tt_q, tt_k, tt_v)
    out = ttnn.to_torch(tt_out)
    assert_with_pcc(golden, out, 0.99)
    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)
    tt_out.deallocate(True)


@_requires_device
@pytest.mark.use_module_device
def test_sdpa_local_window_op(device):
    # op-capability only: symmetric 2W+1 window baked into the mask. The real decoder
    # does NOT window visibility — see build_canvas_denoise_mask docstring.
    _run_canvas_sdpa(device, local_window=True, window_half=64)


# --- W2b long-prompt non-causal SDPA spikes (#47462) ----------------------------------------
# The S1/S2 harness from ``plan.md`` Appendix A (W2b): canvas queries [1, H, 256, DH] attend
# bidirectionally to a long [1, Hkv, Sk, DH] prefix+canvas K/V rectangle. S1 runs the target
# maskless non-causal path; S2 keeps an explicit all-zero mask as the A/B control. The long
# cases are expensive and intentionally opt-in:
#   DG_RUN_DEVICE=1 DG_W2B_SDPA_SWEEP=full pytest .../test_attention.py

CANVAS_LEN = 256
ORACLE_K_CHUNK = 2048
S_SWEEP = (8192, 32768, 33000, 65536, 131072, 262144)
HEAD_DIM_SWEEP = (256, 512)
SMOKE_CASES = {
    (8192, 256, False),
    (33000, 256, False),
    (8192, 256, True),
}


def _requires_full_sweep():
    return pytest.mark.skipif(
        os.environ.get("DG_W2B_SDPA_SWEEP") != "full",
        reason="set DG_W2B_SDPA_SWEEP=full to run the expensive W2b acceptance sweep",
    )


def _w2b_sweep_params():
    params = []
    for masked in (False, True):
        spike = "s2-masked" if masked else "s1-maskless"
        for head_dim in HEAD_DIM_SWEEP:
            for sk in S_SWEEP:
                marks = () if (sk, head_dim, masked) in SMOKE_CASES else (_requires_full_sweep(),)
                params.append(pytest.param(sk, head_dim, masked, marks=marks, id=f"{spike}-sk{sk}-d{head_dim}"))
    return params


def _torch_online_sdpa(q, k, v, *, k_chunk=ORACLE_K_CHUNK):
    """Memory-bounded fp32 all-attend oracle for very long K sequences."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    repeat = q.shape[1] // k.shape[1]
    running_max = torch.full(q.shape[:-1], -torch.inf, dtype=torch.float32)
    running_sum = torch.zeros(q.shape[:-1], dtype=torch.float32)
    running_out = torch.zeros_like(q, dtype=torch.float32)

    q = q.float()
    for start in range(0, k.shape[-2], k_chunk):
        k_chunk_t = k[:, :, start : start + k_chunk, :].float()
        v_chunk_t = v[:, :, start : start + k_chunk, :].float()
        if repeat != 1:
            k_chunk_t = k_chunk_t.repeat_interleave(repeat, dim=1)
            v_chunk_t = v_chunk_t.repeat_interleave(repeat, dim=1)

        scores = torch.einsum("bhqd,bhkd->bhqk", q, k_chunk_t) * scale
        chunk_max = torch.max(scores, dim=-1).values
        new_max = torch.maximum(running_max, chunk_max)
        old_scale = torch.exp(running_max - new_max)
        exp_scores = torch.exp(scores - new_max.unsqueeze(-1))
        chunk_sum = torch.sum(exp_scores, dim=-1)
        chunk_out = torch.einsum("bhqk,bhkd->bhqd", exp_scores, v_chunk_t)

        running_out = running_out * old_scale.unsqueeze(-1) + chunk_out
        running_sum = running_sum * old_scale + chunk_sum
        running_max = new_max

    return running_out / running_sum.unsqueeze(-1)


class _TinyGemma4Text(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            embed_scale=config.hidden_size**0.5,
        )
        self.layers = torch.nn.ModuleList([Gemma4TextDecoderLayer(config, layer_idx=0)])
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)


def _tiny_attention_config(layer_type):
    layer_types = ["sliding_attention", "full_attention"] if layer_type == "sliding_attention" else ["full_attention"]
    config = Gemma4TextConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=4,
        num_global_key_value_heads=4,
        head_dim=32,
        global_head_dim=32,
        layer_types=layer_types,
        sliding_window=1024,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        hidden_activation="gelu_pytorch_tanh",
        attention_bias=False,
        attention_k_eq_v=False,
        enable_moe_block=False,
        hidden_size_per_layer_input=0,
        final_logit_softcapping=0.0,
        rope_parameters={
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
            },
            "full_attention": {
                "rope_type": "default",
                "rope_theta": 1000000.0,
            },
        },
    )
    config._attn_implementation = "eager"
    return config


def _to_tt_state(hf_model):
    return {f"model.{key}": value for key, value in hf_model.state_dict().items()}


def _to_device_hidden(device, value):
    return ttnn.from_torch(
        value.unsqueeze(0),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=_mesh_mapper(device),
    )


def _to_torch_hidden(device, value):
    is_mesh = hasattr(device, "shape") and device.get_num_devices() > 1
    return ttnn.to_torch(ttnn.get_device_tensors(value)[0]) if is_mesh else ttnn.to_torch(value)


def _torch_tiny_denoise_attention_reference(hf_model, layer_idx, prompt_hidden, canvas_hidden):
    config = hf_model.config
    layer_type = config.layer_types[layer_idx]
    attn = hf_model.layers[layer_idx].self_attn
    kv_hidden = torch.cat([prompt_hidden, canvas_hidden], dim=1)
    total_len = kv_hidden.shape[1]
    canvas_len = canvas_hidden.shape[1]
    rope = Gemma4TextRotaryEmbedding(config)
    cos, sin = rope(kv_hidden, torch.arange(total_len).unsqueeze(0), layer_type=layer_type)
    q_cos = cos[:, -canvas_len:, :]
    q_sin = sin[:, -canvas_len:, :]

    q_heads = attn.q_proj.out_features // attn.head_dim
    kv_heads = attn.k_proj.out_features // attn.head_dim
    q_shape = (*canvas_hidden.shape[:-1], q_heads, attn.head_dim)
    kv_shape = (*kv_hidden.shape[:-1], kv_heads, attn.head_dim)
    query = attn.q_norm(attn.q_proj(canvas_hidden).view(q_shape))
    query = apply_rotary_pos_emb(query, q_cos, q_sin, unsqueeze_dim=2).transpose(1, 2)
    key = attn.k_norm(attn.k_proj(kv_hidden).view(kv_shape))
    key = apply_rotary_pos_emb(key, cos, sin, unsqueeze_dim=2).transpose(1, 2)
    value = attn.v_norm(attn.v_proj(kv_hidden).view(kv_shape)).transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(query, key, value, is_causal=False, scale=1.0)
    out = out.transpose(1, 2).reshape(canvas_hidden.shape[0], canvas_len, config.hidden_size)
    return attn.o_proj(out)


def _run_long_noncausal_sdpa(device, *, sk, head_dim, masked, pcc=0.99):
    torch.manual_seed(47462 + sk + head_dim + int(masked))
    q = torch.randn(1, 1, CANVAS_LEN, head_dim)
    k = torch.randn(1, 1, sk, head_dim)
    v = torch.randn(1, 1, sk, head_dim)
    golden = _torch_online_sdpa(q, k, v)

    tt_q = ttnn.from_torch(q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_k = ttnn.from_torch(k, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_v = ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_mask = None
    if masked:
        mask = torch.zeros(1, 1, CANVAS_LEN, sk, dtype=torch.float32)
        tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tt_out = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k,
        tt_v,
        attn_mask=tt_mask,
        is_causal=False,
        program_config=_denoise_sdpa_program_config(head_dim, CANVAS_LEN, sk),
        compute_kernel_config=compute_kernel_config,
    )
    out = ttnn.to_torch(tt_out)[:, :, :CANVAS_LEN, :]
    assert_with_pcc(golden, out, pcc)

    tt_q.deallocate(True)
    tt_k.deallocate(True)
    tt_v.deallocate(True)
    tt_out.deallocate(True)
    if tt_mask is not None:
        tt_mask.deallocate(True)


@_requires_device_w2b
@pytest.mark.use_module_device
@pytest.mark.parametrize(
    ("sk", "head_dim", "masked"),
    _w2b_sweep_params(),
)
def test_w2b_long_prompt_noncausal_sdpa(device, sk, head_dim, masked):
    _run_long_noncausal_sdpa(device, sk=sk, head_dim=head_dim, masked=masked)


@_requires_device_w2b
@pytest.mark.use_module_device
def test_w2b_rope_slice_reaches_256k(device):
    cache_len = 262144
    canvas_len = 256
    prompt_len = cache_len - canvas_len
    cache = ttnn.from_torch(
        torch.zeros(1, 1, cache_len, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    sliced = _slice_rope_cache(cache, prompt_len, canvas_len)
    assert sliced.shape[-2] == canvas_len
    sliced.deallocate(True)
    with pytest.raises(ValueError, match="exceeds cache length"):  # allow-pytest.raises: device test, no fixture args
        _slice_rope_cache(cache, cache_len, 32)
    cache.deallocate(True)


@_requires_device_w2b
@pytest.mark.use_module_device
@pytest.mark.parametrize(
    ("layer_type", "prompt_len"),
    [
        pytest.param("full_attention", 33024, id="full-sk33280"),
        pytest.param("sliding_attention", 33024, id="sliding-sk33280"),
        pytest.param("full_attention", 261888, marks=_requires_full_sweep(), id="full-sk262144"),
        pytest.param("sliding_attention", 261888, marks=_requires_full_sweep(), id="sliding-sk262144"),
    ],
)
def test_w2b_integrated_long_prompt_denoise_attention(device, layer_type, prompt_len):
    torch.manual_seed(47462)
    canvas_len = CANVAS_LEN
    total_len = prompt_len + canvas_len
    config = _tiny_attention_config(layer_type)
    layer_idx = 0
    hf_model = _TinyGemma4Text(config).eval()
    model_args = Gemma4ModelArgs.from_hf_config(config)
    model_args._hf_text_config = config
    tp = device.shape[1] if hasattr(device, "shape") else 1
    mesh_config = MeshConfig(device.shape, decode=ModeConfig(tp=tp))
    tt_model = DiffusionGemma4Model(
        mesh_device=device,
        hf_config=model_args,
        state_dict=_to_tt_state(hf_model),
        ccl_manager=CCLManager(device, num_links=1) if tp > 1 else None,
        dtype=ttnn.bfloat16,
        tensor_cache_path=None,
        mesh_config=mesh_config,
        max_seq_len=total_len,
        max_local_batch_size=1,
        num_layers=layer_idx + 1,
        create_kv_cache=False,
    )

    prompt_hidden = torch.randn(1, prompt_len, config.hidden_size)
    canvas_hidden = torch.randn(1, canvas_len, config.hidden_size)
    with torch.no_grad():
        golden = _torch_tiny_denoise_attention_reference(hf_model, layer_idx, prompt_hidden, canvas_hidden)

    tt_prompt_hidden = _to_device_hidden(device, prompt_hidden)
    tt_canvas_hidden = _to_device_hidden(device, canvas_hidden)
    tt_out = denoise_attention_forward(
        tt_model,
        layer_idx=layer_idx,
        prompt_hidden=tt_prompt_hidden,
        canvas_hidden=tt_canvas_hidden,
    )
    out = _to_torch_hidden(device, tt_out).squeeze(0)
    assert_with_pcc(golden, out, 0.99)

    tt_prompt_hidden.deallocate(True)
    tt_canvas_hidden.deallocate(True)
    tt_out.deallocate(True)
