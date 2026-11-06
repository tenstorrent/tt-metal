# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only regression tests for the serving-path review fixes of tt-metal PR #60857 (no device is opened).

* the lab diagnostics (SDPA dump / sync / keep-refs knobs, the import-time phase-timing monkeypatch) are gone;
* a sliding-tail rebuild lets a real failure propagate instead of attending without the window tail;
* ``Gemma4Generator`` keeps one ``_mock_tokens`` and initialises ``_prev_decode_batch``;
* the decode-trace guard's baselines are per instance;
* the vision helpers live once, on the shared mixin;
* the per-layer block-size resolver only hides a cache without a shape, not an unaddressable view;
* a failed boot warmup turns its path off (lane router / lane-batched walks) or refuses multimodal inputs;
* a prefix-cache hit whose SDPA-aligned restart lands inside a stripped multimodal item restores that item's soft
  tokens from the per-process cache (the 64-aligned hit vs the 128-aligned restart).

The vLLM bridge tests skip where vLLM is not importable (the tt-metal unit runners), like the lane-engine host tests.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch

from models.demos.gemma4.tt import generator as generator_mod
from models.demos.gemma4.tt.attention import prefill as prefill_mod
from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin, Gemma4Generator


@pytest.fixture(scope="module")
def bridge():
    """The vLLM bridge module (``generator_vllm`` imports vLLM at module scope)."""
    pytest.importorskip("vllm")
    from models.demos.gemma4.tt import generator_vllm

    return generator_vllm


# ── item 8: lab diagnostics removed ─────────────────────────────────────────────────────────────────────────────


def test_lab_diagnostics_are_gone():
    assert not hasattr(prefill_mod, "_SDPA_KEEP") and not hasattr(prefill_mod, "_dump_sdpa_inputs")
    src = inspect.getsource(prefill_mod._sdpa_single_chunk)
    assert "os.environ" not in src, "the single-chunk SDPA carries no env-gated diagnostics"
    assert "GEMMA4_PREFILL_PHASE_TIMING" not in inspect.getsource(generator_mod)
    assert "_g4_single_user_with_phases" not in inspect.getsource(generator_mod)


# ── item 3: a sliding-tail rebuild propagates real failures ──────────────────────────────────────────────────────


class _Cache:
    """A paged K or V pool: ``[blocks, kv_heads, block, head_dim]``."""

    def __init__(self, blocks=8, kv=2, block=64, hd=256):
        self.padded_shape = [blocks, kv, block, hd]
        self.dtype = "bf16"


def _ttnn_stub(monkeypatch, *, slice_raises=None, concat_raises=None):
    calls = {"slice": 0, "concat": 0}

    def _slice(*a, **k):
        calls["slice"] += 1
        if slice_raises is not None:
            raise slice_raises
        return SimpleNamespace(dtype="bf16", deallocate=lambda *_: None)

    def _concat(parts, dim):
        calls["concat"] += 1
        if concat_raises is not None:
            raise concat_raises
        return SimpleNamespace(dtype="bf16", deallocate=lambda *_: None)

    stub = SimpleNamespace(Tensor=type("NotATensor", (), {}), slice=_slice, concat=_concat, typecast=None)
    monkeypatch.setattr(prefill_mod, "ttnn", stub)
    return calls


def test_bounded_ring_tail_rebuild_raises_on_a_failing_slice(monkeypatch, expect_error):
    calls = _ttnn_stub(monkeypatch, slice_raises=RuntimeError("device op failed"))
    row = torch.arange(16, dtype=torch.int32)  # a 1024-position ring of 16 blocks
    with expect_error(RuntimeError, "device op failed"):
        prefill_mod._read_sliding_tail_from_bounded_ring(
            _Cache(blocks=16), _Cache(blocks=16), row, 0, 1024, 1024, 1024, 256, None, kv_local=2
        )
    assert calls["slice"] == 1


def test_bounded_ring_tail_rebuild_keeps_the_no_plan_branch(monkeypatch):
    calls = _ttnn_stub(monkeypatch, slice_raises=RuntimeError("must not be reached"))
    row = torch.arange(2, dtype=torch.int32)  # too short to hold the window: the warned no-tail path
    out = prefill_mod._read_sliding_tail_from_bounded_ring(
        _Cache(blocks=16), _Cache(blocks=16), row, 0, 1024, 1024, 1024, 256, None, kv_local=2
    )
    assert out is None and calls["slice"] == 0


def test_paged_tail_rebuild_raises_on_a_failing_concat(monkeypatch, expect_error):
    calls = _ttnn_stub(monkeypatch, concat_raises=RuntimeError("concat failed"))
    row = torch.arange(1, 33, dtype=torch.int32)  # 32 real blocks (block 0 is the null block)
    with expect_error(RuntimeError, "concat failed"):
        prefill_mod._read_sliding_tail_from_paged_cache(
            _Cache(blocks=64), _Cache(blocks=64), row, 0, 1024, 1024, 256, None, kv_local=2
        )
    assert calls["slice"] == 32 and calls["concat"] == 1
    # the legitimate None: a row too short for the window's covering blocks (a chunk start inside a block is
    # served from the covering blocks now: the admission chunks at 128 while the sliding group's blocks are 256)
    assert (
        prefill_mod._read_sliding_tail_from_paged_cache(
            _Cache(blocks=64), _Cache(blocks=64), row[:8], 0, 1024, 1024, 256, None, kv_local=2
        )
        is None
    )


# ── item 5: one _mock_tokens on the demo generator, _prev_decode_batch initialised ──────────────────────────────


def test_demo_generator_has_one_mock_tokens_and_initialises_prev_decode_batch(monkeypatch):
    src = inspect.getsource(Gemma4Generator)
    assert src.count("def _mock_tokens(") == 1
    assert "return ret\n        # Used by" not in src, "no statement after _mock_tokens's return"
    from models.tt_transformers.tt import generator as tt_generator

    monkeypatch.setattr(tt_generator.Generator, "__init__", lambda self, *a, **k: None)
    g = object.__new__(Gemma4Generator)
    Gemma4Generator.__init__(g, [], [], None)
    assert "_prev_decode_batch" in g.__dict__ and g._prev_decode_batch is None
    assert g.is_multimodal is False and g.enable_split_sampling is False


# ── item 6: the decode-trace guard's baselines are per instance ───────────────────────────────────────────────────


def test_trace_alloc_baselines_are_per_instance():
    a = object.__new__(ChunkedPrefillPageTableGuardMixin)
    b = object.__new__(ChunkedPrefillPageTableGuardMixin)
    a._g4_trace_alloc_baseline["k"] = (1, 2)  # mutated in place, as the capture paths do
    assert b._g4_trace_alloc_baseline == {} and a._g4_trace_alloc_baseline == {"k": (1, 2)}
    assert not isinstance(ChunkedPrefillPageTableGuardMixin.__dict__.get("_g4_trace_alloc_baseline"), dict)
    # assignment (the guard's release and the host tests) replaces this instance's dict only
    b._g4_trace_alloc_baseline = {"w": (3, 4)}
    assert a._g4_trace_alloc_baseline == {"k": (1, 2)} and b._g4_trace_alloc_baseline["w"] == (3, 4)


# ── item 4: the per-layer block-size resolver hides only a cache without a shape ──────────────────────────────────


def test_block_size_resolver_raises_on_an_unaddressable_view(bridge, expect_error):
    class _Cache:
        shape = [100, 1, 64, 256]  # 1 head x 256 per block: not addressable as 3 heads x 512

    class _NoShape:
        pass

    cfg = SimpleNamespace(head_dim=512, num_key_value_heads=3)
    layer = SimpleNamespace(self_attn=SimpleNamespace(config=cfg, mesh_config=None, weights=None))
    fake = SimpleNamespace(model=[SimpleNamespace(layers=[layer, layer])])
    with expect_error(ValueError, "not addressable"):
        bridge.Gemma4ForCausalLM._prefill_page_table_block_sizes(fake, [[_Cache(), _Cache()], [_Cache(), _Cache()]])
    # a cache object without a shape keeps the declared size (the TypeError / AttributeError fallback)
    cfg64 = SimpleNamespace(head_dim=256, num_key_value_heads=1)
    layer64 = SimpleNamespace(self_attn=SimpleNamespace(config=cfg64, mesh_config=None, weights=None))
    fake64 = SimpleNamespace(model=[SimpleNamespace(layers=[layer64, layer64])])
    sizes = bridge.Gemma4ForCausalLM._prefill_page_table_block_sizes(
        fake64, [[_Cache(), _Cache()], [_NoShape(), _NoShape()]]
    )
    assert sizes == [64, 64]


# ── item 7: the vision helpers live once, on the shared mixin ────────────────────────────────────────────────────


def _ttnn_torch_stub(monkeypatch):
    """``generator_mod.ttnn`` as plain torch for the gather: from_torch keeps the index, gather / reshape are torch's."""
    stub = SimpleNamespace(
        ReplicateTensorToMesh=lambda mesh: ("replicate", mesh),
        from_torch=lambda t, **k: t.to(torch.long),
        gather=lambda t, dim, index: torch.gather(t, dim, index),
        reshape=lambda t, shape: t.reshape(shape),
        uint16="u16",
        TILE_LAYOUT="tile",
        DRAM_MEMORY_CONFIG="dram",
    )
    monkeypatch.setattr(generator_mod, "ttnn", stub)


def _multimodal_stub(cls, projected, mask):
    inst = object.__new__(cls)
    inst.is_multimodal = True
    inst.mesh_device = None  # no mesh: no mapper
    inst.embed_vision = None
    calls = []

    def tower(pixel_values, image_position_ids, *rest):
        calls.append((tuple(pixel_values.shape), tuple(image_position_ids.shape)))
        return projected, mask

    tower.projects_to_text = True  # the unified embedder's shape
    tower.patch_dim = 768
    inst.vision_tower = tower
    return inst, calls


@pytest.mark.parametrize("which", ["demo", "bridge"])
def test_encode_vision_and_warmup_are_the_mixins(monkeypatch, request, which):
    cls = Gemma4Generator if which == "demo" else request.getfixturevalue("bridge").Gemma4ForCausalLM
    for name in ("encode_vision", "warmup_vision", "_replicate_mapper", "_gather_valid_soft_tokens"):
        assert getattr(cls, name) is getattr(ChunkedPrefillPageTableGuardMixin, name)
        assert name not in cls.__dict__
    _ttnn_torch_stub(monkeypatch)
    projected = torch.arange(5 * 4, dtype=torch.float32).reshape(1, 1, 5, 4)
    mask = torch.tensor([[True, False, True, True, False]])
    inst, calls = _multimodal_stub(cls, projected, mask)
    out = inst.encode_vision(torch.zeros(1, 5, 768), torch.zeros(1, 5, 2, dtype=torch.long))
    assert tuple(out.shape) == (3, 4) and torch.equal(out, projected[0, 0][mask[0]])
    inst.warmup_vision()  # no arguments: the dummy image of 280 patches at the tower's patch_dim
    assert calls[-1] == ((1, 280, 768), (1, 280, 2))


def test_bridge_keeps_no_vision_duplicates(bridge):
    cls = bridge.Gemma4ForCausalLM
    assert not hasattr(cls, "encode_vision_batch"), "no caller on the bridge; the batched demo keeps its own"
    assert "encode_vision_batch" in Gemma4Generator.__dict__
    assert "fully on-device" not in inspect.getsource(cls), "the bridge's merge is host-side"


# ── item 2: a failed boot warmup turns its path off ──────────────────────────────────────────────────────────────

LANES, LANE_SLOTS, CHUNK, BLOCKS = 4, 32, 4096, 32


def _warmup_instance(bridge, monkeypatch, *, lanes_fail_at=None, vision_fails=False, cls=None, bounded=True):
    """A bridge instance with everything ``warmup_model_prefill`` touches stubbed on the host (as the lane-engine
    host tests build it); ``lanes_fail_at`` = the 1-based lane call that raises, ``vision_fails`` = the vision warmup.
    """
    cls = cls or bridge.Gemma4ForCausalLM
    calls = {"lanes": [], "captures": [], "vision": 0}
    monkeypatch.setattr(bridge, "warmup_gemma4_model_prefill", lambda *a, **k: None)
    monkeypatch.setattr(bridge.Gemma4ForCausalLM, "_lane_prefill_group", classmethod(lambda c: True))
    monkeypatch.setattr(bridge.Gemma4ForCausalLM, "_lane_prefill_prompts", classmethod(lambda c: 8))
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_BATCHED", "1")
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_TRACE", "1")  # the opt-in frames: a one-round and an eight-round one
    monkeypatch.setenv("GEMMA4_LANE_PREFILL_TRACE_BUCKETS", "1x128,8x128")

    def vision(self):
        calls["vision"] += 1
        if vision_fails:
            raise ValueError("patch assertion")

    monkeypatch.setattr(bridge.Gemma4ForCausalLM, "warmup_vision", vision)
    inst = object.__new__(cls)
    inst._bounded_sliding_kv_cache = bounded
    inst._decode_warmup_complete = True
    inst.is_multimodal = True
    inst.model = [
        SimpleNamespace(
            mesh_config=SimpleNamespace(lane_sharded=True, lanes=LANES),
            lane_slots=LANE_SLOTS,
            mesh_device=SimpleNamespace(shape=(8, 4)),
        )
    ]
    inst.model_args = [SimpleNamespace(max_prefill_chunk_size=CHUNK, max_seq_len=8192)]
    inst._mock_tokens = lambda b, s, kv, m: {
        "tokens": torch.zeros(b, s, dtype=torch.long),
        "page_table": torch.arange(b * BLOCKS, dtype=torch.int32).reshape(b, BLOCKS),
    }

    def _lanes(tokens, page_tables, kv_cache, prompt_lens, slot_ids=None):
        calls["lanes"].append(tuple(tokens.shape))
        if lanes_fail_at is not None and len(calls["lanes"]) == lanes_fail_at:
            raise RuntimeError("lane walk failed")
        return torch.zeros(tokens.shape[0] * (tokens.shape[1] if tokens.dim() == 3 else 1), 8)

    inst.prefill_forward_lanes = _lanes
    inst._g4_capture_lane_prefill_trace = lambda bucket, kv: calls["captures"].append(tuple(bucket))
    return inst, calls


def test_failed_lane_group_warmup_turns_the_router_off(bridge, monkeypatch):
    inst, calls = _warmup_instance(bridge, monkeypatch, lanes_fail_at=1, bounded=False)
    bridge.Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert inst._g4_lane_router_off.startswith("RuntimeError: lane walk failed")
    assert calls["lanes"] == [(LANES, CHUNK)] and calls["captures"] == [], "no batched walk, no frame after it"
    assert not hasattr(inst.model[0], "_active_page_tables_per_layer")
    # the router rejects every step from here (the serial per-user path serves)
    kwargs = {"tokens": torch.zeros(4, 128, dtype=torch.long), "empty_slots": [0, 32, 64, 96], "prompt_lens": [128] * 4}
    assert (
        bridge.Gemma4ForCausalLM._try_lane_parallel_prefill(inst, kwargs, [torch.ones(4, 8, dtype=torch.int32)]) is None
    )


def test_failed_lane_batched_warmup_turns_the_batched_walks_off(bridge, monkeypatch):
    inst, calls = _warmup_instance(bridge, monkeypatch, lanes_fail_at=2, bounded=False)
    bridge.Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert not getattr(inst, "_g4_lane_router_off", None) and inst._g4_lane_batched_off
    assert calls["lanes"] == [(LANES, CHUNK), (LANES, 8, CHUNK // 8)]
    assert calls["captures"] and all(k == 1 for k, _ in calls["captures"]), "only one-round frames are captured"
    assert bridge.Gemma4ForCausalLM._g4_lane_batched_walks(inst) is False
    # the healthy path captures the k > 1 frames too
    inst, calls = _warmup_instance(bridge, monkeypatch, bounded=False)
    bridge.Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert any(k > 1 for k, _ in calls["captures"]) and bridge.Gemma4ForCausalLM._g4_lane_batched_walks(inst)


def test_failed_vision_warmup_refuses_multimodal_inputs_on_a_text_class(bridge, monkeypatch, expect_error):
    inst, calls = _warmup_instance(bridge, monkeypatch, vision_fails=True)
    bridge.Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)
    assert calls["vision"] == 1 and inst._g4_multimodal_refused == "ValueError: patch assertion"
    inst._g4_check_multimodal_admissible({"pixel_values": [None, None]})  # a text batch passes
    with expect_error(RuntimeError, "multimodal inputs refused"):
        inst._g4_check_multimodal_admissible({"pixel_values": [None, torch.zeros(2, 3)]})


def test_failed_vision_warmup_fails_the_boot_of_a_multimodal_class(bridge, monkeypatch, expect_error):
    inst, _calls = _warmup_instance(
        bridge, monkeypatch, vision_fails=True, cls=bridge.Gemma4VisionForConditionalGeneration
    )
    with expect_error(RuntimeError, "does not start without it"):
        bridge.Gemma4ForCausalLM.warmup_model_prefill(inst, [object()], enable_trace=True, can_sample_on_device=False)


# ── item 1: a prefix-cache hit whose aligned restart lands inside a stripped multimodal item ────────────────────

IMG, PAD, HIDDEN = 262144, 0, 8


def _mm_prompt(n=1200, image=(700, 260)):
    """A prompt of ``n`` text ids with one image's placeholders at ``[offset, offset + length)``."""
    ids = torch.arange(100, 100 + n, dtype=torch.long)
    ids[image[0] : image[0] + image[1]] = IMG
    return ids.view(1, -1)


def _mm_instance(bridge):
    """A bridge instance whose vision encoder is a host stub: soft token r of an image carries (image id, r)."""
    inst = object.__new__(bridge.Gemma4ForCausalLM)
    inst.is_multimodal = True
    inst.image_token_id, inst.video_token_id, inst.audio_token_id, inst.pad_token_id = IMG, None, None, PAD
    inst.embed_audio = None
    inst.model = [
        SimpleNamespace(
            hf_config=SimpleNamespace(layer_types=["sliding_attention", "full_attention"]),
            hidden_size_per_layer_input=0,
            _embed_weight_cpu=torch.randn(100 + 1200, HIDDEN),
            embed_scale=1.0,
        )
    ]
    inst._prefill_page_table_block_sizes = lambda kv: [64, 64]
    encoded = []

    def encode_vision(pixel_values, position_ids):
        n = int(position_ids.shape[1])
        soft = torch.zeros(n, HIDDEN)
        soft[:, 0] = float(pixel_values[0, 0, 0])
        soft[:, 1] = torch.arange(n)
        encoded.append(float(pixel_values[0, 0, 0]))
        return soft

    inst.encode_vision = encode_vision
    inst._vision_to_host = lambda t: t
    return inst, encoded


def _row(blocks):
    return torch.tensor([list(blocks)], dtype=torch.int32)


def _bf16(values):
    """The merged embeddings are bf16: integers above 256 round."""
    return torch.tensor(list(values), dtype=torch.float32).to(torch.bfloat16).tolist()


def _stage(inst, tokens, *, start_pos, computed, row, payload):
    """One user's staging call as prefill_forward makes it: ``start_pos`` the SDPA-aligned restart,
    ``computed`` vLLM's own count, ``payload`` the image's pixel values or None when the plugin stripped it."""
    kwargs = {
        "start_pos": [start_pos] if start_pos is not None else None,
        "page_table": row,
        "kv_cache": [object()],
        "pixel_values": [[payload]],
        "pixel_position_ids": [[torch.zeros(260, 2, dtype=torch.long) if payload is not None else None]],
        "mm_placeholder_offsets": [[700]],
        "mm_placeholder_lengths": [[260]],
    }
    users = inst._stage_multimodal_prefill(
        tokens,
        kwargs,
        [int(tokens.shape[1])],
        num_computed=[computed] if computed else None,
        page_tables_per_layer=[row, row],
    )
    assert set(kwargs) == {"start_pos", "page_table", "kv_cache", "pixel_values"} and users == int(
        kwargs["pixel_values"][0] is not None
    )
    return kwargs["pixel_values"][0]


def test_stripped_item_inside_the_aligned_restart_is_restored_from_the_cache(bridge, expect_error):
    inst, encoded = _mm_instance(bridge)
    tokens = _mm_prompt()
    image = torch.full((260, 3), 7.0)
    # the request that encoded the image: blocks 1..19 of a 64-token grid
    merged = _stage(inst, tokens, start_pos=None, computed=None, row=_row(range(1, 20)), payload=image)
    assert tuple(merged.shape) == (1200, HIDDEN) and encoded == [7.0]
    assert (merged[700:960, 0] == 7).all() and merged[700:960, 1].tolist() == _bf16(range(260))
    assert len(inst._g4_mm_soft_token_cache()) == 1
    # a prefix-cache hit of 960 tokens (64-aligned, not 128-aligned) sharing blocks 1..15: the plugin stripped the
    # image (it ends at 960), the SDPA alignment restarts the prefill at 896, and placeholders 896..959 need it
    hit = _row(list(range(1, 16)) + [40, 41, 42, 43])
    merged = _stage(inst, tokens, start_pos=896, computed=960, row=hit, payload=None)
    assert encoded == [7.0], "restored, not re-encoded"
    assert (merged[896:960, 0] == 7).all() and merged[896:960, 1].tolist() == _bf16(range(196, 260))
    # a hit on other blocks holds another prompt's image: no soft tokens for it, said so
    with expect_error(ValueError, "prefix-cache hit of 960 tokens"):
        _stage(inst, tokens, start_pos=896, computed=960, row=_row(range(100, 119)), payload=None)


def test_stripped_item_below_the_aligned_restart_needs_no_cache(bridge):
    inst, encoded = _mm_instance(bridge)
    tokens = _mm_prompt()
    # a 128-aligned hit of 1024: the image lies wholly inside the restart, its KV stays, nothing to restore; the
    # user has no soft token to merge, so the text prefill runs from 1024 on the token ids
    merged = _stage(inst, tokens, start_pos=1024, computed=1024, row=_row(range(1, 20)), payload=None)
    assert encoded == [] and merged is None


def test_prefill_forward_hands_the_computed_count_to_the_staging(bridge):
    src = inspect.getsource(bridge.Gemma4ForCausalLM.prefill_forward)
    assert "num_computed=num_computed" in src and "page_tables_per_layer=full_page_tables" in src


# ── review: payload-free restoration + restart outside a vision item (prefill_forward) ──────────


def test_restart_moves_out_of_a_vision_item_and_restores_its_rows(bridge):
    tokens = _mm_prompt()  # image at [700, 960)
    aligned = bridge.gemma4_mm_restart_outside_vision_items(
        [896, 1024, 0], tokens.repeat(3, 1), [[700]] * 3, [[260]] * 3, [IMG]
    )
    # 896 cuts the image: restart at 640 (700 aligned down); 1024 is past it; 0 stays
    assert aligned == [640, 1024, 0]
    # an audio item (another placeholder id) may be cut
    assert bridge.gemma4_mm_restart_outside_vision_items([896], tokens, [[700]], [[260]], [999]) == [896]
    # no vision ids / no metadata: unchanged
    assert bridge.gemma4_mm_restart_outside_vision_items([896], tokens, [None], [None], [IMG]) == [896]
    assert bridge.gemma4_mm_items_to_restore(
        [640, 1024, 0],
        [1200, 1200, 1200],
        [[700]] * 3,
        [[260]] * 3,
    ) == [0, 2]
    # an item placed past this call's window waits for a later chunk
    assert bridge.gemma4_mm_items_to_restore([640], [700], [[700]], [[260]]) == []


def _base_text_prefill_class(cls):
    """The class ``super().prefill_forward_text`` resolves to from the ``prefill_forward`` under test."""
    mro = list(cls.__mro__)
    definer = next(c for c in mro if "prefill_forward" in c.__dict__)
    return next(c for c in mro[mro.index(definer) + 1 :] if "prefill_forward_text" in c.__dict__)


def _prefill_forward_harness(inst, recorded):
    """Stub everything prefill_forward touches on the single-user serial path except the staging under test;
    ``recorded`` receives the kwargs the base text prefill is handed."""
    import contextlib

    inst._g4_alloc_dump = lambda *a, **k: None
    inst._g4_arm_alloc_check = lambda *a, **k: None
    inst._g4_prefill_trace_ids = lambda *a, **k: set()
    inst._prefill_user_chunk_plan = lambda tokens, kwargs: (None, None)
    inst._maybe_disable_pli_prefill_trace = lambda enable_trace, batch_size=1: False
    inst._build_per_layer_page_tables = lambda per_layer, page_table: list(per_layer)
    inst._mask_stale_prefill_page_tables = lambda tables, kwargs: (tables, kwargs)
    inst._pad_sliding_page_tables_for_bounded = lambda tables, kv_cache, row_slots=None: tables
    inst._pad_page_tables_batch_to_max = lambda tables: tables
    inst._bounded_sliding_kv_cache = False
    inst._try_lane_parallel_prefill = lambda *a, **k: None
    inst._route_per_layer_page_tables = lambda per_submesh: contextlib.nullcontext()
    inst._clear_sequential_batch_page_tables = lambda: None
    inst._g4_arm_after_prefill = lambda **k: []
    inst._g4_lane_phases = lambda *a, **k: ""
    inst.model_args = [SimpleNamespace(max_prefill_chunk_size=4096, disable_batched_prefill=False)]
    inst.data_parallel = 1
    inst.model[0].mesh_device = SimpleNamespace(shape=(1, 4))
    inst.model[0]._g4_pf_ms = {}


def test_prefill_forward_restores_a_stripped_image_the_restart_cuts(bridge, monkeypatch):
    """The reviewed defect: a 960-token hit covering an image at [700, 960) arrives with every payload
    stripped; the payload predicate is false, yet the restart (896 aligned, moved to 640 here) needs the
    image's soft tokens. Driven through prefill_forward with the heavy members stubbed."""
    inst, encoded = _mm_instance(bridge)
    tokens = _mm_prompt()
    row = _row(range(1, 20))
    # the request that encoded the image, staged directly (as prefill_forward would)
    _stage(inst, tokens, start_pos=None, computed=None, row=row, payload=torch.full((260, 3), 7.0))
    assert encoded == [7.0] and len(inst._g4_mm_soft_token_cache()) == 1

    recorded = {}

    def base_prefill_forward_text(self, **kwargs):
        recorded.update(kwargs)
        return "logits"

    monkeypatch.setattr(bridge, "resolve_gemma4_prefill_trace_enable", lambda enable, *a, **k: False)
    monkeypatch.setattr(
        _base_text_prefill_class(bridge.Gemma4ForCausalLM), "prefill_forward_text", base_prefill_forward_text
    )
    _prefill_forward_harness(inst, recorded)
    hit = _row(list(range(1, 16)) + [40, 41, 42, 43])
    kwargs = {
        "tokens": tokens,
        "prompt_lens": [1200],
        "start_pos": [960],
        "page_table": hit,
        "kv_cache": [object()],
        "pixel_values": [[None]],
        "pixel_position_ids": [[None]],
        "mm_placeholder_offsets": [[700]],
        "mm_placeholder_lengths": [[260]],
    }
    out = inst.prefill_forward(page_tables_per_layer=[hit, hit], **kwargs)
    assert out == "logits" and encoded == [7.0], "restored from the cache, not re-encoded"
    # the restart left the image whole and the merged rows carry the image's soft tokens
    assert recorded["start_pos"] == [640]
    merged = recorded["pixel_values"][0]
    assert torch.is_tensor(merged) and tuple(merged.shape) == (1200, HIDDEN)
    assert (merged[700:960, 0] == 7).all() and merged[700:960, 1].tolist() == _bf16(range(260))
    assert "mm_placeholder_offsets" not in recorded and recorded["enable_trace"] is False


def test_prefill_forward_without_metadata_or_payloads_stays_text(bridge, monkeypatch):
    inst, encoded = _mm_instance(bridge)
    recorded = {}
    monkeypatch.setattr(bridge, "resolve_gemma4_prefill_trace_enable", lambda enable, *a, **k: True)
    monkeypatch.setattr(
        _base_text_prefill_class(bridge.Gemma4ForCausalLM),
        "prefill_forward_text",
        lambda self, **kwargs: recorded.update(kwargs),
    )
    _prefill_forward_harness(inst, recorded)
    tokens = torch.arange(100, 100 + 1200).view(1, -1)
    inst.prefill_forward(
        tokens=tokens,
        prompt_lens=[1200],
        start_pos=[960],
        page_table=_row(range(1, 20)),
        kv_cache=[object()],
        pixel_values=[[None]],
        page_tables_per_layer=[_row(range(1, 20))] * 2,
    )
    assert recorded["start_pos"] == [896] and recorded["pixel_values"] == [[None]] and encoded == []
