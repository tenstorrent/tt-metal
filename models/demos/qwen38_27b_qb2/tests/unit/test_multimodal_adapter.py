# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Execute production adapter methods with explicit CPU-only device boundaries.

AST loading skips TTNN/vLLM imports on macOS. It preserves the actual class and
method bodies; these are host contract tests, not vision or device validation.
"""

import ast
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from models.demos.qwen38_27b_qb2.tt.multimodal import (
    build_plan,
    gather_media,
    item_identity,
    validate_grid,
    vision_boundaries,
)

ROOT = Path(__file__).parents[2] / "tt"
IMAGE, VIDEO = 100, 101


class BaseAdapter:
    model_capabilities = {"supports_async_decode": True, "supports_prefix_caching": False}

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        self.prefill_calls.append(kwargs)
        self._decode_bound = False
        return torch.zeros(len(prompt_lens), 1), torch.zeros(len(prompt_lens))


def adapter_class():
    tree = ast.parse((ROOT / "generator_vllm_multimodal.py").read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38ForConditionalGeneration"
    )
    cls.decorator_list = []
    namespace = dict(
        torch=torch,
        Qwen38ForCausalLM=BaseAdapter,
        SupportsMultiModal=type("SupportsMultiModal", (), {}),
        build_plan=build_plan,
        gather_media=gather_media,
        item_identity=item_identity,
    )
    exec(
        compile(ast.Module(body=[cls], type_ignores=[]), str(ROOT / "generator_vllm_multimodal.py"), "exec"), namespace
    )
    return namespace[cls.name]


@pytest.fixture
def adapter():
    obj = object.__new__(adapter_class())
    obj.batch_size, obj.context = 16, 262144
    obj.mm_config = SimpleNamespace(
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_config=SimpleNamespace(spatial_merge_size=2),
        text_config=SimpleNamespace(hidden_size=8),
    )
    obj._mm_plans, obj._mm_deltas = {}, torch.zeros(16, dtype=torch.int64)
    obj._decode_bound, obj._last_device_sampling = False, None
    obj.prefill_calls, obj.decode_calls, obj.sampling_calls, obj.encoding_calls = [], [], [], []
    obj.cache = object()
    obj._cache = lambda cache: None if cache is obj.cache else (_ for _ in ()).throw(ValueError("cache"))
    obj._table = lambda table: table

    def encode(pixels, grid):
        obj.encoding_calls.append(grid.tolist())
        return torch.ones(int(grid.prod(-1).sum()) // 4, 8)

    def sample(params, **kwargs):
        obj.sampling_calls.append(kwargs)
        return params is not None

    def decode(**kwargs):
        obj.decode_calls.append(kwargs)
        return torch.zeros(16, 1)

    obj.vision_encoder, obj._sampling = encode, sample
    obj.generator = SimpleNamespace(
        _release_traces=lambda: None,
        page_table=object(),
        remap_recurrent_slots=lambda remap: None,
        decode_forward=decode,
    )
    obj.read_decode_output = lambda output: output
    obj.process_decode_output_host = lambda output, **kwargs: output
    return obj


def request(obj, *, request_id="a", start=0, end=3, slot=7, full=None, pixels=True, identity="image-a", history=None):
    full = [1, IMAGE, IMAGE, IMAGE, IMAGE, 2] if full is None else full
    kwargs = dict(
        mm_request_ids=[request_id],
        mm_prompt_token_ids=[full],
        mm_item_spans=[[{"modality": "image", "identifier": identity, "offset": 1, "length": 4}]],
        pixel_values=[[torch.zeros(16, 24)]] if pixels else [[None]],
        image_grid_thw=[[torch.tensor([1, 4, 4])]] if pixels else [[None]],
    )
    return obj.prefill_forward(
        torch.tensor([(full if history is None else history)[:end]]),
        torch.zeros(1, 32),
        obj.cache,
        [end],
        start_pos=[start],
        empty_slots=[slot],
        **kwargs,
    )


def decode(obj, *, order=None, deltas=None, **commands):
    return obj.decode_forward(
        torch.zeros(16, 1, dtype=torch.int32),
        torch.tensor([10] * 2 + [-1] * 14),
        torch.zeros(16, 32),
        obj.cache,
        sampling_params=object(),
        slot_remap=order,
        rope_deltas_all_users=deltas,
        **commands,
    )


def test_visual_chunk_continuation_encodes_once_and_keeps_real_delta(adapter, expect_error):
    _, first = request(adapter)
    _, second = request(adapter, start=3, end=6, pixels=False)
    assert first.tolist() == second.tolist() == [-2]
    assert len(adapter.encoding_calls) == 1
    assert adapter.prefill_calls[0]["multimodal_plans"][0] is adapter.prefill_calls[1]["multimodal_plans"][0]
    assert adapter._mm_deltas[7] == -2
    with expect_error(ValueError, "No matching"):
        request(adapter, request_id="different", start=3, end=6, pixels=False)
    with expect_error(ValueError, "No matching"):
        request(adapter, start=3, end=6, pixels=False, identity="different-image")


def test_new_slot_occupant_never_reuses_old_media_encoding(adapter, expect_error):
    request(adapter)
    request(adapter, request_id="new")
    assert len(adapter.encoding_calls) == 2
    with expect_error(ValueError, "Missing multimodal"):
        request(adapter, request_id="third", pixels=False)


def test_visual_decode_row_in_prefill_and_preemption_replay_extend_as_text(adapter):
    original = [1, IMAGE, IMAGE, IMAGE, IMAGE, 2]
    history = original + [3, IMAGE, VIDEO, 4]
    request(adapter, full=original, end=6)
    request(adapter, full=original, start=6, end=10, history=history, pixels=False)
    assert len(adapter.encoding_calls) == 1
    continued = adapter._mm_plans[7]
    tail = continued.chunk(6, 4)
    assert tail.vision_mask.sum() == 0
    torch.testing.assert_close(tail.rope_positions, (torch.arange(6, 10) + continued.rope_delta).expand(3, 1, 4))
    # Preemption replays from zero, including already sampled text tokens.
    request(adapter, full=original, end=10, history=history)
    assert len(adapter.encoding_calls) == 2
    replay = adapter._mm_plans[7]
    torch.testing.assert_close(replay.positions, continued.positions)
    torch.testing.assert_close(replay.features, continued.features)


def test_direct_text_warmup_is_allowed_but_visual_payload_is_not_silently_ignored(adapter, expect_error):
    adapter.prefill_forward(torch.tensor([[1, 2, 3]]), torch.zeros(1, 32), adapter.cache, [3])
    assert "multimodal_plans" not in adapter.prefill_calls[-1]
    assert not adapter.encoding_calls
    with expect_error(ValueError, "requires complete"):
        adapter.prefill_forward(
            torch.tensor([[1, IMAGE]]), torch.zeros(1, 32), adapter.cache, [2], pixel_values=[[torch.zeros(4, 24)]]
        )
    with expect_error(ValueError, "without media identity"):
        adapter.prefill_forward(torch.tensor([[1, IMAGE]]), torch.zeros(1, 32), adapter.cache, [2])


def test_visual_payload_without_identity_cannot_fall_back_to_text(adapter, expect_error):
    with expect_error(ValueError, "Visual payload arrived without media identity"):
        adapter.prefill_forward(
            torch.tensor([[1, 2, 3]]),
            torch.zeros(1, 32),
            adapter.cache,
            [3],
            mm_request_ids=["a"],
            mm_prompt_token_ids=[[1, 2, 3]],
            mm_item_spans=[[]],
            pixel_values=[[torch.zeros(16, 24)]],
            image_grid_thw=[[torch.tensor([1, 4, 4])]],
        )
    assert adapter.prefill_calls == []
    assert adapter.encoding_calls == []


def test_scheduler_reordering_moves_media_state_and_independent_rotary_delta(adapter):
    request(adapter, slot=7)
    plan = adapter._mm_plans[7]
    order = list(range(16))
    order[0], order[7] = order[7], order[0]
    decode(adapter, order=order, deltas=[-2, 0])
    assert adapter._mm_plans[0] is plan
    assert 7 not in adapter._mm_plans
    torch.testing.assert_close(adapter.decode_calls[-1]["rope_deltas"], torch.tensor([-2, 0] + [0] * 14))
    assert adapter.decode_calls[-1]["start_pos"][0] == 10, "KV positions must not absorb rotary offsets"


def test_steady_decode_and_page_growth_do_not_reset_device_rng_or_rotary_positions(adapter):
    decode(adapter, deltas=[-2, 0])
    count = len(adapter.sampling_calls)
    for page in (False, True):
        decode(
            adapter,
            reload_inputs=False,
            reload_page_table=page,
            reload_sampling_params=False,
            reset_sampling_state=False,
        )
        assert len(adapter.sampling_calls) == count
        assert adapter.decode_calls[-1]["tokens"] is None
        assert adapter.decode_calls[-1]["start_pos"] is None
        assert adapter.decode_calls[-1]["rope_deltas"] is None


def test_rope_only_refresh_preserves_sampling_state_and_retains_delta_after_prefill(adapter, expect_error):
    decode(adapter, deltas=[-2, 0])
    before = len(adapter.sampling_calls)
    decode(adapter, deltas=[-3, 0], reload_inputs=True, reload_sampling_params=False, reset_sampling_state=False)
    assert len(adapter.sampling_calls) == before
    assert adapter.decode_calls[-1]["rope_deltas"][0] == -3
    adapter._decode_bound = False
    decode(adapter, reload_inputs=True, reload_sampling_params=False, reset_sampling_state=False)
    assert adapter.decode_calls[-1]["rope_deltas"][0] == -3
    with expect_error(ValueError, "require input reload"):
        decode(adapter, deltas=[-4, 0], reload_inputs=False, reload_sampling_params=False, reset_sampling_state=False)


def test_failed_decode_does_not_commit_slot_mapping_or_delta(adapter, expect_error):
    request(adapter, slot=7)
    original = dict(adapter._mm_plans)
    original_delta = adapter._mm_deltas.clone()

    def fail(**kwargs):
        raise RuntimeError("submission failed")

    adapter.generator.decode_forward = fail
    order = list(range(16))
    order[0], order[7] = order[7], order[0]
    with expect_error(RuntimeError, "submission failed"):
        decode(adapter, order=order, deltas=[-9, 0])
    assert adapter._mm_plans == original
    torch.testing.assert_close(adapter._mm_deltas, original_delta)


@pytest.mark.parametrize(
    "commands",
    [
        {"reload_inputs": False},
        {"reload_page_table": True},
        {"reload_inputs": "yes"},
        {"reload_sampling_params": False},
    ],
)
def test_incoherent_reload_commands_are_rejected(adapter, commands, expect_error):
    with expect_error(ValueError, "reload"):
        decode(adapter, **commands)


def test_model_prefill_splices_features_but_uses_logical_positions_for_kv():
    tree = ast.parse((ROOT / "model.py").read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen38Model")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "prefill")
    ops = SimpleNamespace(
        uint32=torch.int32,
        int32=torch.int32,
        ROW_MAJOR_LAYOUT="row",
        where=lambda mask, a, b: torch.where(mask.bool(), a, b),
        reshape=torch.reshape,
        typecast=lambda tensor, dtype: tensor.to(dtype),
    )
    namespace = dict(torch=torch, ttnn=ops)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / "model.py"), "exec"), namespace)
    recorded = {}

    def layer(x, **kwargs):
        recorded.update(kwargs)
        recorded["hidden"] = x.clone()
        return x

    def rotary(x, positions):
        recorded["rotary_positions"] = positions.clone()
        return torch.zeros(1, 4, 4), torch.ones(1, 4, 4)

    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=8),
        embed=lambda *a, **kw: torch.full((1, 4, 8), 5.0),
        upload=lambda tensor, **kwargs: tensor.clone(),
        _host_rotary=rotary,
        layers=[SimpleNamespace(kind="full_attention", prefill_forward=layer)],
        logits=lambda x, **kw: x,
    )
    chunk = SimpleNamespace(
        vision_values=torch.full((1, 4, 8), 9.0),
        vision_mask=torch.tensor([[[0], [1], [1], [0]]]),
        rope_positions=torch.tensor([[[4, 5, 5, 6]], [[4, 5, 6, 7]], [[4, 5, 6, 7]]]),
    )
    namespace["prefill"](
        model,
        torch.ones(1, 4),
        cache=SimpleNamespace(layers=[object()], batch_size=1),
        page_table=torch.zeros(1, 32),
        length=4,
        start_pos=10,
        all_logits=True,
        multimodal=chunk,
    )
    assert recorded["positions"].reshape(-1).tolist() == [10, 11, 12, 13]
    assert recorded["start_pos"] == 10
    assert recorded["hidden"][0, :, 0].tolist() == [5, 9, 9, 5]
    torch.testing.assert_close(recorded["rotary_positions"], chunk.rope_positions)


def method_from_source(filename, class_name, name, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(ROOT / filename), "exec"), namespace)
    return namespace[name]


def test_encoder_passes_actual_frame_and_padding_windows_into_every_native_block():
    seen = []
    ops = SimpleNamespace(
        from_torch=lambda value, **kwargs: value.clone(),
        bfloat16=torch.bfloat16,
        int32=torch.int32,
        TILE_LAYOUT="tile",
        ROW_MAJOR_LAYOUT="row",
        ReplicateTensorToMesh=lambda mesh: None,
        ConcatMeshToTensor=lambda *args, **kwargs: None,
        to_torch=lambda tensor, **kwargs: tensor,
    )

    def preprocess(seq_len, **kwargs):
        return None, (torch.ones(seq_len, 8), torch.zeros(seq_len, 8))

    forward = method_from_source(
        "vision.py",
        "Qwen38VisionEncoder",
        "__call__",
        dict(
            torch=torch,
            ttnn=ops,
            validate_grid=validate_grid,
            vision_boundaries=vision_boundaries,
            qwen3_5_vision_transformer_preprocess=preprocess,
            convert_rope_style_hf_to_meta=lambda a, b: (a, b),
        ),
    )

    def block(x, **kwargs):
        seen.append(kwargs["cu_window_seqlens"].tolist())
        return x

    encoder = SimpleNamespace(
        mesh=object(),
        max_patches=32768,
        config=SimpleNamespace(
            spatial_merge_size=2,
            in_channels=3,
            temporal_patch_size=2,
            patch_size=2,
            hidden_size=8,
            num_heads=1,
            out_hidden_size=8,
        ),
        reference=SimpleNamespace(
            patch_embed=lambda pixels: pixels[:, :8],
            fast_pos_embed_interpolate=lambda grid: torch.zeros(int(grid.prod(-1).sum()), 8),
        ),
        tower=SimpleNamespace(
            prepare_input=lambda patches, padded: torch.nn.functional.pad(patches, (0, 0, 0, padded - len(patches)))[
                None, None
            ],
            blocks=[block, block, block],
            patch_merger=lambda x: x.reshape(1, 1, -1, 4, 8).mean(-2),
        ),
    )
    output = forward(encoder, torch.ones(16, 24), torch.tensor([[2, 2, 4]]))
    assert seen == [[0, 8, 16, 2048]] * 3
    assert tuple(output.shape) == (4, 8)
    torch.testing.assert_close(output, torch.ones(4, 8, dtype=torch.bfloat16))


def test_generator_visual_prefill_slices_absolute_chunks_and_disables_grouping():
    captured = []
    cache = SimpleNamespace(batch_size=16, capacity=262144)
    table = torch.zeros(16, 64)
    plan = SimpleNamespace(chunk=lambda start, count: ("visual-chunk", start, count))
    model = SimpleNamespace(
        config=SimpleNamespace(vocab_size=200),
        upload=lambda tensor, **kwargs: tensor,
        prefill=lambda ids, **kwargs: captured.append(kwargs) or torch.zeros(1, 1, 8),
    )
    generator = SimpleNamespace(
        model=model,
        prefill_prepared=None,
        cache=cache,
        prefill_signatures=set(),
        page_table=table,
        batched_prefill=True,
        counters=Counter(),
        _refresh_table=lambda value: None,
        _release_traces=lambda: None,
    )
    ops = SimpleNamespace(uint32=torch.int32, ROW_MAJOR_LAYOUT="row")
    prefill = method_from_source("generator.py", "Qwen38Generator", "prefill_forward", dict(torch=torch, ttnn=ops))
    prefill(
        generator,
        torch.ones(2, 5000, dtype=torch.int64),
        page_table=table,
        kv_cache=cache,
        prompt_lens=[5000, 5000],
        slots=[3, 4],
        start_pos=[64, 64],
        multimodal_plans=[plan, None],
    )
    assert [(call["slot"], call["start_pos"], call["length"]) for call in captured] == [
        (3, 64, 4096),
        (3, 4160, 904),
        (4, 64, 4096),
        (4, 4160, 904),
    ]
    assert captured[0]["multimodal"] == ("visual-chunk", 64, 4096)
    assert captured[1]["multimodal"] == ("visual-chunk", 4160, 904)
    assert "multimodal" not in captured[2] and "multimodal" not in captured[3]


def test_generator_refreshes_rotary_and_kv_positions_independently(expect_error):
    copies = {}
    cache = SimpleNamespace(batch_size=3, capacity=100)
    ops = SimpleNamespace(execute_trace=lambda *args, **kwargs: None)
    decode = method_from_source("generator.py", "Qwen38Generator", "decode_forward", dict(torch=torch, ttnn=ops))
    generator = SimpleNamespace(
        cache=cache,
        host_sampling=False,
        reset_active_slots=False,
        active_slots=(0, 1),
        model=SimpleNamespace(context=100),
        remaining_steps=None,
        positions="kv",
        rope_indices="rope",
        trace="decode",
        sample_trace="sample",
        mesh=object(),
        trace_records_history=False,
        tokens=object(),
        counters=Counter(),
        _refresh_table=lambda table: None,
        _copy=lambda source, target, counter: copies.update({target: source.clone()}),
    )
    decode(
        generator,
        start_pos=torch.tensor([10, 20, -1]),
        page_table=object(),
        kv_cache=cache,
        rope_deltas=torch.tensor([-2, -5, 0]),
        read_from_device=False,
    )
    assert copies["kv"].tolist() == [10, 20, -1]
    assert copies["rope"].tolist() == [8, 15, 0]
    assert generator.remaining_steps == 79
    copies.clear()
    decode(generator, page_table=object(), kv_cache=cache, read_from_device=False)
    assert copies == {}, "Steady asynchronous replay must not use stale host positions"
    with expect_error(ValueError, "outside the rotary table"):
        decode(
            generator,
            start_pos=torch.tensor([10, 20, -1]),
            page_table=object(),
            kv_cache=cache,
            rope_deltas=torch.tensor([-11, 0, 0]),
            read_from_device=False,
        )
