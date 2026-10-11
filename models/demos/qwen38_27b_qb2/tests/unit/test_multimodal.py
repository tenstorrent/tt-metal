# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU reference checks; these do not qualify the TT vision kernels."""

import json
from types import MethodType, SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5VisionModel

from models.demos.qwen38_27b_qb2.tt.multimodal import (
    MAX_PROCESSOR_PIXELS,
    bounded_processor_kwargs,
    build_plan,
    gather_media,
    item_identity,
    modality_positions,
    validate_grid,
    validate_placeholder_budget,
    validate_processed_media,
    vision_boundaries,
)
from models.demos.qwen38_27b_qb2.tt.vision_weights import load_reference_vision, vision_state_dict

IMAGE, VIDEO = 100, 101


def config():
    return SimpleNamespace(
        image_token_id=IMAGE,
        video_token_id=VIDEO,
        vision_config=SimpleNamespace(spatial_merge_size=2),
        text_config=SimpleNamespace(hidden_size=8),
    )


@pytest.mark.parametrize(
    "ids,images,videos",
    [
        ([1, 2, 3], None, None),
        ([1] + [IMAGE] * 6 + [2, 3], [[1, 4, 6]], None),
        ([1] + [IMAGE] * 6 + [2] + [IMAGE] * 2 + [3], [[1, 4, 6], [1, 2, 4]], None),
        ([1] + [VIDEO] * 6 + [2, 3] + [VIDEO] * 6 + [4], None, [[2, 4, 6]]),
        ([1, IMAGE, 2] + [VIDEO] * 2 + [3, 4] + [VIDEO] * 2 + [5], [[1, 2, 2]], [[2, 2, 4]]),
    ],
)
def test_positions_match_pinned_hf_without_constructing_language_model(ids, images, videos):
    image_grid = None if images is None else torch.tensor(images)
    video_grid = None if videos is None else torch.tensor(videos)
    actual, delta = modality_positions(ids, image_grid, video_grid, image_token_id=IMAGE, video_token_id=VIDEO)
    oracle = SimpleNamespace(config=config())
    oracle.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, oracle)
    tokens = torch.tensor([ids])
    kinds = torch.zeros_like(tokens)
    kinds[tokens == IMAGE], kinds[tokens == VIDEO] = 1, 2
    expected, expected_delta = Qwen3_5Model.get_rope_index(
        oracle,
        tokens,
        kinds,
        image_grid_thw=image_grid,
        video_grid_thw=video_grid,
    )
    torch.testing.assert_close(actual, expected[:, 0], rtol=0, atol=0)
    assert delta == expected_delta.item()


@pytest.mark.parametrize("grid", [[], [[0, 2, 2]], [[1, 3, 2]], [[1.0, 2.0, 2.0]], [[True, True, True]], [1, 2]])
def test_rejects_invalid_grids(grid, expect_error):
    with expect_error(ValueError, "Visual"):
        validate_grid(grid)


@pytest.mark.parametrize(
    "ids,images,videos",
    [
        ([IMAGE], None, None),
        ([IMAGE] * 2, [[1, 2, 2]], None),
        ([1], [[1, 2, 2]], None),
        ([VIDEO] * 4, None, [[2, 2, 4]]),
        ([VIDEO] * 2, None, [[2, 2, 4]]),
    ],
)
def test_inconsistent_media_cannot_be_silently_ignored(ids, images, videos, expect_error):
    with expect_error(ValueError, "Visual"):
        modality_positions(ids, images, videos, image_token_id=IMAGE, video_token_id=VIDEO)


def test_frame_and_padding_windows_match_independent_attention(expect_error):
    boundaries = vision_boundaries([[2, 2, 4], [1, 2, 2]], 32)
    assert boundaries.tolist() == [0, 8, 16, 20, 32]
    torch.manual_seed(9)
    q, k, v = [torch.randn(1, 2, 32, 8) for _ in range(3)]
    # Adversarial padding cannot affect any real frame.
    k[:, :, 20:] *= 100
    v[:, :, 20:] += 100
    mask = torch.zeros(32, 32, dtype=torch.bool)
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        mask[begin:end, begin:end] = True
    actual = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    for begin, end in ((0, 8), (8, 16), (16, 20)):
        expected = torch.nn.functional.scaled_dot_product_attention(
            q[:, :, begin:end], k[:, :, begin:end], v[:, :, begin:end]
        )
        torch.testing.assert_close(actual[:, :, begin:end], expected)
    assert vision_boundaries([[1, 2, 4]], 8).tolist() == [0, 8]
    with expect_error(ValueError, "cover"):
        vision_boundaries([[1, 2, 4]], 7)


def make_plan():
    ids = [1] + [IMAGE] * 6 + [2, 3] + [VIDEO] * 2 + [4] + [VIDEO] * 2 + [5]
    identity = item_identity(
        [
            {"modality": "image", "identifier": "image-hash", "offset": 1, "length": 6},
            {"modality": "video", "identifier": "video-hash", "offset": 9, "length": 5},
        ]
    )
    calls = []

    def encode(pixels, grid):
        calls.append(grid.tolist())
        return torch.arange(int(grid.prod(-1).sum()) // 4 * 8).reshape(-1, 8).float()

    plan = build_plan(
        "request-a",
        identity,
        ids,
        (torch.zeros(24, 24), torch.tensor([[1, 4, 6]])),
        (torch.zeros(16, 24), torch.tensor([[2, 2, 4]])),
        encode,
        config=config(),
    )
    assert len(calls) == 2
    return plan


def test_chunks_across_image_video_and_text_boundaries_reconstruct_whole_plan(expect_error):
    plan = make_plan()
    whole = plan.chunk(0, len(plan.prompt_ids))
    chunks = [plan.chunk(start, min(3, len(plan.prompt_ids) - start)) for start in range(0, len(plan.prompt_ids), 3)]
    torch.testing.assert_close(torch.cat([x.vision_values for x in chunks], dim=1), whole.vision_values)
    torch.testing.assert_close(torch.cat([x.vision_mask for x in chunks], dim=1), whole.vision_mask)
    torch.testing.assert_close(torch.cat([x.rope_positions for x in chunks], dim=2), whole.rope_positions)
    assert whole.vision_mask.sum() == 10
    assert not torch.equal(plan.positions[1], torch.arange(len(plan.prompt_ids)))
    assert plan.rope_delta < 0
    with expect_error(ValueError, "outside"):
        plan.chunk(len(plan.prompt_ids) - 1, 2)


def test_request_identity_checks_prevent_slot_reuse_and_cross_media_reuse():
    plan = make_plan()
    assert plan.matches(plan.request_id, plan.item_identity, plan.prompt_ids)
    assert not plan.matches("different-request", plan.item_identity, plan.prompt_ids)
    assert not plan.matches(plan.request_id, (), plan.prompt_ids)
    assert not plan.matches(plan.request_id, plan.item_identity, plan.prompt_ids[:-1])


def test_generated_text_extension_preserves_original_media_and_decode_delta(expect_error):
    original = make_plan()
    length = len(original.prompt_ids)
    extended = original.extend_text(length + 4)
    assert extended.matches(original.request_id, original.item_identity, original.prompt_ids)
    assert extended.features is original.features
    torch.testing.assert_close(extended.positions[:, :length], original.positions)
    tail = extended.chunk(length, 4)
    assert tail.vision_mask.sum() == 0
    torch.testing.assert_close(
        tail.rope_positions, (torch.arange(length, length + 4) + original.rope_delta).expand(3, 1, 4)
    )
    assert extended.extend_text(length + 2) is extended
    assert extended.rope_delta == original.rope_delta
    with expect_error(ValueError, "cannot shorten"):
        original.extend_text(length - 1)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"max_pixels": 2**30}, {"size": {"shortest_edge": 65536, "longest_edge": 2**30}}, {"do_resize": False}],
)
def test_processor_defaults_and_overrides_cannot_raise_pixel_budget(kwargs):
    bounded = bounded_processor_kwargs(kwargs)
    assert bounded["max_pixels"] == MAX_PROCESSOR_PIXELS
    if "size" in bounded:
        assert bounded["size"]["longest_edge"] == MAX_PROCESSOR_PIXELS
    assert kwargs.get("size", {}).get("longest_edge") != MAX_PROCESSOR_PIXELS


def test_smaller_requested_processor_budget_is_preserved(expect_error):
    assert bounded_processor_kwargs({"max_pixels": 65536})["max_pixels"] == 65536
    with expect_error(ValueError, "min_pixels"):
        bounded_processor_kwargs({"min_pixels": MAX_PROCESSOR_PIXELS + 1})


@pytest.mark.parametrize(
    "outputs",
    [
        {"image_grid_thw": torch.tensor([[1, 128, 258]])},
        {"image_grid_thw": torch.tensor([[1, 64, 256]] * 3)},
        {"image_grid_thw": torch.tensor([[1, 2, 2]] * 5)},
        {"video_grid_thw": torch.tensor([[1024, 2, 2]])},
        {"pixel_values": torch.ones(1, 1)},
    ],
)
def test_actual_processor_outputs_cannot_bypass_budget(outputs, expect_error):
    with expect_error(ValueError, "budget|require a grid"):
        validate_processed_media(outputs)


def test_final_placeholder_budget_covers_cached_payloads_and_video_timestamp_masks(expect_error):
    video = SimpleNamespace(get_num_embeds=lambda: 8192)
    validate_placeholder_budget({"video": [video]})
    over = SimpleNamespace(get_num_embeds=lambda: 8193)
    with expect_error(ValueError, "budget"):
        validate_placeholder_budget({"video": [over]})
    with expect_error(ValueError, "modality/count"):
        validate_placeholder_budget({"image": [video] * 5})
    validate_processed_media({"image_grid_thw": torch.tensor([[1, 64, 128]] * 4)})


@pytest.mark.parametrize(
    "pixels,grids", [(None, [[1, 2, 2]]), ([None], [None]), ([], []), ([torch.zeros(3, 24)], [torch.tensor([1, 2, 2])])]
)
def test_missing_cached_or_mismatched_pixels_are_errors(pixels, grids, expect_error):
    with expect_error(ValueError, "Visual|Missing"):
        gather_media(pixels, grids)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_loader_reads_only_vision_tensors_and_preserves_reference_output(tmp_path, monkeypatch, dtype):
    from transformers.models.qwen3_5 import modeling_qwen3_5

    from models.demos.qwen38_27b_qb2.tt import vision_weights

    cfg = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=32,
        patch_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
    )
    original = Qwen3_5VisionModel(cfg).to(dtype).eval()
    reference_dir = tmp_path / "reference-vision-only"
    original.save_pretrained(reference_dir)
    # A blanket .to(BF16) also rounds the nonpersistent rotary frequencies.
    # HF from_pretrained recreates that buffer in FP32; compare the loader to
    # this actual checkpoint-load path, not to a differently cast reference.
    original = Qwen3_5VisionModel.from_pretrained(reference_dir, dtype=dtype).eval()
    state = {f"model.visual.{name}": tensor.contiguous() for name, tensor in original.state_dict().items()}
    state["model.language_model.fake.weight"] = torch.ones(1)
    save_file(state, tmp_path / "first.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "first.safetensors" for name in state}})
    )
    reads = []
    safe_open = vision_weights.safe_open

    class GuardedShard:
        def __init__(self, *args, **kwargs):
            self.source = safe_open(*args, **kwargs)

        def __enter__(self):
            self.source.__enter__()
            return self

        def __exit__(self, *args):
            return self.source.__exit__(*args)

        def get_tensor(self, name):
            assert name.startswith("model.visual."), "Text weights must never be materialized"
            reads.append(name)
            return self.source.get_tensor(name)

    def forbidden(*args, **kwargs):
        raise AssertionError("Full conditional/language model construction is forbidden")

    monkeypatch.setattr(vision_weights, "safe_open", GuardedShard)
    monkeypatch.setattr(modeling_qwen3_5.Qwen3_5ForConditionalGeneration, "__init__", forbidden)
    monkeypatch.setattr(modeling_qwen3_5.Qwen3_5TextModel, "__init__", forbidden)
    loaded = load_reference_vision(tmp_path, cfg)
    assert len(reads) == len(original.state_dict())
    assert all(not value.is_meta for value in (*loaded.parameters(), *loaded.buffers()))
    torch.testing.assert_close(loaded.rotary_pos_emb.inv_freq, original.rotary_pos_emb.inv_freq, rtol=0, atol=0)
    pixels = torch.randn(16, 24)
    grid = torch.tensor([[1, 4, 4]])
    with torch.inference_mode():
        torch.testing.assert_close(
            loaded(pixels, grid).pooler_output, original(pixels, grid).pooler_output, rtol=0, atol=0
        )


def test_no_vision_weights_or_escaping_shard_is_rejected(tmp_path, expect_error):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"model.language_model.x": "weights.safetensors"}}))
    with expect_error(ValueError, "no model.visual"):
        vision_state_dict(tmp_path)
    index.write_text(json.dumps({"weight_map": {"model.visual.x": "../weights.safetensors"}}))
    with expect_error(ValueError, "escapes"):
        vision_state_dict(tmp_path)
