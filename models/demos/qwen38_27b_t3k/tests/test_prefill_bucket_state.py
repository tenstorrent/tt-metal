# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A prompt rounded up to its pages must leave exactly what the unrounded prompt left, on device.

Rounding is what stops the stack being compiled per prompt length, and it is only sound because
the 48 recurrent layers can be told to treat the rounding as identity steps. That claim is the
whole risk: get the mask wrong and 48 of 64 layers silently fold padding into state that no
content hash stands for, which reads as a slightly worse answer rather than a failure.

Three arms. The rounded prompt against the exact one is the equality under test. Declaring the
padding real is the control: without it an inert mask and a correct mask look alike, because both
leave the exact arm untouched.

Two layers, one of each kind, so a rounding that corrupted the attention path would also show.
"""

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_t3k.tt.decoder_tp import native_mesh_shape
from models.demos.qwen38_27b_t3k.tt.generator import build_generator, configure_fabric

VALID = 100
BUCKET = 128
LAYERS = [0, 3]  # layer_types alternates every fourth index: 0 is linear_attention, 3 full


@pytest.fixture(scope="module")
def harness():
    from pathlib import Path

    root = Path(__file__).parents[1]
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*native_mesh_shape()), trace_region_size=200000000)
    try:
        gen = build_generator(root, mesh, precision_config=root / "config/precision.json", layer_indices=LAYERS)
        cache = gen.model.allocate_cache(batch_size=1, capacity=1024)
        table = torch.arange(cache.num_pages, dtype=torch.int32).reshape(1, -1)
        gen.bind_cache(cache, table)
        yield gen, mesh, cache, table
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


def _state(gen, mesh, cache):
    """Recurrent and convolution state of slot 0, as host tensors."""
    out = []
    for layer, state in zip(gen.model.layers, cache.layers):
        if layer.kind != "linear_attention":
            continue
        for name in ("recurrent", "conv"):
            tensor = getattr(state, name)
            out.append(ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float().clone())
    assert out, "the harness must load a linear_attention layer or this test measures nothing"
    return out


def _run(gen, mesh, cache, table, tokens, *, length, valid):
    gen.reset_recurrent_slots([0])
    ids = gen.model.upload(tokens[:, :length].int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    logits = gen.model.prefill(ids, cache=cache, page_table=gen.page_table, length=length, valid=valid)
    whole = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
    return whole.reshape(-1).clone(), _state(gen, mesh, cache)


def _gap(left, right):
    return max(float((a - b).abs().max()) for a, b in zip(left, right))


def test_a_rounded_prompt_leaves_the_state_the_exact_prompt_leaves(harness):
    gen, mesh, cache, table = harness
    prompt = torch.randint(1000, 5000, (1, BUCKET), dtype=torch.int32)

    exact_logits, exact_state = _run(gen, mesh, cache, table, prompt, length=VALID, valid=VALID)
    rounded_logits, rounded_state = _run(gen, mesh, cache, table, prompt, length=BUCKET, valid=VALID)
    # Control: the same rounded tokens with the padding declared real. If this matches too, the
    # mask is inert and the equality above is meaningless.
    absorbed_logits, absorbed_state = _run(gen, mesh, cache, table, prompt, length=BUCKET, valid=BUCKET)

    control = _gap(absorbed_state, exact_state)
    assert control > 0.0, "absorbing 28 padding tokens left the state identical; this test has no teeth"

    drift = _gap(rounded_state, exact_state)
    assert drift == 0.0, f"rounded state differs from exact by {drift}, against {control} when padding is absorbed"
    assert torch.equal(rounded_logits, exact_logits), (
        f"rounded logits differ from exact by {float((rounded_logits - exact_logits).abs().max())}, "
        f"against {float((absorbed_logits - exact_logits).abs().max())} when padding is absorbed"
    )


def test_rounding_reports_the_prompt_length_not_the_bucket(harness):
    gen, _, cache, table = harness
    prompt = torch.randint(1000, 5000, (1, BUCKET), dtype=torch.int32)
    gen.reset_recurrent_slots([0])
    gen.prefill_forward(
        prompt[:, :VALID], page_table=table, kv_cache=cache, prompt_lens=[VALID], start_pos=[0], slots=[0]
    )
    # Decode continues from here, and the padding is not part of the prefix.
    assert gen._slot_prefix_len[0] == VALID


def test_a_rounded_prefill_and_its_continuation_match_one_unrounded_prompt(harness):
    gen, mesh, cache, table = harness
    prompt = torch.randint(1000, 5000, (1, VALID + 64), dtype=torch.int32)

    gen.reset_recurrent_slots([0])
    gen.prefill_forward(prompt, page_table=table, kv_cache=cache, prompt_lens=[VALID + 64], start_pos=[0], slots=[0])
    whole = _state(gen, mesh, cache)

    # The first chunk rounds 100 to 128, so the continuation has to start at 100 regardless.
    gen.reset_recurrent_slots([0])
    gen.prefill_forward(
        prompt[:, :VALID], page_table=table, kv_cache=cache, prompt_lens=[VALID], start_pos=[0], slots=[0]
    )
    gen.prefill_forward(
        prompt[:, VALID:], page_table=table, kv_cache=cache, prompt_lens=[64], start_pos=[VALID], slots=[0]
    )
    split = _state(gen, mesh, cache)

    assert gen._slot_prefix_len[0] == VALID + 64
    drift = _gap(split, whole)
    scale = max(float(a.abs().max()) for a in whole)
    assert drift / scale < 0.02, f"continuing after a rounded chunk drifted {drift} against a scale of {scale}"


def test_two_prompt_lengths_in_one_bucket_share_one_captured_prefill(harness):
    """The point of the selectors: a second prompt length reaches the first one's trace."""
    gen, mesh, cache, table = harness
    prompt = torch.randint(1000, 5000, (1, BUCKET), dtype=torch.int32)
    served = dict(page_table=table, kv_cache=cache, slots=[0], start_pos=[0])

    gen.reset_recurrent_slots([0])
    gen.serving_prefill_tokens(prompt[:, :VALID], prompt_lens=[VALID], **served)
    # Capturing is what a decode does, so the second prompt has something to replay.
    gen.decode_forward(tokens=[1], start_pos=[VALID], page_table=table, kv_cache=cache)
    before = dict(gen.counters)

    longer = VALID + 13
    gen.reset_recurrent_slots([0])
    gen.serving_prefill_tokens(prompt[:, :longer], prompt_lens=[longer], **served)
    replayed = _state(gen, mesh, cache)

    assert gen.counters["prefill_replays"] > before.get("prefill_replays", 0), (
        "the second prompt length did not reach the captured trace: " f"{dict(gen.counters)} against {before}"
    )
    assert gen.counters["prefill_eager_calls"] == before["prefill_eager_calls"]

    gen.reset_recurrent_slots([0])
    exact = _run(gen, mesh, cache, table, prompt, length=longer, valid=longer)[1]
    drift = _gap(replayed, exact)
    scale = max(float(a.abs().max()) for a in exact)
    assert drift / scale < 0.02, f"the replayed prefill drifted {drift} from an exact one, scale {scale}"
