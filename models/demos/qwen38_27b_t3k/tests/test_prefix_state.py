# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Restoring a saved slot reproduces the decode it was saved from, on device.

The 16 full-attention layers keep their prefix in pages the scheduler can hand to another
request. The 48 linear-attention layers do not: `recurrent` and `conv` summarise the tokens one
slot has seen. A prefix cache therefore has to save and reinstate that state explicitly, and this
test is what says the reinstatement is exact rather than approximately right.

Three arms, because an assertion that a restored run matches the original cannot by itself tell a
correct restore from a restore that did nothing: the slot would still hold the state. The reset
arm is the control that gives the comparison its teeth.

The comparison is on logits rather than sampled tokens. A model truncated to two layers is
degenerate enough to emit the same token whatever the recurrent state holds, which makes argmax
blind to exactly the thing under test.

Two layers, one of each kind, so a restore that corrupted the attention path would also show.
"""

import pytest
import torch

import ttnn
from models.demos.qwen38_27b_t3k.tt.decoder_tp import native_mesh_shape
from models.demos.qwen38_27b_t3k.tt.generator import build_generator, configure_fabric

PROMPT = 96
STEPS = 6
LAYERS = [0, 3]  # layer_types alternates every fourth index: 0 is linear_attention, 3 full
# Where a 32-token page size stops a 96-token prompt: the last boundary strictly below it,
# so the pass that follows still has tokens to run.
BOUNDARY = 64


@pytest.fixture(scope="module")
def harness():
    from pathlib import Path

    root = Path(__file__).parents[1]
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(*native_mesh_shape()), trace_region_size=200000000)
    try:
        gen = build_generator(root, mesh, precision_config=root / "config/precision.json", layer_indices=LAYERS)
        kinds = [layer.kind for layer in gen.model.layers]
        assert "linear_attention" in kinds and "full_attention" in kinds, kinds
        cache = gen.model.allocate_cache(batch_size=1, capacity=1024)
        table = torch.arange(cache.num_pages, dtype=torch.int32).reshape(1, -1)
        gen.bind_cache(cache, table)
        yield gen, mesh, cache, table
        gen.close()
    finally:
        ttnn.close_mesh_device(mesh)


def _prefill(gen, cache, table, prompt):
    gen.reset_recurrent_slots([0])
    gen.prefill_forward(prompt, page_table=table, kv_cache=cache, prompt_lens=[PROMPT], start_pos=[0], slots=[0])


def _decode(gen, mesh, cache, first_token):
    """Decode from the slot's current state, returning each step's full-vocabulary logits."""
    model = gen.model
    tokens, out = first_token, []
    for step in range(STEPS):
        ids = model.upload(
            torch.full((1, 1, 1, 32), tokens, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        position = torch.tensor([PROMPT + step], dtype=torch.int32)
        logits = model.decode(
            ids,
            model.upload(position, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT),
            cache=cache,
            page_table=gen.page_table,
            rope_indices=model.upload(position, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        )
        whole = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
        tokens = int(whole.reshape(-1).argmax())
        out.append(whole.reshape(-1).clone())
    return out


def test_a_restored_slot_decodes_exactly_what_the_snapshot_was_taken_from(harness):
    gen, mesh, cache, table = harness
    prompt = torch.randint(1000, 5000, (1, PROMPT), dtype=torch.int32)
    first = int(prompt[0, -1])

    _prefill(gen, cache, table, prompt)
    assert gen._slot_prefix_len[0] == PROMPT
    handle = gen.save_slot_state(0)
    assert handle is not None
    reference = _decode(gen, mesh, cache, first)

    # Control: zeroing the slot must change the continuation, or the comparison below proves
    # nothing about whether the restore actually wrote anything.
    gen.reset_recurrent_slots([0])
    assert gen._slot_prefix_len[0] == 0
    zeroed = _decode(gen, mesh, cache, first)

    gen.reset_recurrent_slots([0])
    assert gen.restore_slot_state(0, handle)
    assert gen._slot_prefix_len[0] == PROMPT
    restored = _decode(gen, mesh, cache, first)

    control = max(float((a - b).abs().max()) for a, b in zip(zeroed, reference))
    assert control > 0.0, (
        "zeroing the recurrent state left every logit identical, so this test cannot "
        "distinguish a real restore from a no-op"
    )
    for step, (got, want) in enumerate(zip(restored, reference)):
        assert torch.equal(got, want), (
            f"step {step} differs from the snapshot it was taken from by "
            f"{float((got - want).abs().max())}, against {control} for a zeroed slot"
        )


def test_a_restored_slot_continues_prefill_where_the_snapshot_left_off(harness):
    gen, mesh, cache, table = harness
    prompt = torch.randint(1000, 5000, (1, PROMPT + 32), dtype=torch.int32)

    _prefill(gen, cache, table, prompt[:, :PROMPT])
    handle = gen.save_slot_state(0)
    gen.reset_recurrent_slots([0])
    assert gen.restore_slot_state(0, handle)

    # The continuity gate reads the reinstated length, so the next chunk is accepted as the
    # continuation it is rather than refused as a prefix-cache hit with no state behind it.
    # Prefill reads its tokens from index zero, so the chunk is what gets passed, not the prompt.
    gen.prefill_forward(
        prompt[:, PROMPT:], page_table=table, kv_cache=cache, prompt_lens=[32], start_pos=[PROMPT], slots=[0]
    )
    assert gen._slot_prefix_len[0] == PROMPT + 32


def test_a_freed_snapshot_is_refused_rather_than_restoring_stale_rows(harness):
    gen, _, cache, table = harness
    prompt = torch.randint(1000, 5000, (1, PROMPT), dtype=torch.int32)
    _prefill(gen, cache, table, prompt)
    handle = gen.save_slot_state(0)
    gen.free_slot_state(handle)
    assert not gen.restore_slot_state(0, handle)


def _gdn_state(gen, mesh, slot=0):
    """Every GDN layer's conv and recurrent rows for one slot, as host tensors.

    Read straight off the cache rather than through the model's output: the state is what the
    split has to preserve, and two layers of model between it and the logits dilute a wrong
    answer until it reads like rounding. These tensors are sharded, so the composer is not
    optional -- one device's share would look plausible and be an eighth of the state.
    """
    out = {}
    for index, layer in enumerate(gen.cache.layers):
        for name in ("conv", "recurrent"):
            tensor = getattr(layer, name)
            if tensor is not None:
                whole = ttnn.to_torch(tensor[slot : slot + 1], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                out[(index, name)] = whole.float()
    return out


def _state_drift(a, b):
    """Largest elementwise difference across every layer, relative to the state's own scale."""
    worst = 0.0
    for key, left in a.items():
        right = b[key]
        scale = max(float(left.abs().max()), 1e-6)
        worst = max(worst, float((left - right).abs().max()) / scale)
    return worst


def test_a_split_prefill_leaves_the_recurrent_state_a_single_pass_would(harness):
    """The state at the boundary is what the snapshot keeps, so it is what has to survive.

    Bit-exactness was never available -- two passes are different matmul and GDN scan shapes than
    one, and prefill already re-chunks every 4096 tokens -- so the measure is relative drift,
    read against the drift of a prefix that is actually wrong.
    """
    gen, mesh, cache, table = harness
    prompt = torch.randint(1000, 5000, (1, PROMPT), dtype=torch.int32)
    other = torch.randint(1000, 5000, (1, PROMPT), dtype=torch.int32)

    def prefill(chunks, tokens=prompt):
        gen.reset_recurrent_slots([0])
        for begin, stop in chunks:
            gen.prefill_forward(
                tokens[:, begin:stop],
                page_table=table,
                kv_cache=cache,
                prompt_lens=[stop - begin],
                start_pos=[begin],
                slots=[0],
            )
        return _gdn_state(gen, mesh)

    whole = prefill([(0, PROMPT)])
    measured = {
        "split@64": _state_drift(prefill([(0, 64), (64, PROMPT)]), whole),
        "split@32": _state_drift(prefill([(0, 32), (32, PROMPT)]), whole),
        "another_prompt": _state_drift(prefill([(0, PROMPT)], tokens=other), whole),
    }
    print("STATE_DRIFT=" + repr(measured))

    # Re-chunking has to be an order of magnitude below computing the wrong prefix entirely,
    # or nothing here distinguishes a split from a corrupted restore.
    assert measured["another_prompt"] > 10 * max(measured["split@64"], measured["split@32"]), measured


def test_a_snapshot_taken_at_the_boundary_serves_the_rest_of_the_prompt(harness):
    """The whole point, end to end: a later request skips what an earlier one already computed.

    The snapshot is taken where the split stops, the slot is reused by something else, and the
    prefix is then reinstated and only the tail prefilled. The state that leaves has to be the
    state a full prefill would have left, and the wrong-prefix arm is what says the comparison
    can tell the difference.
    """
    gen, mesh, cache, table = harness
    prompt = torch.randint(1000, 5000, (1, PROMPT), dtype=torch.int32)
    other = torch.randint(1000, 5000, (1, PROMPT), dtype=torch.int32)

    def prefill(chunks, tokens=prompt):
        for begin, stop in chunks:
            gen.prefill_forward(
                tokens[:, begin:stop],
                page_table=table,
                kv_cache=cache,
                prompt_lens=[stop - begin],
                start_pos=[begin],
                slots=[0],
            )

    gen.reset_recurrent_slots([0])
    prefill([(0, PROMPT)])
    whole = _gdn_state(gen, mesh)

    def serve_from(tokens):
        gen.reset_recurrent_slots([0])
        prefill([(0, BOUNDARY)], tokens=tokens)
        handle = gen.save_slot_state(0)
        assert handle is not None
        # Whatever held the slot next leaves nothing of the prefix behind.
        gen.reset_recurrent_slots([0])
        assert gen.restore_slot_state(0, handle)
        assert gen._slot_prefix_len[0] == BOUNDARY
        prefill([(BOUNDARY, PROMPT)])
        state = _gdn_state(gen, mesh)
        gen.free_slot_state(handle)
        return state

    served = _state_drift(serve_from(prompt), whole)
    wrong = _state_drift(serve_from(other), whole)
    print(f"SERVED_STATE_DRIFT={served} WRONG_PREFIX_DRIFT={wrong}")
    assert wrong > 10 * served, (served, wrong)
