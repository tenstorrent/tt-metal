# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The adapter's hybrid single mode (``QWEN38_HYBRID_SINGLE``) on a fake chain and a fake lane decoder (no device).

A lone plain greedy request stays on the B=1 chain (no import at prefill, a chain step per decode call, ``[lanes, 1]``
outputs); a second prefill, a second live row, a slot remap that moves the request, a sampled or penalized policy, a
logprobs request or the logits form demote it into a lane (``import_lane`` with the chain's length, PLE context and the
last page row, before the chain is touched); release clears it; knob off is the old import-right-after-prefill path.
"""

from __future__ import annotations

from collections import deque

import pytest
import torch

from models.demos.blackhole.qwen38_flash_next.tests.test_qwen38_vllm_no_device import (  # noqa: F401  (fixture)
    ALLOCATED_CONTEXT,
    construction,
)
from models.demos.blackhole.qwen38_flash_next.tests.test_vllm_batch_adapter_no_device import (  # noqa: F401
    BLOCK,
    LANES,
    Rig,
    _prompt,
    _single_path_prefill,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import SEED_TOKEN_ID, greedy_batch_sampling_params


@pytest.fixture
def rig(monkeypatch) -> Rig:
    rig = Rig(monkeypatch)
    rig.model.hybrid_single = True
    decoder, chain = rig.decoder, rig.chain
    import_lane = decoder.import_lane

    def marked_import(vm, lane, length, context, pages):
        chain.log.append("import")  # where the import sits among the chain's calls
        import_lane(vm, lane, length, context, pages)

    decoder.import_lane = marked_import
    return rig


def _imports(rig: Rig) -> list:
    return [entry for entry in rig.decoder.log if isinstance(entry, tuple) and entry[0] == "import"]


def _sampled(rows: int, **overrides):
    params = greedy_batch_sampling_params(rows)
    for name, value in overrides.items():
        setattr(params, name, [value] * rows)
    return params


def _start(rig: Rig, length: int = 40, blocks=(5,), slot: int = 0):
    out, _ = rig.prefill([_prompt(length, 1000)], [slot], [list(blocks)], greedy_batch_sampling_params(1))
    return int(out[0, 0])


def _chain_decode(rig: Rig, token: int, position: int, blocks, **overrides):
    return rig.decode([(token, position, list(blocks))], **overrides)


def test_a_lone_plain_greedy_row_starts_the_hybrid_without_an_import(rig: Rig) -> None:
    token = _start(rig)
    model = rig.model
    assert model.hybrid_slot == 0 and model.hybrid_pages == [5]
    assert _imports(rig) == [] and rig.decoder.positions == [None] * LANES
    assert len(model.slot.committed) == 40 and model.slot.policy is not None
    assert token >= 0


def test_decode_runs_on_the_chain_with_the_live_row_set_in_a_lanes_by_one_output(rig: Rig) -> None:
    token = _start(rig)
    model, chain = rig.model, rig.chain
    chain.log.clear()
    out = _chain_decode(rig, token, 40, [5])
    assert tuple(out.shape) == (LANES, 1) and out.dtype == torch.int32
    assert int(out[1:].abs().sum()) == 0 and int(out[0, 0]) >= 0
    assert len(model.slot.committed) == 41 and rig.decoder.steps == []  # no lane replay
    assert any(entry.startswith("head:") for entry in chain.log)
    stats = model.decode_stats[-1]
    assert stats["rows"] == 1 and stats["bucket"] == 0 and model.decode_calls == 1


def test_the_page_row_is_kept_from_each_decode_call(rig: Rig) -> None:
    token = _start(rig, length=63, blocks=(5,))
    _chain_decode(rig, token, 63, [5])
    assert rig.model.hybrid_pages == [5]
    _chain_decode(rig, token, 64, [5, 9])  # position 64 needs the second block
    assert rig.model.hybrid_pages == [5, 9]


def test_the_batch_decode_log_line_works_with_the_step_split_and_the_chain_row(rig: Rig, monkeypatch) -> None:
    from models.demos.blackhole.qwen38_flash_next.tools import qwen38_vllm as module

    monkeypatch.setattr(module, "STEP_SPLIT", True)
    monkeypatch.setattr(module, "DECODE_LOG_INTERVAL", 2)
    rig.model._split_outside = deque(maxlen=2)  # the split state the constructor builds under QWEN38_STEP_SPLIT
    token = _start(rig)
    for step in range(2):
        _chain_decode(rig, token, 40 + step, [5])
    assert "split" in rig.model.decode_stats[-1] and rig.model.decode_stats[-1]["step_split"]["exe"] >= 0.0


def test_a_second_prefill_demotes_with_the_chain_length_context_and_last_pages_before_the_chain_reset(rig: Rig) -> None:
    token = _start(rig)
    model, chain = rig.model, rig.chain
    for step in range(3):
        token = int(_chain_decode(rig, token, 40 + step, [5])[0, 0])
    context = model.slot.ple_context
    slot_before = model.slot
    chain.log.clear()
    out, _ = rig.prefill([_prompt(20, 5000)], [1], [[7]], greedy_batch_sampling_params(1))
    assert _imports(rig)[0] == ("import", 0, 43, (5,), context)
    assert chain.log.index("import") < next(i for i, e in enumerate(chain.log) if e.startswith("reset:"))
    assert model.hybrid_slot is None and model.lane_slots[0] is slot_before and model.slot is not slot_before
    assert rig.decoder.last_admit["hybrid_import_s"] >= 0.0
    # the new request imported into a lane too (no hybrid: a lane holds a request)
    assert rig.decoder.positions[:2] == [43, 20]
    assert int(out[0, 0]) >= 0


def test_a_hybrid_request_continues_on_the_lane_after_the_demote(rig: Rig) -> None:
    token = _start(rig)
    token = int(_chain_decode(rig, token, 40, [5])[0, 0])
    rig.prefill([_prompt(20, 5000)], [1], [[7]], greedy_batch_sampling_params(1))
    out = rig.decode([(token, 41, [5]), (9, 20, [7])])
    assert rig.decoder.steps and rig.decoder.steps[-1]["positions"][:2] == [41, 20]
    assert int(out[0, 0]) == rig.decoder.next_id(token, 41)


def test_the_hybrid_starts_only_for_one_row_with_idle_lanes_and_no_hybrid(rig: Rig) -> None:
    prompts = [_prompt(30, 100), _prompt(35, 200)]
    rig.prefill(prompts, [0, 1], [[3], [4]], greedy_batch_sampling_params(2))
    assert rig.model.hybrid_slot is None and len(_imports(rig)) == 2
    rig.prefill([_prompt(20, 300)], [2], [[8]], greedy_batch_sampling_params(1))  # lanes are live
    assert rig.model.hybrid_slot is None and len(_imports(rig)) == 3


@pytest.mark.parametrize("sampling", ("sampled", "penalized", "logprobs"))
def test_a_prefill_without_plain_greedy_params_never_starts_the_hybrid(rig: Rig, sampling: str) -> None:
    params = {
        "sampled": _sampled(1, temperature=0.8, top_k=20, top_p=0.9),
        "penalized": _sampled(1, presence_penalty=0.5),
        "logprobs": _sampled(1, enable_log_probs=True),
    }[sampling]
    rig.prefill([_prompt(40, 1000)], [0], [[5]], params)
    assert rig.model.hybrid_slot is None and len(_imports(rig)) == 1


def _start_logits(rig: Rig, length: int = 40, blocks=(5,), slot: int = 0):
    """The server's prefill: ``sampling_params=None``, the logits come back and the plugin samples on the host."""

    out, _ = rig.prefill([_prompt(length, 1000)], [slot], [list(blocks)], None)
    return out


def test_a_prefill_without_sampling_params_starts_the_hybrid_and_returns_the_single_paths_logits(
    rig: Rig, monkeypatch
) -> None:
    out = _start_logits(rig)
    model = rig.model
    assert model.hybrid_slot == 0 and model.hybrid_pages == [5] and _imports(rig) == []
    assert len(model.slot.committed) == 40 and model.slot.policy is None
    reference = Rig(monkeypatch)
    reference.model.hybrid_single = False
    expected = _start_logits(reference)
    assert len(_imports(reference)) == 1 and reference.model.hybrid_slot is None
    assert out.shape == expected.shape == (1, 1, out.shape[-1]) and out.dtype == expected.dtype
    assert torch.equal(out, expected)


def test_the_first_decode_after_a_logits_prefill_feeds_the_plugins_token_and_stays_on_the_chain(rig: Rig) -> None:
    out = _start_logits(rig)
    token = int(torch.argmax(out[0, 0]))  # the plugin's host sampling
    rig.chain.log.clear()
    result = _chain_decode(rig, token, 40, [5])
    assert f"feed:{token}" in rig.chain.log and rig.model.hybrid_slot == 0 and _imports(rig) == []
    assert result.shape == (LANES, 1) and len(rig.model.slot.committed) == 41
    assert rig.model.slot.policy is not None  # adopted from the decode's real policy


def test_the_first_decode_after_a_logits_prefill_demotes_when_the_policy_is_sampled(rig: Rig) -> None:
    out = _start_logits(rig)
    token = int(torch.argmax(out[0, 0]))
    rig.decode([(token, 40, [5])], sampling_params=_sampled(LANES, temperature=0.8, top_k=20, top_p=0.9))
    assert rig.model.hybrid_slot is None and len(_imports(rig)) == 1 and rig.decoder.steps


def test_the_knob_off_imports_right_after_the_prefill(rig: Rig) -> None:
    rig.model.hybrid_single = False
    _start(rig)
    assert rig.model.hybrid_slot is None and len(_imports(rig)) == 1 and rig.decoder.positions[0] == 40
    assert rig.chain.log.index("import") > rig.chain.log.index(f"reset:{SEED_TOKEN_ID}")


def test_release_clears_the_hybrid(rig: Rig) -> None:
    _start(rig)
    rig.model.release_request(0)
    assert rig.model.hybrid_slot is None and rig.model.slot.committed == [] and _imports(rig) == []
    rig.prefill([_prompt(20, 5000)], [1], [[7]], greedy_batch_sampling_params(1))  # starts a new hybrid, no demote
    assert rig.model.hybrid_slot == 1 and _imports(rig) == []


def test_release_persistent_capture_clears_the_hybrid(rig: Rig) -> None:
    _start(rig)
    rig.model.release_persistent_capture()
    assert rig.model.hybrid_slot is None


def _demoted_by(rig: Rig, **decode_overrides) -> None:
    token = _start(rig)
    assert rig.model.hybrid_slot == 0
    rig.decode([(token, 40, [5])], **decode_overrides)
    assert rig.model.hybrid_slot is None and rig.model.slot.committed == []
    assert _imports(rig) == [("import", rig.model.lane_of_slot[0], 40, (5,), rig.decoder.contexts[0])]


def test_a_sampled_policy_demotes_at_the_decode(rig: Rig) -> None:
    _demoted_by(rig, sampling_params=_sampled(LANES, temperature=0.8, top_k=20, top_p=0.9))
    assert rig.decoder.steps


def test_a_penalized_policy_demotes_at_the_decode(rig: Rig) -> None:
    _demoted_by(
        rig,
        sampling_params=_sampled(LANES, presence_penalty=0.5),
        prompt_tokens=[list(_prompt(40, 1000))] * LANES,
        output_tokens=[[]] * LANES,
    )


def test_a_logprobs_request_demotes_at_the_decode(rig: Rig) -> None:
    token = _start(rig)
    out = rig.decode([(token, 40, [5])], sampling_params=_sampled(LANES, enable_log_probs=True))
    assert isinstance(out, tuple) and rig.model.hybrid_slot is None and len(_imports(rig)) == 1


def test_the_logits_form_demotes_at_the_decode(rig: Rig) -> None:
    token = _start(rig)
    out = rig.decode([(token, 40, [5])], sampling_params=None)
    assert out.shape[0] == LANES and rig.model.hybrid_slot is None and len(_imports(rig)) == 1


def test_a_second_live_row_demotes_at_the_decode(rig: Rig, expect_error) -> None:
    token = _start(rig)
    # a second live row maps to a lane with no request: the lane path refuses it as a contract error, after the demote
    with expect_error(Exception, "no prefilled request"):
        rig.decode([(token, 40, [5]), (7, 10, [6])])
    assert rig.model.hybrid_slot is None and len(_imports(rig)) == 1


def test_a_slot_remap_that_does_not_serve_the_hybrid_slot_demotes(rig: Rig, expect_error) -> None:
    token = _start(rig, slot=2)
    assert rig.model.hybrid_slot == 2
    remap = [2, 1, 0] + list(range(3, LANES))  # row 0 resolves to the request's slot 2: stays
    out = rig.decode([(token, 40, [5])], slot_remap=remap)
    assert rig.model.hybrid_slot == 0 and rig.model.lane_of_slot[0] == 2 and not rig.decoder.steps
    assert tuple(out.shape) == (LANES, 1)
    # row 0 now resolves to slot 5, not the request's slot 0: demote, then the lane path refuses the stranger row
    mismatch = [5, 1, 2, 3, 4, 0, 6, 7]
    with expect_error(Exception, "no prefilled request|hold"):
        rig.decode([(int(out[0, 0]), 41, [5])], slot_remap=mismatch)
    assert rig.model.hybrid_slot is None and len(_imports(rig)) == 1


def test_a_failed_chain_step_marks_the_slot_and_a_demote_drops_the_request(rig: Rig, monkeypatch, expect_error) -> None:
    token = _start(rig)

    def boom(*_args, **_kwargs):
        raise RuntimeError("device")

    monkeypatch.setattr(rig.chain, "execute_tail", boom)
    with expect_error(RuntimeError, "device|import"):
        _chain_decode(rig, token, 40, [5])
    assert rig.model.slot.failed
    monkeypatch.undo()
    monkeypatch.setenv("QWEN38_PREFILL_PACK", "0")
    rig.prefill([_prompt(20, 5000)], [1], [[7]], greedy_batch_sampling_params(1))
    assert [entry[2] for entry in _imports(rig)] == [20]  # the failed request is dropped, only the new one imported
    assert rig.model.hybrid_slot is None and not rig.model.slot.failed


def test_an_import_failure_at_the_demote_sets_batch_failed_and_reraises(rig: Rig, expect_error) -> None:
    _start(rig)

    def boom(*_args, **_kwargs):
        raise RuntimeError("import")

    rig.decoder.import_lane = boom
    with expect_error(RuntimeError, "device|import"):
        rig.prefill([_prompt(20, 5000)], [1], [[7]], greedy_batch_sampling_params(1))
    assert rig.model.batch_failed
