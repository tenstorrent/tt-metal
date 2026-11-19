# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU fixtures for attribution, permutation direction, reuse and ambiguity."""

from copy import deepcopy

from models.autoports.qwen_qwen3_6_35b_a3b.tests.analyze_vllm_mixed_routes import analyze


def request(case, identity, digest, start, finish):
    return dict(
        case_index=case,
        response_request_id=identity,
        prompt_token_digest=digest,
        prompt_token_ids=[1, 2],
        token_ids=[3, 4],
        payload={},
        submitted=dict(monotonic_ns=int(start * 1e9)),
        completed=dict(monotonic_ns=int(finish * 1e9)),
    )


def fixtures():
    probe = dict(
        batches=[
            dict(batch_index=0, requests=[request(0, "a", "digest-a", 1, 8), request(1, "b", "digest-b", 1, 8)]),
            dict(batch_index=1, requests=[request(0, "a-reused", "digest-a", 9, 14)]),
        ]
    )
    events = [
        dict(
            event="prefill",
            monotonic=2,
            sampling="device",
            prompt_chunks=[
                dict(slot=0, start=0, end=2, token_digest="digest-a"),
                dict(slot=1, start=0, end=2, token_digest="digest-b"),
            ],
        ),
        # A three-cycle distinguishes destination<-source from its inverse.
        dict(event="decode", monotonic=3, submission_id=1, sampling="host_logits", slot_remap=[1, 2, 0]),
        dict(event="decode_submitted", monotonic=3.1, submission_id=1, active_slots=[0, 2]),
        dict(
            event="prefill",
            monotonic=10,
            sampling="device",
            prompt_chunks=[
                dict(slot=0, start=0, end=2, token_digest="digest-a"),
            ],
        ),
        dict(event="decode", monotonic=11, submission_id=2, sampling="device", slot_remap=None),
        dict(event="decode_submitted", monotonic=11.1, submission_id=2, active_slots=[0]),
    ]
    for index, event in enumerate(events, 1):
        event["log_line"] = index
    return probe, events


def test_three_cycle_remap_and_slot_reuse():
    probe, events = fixtures()
    result = analyze(probe, events, same_host_clock=True)
    assert result["status"] == "COMPLETE_ATTRIBUTION"
    records = {row["response_request_id"]: row for row in result["requests"]}
    assert [row["slot"] for row in records["a"]["routes"]] == [0, 2]
    assert [row["slot"] for row in records["b"]["routes"]] == [1, 0]
    assert [row["slot"] for row in records["a-reused"]["routes"]] == [0, 0]
    assert records["a"]["route_sequence"] == ["device", "host"]
    assert records["a-reused"]["route_sequence"] == ["device", "device"]
    assert result["admissions"][-1]["replaced_response_request_id"] == "b"
    assert result["pair_comparisons"][0]["route_sequences_equal"] is False


def test_duplicate_prompt_with_overlapping_intervals_is_ambiguous():
    probe, events = fixtures()
    duplicate = deepcopy(probe["batches"][0]["requests"][0])
    duplicate["response_request_id"] = "overlapping-duplicate"
    probe["batches"][0]["requests"].append(duplicate)
    result = analyze(probe, events, same_host_clock=True)
    assert result["status"] == "INCOMPLETE"
    assert any(issue["kind"] == "ambiguous_or_unmatched_prefill" for issue in result["issues"])
    assert all(row["mapping_status"] != "CONFIDENT" for row in result["requests"])


def test_mismatched_submission_ids_do_not_assign_decode_routes():
    probe, events = fixtures()
    events[2]["submission_id"] = 99
    result = analyze(probe, events, same_host_clock=True)
    assert result["status"] == "INCOMPLETE"
    assert any(issue["kind"] == "orphan_decode_submitted" for issue in result["issues"])
    assert all(route.get("submission_id") != 1 for row in result["requests"] for route in row["routes"])


def test_clock_domain_must_be_explicit():
    probe, events = fixtures()
    result = analyze(probe, events)
    assert result["status"] == "INCOMPLETE"
    assert result["issues"][-1]["kind"] == "clock_domain_unverified"
    assert all(not row["routes"] for row in result["requests"])


def test_missing_prefill_identity_never_uses_client_order():
    probe, events = fixtures()
    events[0]["prompt_chunks"] = []
    result = analyze(probe, events, same_host_clock=True)
    assert result["status"] == "INCOMPLETE"
    assert any(issue["kind"] == "unmapped_active_slot" for issue in result["issues"])
    first_batch = [row for row in result["requests"] if row["batch_index"] == 0]
    assert all(not row["routes"] for row in first_batch)


def test_missing_actual_response_id_is_incomplete():
    probe, events = fixtures()
    probe["batches"][0]["requests"][0]["response_request_id"] = None
    result = analyze(probe, events, same_host_clock=True)
    assert result["status"] == "INCOMPLETE"
    assert result["issues"][-1]["kind"] == "missing_or_duplicate_response_identity"


def test_invalid_remap_does_not_guess_ownership():
    probe, events = fixtures()
    events[1]["slot_remap"] = [0, 0, 1]
    result = analyze(probe, events, same_host_clock=True)
    assert result["status"] == "INCOMPLETE"
    assert any(issue["kind"] == "invalid_slot_remap" for issue in result["issues"])
    assert all(route.get("submission_id") != 1 for row in result["requests"] for route in row["routes"])


def test_complete_legacy_cpu_tensor_repr_preserves_mapping_and_raw():
    probe, events = fixtures()
    raw = "tensor([1, 2,\n 0],\n dtype=torch.int32)"
    events[1]["slot_remap"] = raw
    result = analyze(probe, events, same_host_clock=True)
    assert result["status"] == "COMPLETE_ATTRIBUTION"
    assert result["submissions"][0]["slot_remap"] == [1, 2, 0]
    assert result["submissions"][0]["slot_remap_raw"] == raw
    assert result["submissions"][0]["slot_remap_encoding"] == "legacy_cpu_tensor_repr"
    assert [route["slot"] for route in result["requests"][0]["routes"]] == [0, 2]


def test_legacy_tensor_parser_rejects_truncation_expressions_and_other_formats():
    invalid = [
        "tensor([1, ..., 0], dtype=torch.int32)",
        "tensor([1, 2, 0], dtype=torch.int64)",
        "tensor([1, 2, 0], device='cuda:0', dtype=torch.int32)",
        "tensor([1, 1+1, 0], dtype=torch.int32)",
        "tensor([True, 2, 0], dtype=torch.int32)",
        "tensor([1.0, 2, 0], dtype=torch.int32)",
        "tensor([1, -1, 0], dtype=torch.int32)",
        "tensor([1, 2, 0], dtype=torch.int32); print('unsafe')",
        "tensor([1, 4096, 0], dtype=torch.int32)",
        "tensor([1, 2, 0], dtype=torch.int32)" + " " * 65536,
    ]
    for raw in invalid:
        probe, events = fixtures()
        events[1]["slot_remap"] = raw
        result = analyze(probe, events, same_host_clock=True)
        assert result["status"] == "INCOMPLETE"
        assert any(issue["kind"] == "invalid_slot_remap_encoding" for issue in result["issues"])
        assert all(route.get("submission_id") != 1 for row in result["requests"] for route in row["routes"])
