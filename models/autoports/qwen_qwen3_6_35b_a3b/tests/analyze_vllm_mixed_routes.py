# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Attribute optional adapter route events to completed mixed-probe requests.

CPU/file only. The response's prompt digest and overlapping local monotonic
interval identify a prefill; client submission order never assigns a slot.
"""

import argparse
import ast
import json
import re
from pathlib import Path
from urllib.parse import urlparse

MARKER = "QWEN_VLLM_EVENT "
LEGACY_REMAP = re.compile(r"tensor\((\[\s*[0-9]{1,4}(?:\s*,\s*[0-9]{1,4})*\s*\]),\s*dtype=torch\.int32\)")


def parse_slot_remap(value):
    """Read complete legacy CPU int32 reprs, never truncated/arbitrary code."""
    if value is None or isinstance(value, list):
        return value
    if not isinstance(value, str) or len(value) > 65536:
        raise ValueError("Unsupported remap encoding or oversized tensor repr")
    match = LEGACY_REMAP.fullmatch(value)
    if match is None:
        raise ValueError("Expected a complete tensor([nonnegative integers], dtype=torch.int32) CPU repr")
    try:
        values = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError) as error:
        raise ValueError("Malformed integer list in legacy tensor repr") from error
    if len(values) > 4096 or any(type(slot) is not int or slot >= 4096 for slot in values):
        raise ValueError("Legacy tensor repr exceeds bounded slot range")
    return values


def read_events(path):
    events, issues = [], []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if MARKER not in line:
            continue
        try:
            event, _ = json.JSONDecoder().raw_decode(line.split(MARKER, 1)[1])
            event["log_line"] = line_number
            events.append(event)
        except (ValueError, TypeError) as error:
            issues.append(dict(kind="unparseable_event", log_line=line_number, error=str(error)))
    return events, issues


def analyze(probe, events, *, same_host_clock=False, parse_issues=()):
    issues = list(parse_issues)
    requests = []
    for batch in probe.get("batches", []):
        for request in batch.get("requests", []):
            record = dict(
                batch_index=batch["batch_index"],
                case_index=request["case_index"],
                response_request_id=request.get("response_request_id"),
                prompt_token_digest=request.get("prompt_token_digest"),
                payload=request.get("payload", {}),
                returned_output_token_count=len(request.get("token_ids") or []),
                prompt_length=len(request.get("prompt_token_ids") or []),
                submitted_ns=request.get("submitted", {}).get("monotonic_ns"),
                completed_ns=request.get("completed", {}).get("monotonic_ns"),
                prefill_events=[],
                routes=[],
                evidence_issues=[],
            )
            requests.append(record)
    result = dict(
        status="INCOMPLETE",
        evidence_scope="Adapter forward routes; host draws happen in plugin after logits return. Submitted decode can exceed emitted outputs under async scheduling.",
        causality="Output differences and route differences alone do not establish logits equality or a causal explanation of a mismatch.",
        requests=requests,
        admissions=[],
        submissions=[],
        issues=issues,
    )
    if not same_host_clock:
        issues.append(
            dict(
                kind="clock_domain_unverified",
                detail="Require probe and server monotonic clocks from the same host boot",
            )
        )
        return result
    if not requests or any(item["submitted_ns"] is None or item["completed_ns"] is None for item in requests):
        issues.append(dict(kind="incomplete_request_intervals"))
        return result
    identities = [item["response_request_id"] for item in requests]
    if any(not identity for identity in identities) or len(set(identities)) != len(identities):
        issues.append(dict(kind="missing_or_duplicate_response_identity"))
        return result
    lower = min(item["submitted_ns"] for item in requests) / 1e9
    upper = max(item["completed_ns"] for item in requests) / 1e9
    owners, pending = {}, {}
    event_index = 0

    def issue(kind, event, **fields):
        item = dict(kind=kind, log_line=event.get("log_line"), submission_id=event.get("submission_id"), **fields)
        issues.append(item)
        return item

    def route_name(event):
        return {"device": "device", "host_logits": "host"}.get(event.get("sampling"), "unknown")

    def add_route(owner, event, phase, slot, **fields):
        request = requests[owner]
        if not request["submitted_ns"] / 1e9 <= event["monotonic"] <= request["completed_ns"] / 1e9:
            request["evidence_issues"].append(issue("forward_outside_request_interval", event, owner=owner, slot=slot))
        request["routes"].append(
            dict(
                phase=phase,
                slot=slot,
                route=route_name(event),
                monotonic=event["monotonic"],
                log_line=event.get("log_line"),
                **fields,
            )
        )
        if route_name(event) == "unknown":
            request["evidence_issues"].append(issue("unknown_route", event, owner=owner))

    for event in events:
        stamp = event.get("monotonic")
        if not isinstance(stamp, (int, float)):
            issue("missing_event_clock", event)
            continue
        if not lower <= stamp <= upper:
            continue
        kind = event.get("event")
        if kind not in ("prefill", "decode", "decode_submitted"):
            continue
        event_index += 1
        if kind == "prefill":
            if pending:
                issue("prefill_interleaves_unpaired_decode", event)
                owners.clear()
            chunks = event.get("prompt_chunks")
            if not chunks:
                issue("missing_prompt_chunks", event)
                owners.clear()
                continue
            for chunk in chunks:
                slot = chunk.get("slot")
                if not isinstance(slot, int):
                    issue("invalid_prefill_slot", event)
                    owners.clear()
                    continue
                candidates = [
                    index
                    for index, request in enumerate(requests)
                    if request["prompt_token_digest"] is not None
                    and request["prompt_token_digest"] == chunk.get("token_digest")
                    and chunk.get("start") == 0
                    and chunk.get("end") == request["prompt_length"]
                    and request["submitted_ns"] / 1e9 <= stamp <= request["completed_ns"] / 1e9
                ]
                if len(candidates) != 1:
                    issue(
                        "ambiguous_or_unmatched_prefill",
                        event,
                        slot=slot,
                        candidate_response_ids=[requests[i]["response_request_id"] for i in candidates],
                    )
                    owners.pop(slot, None)
                    continue
                owner = candidates[0]
                if requests[owner]["prefill_events"]:
                    requests[owner]["evidence_issues"].append(issue("multiple_terminal_prefills", event, owner=owner))
                previous = owners.get(slot)
                owners[slot] = owner
                admission = dict(
                    log_line=event.get("log_line"),
                    monotonic=stamp,
                    slot=slot,
                    response_request_id=requests[owner]["response_request_id"],
                    batch_index=requests[owner]["batch_index"],
                    case_index=requests[owner]["case_index"],
                    replaced_response_request_id=(
                        None if previous is None else requests[previous]["response_request_id"]
                    ),
                )
                requests[owner]["prefill_events"].append(admission)
                result["admissions"].append(admission)
                add_route(owner, event, "terminal_prefill_forward", slot)
        elif kind == "decode":
            sid = event.get("submission_id")
            if sid is None or sid in pending or pending:
                issue("missing_duplicate_or_overlapping_submission_id", event)
                owners.clear()
                pending.clear()
            if sid is not None:
                pending[sid] = (event, dict(owners), event_index)
        else:
            sid = event.get("submission_id")
            item = pending.pop(sid, None)
            if item is None:
                issue("orphan_decode_submitted", event)
                owners.clear()
                continue
            before, old_owners, before_index = item
            if event_index != before_index + 1:
                issue("nonadjacent_decode_pair", event)
                owners.clear()
                continue
            remap_raw = before.get("slot_remap")
            try:
                remap = parse_slot_remap(remap_raw)
            except ValueError as error:
                issue("invalid_slot_remap_encoding", event, slot_remap_raw=remap_raw, error=str(error))
                owners.clear()
                continue
            if remap is None:
                owners = old_owners
            elif (
                isinstance(remap, list)
                and all(type(slot) is int for slot in remap)
                and sorted(remap) == list(range(len(remap)))
            ):
                if any(slot >= len(remap) for slot in old_owners):
                    issue("remap_omits_known_slot", event)
                    owners.clear()
                    continue
                # Contract: destination slot i receives old slot remap[i].
                owners = {
                    destination: old_owners[source] for destination, source in enumerate(remap) if source in old_owners
                }
            else:
                issue("invalid_slot_remap", event, slot_remap=remap)
                owners.clear()
                continue
            active = event.get("active_slots")
            if (
                not isinstance(active, list)
                or any(type(slot) is not int for slot in active)
                or len(set(active)) != len(active)
            ):
                issue("invalid_active_slots", event)
                owners.clear()
                continue
            co_resident = [requests[owners[slot]]["response_request_id"] for slot in active if slot in owners]
            result["submissions"].append(
                dict(
                    submission_id=sid,
                    decode_log_line=before.get("log_line"),
                    submitted_log_line=event.get("log_line"),
                    route=route_name(before),
                    slot_remap=remap,
                    slot_remap_raw=remap_raw,
                    slot_remap_encoding="legacy_cpu_tensor_repr" if isinstance(remap_raw, str) else "json_list_or_null",
                    active_slots=active,
                    mapped_response_request_ids=co_resident,
                    reload_inputs=before.get("reload_inputs"),
                    reload_sampling_params=before.get("reload_sampling_params"),
                    reset_sampling_state=before.get("reset_sampling_state"),
                )
            )
            for slot in active:
                if slot not in owners:
                    issue("unmapped_active_slot", event, slot=slot)
                    continue
                add_route(
                    owners[slot],
                    before,
                    "decode_forward",
                    slot,
                    submission_id=sid,
                    co_resident_response_request_ids=co_resident,
                )
    for sid, (event, _, _) in pending.items():
        issue("unpaired_decode", event, pending_submission_id=sid)
    for request in requests:
        if not request["prefill_events"]:
            request["evidence_issues"].append(dict(kind="no_mapped_prefill"))
        request["route_sequence"] = [item["route"] for item in request["routes"]]
        request["route_transitions"] = [
            dict(at_forward_index=index, previous=previous, current=current)
            for index, (previous, current) in enumerate(
                zip(request["route_sequence"], request["route_sequence"][1:]), 1
            )
            if previous != current
        ]
        request["submitted_forward_count"] = len(request["routes"])
        if len(request["routes"]) < request["returned_output_token_count"]:
            request["evidence_issues"].append(dict(kind="fewer_mapped_forwards_than_returned_tokens"))
        request["mapping_status"] = (
            "CONFIDENT" if not request["evidence_issues"] and not issues else "INCOMPLETE_OR_AMBIGUOUS"
        )
    result["pair_comparisons"] = []
    for case in sorted({request["case_index"] for request in requests}):
        pair = sorted(
            (request for request in requests if request["case_index"] == case),
            key=lambda request: request["batch_index"],
        )
        if len(pair) == 2:
            result["pair_comparisons"].append(
                dict(
                    case_index=case,
                    response_request_ids=[request["response_request_id"] for request in pair],
                    route_sequences_equal=pair[0]["route_sequence"] == pair[1]["route_sequence"],
                    any_observed_route_transition=any(request["route_transitions"] for request in pair),
                    mapping_status=(
                        "CONFIDENT"
                        if all(request["mapping_status"] == "CONFIDENT" for request in pair)
                        else "INCOMPLETE_OR_AMBIGUOUS"
                    ),
                )
            )
    result["status"] = (
        "COMPLETE_ATTRIBUTION"
        if requests and all(request["mapping_status"] == "CONFIDENT" for request in requests)
        else "INCOMPLETE"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--server-log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--same-host-clock", action="store_true", help="Confirm client/server use the same host boot's monotonic clock"
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence; choose a fresh output path")
    probe = json.loads(args.probe.read_text())
    events, issues = read_events(args.server_log)
    result = analyze(probe, events, same_host_clock=args.same_host_clock, parse_issues=issues)
    result.update(
        probe_path=str(args.probe),
        server_log_path=str(args.server_log),
        probe_status=probe.get("status"),
        probe_host=urlparse(probe.get("url", "")).hostname,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=result["status"],
                requests=len(result["requests"]),
                issues=len(result["issues"]),
                output=str(args.output),
            )
        )
    )
    if result["status"] != "COMPLETE_ATTRIBUTION":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
