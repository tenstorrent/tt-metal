"""Validate the prepared fixture and cost arithmetic; this does not execute the C++ bridge."""

import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(doc):
    require(doc["status"] == "SOURCE_PREPARATION_ONLY_NO_DISPATCH", "must remain closed")
    fixture = Path(doc["fixture_path"])
    require(hashlib.sha256(fixture.read_bytes()).hexdigest() == doc["fixture_sha256"], "fixture hash")
    tokens = json.loads(fixture.read_text())["tokens"]
    expected = [
        ("A", 1033, 0, 0, 0, 33, 0),
        ("A", 1033, 0, 0, 32, 257, 1024),
        ("A", 1033, 0, 0, 256, 1033, 1024),
        ("B", 257, 1, 1, 0, 257, 0),
        ("C", 33, 0, 1, 0, 33, 0),
        ("C", 65, 0, 1, 32, 65, 32),
    ]
    require(len(doc["phases"]) == 6, "phase count")
    calls = pages = layer_commands = 0
    seen_uuid = set()
    for p, contract in zip(doc["phases"], expected):
        key, n, src, dst, lo, hi, reused = contract
        command, passive = p["source_command"], p["passive_command"]
        require(
            (
                p["fixture"],
                p["valid_prompt_tokens"],
                p["source_slot"],
                p["destination_slot"],
                command["from"],
                command["to"],
                p["reused_tokens"],
            )
            == contract,
            "phase contract",
        )
        require(command["tokens"] == tokens[key][:n], "token identity")
        require(command["uuid"] not in seen_uuid, "generation UUID duplicate")
        seen_uuid.add(command["uuid"])
        require(command["op"] == ("remount" if reused else "register"), "admission kind")
        require(
            passive["uuid"] == command["uuid"]
            and passive["slot"] == dst
            and (passive["from"], passive["to"]) == (lo, hi),
            "passive mapping/range",
        )
        require(passive["expected_reused"] == max(0, min(reused, hi) - lo), "source reuse metadata")
        require(p["expected_completion_end"] == hi, "native completion endpoint")
        require(p["selected_token_count"] == hi - lo, "selected token count")
        require(0 <= lo < hi <= n <= 2048 and lo % 32 == 0, "logical range")
        require(reused < n and reused % 32 == 0, "nonempty remount tail")
        expected_calls = [
            {"slot": src, "request_id": command["request_id"], "begin": b, "end": min(b + 1024, n)}
            for b in range(reused, n, 1024)
        ]
        require(p["compute_calls"] == expected_calls, "compute sequence")
        calls += len(expected_calls)
        page_positions = range(lo // 32 * 32, ((hi + 31) // 32) * 32, 32)
        require(len(page_positions) == p["selected_pages_per_config_layer"], "page accounting")
        pages += len(page_positions) * 16 * 32
        for index, chunk in enumerate(expected_calls):
            ranges = [(chunk["begin"], chunk["end"])]
            if index == 0 and reused:
                ranges.insert(0, (0, reused))
            layer_commands += 32 * sum(max(a, lo) < min(b, hi) for a, b in ranges)
        for field, slot in [
            ("retire_source_after_verified_landing", src),
            ("retire_destination_after_verified_readback", dst),
        ]:
            require(
                p[field]["op"] == "retire" and p[field]["uuid"] == command["uuid"] and p[field]["slot"] == slot,
                "retirement identity",
            )
    require(
        doc["between_B_and_C"] == {"op": "reclaim", "id": 1039, "slot": 0, "uuid": 102},
        "third prompt must reclaim retired A generation",
    )
    require(
        tokens["C"][:32] != tokens["A"][:32] and tokens["C"][:32] != tokens["B"][:32],
        "C must not accidentally remount A or B",
    )
    totals = {
        "generations": 6,
        "compute_calls": calls,
        "layer_acks": calls * 32,
        "native_layer_commands": layer_commands,
        "selected_page_visits": pages,
        "selected_packed_bytes": pages * 4352,
    }
    require(totals == doc["expected"], "independent totals")
    return totals


if __name__ == "__main__":
    print(json.dumps(validate(json.loads(Path(__file__).with_name("scenario.json").read_text())), indent=2))
