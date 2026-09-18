"""Fault mutations of a complete small capacity evidence record; no device import."""

import unittest

from capacity_execution import make_cases, sentinel_keys
from capacity_warmup import token_digest
from checks_fixtures import token_manifest
from verify_capacity import verify_report


def terminal(phase, role, retired=False):
    spec = phase["source_command"]
    slot = phase["source_slot"]
    dst = phase["destination_slot"]
    uuid = spec["uuid"]
    row = dict(
        uuid=uuid,
        slot=slot if role == "source" else dst,
        **{"from": spec["from"], "to": spec["to"]},
        selected_token_count=1024,
        retired=retired,
    )
    if role == "passive":
        row.update(
            complete=True,
            status=0,
            tokens=spec["to"],
            completion_end=spec["to"],
            expected_reused=0,
            peer_reused=0,
            peer_reused_known=True,
        )
        return dict(inbound_generations=[row])
    row.update(
        successful=True,
        request_id=spec["request_id"],
        prompt_len=phase["valid_prompt_tokens"],
        reused=0,
        chunk_count=len(phase["compute_calls"]),
    )
    base = dict(transfer=uuid, src=slot, dst=dst)
    calls = [
        dict(base, op="register", uuid=uuid, **{"from": 0}),
        dict(base, op="peer_ready", **{"from": spec["from"], "to": spec["to"]}),
    ]
    for call in phase["compute_calls"]:
        begin = max(call["begin"], spec["from"])
        end = min(call["end"], spec["to"])
        if begin < end:
            calls.extend(dict(base, op="layer", layer=layer, **{"from": begin, "to": end}) for layer in range(32))
    calls.extend(
        [dict(base, op="seal"), dict(base, op="completion", status=0, tokens=spec["to"], completion_end=spec["to"])]
    )
    resident = dict(
        slot=slot,
        position=phase["valid_prompt_tokens"],
        request_id=spec["request_id"],
        pins=0,
        in_flight=0,
        pending=0,
        idle_resident=True,
        evict_pending=False,
    )
    return dict(generations=[row], slots=[resident], calls=calls)


def report(role):
    with token_manifest() as (path, digest):
        doc, _ = make_cases(4096, path, digest)
    warm = []
    requests = []
    captures = []
    phases = []
    before = {str(c) + ":" + str(l): "a" * 64 for c in range(16) for l in range(32)}
    after = {key: "b" * 64 for key in before}
    allcalls = []
    for phase in doc["phases"]:
        spec = phase["source_command"]
        tokens = spec["tokens"]
        for call in phase["compute_calls"]:
            i = len(requests)
            warm.append(
                dict(
                    slot=call["slot"],
                    begin=call["begin"],
                    end=call["end"],
                    native_acks=0,
                    real_prompt_sha256=token_digest(tokens),
                    prompt_sha256=token_digest([(x + 1) % 128256 for x in tokens]),
                )
            )
            requests.append(
                dict(call, ordinal=i, routed_acks=32, borrowed_input_preserved=True, all32_metadata_equal=True)
            )
            overlap = max(0, min(call["end"], spec["to"]) - max(call["begin"], spec["from"]))
            capture = dict(request=i, selected_pages=overlap // 32 * 512, snapshot_complete_ns=15 + 100 * i)
            if call["end"] == phase["valid_prompt_tokens"]:
                capture.update(selected={}, changed_from_pre_request=dict(before=before, after=after))
            captures.append(capture)
        raw = terminal(phase, role)
        allcalls += raw.get("calls", [])
        phases.append(
            dict(
                uuid=spec["uuid"],
                terminal=raw,
                retired=terminal(phase, role, True),
                readback=dict(
                    exact=True,
                    pages=16384,
                    untouched_samples=dict(unchanged_sha256=True, pages=len(sentinel_keys(4096, phase))),
                ),
            )
        )
    result = dict(
        ok=True,
        owner_cleanup_complete=True,
        errors=[],
        cleanup_errors=[],
        capacity=4096,
        manager_exit=0,
        bridge_exit_code=0,
        native_transfer_tested=True,
        model_executed=role == "source",
        persistent_h2d_tested=role == "source",
        compile_warmup_full32_calls=2 if role == "source" else 0,
        warmup_barrier_before_native_clients=True,
        capacity_warmup=dict(
            full32_calls=8, calls=warm, native_acks=0, serving_request_id_before=-1, serving_request_id_after=-1
        ),
        phases=phases,
        requests=requests,
        selected_captures=captures,
        terminal=dict(successful=True, calls=allcalls),
        selected_source_config_hashes=[{str(c): "a" for c in range(16)}, {str(c): "b" for c in range(16)}],
        after_manager_shutdown=[dict(exact=True, pages=16384)] * 2,
        final_untouched_samples=dict(
            unchanged_sha256=True,
            pages=len({k for phase in doc["phases"] for k in sentinel_keys(4096, phase, doc["phases"])}),
        ),
        manager_memory_after_tables=dict(rss_bytes=100, hwm_bytes=200, limit_bytes=300),
        manager_memory_after_transfer=dict(rss_bytes=200, hwm_bytes=250, limit_bytes=300),
        acks=[
            dict(request_id=i, layer=l, synchronized_ns=10 + 100 * i, ack_ns=20 + 100 * i + l)
            for i in range(8)
            for l in range(32)
        ],
        published=[dict(request_id=i, layer=l, published_ns=60 + 100 * i + l) for i in range(8) for l in range(32)],
    )
    return doc, result


class VerifierTests(unittest.TestCase):
    # Accept complete independent source/passive records, including exact per-range native audit sequences.
    def test_complete_record(self):
        for role in ("source", "passive"):
            doc, value = report(role)
            self.assertEqual(verify_report(value, role, doc)["capacity"], 4096)

    # Reject incomplete snapshots, stale warmup data, missing acknowledgements and native nonzero exits.
    def test_source_faults(self):
        edits = [
            lambda r: r["selected_captures"][-1].update(selected_pages=0),
            lambda r: r["selected_captures"][-1]["changed_from_pre_request"]["after"].update({"15:31": "a" * 64}),
            lambda r: r["acks"].pop(),
            lambda r: r.update(bridge_exit_code=2),
            lambda r: r["capacity_warmup"]["calls"][-1].update(prompt_sha256="wrong"),
            lambda r: r["manager_memory_after_transfer"].update(hwm_bytes=301),
        ]
        for edit in edits:
            doc, value = report("source")
            edit(value)
            with self.assertRaises((ValueError, RuntimeError)):
                verify_report(value, "source", doc)

    # Retired/untouched evidence must cover the actual destination, not only successful generation metadata.
    def test_passive_faults(self):
        edits = [
            lambda r: r["phases"][1]["readback"]["untouched_samples"].update(pages=0),
            lambda r: r["phases"][0]["retired"]["inbound_generations"][0].update(retired=False),
            lambda r: r["final_untouched_samples"].update(pages=0),
        ]
        for edit in edits:
            doc, value = report("passive")
            edit(value)
            with self.assertRaises((ValueError, RuntimeError)):
                verify_report(value, "passive", doc)


if __name__ == "__main__":
    unittest.main()
