"""Import-free faults at the actual paired phase/runtime/page helper seams."""
import copy
import hashlib
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from edge_coverage import Call
from range_contract import AckRecorder, generation, phase_receipt, scenario
from range_driver import run_phases
from range_pages import PageEffect, source_progress_policy
from runner_support import CapturedCompletionSink
from runtime_call import execute_call
from transfer_contract import validate_plan

DOC = scenario(Path(__file__).with_name("scenario.json"))
NONCE = "a" * 32


def source_state(phase, retired=False, acks=224):
    command = phase["source_command"]
    uuid = command["uuid"]
    slot = phase["source_slot"]
    transfer = uuid + 1000
    calls = [
        dict(op="register", transfer=transfer, uuid=uuid, src=slot, **{"from": phase["reused_tokens"]}),
        dict(
            op="peer_ready",
            transfer=transfer,
            src=slot,
            dst=phase["destination_slot"],
            **{"from": command["from"], "to": command["to"]},
        ),
    ]
    ranges = {
        100: [(0, 33)],
        101: [(32, 257)],
        102: [(256, 1024), (1024, 1033)],
        103: [(0, 257)],
        104: [(0, 33)],
        105: [(32, 65)],
    }[uuid]
    # The fixed matrix supplies an independent expected wire-range order.
    for layer in range(32):
        for begin, end in ranges:
            calls.append(
                dict(
                    op="layer",
                    transfer=transfer,
                    src=slot,
                    dst=phase["destination_slot"],
                    layer=layer,
                    **{"from": begin, "to": end},
                )
            )
    calls.extend(
        [
            dict(op="seal", transfer=transfer),
            dict(
                op="completion",
                transfer=transfer,
                src=slot,
                status=0,
                tokens=command["to"],
                completion_end=command["to"],
            ),
        ]
    )
    row = dict(
        uuid=uuid,
        slot=slot,
        request_id=command["request_id"],
        prompt_len=phase["valid_prompt_tokens"],
        reused=phase["reused_tokens"],
        successful=True,
        retired=retired,
        chunk_count=len(phase["compute_calls"]),
        selected_token_count=command["to"] - command["from"],
        **{"from": command["from"], "to": command["to"]},
    )
    return dict(
        successful=True,
        retired=True,
        acks=acks,
        calls=calls,
        generations=[row],
        slots=[
            dict(
                slot=slot,
                position=phase["valid_prompt_tokens"],
                request_id=command["request_id"],
                pins=0,
                in_flight=0,
                pending=0,
                idle_resident=True,
                evict_pending=False,
            )
        ],
    )


def passive_state(phase, retired=False):
    command = phase["source_command"]
    return dict(
        inbound_generations=[
            dict(
                uuid=command["uuid"],
                slot=phase["destination_slot"],
                complete=True,
                status=0,
                tokens=command["to"],
                completion_end=command["to"],
                expected_reused=phase["passive_command"]["expected_reused"],
                peer_reused=phase["passive_command"]["expected_reused"],
                peer_reused_known=True,
                retired=retired,
                selected_token_count=command["to"] - command["from"],
                **{"from": command["from"], "to": command["to"]},
            )
        ]
    )


class FakeBridge:
    def __init__(self, role):
        self.role = role
        self.phase = None
        self.retired = False
        self.log = []
        self.acks = 0

    def rpc(self, op, **fields):
        self.log.append((op, fields))
        if op in ("register", "remount", "expect"):
            self.phase = next(p for p in DOC["phases"] if p["source_command"]["uuid"] == fields["uuid"])
            self.retired = False
            return dict(slot=self.phase["source_slot"], reused=self.phase["reused_tokens"])
        if op == "retire":
            self.retired = True
        if op == "snapshot":
            return self.state()
        return {}

    def state(self):
        return (
            source_state(self.phase, self.retired, self.acks)
            if self.role == "source"
            else passive_state(self.phase, self.retired)
        )

    def snapshot_until(self, predicate):
        value = self.state()
        if not predicate(value):
            raise RuntimeError("fake did not satisfy progress predicate")
        return value


class ContractTests(unittest.TestCase):
    # The copied fixture remains exactly six native generations and seven full32 runtime calls.
    def test_fixture_inventory(self):
        self.assertEqual((len(DOC["phases"]), sum(len(p["compute_calls"]) for p in DOC["phases"])), (6, 7))
        self.assertEqual(DOC["expected"]["layer_acks"], 224)

    # The template cannot be executed with absent assignments and unbuilt range binaries.
    def test_closed_plan(self):
        with self.assertRaises(RuntimeError):
            validate_plan(json.loads(Path(__file__).with_name("plan.example.json").read_text()))

    # Every phase's endpoint and independently stated wire ranges satisfy both role contracts.
    def test_all_six_generation_receipts(self):
        for phase in DOC["phases"]:
            generation(source_state(phase), phase, "source")
            generation(passive_state(phase), phase, "passive")

    # A source selected-width225 must not stand in for endpoint257.
    def test_source_endpoint_width_rejected(self):
        phase = DOC["phases"][1]
        value = source_state(phase)
        value["calls"][-1]["tokens"] = 225
        with self.assertRaises(RuntimeError):
            generation(value, phase, "source")

    # The real passive completion uses endpoint65, not selected-width33.
    def test_passive_endpoint_width_rejected(self):
        phase = DOC["phases"][5]
        value = passive_state(phase)
        value["inbound_generations"][0]["completion_end"] = 33
        with self.assertRaises(RuntimeError):
            generation(value, phase, "passive")

    # Warm source reuse is the overlap225 even though the endpoint is257.
    def test_reuse_metadata_rejected(self):
        phase = DOC["phases"][1]
        value = passive_state(phase)
        value["inbound_generations"][0]["peer_reused"] = 257
        with self.assertRaises(RuntimeError):
            generation(value, phase, "passive")

    # Removing a native layer command cannot be hidden by a successful summary flag.
    def test_missing_layer_rejected(self):
        phase = DOC["phases"][2]
        value = source_state(phase)
        value["calls"].pop(6)
        with self.assertRaises(RuntimeError):
            generation(value, phase, "source")

    # A callback does not authorize retirement while the source pin remains live.
    def test_pending_source_pin_rejected(self):
        phase = DOC["phases"][0]
        value = source_state(phase)
        value["slots"][0]["pins"] = 1
        with self.assertRaises(RuntimeError):
            generation(value, phase, "source")

    # A receipt from another run or transfer UUID cannot advance the barrier.
    def test_stale_receipt_rejected(self):
        for bad in ({"run_nonce": "b" * 32}, {"uuid": 999}):
            value = dict(run_nonce=NONCE, role="passive", ok=True, uuid=100)
            value.update(bad)
            with self.assertRaises(RuntimeError):
                phase_receipt(value, DOC["phases"][0], NONCE, "passive")


class PageTests(unittest.TestCase):
    def source_checker(self, values):
        phase = DOC["phases"][5]
        return PageEffect("source", phase=phase, call=phase["compute_calls"][0], decode=lambda raw: values)

    def selected(self, checker, raw=b"1" * 4352):
        checker.count = 1
        checker.accept((0, 0, 0, 32), b"0" * 4352, raw)

    # Replayed A bytes may match the prior snapshot; finite nonzero non-sentinel rows still pass.
    def test_deterministic_replay_allowed(self):
        phase = DOC["phases"][1]
        values = [[2.0] * 128 for _ in range(9)] + [[0.0] * 128 for _ in range(23)]
        checker = PageEffect("source", phase=phase, call=phase["compute_calls"][0], decode=lambda raw: values)
        checker.count = 32
        checker.accept((0, 0, 0, 1024), b"1" * 4352, b"1" * 4352)

    # A selected all-zero row is not a valid structural model write.
    def test_zero_valid_row_rejected(self):
        with self.assertRaises(ValueError):
            self.selected(self.source_checker([[0.0] * 128 for _ in range(32)]))

    # Keeping the seeded baseline across a selected write is rejected.
    def test_seed_sentinel_rejected(self):
        with self.assertRaises(ValueError):
            self.selected(self.source_checker([[1 / 16] * 128 for _ in range(32)]))

    # C's final page has only one valid row; rows65..95 must be exactly zero.
    def test_nonzero_padding_rejected(self):
        checker = self.source_checker([[2.0] * 128 for _ in range(32)])
        checker.count = 2
        with self.assertRaises(ValueError):
            checker.accept((0, 0, 0, 64), b"0" * 4352, b"1" * 4352)

    # The source's other slot remains byte-identical during C's slot0 continuation.
    def test_other_slot_rejected(self):
        checker = self.source_checker([])
        checker.count = 32768
        with self.assertRaises(RuntimeError):
            checker.accept((0, 1, 0, 0), b"0" * 4352, b"1" * 4352)

    # A whole copied boundary page may contain valid source rows beyond the selected endpoint.
    def test_full_destination_inventory_and_boundary_page(self):
        phase = DOC["phases"][0]
        checker = PageEffect("passive", phase=phase)
        for slot in (0, 1):
            for config in range(16):
                for layer in range(32):
                    for position in range(0, 2048, 32):
                        key = (config, slot, layer, position)
                        before = hashlib.sha256(repr(key).encode()).digest() * 136
                        selected = slot == 0 and position < 64
                        source = hashlib.sha256(("source" + repr(key)).encode()).digest() * 136
                        checker.accept(key, before, source if selected else before, source if selected else None)
        result = checker.finish()
        self.assertEqual((result["selected_pages"], result["untouched_pages"]), (1024, 64512))

    # A valid page from another layer cannot substitute for the exact selected source coordinate.
    def test_wrong_source_page_rejected(self):
        checker = PageEffect("passive", phase=DOC["phases"][0])
        with self.assertRaises(RuntimeError):
            checker.accept((0, 0, 0, 0), b"0" * 4352, b"1" * 4352, b"2" * 4352)

    # A page at ceil32(end) is an untouched suffix, not part of the transfer's boundary tile.
    def test_suffix_rejected(self):
        checker = PageEffect("passive", phase=DOC["phases"][0])
        checker.count = 2
        with self.assertRaises(RuntimeError):
            checker.accept((0, 0, 0, 64), b"0" * 4352, b"1" * 4352)


class DriverTests(unittest.TestCase):
    def run_role(self, role, *, bad_hash=False, fail_bytes=False):
        bridge = FakeBridge(role)
        sent = []
        calls = []

        def publish(name, **fields):
            sent.append((name, fields))

        def wait(name):
            uuid = int(name.split("-")[1])
            phase = next(p for p in DOC["phases"] if p["source_command"]["uuid"] == uuid)
            value = dict(run_nonce=NONCE, role="passive" if role == "source" else "source", ok=True, uuid=uuid)
            if name.endswith("bytes-verified"):
                if fail_bytes:
                    raise RuntimeError("peer exact bytes failed")
                value["source_snapshot_sha256"] = "wrong" if bad_hash else str(uuid)
            if name.endswith("source-done"):
                value.update(terminal=source_state(phase), snapshot={"receipt_sha256": str(uuid)})
            return value

        def call(phase, row, ordinal):
            calls.append((phase["name"], row, ordinal))
            bridge.acks += 32
            return {"receipt_sha256": str(phase["source_command"]["uuid"])}

        def verify(phase, row):
            if fail_bytes:
                raise RuntimeError("exact destination mismatch")
            return dict(source_snapshot_sha256=row["receipt_sha256"])

        self.bridge = bridge
        self.calls = calls
        return run_phases(role, DOC, NONCE, bridge, publish, wait, call, verify, lambda: None), sent

    # Source computes seven calls, retires six generations, and reclaims A only after the fourth retirement.
    def test_source_order_and_reclaim(self):
        self.run_role("source")
        ops = [op for op, _ in self.bridge.log]
        self.assertEqual(len(self.calls), 7)
        self.assertEqual(ops.count("retire"), 6)
        self.assertEqual(ops[: ops.index("reclaim")].count("retire"), 4)

    # Passive verification precedes every explicit destination retirement.
    def test_passive_six_landings(self):
        result, sent = self.run_role("passive")
        self.assertEqual(len(result[0]), 6)
        self.assertEqual(sum(name.endswith("bytes-verified") for name, _ in sent), 6)

    # A peer byte failure leaves the source generation mounted and unreclaimed.
    def test_source_byte_failure_blocks_retire(self):
        with self.assertRaises(RuntimeError):
            self.run_role("source", fail_bytes=True)
        self.assertNotIn("retire", [x[0] for x in self.bridge.log])

    # Local destination mismatch prevents passive-slot retirement and later reuse.
    def test_passive_byte_failure_blocks_retire(self):
        with self.assertRaises(RuntimeError):
            self.run_role("passive", fail_bytes=True)
        self.assertNotIn("retire", [x[0] for x in self.bridge.log])

    # Source retirement requires readback of this generation's immutable snapshot.
    def test_wrong_snapshot_hash_blocks_retire(self):
        with self.assertRaises(RuntimeError):
            self.run_role("source", bad_hash=True)
        self.assertNotIn("retire", [x[0] for x in self.bridge.log])


class RuntimeTests(unittest.TestCase):
    def setup_call(self, fail=None):
        call = Call("A", "A", 0, 0, 1024)
        recorder = AckRecorder([dict(slot=0, begin=0, end=1024)])
        events = []
        borrowed = object()
        meta = object()

        def capture(request):
            events.append("capture")
            if fail == "capture":
                raise RuntimeError("capture failed")
            return {}

        def push(layer, request):
            events.append(("ack", layer))
            if fail == "push":
                raise RuntimeError("push failed")

        sink = CapturedCompletionSink(recorder, capture, push, clock=lambda: 3)

        def forward(tokens, cache, **kwargs):
            self.assertIs(tokens, borrowed)
            self.assertIs(kwargs["metadata_msg"], meta)
            events.append("forward")
            if fail == "forward":
                raise RuntimeError("forward failed")
            if fail == "sync":
                raise RuntimeError("sync failed")
            recorder.synchronized(2)
            events.append("sync")
            for layer in range(32):
                sink(layer, 0)

        runtime = SimpleNamespace(prefill_chunk=forward)
        bridge = SimpleNamespace(snapshot_until=lambda predicate: {"retired": True, "acks": 32})
        packet = dict(tokens=borrowed, metadata=meta, metadata_rows=[[0, 0, 1024] for _ in range(32)], began_ns=1)
        return call, recorder, runtime, bridge, packet, events

    # The borrowed object reaches the runtime unchanged; snapshot completes after sync and before32 callbacks.
    def test_real_call_seam_order(self):
        call, rec, runtime, bridge, packet, events = self.setup_call()
        execute_call(runtime, None, rec, bridge, call, 0, packet, lambda *_: None, lambda: None)
        self.assertEqual(events[:4], ["forward", "sync", "capture", ("ack", 0)])
        self.assertEqual(len(rec.rows), 32)

    # Forward and synchronization failures never publish readiness.
    def test_forward_sync_failures_no_acks(self):
        for fault in ("forward", "sync"):
            call, rec, runtime, bridge, packet, events = self.setup_call(fault)
            with self.assertRaises(RuntimeError):
                execute_call(runtime, None, rec, bridge, call, 0, packet, lambda *_: None, lambda: None)
            self.assertFalse(rec.rows)
            self.assertFalse(any(isinstance(e, tuple) for e in events))

    # An incomplete packed capture cannot make even the first layer ready.
    def test_capture_failure_no_acks(self):
        call, rec, runtime, bridge, packet, events = self.setup_call("capture")
        with self.assertRaises(RuntimeError):
            execute_call(runtime, None, rec, bridge, call, 0, packet, lambda *_: None, lambda: None)
        self.assertFalse(rec.rows)

    # A failed real callback publisher does not finish the active call or report success.
    def test_push_failure_is_not_success(self):
        call, rec, runtime, bridge, packet, events = self.setup_call("push")
        with self.assertRaises(RuntimeError):
            execute_call(runtime, None, rec, bridge, call, 0, packet, lambda *_: None, lambda: None)
        self.assertIsNotNone(rec.active)
        self.assertEqual(len(rec.rows), 1)

    # Wrong metadata on any one of32 chips rejects before the model call.
    def test_metadata_mismatch_precedes_forward(self):
        call, rec, runtime, bridge, packet, events = self.setup_call()
        packet["metadata_rows"][31] = [1, 0, 1024]
        with self.assertRaises(ValueError):
            execute_call(runtime, None, rec, bridge, call, 0, packet, lambda *_: None, lambda: None)
        self.assertFalse(events)


class ReportTests(unittest.TestCase):
    def reports(self):
        source = dict(
            role="source",
            ok=True,
            owner_cleanup_complete=True,
            errors=[],
            cleanup_errors=[],
            manager_exit=0,
            native_transfer_tested=True,
            persistent_h2d_tested=True,
            model_executed=True,
            after_manager_shutdown=dict(pages=65536, bytes=285212672, packed_bytes_equal=True),
            phases=[],
            snapshots=[],
            requests=[],
            acks=[],
            published=[],
        )
        passive = copy.deepcopy(source)
        passive.update(role="passive", persistent_h2d_tested=False, model_executed=False)
        ordinal = 0
        for phase in DOC["phases"]:
            uuid = phase["source_command"]["uuid"]
            source["phases"].append(dict(uuid=uuid, terminal=source_state(phase), retired=source_state(phase, True)))
            passive["phases"].append(dict(uuid=uuid, terminal=passive_state(phase), retired=passive_state(phase, True)))
            passive["snapshots"].append({})
            for call in phase["compute_calls"]:
                policy = source_progress_policy(phase, call)
                source["snapshots"].append(
                    dict(
                        identity=dict(ordinal=ordinal, uuid=uuid, **call),
                        snapshot_complete_ns=2,
                        checks=dict(policy, valid_change_groups=policy["expected_valid_change_groups"]),
                    )
                )
                source["requests"].append(
                    dict(
                        ordinal=ordinal,
                        uuid=uuid,
                        **call,
                        routed_acks=32,
                        borrowed_input_preserved=True,
                        all32_metadata_equal=True,
                    )
                )
                for layer in range(32):
                    source["acks"].append(
                        dict(
                            request_id=ordinal,
                            slot=call["slot"],
                            start=call["begin"],
                            end=call["end"],
                            layer=layer,
                            synchronized_ns=1,
                            ack_ns=3,
                        )
                    )
                    source["published"].append(dict(request_id=ordinal, layer=layer, published_ns=4))
                ordinal += 1
        return source, passive

    # The final report requires all six generations and224 ordered callbacks; success flags alone are insufficient.
    def test_complete_report(self):
        from verify_ranges import check_reports

        source, passive = self.reports()
        self.assertEqual(check_reports(source, passive, DOC)["post_sync_acks"], 224)

    # Dropping a final callback leaves a failed inventory even if every summary says success.
    def test_missing_ack_rejected(self):
        from verify_ranges import check_reports

        source, passive = self.reports()
        source["acks"].pop()
        with self.assertRaises(RuntimeError):
            check_reports(source, passive, DOC)

    # A callback timestamp before its immutable snapshot cannot satisfy readiness evidence.
    def test_snapshot_order_rejected(self):
        from verify_ranges import check_reports

        source, passive = self.reports()
        source["snapshots"][0]["snapshot_complete_ns"] = 5
        with self.assertRaises(RuntimeError):
            check_reports(source, passive, DOC)

    # A success summary cannot omit the new-content progress proof recorded by the actual page checker.
    def test_missing_valid_progress_rejected(self):
        from verify_ranges import check_reports

        source, passive = self.reports()
        source["snapshots"][5]["checks"]["valid_change_groups"] = 0
        with self.assertRaises(RuntimeError):
            check_reports(source, passive, DOC)

    # Native exit alone cannot replace the full retained-cache readback after shutdown.
    def test_missing_post_shutdown_bytes_rejected(self):
        from verify_ranges import check_reports

        source, passive = self.reports()
        source.pop("after_manager_shutdown")
        with self.assertRaises(RuntimeError):
            check_reports(source, passive, DOC)

    # A cleanup error remains terminal failure even after the transport itself passed.
    def test_cleanup_failure_rejected(self):
        from verify_ranges import check_reports

        source, passive = self.reports()
        passive["cleanup_errors"] = ["mesh close failed"]
        with self.assertRaises(RuntimeError):
            check_reports(source, passive, DOC)


if __name__ == "__main__":
    unittest.main(verbosity=2)
