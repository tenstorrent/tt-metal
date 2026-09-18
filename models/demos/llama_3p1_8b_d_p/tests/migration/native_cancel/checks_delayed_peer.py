"""Named barrier tests; fake bridge I/O only, no clock-based readiness assumption."""

import unittest

from delayed_peer import check_unarmed, passive_arm, source_before_arm


def pending():
    return dict(
        acks=32,
        retired=True,
        calls=[dict(op="register", uuid=700, src=0)],
        generations=[dict(uuid=700, request_id=11, slot=0, prompt_len=2048, chunk_count=1, **{"from": 0, "to": 2048})],
        slots=[dict(slot=0, position=1024, pins=1, in_flight=0, pending=0)],
    )


def pages():
    return {f"{c}:{l}:{x}": "sentinel" for c in range(16) for l in range(32) for x in range(0, 1024, 32)}


class DelayedPeerTests(unittest.TestCase):
    # The actual reader may consume all32 acks before arm; only register appears on
    # the native audit. Every early peer-ready/layer/seal or lost source pin fails.
    def test_consumed_acks_without_native_layer_is_valid(self):
        check_unarmed(pending())
        for op in ("peer_ready", "layer", "seal", "cancel", "completion"):
            bad = pending()
            bad["calls"].append(dict(op=op))
            with self.subTest(op=op), self.assertRaises(ValueError):
                check_unarmed(bad)
        bad = pending()
        bad["slots"][0]["pins"] = 0
        with self.assertRaises(ValueError):
            check_unarmed(bad)

    # Use an actual native journal row. Main.cpp audit_json emits src, while
    # src_slot is the C++ member name and must not become a wire-field alias.
    def test_actual_native_register_serialization(self):
        import json
        from pathlib import Path

        row = json.loads((Path(__file__).parent / "native-register-schema.json").read_bytes())["observed_register"]
        row = dict(row, uuid=700)
        snapshot = pending()
        snapshot["calls"] = [row]
        check_unarmed(snapshot)
        wrong = dict(row)
        wrong["src_slot"] = wrong.pop("src")
        snapshot["calls"] = [wrong]
        with self.assertRaises(ValueError):
            check_unarmed(snapshot)

    # Source produces once while passive remains held, validates two snapshots,
    # and explicitly permits arm only after passive certifies unchanged bytes.
    def test_source_requires_delayed_receipt_before_arm_permission(self):
        events = []

        class Bridge:
            def snapshot_until(self, predicate):
                events.append("snapshot")
                row = pending()
                assert predicate(row)
                return row

            def rpc(self, op):
                events.append("rpc-" + op)
                return pending()

        def wait(name):
            events.append("wait-" + name)
            return {}

        def publish(name, **kw):
            events.append("publish-" + name)

        def produce(count):
            events.append("produce-" + str(count))
            return {"real": True}

        source_before_arm(Bridge(), produce, publish, wait, lambda: None)
        self.assertEqual(
            events,
            [
                "wait-delay-held",
                "produce-32",
                "snapshot",
                "publish-unarmed-complete",
                "wait-delay-verified",
                "rpc-snapshot",
                "publish-arm-permitted",
                "wait-cancel-armed",
            ],
        )

    # A source snapshot that already issued a layer cannot release the arm barrier.
    def test_source_rejects_early_native_layer(self):
        published = []

        class Bridge:
            def snapshot_until(self, predicate):
                row = pending()
                row["calls"].append({"op": "layer"})
                return row

        with self.assertRaises(ValueError):
            source_before_arm(
                Bridge(), lambda n: None, lambda name, **k: published.append(name), lambda name: {}, lambda: None
            )
        self.assertEqual(published, [])

    # Passive checks the whole intended first1K (16,384 pages) and cannot call
    # expect on a corrupted sentinel or before the named arm-permitted receipt.
    def test_passive_verifies_all_first_chunk_pages_before_expect(self):
        for corrupt in (False, True):
            events = []
            baseline = pages()

            class Bridge:
                def rpc(self, op, **kw):
                    events.append("rpc-" + op)
                    return {"calls": [], "inbound_generations": []}

            def capture(label):
                events.append(label)
                row = dict(baseline)
                if corrupt and label == "delay-after":
                    row["15:31:992"] = "changed"
                return row

            def publish(name, **k):
                events.append("publish-" + name)

            def wait(name):
                events.append("wait-" + name)
                return {"snapshot": pending()}

            if corrupt:
                with self.assertRaises(ValueError):
                    passive_arm(Bridge(), capture, publish, wait, lambda: None)
                self.assertNotIn("rpc-expect", events)
            else:
                passive_arm(Bridge(), capture, publish, wait, lambda: None)
                self.assertLess(events.index("delay-after"), events.index("publish-delay-verified"))
                self.assertLess(events.index("wait-arm-permitted"), events.index("rpc-expect"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
