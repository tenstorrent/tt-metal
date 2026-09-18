"""Real callback use at the scenario seam; native I/O itself remains unrun."""

import unittest

from cancel_restart_scenario import run_restart_epoch


class ScenarioRuntimeTests(unittest.TestCase):
    # A source snapshot made before produce would certify stale epoch-A data.
    # The actual choreography must use the callback's post-write map and must
    # never call the passive pre-transfer snapshot callback on the source.
    def test_restart_oracle_is_returned_after_real_production(self):
        events = []
        published = {}
        tokens = list(range(32))
        fresh = {str(x): "new" for x in range(512)}
        row = dict(source_after=fresh, epoch="b")

        class Bridge:
            def rpc(self, op, **kw):
                events.append(op)
                if op == "register":
                    self_tokens = kw["tokens"]
                    assert self_tokens == tokens
                    return {"slot": 0}
                return {}

            def snapshot_until(self, predicate):
                events.append("snapshot")
                v = {"generations": [{"uuid": 701, "successful": True}], "calls": []}
                assert predicate(v)
                return v

            def drain(self):
                events.append("drain")
                return {"bridge_exit_code": 0}

        def produce(count):
            events.append("real-produce")
            self.assertEqual(count, 32)
            return row

        def pages(label):
            self.fail("source snapshot must be from its post-write callback")

        run_restart_epoch(
            "source",
            Bridge(),
            lambda name, **fields: published.update({name: fields}),
            lambda name: {},
            produce,
            pages,
            lambda: {"new_nonce": "b", "old_nonce": "a"},
            lambda: None,
            tokens=tokens,
        )
        receipt = published["restart-terminal"]["receipt"]
        self.assertIs(receipt["source_before"], fresh)
        self.assertIs(receipt["runtime_receipt"], row)
        self.assertEqual(receipt["real_layer_acks"], 32)
        self.assertEqual(receipt["synthetic_acks"], 0)
        self.assertEqual(events, ["register", "prepare", "real-produce", "snapshot", "drain"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
