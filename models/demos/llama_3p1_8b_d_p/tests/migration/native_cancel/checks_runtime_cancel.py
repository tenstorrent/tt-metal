"""Focused host faults for real runtime cancellation; no native imports."""

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

from runtime_cancel import produce_real_chunk, prompt_pair, validate_real_chunk

RUNTIME = Path(__file__).resolve().parents[3] / "tt/tt_prefill_runtime.py"
GEOMETRY = RUNTIME.with_name("prefill_geometry.py")
ns = {}
exec(compile(ast.parse(GEOMETRY.read_text()), str(GEOMETRY), "exec"), ns)


def integer(name, value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name)


ns["integer"] = integer
node = next(
    n for n in ast.parse(RUNTIME.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "TtPrefillRuntime"
)
exec(compile(ast.Module(body=[node], type_ignores=[]), str(RUNTIME), "exec"), ns)
Runtime = ns["TtPrefillRuntime"]
GROUPS = [f"{c}:{l}" for c in range(16) for l in range(32)]


class RealRuntimeContract(unittest.TestCase):
    def make(self, *, write=True, sync_failure=False, send_failure=None, layers=32):
        state = {"value": "old", "pending": None, "events": [], "acks": [], "clock": 0}

        class Model:
            num_layers = layers
            max_seq_len = 2048

            def prefill_chunk(self, tokens, cache, **kw):
                state["events"].append("forward")
                state["pending"] = "new" if write else "old"

        def sync(mesh):
            state["events"].append("sync")
            if sync_failure:
                raise RuntimeError("sync fault")
            state["value"] = state["pending"]

        cfg = SimpleNamespace(num_layers=layers, max_seq_len=2048, num_users=2, first_layer_idx=0)
        runtime = Runtime("mesh", config=cfg, model=Model(), synchronize=sync, upload=lambda *_: None)
        runtime.compiled = True
        cache = SimpleNamespace(num_users=2, num_layers=32, max_seq_len=2048, sp=4)

        def capture(label):
            state["events"].append(label)
            return {key: state["value"] for key in GROUPS}

        def receive(ids, end):
            return dict(tokens=list(ids), metadata=object(), metadata_rows=[[0, 0, end]] * 32)

        def borrowed(packet):
            state["events"].append("borrowed")

        def send(layer, request):
            if layer == send_failure:
                raise RuntimeError("sink fault")
            state["acks"].append((layer, request))
            state["events"].append("ack")

        def clock():
            state["clock"] += 1
            return state["clock"]

        def run():
            return produce_real_chunk(
                runtime,
                cache,
                epoch="b",
                nonce="b" * 32,
                request_id=1,
                ids=[19] * 32,
                receive=receive,
                check_borrowed=borrowed,
                capture=capture,
                push=send,
                clock=clock,
            )

        return runtime, state, run

    # Queued writes become visible only at the production runtime's synchronize, and
    # its first layer callback must save those bytes before any real-channel publish.
    def test_actual_runtime_orders_write_sync_capture_and_acks(self):
        runtime, state, run = self.make()
        row = run()
        self.assertEqual(state["events"][:4], ["prewrite", "forward", "sync", "postwrite"])
        self.assertEqual(state["acks"], [(x, 1) for x in range(32)])
        self.assertEqual(row["source_after"], {k: "new" for k in GROUPS})
        validate_real_chunk(row, epoch="b", nonce="b" * 32, ids=[19] * 32)

    # Reusing a valid epoch-A page map while claiming epoch B must fail before ack0.
    def test_exact_stale_epoch_a_snapshot_is_rejected(self):
        runtime, state, run = self.make(write=False)
        with self.assertRaisesRegex(ValueError, "valid source write"):
            run()
        self.assertEqual(state["acks"], [])
        self.assertTrue(runtime._failed)

    # A failed wait cannot certify partially written device memory.
    def test_sync_failure_has_no_snapshot_or_ack(self):
        runtime, state, run = self.make(sync_failure=True)
        with self.assertRaisesRegex(RuntimeError, "sync fault"):
            run()
        self.assertNotIn("postwrite", state["events"])
        self.assertEqual(state["acks"], [])

    # One publish failure preserves the runtime error state and never resumes a later layer.
    def test_ack_failure_stops_and_poison_runtime(self):
        runtime, state, run = self.make(send_failure=3)
        with self.assertRaisesRegex(RuntimeError, "sink fault"):
            run()
        self.assertEqual(state["acks"], [(0, 1), (1, 1), (2, 1)])
        self.assertTrue(runtime._failed)

    # The receipt cannot pass if a wrong stack emitted fewer than32 completed layer callbacks.
    def test_missing_layer_is_rejected(self):
        runtime, state, run = self.make(layers=31)
        with self.assertRaisesRegex(ValueError, "32"):
            run()

    # A stale nonce, token hash, page map, or timing cannot relabel an old snapshot as the restart.
    def test_offline_receipt_rejects_identity_and_order_mutants(self):
        _, _, run = self.make()
        row = run()
        for field, value in [
            ("run_nonce", "a" * 32),
            ("epoch", "a"),
            ("request_id", 0),
            ("source_after", row["source_before"]),
            ("captured_ns", row["synchronized_ns"] - 1),
            ("acks", row["acks"][:-1]),
            ("acks", row["acks"] + [row["acks"][-1]]),
        ]:
            with self.subTest(field=field):
                bad = copy.deepcopy(row)
                bad[field] = value
                with self.assertRaises(ValueError):
                    validate_real_chunk(bad, epoch="b", nonce="b" * 32, ids=[19] * 32)
        with self.assertRaises(ValueError):
            validate_real_chunk(row, epoch="b", nonce="b" * 32, ids=[17] * 32)

    # The registered2K request and restarted32-token request are exact distinct frozen input slices.
    def test_fixture_uses_real_distinct_prompt_and_rejects_replay(self):
        source = [list(range(2048)), list(range(1, 2049))]
        fixture = {"tokens": {"A": source[0][:1033], "C": source[1][:65]}}
        a, b = prompt_pair({"slots": source}, fixture)
        self.assertEqual(a, source[0])
        self.assertEqual(b, source[1][:32])
        with self.assertRaises(ValueError):
            prompt_pair({"slots": source}, {"tokens": {"A": source[0][:1033], "C": source[0][:65]}})


if __name__ == "__main__":
    unittest.main(verbosity=2)
