"""Host-only warmup sequencing and stale-cache failure regressions."""

import unittest

import capacity_execution as c
import capacity_pages as p
import capacity_warmup as w


class Chunk:
    def __init__(self, tokens):
        self.tokens = tokens
        self.freed = False

    def deallocate(self, force):
        self.freed = True


class Runtime:
    def __init__(self, fail_at=None):
        self.compiled = True
        self._layer_completion_sink = None
        self._last_request_id = -1
        self._active = False
        self._failed = False
        self.calls = []
        self.inputs = []
        self.fail_at = fail_at

    def _check_ready(self):
        if self._active or self._failed:
            raise RuntimeError("not ready")

    def make_chunk_input(self, tokens, actual_start):
        value = Chunk(tokens)
        self.inputs.append(value)
        return value

    def _run(self, tokens, cache, slot, start, end, request, sink):
        self.calls.append((slot, start, end, request, sink))
        if len(self.calls) == self.fail_at:
            raise RuntimeError("warmup failed")


def fixture():
    length = 4096
    phases = []
    tokens = {}
    for slot in (0, 1):
        end = length - 32 * slot
        name = str(slot)
        tokens[name] = [(x + slot * 17) % 128256 for x in range(end)]
        phases.append(
            dict(
                fixture=name,
                compute_calls=[dict(slot=slot, begin=x, end=min(x + 1024, end)) for x in range(0, end, 1024)],
            )
        )
    return dict(capacity=length, phases=phases), tokens


class WarmupTests(unittest.TestCase):
    # Warm every real width and both physical slots, without consuming serving sequence IDs.
    def test_complete_distinct_geometry_and_no_native_acks(self):
        doc, tokens = fixture()
        runtime = Runtime()
        events = []
        receipt = w.warmup_geometry(runtime, object(), doc, tokens, lambda: None, events.append)
        self.assertEqual(receipt["full32_calls"], 8)
        self.assertEqual(
            runtime.calls,
            [(v["slot"], v["begin"], v["end"], None, None) for phase in doc["phases"] for v in phase["compute_calls"]],
        )
        self.assertEqual(runtime._last_request_id, -1)
        self.assertTrue(all(value.freed for value in runtime.inputs))
        self.assertTrue(
            all(row["native_acks"] == 0 and row["prompt_sha256"] != row["real_prompt_sha256"] for row in events)
        )
        self.assertFalse(runtime._active)
        self.assertEqual(c.resources(65536)["total_full32_calls"], 258)

    # A partial warmup fails the runtime and releases only its owned input; no barrier is published.
    def test_failure_prevents_barrier_and_marks_runtime_failed(self):
        doc, tokens = fixture()
        runtime = Runtime(fail_at=3)
        published = []

        def run():
            return w.warmup_geometry(runtime, None, doc, tokens, lambda: None, lambda row: None)

        with self.assertRaisesRegex(RuntimeError, "warmup failed"):
            w.warmup_barrier("source", run, lambda *a, **kw: published.append(a), lambda *a, **kw: None, 1200)
        self.assertEqual(published, [])
        self.assertTrue(runtime._failed)
        self.assertFalse(runtime._active)
        self.assertTrue(all(value.freed for value in runtime.inputs))

    # Source cannot leave the barrier before peer observation; passive cannot leave before source completion.
    def test_both_roles_wait_before_bridge_construction(self):
        for role in ("source", "passive"):
            events = []

            def run():
                events.append("run")
                return {"full32_calls": 8}

            def publish(label, **fields):
                events.append(label)

            def wait(label, timeout):
                events.append((label, timeout))
                return {"warmup": {"full32_calls": 8}}

            w.warmup_barrier(role, run, publish, wait, 1700)
            events.append("construct bridge")
            self.assertEqual(
                events,
                (
                    ["run", "warmup-finished", ("warmup-observed", 1700), "construct bridge"]
                    if role == "source"
                    else [("warmup-finished", 1700), "warmup-observed", "construct bridge"]
                ),
            )

    # Installing a callback early must reject warmup instead of accidentally acknowledging synthetic data.
    def test_sink_or_busy_runtime_rejected(self):
        doc, tokens = fixture()
        for field, value in (("_layer_completion_sink", object()), ("_active", True), ("_failed", True)):
            runtime = Runtime()
            setattr(runtime, field, value)
            with self.assertRaises((RuntimeError, ValueError)):
                w.warmup_geometry(runtime, None, doc, tokens, lambda: None, lambda row: None)
            self.assertEqual(runtime.calls, [])

    # Every one of the512 config/layer groups must change; aggregate changed bytes are insufficient.
    def test_one_stale_or_missing_group_rejected(self):
        before = {str(c) + ":" + str(l): "a" * 64 for c in range(16) for l in range(32)}
        after = {key: "b" * 64 for key in before}
        self.assertEqual(p.require_changed_groups(before, after)["groups"], 512)
        after["15:31"] = before["15:31"]
        with self.assertRaises(ValueError):
            p.require_changed_groups(before, after)
        del after["15:31"]
        with self.assertRaises(ValueError):
            p.require_changed_groups(before, after)


if __name__ == "__main__":
    unittest.main()
