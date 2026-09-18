"""Deterministic regression for a bridge child that exits after its final journal append."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from runner_support import Bridge

CHILD = r"""
import json
from pathlib import Path
import sys
import time
journal, gate, payload, code, newline = sys.argv[1:]
while not Path(gate).exists():
    time.sleep(.001)
raw = payload.encode() + (b"\n" if newline == "1" else b"")
with Path(journal).open("ab", buffering=0) as stream:
    stream.write(raw)
    stream.flush()
    import os
    os.fsync(stream.fileno())
raise SystemExit(int(code))
"""


class BridgeTerminalTests(unittest.TestCase):
    def make_bridge(self, record, exit_code, newline=True):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        journal = output / "bridge.jsonl"
        journal.write_bytes(b"")
        gate = output / "release"
        child = subprocess.Popen(
            [
                sys.executable,
                "-c",
                CHILD,
                str(journal),
                str(gate),
                json.dumps(record, separators=(",", ":")),
                str(exit_code),
                "1" if newline else "0",
            ]
        )
        self.addCleanup(lambda: child.poll() is None and child.kill())
        bridge = Bridge.__new__(Bridge)
        bridge.check = lambda: None
        bridge.output = output
        bridge.journal = journal
        bridge.process = child
        bridge.sequence = 7
        original_rows = bridge.rows
        first = True

        def stale_first_read():
            nonlocal first
            rows = original_rows()
            if first:
                first = False
                gate.touch()
                child.wait(timeout=5)
            return rows

        bridge.rows = stale_first_read
        return bridge, output

    # A clean child may append the matching final reply and exit between the first journal read and poll.
    def test_clean_exit_final_reply_race_is_accepted(self):
        bridge, output = self.make_bridge({"reply": 7, "op": "drain"}, 0)
        rows = bridge.wait(lambda value: any(row.get("reply") == 7 for row in value), terminal=True)
        self.assertEqual(rows[-1]["op"], "drain")
        self.assertEqual(bridge.terminal_exit_code, 0)
        self.assertEqual(json.loads((output / "bridge-exit.json").read_text())["exit_code"], 0)

    # Exit zero without the requested reply is still a protocol failure, with the actual code preserved.
    def test_clean_exit_without_reply_is_rejected(self):
        bridge, output = self.make_bridge({"event": "unrelated"}, 0)
        with self.assertRaisesRegex(RuntimeError, "requested phase.*exit_code=0"):
            bridge.wait(lambda value: any(row.get("reply") == 7 for row in value), terminal=True)
        self.assertEqual(json.loads((output / "bridge-exit.json").read_text())["exit_code"], 0)

    # A matching reply cannot conceal a nonzero child exit.
    def test_nonzero_exit_with_reply_is_rejected(self):
        bridge, output = self.make_bridge({"reply": 7, "op": "drain"}, 3)
        with self.assertRaisesRegex(RuntimeError, "exit_code=3"):
            bridge.wait(lambda value: any(row.get("reply") == 7 for row in value), terminal=True)
        self.assertEqual(json.loads((output / "bridge-exit.json").read_text())["exit_code"], 3)

    # An explicit journal error remains terminal and records the child status used for diagnosis.
    def test_error_row_is_rejected_with_exit_code(self):
        bridge, output = self.make_bridge({"event": "error", "message": "boom"}, 1)
        with self.assertRaisesRegex(RuntimeError, "error.*exit_code=1"):
            bridge.wait(lambda value: any(row.get("reply") == 7 for row in value), terminal=True)
        self.assertEqual(json.loads((output / "bridge-exit.json").read_text())["exit_code"], 1)

    # A torn final append is not a reply and must not be accepted merely because the child exited zero.
    def test_incomplete_final_line_is_rejected(self):
        bridge, output = self.make_bridge({"reply": 7, "op": "drain"}, 0, newline=False)
        with self.assertRaisesRegex(RuntimeError, "requested phase.*exit_code=0"):
            bridge.wait(lambda value: any(row.get("reply") == 7 for row in value), terminal=True)
        self.assertEqual(json.loads((output / "bridge-exit.json").read_text())["complete_rows"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
