import json
import unittest
from pathlib import Path
from observe import fresh, observe, date

START = "2026-11-06T20:00:00Z"
END = "2026-11-06T20:00:02Z"


def body(rid):
    return {"id": rid, "choices": [{"message": {"content": None, "reasoning": "private"}, "finish_reason": "length"}]}


def logs():
    s = "2026-11-06T19:59:00Z EVAL_BOUNDARY\n"
    for i in range(4):
        s += f"2026-11-06T19:59:01Z Received request r{i}: params\n"
    for i in range(4):
        s += "2026-11-06T19:59:02Z response_body={" + json.dumps(body(f"r{i}")) + "}\n"
    return s.encode()


class Tests(unittest.TestCase):
    def test_hex_ci_filter_and_partial_append(self):
        import logging, sys, subprocess

        sys.path.insert(0, str(Path.cwd()))
        from models.autoports.google_gemma_4_31b_it.tt.native_log_formatter import NativeHexFormatter

        formatter = NativeHexFormatter()
        stamp = "2026-11-06T21:39:06.4653537Z"
        rid = "request-InFo-id"
        reply = body(rid)
        reply["choices"][0]["message"]["reasoning"] = "雪 INFO\nmultiline"
        messages = [
            ("vllm.entrypoints.serve.utils.request_logger", "Received request " + rid + ": params"),
            (
                "vllm.entrypoints.serve.middleware.log_response",
                "response_body={" + json.dumps(reply, ensure_ascii=False, indent=2) + "}",
            ),
        ]
        records = [
            formatter.format(logging.LogRecord(name, logging.INFO, "", 0, msg, (), None)) for name, msg in messages
        ]
        raw = "BOUNDARY\n" + "\n".join(records) + "\n"
        filtered = subprocess.run(
            ["bash", "-o", "pipefail", "-c", 'grep -v -i INFO | sed "s/^/[container] /"'],
            input=raw,
            text=True,
            capture_output=True,
            check=True,
        ).stdout
        blob = ("\n".join(stamp + " " + line for line in filtered.splitlines()) + "\n").encode()
        state = fresh(1, "BOUNDARY", 2026)
        partial = observe(state, blob[:-20], stamp, "2026-11-06T21:39:07.4653537Z")
        self.assertFalse(partial["all_initial_server_replies_observed"])
        result = observe(state, blob, stamp, "2026-11-06T21:39:07.4653537Z")
        self.assertTrue(result["all_initial_server_replies_observed"])
        self.assertEqual(result["initial_requests"][0]["id"], rid)
        self.assertEqual(result["initial_requests"][0]["timestamp"], stamp)
        self.assertEqual(result["no_score_currently"], "unknown")
        self.assertEqual(
            result["initial_requests"][0]["server_response_records"][0]["json_sha256"],
            __import__("hashlib").sha256(json.dumps(reply, ensure_ascii=False, indent=2).encode()).hexdigest(),
        )
        bad = observe(fresh(1, "BOUNDARY", 2026), b"BOUNDARY\nTTNATIVEHEX1 broken\n", START, END)
        self.assertTrue(bad["source_coverage_compromised"])
        self.assertFalse(bad["all_initial_server_replies_observed"])

    def test_ci_fraction_precision(self):
        stamp = "2026-11-06T21:39:06.4653537Z"
        self.assertEqual(date(stamp).microsecond, 465353)
        for digits in ("4", "46", "465", "4653", "46535", "465353", "4653537", "465353712"):
            self.assertEqual(date("2026-11-06T21:39:06." + digits + "Z").microsecond, int((digits + "000000")[:6]))
        self.assertEqual(date(stamp), date("2026-11-06T21:39:06.465353+00:00"))
        r = observe(fresh(1, "BOUNDARY", 2026), (stamp + " BOUNDARY\n").encode(), stamp, "2026-11-06T21:39:07.4653537Z")
        self.assertEqual(r["latest_log_event"]["timestamp"], stamp)
        self.assertEqual(r["last_event_lag_seconds"], 1)
        self.assertFalse(r["all_initial_server_replies_observed"])
        self.assertEqual(r["no_score_currently"], "unknown")
        with self.assertRaises(ValueError):
            date("2026-11-06T21:39:06.badZ")

    def test_complete_and_stale(self):
        s = fresh(4, "EVAL_BOUNDARY", 2026)
        r = observe(s, logs(), START, END)
        self.assertTrue(r["all_initial_server_replies_observed"])
        self.assertEqual(r["no_score_currently"], "unknown")
        self.assertEqual(r["last_event_lag_seconds"], 60)
        r2 = observe(s, logs(), START, "2026-11-06T20:10:02Z")
        self.assertEqual(r2["last_event_lag_seconds"], 660)
        self.assertEqual(len(r2["initial_requests"][0]["server_response_records"]), 1)

    def test_incremental_partial_line(self):
        s = fresh(4, "EVAL_BOUNDARY", 2026)
        b = logs()
        r = observe(s, b[:-20], START, END)
        self.assertFalse(r["all_initial_server_replies_observed"])
        r = observe(s, b, START, END)
        self.assertTrue(r["all_initial_server_replies_observed"])

    def test_split_utf8(self):
        s = fresh(1, "EVAL_BOUNDARY", 2026)
        b = (
            "EVAL_BOUNDARY\nReceived request r0: params\n".encode()
            + ("response_body={" + json.dumps(body("r0"), ensure_ascii=False).replace("private", "é") + "}\n").encode()
        )
        split = b.index("é".encode()) + 1
        observe(s, b[:split], START, END)
        self.assertTrue(observe(s, b, START, END)["all_initial_server_replies_observed"])

    def test_multiline_ci_prefix(self):
        b = "2026-11-06T19:59:00Z EVAL_BOUNDARY\n2026-11-06T19:59:01Z Received request r0: params\n"
        lines = json.dumps(body("r0"), indent=2).splitlines()
        b += "2026-11-06T19:59:02Z response_body={" + lines[0] + "\n"
        b += "\n".join("2026-11-06T19:59:02Z " + x for x in lines[1:]) + "}\n"
        r = observe(fresh(1, "EVAL_BOUNDARY", 2026), b.encode(), START, END)
        self.assertTrue(r["all_initial_server_replies_observed"])

    def test_truncation_and_next_logger(self):
        b = 'EVAL_BOUNDARY\nReceived request r0: params\nresponse_body={{"id":"r0",\noutput truncated\nINFO 11-06 19:59:04 other event\n'
        r = observe(fresh(1, "EVAL_BOUNDARY", 2026), b.encode(), START, END)
        self.assertFalse(r["all_initial_server_replies_observed"])
        self.assertTrue(r["source_coverage_compromised"])
        self.assertEqual(r["incomplete_response_records"], 1)

    def test_non_append_and_no_boundary(self):
        s = fresh(4, "EVAL_BOUNDARY", 2026)
        observe(s, logs(), START, END)
        with self.assertRaises(ValueError):
            observe(s, b"rotated\n", START, END)
        r = observe(fresh(4, "MISSING_BOUNDARY", 2026), logs(), START, END)
        self.assertFalse(r["initial_c_identified"])

    def test_client_progress_not_ack(self):
        b = (
            logs()
            + b'2026-11-06T19:59:03Z Requesting API: 10% 4/40\n2026-11-06T19:59:04Z "POST /v1/chat/completions HTTP/1.1" 200 OK\n'
        )
        r = observe(fresh(4, "EVAL_BOUNDARY", 2026), b, START, END)
        self.assertEqual(r["progress"]["completed"], 4)
        self.assertEqual(r["http_status_counts"], {"200": 1})
        self.assertTrue(all(x["client_acknowledgement"] == "unknown" for x in r["initial_requests"]))

    def test_actual_private_native_logs(self):
        root = Path("bringup/artifacts/first-agent-reply-observability/live")
        native = [json.loads((root / "four-probes" / f"{i}-response.json").read_text()) for i in range(4)]
        lines = (root / "server/server.log").read_bytes()
        first = min(lines.index(("Received request " + r["id"] + ":").encode()) for r in native)
        start = lines.rfind(b"\n", 0, first) + 1
        b = b"EVAL_BOUNDARY\n" + lines[start:]
        r = observe(fresh(4, "EVAL_BOUNDARY", 2026), b, START, END)
        self.assertEqual({x["id"] for x in r["initial_requests"]}, {x["id"] for x in native})
        self.assertTrue(r["all_initial_server_replies_observed"])
        self.assertTrue(all(x["server_response_records"][0]["visible_empty"] for x in r["initial_requests"]))
        # Assertions validate native logs without overwriting prior observation evidence.


if __name__ == "__main__":
    unittest.main()
