"""Mock and stdlib coverage only. This suite must not import Torch or native modules."""
import copy
import datetime
import hashlib
import importlib.abc
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class NoNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("torch", "ttnn", "tt_lib", "tt_d_gen", "transformers"):
            raise RuntimeError("native/model import prohibited in host tests: " + fullname)


sys.meta_path.insert(0, NoNative())
from edge_checks import EffectCheck, check_values, drive_requests
from edge_coverage import PAGE_BYTES, RUNTIME_CALLS
from edge_guard import check_health, check_plan
from edge_snapshots import capture
from gate_contract import AckRecorder
from reference.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
from support import CapturedCompletionSink, write_json
from verify_report import verify

FIXTURES = json.loads(Path(__file__).with_name("fixtures.json").read_bytes())["tokens"]


class Model:
    num_layers = 32
    max_seq_len = 2048

    def __init__(self):
        self.calls = []

    def prefill_chunk(self, tokens, cache, **kwargs):
        self.calls.append((tokens, kwargs))
        return None


def mock_run(*, bad_metadata=False, capture_failure=False, sync_failure=False, push_failure=False):
    model = Model()
    recorder = AckRecorder()
    cache = SimpleNamespace(num_users=2, num_layers=32, max_seq_len=2048, sp=4)

    def sync(mesh):
        if sync_failure:
            raise RuntimeError("sync failed")
        recorder.synchronized(time.monotonic_ns())

    runtime = TtPrefillRuntime(
        object(),
        config=TtPrefillRuntimeConfig(2048, 1024, 2),
        model=model,
        synchronize=sync,
        upload=lambda *a, **k: (_ for _ in ()).throw(AssertionError("borrowed input must not use uploader")),
    )
    runtime.compiled = True
    packets = []
    records = []
    events = []
    routed = []

    def snapshot(request):
        events.append(("snapshot", request, len(recorder.rows)))
        if capture_failure:
            raise RuntimeError("capture failed")
        return dict(request=request)

    def push(layer, request):
        if push_failure:
            raise RuntimeError("sink failed")
        routed.append((request, layer))
        events.append(("push", request, layer))

    sink = CapturedCompletionSink(recorder, snapshot, push)
    runtime.set_layer_completion_sink(sink)

    def receive(call, tokens):
        assert len(tokens) == call.end - call.begin
        assert recorder.active is None
        packet = dict(
            tokens=object(),
            metadata=object(),
            metadata_rows=[[call.slot, call.begin, call.end] for _ in range(32)],
            began_ns=time.monotonic_ns(),
        )
        if bad_metadata:
            packet["metadata_rows"][31][0] = 1 - call.slot
        packets.append(packet)
        return packet

    def consume():
        n = len(routed)
        routed.clear()
        return n

    def check_input(packet, call):
        assert model.calls[-1][0] is packet["tokens"]

    try:
        drive_requests(
            runtime, cache, recorder, FIXTURES, receive, check_input, consume, lambda *args: records.append(args)
        )
        error = None
    except BaseException as exc:
        error = exc
    return SimpleNamespace(
        model=model,
        recorder=recorder,
        runtime=runtime,
        packets=packets,
        records=records,
        events=events,
        sink=sink,
        error=error,
    )


class RuntimeTests(unittest.TestCase):
    # The real runtime borrows all five H2D inputs and emits 160 ordered callbacks after synchronization and capture.
    def test_five_calls_use_actual_runtime_and_borrowed_identity(self):
        result = mock_run()
        self.assertIsNone(result.error)
        self.assertEqual(len(result.records), 5)
        self.assertEqual(len(result.recorder.rows), 160)
        self.assertEqual(len(result.sink.published), 160)
        for i, call in enumerate(RUNTIME_CALLS):
            self.assertIs(result.model.calls[i][0], result.packets[i]["tokens"])
            self.assertEqual(
                result.model.calls[i][1],
                dict(slot_idx=call.slot, actual_start=call.begin, actual_end=call.end, skip_lm_head=True),
            )
            self.assertEqual(result.events[i * 33], ("snapshot", i, i * 32))
        self.assertIsNone(result.recorder.active)

    # A wrong slot on even one metadata shard must stop before any model call or acknowledgement.
    def test_wrong_metadata_stops_before_dispatch(self):
        result = mock_run(bad_metadata=True)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.model.calls, [])
        self.assertEqual(result.recorder.rows, [])

    # A failed device synchronization cannot produce a snapshot or an acknowledgement.
    def test_sync_failure_has_no_snapshot_or_ack(self):
        result = mock_run(sync_failure=True)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.events, [])
        self.assertEqual(result.recorder.rows, [])
        self.assertTrue(result.runtime._failed)

    # A failed immutable capture suppresses the first acknowledgement and poisons the runtime generation.
    def test_capture_failure_has_no_ack(self):
        result = mock_run(capture_failure=True)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.recorder.rows, [])
        self.assertEqual(result.sink.published, [])
        self.assertTrue(result.runtime._failed)

    # A failed publication cannot masquerade as a completed request or permit slot reuse.
    def test_sink_failure_does_not_finish_or_reuse(self):
        result = mock_run(push_failure=True)
        self.assertIsNotNone(result.error)
        self.assertEqual(len(result.packets), 1)
        self.assertEqual(result.records, [])
        self.assertTrue(result.runtime._failed)

    # The recorder rejects acknowledgements with a wrong request or layer even after a synchronization.
    def test_wrong_callback_identity_rejected(self):
        recorder = AckRecorder()
        recorder.begin(0, 0, 0, 1024, 1)
        recorder.synchronized(2)
        for layer, request in ((1, 0), (0, 1)):
            with self.assertRaises(ValueError):
                recorder.ack(layer, request, 3)


class EffectTests(unittest.TestCase):
    # The overlapping C continuation changes pages32/64, checks padding65..96, and preserves every other page including slot1.
    def test_complete_continuation_inventory(self):
        call = RUNTIME_CALLS[4]
        old = b"a" * PAGE_BYTES
        full = b"b" * PAGE_BYTES
        tail = b"c" * PAGE_BYTES
        rows = [[1.0] * 127 + [0.5] for _ in range(32)]
        tailrows = [[1.0] * 127 + [0.5]] + [[0.0] * 128 for _ in range(31)]
        check = EffectCheck(call, lambda raw: tailrows if raw == tail else rows)
        for s in (0, 1):
            for c in range(16):
                for l in range(32):
                    for p in range(0, 2048, 32):
                        check.accept(
                            (c, s, l, p), old, (full if p == 32 else tail) if s == 0 and p in (32, 64) else old
                        )
        result = check.finish()
        self.assertEqual(result["changed_pages"], 1024)
        self.assertEqual(result["untouched_pages"], 64512)
        self.assertEqual(result["valid_values"], 33 * 128 * 512)
        self.assertEqual(result["padding_values"], 31 * 128 * 512)

    # Prefix, suffix, and the other slot remain byte-exact; each fault is tested at its real logical coordinate.
    def test_untouched_region_faults(self):
        for c, s, l, p in ((0, 0, 0, 0), (0, 0, 0, 96), (0, 1, 0, 0)):
            check = EffectCheck(RUNTIME_CALLS[4], lambda raw: [])
            check.count = ((s * 16 + c) * 32 + l) * 64 + p // 32
            with self.assertRaisesRegex(ValueError, "untouched"):
                check.accept((c, s, l, p), b"a" * PAGE_BYTES, b"b" * PAGE_BYTES)

    # Duplicate or missing stream positions cannot satisfy the complete-page inventory.
    def test_page_inventory_rejects_gaps_duplicates_and_short_capture(self):
        check = EffectCheck(RUNTIME_CALLS[4], lambda raw: [])
        raw = b"a" * PAGE_BYTES
        check.accept((0, 0, 0, 0), raw, raw)
        for key in ((0, 0, 0, 0), (0, 0, 0, 64)):
            with self.assertRaises(ValueError):
                check.accept(key, raw, raw)
        with self.assertRaises(ValueError):
            check.finish()

    # An unchanged selected page detects stale inputs or skipped writes for this fixed, distinct fixture sequence.
    def test_selected_old_contents_rejected(self):
        check = EffectCheck(RUNTIME_CALLS[4], lambda raw: [])
        check.count = 1
        raw = b"a" * PAGE_BYTES
        with self.assertRaisesRegex(ValueError, "old contents"):
            check.accept((0, 0, 0, 32), raw, raw)

    # Finite valid rows and exact zero padding are structural checks, with no new numerical tolerance.
    def test_decoded_nonfinite_padding_shape_and_seed_faults(self):
        rows = [[1.0] * 128] + [[0.0] * 128 for _ in range(31)]
        check_values(rows, valid_rows=1)
        for mutated in ([[float("nan")] * 128] + rows[1:], rows[:-1], [[1.0] * 128 for _ in range(32)]):
            with self.assertRaises(ValueError):
                check_values(mutated, valid_rows=1)
        with self.assertRaises(ValueError):
            check_values(rows, seed=2)
        with self.assertRaises(ValueError):
            check_values(rows, valid_rows=33)

    # Failed reads preserve the partial binary but never publish a completed immutable snapshot receipt.
    def test_capture_failure_preserves_partial_without_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "capture"
            with patch("edge_snapshots.geometry", return_value=2048), patch(
                "edge_snapshots.read_page", side_effect=RuntimeError("read failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "read failed"):
                    capture(object(), output, {}, lambda raw: [])
            self.assertTrue((output / "slot0.bin").exists())
            self.assertFalse((output / "snapshot.json").exists())

    # Completed receipts are immutable even when a caller tries to reuse the same path.
    def test_receipt_overwrite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            write_json(path, dict(value=1))
            with self.assertRaises(RuntimeError):
                write_json(path, dict(value=2))
            self.assertEqual(json.loads(path.read_bytes()), dict(value=1))


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.plan = dict(
            reviewed=True,
            dispatch_enabled=True,
            scope="2k_runtime_edges_without_manager",
            node="bh-glx-110-c10u14",
            job_id="109097",
            run_nonce="a" * 32,
            lock_path="/tmp/prefill-device-110-c10u14.lock",
        )
        self.env = dict(
            SLURM_JOB_ID="109097",
            SLURM_CPUS_PER_TASK="1",
            PREFILL_FABRIC_MODE="1d_ring",
            **{k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        )

    # The unarmed plan and any wrong endpoint, lock, thread budget, or dispatch mode fail before native imports.
    def test_closed_and_endpoint_environment_guards(self):
        check_plan(self.plan, self.env, self.plan["node"])
        for key, value in (
            ("reviewed", False),
            ("dispatch_enabled", False),
            ("node", "other"),
            ("lock_path", "other"),
            ("run_nonce", ""),
        ):
            with self.assertRaises(ValueError):
                check_plan(dict(self.plan, **{key: value}), self.env, self.plan["node"])
        for key, value in (
            ("SLURM_JOB_ID", "wrong"),
            ("MKL_NUM_THREADS", "2"),
            ("TT_METAL_SLOW_DISPATCH_MODE", "1"),
            ("PREFILL_FABRIC_MODE", ""),
        ):
            with self.assertRaises(ValueError):
                check_plan(self.plan, dict(self.env, **{key: value}), self.plan["node"])

    # A root health receipt must name this job/node, prove clean accepted completion, and still be fresh.
    def test_health_identity_success_and_freshness(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        row = dict(
            node=self.plan["node"],
            job_id=self.plan["job_id"],
            actual_exit=0,
            verified_exit=0,
            clean_close="yes",
            ended_utc=now.isoformat(),
        )
        check_health(row, self.plan, now)
        for key, value in (
            ("job_id", "wrong"),
            ("verified_exit", 90),
            ("clean_close", None),
            ("ended_utc", (now - datetime.timedelta(seconds=1501)).isoformat()),
        ):
            with self.assertRaises(ValueError):
                check_health(dict(row, **{key: value}), self.plan, now)


def synthetic_report():
    common = dict(
        run_nonce="a" * 32,
        table_sha256="table",
        device_map_sha256="map",
        cache_bases=[1, 2],
        fixture_sha256="fixture",
        plan_sha256="plan",
    )
    files = [dict(path=str(s), slot=s, begin=0, end=2048, pages=32768, bytes=142606336, sha256="bytes") for s in (0, 1)]
    base = dict(
        identity=dict(common, ordinal=-1),
        files=files,
        checks=dict(pages=65536, nonzero_pages=65536, seed_groups=1024),
        capture_finished_ns=1,
        receipt_path="base",
        receipt_sha256="base",
    )
    report = dict(
        gate_passed=True,
        owner_cleanup_complete=True,
        errors=[],
        cleanup_errors=[],
        mesh_devices=32,
        configs=16,
        table_entries=65536,
        page_bytes=4352,
        seed_writer_calls=128,
        warmup_full32_calls=2,
        run_nonce="a" * 32,
        baseline=base,
        requests=[],
        snapshots=[],
        acks=[],
        published=[],
    )
    for k in (
        "native_manager_tested",
        "native_source_pin_or_retirement_tested",
        "native_transfer_tested",
        "kv_golden_comparison_performed",
        "new_model_numerical_acceptance",
        "decoder_tested",
    ):
        report[k] = False
    report["recovery_required"] = False
    report["cleanup_attempts"] = [
        "service.drop",
        "service.collect",
        "channel.drop",
        "producer",
        "router",
        "saved",
        "cache.k",
        "cache.v",
        "model",
        "model.collect",
        "mesh.synchronize",
        "fabric.disable",
        "mesh.close",
    ]
    previous = "base"
    for request, call in enumerate(RUNTIME_CALLS):
        t = 2 + request * 100
        rounded = (call.end + 31) // 32 * 32
        selected = (rounded - call.begin) // 32 * 512
        ident = dict(
            common,
            ordinal=request,
            prompt=call.prompt,
            slot=call.slot,
            begin=call.begin,
            end=call.end,
            previous_snapshot_sha256=previous,
            token_ids_sha256=hashlib.sha256(
                json.dumps(FIXTURES[call.prompt][call.begin : call.end], separators=(",", ":")).encode()
            ).hexdigest(),
        )
        checks = dict(
            pages=65536,
            changed_pages=selected,
            untouched_pages=65536 - selected,
            configs=16,
            layers=32,
            slots=2,
            valid_values=(call.end - call.begin) * 128 * 512,
            padding_values=(rounded - call.end) * 128 * 512,
            semantic_valid_end=call.end,
            packed_end=rounded,
            golden_comparison=False,
            structural_write_checked=True,
        )
        snap = dict(
            identity=ident,
            files=copy.deepcopy(files),
            checks=checks,
            capture_finished_ns=t + 1,
            snapshot_complete_ns=t + 2,
            receipt_path=str(request),
            receipt_sha256=str(request),
        )
        previous = str(request)
        report["snapshots"].append(snap)
        report["requests"].append(
            dict(
                request_id=request,
                prompt=call.prompt,
                slot=call.slot,
                begin=call.begin,
                end=call.end,
                routed_acks=32,
                borrowed_input_preserved=True,
                all32_metadata_equal=True,
                native_retirement_checked=False,
                runtime_only_reuse=request in (3, 4),
            )
        )
        for layer in range(32):
            report["acks"].append(
                dict(
                    request_id=request,
                    slot=call.slot,
                    start=call.begin,
                    end=call.end,
                    layer=layer,
                    synchronized_ns=t,
                    ack_ns=t + 3 + layer * 2,
                )
            )
            report["published"].append(dict(request_id=request, layer=layer, published_ns=t + 4 + layer * 2))
    return report


def receipt_loader(summary):
    return {k: v for k, v in summary.items() if k not in ("receipt_path", "receipt_sha256", "snapshot_complete_ns")}


class ReportTests(unittest.TestCase):
    # The independent reducer accepts the exact five-call/160-ack inventory, without claiming transport or numerical accuracy.
    def test_exact_report_inventory(self):
        self.assertEqual(verify(synthetic_report(), FIXTURES, receipt_loader)["selected_pages"], 23552)

    # Missing/duplicate acknowledgements, wrong snapshot generation, or callback-before-capture ordering fail independently.
    def test_missing_duplicate_stale_and_early_ack_fail(self):
        for fault in ("missing", "duplicate", "stale", "early", "slot", "hidden"):
            row = synthetic_report()
            if fault == "missing":
                row["acks"].pop()
            if fault == "duplicate":
                row["acks"][1] = row["acks"][0]
            if fault == "stale":
                row["snapshots"][4]["identity"]["previous_snapshot_sha256"] = "stale"
            if fault == "early":
                row["acks"][128]["ack_ns"] = 0
            if fault == "slot":
                row["snapshots"][3]["identity"]["slot"] = 1
            if fault == "hidden":
                row["cleanup_errors"] = ["failure"]
            with self.assertRaises(RuntimeError, msg=fault):
                verify(row, FIXTURES, receipt_loader)

    # Missing snapshot slots and omitted padding coverage cannot pass using only aggregate request counts.
    def test_snapshot_slot_and_padding_coverage(self):
        for fault in ("slot", "padding", "body"):
            row = synthetic_report()
            if fault == "slot":
                row["snapshots"][1]["files"].pop()
            if fault == "padding":
                row["snapshots"][4]["checks"]["padding_values"] = 0
            load = (
                (lambda summary: dict(receipt_loader(summary), unexpected=True)) if fault == "body" else receipt_loader
            )
            with self.assertRaises(RuntimeError, msg=fault):
                verify(row, FIXTURES, load)


if __name__ == "__main__":
    unittest.main()
