# SPDX-License-Identifier: Apache-2.0
"""CPU-only token diagnostics; no TTNN import, model construction or device access."""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("token_retention_phases", ROOT / "tt/benchmark_phases.py")
phases = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phases)
tree = ast.parse((ROOT / "tt/generator_vllm.py").read_text())
node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Gemma4ForCausalLM")
namespace = {"torch": torch, "json": json}
exec(compile(ast.Module(body=[node], type_ignores=[]), "generator_vllm.py", "exec"), namespace)
Adapter = namespace["Gemma4ForCausalLM"]


class TokenRetentionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "tokens.jsonl"

    def recorder(self, **kwargs):
        recorder = phases.PhaseRecorder(
            self.path, {}, auto_export_interval=3600, retain_decode_tokens=kwargs.pop("retain", True), **kwargs
        )
        self.addCleanup(lambda: Path(recorder.socket_path).unlink(missing_ok=True))
        self.addCleanup(lambda: recorder.export_stream.close() if recorder.export_stream else None)
        return recorder

    def begin(self, recorder, positions, **metadata):
        return recorder.observe(
            "begin", "decode", batch_size=len(positions), sampling_mode="device", positions=positions, **metadata
        )

    def test_reused_device_delayed_completion_keeps_submission_positions_and_values(self):
        recorder = self.recorder()
        device, host_a, host_b = object(), object(), object()
        a = self.begin(recorder, recorder.observe("decode_positions", [20, 30], True), slot_remap=None)
        recorder.observe("submitted_output", a, device)
        recorder.observe("bind_readback", device, host_a)
        b = self.begin(recorder, recorder.observe("decode_positions", [31, 21], True), slot_remap=[1, 0])
        recorder.observe("submitted_output", b, device)
        recorder.observe("bind_readback", device, host_b)
        ids_b = torch.tensor([[41], [42]])
        recorder.observe("complete_output", host_b, sampled_tokens=ids_b)
        recorder.observe("complete_output", host_a, sampled_tokens=torch.tensor([[11], [12]]))
        ids_b.fill_(99)
        result = recorder.command("drain")
        events = [e for e in result["events"] if e["event"] == "complete"]
        self.assertEqual(
            [(e["step_id"], e["positions"], e["sampled_decode_token_ids"]) for e in events],
            [(b, [31, 21], [41, 42]), (a, [20, 30], [11, 12])],
        )
        self.assertEqual(events[0]["slot_remap"], [1, 0])
        self.assertFalse(result["token_retention"]["loss"])
        self.assertFalse(result["token_retention"]["pending_incomplete"])
        self.assertFalse(result["token_retention"]["request_ids_available"])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_supported_batches_and_inactive_rows_are_physical_samples(self):
        recorder = self.recorder()
        for batch in (1, 8, 16, 32):
            positions = [10] + [-1] * (batch - 1)
            step = self.begin(recorder, positions)
            recorder.observe("complete", step, sampled_tokens=torch.arange(batch).reshape(batch, 1))
        result = recorder.command("drain")
        events = [e for e in result["events"] if e["event"] == "complete"]
        self.assertEqual([len(e["sampled_decode_token_ids"]) for e in events], [1, 8, 16, 32])
        self.assertEqual(events[-1]["positions"], [10] + [-1] * 31)
        self.assertFalse(result["token_retention"]["client_emission_verified"])

    def test_disabled_never_inspects_payload(self):
        recorder = self.recorder(retain=False)
        step = self.begin(recorder, [0])
        recorder.observe("complete", step, sampled_tokens=object())
        result = recorder.command("drain")
        self.assertNotIn("sampled_decode_token_ids", json.dumps(result))
        self.assertNotIn("token_retention", result)

    def test_host_sampling_and_prefill_have_explicit_noncoverage(self):
        recorder = self.recorder()
        for phase in ("prefill", "decode"):
            step = recorder.observe("begin", phase, batch_size=1, sampling_mode="host_compatibility")
            recorder.observe("complete", step)
        result = recorder.command("drain")
        self.assertFalse(result["token_retention"]["prefill_tokens_retained"])
        self.assertFalse(result["token_retention"]["host_sampled_tokens_retained"])
        self.assertEqual(result["events"][-1]["token_retention_status"], "host_sampled_tokens_unavailable")

    def test_unknown_completion_disables_without_guessing_or_retaining_references(self):
        recorder = self.recorder()
        step = self.begin(recorder, [1])
        recorder.observe("submitted_output", step, object())
        recorder.observe("complete_output", object(), sampled_tokens=torch.tensor([[1]]))
        result = recorder.command("check")
        self.assertTrue(result["token_retention"]["loss"])
        self.assertEqual(result["token_retention"]["diagnostic_failures"], 1)
        self.assertEqual(len(recorder.pending) + len(recorder.unbound) + len(recorder.readbacks), 0)
        self.assertIsNone(self.begin(recorder, [1]))

    def test_pending_bound_and_bad_payload_disable_diagnostics(self):
        recorder = self.recorder(max_pending_steps=1)
        self.begin(recorder, [1])
        self.assertIsNone(self.begin(recorder, [2]))
        self.assertTrue(recorder.token_diagnostic_disabled)
        self.assertEqual(len(recorder.pending), 0)

    def test_adapter_async_returns_identical_tensor_even_if_capture_fails(self):
        recorder = self.recorder()
        adapter = Adapter.__new__(Adapter)
        adapter._benchmark_phases = recorder
        result = torch.tensor([[9]], dtype=torch.int64)
        adapter.generator = SimpleNamespace(process_decode_output_host=Mock(return_value=result))
        adapter._validate_token_ids = lambda tensor: tensor
        host = object()
        step = self.begin(recorder, [30])
        device = object()
        recorder.observe("submitted_output", step, device)
        recorder.observe("bind_readback", device, host)
        with patch.object(recorder, "_token_payload", side_effect=RuntimeError("export conversion failed")):
            self.assertIs(adapter.process_decode_output_host(host, is_tokens=True), result)
        self.assertEqual(result.tolist(), [[9]])
        self.assertTrue(recorder.token_diagnostic_disabled)
        adapter._validate_token_ids = Mock(side_effect=ValueError("model validation"))
        with self.assertRaisesRegex(ValueError, "model validation"):
            adapter.process_decode_output_host(host, is_tokens=True)

    def test_failed_export_is_loss_and_future_outputs_continue(self):
        recorder = self.recorder()
        step = self.begin(recorder, [30])
        recorder.observe("complete", step, sampled_tokens=torch.tensor([[7]]))
        with patch.object(recorder, "_write_export", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                recorder._export(automatic=True)
        self.assertTrue(recorder.command("check")["token_retention"]["export_failed"])
        self.assertIsNone(self.begin(recorder, [31]))
        self.assertIsNone(recorder.observe("complete_output", object(), sampled_tokens=torch.tensor([[8]])))

    def test_file_cap_persists_loss_status(self):
        recorder = self.recorder(max_file_bytes=4096)
        for _ in range(4):
            step = self.begin(recorder, [1] * 32)
            recorder.observe("complete", step, sampled_tokens=torch.arange(32).reshape(32, 1))
        recorder.command("drain")
        record = json.loads(self.path.read_text())
        self.assertTrue(record["token_retention"]["loss"])
        self.assertTrue(record["token_retention"]["file_exhausted"])
        self.assertLessEqual(self.path.stat().st_size, 4096)

    def test_direct_sync_adapter_returns_exact_sample_ids(self):
        recorder = self.recorder()
        adapter = Adapter.__new__(Adapter)
        adapter._benchmark_phases = recorder
        adapter.batch = 2
        adapter._pages = Mock()
        adapter._log = Mock()
        adapter._advance_sampler = Mock()
        adapter._validate_token_ids = lambda tensor: tensor
        adapter.generator = SimpleNamespace(decode_forward=Mock(return_value=[3, 4]), page_table=object())
        result = adapter.decode_forward(
            torch.zeros(2, 1),
            torch.tensor([30, -1]),
            page_table=object(),
            kv_cache=object(),
            reload_inputs=True,
            reload_page_table=False,
            reload_sampling_params=False,
            reset_sampling_state=False,
            sampling_params=object(),
        )
        self.assertEqual(result.tolist(), [[3], [4]])
        event = recorder.command("drain")["events"][-1]
        self.assertEqual(event["sampled_decode_token_ids"], [3, 4])
        self.assertEqual(event["positions"], [30, -1])
        self.assertIsNone(event["slot_remap"])
        adapter.generator.decode_forward.assert_called_once()

    def test_bad_tensor_payload_drops_diagnostics_not_model_values(self):
        recorder = self.recorder()
        step = self.begin(recorder, [1])
        recorder.observe("complete", step, sampled_tokens=torch.tensor([[1.5]]))
        self.assertTrue(recorder.token_diagnostic_disabled)
        self.assertTrue(recorder.command("check")["token_retention"]["loss"])

    def test_duplicate_binding_and_buffer_overflow_are_explicit_loss(self):
        recorder = self.recorder(max_events=1)
        device, host = object(), object()
        first = self.begin(recorder, [1])
        recorder.observe("submitted_output", first, device)
        recorder.observe("bind_readback", device, host)
        second = self.begin(recorder, [2])
        recorder.observe("submitted_output", second, device)
        recorder.observe("bind_readback", device, host)
        status = recorder.command("check")["token_retention"]
        self.assertTrue(status["diagnostic_disabled"])
        self.assertTrue(status["loss"])
        self.assertFalse(status["pending_incomplete"])

    def test_manual_reset_cannot_silently_drop_retained_tokens(self):
        recorder = self.recorder()
        step = self.begin(recorder, [1])
        recorder.observe("complete", step, sampled_tokens=torch.tensor([[9]]))
        with self.assertRaisesRegex(RuntimeError, "drain retained"):
            recorder.command("reset")
        recorder.command("drain")
        recorder.command("reset")

    def test_nonidentity_remap_without_reload_marks_positions_uncertain(self):
        recorder = self.recorder()
        self.assertEqual(recorder.observe("decode_positions", [20, 30], True), [20, 30])
        self.assertEqual(recorder.observe("decode_positions", [0, 0], False, slot_remap=[0, 1]), [21, 31])
        self.assertIsNone(recorder.observe("decode_positions", [0, 0], False, slot_remap=[1, 0]))
        status = recorder.command("check")["token_retention"]
        self.assertTrue(status["position_identity_uncertain"])
        self.assertTrue(status["loss"])

    def test_insecure_setup_is_rejected_without_affecting_model_setup(self):
        self.assertIsNone(phases.PhaseRecorder.create(self.path, {}, auto_export_interval=0, retain_decode_tokens=True))
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
