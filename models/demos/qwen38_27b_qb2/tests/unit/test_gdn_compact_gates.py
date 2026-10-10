# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Packed gate batching and strict physical-screen receipt coverage."""

import copy
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from models.demos.qwen38_27b_qb2.tests.gdn_compact_gates import CASES, validate_report
from models.demos.qwen38_27b_qb2.tests.gdn_epilogue import compare_timings
from models.demos.qwen38_27b_qb2.tt.gdn_step import flat_prepare, model_adapter, op
from models.demos.qwen38_27b_qb2.tt.gdn_step.gates import from_packed


def receipt():
    hashes = [[str(rank) * 64 for rank in range(4)] for _ in range(4)]
    other = [[str(rank + 4) * 64 for rank in range(4)] for _ in range(4)]
    timings = [dict(variant=v, traced_call_us=[t] * 5) for v, t in (("native", 100), ("fused", 90), ("native", 101))]
    cases = [
        dict(
            batch=b,
            placement=p,
            input_immutability=True,
            stable_addresses=True,
            changed_input_trace=True,
            checks=[
                dict(allocation=i, prepared_sha256=copy.deepcopy(h), bit_identical=True, finite=True)
                for i, h in ((0, hashes), (1, other), (0, hashes))
            ],
            timings=copy.deepcopy(timings),
            comparison=compare_timings(timings),
        )
        for b, p in CASES
    ]
    layers = [
        dict(
            batch=b,
            checkpoints=[
                dict(step=s, state_history_output_sha256=copy.deepcopy(hashes[:3]), bit_identical=True, finite=True)
                for s in (1, 2, 4, 8, 16, 32, 64)
            ],
        )
        for b in (16, 32)
    ]
    return dict(state="completed", cleanup_completed=True, device_ids=[0, 4, 12, 8], cases=cases, layers=layers)


class CompactGateTests(unittest.TestCase):
    def test_compact_math_retains_every_user_and_gate_channel(self):
        fake = SimpleNamespace(
            reshape=torch.reshape,
            sigmoid=torch.sigmoid,
            float32=torch.float32,
            typecast=lambda x, d: x.to(d),
            add=torch.add,
            mul=lambda a, b, **kw: a
            * F.softplus(
                b, beta=kw["input_tensor_b_activations"][0][1], threshold=kw["input_tensor_b_activations"][0][2]
            ),
            UnaryWithParam=lambda *args: args,
            UnaryOpType=SimpleNamespace(SOFTPLUS="softplus"),
        )
        for batch in (1, 16, 17, 31, 32):
            with self.subTest(batch=batch), patch.dict(sys.modules, {"ttnn": fake}):
                rng = torch.Generator().manual_seed(32 + batch)
                packed = torch.randn(1, batch, 4160, generator=rng).bfloat16()
                a = -torch.rand(1, 1, 12, generator=rng)
                bias = torch.randn(1, 1, 12, generator=rng)
                public = from_packed(packed, a, bias)
                compact = from_packed(packed, a, bias, compact=True)
                for x, y in zip(public, compact):
                    self.assertEqual(y.shape, torch.Size((1, batch, 12)))
                    self.assertTrue(torch.equal(x.reshape(batch, 12), y.reshape(batch, 12)))

    def test_adapter_exponentiates_all_compact_users(self):
        observed = []
        fake = SimpleNamespace(
            reshape=lambda v, shape: SimpleNamespace(shape=shape, buffer_address=v.buffer_address),
            exp=lambda v: observed.append(v.clone()) or v.exp(),
        )
        for batch in (16, 32):
            state = SimpleNamespace(shape=(batch, 12, 128, 128), buffer_address=lambda: 123)
            q = SimpleNamespace(shape=(1, batch, 512))
            v = SimpleNamespace(shape=(1, batch, 1536))
            gates = -torch.arange(1, batch * 12 + 1, dtype=torch.float32).reshape(1, batch, 12) / 100
            output = object()
            prepared, recurrence = [], []
            with patch.dict(sys.modules, {"ttnn": fake}), patch.object(
                flat_prepare, "prepare", lambda *a, **k: prepared.append((a, k))
            ), patch.object(op, "step", lambda *a, **k: recurrence.append(k)):
                result = model_adapter.step_from_flat(
                    q,
                    q,
                    v,
                    gates,
                    gates,
                    state,
                    output,
                    shared_qk_outputs=(object(), object()),
                    flat_prepare_outputs=(object(), object()),
                    compact_qkv=True,
                    compact_gates=True,
                    resident_state=True,
                    raw_output=True,
                )
            self.assertIs(result, output)
            self.assertTrue(torch.equal(observed[-1], gates))
            self.assertTrue(torch.equal(prepared[0][0][3], gates.exp()))
            self.assertTrue(prepared[0][1]["compact_gates"])
            self.assertTrue(recurrence[0]["resident_state"])

    def test_candidate_flags_reject_inconsistent_layouts(self):
        fake = SimpleNamespace()
        with patch.dict(sys.modules, {"ttnn": fake}):
            for flag in (1, "yes", None):
                with self.subTest(flag=flag), self.assertRaises(ValueError):
                    from_packed(SimpleNamespace(shape=(1, 16, 4160)), None, None, compact=flag)
            with self.assertRaises(ValueError):
                model_adapter.step_from_flat(None, None, None, None, None, None, None, compact_gates=True)

    def test_resident_adapter_rejects_unprepared_public_inputs(self):
        with self.assertRaisesRegex(ValueError, "compact direct-preparation"):
            model_adapter.step_from_flat(None, None, None, None, None, None, None, resident_state=True)

    def test_valid_report_does_not_promote_model(self):
        result = validate_report(receipt())
        self.assertTrue(result["correctness_passed"])
        self.assertFalse(result["full_model_qualified"])

    def test_reject_missing_shape_rank_changed_input_or_state_checkpoint(self):
        edits = [
            lambda r: r["cases"].pop(),
            lambda r: r["cases"][0]["checks"][0]["prepared_sha256"][0].pop(),
            lambda r: r["cases"][0].update(changed_input_trace=False),
            lambda r: r["layers"][1]["checkpoints"].pop(),
            lambda r: r.update(cleanup_completed=False),
            lambda r: r["layers"][0]["checkpoints"][0].update(bit_identical=False),
        ]
        for edit in edits:
            r = receipt()
            edit(r)
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                validate_report(r)

    def test_timing_is_recomputed_and_drift_gets_no_speed_credit(self):
        r = receipt()
        r["cases"][0]["comparison"]["qualified_speedup"] = 9
        with self.assertRaises(ValueError):
            validate_report(r)
        r = receipt()
        r["cases"][0]["timings"][2]["traced_call_us"] = [120] * 5
        r["cases"][0]["comparison"] = compare_timings(r["cases"][0]["timings"])
        self.assertIsNone(validate_report(r)["timings"][0]["qualified_speedup"])


if __name__ == "__main__":
    unittest.main()
