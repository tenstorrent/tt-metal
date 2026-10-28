# SPDX-License-Identifier: Apache-2.0
"""Separate allocator domains even when graph placement/address coincide."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("inventory", Path(__file__).with_name("physical_memory_inventory.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def event(kind="buffer_allocate", buffer_type=0, coarse="DRAM", address=4096, per_bank=64):
    return dict(
        node_type=kind,
        params=dict(
            device_id="0",
            type=coarse,
            buffer_type=str(buffer_type),
            address=str(address),
            max_size_per_bank=str(per_bank),
            size="100",
        ),
    )


class DomainsTest(unittest.TestCase):
    def ledger(self):
        return m.AllocationLedger({"DRAM": 8, "L1": 80})

    def test_same_address_regular_and_trace_domains_reconcile_independently(self):
        ledger = self.ledger()
        ledger.update([event(), event(buffer_type=4, per_bank=128)])
        snap = ledger.snapshot(
            [dict(buffer_type="DRAM", address=4096, path="weight")],
            domain=0,
            allocator_bytes=512,
            allocator_bytes_by_domain={"TRACE": 1024},
        )
        self.assertEqual(snap["dram_bytes_per_device"], 512)
        self.assertEqual(snap["trace_arena_used_bytes_per_device"], 1024)
        self.assertEqual(snap["named_dram_bytes_per_device"], 512)
        self.assertEqual(snap["independently_reconciled_allocator_domains"], ["DRAM", "TRACE"])
        trace = next(r for r in snap["allocations"] if r["type"] == "TRACE")
        self.assertEqual(trace["raw_params"]["type"], "DRAM")
        self.assertFalse(trace["aliases"])

    def test_trace_deallocate_preserves_same_address_regular_dram(self):
        ledger = self.ledger()
        ledger.update([event(), event(buffer_type=4), event("buffer_deallocate", buffer_type=4)])
        self.assertEqual(ledger.snapshot([], domain=0, allocator_bytes=512)["trace_arena_used_bytes_per_device"], 0)

    def test_missing_numeric_domain_fails(self):
        item = event()
        del item["params"]["buffer_type"]
        with self.assertRaisesRegex(ValueError, "numeric buffer_type"):
            self.ledger().update([item])

    def test_unknown_numeric_domain_fails(self):
        with self.assertRaisesRegex(ValueError, "Unknown numeric"):
            self.ledger().update([event(buffer_type=99)])

    def test_actual_trace_mismatch_is_not_exempted(self):
        ledger = self.ledger()
        ledger.update([event(buffer_type=4)])
        with self.assertRaisesRegex(ValueError, "event TRACE=512"):
            ledger.snapshot([], domain=0, allocator_bytes=0, allocator_bytes_by_domain={"TRACE": 256})

    def test_l1_small_is_separate_from_l1(self):
        ledger = self.ledger()
        ledger.update([event(buffer_type=1, coarse="L1"), event(buffer_type=3, coarse="L1")])
        snap = ledger.snapshot(
            [], domain=0, allocator_bytes=0, allocator_bytes_by_domain={"L1": 5120, "L1_SMALL": 5120}
        )
        self.assertEqual(snap["allocator_domain_bytes_per_device"]["L1_SMALL"], 5120)

    def test_dynamic_trace_real_dram_domain_still_counts_in_dram(self):
        ledger = self.ledger()
        ledger.update([event(per_bank=128)])
        self.assertEqual(ledger.snapshot([], domain=0, allocator_bytes=1024)["trace_arena_used_bytes_per_device"], 0)

    def test_nonzero_baselines_apply_per_domain(self):
        ledger = self.ledger()
        ledger.update([event(), event(buffer_type=4)])
        snap = ledger.snapshot(
            [],
            domain=0,
            allocator_bytes=1536,
            baseline_bytes=1024,
            allocator_bytes_by_domain={"TRACE": 768},
            baseline_bytes_by_domain={"TRACE": 256},
        )
        self.assertTrue(snap["exact_allocator_reconciliation"])
        self.assertEqual(snap["event_trace_arena_peak_bytes_per_device"], 768)


if __name__ == "__main__":
    unittest.main()
