# SPDX-License-Identifier: Apache-2.0
"""CPU checks for exact event accounting and alias deduplication."""
import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "physical_inventory", Path(__file__).with_name("physical_memory_inventory.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def event(kind, address, per_bank=64):
    return dict(
        node_type=kind,
        params=dict(device_id="7", type="DRAM", address=str(address), max_size_per_bank=str(per_bank), size="100"),
    )


class LedgerTest(unittest.TestCase):
    def test_shared_phase_and_pool_owners_count_once_with_bank_padding(self):
        ledger = module.AllocationLedger({"DRAM": 8, "L1": 80})
        ledger.update([event("buffer_allocate", 100), event("buffer_allocate", 200, 128)])
        aliases = [
            dict(path=name, address=100, buffer_type="DRAM")
            for name in ("model.gate", "model.prefill.gate", "model.decode.gate")
        ]
        out = ledger.snapshot(aliases, domain=7, allocator_bytes=1536 + 80, baseline_bytes=80)
        self.assertEqual(out["dram_bytes_per_device"], 1536)
        self.assertEqual(out["named_dram_bytes_per_device"], 512)
        self.assertEqual(out["unnamed_dram_bytes_per_device"], 1024)
        self.assertEqual(len(out["allocations"][0]["aliases"]), 3)

    def test_address_reuse_across_windows_keeps_current_owner(self):
        ledger = module.AllocationLedger({"DRAM": 8})
        ledger.update([event("buffer_allocate", 100)])
        ledger.update([event("buffer_deallocate", 100), event("buffer_allocate", 100, 128)])
        self.assertEqual(ledger.snapshot([], domain=7, allocator_bytes=1024)["dram_bytes_per_device"], 1024)
        self.assertEqual(ledger.peak[(7, "DRAM")], 1024)

    def test_missing_event_is_a_failure(self):
        ledger = module.AllocationLedger({"DRAM": 8})
        ledger.update([event("buffer_allocate", 100)])
        with self.assertRaisesRegex(ValueError, "reconciliation"):
            ledger.snapshot([], domain=7, allocator_bytes=1024)

    def test_capture_must_precede_owned_allocations(self):
        ledger = module.AllocationLedger({"DRAM": 8})
        with self.assertRaisesRegex(ValueError, "pre-capture"):
            ledger.update([event("buffer_deallocate", 100)])


if __name__ == "__main__":
    unittest.main()
