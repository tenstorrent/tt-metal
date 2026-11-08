# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks of the vLLM adapter's explicit decode reload contract (version 1).

The generator is replaced by a recorder, so this exercises only the adapter's
translation of the plugin's four reload commands into generator calls: which
inputs are fed, which page-table width is selected, and when the resident
device inputs are read back to rebuild a wider state.
"""

import sys
import types
import unittest
from types import SimpleNamespace

import torch

# The adapter module imports the real generator, whose model stack needs a
# device-era ttnn build. This check replaces it before the import: only the
# adapter's translation of the reload commands is under test.
_GENERATOR_MODULE = "models.demos.k2_horizon_7b_qb2.tt.generator"
if _GENERATOR_MODULE not in sys.modules:
    _stub = types.ModuleType(_GENERATOR_MODULE)
    _stub.K2Generator = type("K2Generator", (), {})
    sys.modules[_GENERATOR_MODULE] = _stub

from models.demos.k2_horizon_7b_qb2.tt.generator_vllm import K2HorizonForCausalLM  # noqa: E402

VOCAB = 250624


class RecordingGenerator:
    def __init__(self):
        self.state = None
        self.counters = {}
        self.calls = []
        self.sampling = []
        self.readbacks = 0
        self.device_tokens = torch.arange(32, dtype=torch.long) + 1000
        self.device_positions = torch.full((32,), -1, dtype=torch.long)

    def configure_sampling_batch(self, **kwargs):
        self.sampling.append(kwargs)

    def read_resident_inputs(self, *, batch):
        self.readbacks += 1
        return self.device_tokens[:batch].clone(), self.device_positions[:batch].clone()

    def decode_forward(self, tokens, positions, *, page_table, kv_cache, sampling_mode, read_from_device):
        if tokens is None and (self.state is None or self.state["trace"] is None):
            raise AssertionError("resident decode without established state")
        self.calls.append(
            dict(
                tokens=None if tokens is None else tokens.clone(),
                positions=None if positions is None else positions.clone(),
                shape=tuple(page_table.shape),
                mode=sampling_mode,
            )
        )
        batch = page_table.shape[0]
        self.state = {"trace": object(), "key": (batch, tuple(page_table.shape), ())}
        if sampling_mode == "device":
            return torch.zeros(batch, dtype=torch.long)
        return torch.zeros(batch, VOCAB)


def make_adapter():
    adapter = K2HorizonForCausalLM.__new__(K2HorizonForCausalLM)
    adapter.generator = RecordingGenerator()
    adapter.mesh = None
    adapter.max_batch_size = 32
    adapter.force_host_sampling = False
    adapter.host_sampler = None
    adapter.mode = None
    adapter.host_fallback_calls = 0
    adapter.decode_table_width = None
    adapter.decode_batch = 32
    adapter.audit_path = None
    adapter.benchmark_phase = None
    return adapter


def inputs(active_rows, max_position, *, pages_per_row=None):
    """Wire-shaped batch: 32 slots, the first ``active_rows`` live at ``max_position``."""
    tokens = torch.zeros(32, dtype=torch.long)
    start_pos = torch.full((32,), -1, dtype=torch.long)
    start_pos[:active_rows] = max_position
    table = torch.zeros((32, 16384), dtype=torch.int32)
    pages = (max_position // 32) + 1 if pages_per_row is None else pages_per_row
    for row in range(active_rows):
        # vLLM never maps a live block to page 0 (its null block).
        table[row, :pages] = torch.arange(pages, dtype=torch.int32) + 1 + row * pages
    return tokens, start_pos, table


DEVICE = SimpleNamespace(top_k=[1] * 32, top_p=[0.0] * 32, temperature=[0.0] * 32, seed=[None] * 32)


class ReloadContractTest(unittest.TestCase):
    def test_contract_version_and_legacy_keyword(self):
        self.assertEqual(K2HorizonForCausalLM.decode_input_update_contract, 1)
        adapter = make_adapter()
        tokens, start_pos, table = inputs(1, 10)
        with self.assertRaises(TypeError):
            adapter.decode_forward(tokens, start_pos, table, [], sampling_params=DEVICE, reset_batch=True)

    def test_sampling_reset_requires_input_reload(self):
        adapter = make_adapter()
        tokens, start_pos, table = inputs(1, 10)
        with self.assertRaises(ValueError):
            adapter.decode_forward(
                tokens, start_pos, table, [], sampling_params=DEVICE, reload_inputs=False, reset_sampling_state=True
            )

    def test_resident_step_needs_established_state(self):
        adapter = make_adapter()
        tokens, start_pos, table = inputs(1, 10)
        with self.assertRaises(ValueError):
            adapter.decode_forward(
                tokens,
                start_pos,
                table,
                [],
                sampling_params=DEVICE,
                reload_inputs=False,
                reload_sampling_params=False,
                reset_sampling_state=False,
            )

    def test_input_reload_feeds_host_inputs_and_selects_buckets(self):
        adapter = make_adapter()
        tokens, start_pos, table = inputs(3, 100)
        out = adapter.decode_forward(tokens, start_pos, table, [], sampling_params=DEVICE)
        call = adapter.generator.calls[-1]
        self.assertEqual(adapter.decode_batch, 4)  # 3 live rows -> power-of-two batch 4
        self.assertEqual(adapter.decode_table_width, 128)  # 4096-token capacity floor
        self.assertEqual(call["shape"], (4, 128))
        self.assertTrue(torch.equal(call["positions"], start_pos[:4]))
        self.assertEqual(call["mode"], "device")
        # Sampling is configured for every wire slot; inactive rows carry -1.
        self.assertEqual(adapter.generator.sampling[-1]["positions"], [100] * 3 + [-1] * 29)
        self.assertEqual(tuple(out.shape), (32, 1))

    def test_resident_steps_feed_nothing_and_refresh_within_bucket(self):
        adapter = make_adapter()
        tokens, start_pos, table = inputs(3, 100)
        adapter.decode_forward(tokens, start_pos, table, [], sampling_params=DEVICE)
        flags = dict(reload_inputs=False, reload_sampling_params=False, reset_sampling_state=False)
        adapter.decode_forward(tokens, start_pos, table, [], sampling_params=DEVICE, **flags)
        tokens, start_pos, table = inputs(3, 200)
        adapter.decode_forward(tokens, start_pos, table, [], sampling_params=DEVICE, reload_page_table=True, **flags)
        for call in adapter.generator.calls[1:]:
            self.assertIsNone(call["tokens"])
            self.assertIsNone(call["positions"])
            self.assertEqual(call["shape"], (4, 128))
        self.assertEqual(adapter.generator.readbacks, 0)
        self.assertEqual(len(adapter.generator.sampling), 1)

    def test_bucket_crossing_rebuilds_from_resident_inputs(self):
        adapter = make_adapter()
        tokens, start_pos, table = inputs(3, 4000)
        adapter.decode_forward(tokens, start_pos, table, [], sampling_params=DEVICE)
        self.assertEqual(adapter.decode_table_width, 128)
        flags = dict(reload_inputs=False, reload_sampling_params=False, reset_sampling_state=False)
        # Host positions 4094: the device may already be at 4095, still inside 4096.
        tokens, start_pos, table = inputs(3, 4094)
        adapter.decode_forward(tokens, start_pos, table, [], sampling_params=DEVICE, reload_page_table=True, **flags)
        self.assertEqual(adapter.decode_table_width, 128)
        self.assertEqual(adapter.generator.readbacks, 0)
        # Host positions 4095: the next device position is 4096, outside the bucket.
        tokens, start_pos, table = inputs(3, 4095)
        adapter.generator.device_positions[:3] = 4096
        adapter.decode_forward(tokens, start_pos, table, [], sampling_params=DEVICE, reload_page_table=True, **flags)
        call = adapter.generator.calls[-1]
        self.assertEqual(adapter.decode_table_width, 256)
        self.assertEqual(adapter.generator.readbacks, 1)
        self.assertEqual(call["shape"], (4, 256))
        self.assertTrue(torch.equal(call["tokens"], adapter.generator.device_tokens[:4]))
        self.assertTrue(torch.equal(call["positions"], torch.tensor([4096, 4096, 4096, -1])))
        self.assertEqual(len(adapter.generator.sampling), 1)  # seeds keep their stream

    def test_populated_pages_bound_the_bucket_regardless_of_stale_positions(self):
        adapter = make_adapter()
        tokens, start_pos, table = inputs(3, 100)
        adapter.decode_forward(tokens, start_pos, table, [], sampling_params=DEVICE)
        flags = dict(reload_inputs=False, reload_sampling_params=False, reset_sampling_state=False)
        tokens, start_pos, table = inputs(3, 100, pages_per_row=131)  # a live page past column 128
        adapter.decode_forward(tokens, start_pos, table, [], sampling_params=DEVICE, reload_page_table=True, **flags)
        self.assertEqual(adapter.decode_table_width, 256)
        self.assertEqual(adapter.generator.readbacks, 1)

    def test_host_sampling_reloads_every_step(self):
        adapter = make_adapter()
        tokens, start_pos, table = inputs(2, 50)
        out = adapter.decode_forward(
            tokens, start_pos, table, [], sampling_params=None, reload_sampling_params=False, reset_sampling_state=False
        )
        call = adapter.generator.calls[-1]
        self.assertEqual(call["mode"], "host")
        self.assertIsNotNone(call["tokens"])
        self.assertEqual(adapter.host_fallback_calls, 1)
        self.assertEqual(adapter.generator.sampling, [])
        self.assertEqual(tuple(out.shape), (32, 1, VOCAB))
        # A sampling-mode change without an input reload is a contract violation.
        with self.assertRaises(ValueError):
            adapter.decode_forward(
                tokens,
                start_pos,
                table,
                [],
                sampling_params=DEVICE,
                reload_inputs=False,
                reload_sampling_params=False,
                reset_sampling_state=False,
            )


if __name__ == "__main__":
    unittest.main()
