# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU orchestration control with a deliberately shape-sensitive recurrence."""
import unittest
from types import SimpleNamespace

import torch

from .prefill_tail_checkpoint_diagnostic import TailCheckpointDiagnostic, compact_state


def fixture():
    state = SimpleNamespace(recurrent=torch.zeros(1), attention=None)
    cache = SimpleNamespace(layers=[state], position=0)
    gen = SimpleNamespace(caches=[cache], prefill_cache=cache, page_tables={})
    helper = TailCheckpointDiagnostic.__new__(TailCheckpointDiagnostic)
    helper.gen, helper.valid, helper.active = gen, {}, None
    helper.tt = SimpleNamespace(copy=lambda source, destination: destination.copy_(source))
    helper.buffers = [[("layer0.recurrent", torch.zeros(1))]]
    helper.report = dict(saved=[], restored=[], missing_or_invalid=[], invalidated=[])
    output = []

    def reset(owned):
        owned.position = 0
        owned.layers[0].recurrent.zero_()

    def chunk(count, slot):
        offset = helper.active[slot]["offset"]
        for index in range(count):
            state.recurrent += 1 if count == 1 else 2
            output.append(state.recurrent.clone())
        cache.position = offset + count

    def prefill(tokens, **kwargs):
        start, end = kwargs["start_pos"][0], kwargs["prompt_lens"][0]
        if start == 0:
            helper.reset_cache(cache)
        output.clear()
        while start < end:
            available = min(end, (start // 32 + 1) * 32) - start
            count = 1 << (available.bit_length() - 1)
            helper.chunk(count, 0)
            start += count
        return torch.stack(output).unsqueeze(0)

    helper.original = dict(
        prefill_forward=prefill, _prefill_chunk=chunk, _reset_cache=reset, decode_forward=lambda *a, **kw: None
    )
    return helper


class TailCheckpointCPU(unittest.TestCase):
    def run_prefill(self, helper, end, start=0):
        return helper.prefill(
            torch.arange(300).reshape(1, -1), prompt_lens=[end], start_pos=[start], slots=(0,), return_all_logits=True
        )

    def test_replay_old_row_is_sliced_and_committed_odd_cache_preserved(self):
        baseline = self.run_prefill(fixture(), 265)
        helper = fixture()
        prefix = self.run_prefill(helper, 31)
        self.assertEqual(helper.gen.caches[0].position, 31)
        self.assertEqual(float(helper.gen.caches[0].layers[0].recurrent), float(prefix[0, -1]))
        continued = self.run_prefill(helper, 265, 31)
        self.assertEqual(tuple(continued.shape), (1, 234, 1))
        self.assertTrue(torch.equal(continued, baseline[:, 31:265]))
        self.assertEqual(helper.report["restored"][0]["replay_from"], 30)

    def test_missing_and_decode_invalidated_checkpoints_keep_negative_control(self):
        baseline = self.run_prefill(fixture(), 265)
        for invalidation in ("missing", "decode", "wrong_owner_end"):
            helper = fixture()
            self.run_prefill(helper, 31)
            if invalidation == "missing":
                helper.valid.clear()
            elif invalidation == "decode":
                helper.decode()
            else:
                helper.valid[0]["end"] = 29
            continued = self.run_prefill(helper, 265, 31)
            self.assertFalse(torch.equal(continued, baseline[:, 31:265]))
            self.assertEqual(len(helper.report["missing_or_invalid"]), 1)

    def test_compact_inventory_excludes_kv_index_and_ownership(self):
        attention = SimpleNamespace(
            **{
                name: object()
                for name in ("raw_history", "position", "k_cache", "v_cache", "compressed_keys", "page_table")
            }
        )
        cache = SimpleNamespace(layers=[SimpleNamespace(attention=attention, recurrent=object())])
        self.assertEqual(
            [name for name, _ in compact_state(cache)],
            ["layer0.recurrent", "layer0.attention.raw_history", "layer0.attention.position"],
        )
        self.assertTrue(all((end - 1) % 4 != 3 for end in range(1, 385, 2)))


if __name__ == "__main__":
    unittest.main()
