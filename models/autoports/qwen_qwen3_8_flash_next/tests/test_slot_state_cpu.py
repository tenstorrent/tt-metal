# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU structural tests; do not import TTNN or touch a device."""

import itertools
import unittest
from types import SimpleNamespace

from models.autoports.qwen_qwen3_8_flash_next.tt.slot_state import (
    SlotStateMover,
    _state_tensors,
    remap_slot_mapping,
    validate_remap,
)


def cache(value):
    def tensor(offset):
        return SimpleNamespace(value=value * 100 + offset, shape=(1, 32), dtype="uint32", layout="row_major")

    return SimpleNamespace(
        capacity=128,
        position=value * 11,
        layers=[
            SimpleNamespace(
                conv=tensor(0), recurrent=tensor(1), ple_conv=tensor(2), token_history=tensor(3), attention=None
            ),
            SimpleNamespace(
                conv=None,
                recurrent=None,
                ple_conv=None,
                token_history=None,
                attention=SimpleNamespace(
                    **{
                        name: tensor(index + 4)
                        for index, name in enumerate(
                            ("position", "page_table", "k_cache", "v_cache", "raw_history", "compressed_keys")
                        )
                    }
                ),
            ),
        ],
    )


def copy_tensor(source, destination):
    destination.value = source.value


class SlotStateTest(unittest.TestCase):
    def test_prepared_scratch_copies_read_current_state_and_preserve_bindings(self):
        caches, scratch = [cache(0), cache(1), cache(2)], cache(99)
        recording = None
        signatures = []

        def copy(source, destination):
            if recording is None:
                copy_tensor(source, destination)
            else:
                recording.append((source, destination))

        def capture(body, signature):
            nonlocal recording
            recording = []
            body()
            commands, recording = recording, None
            signatures.append(signature)
            return lambda: [copy_tensor(source, destination) for source, destination in commands]

        mover = SlotStateMover(caches, scratch, copy_tensor=copy)
        addresses = [[id(tensor) for _, tensor in _state_tensors(c)] for c in (*caches, scratch)]
        mover.prepare_copy_traces(capture)
        mover.prepare_copy_traces(capture)
        self.assertEqual(len(signatures), 6)
        self.assertEqual(scratch.position, 1089)
        self.assertEqual(scratch.layers[0].conv.value, 9900)
        for step in range(3):
            caches[0].layers[0].conv.value = 123 + step
            caches[0].position = 45 + step
            mover._copy(caches[0], scratch)
            mover._copy(scratch, caches[1])
            self.assertEqual(caches[1].layers[0].conv.value, 123 + step)
            self.assertEqual(caches[1].position, 45 + step)
        for index, item in enumerate(caches):
            for (_, src), (_, dst) in zip(_state_tensors(cache(index + 7)), _state_tensors(item), strict=True):
                copy_tensor(src, dst)
            item.position = 70 + index
        before = [[tensor.value for _, tensor in _state_tensors(item)] for item in caches]
        mover.apply((2, 0, 1))
        for destination, source in enumerate((2, 0, 1)):
            self.assertEqual([tensor.value for _, tensor in _state_tensors(caches[destination])], before[source])
            self.assertEqual(caches[destination].position, 70 + source)
        self.assertEqual([[id(tensor) for _, tensor in _state_tensors(c)] for c in (*caches, scratch)], addresses)

    def test_shared_kv_is_not_part_of_prepared_copy(self):
        caches, scratch = [cache(0), cache(1)], cache(99)
        shared = caches[0].layers[1].attention.k_cache
        for item in (*caches, scratch):
            item.shared_kv = True
            item.layers[1].attention.k_cache = shared
            item.layers[1].attention.v_cache = shared
        copies = []
        mover = SlotStateMover(caches, scratch, copy_tensor=lambda src, dst: copies.append((src, dst)))
        mover.prepare_copy_traces(lambda body, signature: body)
        mover._copy(caches[0], scratch)
        mover._copy(scratch, caches[1])
        self.assertTrue(copies)
        self.assertFalse(any(src is shared or dst is shared for src, dst in copies))

    def test_every_permutation_through_six_slots(self):
        for width in range(1, 7):
            for remap in itertools.permutations(range(width)):
                caches = [cache(i) for i in range(width)]
                addresses = [[id(tensor) for _, tensor in _state_tensors(item)] for item in caches]
                mover = SlotStateMover(caches, cache(99), copy_tensor=copy_tensor)
                mover.warmup()
                mover.apply(remap)
                for destination, source in enumerate(remap):
                    self.assertEqual(caches[destination].position, source * 11)
                    self.assertEqual(
                        [tensor.value for _, tensor in _state_tensors(caches[destination])],
                        [tensor.value for _, tensor in _state_tensors(cache(source))],
                    )
                    self.assertEqual(
                        [id(tensor) for _, tensor in _state_tensors(caches[destination])], addresses[destination]
                    )

    def test_invalid_remap_does_not_mutate(self):
        calls = []
        mover = SlotStateMover([cache(0), cache(1)], cache(99), copy_tensor=lambda *args: calls.append(args))
        for remap in ([0], [0, 0], [0, 2], [-1, 1], [0.0, 1.0], [[0, 1]]):
            with self.assertRaises(ValueError):
                mover.apply(remap)
        self.assertEqual(calls, [])
        mover.apply([0, 1])
        self.assertEqual(calls, [])

    def test_reject_alias_and_geometry_mismatch_before_mutation(self):
        first, second, scratch = cache(0), cache(1), cache(99)
        second.layers[0].recurrent = first.layers[0].recurrent
        with self.assertRaisesRegex(ValueError, "Aliased"):
            SlotStateMover([first, second], scratch)
        second = cache(1)
        second.layers[1].attention.k_cache.dtype = "bfloat8_b"
        with self.assertRaisesRegex(ValueError, "identical"):
            SlotStateMover([first, second], scratch)
        with self.assertRaisesRegex(ValueError, "scratch"):
            SlotStateMover([first, cache(1)])

    def test_off_batch_metadata_and_dormant_sampler_move_once(self):
        sampler = object()
        mapping = {0: "table-a", 2: sampler}
        remapped = remap_slot_mapping(mapping, [2, 0, 3, 1])
        self.assertEqual(remapped, {0: sampler, 1: "table-a"})
        self.assertIs(remapped[0], sampler)
        self.assertEqual(mapping, {0: "table-a", 2: sampler})

    def test_device_remap_metadata_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "host scheduler"):
            validate_remap(SimpleNamespace(device=SimpleNamespace(type="tt")), 1)


if __name__ == "__main__":
    unittest.main()
