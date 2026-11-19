# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Protocol metadata tests; run as a module without TTNN/pytest device fixtures."""

import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from models.autoports.ornith_ai_ornith_1_5_9b.tt.decode_contract import DecodeCommands, DecodeSubmission, ForwardInputs


class ForbiddenStaleHost:
    def __getattribute__(self, name):
        raise AssertionError(f"Resident decode read stale host {name}")


class DecodeContractTests(unittest.TestCase):
    def context(self, command, **updates):
        defaults = {
            "mode": "device",
            "slot_capacity": 8,
            "batch_size": 2,
            "forward_initialized": True,
            "sampling_params_initialized": True,
            "sampling_state_initialized": True,
        }
        defaults.update(updates)
        command.validate_context(**defaults)

    def test_direct_defaults_and_strict_keywords(self):
        command = DecodeCommands.from_kwargs({})
        self.assertTrue(command.reload_inputs)
        self.assertEqual(command.required_action_order, ("reload_all_forward_inputs",))
        self.context(command, mode="host")
        for unknown in ("reset_batch", "decode_layout_changed", "reload_input", "ignored_magic"):
            with self.assertRaises(TypeError):
                DecodeCommands.from_kwargs({unknown: False})
        with self.assertRaises(TypeError):
            DecodeCommands(reload_inputs=1)

    def test_invalid_command_combinations(self):
        for kwargs in (
            {"reload_inputs": True, "reload_page_table": True},
            {"reload_inputs": False, "reset_sampling_state": True},
            {"sampling_state_slots": [0]},
            {"reset_sampling_state": True, "sampling_state_slots": [1, 1]},
            {"slot_remap": [0, -1]},
        ):
            with self.assertRaises(ValueError):
                DecodeCommands(**kwargs)

    def test_resident_mode_never_accesses_stale_host_inputs(self):
        resident = ForwardInputs(object(), object(), object(), object())
        command = DecodeCommands(reload_inputs=False)
        actual = command.select_forward_inputs(host=ForbiddenStaleHost(), resident=resident)
        self.assertIs(actual.tokens, resident.tokens)
        self.assertIs(actual.positions, resident.positions)
        self.assertIs(actual.rope, resident.rope)
        self.assertIs(actual.page_table, resident.page_table)
        self.context(command)
        with self.assertRaises(ValueError):
            self.context(command, forward_initialized=False)
        with self.assertRaises(ValueError):
            self.context(command, mode="host")

    def test_page_only_reload_does_not_touch_token_position_rope(self):
        resident = ForwardInputs(object(), object(), object(), object())

        class OnlyPages:
            page_table = object()

            def __getattr__(self, name):
                raise AssertionError(f"Page-only decode accessed stale {name}")

        host = OnlyPages()
        command = DecodeCommands(reload_inputs=False, reload_page_table=True)
        selected = command.select_forward_inputs(host=host, resident=resident)
        self.assertIs(selected.tokens, resident.tokens)
        self.assertIs(selected.positions, resident.positions)
        self.assertIs(selected.rope, resident.rope)
        self.assertIs(selected.page_table, host.page_table)
        self.context(command, pending_submission=True)

    def test_first_sampling_needs_explicit_params_and_state_even_none_seed(self):
        for command in (DecodeCommands(), DecodeCommands(reload_sampling_params=True)):
            with self.assertRaises(ValueError):
                self.context(
                    command,
                    sampling_params_initialized=False,
                    sampling_state_initialized=False,
                    sampling_params_available=True,
                )
        command = DecodeCommands(reload_sampling_params=True, reset_sampling_state=True)
        self.context(
            command,
            forward_initialized=False,
            sampling_params_initialized=False,
            sampling_state_initialized=False,
            sampling_params_available=True,
        )
        with self.assertRaises(ValueError):
            self.context(command, sampling_params_available=False)
        # Once initialized, no seed equality heuristic resets it again.
        self.context(DecodeCommands(reload_inputs=False), sampling_params_available=False)

    def test_full_slot_permutation_covers_off_batch_and_dormant_sampler(self):
        remap = [1, 0, 3, 2, 7, 6, 5, 4]
        command = DecodeCommands(slot_remap=remap)
        remap[0] = 0  # Immutable snapshot must not change with caller's list.
        self.assertEqual(command.slot_remap, (1, 0, 3, 2, 7, 6, 5, 4))
        self.context(command, mode="host", batch_size=2)
        self.assertEqual(command.required_action_order[0], "remap_all_slot_state")
        for wrong in ([1, 0], [1, 0, 2, 3, 4, 5, 6, 8]):
            with self.assertRaises(ValueError):
                self.context(DecodeCommands(slot_remap=wrong))

    def test_host_tensor_metadata_protocol_without_importing_torch(self):
        class HostTensor:
            device = "cpu"

            def tolist(self):
                return [1, 0, 2, 3, 4, 5, 6, 7]

        self.context(DecodeCommands(slot_remap=HostTensor()))

        class DeviceTensor:
            device = "cuda"

            def tolist(self):
                raise AssertionError("Validation must never trigger a device transfer")

        with self.assertRaises(ValueError):
            DecodeCommands(slot_remap=DeviceTensor())

    def test_reset_scope_and_order(self):
        command = DecodeCommands(
            reload_sampling_params=True,
            reset_sampling_state=True,
            sampling_state_slots=[3, 1],
            slot_remap=[1, 0, 2, 3, 4, 5, 6, 7],
        )
        self.context(command, sampling_params_available=True)
        self.assertEqual(command.sampling_state_slots, (3, 1))
        self.assertEqual(
            command.required_action_order,
            ("remap_all_slot_state", "reload_all_forward_inputs", "reload_sampling_params", "reset_sampling_state"),
        )
        with self.assertRaises(ValueError):
            self.context(command, sampling_params_available=True, pending_submission=True)
        with self.assertRaises(ValueError):
            self.context(DecodeCommands(reset_sampling_state=True, sampling_state_slots=[8]))

    def test_submission_owns_exact_output_and_read_is_observational(self):
        tensor, host, keepalive = object(), object(), object()
        handle = DecodeSubmission(0, "tokens", 2, [3, 1], 8, token_tensor=tensor, keepalive=[keepalive])
        self.assertIs(handle.token_tensor, tensor)
        self.assertEqual(handle.row_order, (3, 1))
        with self.assertRaises(FrozenInstanceError):
            handle.row_order = (1, 3)
        self.assertEqual(handle.expected_output_shape, (2, 1))
        completion = handle.complete_read(host, shape=(2, 1), row_order=[3, 1])
        self.assertIs(completion, handle.complete_read(host, shape=(2, 1), row_order=[3, 1]))
        self.assertIs(handle.token_tensor, tensor)
        self.assertEqual(handle.keepalive, (keepalive,))
        with self.assertRaises(ValueError):
            handle.complete_read(object(), shape=(2, 1), row_order=[3, 1])
        with self.assertRaises(ValueError):
            handle.complete_read(host, shape=(2, 1), row_order=[1, 3])
        with self.assertRaises(ValueError):
            handle.complete_read(host, shape=(2,), row_order=[3, 1])

    def test_logits_shape_includes_full_vocab_and_inactive_gap_rows(self):
        handle = DecodeSubmission(4, "logits", 4, [0, 1, 2, 3], 8, logits_tensor=object(), vocab_size=248320)
        self.assertEqual(handle.expected_output_shape, (4, 1, 248320))
        with self.assertRaises(ValueError):
            handle.complete_read(object(), shape=(2, 1, 248320), row_order=[0, 2])
        with self.assertRaises(ValueError):
            DecodeSubmission(4, "tokens", 2, [0, 0], 8, token_tensor=object())

    def test_no_device_imports(self):
        # Repository pytest fixtures may already import device packages. Isolate
        # the production module import, including its transitive dependencies.
        # -I/-S exclude inherited Python paths and site initialization; only the
        # explicit checkout root is added, so this check also works outside cwd.
        probe = (
            "import sys; sys.path.insert(0, sys.argv[1]); "
            "__import__(sys.argv[2]); "
            "found = sorted(name for name in sys.modules "
            "if name.split('.')[0] in ('ttnn', 'torch', 'vllm')); "
            "assert not found, 'Host-only module imported device packages: ' + repr(found)"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-c",
                probe,
                str(Path(__file__).resolve().parents[4]),
                "models.autoports.ornith_ai_ornith_1_5_9b.tt.decode_contract",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()
