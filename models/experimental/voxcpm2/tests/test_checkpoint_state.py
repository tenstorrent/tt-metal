# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint selection must use the weights verified against the CUDA oracle."""

from pathlib import Path
import tempfile
import unittest

from safetensors.torch import save_file
import torch

from models.experimental.voxcpm2.validation.manifest import sha256_file
from models.experimental.voxcpm2.validation.replay_component import checkpoint_state


class TestCheckpointState(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.checkpoint = Path(temporary.name)

    def manifest(self, *names):
        return {"checkpoint": {"files": {name: {"sha256": sha256_file(self.checkpoint / name)} for name in names}}}

    def test_unrecorded_safetensors_cannot_override_recorded_binary(self):
        for codec, binary, safe in (
            (False, "pytorch_model.bin", "model.safetensors"),
            (True, "audiovae.pth", "audiovae.safetensors"),
        ):
            with self.subTest(codec=codec):
                expected = {"weight": torch.tensor([1.0, 2.0])}
                torch.save(expected, self.checkpoint / binary)
                manifest = self.manifest(binary)
                save_file({"weight": torch.tensor([-3.0, -4.0])}, self.checkpoint / safe)
                actual = checkpoint_state(self.checkpoint, manifest, codec=codec)
                self.assertTrue(torch.equal(actual["weight"], expected["weight"]))

    def test_recorded_safetensors_load_for_model_and_codec(self):
        for codec, name in (
            (False, "model.safetensors"),
            (True, "audiovae.safetensors"),
        ):
            with self.subTest(codec=codec):
                expected = {"weight": torch.tensor([3.0, 4.0])}
                save_file(expected, self.checkpoint / name)
                actual = checkpoint_state(self.checkpoint, self.manifest(name), codec=codec)
                self.assertTrue(torch.equal(actual["weight"], expected["weight"]))

    def test_unsupported_recorded_weights_do_not_fall_back_to_directory(self):
        (self.checkpoint / "unsupported.bin").write_bytes(b"recorded but unsupported")
        manifest = self.manifest("unsupported.bin")
        for codec, name in (
            (False, "pytorch_model.bin"),
            (True, "audiovae.pth"),
        ):
            with self.subTest(codec=codec):
                torch.save({"weight": torch.ones(2)}, self.checkpoint / name)
                with self.assertRaisesRegex(ValueError, "records no supported weights"):
                    checkpoint_state(self.checkpoint, manifest, codec=codec)

    def test_modified_recorded_weights_are_rejected(self):
        name = "pytorch_model.bin"
        torch.save({"weight": torch.ones(2)}, self.checkpoint / name)
        manifest = self.manifest(name)
        torch.save({"weight": torch.zeros(2)}, self.checkpoint / name)
        with self.assertRaisesRegex(ValueError, "differs from CUDA oracle"):
            checkpoint_state(self.checkpoint, manifest)

    def test_missing_recorded_weights_do_not_fall_back_to_other_files(self):
        name = "pytorch_model.bin"
        torch.save({"weight": torch.ones(2)}, self.checkpoint / name)
        manifest = self.manifest(name)
        (self.checkpoint / name).unlink()
        save_file({"weight": torch.zeros(2)}, self.checkpoint / "model.safetensors")
        with self.assertRaises(FileNotFoundError):
            checkpoint_state(self.checkpoint, manifest)

    def test_binary_state_dict_wrapper_is_supported(self):
        expected = {"weight": torch.ones(2)}
        name = "audiovae.pth"
        torch.save({"state_dict": expected}, self.checkpoint / name)
        actual = checkpoint_state(self.checkpoint, self.manifest(name), codec=True)
        self.assertTrue(torch.equal(actual["weight"], expected["weight"]))
