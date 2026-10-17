# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host checks for serving boundaries; these do not qualify device execution."""

import importlib.util
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from ..tt.checkpoint import MODEL_ID, REVISION, resolve_checkpoint
from ..tt.precision import load_precision


def adapter_class():
    name = "models.demos.granite42_30b_qb2.tt.generator_vllm"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "tt/generator_vllm.py")
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "ttnn": SimpleNamespace(FabricConfig=SimpleNamespace(FABRIC_1D_RING=object())),
        name.rsplit(".", 1)[0] + ".generator": SimpleNamespace(GraniteGenerator=object),
        name.rsplit(".", 1)[0] + ".model": SimpleNamespace(GraniteModel=object),
    }
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module.GraniteForCausalLM


class TestServingContract(unittest.TestCase):
    def setUp(self):
        self.adapter = adapter_class()()
        self.adapter.generator = Mock()
        self.adapter.generator.pages_per_row = 8
        self.adapter.generator._bucket.side_effect = lambda n: next(b for b in (1, 8, 16) if n <= b)

    def test_real_vllm_generation_protocol_selects_tt_bridge(self):
        from vllm.model_executor.models.interfaces_base import is_text_generation_model

        cls = adapter_class()
        self.assertTrue(is_text_generation_model(cls))
        with self.assertRaisesRegex(RuntimeError, "initialize_vllm_model"):
            cls(vllm_config=object())
        for method, args in (("embed_input_ids", (None,)), ("forward", (None, None)), ("compute_logits", (None,))):
            with self.subTest(method=method), self.assertRaises(RuntimeError):
                getattr(self.adapter, method)(*args)

    def test_host_logits_exclude_inactive_bucket_rows(self):
        for batch, bucket in ((5, 8), (10, 16)):
            with self.subTest(batch=batch):
                logits = torch.arange(bucket * 17).reshape(bucket, 17).float()
                self.adapter.generator.decode_forward.return_value = logits
                actual = self.adapter.decode_forward(
                    torch.zeros(batch, 1), torch.arange(batch), torch.zeros(batch, 8), [object()], sampling_params=None
                )
                self.assertEqual(actual.shape, (batch, 1, 17))
                torch.testing.assert_close(actual[:, 0], logits[:batch])
                self.assertEqual(self.adapter.generator.decode_forward.call_args.kwargs["sampling_mode"], "host")
                self.adapter.generator.set_sampling_params.assert_not_called()

    def test_host_prefill_preserves_dense_scheduler_order_and_chunk_offsets(self):
        logits = torch.arange(5 * 17).reshape(5, 1, 17).float()
        self.adapter.generator.prefill_forward.return_value = logits
        tokens = torch.arange(5 * 16).reshape(5, 16)
        # The plugin compacts scheduled requests into dense rows; physical KV
        # pages need not be contiguous or ordered with those rows.
        table = torch.tensor([[91, 4], [7, 19], [52, 3], [8, 70], [61, 9]])
        actual = self.adapter.prefill_forward(
            tokens, table, [object()], prompt_lens=[8, 12, 5, 9, 16], start_pos=[4, 8, 1, 5, 12]
        )
        torch.testing.assert_close(actual, logits)
        call = self.adapter.generator.prefill_forward.call_args
        self.assertEqual(call.kwargs["prompt_lens"], [4] * 5)
        for row, start in enumerate((4, 8, 1, 5, 12)):
            torch.testing.assert_close(call.args[0][row], tokens[row, start : start + 4])
        torch.testing.assert_close(call.kwargs["page_table"][:5, :2], table.int())

    def test_scheduler_page_remap_refreshes_without_reloading_tokens(self):
        table = torch.arange(80).reshape(10, 8)
        self.adapter.generator.decode_forward.return_value = torch.zeros(16, 17)
        self.adapter.decode_forward(
            torch.zeros(10, 1), torch.arange(10), table, [object()], reload_inputs=False, reload_page_table=True
        )
        forwarded = self.adapter.generator.decode_forward.call_args.kwargs
        self.assertFalse(forwarded["reload_inputs"])
        self.assertTrue(forwarded["reload_page_table"])
        torch.testing.assert_close(forwarded["page_table"][:10], table.int())
        self.assertEqual(torch.count_nonzero(forwarded["page_table"][10:]), 0)

    def test_greedy_and_stochastic_rows_preserve_sampling_parameters(self):
        params = SimpleNamespace(temperature=[0, 0.7], top_k=[-1, 32], top_p=[1, 0.95], seed=[19, 23])
        self.assertEqual(self.adapter._sampling(params, positions=[4, 12]), "device")
        args = self.adapter.generator.set_sampling_params.call_args.kwargs
        self.assertEqual(args["top_k"][:2], [1, 32])
        self.assertEqual(args["temperature"][:2], [1, 0.7])
        self.assertEqual(args["top_p"][:2], [0, 0.95])
        seeds, positions = self.adapter.generator.reset_sampling_state.call_args.args
        self.assertEqual(seeds[:2], [19, 23])
        self.assertEqual(positions, [4, 12])


class TestPrecisionContract(unittest.TestCase):
    def test_invalid_operator_boundary_is_rejected(self):
        policy = load_precision()
        policy["token_dtype"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "token_dtype"):
            load_precision(policy)

    def test_caller_policy_is_not_mutated(self):
        policy = load_precision()
        loaded = load_precision(policy)
        loaded["weight_groups"]["down"] = "bfloat16"
        self.assertEqual(policy["weight_groups"]["down"], "bfloat4_b")


class TestCheckpointResolution(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)
        (self.path / "config.json").write_text(
            json.dumps({"model_type": "granite", "architectures": ["GraniteForCausalLM"]})
        )
        (self.path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"model.embed_tokens.weight": "model-00001.safetensors"}})
        )
        for name in ("tokenizer.json", "tokenizer_config.json", "model-00001.safetensors"):
            (self.path / name).touch()

    def test_explicit_local_directory_does_not_consult_hub(self):
        download = Mock(side_effect=AssertionError("local checkpoint must not access the hub"))
        with patch.dict(os.environ, {"HF_MODEL": str(self.path)}), patch.dict(
            sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}
        ):
            self.assertEqual(resolve_checkpoint(), self.path)
        download.assert_not_called()

    def test_default_and_repository_id_resolve_pinned_offline_snapshot(self):
        for source in (None, MODEL_ID):
            with self.subTest(source=source):
                download = Mock(return_value=str(self.path))
                with patch.dict(os.environ, {}, clear=True), patch.dict(
                    sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}
                ):
                    if source is not None:
                        os.environ["HF_MODEL"] = source
                    self.assertEqual(resolve_checkpoint(), self.path)
                download.assert_called_once_with(MODEL_ID, revision=REVISION, local_files_only=True)

    def test_missing_shard_fails_before_device_weight_upload(self):
        (self.path / "model-00001.safetensors").unlink()
        with patch.dict(os.environ, {"HF_MODEL": str(self.path)}):
            with self.assertRaisesRegex(FileNotFoundError, "model-00001.safetensors"):
                resolve_checkpoint()

    def test_unrelated_architecture_is_rejected(self):
        (self.path / "config.json").write_text(json.dumps({"model_type": "llama"}))
        with patch.dict(os.environ, {"HF_MODEL": str(self.path)}):
            with self.assertRaisesRegex(ValueError, "GraniteForCausalLM"):
                resolve_checkpoint()
