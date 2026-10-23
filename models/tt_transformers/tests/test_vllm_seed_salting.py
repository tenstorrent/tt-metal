# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``initialize_vllm_text_transformer`` must hand every submesh's ModelArgs
``salt_duplicate_seeds=False``: on the vLLM path, concurrent requests sharing a
seed are independent requests that must reproduce identically. Host-only: the
model, weights and submeshes are stubbed."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


def test_vllm_text_transformer_disables_duplicate_seed_salting():
    from models.tt_transformers.tt import generator_vllm

    hf_config = SimpleNamespace(_name_or_path="meta-llama/Llama-3.1-8B-Instruct")

    def fake_model_args(submesh, **kwargs):
        args = SimpleNamespace(model_name="Llama-3.1-8B", CKPT_DIR="/weights", n_layers=32)
        args.load_state_dict = lambda: {}
        args.weight_cache_path = lambda dtype: "/cache"
        return args

    captured = []

    def fake_transformer(*, args, **kwargs):
        captured.append(args)
        return MagicMock()

    with patch.object(generator_vllm, "ModelArgs", side_effect=fake_model_args), patch.object(
        generator_vllm, "Transformer", side_effect=fake_transformer
    ), patch.object(generator_vllm, "create_submeshes", return_value=[MagicMock(), MagicMock()]):
        tt_model, model_args = generator_vllm.initialize_vllm_text_transformer(
            hf_config, tt_data_parallel=2, mesh_device=MagicMock(), max_batch_size=64, max_seq_len=1024
        )

    assert len(tt_model) == len(model_args) == len(captured) == 2
    assert all(a.salt_duplicate_seeds is False for a in captured)
