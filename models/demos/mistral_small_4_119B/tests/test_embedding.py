# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Embedding parity tests.

Uses the root ``device`` fixture (``ttnn.CreateDevice`` → ``MeshDevice::create_unit_mesh``), not
``mesh_device`` / ``open_mesh_device``, to avoid heavier Open MPI bring-up on single-chip runs.

There are **two** test functions—``Mistral4Embedding1D`` and ``Mistral4Embedding2D``—so pytest
reports two nodes (clear pass/fail per implementation). Each test opens its own device session
(function-scoped ``device``); that duplicates setup vs one combined test but keeps logs and CI
filters obvious (``-k embedding2d``).

**Synthetic tests** (``*_matches_torch``): tiny ``Mistral4Config``, PyTorch-default embedding weights,
random token ids.

**Checkpoint tests** (``*_checkpoint_matches_torch``): real ``embed_tokens.weight`` from
``models/mistral_small_4/`` when a sharded snapshot is present; skipped otherwise. Token ids use
``torch.manual_seed(42)`` then ``randint`` for reproducibility.
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_small_4_119B.tt.decoder_checkpoint import read_embed_tokens_checkpoint_tensor
from models.demos.mistral_small_4_119B.tt.embedding import Mistral4Embedding1D, Mistral4Embedding2D
from models.demos.mistral_small_4_119B.tt.moe.moe import mistral4_text_config_from_snapshot
from models.demos.mistral_small_4_119B.tt_utils.abstract_module import AbstractModule
from models.demos.mistral_small_4_119B.tt_utils.ccl import CCL
from models.demos.mistral_small_4_119B.tt_utils.run_config import create_run_config


def _tiny_mistral4_config():
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")
    from transformers.models.mistral4.configuration_mistral4 import Mistral4Config

    return Mistral4Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=8,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        max_position_embeddings=4096,
        kv_lora_rank=8,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        qk_nope_head_dim=8,
    )


def _assert_hidden_dim_pcc(tt_output: torch.Tensor, reference_output: torch.Tensor, *, pcc_required: float) -> None:
    tt_out = tt_output.cpu().float()
    ref_out = reference_output.cpu().float()

    while tt_out.ndim < ref_out.ndim:
        tt_out = tt_out.unsqueeze(0)
    while ref_out.ndim < tt_out.ndim:
        ref_out = ref_out.unsqueeze(0)

    seq_or_batch = min(tt_out.shape[-2], ref_out.shape[-2])
    tt_out = tt_out[..., :seq_or_batch, :]
    ref_out = ref_out[..., :seq_or_batch, :]

    passing, pcc = comp_pcc(tt_out, ref_out, pcc_required)
    logger.info(f"embedding PCC: {pcc}")
    assert passing, f"PCC {pcc} < required {pcc_required}"


def _iter_weight_tensors(weight_config):
    stack = [weight_config]
    while stack:
        node = stack.pop()
        if isinstance(node, ttnn.Tensor):
            yield node
        elif isinstance(node, dict):
            stack.extend(node.values())
        elif isinstance(node, (list, tuple)):
            stack.extend(node)


_EMBEDDING_CASES = (
    ("decode", 1),
    ("decode", 32),
    ("prefill", 64),
)


def _run_embedding_parity(
    embedding_cls: type[AbstractModule],
    mesh_device: ttnn.MeshDevice,
    tmp_path_weight_dir,
) -> None:
    hf_config = _tiny_mistral4_config()
    assert hf_config.hidden_size % ttnn.TILE_SIZE == 0

    reference = torch.nn.Embedding(hf_config.vocab_size, hf_config.hidden_size, dtype=torch.float32).eval()
    hf_state_dict = reference.state_dict()

    ccl = CCL(mesh_device) if mesh_device.get_num_devices() > 1 else None
    weight_config = embedding_cls.convert_weights(
        hf_config,
        (hf_state_dict,),
        tmp_path_weight_dir,
        mesh_device,
    )
    model_state = embedding_cls.create_state(hf_config, mesh_device, ccl)

    try:
        for mode, seq_len in _EMBEDDING_CASES:
            model_config = (
                embedding_cls.decode_model_config(hf_config, mesh_device)
                if mode == "decode"
                else embedding_cls.prefill_model_config(hf_config, mesh_device)
            )
            run_config = create_run_config(model_config, weight_config, model_state)

            torch_input = torch.randint(0, hf_config.vocab_size, (1, 1, seq_len), dtype=torch.int64)
            reference_output = reference(torch_input).to(torch.bfloat16)

            tt_input_ids = ttnn.from_torch(
                torch_input.to(torch.int32),
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            tt_output = (
                embedding_cls.forward_decode(tt_input_ids, run_config)
                if mode == "decode"
                else embedding_cls.forward_prefill(tt_input_ids, run_config)
            )

            tt_output_torch = ttnn.to_torch(
                tt_output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device,
                    dims=(0, -1),
                    mesh_shape=tuple(mesh_device.shape),
                ),
            )
            tt_output_torch = tt_output_torch[:1]

            ttnn.deallocate(tt_input_ids)
            ttnn.deallocate(tt_output)

            logger.info(f"{embedding_cls.__name__} mode={mode} seq_len={seq_len}")
            _assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)
    finally:
        for tensor in _iter_weight_tensors(weight_config):
            ttnn.deallocate(tensor)


def _mistral4_snapshot_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "mistral_small_4"


def _hf_config_and_embed_weight_from_snapshot() -> tuple[object, torch.Tensor]:
    pytest.importorskip("transformers.models.mistral4.configuration_mistral4")

    snapshot_dir = _mistral4_snapshot_dir()
    if not (snapshot_dir / "config.json").is_file():
        pytest.skip("No config.json under models/mistral_small_4/ (install snapshot per download_model.py)")
    if not (snapshot_dir / "model.safetensors.index.json").is_file():
        pytest.skip("No model.safetensors.index.json (snapshot incomplete)")

    try:
        embed_w = read_embed_tokens_checkpoint_tensor(snapshot_dir)
    except (FileNotFoundError, KeyError, RuntimeError, TypeError) as exc:
        pytest.skip(f"Could not read embed_tokens from snapshot: {exc}")

    hf_config = mistral4_text_config_from_snapshot(snapshot_dir)
    if embed_w.shape != (hf_config.vocab_size, hf_config.hidden_size):
        pytest.skip(
            f"embed shape {tuple(embed_w.shape)} vs config (vocab={hf_config.vocab_size}, hidden={hf_config.hidden_size})"
        )
    if hf_config.hidden_size % ttnn.TILE_SIZE != 0:
        pytest.skip(f"hidden_size {hf_config.hidden_size} not divisible by TILE_SIZE")

    return hf_config, embed_w


def _run_embedding_parity_checkpoint(
    embedding_cls: type[AbstractModule],
    mesh_device: ttnn.MeshDevice,
    tmp_path_weight_dir: Path,
) -> None:
    hf_config, embed_w = _hf_config_and_embed_weight_from_snapshot()

    reference = torch.nn.Embedding(hf_config.vocab_size, hf_config.hidden_size, dtype=torch.float32).eval()
    with torch.no_grad():
        reference.weight.copy_(embed_w.to(torch.float32))

    hf_state_dict = {"weight": reference.weight.detach().clone().contiguous()}
    ccl = CCL(mesh_device) if mesh_device.get_num_devices() > 1 else None
    weight_config = embedding_cls.convert_weights(
        hf_config,
        (hf_state_dict,),
        tmp_path_weight_dir,
        mesh_device,
    )
    model_state = embedding_cls.create_state(hf_config, mesh_device, ccl)

    torch.manual_seed(42)
    try:
        for mode, seq_len in _EMBEDDING_CASES:
            model_config = (
                embedding_cls.decode_model_config(hf_config, mesh_device)
                if mode == "decode"
                else embedding_cls.prefill_model_config(hf_config, mesh_device)
            )
            run_config = create_run_config(model_config, weight_config, model_state)

            torch_input = torch.randint(0, hf_config.vocab_size, (1, 1, seq_len), dtype=torch.int64)
            reference_output = reference(torch_input).to(torch.bfloat16)

            tt_input_ids = ttnn.from_torch(
                torch_input.to(torch.int32),
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )

            tt_output = (
                embedding_cls.forward_decode(tt_input_ids, run_config)
                if mode == "decode"
                else embedding_cls.forward_prefill(tt_input_ids, run_config)
            )

            tt_output_torch = ttnn.to_torch(
                tt_output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device,
                    dims=(0, -1),
                    mesh_shape=tuple(mesh_device.shape),
                ),
            )
            tt_output_torch = tt_output_torch[:1]

            ttnn.deallocate(tt_input_ids)
            ttnn.deallocate(tt_output)

            logger.info(f"{embedding_cls.__name__} [checkpoint] mode={mode} seq_len={seq_len}")
            _assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=0.98)
    finally:
        for tensor in _iter_weight_tensors(weight_config):
            ttnn.deallocate(tensor)


def test_mistral4_embedding1d_forward_matches_torch(device, tmp_path):
    """``Mistral4Embedding1D`` vs ``torch.nn.Embedding`` (decode + prefill lengths)."""
    _run_embedding_parity(Mistral4Embedding1D, device, tmp_path / "Mistral4Embedding1D")


def test_mistral4_embedding2d_forward_matches_torch(device, tmp_path):
    """``Mistral4Embedding2D`` vs ``torch.nn.Embedding``; on 1-row meshes matches 1D (no row reduce-scatter)."""
    _run_embedding_parity(Mistral4Embedding2D, device, tmp_path / "Mistral4Embedding2D")


def test_mistral4_embedding1d_forward_checkpoint_matches_torch(device, tmp_path):
    """``Mistral4Embedding1D`` with snapshot ``embed_tokens.weight`` vs torch (when snapshot present)."""
    _run_embedding_parity_checkpoint(Mistral4Embedding1D, device, tmp_path / "Mistral4Embedding1D_ckpt")


def test_mistral4_embedding2d_forward_checkpoint_matches_torch(device, tmp_path):
    """``Mistral4Embedding2D`` with snapshot ``embed_tokens.weight`` vs torch (when snapshot present)."""
    _run_embedding_parity_checkpoint(Mistral4Embedding2D, device, tmp_path / "Mistral4Embedding2D_ckpt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
