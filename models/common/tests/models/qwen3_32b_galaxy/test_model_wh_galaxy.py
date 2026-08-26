# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step-5 hardware qualification for the Galaxy Qwen3-32B model.

One decoder layer, prefill 128 and one decode step, compared against the
Hugging Face reference for the *same* single layer. It is the deliberate mirror
of ``models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py``:
same staging, same comparison surface, same failure points, so a divergence
between the two models is attributable to the Qwen-specific parts of the graph.

What is Qwen-specific here:

1. **Per-head Q/K normalization.** ``Attention2D`` applies ``RMSNorm2D`` with
   ``HEAD_LOCAL`` geometry to the created Q and K heads. This is its first
   hardware exercise inside the attention module rather than standalone.
2. **Decoupled attention width.** ``n_heads * head_dim`` is 8192 while ``dim``
   is 5120, so ``wo`` is ``[8192, 5120]``. The Milestone A Qwen attention
   evidence used a 40-head fixture where the two coincided.
3. **A pinned checkpoint revision**, because Qwen3-32B's published weights have
   moved.

**This file has never been executed.** It was written without a Galaxy mesh.
Treat every assertion as a hypothesis until it runs. The assumptions it encodes,
most likely to be wrong first:

1. **Contiguous KV, not paged.** The model is built with
   ``paged_attention_config=None`` so the comparison isolates the layer graph
   from paging; the paged path has its own test.
2. **Prefill writes one column-local user.** ``user_ids=(0,)`` fills local user
   0 of every column shard, so global rows 0, 8, 16 and 24 hold the prefix.
3. **Decode positions are one column wide**, exactly as the qualified attention
   test passed them, while the RoPE indices carry the full physical batch.
4. **Logits are the comparison surface.** ``LMHead2D`` masks padded vocabulary
   to ``-inf``, so only ``[:vocab_size]`` is compared.

Run it as::

    pytest models/common/tests/models/qwen3_32b_galaxy/test_model_wh_galaxy.py -v
"""

from __future__ import annotations

import gc
import os
import traceback
from typing import Any

import pytest
import torch

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.models.qwen3_32b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, convert_hf_model_weights
from models.common.models.qwen3_32b_galaxy.model import (
    DEFAULT_HF_REVISION,
    QWEN3_32B_GALAXY_ACCURACY,
    build_qwen3_32b_galaxy_model,
    parameters_from_hf_config,
)
from models.common.modules.lazy_weight import LazyWeight
from models.common.tests.models.galaxy.galaxy_hardware import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    deallocate,
    hf_config_or_skip,
    local_files_only,
)
from models.common.utility_functions import comp_pcc

_MESH_ROWS, _MESH_COLUMNS = GALAXY_MESH_SHAPE
_PREFILL_LENGTH = 128
_MAX_SEQ_LEN = 2048
_PCC = 0.99


def _hf_model() -> str:
    return os.getenv("QWEN3_32B_HF_MODEL", DEFAULT_HF_MODEL)


def _hf_revision() -> str | None:
    return DEFAULT_HF_REVISION if _hf_model() == DEFAULT_HF_MODEL else None


def _one_layer_reference(hf_model: str) -> Any:
    """Load the checkpoint and truncate it to its first decoder layer.

    The same truncated module supplies both the TT weights and the reference
    logits, so a weight-conversion error cannot cancel itself out across the two
    sides of the comparison.
    """

    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        hf_model, revision=_hf_revision(), torch_dtype=torch.bfloat16, local_files_only=local_files_only()
    )
    hf.eval()
    hf.model.layers = torch.nn.ModuleList([hf.model.layers[0]])
    hf.config.num_hidden_layers = 1
    gc.collect()
    return hf


def _replicated_tokens(tokens: torch.Tensor, mesh_device: ttnn.MeshDevice) -> LazyWeight:
    """Stage a `[1, sequence]` token row; Embedding2D replicates and recasts it."""

    return LazyWeight(source=tokens, device=mesh_device)


def _contiguous_kv_cache(
    mesh_device: ttnn.MeshDevice, *, n_layers: int, n_local_kv_heads: int, head_dim: int, dtype: ttnn.DataType
) -> list[list[ttnn.Tensor]]:
    """Allocate one zeroed contiguous K/V pair per layer.

    The cache holds one mesh column's users. Every device starts from the same
    zeros; each mesh row then fills its own KV head slice.
    """

    shape = (GALAXY_PHYSICAL_BATCH // _MESH_COLUMNS, n_local_kv_heads, _MAX_SEQ_LEN, head_dim)
    source = torch.zeros(shape, dtype=torch.bfloat16)
    cache: list[list[ttnn.Tensor]] = []
    for _ in range(n_layers):
        cache.append(
            [
                ttnn.from_torch(
                    source,
                    device=mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            ]
        )
    return cache


def _logits(output: ttnn.Tensor, vocab_size: int) -> torch.Tensor:
    """Compose device logits and drop the masked vocabulary padding."""

    composed = to_torch_auto_compose(output).float()
    return composed.reshape(-1, composed.shape[-1])[:, :vocab_size]


def _assert_pcc(expected: torch.Tensor, actual: torch.Tensor, case: str) -> None:
    passing, message = comp_pcc(expected.float(), actual.float(), _PCC)
    assert passing, f"{case} failed PCC>={_PCC}: {message}"


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_one_layer_prefill_and_decode(mesh_device: ttnn.MeshDevice):
    hf_model = _hf_model()
    hf_config = hf_config_or_skip(hf_model, revision=_hf_revision())
    params = parameters_from_hf_config(
        hf_config,
        n_layers=1,
        max_seq_len=_MAX_SEQ_LEN,
        prefill_sequence_lengths=(_PREFILL_LENGTH,),
    )
    precision = QWEN3_32B_GALAXY_ACCURACY
    ttnn.SetDefaultDevice(mesh_device)
    torch.manual_seed(11)
    tokens = torch.randint(0, params.vocab_size, (1, _PREFILL_LENGTH + 1), dtype=torch.long)
    prefill_tokens, decode_token = tokens[:, :_PREFILL_LENGTH], tokens[:, _PREFILL_LENGTH:]

    hf = _one_layer_reference(hf_model)
    try:
        weights = convert_hf_model_weights(hf, params=params)
        reference = hf(input_ids=tokens).logits.float()[0]
    finally:
        del hf
        gc.collect()
    expected_prefill = reference[:_PREFILL_LENGTH]
    expected_decode = reference[_PREFILL_LENGTH]

    model = build_qwen3_32b_galaxy_model(
        mesh_device,
        params=params,
        weights=weights,
        precision=precision,
        # See assumption 1 in the module docstring: paging has its own test.
        paged_attention_config=None,
        enable_device_sampling=False,
    )
    del weights
    gc.collect()
    kv_cache: list[list[ttnn.Tensor]] = []
    try:
        kv_cache = _contiguous_kv_cache(
            mesh_device,
            n_layers=params.n_layers,
            n_local_kv_heads=params.n_kv_heads // _MESH_ROWS,
            head_dim=params.head_dim,
            dtype=precision.kv_cache_dtype,
        )
        model.set_kv_cache(kv_cache)

        # --- Prefill the single column-local user -------------------------------
        model.activate("prefill")
        rot_mats = model.prepare_prefill_rot_mats(0, _PREFILL_LENGTH)
        output = None
        try:
            x_embed = model.embed_prefill(_replicated_tokens(prefill_tokens, mesh_device))
            output = model.prefill_forward(x_embed, rot_mats, sequence_length=_PREFILL_LENGTH, user_ids=(0,))
            actual = _logits(output, params.vocab_size)[:_PREFILL_LENGTH]
            _assert_pcc(expected_prefill, actual, "prefill 128")
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            deallocate(output)
            for tensor in rot_mats:
                deallocate(tensor)

        # --- One decode step at position 128 -----------------------------------
        model.activate("decode")
        positions = torch.full((GALAXY_PHYSICAL_BATCH,), _PREFILL_LENGTH, dtype=torch.long)
        decode_row = decode_token.reshape(1, 1).repeat(1, GALAXY_PHYSICAL_BATCH)
        rot_mats = model.prepare_decode_rot_mats(positions)
        tt_positions = output = None
        try:
            tt_positions = ttnn.from_torch(
                positions[: model.geometry.users_per_column].to(torch.int32),
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            x_embed = model.embed_decode(_replicated_tokens(decode_row, mesh_device))
            output = model.decode_forward(x_embed, tt_positions, rot_mats)
            actual = _logits(output, params.vocab_size)
            # Assumption 2: prefill filled local user 0 of every column shard.
            for user in range(0, GALAXY_PHYSICAL_BATCH, model.geometry.users_per_column):
                _assert_pcc(expected_decode, actual[user], f"decode position {_PREFILL_LENGTH} user {user}")
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            deallocate(output)
            deallocate(tt_positions)
            for tensor in rot_mats:
                deallocate(tensor)
    finally:
        try:
            model.close()
        finally:
            for pair in kv_cache:
                for tensor in pair:
                    deallocate(tensor)
            del model, kv_cache
            gc.collect()
