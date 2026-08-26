# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step-2 hardware qualification for the Galaxy Llama-3.3-70B model.

One decoder layer, prefill 128 and one decode step, compared against the
Hugging Face reference for the *same* single layer.

**This file has never been executed.** It was written statically, without a
Galaxy mesh, so that the first hardware session starts from a concrete test
rather than an empty file. Treat every assertion as a hypothesis until it runs.
The assumptions it encodes, in the order they are most likely to be wrong:

1. **Contiguous KV, not paged.** The model is built with
   ``paged_attention_config=None``. ``Attention2D`` validates a decode page
   table against ``users = range(max_batch_size)``, i.e. it demands at least 32
   rows, while the Galaxy decode SDPA batch is the eight users of one mesh
   column; the qualified Milestone A attention test therefore also ran
   contiguous. Qualify paging separately.
2. **Prefill writes one column-local user.** ``user_ids=(0,)`` fills local user
   0 of every column shard, so global rows 0, 8, 16 and 24 all hold the prefilled
   prefix; every other row attends to a zeroed cache. Only the prefilled rows
   are compared.
3. **Decode positions are one column wide.** ``current_positions`` carries
   ``users_per_column`` entries, replicated, exactly as the qualified attention
   test passed them, while the RoPE indices carry the full physical batch.
4. **Logits are the comparison surface.** LMHead2D masks the padded vocabulary
   to ``-inf``, so only ``[:vocab_size]`` is compared.

Run it as::

    pytest models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py -v

It skips unless the checkpoint is resolvable from the local Hugging Face cache
(or ``LLAMA33_70B_HF_MODEL`` names another copy of the same geometry). Loading
the checkpoint needs ~140 GB of host memory and several minutes.
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
from models.common.models.llama33_70b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, convert_hf_model_weights
from models.common.models.llama33_70b_galaxy.model import (
    LLAMA33_70B_GALAXY_ACCURACY,
    build_llama33_70b_galaxy_model,
    parameters_from_hf_config,
)
from models.common.modules.lazy_weight import LazyWeight
from models.common.utility_functions import comp_pcc

_MESH_SHAPE = (8, 4)
_MESH_ROWS, _MESH_COLUMNS = _MESH_SHAPE
_PHYSICAL_BATCH = 32
_PREFILL_LENGTH = 128
_MAX_SEQ_LEN = 2048
_PCC = 0.99


def _local_files_only() -> bool:
    return any(
        os.getenv(name, "").lower() in {"1", "true", "yes"} for name in ("CI", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _hf_config_or_skip(hf_model: str) -> Any:
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(hf_model, local_files_only=_local_files_only())
    except BaseException as error:  # noqa: BLE001 - any resolution failure is a skip, not a defect
        pytest.skip(f"Llama-3.3-70B checkpoint {hf_model!r} is unavailable: {error}")


def _one_layer_reference(hf_model: str) -> Any:
    """Load the checkpoint and truncate it to its first decoder layer.

    The same truncated module supplies both the TT weights and the reference
    logits, so a weight-conversion error cannot cancel itself out across the two
    sides of the comparison.
    """

    from transformers import AutoModelForCausalLM

    hf = AutoModelForCausalLM.from_pretrained(
        hf_model, torch_dtype=torch.bfloat16, local_files_only=_local_files_only()
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

    Users shard over the four mesh columns; the row shards are replicas because
    each mesh row owns its own KV head slice of an identically shaped cache.
    """

    shape = (_PHYSICAL_BATCH, n_local_kv_heads, _MAX_SEQ_LEN, head_dim)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=_MESH_SHAPE)
    cache: list[list[ttnn.Tensor]] = []
    for _ in range(n_layers):
        cache.append(
            [
                ttnn.from_torch(
                    torch.zeros(shape, dtype=torch.bfloat16),
                    device=mesh_device,
                    mesh_mapper=mapper,
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            ]
        )
    return cache


def _deallocate(tensor: Any) -> None:
    if tensor is None:
        return
    deallocate = getattr(tensor, "deallocate", None)
    if callable(deallocate):
        deallocate(True)


def _logits(output: ttnn.Tensor, vocab_size: int) -> torch.Tensor:
    """Compose device logits and drop the masked vocabulary padding."""

    composed = to_torch_auto_compose(output).float()
    return composed.reshape(-1, composed.shape[-1])[:, :vocab_size]


def _assert_pcc(expected: torch.Tensor, actual: torch.Tensor, case: str) -> None:
    passing, message = comp_pcc(expected.float(), actual.float(), _PCC)
    assert passing, f"{case} failed PCC>={_PCC}: {message}"


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_llama33_70b_galaxy_one_layer_prefill_and_decode(mesh_device: ttnn.MeshDevice):
    hf_model = os.getenv("LLAMA33_70B_HF_MODEL", DEFAULT_HF_MODEL)
    hf_config = _hf_config_or_skip(hf_model)
    params = parameters_from_hf_config(
        hf_config,
        n_layers=1,
        max_seq_len=_MAX_SEQ_LEN,
        prefill_sequence_lengths=(_PREFILL_LENGTH,),
    )
    precision = LLAMA33_70B_GALAXY_ACCURACY
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

    model = build_llama33_70b_galaxy_model(
        mesh_device,
        params=params,
        weights=weights,
        precision=precision,
        # See assumption 1 in the module docstring: paged decode is unqualified.
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
        x_embed = output = None
        try:
            x_embed = model.embed_prefill(_replicated_tokens(prefill_tokens, mesh_device))
            output = model.prefill_forward(x_embed, rot_mats, sequence_length=_PREFILL_LENGTH, user_ids=(0,))
            actual = _logits(output, params.vocab_size)[:_PREFILL_LENGTH]
            _assert_pcc(expected_prefill, actual, "prefill 128")
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            _deallocate(output)
            for tensor in rot_mats:
                _deallocate(tensor)

        # --- One decode step at position 128 -----------------------------------
        model.activate("decode")
        positions = torch.full((_PHYSICAL_BATCH,), _PREFILL_LENGTH, dtype=torch.long)
        decode_row = decode_token.reshape(1, 1).repeat(1, _PHYSICAL_BATCH)
        rot_mats = model.prepare_decode_rot_mats(positions)
        x_embed = tt_positions = output = None
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
            for user in range(0, _PHYSICAL_BATCH, model.geometry.users_per_column):
                _assert_pcc(expected_decode, actual[user], f"decode position {_PREFILL_LENGTH} user {user}")
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            _deallocate(output)
            _deallocate(tt_positions)
            for tensor in rot_mats:
                _deallocate(tensor)
    finally:
        try:
            model.close()
        finally:
            for pair in kv_cache:
                for tensor in pair:
                    _deallocate(tensor)
            del model, kv_cache
            gc.collect()
