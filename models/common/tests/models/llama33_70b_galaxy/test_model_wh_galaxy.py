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
(or ``LLAMA33_70B_HF_MODEL`` names another copy of the same geometry). Only the
shards holding layer 0, the embedding, the final norm and the LM head are read
- about 12 GB of the 141 GB checkpoint - so a fresh process costs seconds, not
the ten minutes a whole-checkpoint load costs.
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
from models.common.tests.models.galaxy.galaxy_checkpoint import CheckpointUnavailable, load_layer_subset_causal_lm
from models.common.utility_functions import comp_pcc

_MESH_SHAPE = (8, 4)
_MESH_ROWS, _MESH_COLUMNS = _MESH_SHAPE
_PHYSICAL_BATCH = 32
_PREFILL_LENGTH = 128
_LONG_PREFILL_LENGTH = 2048
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
    """Return the checkpoint's first decoder layer as a runnable causal LM.

    The same one-layer module supplies both the TT weights and the reference
    logits, so a weight-conversion error cannot cancel itself out across the two
    sides of the comparison.

    This reads only the safetensors shards that hold layer 0, the embedding, the
    final norm and the LM head - 3 of Llama-3.3-70B's 30, about 12 GB and 12 GB
    of peak RSS. The `from_pretrained`-then-truncate version this replaced
    materialised all 141 GB of an 80-layer checkpoint to keep 1/80th of it, and
    at roughly ten minutes per process it made the three-runs-in-fresh-processes
    rule unaffordable, which is the whole reason `load_layer_subset_causal_lm`
    exists. Its tensors are bitwise equal to the shards' and it builds the
    rotary module from the checkpoint's own config, so the reference is
    unchanged; only the cost is.
    """

    try:
        hf = load_layer_subset_causal_lm(hf_model, layer_indices=(0,))
    except CheckpointUnavailable as error:
        pytest.skip(str(error))
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


def _reference_logits_and_cache(hf: Any, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return HF's ``(logits, K, V)`` for the one-layer reference.

    ``use_cache=True`` makes the reference hand back the layer's own KV cache,
    which is the independent reference the Milestone B gate wants for the cache
    contents: ``K`` is post-RoPE and ``V`` is the raw value projection, exactly
    what the device writes. Deriving it from a hand-written re-implementation is
    what Milestone A found hides errors on both sides.

    Shapes are ``(1, n_kv_heads, sequence, head_dim)``.
    """

    out = hf(input_ids=tokens, use_cache=True)
    cache = out.past_key_values
    layer = cache.layers[0]
    return out.logits.float()[0], layer.keys.float(), layer.values.float()


def _compose_kv(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """Compose a contiguous KV cache shard into ``(batch, n_kv_heads, seq, head_dim)``.

    The cache is allocated ``(32, n_local_kv_heads, max_seq, head_dim)`` and
    mapped with ``ShardTensor2dMesh(dims=(None, 0))``: mesh *columns* carry
    disjoint users, and the mesh *rows* are allocated as replicas but the model
    writes a different KV head into each one. So the rows must be concatenated
    on the head axis and the columns on the user axis --
    ``dims=(1, 0)`` in ``ConcatMesh2dToTensor``'s
    ``(mesh-row-target, mesh-column-target)`` order.

    `to_torch_auto_compose` is deliberately *not* used here: it would honour the
    mapper's declared row-replication and return one row's heads, silently
    dropping seven eighths of the cache.
    """

    return ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 0), mesh_shape=_MESH_SHAPE),
    ).float()


def _assert_kv_pcc(
    expected_k: torch.Tensor,
    expected_v: torch.Tensor,
    kv_pair: list,
    mesh_device: ttnn.MeshDevice,
    *,
    length: int,
    case: str,
) -> None:
    """Compare the device cache against HF's for every prefilled user row.

    Prefill with ``user_ids=(0,)`` fills local user 0 of each mesh column, so
    global rows 0, 8, 16 and 24 all hold the same prefix. Checking all four
    catches a column that silently wrote nothing, which comparing only row 0
    would not.
    """

    actual_k = _compose_kv(kv_pair[0], mesh_device)
    actual_v = _compose_kv(kv_pair[1], mesh_device)
    users_per_column = _PHYSICAL_BATCH // _MESH_COLUMNS
    for user in range(0, _PHYSICAL_BATCH, users_per_column):
        _assert_pcc(expected_k[0, :, :length, :], actual_k[user, :, :length, :], f"{case} K user {user}")
        _assert_pcc(expected_v[0, :, :length, :], actual_v[user, :, :length, :], f"{case} V user {user}")


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
        reference, reference_k, reference_v = _reference_logits_and_cache(hf, tokens)
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
            # The cache contents, not just the block output: the Milestone B gate
            # is PCC >= 0.99 on both, because a decode step that reads the wrong
            # KV can still produce a passing prefill.
            _assert_kv_pcc(
                reference_k,
                reference_v,
                kv_cache[0],
                mesh_device,
                length=_PREFILL_LENGTH,
                case="prefill 128 cache",
            )
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
            # The decode step appended position 128 to the cache. `reference_k`
            # and `reference_v` already cover 0..128 inclusive, because the HF
            # reference was run over the whole `_PREFILL_LENGTH + 1` row.
            _assert_kv_pcc(
                reference_k,
                reference_v,
                kv_cache[0],
                mesh_device,
                length=_PREFILL_LENGTH + 1,
                case=f"decode position {_PREFILL_LENGTH} cache",
            )
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
def test_llama33_70b_galaxy_one_layer_prefill_2048(mesh_device: ttnn.MeshDevice):
    """Single-row prefill at the full 2048-token recipe, logits and cache.

    Separate from the 128 case rather than parametrized with it, because the
    recipe family is keyed by sequence length: a 2048 prefill resolves a
    different attention program config, a different SDPA geometry and a
    different collective plan, and the point is to exercise those. It carries no
    decode step - position 128's decode is the 128 test's job.
    """

    hf_model = os.getenv("LLAMA33_70B_HF_MODEL", DEFAULT_HF_MODEL)
    hf_config = _hf_config_or_skip(hf_model)
    params = parameters_from_hf_config(
        hf_config,
        n_layers=1,
        max_seq_len=_MAX_SEQ_LEN,
        prefill_sequence_lengths=(_LONG_PREFILL_LENGTH,),
    )
    precision = LLAMA33_70B_GALAXY_ACCURACY
    ttnn.SetDefaultDevice(mesh_device)
    torch.manual_seed(12)
    tokens = torch.randint(0, params.vocab_size, (1, _LONG_PREFILL_LENGTH), dtype=torch.long)

    hf = _one_layer_reference(hf_model)
    try:
        weights = convert_hf_model_weights(hf, params=params)
        reference, reference_k, reference_v = _reference_logits_and_cache(hf, tokens)
    finally:
        del hf
        gc.collect()

    model = build_llama33_70b_galaxy_model(
        mesh_device,
        params=params,
        weights=weights,
        precision=precision,
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
        model.activate("prefill")
        rot_mats = model.prepare_prefill_rot_mats(0, _LONG_PREFILL_LENGTH)
        x_embed = output = None
        try:
            x_embed = model.embed_prefill(_replicated_tokens(tokens, mesh_device))
            output = model.prefill_forward(x_embed, rot_mats, sequence_length=_LONG_PREFILL_LENGTH, user_ids=(0,))
            actual = _logits(output, params.vocab_size)[:_LONG_PREFILL_LENGTH]
            _assert_pcc(reference, actual, "prefill 2048")
            _assert_kv_pcc(
                reference_k,
                reference_v,
                kv_cache[0],
                mesh_device,
                length=_LONG_PREFILL_LENGTH,
                case="prefill 2048 cache",
            )
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            _deallocate(output)
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
