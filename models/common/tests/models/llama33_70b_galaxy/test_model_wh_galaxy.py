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
from models.common.models.galaxy.collectives import compose_galaxy_logits
from models.common.models.llama33_70b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, convert_hf_model_weights
from models.common.models.llama33_70b_galaxy.model import (
    LLAMA33_70B_GALAXY_ACCURACY,
    _relocate,
    build_llama33_70b_galaxy_model,
    parameters_from_hf_config,
)
from models.common.modules.lazy_weight import LazyWeight
from models.common.tests.modules._hf_reference import reverse_permute_1d
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


def _logits(output: ttnn.Tensor, vocab_size: int, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """Compose device logits and drop the masked vocabulary padding.

    `to_torch_auto_compose` is wrong for this tensor and wrong *silently*; see
    `compose_galaxy_logits`, which carries the measurement. It concatenated the
    four mesh columns along the vocabulary axis instead of the eight rows, so a
    128-token prefill composed to 128 x 64128 - four copies of row 0's vocabulary
    slice - and the only reason this surfaced at all is that `comp_pcc` compares
    sizes. A caller that slices `[:, :vocab_size]` would have seen no error.
    """

    return compose_galaxy_logits(output, mesh_device=mesh_device, vocab_size=vocab_size)


def _compose_residual(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """Compose a `[1, 1, rows, local_dim]` residual-stream tensor to `[rows, dim]`.

    The residual stream is sharded over mesh *columns* on its last axis
    (`local_dim = dim / 4`) and replicated over mesh rows, so mesh rows stack on
    the free leading axis and row 0 is the authoritative copy. Explicit, not
    `to_torch_auto_compose`, for the reason `compose_galaxy_logits` documents.
    """

    composed = ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=_MESH_SHAPE),
    ).float()
    first_row = composed[0]
    return first_row.reshape(-1, first_row.shape[-1])


def _reference_decode_stages(hf: Any, tokens: torch.Tensor, position: int) -> dict[str, torch.Tensor]:
    """Return HF's own tensors at each boundary the device graph crosses.

    `output_hidden_states=True` gives the embedding output as `hidden_states[0]`
    and the hidden state after layer 0 as `hidden_states[1]`; the final norm is
    applied here with the checkpoint's own module. One HF forward, four
    comparison points, and no hand-written re-implementation anywhere - which is
    the property Milestone A found matters.
    """

    layer = hf.model.layers[0]
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().float()

        return hook

    handles = [
        layer.input_layernorm.register_forward_hook(capture("attention norm")),
        layer.self_attn.register_forward_hook(capture("attention out")),
        layer.post_attention_layernorm.register_forward_hook(capture("ff norm")),
        layer.mlp.register_forward_hook(capture("mlp out")),
    ]
    try:
        out = hf(input_ids=tokens, use_cache=True, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()

    embedded = out.hidden_states[0].float()
    stages = {
        "embedding": embedded[0, position],
        "attention norm": captured["attention norm"][0, position],
        "attention out": captured["attention out"][0, position],
        # The residual after attention, which is what the device's `h` carries.
        "residual after attention": (embedded + captured["attention out"])[0, position],
        "ff norm": captured["ff norm"][0, position],
        "mlp out": captured["mlp out"][0, position],
        "after layer 0": out.hidden_states[1].float()[0, position],
        "final norm": hf.model.norm(out.hidden_states[1])[0].float()[position],
        "logits": out.logits.float()[0, position],
    }
    return stages


def _compose_decode_rot_mat(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """Compose a decode `(cos, sin)` table to `[batch, head_dim]`.

    `RotarySetup2D` shards the *position indices* over mesh columns and replicates
    them over rows, so each device holds `users_per_column` rows of the table.
    Mesh columns therefore concatenate on the user axis and mesh rows stack on the
    free leading axis.
    """

    composed = ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 2), mesh_shape=_MESH_SHAPE),
    ).float()
    first_row = composed[0]
    return first_row.reshape(-1, first_row.shape[-1])


def _report_pcc(expected: torch.Tensor, actual: torch.Tensor, case: str) -> float:
    """Compute and print a PCC without asserting on it.

    Used for the bisection boundaries: asserting on the first boundary that
    diverges hides the shape of everything after it, and the shape is the
    diagnosis. The comparison that fails the test is still an assertion.
    """

    _, message = comp_pcc(expected.float(), actual.float(), _PCC)
    print(f"[bisect] {case}: {message}", flush=True)
    return 0.0


def _assert_pcc(expected: torch.Tensor, actual: torch.Tensor, case: str) -> None:
    """Compare and *record*. A passing gate is a number, not a silence."""

    passing, message = comp_pcc(expected.float(), actual.float(), _PCC)
    print(f"[pcc] {case}: {message} (gate >= {_PCC})", flush=True)
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


def _report_kv_pcc(
    expected_k: torch.Tensor,
    expected_v: torch.Tensor,
    kv_pair: list,
    mesh_device: ttnn.MeshDevice,
    *,
    length: int,
    case: str,
) -> None:
    """The KV comparison of `_assert_kv_pcc`, reported rather than asserted."""

    actual_k = _compose_kv(kv_pair[0], mesh_device)
    actual_v = _compose_kv(kv_pair[1], mesh_device)
    permuted_k = reverse_permute_1d(expected_k)
    for user in (0, 8):
        # Three windows, not one. `length` covers the position this decode step
        # appended; `length - 1` covers only what prefill wrote; and the single
        # appended row isolates the write itself. One number cannot tell "the
        # decode write is garbage" from "the whole cache is garbage", and a single
        # large garbage row is enough to crush the PCC of the whole window.
        _report_pcc(permuted_k[0, :, : length - 1, :], actual_k[user, :, : length - 1, :], f"{case} K prefix user {user}")
        _report_pcc(permuted_k[0, :, :length, :], actual_k[user, :, :length, :], f"{case} K user {user}")
        _report_pcc(permuted_k[0, :, length - 1, :], actual_k[user, :, length - 1, :], f"{case} K appended row user {user}")
        _report_pcc(expected_v[0, :, : length - 1, :], actual_v[user, :, : length - 1, :], f"{case} V prefix user {user}")
        _report_pcc(expected_v[0, :, length - 1, :], actual_v[user, :, length - 1, :], f"{case} V appended row user {user}")
        print(
            f"[probe] {case} user {user}: appended K device |max|="
            f"{float(actual_k[user, :, length - 1, :].abs().max()):.4g} "
            f"reference |max|={float(permuted_k[0, :, length - 1, :].abs().max()):.4g}",
            flush=True,
        )


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
    # The device K cache is post-RoPE in **Meta interleaved** head-dim order
    # (r0, i0, r1, i1, ...); HF's `past_key_values` keys are in HF's split order
    # (r0, r1, ..., i0, i1, ...). The adaptor converts wq/wk with
    # `reverse_permute` and the cos/sin tables with
    # `permute_hf_rope_to_meta_tables` precisely so the device runs the Meta
    # convention, and the two conventions cancel inside Q.K^T - which is why the
    # *logits* match at PCC >= 0.99 while the raw caches do not. Measured on
    # `(8, 4)`: comparing them unpermuted gives
    #     prefill 128 cache K user 0 failed PCC>=0.99: 0.0386
    # V is not permuted by either side: `wv_meta = wv_raw`.
    expected_k = reverse_permute_1d(expected_k)
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
            actual = _logits(output, params.vocab_size, mesh_device)[:_PREFILL_LENGTH]
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
            actual = _logits(output, params.vocab_size, mesh_device)
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
            actual = _logits(output, params.vocab_size, mesh_device)[:_LONG_PREFILL_LENGTH]
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
def test_llama33_70b_galaxy_decode_bisection(mesh_device: ttnn.MeshDevice):
    """Bisect one decode step by sub-module boundary against the same HF reference.

    `job1_llama.md`: "Bisect by sub-module when a block fails. The individual
    modules are qualified, so a block failure is almost certainly composition...
    Compare the residual stream at each boundary against the reference before you
    suspect a module."

    This is that comparison, and it is a separate test rather than instrumentation
    inside the gate so the gate keeps asserting only what it gates. It runs the
    decode step by hand - embedding, layer, final norm, LM head - and reports the
    PCC at each boundary. **It reports rather than asserts on the intermediates**,
    because the point is to see where a chain diverges, and an assertion on the
    first boundary hides the shape of the rest. The logits assertion at the end is
    the one that fails the test.

    Prefill runs first and must be correct: the decode step at position 128 reads
    the cache prefill wrote, so a decode-only comparison could not tell a bad
    decode from a bad cache.
    """

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
    weights = convert_hf_model_weights(hf, params=params)
    stages = _reference_decode_stages(hf, tokens, _PREFILL_LENGTH)
    _, reference_k_full, reference_v_full = _reference_logits_and_cache(hf, tokens)
    # Kept alive on purpose: `layer.mlp` is re-applied below to the device's *own*
    # MLP input, which is the only way to tell "the MLP is a wrong function" from
    # "the MLP was handed a wrong input".
    reference_mlp = hf.model.layers[0].mlp
    reference_cos = weights.rope_cos.float()
    reference_sin = weights.rope_sin.float()

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
        rot_mats = model.prepare_prefill_rot_mats(0, _PREFILL_LENGTH)
        try:
            x_embed = model.embed_prefill(_replicated_tokens(prefill_tokens, mesh_device))
            output = model.prefill_forward(x_embed, rot_mats, sequence_length=_PREFILL_LENGTH, user_ids=(0,))
            _deallocate(output)
        finally:
            for tensor in rot_mats:
                _deallocate(tensor)

        model.activate("decode")
        positions = torch.full((_PHYSICAL_BATCH,), _PREFILL_LENGTH, dtype=torch.long)
        decode_row = decode_token.reshape(1, 1).repeat(1, _PHYSICAL_BATCH)
        rot_mats = model.prepare_decode_rot_mats(positions)
        tt_positions = None
        try:
            tt_positions = ttnn.from_torch(
                positions[: model.geometry.users_per_column].to(torch.int32),
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            from models.common.models.llama33_70b_galaxy.model import DecodeMetadata

            metadata = DecodeMetadata(current_positions=tt_positions, page_table=None)

            # Is the prefetcher's global circular buffer actually bound for decode?
            # It is created lazily now (`defer_global_cb`), and every prefetched
            # weight matmul reads it, so its absence would corrupt attention and
            # the MLP together. Host-side, free, and it removes a whole family of
            # explanations from the table below.
            decode_prefetch = model.resources.prefetcher.context("decode")
            print(f"[probe] decode global_cb bound: {getattr(decode_prefetch, 'global_cb', None) is not None}", flush=True)

            # The decode RoPE tables, against the Meta-layout tables the adaptor
            # built. All 32 positions are `_PREFILL_LENGTH` here, so every row must
            # equal the table's row at that position. `job1_llama.md` ranks the
            # RoPE/Attention2D pairing as the expected first failure, and this
            # separates "the tables are wrong" from "the rotation is applied wrong".
            cos_host = _compose_decode_rot_mat(rot_mats[0], mesh_device)
            sin_host = _compose_decode_rot_mat(rot_mats[1], mesh_device)
            print(f"[probe] decode cos composed {tuple(cos_host.shape)}", flush=True)
            for user in (0, 8, 16, 24):
                _report_pcc(reference_cos[0, 0, _PREFILL_LENGTH], cos_host[user], f"probe decode cos user {user}")
                _report_pcc(reference_sin[0, 0, _PREFILL_LENGTH], sin_host[user], f"probe decode sin user {user}")

            x_embed = model.embed_decode(_replicated_tokens(decode_row, mesh_device))
            embedded = _compose_residual(x_embed, mesh_device)
            _report_pcc(stages["embedding"], embedded[0], "bisect decode embedding user 0")

            # Walk *inside* layer 0 rather than calling `decode_forward`, so every
            # boundary the block crosses is a comparison. The sequence mirrors
            # `Llama33_70BTransformerBlock2D.decode_forward` exactly; if that
            # method changes, this stops being a bisection of it.
            layer = model.layers[0]
            attention_input, h = layer._decode_attention_norm(x_embed, None)
            _report_pcc(
                stages["attention norm"],
                _compose_residual(attention_input, mesh_device)[0],
                "bisect decode attention norm user 0",
            )
            attention_input = _relocate(attention_input, layer.config.decode_attention_input_memcfg)
            attention_output = layer.attention.decode_forward(attention_input, rot_mats, metadata)
            attention_host = _compose_residual(attention_output, mesh_device)
            # All four column-local users, not just user 0. Prefill filled local
            # user 0 of every mesh column, so these four rows should be identical;
            # if they are not, the fault is a placement across columns rather than
            # an arithmetic one inside attention.
            for user in (0, 8, 16, 24):
                _report_pcc(stages["attention out"], attention_host[user], f"bisect decode attention out user {user}")
            # And the cache the decode step just wrote. `reference_k`/`reference_v`
            # cover 0..128 inclusive, so length 129 includes the position this
            # decode step appended. If K at 129 is right and the attention output
            # is not, the fault is on the read side (SDPA, the mask, the length)
            # rather than the write side.
            _report_kv_pcc(
                reference_k_full,
                reference_v_full,
                kv_cache[0],
                mesh_device,
                length=_PREFILL_LENGTH + 1,
                case="bisect decode cache",
            )
            mlp_input, h = layer.ff_norm.decode_forward(attention_output, residual=h)
            _report_pcc(
                stages["residual after attention"],
                _compose_residual(h, mesh_device)[0],
                "bisect decode residual after attention user 0",
            )
            _report_pcc(
                stages["ff norm"],
                _compose_residual(mlp_input, mesh_device)[0],
                "bisect decode ff norm user 0",
            )
            device_mlp_input = _compose_residual(mlp_input, mesh_device)[0]
            mlp_input = _relocate(mlp_input, layer.config.decode_mlp_input_memcfg, layer.config.decode_mlp_input_dtype)
            x = layer.feed_forward.decode_forward(mlp_input)
            device_mlp_out = _compose_residual(x, mesh_device)[0]
            _report_pcc(stages["mlp out"], device_mlp_out, "bisect decode mlp out user 0")
            # The MLP as a *function*: HF's own MLP applied to the device's own
            # input. If this is also low, the MLP is wrong; if it is high, the MLP
            # is faithfully propagating a wrong input.
            with torch.no_grad():
                mlp_on_device_input = reference_mlp(device_mlp_input.unsqueeze(0).to(torch.bfloat16)).float()[0]
            _report_pcc(mlp_on_device_input, device_mlp_out, "probe mlp on the device's own input")
            normed, residual = model.norm.decode_forward(x, residual=h)
            after_layer = _compose_residual(residual, mesh_device)
            normed_host = _compose_residual(normed, mesh_device)
            _report_pcc(stages["after layer 0"], after_layer[0], "bisect decode after layer 0 user 0")
            _report_pcc(stages["final norm"], normed_host[0], "bisect decode final norm user 0")

            logits = model.lm_head.decode_forward(
                _relocate(normed, model.config.lm_head_config.decode_input_memcfg)
            )
            actual = _logits(logits, params.vocab_size, mesh_device)
            _deallocate(logits)
            _assert_pcc(stages["logits"], actual[0], "bisect decode logits user 0")
        except BaseException:
            traceback.print_exc()
            raise
        finally:
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
            del hf, reference_mlp
            gc.collect()
