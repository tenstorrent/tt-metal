# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step-5 hardware qualification for the Galaxy Qwen3-32B model.

One decoder layer, prefill 128 / 2048 and one decode step, compared against the
Hugging Face reference for the *same* single layer. It is the deliberate mirror
of ``models/common/tests/models/llama33_70b_galaxy/test_model_wh_galaxy.py``:
same staging, same comparison surface, same failure points, so a divergence
between the two models is attributable to the Qwen-specific parts of the graph.

What is Qwen-specific here, and therefore what this file exists to qualify:

1. **The 64-head decoupled geometry.** ``n_heads * head_dim`` is 8192 while
   ``dim`` is 5120, so ``wo`` is ``[8192, 5120]`` and every attention placement
   is ``attention_dim``-wide rather than ``dim``-wide. Milestone A's recorded
   "Qwen3-32B attention qualified" was measured against a **40-head** fixture
   (``test_attention_2d_wh_galaxy.py:86``, ``dim=5120, n_heads=40``) chosen so
   that ``n_heads * head_dim`` happened to equal ``dim``. The square case is the
   only one with prior silicon evidence, so the decoupled path is treated here
   as unqualified, not as a re-run. ``test_..._geometry_is_decoupled`` states
   the geometry on the device before anything numerical runs.
2. **Per-head Q/K normalization.** ``Attention2D`` applies ``RMSNorm2D`` with
   ``HEAD_LOCAL`` geometry to the created Q and K heads. Milestone A's D2 defect
   was that head-local decode aborted in op validation before producing any
   numerical result at all, so there is no prior Qwen Q/K norm number anywhere.
   ``test_..._qk_norm_head_local`` validates it *alone*, in both modes, with the
   real sub-device manager loaded, before the block gate runs it inside
   attention.
3. **A pinned checkpoint revision**, because Qwen3-32B's published weights have
   moved.

Assumptions this file encodes, in the order they are most likely to be wrong:

1. **Contiguous KV, not paged.** ``paged_attention_config=None`` so the
   comparison isolates the layer graph from paging; paging has its own test.
2. **Prefill writes one column-local user.** ``user_ids=(0,)`` fills local user
   0 of every column shard, so global rows 0, 8, 16 and 24 hold the prefix.
3. **Decode positions are one column wide**, exactly as the qualified attention
   test passed them, while the RoPE indices carry the full physical batch.
4. **Logits are the comparison surface.** ``LMHead2D`` masks padded vocabulary
   to ``-inf``, so only ``[:vocab_size]`` is compared.

Run it as::

    HF_HOME=/localdev/ctr-apbernal/hf_data \
    pytest models/common/tests/models/qwen3_32b_galaxy/test_model_wh_galaxy.py -v

It skips unless the checkpoint is resolvable from the local Hugging Face cache
(or ``QWEN3_32B_HF_MODEL`` names another copy of the same geometry). Only the
shards holding layer 0, the embedding, the final norm and the LM head are read,
so a fresh process costs seconds rather than a whole-checkpoint load. **A run
that skipped is a failure of the run, not a result**: `HF_HOME` reaches Qwen3-32B
only under ``/localdev/ctr-apbernal/hf_data``.
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
from models.common.models.qwen3_32b_galaxy.hf_adaptor import DEFAULT_HF_MODEL, convert_hf_model_weights
from models.common.models.qwen3_32b_galaxy.model import (
    DEFAULT_HF_REVISION,
    QWEN3_32B_GALAXY_ACCURACY,
    _relocate,
    build_qwen3_32b_galaxy_model,
    parameters_from_hf_config,
)
from models.common.models.qwen3_32b_galaxy.weight_utils import reverse_permute_1d as qwen_reverse_permute_1d
from models.common.modules.lazy_weight import LazyWeight
from models.common.tests.models.galaxy.galaxy_checkpoint import CheckpointUnavailable, load_layer_subset_causal_lm
from models.common.tests.models.galaxy.galaxy_hardware import hf_config_or_skip
from models.common.tests.modules._hf_reference import reverse_permute_1d
from models.common.utility_functions import comp_pcc

_MESH_SHAPE = (8, 4)
_MESH_ROWS, _MESH_COLUMNS = _MESH_SHAPE
_PHYSICAL_BATCH = 32
_PREFILL_LENGTH = 128
_LONG_PREFILL_LENGTH = 2048
_MAX_SEQ_LEN = 2048
_PCC = 0.99

_DEVICE_PARAMS = {
    "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
}


def _hf_model() -> str:
    return os.getenv("QWEN3_32B_HF_MODEL", DEFAULT_HF_MODEL)


def _hf_revision() -> str | None:
    """A user-supplied checkpoint copy is not the pinned revision."""

    return DEFAULT_HF_REVISION if _hf_model() == DEFAULT_HF_MODEL else None


def _one_layer_reference(hf_model: str) -> Any:
    """Return the checkpoint's first decoder layer as a runnable causal LM.

    The same one-layer module supplies both the TT weights and the reference
    logits, so a weight-conversion error cannot cancel itself out across the two
    sides of the comparison.

    This reads only the safetensors shards that hold layer 0, the embedding, the
    final norm and the LM head - 3 of Qwen3-32B's 17 - rather than materialising
    all 62 GB of a 64-layer checkpoint to keep 1/64th of it. The
    three-runs-in-fresh-processes rule multiplies that cost by three.
    """

    try:
        hf = load_layer_subset_causal_lm(hf_model, layer_indices=(0,), revision=_hf_revision())
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
    release = getattr(tensor, "deallocate", None)
    if callable(release):
        release(True)


def _logits(output: ttnn.Tensor, vocab_size: int, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """Compose device logits and drop the masked vocabulary padding.

    `to_torch_auto_compose` is wrong for this tensor and wrong *silently*; see
    `compose_galaxy_logits`, which carries the measurement. It concatenated the
    four mesh columns along the vocabulary axis instead of the eight rows, and a
    caller that slices `[:, :vocab_size]` would have seen no error.
    """

    return compose_galaxy_logits(output, mesh_device=mesh_device, vocab_size=vocab_size)


def _compose_residual(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """Compose a `[1, 1, rows, local_dim]` residual-stream tensor to `[rows, dim]`.

    The residual stream is sharded over mesh *columns* on its last axis
    (`local_dim = dim / 4 = 1280`) and replicated over mesh rows, so mesh rows
    stack on the free leading axis and row 0 is the authoritative copy.
    """

    composed = ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=_MESH_SHAPE),
    ).float()
    first_row = composed[0]
    return first_row.reshape(-1, first_row.shape[-1])


def _compose_head_local(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice, *, rows: int) -> list[torch.Tensor]:
    """Return all 32 devices' copies of a replicated head-local tensor.

    A per-head Q/K norm input is `head_dim` wide on every device and carries no
    sharded axis at all in this test - it is replicated over the whole mesh - so
    there is no composition to do, only an unstacking. Mesh rows go to the
    (size-1) leading axis and mesh columns to the row axis, which makes device
    `(r, c)`'s copy `composed[r, c * rows : (c + 1) * rows]`.

    All 32 are returned rather than device (0, 0)'s alone because a head-local
    program that ran on only part of the mesh would otherwise read as a pass.
    """

    composed = ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=_MESH_SHAPE),
    ).float()
    return [
        composed[row, column * rows : (column + 1) * rows]
        for row in range(_MESH_ROWS)
        for column in range(_MESH_COLUMNS)
    ]


def _compose_decode_rot_mat(tensor: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> torch.Tensor:
    """Compose a decode `(cos, sin)` table to `[batch, head_dim]`."""

    composed = ttnn.to_torch(
        tensor,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 2), mesh_shape=_MESH_SHAPE),
    ).float()
    first_row = composed[0]
    return first_row.reshape(-1, first_row.shape[-1])


def _report_pcc(expected: torch.Tensor, actual: torch.Tensor, case: str) -> None:
    """Compute and print a PCC without asserting on it.

    Used for the bisection boundaries: asserting on the first boundary that
    diverges hides the shape of everything after it, and the shape is the
    diagnosis.
    """

    _, message = comp_pcc(expected.float(), actual.float(), _PCC)
    print(f"[bisect] {case}: {message}", flush=True)


def _assert_pcc(expected: torch.Tensor, actual: torch.Tensor, case: str) -> None:
    """Compare and *record*. A passing gate is a number, not a silence."""

    passing, message = comp_pcc(expected.float(), actual.float(), _PCC)
    print(f"[pcc] {case}: {message} (gate >= {_PCC})", flush=True)
    assert passing, f"{case} failed PCC>={_PCC}: {message}"


def _reference_logits_and_cache(hf: Any, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return HF's ``(logits, K, V)`` for the one-layer reference.

    ``use_cache=True`` makes the reference hand back the layer's own KV cache:
    ``K`` is post-RoPE (and, for Qwen3, post per-head K norm) and ``V`` is the
    raw value projection, exactly what the device writes.
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
    writes a different KV head into each one.
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
        # appended row isolates the write itself.
        _report_pcc(
            permuted_k[0, :, : length - 1, :], actual_k[user, :, : length - 1, :], f"{case} K prefix user {user}"
        )
        _report_pcc(permuted_k[0, :, :length, :], actual_k[user, :, :length, :], f"{case} K user {user}")
        _report_pcc(
            permuted_k[0, :, length - 1, :], actual_k[user, :, length - 1, :], f"{case} K appended row user {user}"
        )
        _report_pcc(
            expected_v[0, :, : length - 1, :], actual_v[user, :, : length - 1, :], f"{case} V prefix user {user}"
        )
        _report_pcc(
            expected_v[0, :, length - 1, :], actual_v[user, :, length - 1, :], f"{case} V appended row user {user}"
        )
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
    catches a column that silently wrote nothing.
    """

    actual_k = _compose_kv(kv_pair[0], mesh_device)
    actual_v = _compose_kv(kv_pair[1], mesh_device)
    # The device K cache is post-RoPE in **Meta interleaved** head-dim order
    # (r0, i0, r1, i1, ...); HF's `past_key_values` keys are in HF's split order.
    # The adaptor converts wq/wk with `reverse_permute`, the per-head Q/K norm
    # weights with `reverse_permute_1d`, and the cos/sin tables with
    # `permute_hf_rope_to_meta_tables`, precisely so the device runs the Meta
    # convention; the two conventions cancel inside Q.K^T, which is why the
    # *logits* can match while the raw caches do not. V is not permuted by
    # either side: `wv_meta = wv_raw`.
    expected_k = reverse_permute_1d(expected_k)
    users_per_column = _PHYSICAL_BATCH // _MESH_COLUMNS
    for user in range(0, _PHYSICAL_BATCH, users_per_column):
        _assert_pcc(expected_k[0, :, :length, :], actual_k[user, :, :length, :], f"{case} K user {user}")
        _assert_pcc(expected_v[0, :, :length, :], actual_v[user, :, :length, :], f"{case} V user {user}")


def _params(hf_config: Any, *, prefill_length: int) -> Any:
    return parameters_from_hf_config(
        hf_config,
        n_layers=1,
        max_seq_len=_MAX_SEQ_LEN,
        prefill_sequence_lengths=(prefill_length,),
    )


def _print_geometry(params: Any) -> None:
    """State the decoupled geometry in the log before any number depends on it."""

    geometry = params.geometry()
    print(
        f"[geometry] dim={params.dim} n_heads={params.n_heads} head_dim={params.head_dim} "
        f"attention_dim={params.attention_dim} (dim*{params.attention_dim / params.dim:.2f})\n"
        f"[geometry] local_dim={geometry.local_dim} local_attention_dim={geometry.local_attention_dim} "
        f"local_qkv_size={geometry.local_qkv_size} local_hidden_dim={geometry.local_hidden_dim}\n"
        f"[geometry] wo is [{params.attention_dim}, {params.dim}], per mesh row "
        f"[{geometry.local_attention_dim}, {geometry.local_dim}]",
        flush=True,
    )


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_geometry_is_decoupled_8x4_qwen3_32b(mesh_device: ttnn.MeshDevice):
    """The 64-head geometry, resolved on the real mesh, before anything numerical.

    `local_qkv_size == local_dim == 1280` for this model, so a confusion between
    the fused-QKV width and the residual width is shape-invisible;
    `local_attention_dim` (1024) is the one that differs, and it is what `wo`'s
    DRAM-sharded placement must be built from. This test states all three and
    checks the `wo` placement against `local_attention_dim`, so a later PCC
    failure cannot be blamed on a geometry nobody wrote down.
    """

    from models.common.models.galaxy.recipes import dram_sharded_weight_memory_config

    hf_config = hf_config_or_skip(_hf_model(), revision=_hf_revision())
    params = _params(hf_config, prefill_length=_PREFILL_LENGTH)
    _print_geometry(params)
    geometry = params.geometry()

    assert params.attention_dim == params.n_heads * params.head_dim == 8192
    assert params.dim == 5120 and params.attention_dim != params.dim
    assert geometry.local_attention_dim == 8192 // _MESH_ROWS == 1024
    assert geometry.local_dim == 5120 // _MESH_COLUMNS == 1280
    # The trap: these two are equal for Qwen3-32B, so no shape check can catch a
    # confusion between them.
    assert geometry.local_qkv_size == geometry.local_dim == 1280

    wo_memcfg = dram_sharded_weight_memory_config(mesh_device, geometry.local_attention_dim, geometry.local_dim)
    dim_memcfg = dram_sharded_weight_memory_config(mesh_device, geometry.local_dim, geometry.local_dim)
    print(f"[geometry] wo memcfg (local_attention_dim) : {wo_memcfg}", flush=True)
    print(f"[geometry] wo memcfg if dim were used      : {dim_memcfg}", flush=True)
    assert wo_memcfg != dim_memcfg, "a dim-vs-attention_dim confusion would be invisible in the wo placement"


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_qk_norm_head_local_8x4_qwen3_32b_decode_and_prefill(mesh_device: ttnn.MeshDevice):
    """The per-head Q/K norm, alone, in both modes, against the HF reference.

    Milestone A's D2 defect was that a `HEAD_LOCAL` decode aborted in op
    validation before producing any numerical result, so no prior Qwen Q/K norm
    number exists anywhere. The brief asks for this to be validated *before* it
    is trusted inside the block, because a failure here and a failure in the
    decoupled attention geometry look identical from the block's logits.

    It runs with the real one-layer model built - so the prefetcher's sub-device
    managers are loaded and `activate()` has partitioned the grid - but calls
    `Attention2D`'s own `RMSNorm2D` instances directly. A head-local norm builds
    a plain `ttnn.rms_norm` program, and whether that program is placeable under
    the decode manager is exactly the open question D2 left.

    The device weight is `reverse_permute_1d` of HF's, because the created heads
    are in Meta interleaved order. RMSNorm scales elementwise after a mean-square
    over the whole 128-wide head, so the permutation commutes with the norm: the
    reference is `permute(hf_norm(x))` against `device_norm(permute(x))`.
    """

    hf_config = hf_config_or_skip(_hf_model(), revision=_hf_revision())
    params = _params(hf_config, prefill_length=_PREFILL_LENGTH)
    precision = QWEN3_32B_GALAXY_ACCURACY
    ttnn.SetDefaultDevice(mesh_device)
    torch.manual_seed(17)

    hf = _one_layer_reference(_hf_model())
    weights = convert_hf_model_weights(hf, params=params)
    reference_q_norm = hf.model.layers[0].self_attn.q_norm
    reference_k_norm = hf.model.layers[0].self_attn.k_norm
    assert reference_q_norm is not None and reference_k_norm is not None, "Qwen3-32B must carry per-head Q/K norms"
    print(
        f"[qk-norm] q_norm weight {tuple(reference_q_norm.weight.shape)} "
        f"k_norm weight {tuple(reference_k_norm.weight.shape)} eps={params.rms_norm_eps}",
        flush=True,
    )
    assert int(reference_q_norm.weight.shape[-1]) == params.head_dim

    model = build_qwen3_32b_galaxy_model(
        mesh_device,
        params=params,
        weights=weights,
        precision=precision,
        paged_attention_config=None,
        enable_device_sampling=False,
    )
    del weights
    gc.collect()

    # `[1, rows, heads, head_dim]` is the shape `nlp_create_qkv_heads_decode`
    # hands the norm: eight local Q heads and one local K head per mesh row.
    local_q_heads = params.n_heads // _MESH_ROWS
    local_k_heads = params.n_kv_heads // _MESH_ROWS
    try:
        attention = model.layers[0].attention
        assert attention._q_norm is not None and attention._k_norm is not None

        for mode in ("prefill", "decode"):
            model.activate(mode)
            forward = "decode_forward" if mode == "decode" else "prefill_forward"
            rows = _PHYSICAL_BATCH if mode == "decode" else _PREFILL_LENGTH
            for name, norm, reference, heads in (
                ("q_norm", attention._q_norm, reference_q_norm, local_q_heads),
                ("k_norm", attention._k_norm, reference_k_norm, local_k_heads),
            ):
                source_hf = torch.randn(1, rows, heads, params.head_dim, dtype=torch.float32)
                with torch.no_grad():
                    expected_hf = reference(source_hf.to(torch.bfloat16)).float()
                expected = qwen_reverse_permute_1d(expected_hf)
                source = qwen_reverse_permute_1d(source_hf).to(torch.bfloat16)
                staged = out = None
                print(f"[stage] {mode} {name} rows={rows} heads={heads} enter", flush=True)
                try:
                    staged = ttnn.from_torch(
                        source,
                        device=mesh_device,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                        dtype=precision.norm_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    out = getattr(norm, forward)(staged)
                    print(f"[stage] {mode} {name} returned {tuple(out.shape)} {out.memory_config()}", flush=True)
                    copies = _compose_head_local(out, mesh_device, rows=rows)
                    flat_expected = expected.reshape(-1, params.head_dim)
                    for index, copy in enumerate(copies):
                        _assert_pcc(
                            flat_expected,
                            copy.reshape(-1, params.head_dim),
                            f"{mode} {name} device {index // _MESH_COLUMNS},{index % _MESH_COLUMNS}",
                        )
                except BaseException:
                    traceback.print_exc()
                    raise
                finally:
                    _deallocate(out)
                    _deallocate(staged)
    finally:
        try:
            model.close()
        finally:
            del model
            gc.collect()
            del hf, reference_q_norm, reference_k_norm
            gc.collect()


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_one_layer_prefill_and_decode_8x4_qwen3_32b_b32_s128(mesh_device: ttnn.MeshDevice):
    """The Milestone B step-5 gate: prefill 128 and decode, logits and both caches."""

    hf_model = _hf_model()
    hf_config = hf_config_or_skip(hf_model, revision=_hf_revision())
    params = _params(hf_config, prefill_length=_PREFILL_LENGTH)
    _print_geometry(params)
    precision = QWEN3_32B_GALAXY_ACCURACY
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

    model = build_qwen3_32b_galaxy_model(
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
        print("[stage] prefill 128 enter", flush=True)
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
        print("[stage] decode 128 enter", flush=True)
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


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_one_layer_prefill_2048_8x4_qwen3_32b_b1_s2048(mesh_device: ttnn.MeshDevice):
    """Single-row prefill at the full 2048-token recipe, logits and cache.

    Separate from the 128 case rather than parametrized with it, because the
    recipe family is keyed by sequence length: a 2048 prefill resolves a
    different attention program config, a different SDPA geometry and a
    different collective plan.
    """

    hf_model = _hf_model()
    hf_config = hf_config_or_skip(hf_model, revision=_hf_revision())
    params = _params(hf_config, prefill_length=_LONG_PREFILL_LENGTH)
    precision = QWEN3_32B_GALAXY_ACCURACY
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

    model = build_qwen3_32b_galaxy_model(
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
        print("[stage] prefill 2048 enter", flush=True)
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


def _reference_decode_stages(hf: Any, tokens: torch.Tensor, position: int) -> dict[str, torch.Tensor]:
    """Return HF's own tensors at each boundary the device graph crosses."""

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
    return {
        "embedding": embedded[0, position],
        "attention norm": captured["attention norm"][0, position],
        "attention out": captured["attention out"][0, position],
        "residual after attention": (embedded + captured["attention out"])[0, position],
        "ff norm": captured["ff norm"][0, position],
        "mlp out": captured["mlp out"][0, position],
        "after layer 0": out.hidden_states[1].float()[0, position],
        "final norm": hf.model.norm(out.hidden_states[1])[0].float()[position],
        "logits": out.logits.float()[0, position],
    }


@pytest.mark.parametrize("device_params", [_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [pytest.param(_MESH_SHAPE, id="8x4")], indirect=True)
@torch.no_grad()
def test_qwen3_32b_galaxy_decode_bisection_8x4_qwen3_32b_b32_s128(mesh_device: ttnn.MeshDevice):
    """Bisect one decode step by sub-module boundary against the same HF reference.

    It runs the decode step by hand - embedding, layer, final norm, LM head - and
    reports the PCC at each boundary. **It reports rather than asserts on the
    intermediates**, because the point is to see where a chain diverges. The
    logits assertion at the end is the one that fails the test.

    Prefill runs first and must be correct: the decode step at position 128 reads
    the cache prefill wrote, so a decode-only comparison could not tell a bad
    decode from a bad cache.
    """

    hf_model = _hf_model()
    hf_config = hf_config_or_skip(hf_model, revision=_hf_revision())
    params = _params(hf_config, prefill_length=_PREFILL_LENGTH)
    _print_geometry(params)
    precision = QWEN3_32B_GALAXY_ACCURACY
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

    model = build_qwen3_32b_galaxy_model(
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

        print("[stage] bisect prefill enter", flush=True)
        model.activate("prefill")
        rot_mats = model.prepare_prefill_rot_mats(0, _PREFILL_LENGTH)
        try:
            x_embed = model.embed_prefill(_replicated_tokens(prefill_tokens, mesh_device))
            output = model.prefill_forward(x_embed, rot_mats, sequence_length=_PREFILL_LENGTH, user_ids=(0,))
            _deallocate(output)
        finally:
            for tensor in rot_mats:
                _deallocate(tensor)

        print("[stage] bisect decode enter", flush=True)
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
            from models.common.models.qwen3_32b_galaxy.model import DecodeMetadata

            metadata = DecodeMetadata(current_positions=tt_positions, page_table=None)

            decode_prefetch = model.resources.prefetcher.context("decode")
            print(
                f"[probe] decode global_cb bound: {getattr(decode_prefetch, 'global_cb', None) is not None}", flush=True
            )

            cos_host = _compose_decode_rot_mat(rot_mats[0], mesh_device)
            sin_host = _compose_decode_rot_mat(rot_mats[1], mesh_device)
            print(f"[probe] decode cos composed {tuple(cos_host.shape)}", flush=True)
            for user in (0, 8, 16, 24):
                _report_pcc(reference_cos[0, 0, _PREFILL_LENGTH], cos_host[user], f"probe decode cos user {user}")
                _report_pcc(reference_sin[0, 0, _PREFILL_LENGTH], sin_host[user], f"probe decode sin user {user}")

            x_embed = model.embed_decode(_replicated_tokens(decode_row, mesh_device))
            embedded = _compose_residual(x_embed, mesh_device)
            _report_pcc(stages["embedding"], embedded[0], "bisect decode embedding user 0")

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

            logits = model.lm_head.decode_forward(_relocate(normed, model.config.lm_head_config.decode_input_memcfg))
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
