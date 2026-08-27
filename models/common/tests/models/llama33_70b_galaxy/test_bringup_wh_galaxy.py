# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Milestone B step-1 hardware bring-up for the Galaxy Llama-3.3-70B model.

Mesh: WH Galaxy ``(8, 4)``. Checkpoint: ``meta-llama/Llama-3.3-70B-Instruct``,
layer 0 only, real weights. Mode: construction and single-step execution.

This file deliberately asserts *nothing about numerics*. The brief's step 1
requires construction, prefetcher sealing, CCL resource resolution and clean
teardown to be proven before a single PCC number is looked at, because each of
the four things below fails in a different and differently-diagnosable way, and
a combined test reports only the first:

``test_one_layer_model_constructs_and_closes``
    The C1/D1 site. ``RMSNorm2D`` resolves its fused decode statistics
    placement from the decode input placement, and ``plans.py`` derives the
    persistent all-gather buffer from the same residual placement. A
    re-introduced independent stats placement raises
    ``ValueError: fused decode stats buffer must be L1-sharded ...`` here, at
    construction, rather than corrupting a decode silently.

``test_one_decode_step_executes``
    Two unproven compositions at once, both recorded as risks:
    ``GalaxyAttentionCollectives.rotary`` issuing the production
    ``rotary_embedding_llama`` against Q/K that ``Attention2D`` produced
    (Milestone A qualified attention with an *identity* rotary), and the
    ring/``gather_in0`` decode QKV matmul running inside the prefetcher's worker
    subdevice partition. The latter is Milestone A limitation L3, recorded there
    as terminal against the old straddling ``(7, 1)`` grid. Either failure is a
    ``TT_FATAL`` or a hang, not a bad number, so it belongs before PCC.

``test_one_prefill_executes``
    The single-row prefill path and its own recipe family.

``test_two_models_in_one_process``
    Milestone A limitation L1: ``Prefetcher2D.cleanup()`` cannot free the global
    circular buffer, so ~55 MB of L1 stays resident until every
    ``Prefetcher2DContext`` handle dies. A second construction in the same
    process is the cheapest probe of whether teardown ordering actually returns
    that L1.

Every test skips - never invents weights - if the checkpoint cannot be resolved.
"""

from __future__ import annotations

import contextlib
import gc
import os
import sys
import traceback

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
from models.common.tests.models.galaxy.galaxy_checkpoint import (
    CheckpointUnavailable,
    load_layer_subset_causal_lm,
    local_files_only,
)
from models.common.tests.models.galaxy.galaxy_hardware import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH_SHAPE,
    GALAXY_PHYSICAL_BATCH,
    deallocate,
)

_MESH_ROWS, _MESH_COLUMNS = GALAXY_MESH_SHAPE
_PREFILL_LENGTH = 128
_MAX_SEQ_LEN = 2048


def _hf_model_name() -> str:
    return os.getenv("LLAMA33_70B_HF_MODEL", DEFAULT_HF_MODEL)


def _params(n_layers: int = 1):
    from transformers import AutoConfig

    hf_model = _hf_model_name()
    try:
        hf_config = AutoConfig.from_pretrained(hf_model, local_files_only=local_files_only())
    except BaseException as error:  # noqa: BLE001
        pytest.skip(f"checkpoint {hf_model!r} is unavailable: {error}")
    return parameters_from_hf_config(
        hf_config,
        n_layers=n_layers,
        max_seq_len=_MAX_SEQ_LEN,
        prefill_sequence_lengths=(_PREFILL_LENGTH,),
    )


@pytest.fixture(scope="module")
def layer0_weights_and_params():
    """Real layer-0 weights, converted once for every test in this module."""

    params = _params(1)
    try:
        hf = load_layer_subset_causal_lm(_hf_model_name(), layer_indices=(0,))
    except CheckpointUnavailable as error:
        pytest.skip(str(error))
    try:
        weights = convert_hf_model_weights(hf, params=params)
    finally:
        del hf
        gc.collect()
    return weights, params


def _contiguous_kv_cache(mesh_device, *, params, dtype) -> list[list[ttnn.Tensor]]:
    """One zeroed contiguous K/V pair per layer.

    Users shard over the four mesh columns; the row shards are replicas because
    each mesh row owns its own KV head slice of an identically shaped cache.
    """

    shape = (GALAXY_PHYSICAL_BATCH, params.n_kv_heads // _MESH_ROWS, _MAX_SEQ_LEN, params.head_dim)
    mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=GALAXY_MESH_SHAPE)
    return [
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
        for _ in range(params.n_layers)
    ]


def _build(mesh_device, weights, params):
    return build_llama33_70b_galaxy_model(
        mesh_device,
        params=params,
        weights=weights,
        precision=LLAMA33_70B_GALAXY_ACCURACY,
        # Paged KV is plan step 7 (mb-coverage); this job qualifies contiguous.
        paged_attention_config=None,
        enable_device_sampling=False,
    )


@contextlib.contextmanager
def _stage(name: str):
    """Name each device stage in the log, and flush before the next one.

    A ``TT_FATAL`` inside a multi-subdevice program leaves the mesh
    un-drainable: teardown then blocks in ``FDMeshCommandQueue``'s destructor,
    the pytest session never reaches its failure summary, and the Python
    traceback is lost with the killed process. Printing and flushing the stage
    name *before* entering it means the last line in the log always identifies
    the call that aborted, with no debugger and no second run.
    """

    print(f"[stage] enter {name}", flush=True)
    try:
        yield
    except BaseException:
        print(f"[stage] FAILED in {name}", flush=True)
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        raise
    print(f"[stage] leave {name}", flush=True)


def _release_kv(kv_cache) -> None:
    for pair in kv_cache:
        for tensor in pair:
            deallocate(tensor)


galaxy = pytest.mark.parametrize("mesh_device", [pytest.param(GALAXY_MESH_SHAPE, id="8x4")], indirect=True)
galaxy_params = pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)


# =============================================================================
# Construction
# =============================================================================


@galaxy_params
@galaxy
@torch.no_grad()
def test_one_layer_model_constructs_and_closes(mesh_device, layer0_weights_and_params):
    """Construction, prefetcher sealing, CCL resolution and teardown."""

    weights, params = layer0_weights_and_params
    ttnn.SetDefaultDevice(mesh_device)
    model = _build(mesh_device, weights, params)
    try:
        assert model.n_layers == 1
        assert model.geometry.max_batch_size == GALAXY_PHYSICAL_BATCH
        assert model.geometry.users_per_column == GALAXY_PHYSICAL_BATCH // _MESH_COLUMNS
        # The prefetcher is sealed by build_llama33_70b_galaxy_model; a sealed
        # prefetcher is what makes the decode weight stream resolvable at all.
        assert model.prefetcher.sealed, "prefetcher was not sealed by construction"
        # Both operation-boundary contexts must *resolve*; resolving is not
        # activating. `activate("decode")` launches the persistent DRAM prefetch
        # program and calls set_sub_device_stall_group, so activating a mode
        # without then running that mode's forward pass - and in particular
        # allocating device tensors underneath a set decode stall group - leaves
        # the command queue undrainable and hangs MeshDevice::close(). That is
        # what an earlier revision of this test did; see the report's D6.
        for mode in ("prefill", "decode"):
            assert model.resources.context(mode) is not None, mode
        kv_cache = _contiguous_kv_cache(mesh_device, params=params, dtype=LLAMA33_70B_GALAXY_ACCURACY.kv_cache_dtype)
        try:
            model.set_kv_cache(kv_cache)
            model.set_kv_cache(None)
        finally:
            _release_kv(kv_cache)
    finally:
        model.close()
        # The prefetcher is the resource owner; an empty tuple after close is
        # the truthful-but-incomplete signal L1 is about (see the L1 test below).
        assert model.prefetcher.owned_resources == (), model.prefetcher.owned_resources
        del model
        gc.collect()


# =============================================================================
# Single-step execution: the two unproven compositions
# =============================================================================


@galaxy_params
@galaxy
@torch.no_grad()
def test_one_decode_step_executes(mesh_device, layer0_weights_and_params):
    """RoPE composed with Attention2D, and the decode matmul on the partition.

    Every device call is its own named stage, so the last ``[stage] enter`` line
    in the log identifies the aborting call even when the mesh is left
    un-drainable and the session never prints a failure summary.
    """

    weights, params = layer0_weights_and_params
    ttnn.SetDefaultDevice(mesh_device)
    torch.manual_seed(21)
    with _stage("build"):
        model = _build(mesh_device, weights, params)
    kv_cache: list[list[ttnn.Tensor]] = []
    try:
        with _stage("allocate kv cache"):
            kv_cache = _contiguous_kv_cache(
                mesh_device, params=params, dtype=LLAMA33_70B_GALAXY_ACCURACY.kv_cache_dtype
            )
        with _stage("bind kv cache"):
            model.set_kv_cache(kv_cache)
        with _stage("activate decode (starts the persistent DRAM prefetch program)"):
            model.activate("decode")
        positions = torch.full((GALAXY_PHYSICAL_BATCH,), 4, dtype=torch.long)
        tokens = torch.randint(0, params.vocab_size, (1, GALAXY_PHYSICAL_BATCH), dtype=torch.long)
        rot_mats = tt_positions = output = None
        try:
            with _stage("prepare decode rot mats (RotarySetup2D.decode_forward)"):
                rot_mats = model.prepare_decode_rot_mats(positions)
            with _stage("stage current positions"):
                tt_positions = ttnn.from_torch(
                    positions[: model.geometry.users_per_column].to(torch.int32),
                    device=mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            with _stage("embed decode"):
                x_embed = model.embed_decode(LazyWeight(source=tokens, device=mesh_device))
            with _stage("decode forward"):
                output = model.decode_forward(x_embed, tt_positions, rot_mats)
            with _stage("synchronize decode"):
                model.synchronize("decode")
            assert output is not None
            # The vocabulary is sharded over the eight mesh rows, so the padded
            # vocabulary is a property of the *composed* logits, not of one
            # device's shard. Asserting it on `output.shape` -- which reports the
            # per-device shard width -- was a hypothesis written without a mesh.
            # Composing is the stronger check: it fails if any row shard is
            # missing or mis-sized, which a local-width assertion cannot see.
            with _stage("compose decode logits"):
                composed = to_torch_auto_compose(output)
            print(f"[shape] device shard {tuple(output.shape)}, composed {tuple(composed.shape)}", flush=True)
            assert composed.shape[-1] == params.padded_vocab_size, (
                f"composed logits are {tuple(composed.shape)}; "
                f"expected a padded vocabulary of {params.padded_vocab_size}"
            )
            assert composed.reshape(-1, composed.shape[-1]).shape[0] >= GALAXY_PHYSICAL_BATCH
        finally:
            deallocate(output)
            deallocate(tt_positions)
            for tensor in rot_mats or ():
                deallocate(tensor)
    finally:
        with _stage("close model"):
            try:
                model.close()
            finally:
                _release_kv(kv_cache)
                del model, kv_cache
                gc.collect()


@galaxy_params
@galaxy
@torch.no_grad()
def test_one_prefill_executes(mesh_device, layer0_weights_and_params):
    weights, params = layer0_weights_and_params
    ttnn.SetDefaultDevice(mesh_device)
    torch.manual_seed(22)
    with _stage("build"):
        model = _build(mesh_device, weights, params)
    kv_cache: list[list[ttnn.Tensor]] = []
    try:
        with _stage("allocate kv cache"):
            kv_cache = _contiguous_kv_cache(
                mesh_device, params=params, dtype=LLAMA33_70B_GALAXY_ACCURACY.kv_cache_dtype
            )
        with _stage("bind kv cache"):
            model.set_kv_cache(kv_cache)
        with _stage("activate prefill"):
            model.activate("prefill")
        tokens = torch.randint(0, params.vocab_size, (1, _PREFILL_LENGTH), dtype=torch.long)
        rot_mats = output = None
        try:
            with _stage("prepare prefill rot mats"):
                rot_mats = model.prepare_prefill_rot_mats(0, _PREFILL_LENGTH)
            with _stage("embed prefill"):
                x_embed = model.embed_prefill(LazyWeight(source=tokens, device=mesh_device))
            with _stage("prefill forward"):
                output = model.prefill_forward(x_embed, rot_mats, sequence_length=_PREFILL_LENGTH, user_ids=(0,))
            with _stage("synchronize prefill"):
                model.synchronize("prefill")
            assert output is not None
            assert tuple(output.shape)[-1] == params.padded_vocab_size, tuple(output.shape)
        finally:
            deallocate(output)
            for tensor in rot_mats or ():
                deallocate(tensor)
    finally:
        with _stage("close model"):
            try:
                model.close()
            finally:
                _release_kv(kv_cache)
                del model, kv_cache
                gc.collect()


# =============================================================================
# L1: global-CB ownership across two constructions
# =============================================================================


@galaxy_params
@galaxy
@torch.no_grad()
def test_two_models_in_one_process(mesh_device, layer0_weights_and_params):
    """Probe Milestone A limitation L1 directly.

    ``Prefetcher2D.cleanup()`` cannot free the global circular buffer, so ~55 MB
    of L1 can stay resident after a truthful ``owned_resources == ()``. If this
    fails with an L1 out-of-memory on the *second* construction, the cause is
    teardown ordering, not insufficient memory, and one model per process is the
    documented workaround.
    """

    weights, params = layer0_weights_and_params
    ttnn.SetDefaultDevice(mesh_device)
    for attempt in range(2):
        model = _build(mesh_device, weights, params)
        try:
            assert model.prefetcher.sealed, f"construction {attempt} did not seal the prefetcher"
        finally:
            model.close()
            del model
            gc.collect()
