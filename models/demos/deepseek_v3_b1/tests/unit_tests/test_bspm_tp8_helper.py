# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free unit tests for moe_routed_expert_bspm_tp8_torch_for_cache.

Verifies the slicing/padding/shuffling logic used to fan a routed expert
weight + BSPM assignment out to a 2D mesh.  No TT device is needed — all
tests run on CPU with numpy/torch only.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from models.demos.deepseek_v3_b1.weights.transforms.moe import moe_routed_expert_bspm_tp8_torch_for_cache

TILE_W = 32
NUM_BANKS = 8
PAD_CODE = 3  # _pad_assignment_cols uses constant_values=3 (zero/bfp0)
UNIFORM_CODE = 1  # np.ones(...) when assignment=None — bfp4 in tt-metal codes


def _pattern_assignment(tiles_h: int, tiles_w: int) -> np.ndarray:
    """Deterministic codes spread across {0, 1, 2, 3} so multiset checks are meaningful."""
    flat = (np.arange(tiles_h * tiles_w, dtype=np.int64) % 4).astype(np.int8)
    return flat.reshape(tiles_h, tiles_w)


def _padded_N(N: int, num_banks: int = NUM_BANKS) -> int:
    lcm = num_banks * TILE_W
    return ((N + lcm - 1) // lcm) * lcm


# --- Output shapes ------------------------------------------------------------


@pytest.mark.parametrize("mesh_shape", [(1, 8), (2, 4), (4, 2)])
def test_col_parallel_output_shapes(mesh_shape):
    K, N = 64, 2048  # per_device_N = 256 = NUM_BANKS * TILE_W → no helper-internal padding
    tp = mesh_shape[0] * mesh_shape[1]
    w = torch.randn(K, N)
    assignment = _pattern_assignment(K // TILE_W, N // TILE_W)

    stacked, stacked_assignment = moe_routed_expert_bspm_tp8_torch_for_cache(
        w, assignment, NUM_BANKS, mesh_shape, shard_dim=1
    )

    per_device_N = N // tp
    n_padded = _padded_N(per_device_N)
    assert stacked.shape == (mesh_shape[0], mesh_shape[1], K, n_padded)
    assert stacked_assignment.shape == (
        mesh_shape[0] * mesh_shape[1] * (K // TILE_W),
        n_padded // TILE_W,
    )
    assert stacked_assignment.dtype == np.int8


@pytest.mark.parametrize("mesh_shape", [(1, 8), (2, 4), (4, 2)])
def test_row_parallel_output_shapes(mesh_shape):
    K, N = 2048, 256  # N = NUM_BANKS * TILE_W → no helper-internal padding
    tp = mesh_shape[0] * mesh_shape[1]
    w = torch.randn(K, N)
    assignment = _pattern_assignment(K // TILE_W, N // TILE_W)

    stacked, stacked_assignment = moe_routed_expert_bspm_tp8_torch_for_cache(
        w, assignment, NUM_BANKS, mesh_shape, shard_dim=0
    )

    per_device_K = K // tp
    n_padded = _padded_N(N)
    assert stacked.shape == (mesh_shape[0], mesh_shape[1], per_device_K, n_padded)
    assert stacked_assignment.shape == (
        mesh_shape[0] * mesh_shape[1] * (per_device_K // TILE_W),
        n_padded // TILE_W,
    )


# --- Partition preserves precision-code multiset ------------------------------


def test_col_parallel_partition_preserves_multiset():
    """Per-rank shuffling is a permutation, so the union of per-rank cells must
    equal the original cells (multiset).  Picks dimensions where no internal
    padding kicks in, so we can compare directly to the input assignment."""
    K, N = 96, 2048
    mesh_shape = (2, 4)
    w = torch.randn(K, N)
    assignment = _pattern_assignment(K // TILE_W, N // TILE_W)

    _, stacked_assignment = moe_routed_expert_bspm_tp8_torch_for_cache(
        w, assignment, NUM_BANKS, mesh_shape, shard_dim=1
    )

    # No padding: per_device_N = 256 = NUM_BANKS * TILE_W.
    assert stacked_assignment.size == assignment.size, "padding leaked into assignment unexpectedly"
    np.testing.assert_array_equal(np.sort(stacked_assignment.flatten()), np.sort(assignment.flatten()))


def test_row_parallel_partition_preserves_multiset():
    K, N = 2048, 256
    mesh_shape = (2, 4)
    w = torch.randn(K, N)
    assignment = _pattern_assignment(K // TILE_W, N // TILE_W)

    _, stacked_assignment = moe_routed_expert_bspm_tp8_torch_for_cache(
        w, assignment, NUM_BANKS, mesh_shape, shard_dim=0
    )

    assert stacked_assignment.size == assignment.size
    np.testing.assert_array_equal(np.sort(stacked_assignment.flatten()), np.sort(assignment.flatten()))


# --- Bits-per-element preservation (Phase 4 will assert this at runtime) ------


def test_be_preservation_no_padding():
    """Sum of per-rank assignment cells equals sum of input cells when no padding."""
    K, N = 64, 2048
    mesh_shape = (2, 4)
    w = torch.randn(K, N)
    assignment = _pattern_assignment(K // TILE_W, N // TILE_W)

    _, stacked_assignment = moe_routed_expert_bspm_tp8_torch_for_cache(
        w, assignment, NUM_BANKS, mesh_shape, shard_dim=1
    )

    assert int(stacked_assignment.sum()) == int(assignment.sum())
    assert float(stacked_assignment.mean()) == pytest.approx(float(assignment.mean()))


# --- Uniform-BFP4 fallback ----------------------------------------------------


def test_uniform_fallback_when_assignment_is_none():
    K, N = 64, 2048
    mesh_shape = (2, 4)
    w = torch.randn(K, N)

    _, stacked_assignment = moe_routed_expert_bspm_tp8_torch_for_cache(w, None, NUM_BANKS, mesh_shape, shard_dim=1)

    # No padding: per_device_N = 256 = NUM_BANKS * TILE_W. Uniform = all UNIFORM_CODE.
    assert np.all(stacked_assignment == UNIFORM_CODE)


# --- Padding tests ------------------------------------------------------------


def test_col_parallel_pad_uses_zero_code():
    """When per_device_N < NUM_BANKS * TILE_W, the helper pads each rank's
    assignment to N_padded_per_device with the zero precision code (3)."""
    K, N = 64, 128
    mesh_shape = (1, 2)
    tp = mesh_shape[0] * mesh_shape[1]
    per_device_N = N // tp  # 64
    n_padded = _padded_N(per_device_N)  # 256
    assert n_padded > per_device_N, "test setup expects helper-internal padding"

    w = torch.randn(K, N)
    assignment = _pattern_assignment(K // TILE_W, N // TILE_W)

    _, stacked_assignment = moe_routed_expert_bspm_tp8_torch_for_cache(
        w, assignment, NUM_BANKS, mesh_shape, shard_dim=1
    )

    # Only (per_device_N // TILE_W) tile columns per rank carry real data;
    # the remaining (n_padded - per_device_N) // TILE_W are all PAD_CODE.
    expected_pad_cells = tp * (K // TILE_W) * ((n_padded - per_device_N) // TILE_W)
    pad_cell_count = int(np.sum(stacked_assignment == PAD_CODE))
    real_pad_in_input = int(np.sum(assignment == PAD_CODE))  # any 3s in the pattern
    assert (
        pad_cell_count >= expected_pad_cells - real_pad_in_input * tp
    ), f"expected at least {expected_pad_cells} pad cells, got {pad_cell_count}"


def test_row_parallel_pad_uses_zero_code():
    """For row-parallel, the helper pads N first, then row-slices per rank.
    Padded cols carry PAD_CODE in every rank."""
    K, N = 2048, 64
    mesh_shape = (2, 4)
    tp = mesh_shape[0] * mesh_shape[1]
    per_device_K = K // tp  # 256
    n_padded = _padded_N(N)  # 256
    assert n_padded > N, "test setup expects helper-internal padding"

    w = torch.randn(K, N)
    assignment = _pattern_assignment(K // TILE_W, N // TILE_W)

    _, stacked_assignment = moe_routed_expert_bspm_tp8_torch_for_cache(
        w, assignment, NUM_BANKS, mesh_shape, shard_dim=0
    )

    # Pad columns are at the right of each rank's assignment.  Counting cells with
    # PAD_CODE: every rank's row block contributes (per_device_K // TILE_W) *
    # ((n_padded - N) // TILE_W) padded cells, plus any organic 3s scattered around.
    expected_pad = tp * (per_device_K // TILE_W) * ((n_padded - N) // TILE_W)
    pad_cell_count = int(np.sum(stacked_assignment == PAD_CODE))
    assert pad_cell_count >= expected_pad, f"expected at least {expected_pad} pad cells, got {pad_cell_count}"
    # Tighter: per-rank contribution exactly matches the slice + pad behavior.
    n_padded_tiles = n_padded // TILE_W
    n_real_tiles = N // TILE_W
    assert n_padded_tiles - n_real_tiles == 6  # 256/32 - 64/32 = 8 - 2


# --- Input validation ---------------------------------------------------------


@pytest.mark.parametrize(
    "K, N",
    [
        (33, 2048),  # K not tile-aligned
        (64, 2049),  # N not tile-aligned
    ],
)
def test_rejects_non_tile_aligned_shapes(K, N):
    w = torch.randn(K, N)
    with pytest.raises(ValueError, match="tile-aligned"):
        moe_routed_expert_bspm_tp8_torch_for_cache(w, None, NUM_BANKS, mesh_shape=(2, 4), shard_dim=1)


def test_rejects_invalid_shard_dim():
    w = torch.randn(64, 2048)
    with pytest.raises(ValueError, match="shard_dim"):
        moe_routed_expert_bspm_tp8_torch_for_cache(w, None, NUM_BANKS, mesh_shape=(2, 4), shard_dim=2)


def test_rejects_per_device_N_not_tile_aligned():
    # mesh (2, 8) → tp=16. N=64 is tile-aligned and divisible by tp, but
    # per_device_N = 64 / 16 = 4, which is NOT a multiple of tile_w=32.
    # The Phase 3 safeguard should catch this before the slicing math goes wrong.
    w = torch.randn(64, 64)
    with pytest.raises(AssertionError, match="per_device_N=.* must be a multiple of tile_w"):
        moe_routed_expert_bspm_tp8_torch_for_cache(w, None, NUM_BANKS, mesh_shape=(2, 8), shard_dim=1)


def test_rejects_per_device_K_not_tile_aligned():
    # Same idea for row-parallel: K=64, tp=16 → per_device_K = 4, not tile-aligned.
    w = torch.randn(64, 256)
    with pytest.raises(AssertionError, match="per_device_K=.* must be a multiple of tile_w"):
        moe_routed_expert_bspm_tp8_torch_for_cache(w, None, NUM_BANKS, mesh_shape=(2, 8), shard_dim=0)


def test_rejects_assignment_K_tile_count_mismatch():
    w = torch.randn(64, 2048)
    bogus_assignment = np.zeros((1, N := 64), dtype=np.int8)  # K_tiles should be 2, not 1
    with pytest.raises(ValueError, match="do not match K tiles"):
        moe_routed_expert_bspm_tp8_torch_for_cache(w, bogus_assignment, NUM_BANKS, mesh_shape=(2, 4), shard_dim=1)


def test_rejects_assignment_with_too_few_N_tiles():
    w = torch.randn(64, 2048)
    too_short = np.zeros((2, 32), dtype=np.int8)  # N_tiles should be >= 64
    with pytest.raises(ValueError, match="do not cover N tiles"):
        moe_routed_expert_bspm_tp8_torch_for_cache(w, too_short, NUM_BANKS, mesh_shape=(2, 4), shard_dim=1)


# --- Phase 4: B/E drift check is silent on the happy path ---------------------


def test_be_drift_check_silent_on_correct_slicing(caplog):
    """The Phase 4 logger.warning should NOT fire on a correctly-sliced helper run."""
    K, N = 64, 2048
    mesh_shape = (2, 4)
    w = torch.randn(K, N)
    assignment = _pattern_assignment(K // TILE_W, N // TILE_W)

    with caplog.at_level("WARNING"):
        moe_routed_expert_bspm_tp8_torch_for_cache(w, assignment, NUM_BANKS, mesh_shape, shard_dim=1)
    assert "BSPM TP8 assignment drift" not in caplog.text


def test_be_drift_check_silent_with_padding(caplog):
    """Pad cells (code 3) added by the helper must be accounted for so the check
    is silent even when per_device_N < num_banks*tile_w."""
    K, N = 64, 128
    mesh_shape = (1, 2)
    w = torch.randn(K, N)
    assignment = _pattern_assignment(K // TILE_W, N // TILE_W)

    with caplog.at_level("WARNING"):
        moe_routed_expert_bspm_tp8_torch_for_cache(w, assignment, NUM_BANKS, mesh_shape, shard_dim=1)
    assert "BSPM TP8 assignment drift" not in caplog.text


# --- Phase 6: hard error when bspm_path is supplied but file missing ----------


def test_prepare_tp8_raises_when_bspm_path_missing(tmp_path):
    """``prepare_moe_routed_experts_bspm_tp8`` must raise FileNotFoundError if
    a non-existent bspm_path is supplied.  The check fires before any device or
    state_dict access, so we can pass ``device=None`` and an empty state dict."""
    from models.demos.deepseek_v3_b1.weights.prepare import prepare_moe_routed_experts_bspm_tp8

    missing = tmp_path / "definitely_missing.bspm"
    assert not missing.exists()

    with pytest.raises(FileNotFoundError, match="BSPM file required"):
        prepare_moe_routed_experts_bspm_tp8(
            device=None,
            state_dict={},
            layer_idx=3,
            num_routed_experts=0,
            num_banks=8,
            mesh_shape=(2, 4),
            bspm_path=missing,
            move_to_device=False,
        )


# --- Phase 1 follow-up: on-disk cache actually hits on second call ------------


def test_disk_cache_hits_on_second_call(tmp_path):
    """Round-trip test of the on-disk TensorCache machinery using the same
    fingerprint shape ``get_or_create_bspm_expert_tp8`` builds.  Bypasses
    CompressedTensor construction (replaces ``reconstruct`` with a passthrough)
    so the test is hardware-free, but the disk I/O (tiles.bin + assignment.npy +
    manifest + metadata) is exercised exactly like the production helper."""
    import hashlib

    from models.demos.deepseek_v3_b1.weights.cache import (
        BspmVariant,
        CacheContext,
        CompressedTensorBuildInputs,
        CompressedTensorTarget,
        SourceTensorSelection,
        TensorCache,
    )

    # Mimic get_or_create_bspm_expert_tp8's fingerprint inputs for one expert/proj.
    K, N = 64, 2048
    mesh_shape = (2, 4)
    tp = mesh_shape[0] * mesh_shape[1]
    K_per_device = K  # col-parallel
    N_padded_per_device = N // tp  # 256 = NUM_BANKS*TILE_W → no helper padding

    flat_w = np.random.RandomState(0).randn(tp * K_per_device, N_padded_per_device).astype(np.float32)
    flat_assignment = np.random.RandomState(0).randint(
        0, 4, size=(tp * K_per_device // TILE_W, N_padded_per_device // TILE_W), dtype=np.int8
    )
    assignment_hash = hashlib.sha256(flat_assignment.tobytes()).hexdigest()[:16]

    target = CompressedTensorTarget(
        name="routed_gate_proj",
        K=tp * K_per_device,  # flattened K for storage (matches helper convention)
        N_padded=N_padded_per_device,
        num_banks=NUM_BANKS,
        bspm_variant=BspmVariant.B,
        bspm_budget=3.5,
        assignment_hash=assignment_hash,
    )
    context = CacheContext(
        schema_version=0,
        hf_model_id="test_model",
        hf_revision="test_rev",
        mesh_shape=mesh_shape,
    )
    fingerprint = context.fingerprint(
        source=SourceTensorSelection(names=("test_key",)),
        target=target,
    )

    cache = TensorCache(tmp_path)

    preprocess_calls = 0
    raw_tensor_calls = 0

    def _preprocess(_tensors):
        nonlocal preprocess_calls
        preprocess_calls += 1
        return {target.name: CompressedTensorBuildInputs(w=flat_w, assignment=flat_assignment)}

    def _raw_tensors():
        nonlocal raw_tensor_calls
        raw_tensor_calls += 1
        return {"test_key": torch.zeros(1)}

    def _reconstruct(inputs, _dev):
        # Passthrough — skips CompressedTensor.from_bspm so the test stays hardware-free.
        return inputs

    # First call: cache miss → preprocess + raw_tensors invoked, disk artifacts written.
    result1 = cache.get_or_create(
        fingerprint,
        device=None,
        move_to_device=False,
        preprocess=_preprocess,
        raw_tensors=_raw_tensors,
        reconstruct=_reconstruct,
    )
    assert preprocess_calls == 1
    assert raw_tensor_calls == 1
    assert isinstance(result1, CompressedTensorBuildInputs)
    preprocess_calls_after_miss = preprocess_calls
    raw_tensor_calls_after_miss = raw_tensor_calls

    # Verify the expected compact-tile artifacts landed on disk.
    artifact_dirs = list((tmp_path / "objects").glob("*/*"))
    assert len(artifact_dirs) == 1
    artifact_dir = artifact_dirs[0]
    for fname in ("tiles.bin", "assignment.npy", "manifest.json", "metadata.json"):
        assert (artifact_dir / fname).is_file(), f"missing cache artifact {fname}"

    # Second call with the same fingerprint must hit: preprocess and raw_tensors
    # are skipped entirely; only reconstruct runs against the loaded inputs.
    result2 = cache.get_or_create(
        fingerprint,
        device=None,
        move_to_device=False,
        preprocess=_preprocess,
        raw_tensors=_raw_tensors,
        reconstruct=_reconstruct,
    )
    assert preprocess_calls == preprocess_calls_after_miss, f"preprocess re-invoked on cache hit: {preprocess_calls}"
    assert raw_tensor_calls == raw_tensor_calls_after_miss, f"raw_tensors re-invoked on cache hit: {raw_tensor_calls}"

    # Assignment must round-trip exactly through the on-disk artifact.
    np.testing.assert_array_equal(result2.assignment, flat_assignment)


def test_disk_cache_misses_when_fingerprint_changes(tmp_path):
    """Two calls with different ``assignment_hash`` must miss independently.
    This guards against regressions where the fingerprint stops including the
    assignment (or some other relevant field) and silently aliases distinct
    BSPMs to the same cache entry."""
    import hashlib

    from models.demos.deepseek_v3_b1.weights.cache import (
        BspmVariant,
        CacheContext,
        CompressedTensorBuildInputs,
        CompressedTensorTarget,
        SourceTensorSelection,
        TensorCache,
    )

    K, N = 64, 2048
    mesh_shape = (2, 4)
    tp = mesh_shape[0] * mesh_shape[1]
    K_per_device = K
    N_padded_per_device = N // tp

    cache = TensorCache(tmp_path)
    context = CacheContext(
        schema_version=0,
        hf_model_id="test_model",
        hf_revision="test_rev",
        mesh_shape=mesh_shape,
    )

    def _make_fp(seed: int):
        flat_assignment = np.random.RandomState(seed).randint(
            0, 4, size=(tp * K_per_device // TILE_W, N_padded_per_device // TILE_W), dtype=np.int8
        )
        target = CompressedTensorTarget(
            name="routed_gate_proj",
            K=tp * K_per_device,
            N_padded=N_padded_per_device,
            num_banks=NUM_BANKS,
            bspm_variant=BspmVariant.B,
            bspm_budget=3.5,
            assignment_hash=hashlib.sha256(flat_assignment.tobytes()).hexdigest()[:16],
        )
        fp = context.fingerprint(
            source=SourceTensorSelection(names=("test_key",)),
            target=target,
        )
        return fp, flat_assignment

    fp_a, asn_a = _make_fp(0)
    fp_b, asn_b = _make_fp(1)
    assert fp_a != fp_b, "different assignments must produce different fingerprints"

    flat_w = np.zeros((tp * K_per_device, N_padded_per_device), dtype=np.float32)
    miss_count = 0

    def _make_callbacks(asn):
        def _preprocess(_tensors):
            nonlocal miss_count
            miss_count += 1
            return {"routed_gate_proj": CompressedTensorBuildInputs(w=flat_w, assignment=asn)}

        return _preprocess

    cache.get_or_create(
        fp_a,
        device=None,
        move_to_device=False,
        preprocess=_make_callbacks(asn_a),
        raw_tensors=lambda: {"test_key": torch.zeros(1)},
        reconstruct=lambda inputs, _dev: inputs,
    )
    cache.get_or_create(
        fp_b,
        device=None,
        move_to_device=False,
        preprocess=_make_callbacks(asn_b),
        raw_tensors=lambda: {"test_key": torch.zeros(1)},
        reconstruct=lambda inputs, _dev: inputs,
    )
    assert miss_count == 2, f"distinct fingerprints aliased to one cache entry: miss_count={miss_count}"
    artifact_dirs = list((tmp_path / "objects").glob("*/*"))
    assert len(artifact_dirs) == 2, f"expected two cache entries, got {len(artifact_dirs)}"
