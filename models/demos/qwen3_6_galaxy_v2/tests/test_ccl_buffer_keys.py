# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CPU-only test for V2-6 — qwen3.6 dual-dtype persistent-buffer keys.

Goal: verify that `TT_CCL` registers the bfloat8_b / bfloat16 variants of the
QKV, WO_AG, FF1, FF3 prefill all-gather buffers and the decode `BINARY_MUL_BF16`
residual buffer when `is_qwen36=True`. Also verifies the `get_qkv_buffer_key`
selection rule (ISL ≥ 4096 → "QKV", ISL < 4096 → "QKV_BF16").

We do not touch a device: every ttnn allocator and semaphore primitive used in
`TT_CCL.__init__` is monkey-patched to return a sentinel object (a `MagicMock`).
Only the registered key set is asserted against.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Lightweight model_args stub matching qwen3.6 fields read by TT_CCL.__init__.
# ---------------------------------------------------------------------------
class _FakeModelArgs:
    def __init__(self):
        self.sub_core_grids = MagicMock(name="sub_core_grids")
        # Dynamic full-band grid: BH P150 galaxy reports compute grid (12,10),
        # so sub_core_grids spans cols 1..11 × 10 rows = 110 cores (was the
        # inherited Wormhole 60). Matches the dynamic derivation in
        # qwen36_model_config.py.
        self.sub_core_grids.num_cores.return_value = 110
        # Required by get_decode_reduce_scatter_buffers via shard_spec.num_cores().
        self.max_top_k = 32
        self.max_batch_size = 32
        self.cluster_shape = [8, 4]
        self.is_qwen36 = True
        # Minimal model_config: only what TT_CCL.__init__ + buffer methods read.
        gather_users_memcfg = MagicMock(name="GATHER_USERS_MEMCFG")
        gather_users_memcfg.return_value = MagicMock(name="GATHER_USERS_MEMCFG_returned")
        rs_interim = MagicMock(name="REDUCE_SCATTER_INTERIM_MEMCFG")
        rs_interim.shard_spec.num_cores.return_value = 32
        rs_create_heads = MagicMock(name="RS_CREATE_HEADS_INTERIM_MEMCFG")
        ff2_in_ring = MagicMock(name="FF2_IN_RING_MEMCFG")
        self.model_config = {
            "CCL_TOPOLOGY": MagicMock(name="CCL_TOPOLOGY"),
            "GATHER_USERS_MEMCFG": gather_users_memcfg,
            "REDUCE_SCATTER_INTERIM_MEMCFG": rs_interim,
            "RS_CREATE_HEADS_INTERIM_MEMCFG": rs_create_heads,
            "FF2_IN_RING_MEMCFG": ff2_in_ring,
            "IS_QWEN36": True,
            "GALAXY_NUM_LINKS": 1,
        }

    def weight_cache_path(self, _dtype):
        # Path-like with __truediv__ so as_tensor cache_file_name keeps working.
        import pathlib

        return pathlib.Path("/tmp/_ccl_test_cache")


def _make_fake_mesh_device():
    md = MagicMock(name="mesh_device")
    md.shape = [8, 4]
    grid = MagicMock(name="grid_size")
    # BH P150 galaxy compute grid (cols 0-11 × rows 0-9).
    grid.x = 12
    grid.y = 10
    md.compute_with_storage_grid_size.return_value = grid
    md.get_num_devices.return_value = 32
    return md


# ---------------------------------------------------------------------------
# Patch the ttnn primitives used by TT_CCL.__init__ and the buffer methods so
# we never touch silicon.  Each returns a sentinel `MagicMock` standing in for
# the allocated tensor / semaphore.
# ---------------------------------------------------------------------------
@pytest.fixture
def patched_ttnn():
    import ttnn

    # CoreCoord / CoreRange / CoreRangeSet are constructed at import time inside
    # TT_CCL.__init__, so they must accept *args; the real ttnn ones do, but
    # if anything breaks we just stub them with MagicMock factories.
    patches = [
        patch.object(ttnn, "from_torch", MagicMock(return_value=MagicMock(name="tt_buffer"))),
        patch.object(ttnn, "as_tensor", MagicMock(return_value=MagicMock(name="tt_as_tensor"))),
        patch.object(
            ttnn,
            "create_global_semaphore",
            MagicMock(return_value=MagicMock(name="global_semaphore")),
        ),
        patch.object(
            ttnn,
            "create_sharded_memory_config",
            MagicMock(return_value=MagicMock(name="sharded_memcfg")),
        ),
        patch.object(ttnn, "ShardSpec", MagicMock(return_value=MagicMock(name="shard_spec"))),
        patch.object(ttnn, "MemoryConfig", MagicMock(return_value=MagicMock(name="memcfg"))),
        patch.object(ttnn, "ReplicateTensorToMesh", MagicMock(return_value=MagicMock(name="replicate"))),
        patch.object(ttnn, "ShardTensor2dMesh", MagicMock(return_value=MagicMock(name="shard_mesh"))),
    ]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


def _construct_ccl(mode="prefill"):
    """Build a TT_CCL helper under the patched ttnn environment."""
    from models.demos.qwen3_6_galaxy_v2.tt.llama_ccl import TT_CCL

    mesh_device = _make_fake_mesh_device()
    model_args = _FakeModelArgs()
    worker_sub_device_id = MagicMock(name="worker_sub_device_id")
    return TT_CCL(
        mesh_device=mesh_device,
        model_args=model_args,
        worker_sub_device_id=worker_sub_device_id,
        mode=mode,
        # is_qwen36 flag is auto-picked from model_args.is_qwen36; we still
        # pass it explicitly to exercise the constructor kw.
        is_qwen36=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_get_qkv_buffer_key_isl_threshold(patched_ttnn):
    """ISL ≥ 4096 → bfloat8_b ("QKV"); ISL < 4096 → bfloat16 ("QKV_BF16")."""
    tt_ccl = _construct_ccl(mode="prefill")
    assert tt_ccl.get_qkv_buffer_key(seq_len=128) == "QKV_BF16"
    assert tt_ccl.get_qkv_buffer_key(seq_len=2048) == "QKV_BF16"
    # 4096 is the threshold — falls into bfloat8_b bucket.
    assert tt_ccl.get_qkv_buffer_key(seq_len=4096) == "QKV"
    assert tt_ccl.get_qkv_buffer_key(seq_len=8192) == "QKV"


def test_get_wo_ag_buffer_key_isl_threshold(patched_ttnn):
    tt_ccl = _construct_ccl(mode="prefill")
    assert tt_ccl.get_wo_ag_buffer_key(seq_len=128) == "WO_AG_BF16"
    assert tt_ccl.get_wo_ag_buffer_key(seq_len=2048) == "WO_AG_BF16"
    assert tt_ccl.get_wo_ag_buffer_key(seq_len=4096) == "WO_AG"


def test_get_ff_buffer_key_isl_threshold(patched_ttnn):
    tt_ccl = _construct_ccl(mode="prefill")
    assert tt_ccl.get_ff_buffer_key("FF1", seq_len=128) == "FF1_BF16"
    assert tt_ccl.get_ff_buffer_key("FF3", seq_len=2048) == "FF3_BF16"
    assert tt_ccl.get_ff_buffer_key("FF1", seq_len=4096) == "FF1"
    assert tt_ccl.get_ff_buffer_key("FF3", seq_len=8192) == "FF3"


def test_prefill_all_gather_buffer_keys_present(patched_ttnn):
    """The qwen3.6 dual-dtype prefill all-gather buffer keys must be registered
    for every supported seqlen bucket."""
    tt_ccl = _construct_ccl(mode="prefill")
    expected_keys = {
        "QKV",
        "QKV_BF16",
        "WO_AG",
        "WO_AG_BF16",
        "FF1",
        "FF1_BF16",
        "FF3",
        "FF3_BF16",
        "FF2",
        "LAYERNORM",
        "SDPA",
        "SDPA_REVERSE",
        "ATTN_REPLICATE",
    }
    # `support_seqlens` is set in __init__ for prefill mode.
    for seqlen in tt_ccl.support_seqlens:
        assert seqlen in tt_ccl.all_gather_buffers, f"seqlen {seqlen} missing from all_gather_buffers"
        bucket = tt_ccl.all_gather_buffers[seqlen]
        missing = expected_keys - set(bucket.keys())
        assert not missing, f"seqlen {seqlen} missing keys: {missing}"


def test_decode_binary_mul_bf16_registered(patched_ttnn):
    """qwen3.6 decode `BINARY_MUL_BF16` (residual-stream bfloat16 buffer)
    must exist alongside the canonical bfloat8_b `BINARY_MUL`."""
    tt_ccl = _construct_ccl(mode="decode")
    assert "BINARY_MUL" in tt_ccl.all_gather_buffers
    assert "BINARY_MUL_BF16" in tt_ccl.all_gather_buffers


def test_non_qwen36_does_not_register_bf16_variants(patched_ttnn):
    """Sanity: turning the flag off keeps the legacy key set."""
    from models.demos.qwen3_6_galaxy_v2.tt.llama_ccl import TT_CCL

    model_args = _FakeModelArgs()
    model_args.is_qwen36 = False
    tt_ccl = TT_CCL(
        mesh_device=_make_fake_mesh_device(),
        model_args=model_args,
        worker_sub_device_id=MagicMock(),
        mode="prefill",
        is_qwen36=False,
    )
    for seqlen in tt_ccl.support_seqlens:
        bucket = tt_ccl.all_gather_buffers[seqlen]
        assert "QKV_BF16" not in bucket
        assert "WO_AG_BF16" not in bucket
        assert "FF1_BF16" not in bucket
        assert "FF3_BF16" not in bucket


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
