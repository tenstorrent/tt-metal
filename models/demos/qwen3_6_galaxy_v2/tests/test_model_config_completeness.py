# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Config-keys completeness test for ``TtQwen36ModelArgs``.

Scans every ``self.model_config[<KEY>]`` / ``self.args.model_config[<KEY>]``
access across ``models/demos/qwen3_6_galaxy_v2/tt/*.py`` and asserts that
the constructed ``args.model_config`` dict actually contains every key.

This is a *pure-CPU* test — the mesh device is mocked, and all ``ttnn``
allocators that touch the real fabric (``MemoryConfig``, sharded mem cfgs,
program configs) are patched to sentinel callables that return ``MagicMock``
instances.  The test's only job is to verify the **keys** are populated;
the values get validated by the first device run.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


TT_DIR = Path(__file__).resolve().parents[1] / "tt"

# Match ``self.model_config["KEY"]`` and ``self.args.model_config["KEY"]``
# (also ``configuration.model_config["KEY"]`` for completeness).
_KEY_RE = re.compile(r"(?:self|configuration|args|self\.args)\.model_config\[\s*[\"\']([A-Z_0-9]+)[\"\']\s*\]")


def _scan_model_config_keys() -> set[str]:
    """Walk every .py file under tt/ and collect every model_config key
    appearing on an indexed access.  Skips the config files themselves
    (qwen36_model_config.py / qwen_model_config.py / model_config.py)
    because those are where keys get *defined*, not consumed."""
    skip = {"qwen36_model_config.py", "qwen_model_config.py", "model_config.py"}
    keys: set[str] = set()
    for path in sorted(TT_DIR.glob("*.py")):
        if path.name in skip:
            continue
        text = path.read_text()
        for m in _KEY_RE.finditer(text):
            keys.add(m.group(1))
    return keys


class _FakeMeshDevice:
    """Stand-in for a real ``MeshDevice``.  Exposes the handful of methods
    the config constructor calls: shape attr, ``get_num_devices()``,
    ``compute_with_storage_grid_size()``, ``dram_grid_size()``."""

    def __init__(self, shape=(8, 4)):
        self.shape = list(shape)

    def get_num_devices(self):
        return self.shape[0] * self.shape[1]

    def compute_with_storage_grid_size(self):
        return SimpleNamespace(x=7, y=10)

    def dram_grid_size(self):
        return SimpleNamespace(x=8, y=1)


def _ttnn_patches():
    """Patch every ttnn entry point the config constructor would have called
    against a real device.  Each is replaced by a callable returning a
    sentinel ``MagicMock`` — keys/values are tracked, but device-state is
    never touched."""
    import ttnn

    return [
        # Memory configs
        patch.object(ttnn, "MemoryConfig", lambda *a, **kw: MagicMock(name="MemoryConfig")),
        patch.object(ttnn, "ShardSpec", lambda *a, **kw: MagicMock(name="ShardSpec")),
        patch.object(ttnn, "create_sharded_memory_config", lambda *a, **kw: MagicMock(name="sharded_mem_cfg")),
        # Program / compute kernel configs
        patch.object(
            ttnn,
            "WormholeComputeKernelConfig",
            lambda *a, **kw: MagicMock(name="WormholeComputeKernelConfig"),
        ),
        patch.object(
            ttnn,
            "LayerNormShardedMultiCoreProgramConfig",
            lambda *a, **kw: MagicMock(name="LayerNormShardedMultiCoreProgramConfig"),
        ),
        patch.object(
            ttnn,
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            lambda *a, **kw: MagicMock(name="MatmulMultiCoreReuseMultiCast1DProgramConfig"),
        ),
        patch.object(
            ttnn,
            "MatmulMultiCoreReuseMultiCastProgramConfig",
            lambda *a, **kw: MagicMock(name="MatmulMultiCoreReuseMultiCastProgramConfig"),
        ),
        patch.object(
            ttnn,
            "MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig",
            lambda *a, **kw: MagicMock(name="MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig"),
        ),
        patch.object(ttnn, "MinimalMatmulConfig", lambda *a, **kw: MagicMock(name="MinimalMatmulConfig")),
        patch.object(ttnn, "SDPAProgramConfig", lambda *a, **kw: MagicMock(name="SDPAProgramConfig")),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_qwen36_model_config_has_every_key_consumers_read():
    """Every key any v2 tt/*.py file looks up by indexed access must be
    present in the constructed ``args.model_config`` dict."""
    keys_consumed = _scan_model_config_keys()
    assert keys_consumed, "regex did not match any model_config keys — broken scan"

    patches = _ttnn_patches()
    for p in patches:
        p.start()
    try:
        from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

        mesh = _FakeMeshDevice(shape=(8, 4))
        # Skip HF param load — no internet, dummy_weights=True is the only
        # branch that bypasses the snapshot read.
        args = TtQwen36ModelArgs(mesh_device=mesh, dummy_weights=True)
    finally:
        for p in patches:
            p.stop()

    missing = sorted(k for k in keys_consumed if k not in args.model_config)
    assert not missing, (
        f"{len(missing)} model_config keys are read by v2/tt/*.py but never populated by "
        f"TtQwen36ModelArgs.__init__: {missing}.\n"
        f"This would KeyError on the first real-device construction."
    )


@pytest.mark.cpu_only
def test_qwen36_model_config_baseline_keys_populated():
    """A small fixed sanity-check covering the keys that the bare-minimum
    construction has always set — guards against accidental regression of
    the topology / link / qwen36 flags."""
    patches = _ttnn_patches()
    for p in patches:
        p.start()
    try:
        from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

        mesh = _FakeMeshDevice(shape=(8, 4))
        args = TtQwen36ModelArgs(mesh_device=mesh, dummy_weights=True)
    finally:
        for p in patches:
            p.stop()

    for key in (
        "CCL_TOPOLOGY",
        "GALAXY_NUM_LINKS",
        "USE_PREFETCHER",
        "IS_QWEN36",
    ):
        assert key in args.model_config, f"baseline key {key!r} missing from model_config"
    assert args.model_config["IS_QWEN36"] is True
    assert args.model_config["USE_PREFETCHER"] is False


@pytest.mark.cpu_only
def test_qwen36_model_config_program_config_keys_populated():
    """Spot-check the categories from the V2-config2 task list — every
    big-ticket program-config family must be present after construction."""
    patches = _ttnn_patches()
    for p in patches:
        p.start()
    try:
        from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs

        mesh = _FakeMeshDevice(shape=(8, 4))
        args = TtQwen36ModelArgs(mesh_device=mesh, dummy_weights=True)
    finally:
        for p in patches:
            p.stop()

    expected = {
        # Norms
        "SHARDED_NORM_ATTN_PRGM_CFG",
        "SHARDED_NORM_MLP_PRGM_CFG",
        "SHARDED_NORM_LM_HEAD_PRGM_CFG",
        # Decode memcfgs
        "SHARDED_ATTN_INPUT_MEMCFG",
        "SHARDED_MLP_INPUT_MEMCFG",
        "SHARDED_FF12_RING_MEMCFG",
        "SHARDED_ATTN_INPUT_RING_MEMCFG",
        "DECODE_RESIDUAL_MEMCFG",
        "CREATE_HEAD_OUTPUT_MEMCFG",
        # Decode matmul progcfgs
        "XQKV_DECODE_RING_PROGCFG",
        "WO_DECODE_RING_PROGCFG",
        "FF1_3_TG_RING_PROGCFG",
        "FF2_TG_RING_PROGCFG",
        # LM head
        "LM_HEAD_INPUT_MEMCFG",
        "LM_HEAD_OUT_RING_RESHARD_MEMCFG",
        "SHARDED_LM_HEAD_INPUT_32_RING_MEMCFG",
        # SDPA
        "SDPA_DECODE_PROGCFG",
        "SDPA_DECODE_COMPUTE_PROGCFG",
        "PAGED_SDPA_DECODE_PROGCFG",
        "SDPA_PROGCFG",
        "SDPA_PROGCFG_FLEXIBLE_CHUNK",
        # CCL
        "REDUCE_SCATTER_OUT_MEMCFG",
        "REDUCE_SCATTER_INTERIM_MEMCFG",
        "RS_CREATE_HEADS_INTERIM_MEMCFG",
        "GATHER_USERS_MEMCFG",
        # Compute kernels
        "COMPUTE_KERNEL_CONFIG_HIFI2",
    }
    missing = sorted(k for k in expected if k not in args.model_config)
    assert not missing, f"expected keys missing: {missing}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
