# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in BFP8 vision weights: the dtype each weight really has on device.

A PCC pass cannot show that BFP8 was loaded (BF16 passes the same bars), so this test reads the
device dtype of every vision weight after ``PplxVisionTower.from_snapshot``:

- ``test_device_weight_dtypes``: default (no kwarg, no env), ``policy="bfp8"``,
  ``$PPLX_DECIDER_VISION_PRECISION=bfp8``, a kwarg that overrides the env, and the two ``except``
  variants. The 7 projection roles must be BFLOAT8_B or BFLOAT16 as listed in ``EXPECTED_BFP8``
  (written out here, not read from ``VISION_PRECISIONS``, so a wrong policy table fails); every bias
  and LayerNorm weight must stay BFLOAT16.
- ``test_unknown_precision_raises``: an unknown name is rejected by the tower and by
  ``PplxDeciderModel.from_snapshot``, also with ``vision=False``, before any weight loads.
- ``test_precision_without_vision_warns``: a valid ``vision_precision`` with ``vision=False`` logs
  a warning that it has no effect.

    pytest models/demos/pplx_decider_v1_27b/tests/vision/test_vision_precision.py -q
"""

from __future__ import annotations

import pytest
from loguru import logger

import ttnn
from models.demos.pplx_decider_v1_27b.tests.vision.vision_test_utils import DEVICE_PARAMS, reader
from models.demos.pplx_decider_v1_27b.tt.optimizations import VISION_PRECISION_ENV

ALL_ROLES = ("vision_patch", "vision_qkv", "vision_proj", "vision_fc1", "vision_fc2", "merger_fc1", "merger_fc2")
# case id -> (policy kwarg, env value, roles whose weights must be BFLOAT8_B; every other role BFLOAT16)
EXPECTED_BFP8 = {
    "default": (None, None, ()),
    "kwarg_bfp8": ("bfp8", None, ALL_ROLES),
    "env_bfp8": (None, "bfp8", ALL_ROLES),
    "kwarg_bf16_over_env_bfp8": ("bf16", "bfp8", ()),
    "except_qkv": ("bfp8_except_qkv", None, tuple(r for r in ALL_ROLES if r != "vision_qkv")),
    "except_merger": ("bfp8_except_merger", None, tuple(r for r in ALL_ROLES if not r.startswith("merger_"))),
}


def _device_dtypes(tower) -> tuple[dict[str, set], dict[str, set], list]:
    """(role -> dtypes of its weights, bias/norm name -> dtypes, every tensor read) over every block."""
    weights, others, tensors = {}, {}, []

    def add(table, key, t):
        table.setdefault(key, set()).add(t.dtype)
        tensors.append(t)

    add(weights, "vision_patch", tower.patch_embed.weight)
    add(others, "patch.bias", tower.patch_embed.bias)
    for b in tower.blocks:
        a, m = b.attention, b.mlp
        for role, w, bias in (
            ("vision_qkv", a.qkv_w, a.qkv_b),
            ("vision_proj", a.proj_w, a.proj_b),
            ("vision_fc1", m.fc1_w, m.fc1_b),
            ("vision_fc2", m.fc2_w, m.fc2_b),
        ):
            add(weights, role, w)
            add(others, f"{role}.bias", bias)
        for name, norm in (("norm1", b.norm1), ("norm2", b.norm2)):
            add(others, f"block.{name}.weight", norm.weight)
            add(others, f"block.{name}.bias", norm.bias)
    g = tower.merger
    add(weights, "merger_fc1", g.fc1_w)
    add(weights, "merger_fc2", g.fc2_w)
    add(others, "merger_fc1.bias", g.fc1_b)
    add(others, "merger_fc2.bias", g.fc2_b)
    add(others, "merger.norm.weight", g.norm.weight)
    add(others, "merger.norm.bias", g.norm.bias)
    return weights, others, tensors


@pytest.mark.use_module_device(DEVICE_PARAMS)
@pytest.mark.timeout(900)
@pytest.mark.parametrize("case", list(EXPECTED_BFP8))
def test_device_weight_dtypes(_device_module_impl, monkeypatch, case):
    from models.demos.pplx_decider_v1_27b.tt.vision.tower import PplxVisionTower

    kwarg, env, bfp8_roles = EXPECTED_BFP8[case]
    monkeypatch.delenv(VISION_PRECISION_ENV, raising=False)
    if env is not None:
        monkeypatch.setenv(VISION_PRECISION_ENV, env)
    kw = {} if kwarg is None else {"policy": kwarg}
    tower = PplxVisionTower.from_snapshot(_device_module_impl, reader(), **kw)
    weights, others, tensors = _device_dtypes(tower)
    n_blocks = len(tower.blocks)
    policy = tower.config.optimizations.policy.name
    for t in tensors:  # free the tower weights before the next case builds its own
        ttnn.deallocate(t)
    del tower

    assert n_blocks == 27
    assert set(weights) == set(ALL_ROLES)
    got = {role: sorted(str(d) for d in weights[role]) for role in ALL_ROLES}
    want = {role: [str(ttnn.bfloat8_b if role in bfp8_roles else ttnn.bfloat16)] for role in ALL_ROLES}
    logger.info(f"{case}: policy {policy}; weight dtypes {got}")
    assert got == want, f"{case} (policy {policy}): device weight dtypes {got} != expected {want}"
    not_bf16 = {k: sorted(str(d) for d in v) for k, v in others.items() if v != {ttnn.bfloat16}}
    assert not not_bf16, f"{case}: biases / LayerNorm must stay BF16, got {not_bf16}"


@pytest.mark.parametrize("vision", [True, False])
def test_unknown_precision_raises(monkeypatch, expect_error, vision):
    """Rejected before any weight loads (no device or snapshot is touched)."""
    from models.demos.pplx_decider_v1_27b.tt.model import PplxDeciderModel
    from models.demos.pplx_decider_v1_27b.tt.vision.tower import PplxVisionTower

    monkeypatch.delenv(VISION_PRECISION_ENV, raising=False)
    with expect_error(ValueError, "Unknown vision precision"):
        PplxDeciderModel.from_snapshot(None, _NoLoadReader(), vision=vision, vision_precision="bfp4")
    with expect_error(ValueError, "Unknown vision precision"):
        PplxVisionTower.from_snapshot(None, reader(), policy="BFP8")  # reads config.json only


def test_precision_without_vision_warns(monkeypatch, expect_error):
    from models.demos.pplx_decider_v1_27b.tt.model import PplxDeciderModel

    monkeypatch.delenv(VISION_PRECISION_ENV, raising=False)
    messages = []
    sink = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        with expect_error(_LoadStarted, "text_config"):
            PplxDeciderModel.from_snapshot(None, _NoLoadReader(), vision=False, vision_precision="bfp8")
    finally:
        logger.remove(sink)
    assert any("vision_precision='bfp8' has no effect with vision=False" in m for m in messages), messages


class _LoadStarted(Exception):
    pass


class _NoLoadReader:
    """A snapshot reader that stops the load at its first use."""

    def __getattr__(self, name):
        raise _LoadStarted(name)
