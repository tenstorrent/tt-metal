# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.37: the carried block's features on MOE_BLOCK_SRC=main (Marko's reworked block, tt/moe_block_main_adapter.py),
on the spec's mesh (the Galaxy 8x4 torus).

* test_se_pack_precise_rounding[on|off]: SE_PACK_PRECISE through the adapter's MoEDtypes (block_dtypes: GEMM_INPUTS
  bfp8 -> the head's bf8 x / h, SE_PACK_PRECISE -> shared_pack_precise) and dims (block_dims of this checkpoint's
  config): the fused head's packed h bit for bit - on: one rounding of the fp32 value at the group's shared exponent
  (the reference's single fp8 cast of h, DS41-137), off: the packer's two (the fork test's crafted head,
  ttnn/ttnn/bringup/moe_block_main_ttnn/tests/test_fused_head_bf8.py: check_pack_rounding).
* test_norm_into_moe_bit_identical: NORM_INTO_MOE on vs off on the all-device model with layer 0 only (OWN_ROWS=off:
  NORM_INTO_MOE needs the column split), chunk 0 of the s10240 golden: on, ffn_norm writes the block's x and the
  head's bf8 xq (rms_norm bfp8_twin) and the head reads them as written; off, the head's xq is a typecast of the
  norm's output. The layer's output must be the same bytes (the twin is typecast bit for bit, DS41-184).

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/test_moe_block_main_parity.py
"""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.reference.golden import Golden
from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import begin_chunk, spec

S = spec()
REPO = Path(__file__).resolve().parents[4]
_OUT: dict = {}


def _fork_test():
    p = REPO / "ttnn/ttnn/bringup/moe_block_main_ttnn/tests/test_fused_head_bf8.py"
    sp = importlib.util.spec_from_file_location("test_fused_head_bf8_ds41", p)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def _settings(monkeypatch, **kv):
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    for k, v in kv.items():
        monkeypatch.setenv(f"DS41_{k}", v)
    settings.thaw()  # (get() reads the environment again: a model's load froze the previous values)
    return settings


MESH = pytest.mark.parametrize("mesh_device", [tuple(S.mesh)], indirect=True, ids=["x".join(map(str, S.mesh))])
BOX = pytest.mark.parametrize("device_params", [harness.device_params(S)], indirect=True, ids=["box"])


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("pack", ["on", "off"])
@BOX
@MESH
def test_se_pack_precise_rounding(mesh_device, pack, monkeypatch):
    from models.demos.common.bringup.reference.golden import hf_path
    from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import Config
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as ad

    _settings(monkeypatch, MOE_BLOCK_SRC="main", SE_PACK_PRECISE=pack, GEMM_INPUTS="bfp8")
    ft = _fork_test()
    cfg = Config.from_dir(hf_path(S))
    dims, _ = ad.block_dims(cfg)
    assert (int(cfg.dim), int(cfg.n_experts), int(cfg.inter)) == (ft.H, ft.E, ft.IS), "the crafted head's dims"
    dtypes = ad.block_dtypes()
    assert dtypes.shared_activations == ttnn.bfloat8_b and dtypes.shared_pack_precise == (pack == "on")
    ad._apply_core_cap(mesh_device)
    n_robust, n_differ = ft.check_pack_rounding(mesh_device, pack == "on", dtypes=dtypes, dims=dims)
    print(f"SE_PACK_PRECISE={pack}: {n_robust} decided values as the rule, {n_differ} where the rules differ")


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("norm_into_moe", ["on", "off"])
@BOX
@MESH
def test_norm_into_moe_bit_identical(mesh_device, norm_into_moe, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as ad

    _settings(monkeypatch, MOE_BLOCK_SRC="main", NORM_INTO_MOE=norm_into_moe, OWN_ROWS="off")
    ad._BLOCKS.clear()  # (each value its own block and input buffers)
    ad._INPUTS.clear()
    g = Golden.for_rung(S, "s10240")
    chunk = g.chunk
    model = S.hooks().device_model(mesh_device, S, [0], lm_head=False)
    blk = model.blocks[0]
    assert isinstance(blk.moe_block, ad.TtMoEBlockMainLayer), type(blk.moe_block)
    assert blk.norm_into_moe == (norm_into_moe == "on"), (blk.norm_into_moe, norm_into_moe)
    used = types.SimpleNamespace(xq_ready=[])
    call = type(blk.moe_block).__call__

    def spy(self, x, xq_ready=False):
        used.xq_ready.append(bool(xq_ready))
        return call(self, x, xq_ready=xq_ready)

    monkeypatch.setattr(type(blk.moe_block), "__call__", spy)
    state = model.new_state(g.seq)
    tokens = g.tokens()[:chunk]
    begin_chunk(model, tokens, 0)
    h = model.embed(tokens)
    out = model.layer(0, h, 0, state)
    model.free(h)
    model.sync()
    y = model.to_host(out).float()
    model.free(out)
    assert used.xq_ready == [norm_into_moe == "on"], used.xq_ready
    assert torch.isfinite(y).all()
    _OUT[norm_into_moe] = y
    if len(_OUT) == 2:
        d = (_OUT["on"] - _OUT["off"]).abs().max().item()
        print(f"NORM_INTO_MOE on vs off (MOE_BLOCK_SRC=main): max |diff| {d}")
        assert torch.equal(_OUT["on"], _OUT["off"]), f"NORM_INTO_MOE on vs off differ: max |diff| {d}"
