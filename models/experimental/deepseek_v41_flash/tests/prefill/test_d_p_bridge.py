# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""metal_2's whole-mesh prefill (``tt/prefill/d_p_bridge.py``) against this model's own KV-only prefill, in two runs
(a parent mesh and its stage submeshes allocate independently, so the two models never share a process; see
``PREFILL_D_P_INTEGRATION.md``):

1. ``test_d_p_export``: metal_2's runtime on the 8 x 4 mesh (its spec's device params) prefills a real-text prompt and
   saves layers 0-20 in the bridge's host format, with the prompt's ids, to ``D_P_STATES``.
2. ``test_d_p_vs_kv_only``: this model's traced KV-only prefill (eight ``1 x 4`` stages) runs the same ids; every
   layer's ``kv_tail`` (layer 20's excepted: metal_2 never writes it) and every KV source's ``compressed_kv`` /
   ``idx_keys`` are compared by PCC, and the saved states are converted with ``to_prefill_states`` (shapes checked on
   the decode stages).

Run (ttnn venv, repo root)::

    D_P_STATES=/tmp/d_p_states.pt D_P_PREFILL_LEN=16384 \\
    BRINGUP_SPEC=models/demos/deepseek_v41_flash_d_p/bringup/ledgers/galaxy_8x4-full/spec.yaml BRINGUP_HF=... \\
      pytest -s models/experimental/deepseek_v41_flash/tests/prefill/test_d_p_bridge.py::test_d_p_export
    D_P_STATES=/tmp/d_p_states.pt DEEPSEEK_V4_CACHE_DIR=... \\
      pytest -s models/experimental/deepseek_v41_flash/tests/prefill/test_d_p_bridge.py::test_d_p_vs_kv_only

``D_P_PREFILL_LEN`` is a multiple of 128 (decode's commit unit); 16384 by default, where this model's prefill is
validated. ``D_P_MIN_PCC`` (default 0.99) is the pass bar. ``D_P_MODEL_DIR`` overrides this model's checkpoint
directory (``DEFAULT_MODEL_DIR``).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

STATES = Path(os.environ.get("D_P_STATES", "/tmp/d_p_states.pt"))
PREFILL_LEN = int(os.environ.get("D_P_PREFILL_LEN", 16384))
MIN_PCC = float(os.environ.get("D_P_MIN_PCC", 0.99))
_TP_SIZE = 4


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    den = a.norm() * b.norm()
    return float((a @ b) / den) if den > 0 else float(torch.equal(a, b))


def _d_p_mesh(fn):
    """metal_2's mesh fixture (its spec's mesh and device params) when its spec is set, else a skip."""
    if not os.environ.get("BRINGUP_SPEC"):
        return pytest.mark.skip(reason="test_d_p_export runs metal_2's runtime: set BRINGUP_SPEC (and BRINGUP_HF)")(fn)
    from models.demos.common.bringup.testing.harness import mesh_parametrize

    return mesh_parametrize(timeout_s=4 * 3600)(fn)


@_d_p_mesh
def test_d_p_export(mesh_device):
    from models.demos.common.bringup.reference.golden import tokenizer
    from models.demos.common.bringup.testing.harness import spec
    from models.demos.deepseek_v41_flash_d_p.tests.test_full_prefill import build_tokens
    from models.experimental.deepseek_v41_flash.tt.prefill.d_p_bridge import DPPrefill, save_host_states

    assert PREFILL_LEN > 0 and PREFILL_LEN % 128 == 0, f"D_P_PREFILL_LEN={PREFILL_LEN} must be a multiple of 128"
    ids, info = build_tokens(tokenizer(spec()), PREFILL_LEN)
    ids = [int(t) for t in ids[:PREFILL_LEN]]
    logger.info(f"prompt: {len(ids)} tokens ({info})")

    d_p = DPPrefill(mesh_device, max_seq=PREFILL_LEN)
    d_p.prefill(ids[: d_p.chunk])  # warm: the later chunks' positions compile on first use
    t0 = time.perf_counter()
    n = d_p.prefill(ids, on_chunk=lambda s, e, sec: logger.info(f"chunk [{s}, {e}) enqueued in {sec * 1e3:.0f} ms"))
    wall = time.perf_counter() - t0
    logger.info(f"metal_2 prefill: {n} tokens in {wall:.2f} s = {n / wall:.0f} tok/s")

    host = d_p.host_states(n)
    for i, layer in host["layers"].items():
        for name, t in layer.items():
            assert torch.isfinite(t.float()).all(), (i, name)
    path = save_host_states(host, STATES, ids=torch.tensor(ids), mesh=tuple(mesh_device.shape))
    logger.info(f"saved layers {sorted(host['layers'])} of {n} tokens to {path}")


@pytest.mark.timeout(6 * 3600)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True, ids=["fabric_2d"]
)
@pytest.mark.parametrize("mesh_device", [(8, _TP_SIZE)], indirect=True, ids=["galaxy_8x4"])
def test_d_p_vs_kv_only(mesh_device):
    from transformers import AutoTokenizer

    from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import _build_rope
    from models.experimental.deepseek_v41_flash.tt.config import CACHE_DIR, DEFAULT_MODEL_DIR, load_config
    from models.experimental.deepseek_v41_flash.tt.engram_lookup import EngramHostLookup, load_engram_table
    from models.experimental.deepseek_v41_flash.tt.prefill.d_p_bridge import (
        free_prefill_states,
        load_host_states,
        to_prefill_states,
    )
    from models.experimental.deepseek_v41_flash.tt.pipeline_parallel_prefill.model import DeepSeekV41PrefillModel
    from models.experimental.deepseek_v4_flash.tt.model import plan_layer_placement
    from models.experimental.deepseek_v4_flash.tt.system_config import load_system_config, set_active_system_config
    from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
    from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

    if not STATES.exists():
        pytest.skip(f"{STATES} not found: run test_d_p_export first")
    host = load_host_states(STATES)
    ids = host["ids"].long()
    T = int(host["seq_len"])
    assert ids.numel() == T and T % 128 == 0, (ids.numel(), T)

    loader = DeepseekV4WeightLoader(os.environ.get("D_P_MODEL_DIR", DEFAULT_MODEL_DIR))
    config = load_config(loader.snapshot_dir)
    set_active_system_config(load_system_config(mesh_device=mesh_device))
    num_layers = max(config.kv_source_layer_ids) + 1
    placement = plan_layer_placement(num_layers, tuple(mesh_device.shape)[0], 1)
    submeshes = [
        mesh_device.create_submesh(ttnn.MeshShape(1, _TP_SIZE), ttnn.MeshCoordinate(k, 0))
        for k in range(placement[-1] + 1)
    ]
    layer_devices = [submeshes[k] for k in placement]
    rope = _build_rope(config, loader.snapshot_dir, T)
    engram_ids = [i for i in config.engram_layer_ids if i < num_layers]
    engram_lookup = None
    if engram_ids:
        tok = AutoTokenizer.from_pretrained(loader.snapshot_dir)
        tables = {i: load_engram_table(loader, i) for i in engram_ids}
        engram_lookup = EngramHostLookup(config, loader.snapshot_dir, tables, batch=1, max_seq=T, tokenizer=tok)
    model = DeepSeekV41PrefillModel(
        config,
        loader,
        layer_devices[0],
        rope,
        num_layers=num_layers,
        cache=WeightCache(CACHE_DIR) if CACHE_DIR else None,
        weight_dtype=ttnn.bfloat8_b,
        expert_dtype=ttnn.bfloat4_b,
        tp_size=_TP_SIZE,
        layer_devices=layer_devices,
        engram_lookup=engram_lookup,
        kv_only=True,
    )
    model.synchronize("uploads")
    model.prepare_traced_prefill(T, 1024)
    _, states = model.prefill_traced(ids.unsqueeze(0))

    pccs, bad = {}, []
    for i, state in enumerate(states):
        theirs = host["layers"][i]
        fields = [("kv_tail", state.kv_tail)] if i != num_layers - 1 else []  # (layer 20: metal_2 never writes it)
        if state.compressed_kv is not None:
            fields += [("compressed_kv", state.compressed_kv), ("idx_keys", state.idx_keys)]
        for name, t in fields:
            mine = model.to_host(t, model.layer_devices[i]).reshape(-1, t.shape[-1]).float()
            other = theirs[name].float()
            assert mine.shape == other.shape, (i, name, tuple(mine.shape), tuple(other.shape))
            p = _pcc(mine, other)
            pccs[(i, name)] = p
            logger.info(f"layer {i:>2} {name:<13} PCC {p:.6f}  max |diff| {(mine - other).abs().max():.4f}")
            if p < MIN_PCC:
                bad.append(f"layer {i} {name}: PCC {p:.6f} < {MIN_PCC}")
    model.free_traced_states(states)
    model.release_traced_prefill()

    converted = to_prefill_states(host, layer_devices, config)
    assert len(converted) == num_layers and all(s.seq_len == T for s in converted)
    free_prefill_states(converted)
    assert not bad, "metal_2 prefill states differ from this model's:\n" + "\n".join(bad)


@pytest.mark.timeout(3600)
def test_prefill_server():
    """A running ``demo/prefill_server.py`` (``D_P_SERVER=host:port``, no device in this process) prefills the
    prompt ``test_d_p_export`` saved, and its KV entries match the saved ones."""
    from models.experimental.deepseek_v41_flash.tt.prefill.d_p_bridge import load_host_states
    from models.experimental.deepseek_v41_flash.tt.prefill.d_p_wire import PrefillClient

    address = os.environ.get("D_P_SERVER")
    if not address:
        pytest.skip("set D_P_SERVER=host:port of a running prefill_server.py")
    if not STATES.exists():
        pytest.skip(f"{STATES} not found: run test_d_p_export first")
    saved = load_host_states(STATES)
    ids = saved["ids"].tolist()

    client = PrefillClient(address)
    logger.info(f"prefill server {address}: {client.info}")
    t0 = time.perf_counter()
    got = client.prefill(ids)
    wall = time.perf_counter() - t0
    client.close()
    T = got["seq_len"]
    logger.info(
        f"{T} tokens: server prefill {got['seconds']:.2f} s ({T / got['seconds']:.0f} tok/s), "
        f"round trip {wall:.2f} s"
    )
    assert T == len(ids) and sorted(got["layers"]) == sorted(saved["layers"])
    bad = []
    for i, layer in saved["layers"].items():
        for name, want in layer.items():
            have = got["layers"][i][name]
            assert have.shape == want.shape, (i, name, tuple(have.shape), tuple(want.shape))
            p = _pcc(have, want)
            if p < MIN_PCC:
                bad.append(f"layer {i} {name}: PCC {p:.6f}")
    assert not bad, "prefill server entries differ from the saved run:\n" + "\n".join(bad)
