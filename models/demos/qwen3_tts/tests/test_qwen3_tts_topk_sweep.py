# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated sweep of the CP sampler's ``ttnn.topk`` width.

``_DeviceSampler.append_sampling`` pads the ``[1,1,1,2048]`` codec logits out to 8192
before ``topk(k=64)``, because ``topk``'s multi-core path used to require
``reduced_width >= multi_core_min_width`` (= 8192). This build added an Ht-aware
relaxation (``topk_utils.cpp::topk_multicore_structurally_eligible``):

    width_gate = (reduced_width >= 8192)
              || (num_tile_rows <= 2 && reduced_width >= 1024)

A ``[1,1,1,2048]`` logit row is ONE tile row, so it already qualifies at its native
width — making the pad (and 4x the sort work) obsolete.

The pad is also a no-op numerically: it fills with -1e4, and there are 2048 real logits,
so the top-64 of the padded tensor is exactly the top-64 of the unpadded one, in the same
order. Removing it should be bit-exact, which this test asserts with ``torch.equal``.

Run (N300):

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) ARCH_NAME=wormhole_b0 MESH_DEVICE=N300
    python -m tracy -p -v -r --no-runtime-analysis -m pytest -s -q \\
      models/demos/qwen3_tts/tests/test_qwen3_tts_topk_sweep.py
    CSV=$(ls -t generated/profiler/reports/*/ops_perf_results_*.csv | head -1)

Each arm's ``REP`` TopK rows appear in the CSV in the order printed below; take the min
per block (see module-perf-optimization/references/matmul-and-mlp.md micro-sweep note).
"""

import os

import pytest
import torch

import ttnn

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(*_a, **_k):
        pass


VOCAB = 2048  # Qwen3TTSCodePredictorConfig.vocab_size
TOPK = 64  # _SAMPLING_TOPK
NEG = -1e4  # _SAMPLING_NEG
REP = 5

# (label, padded width). 2048 = native; 8192 = what ships today.
ARMS = [("w2048_native", 2048), ("w4096_pad", 4096), ("w8192_pad_shipping", 8192)]


@pytest.fixture(scope="module")
def device():
    mesh_shape = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"))
    if mesh_shape is None:
        d = ttnn.open_device(device_id=0, l1_small_size=32768)
        d.enable_program_cache()
        yield d
        ttnn.close_device(d)
        return
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape), l1_small_size=32768)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)
    if mesh_shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _full_grid(device):
    g = device.compute_with_storage_grid_size()
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])


def test_topk_width_sweep(device):
    torch.manual_seed(0)
    logits = torch.randn(1, 1, 1, VOCAB, dtype=torch.bfloat16) * 8.0
    grid = _full_grid(device)

    def _make(width):
        t = torch.full((1, 1, 1, width), NEG, dtype=torch.bfloat16)
        t[..., :VOCAB] = logits
        return ttnn.from_torch(
            t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    inputs = {label: _make(w) for label, w in ARMS}

    # Compile every arm before the measured region.
    for label, _ in ARMS:
        v, i = ttnn.topk(inputs[label], k=TOPK, dim=-1, largest=True, sorted=True, sub_core_grids=grid)
        ttnn.deallocate(v)
        ttnn.deallocate(i)
    ttnn.synchronize_device(device)

    print("\nCSV TopK block order (REP rows each):")
    for label, w in ARMS:
        print(f"  {label:22} width={w}")

    signpost("start")
    for label, _ in ARMS:
        for _ in range(REP):
            v, i = ttnn.topk(inputs[label], k=TOPK, dim=-1, largest=True, sorted=True, sub_core_grids=grid)
            ttnn.deallocate(v)
            ttnn.deallocate(i)
        ttnn.synchronize_device(device)
    signpost("stop")

    # Bit-exactness: dropping the pad must not change the values or the indices.
    def _out(label):
        v, i = ttnn.topk(inputs[label], k=TOPK, dim=-1, largest=True, sorted=True, sub_core_grids=grid)
        from models.demos.qwen3_tts.tt.mesh_utils import to_torch as _mesh_to_torch

        vt, it = _mesh_to_torch(v), _mesh_to_torch(i)
        ttnn.deallocate(v)
        ttnn.deallocate(i)
        # A mesh replicates the input, so every chip must agree; compare chip 0's rows.
        return vt.reshape(-1, TOPK)[0], it.reshape(-1, TOPK)[0]

    v_native, i_native = _out("w2048_native")
    v_ship, i_ship = _out("w8192_pad_shipping")
    assert torch.equal(v_native, v_ship), "topk VALUES differ between native and padded width"
    assert torch.equal(i_native.long(), i_ship.long()), "topk INDICES differ between native and padded width"
    print("[topk_sweep] native width is bit-exact with the shipping 8192 pad (values + indices)")

    for t in inputs.values():
        ttnn.deallocate(t)
