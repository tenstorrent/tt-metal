# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.moe_block.flat_routed_expert on one Blackhole chip (a 10-row grid):

* test_flat_routed_expert_pcc: GLM-5.3-Flash at the Galaxy per-chip load against a torch reference of the
  bfp4-quantized weights, two calls (the second with a moved arena: the cache hit must patch every address).
* test_flat_routed_expert_bit_exact: bit-exact against saved reference outputs; runs only with MOE_BLOCK_REFS set.
"""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]
OP = "flat_routed_expert"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cases = _load("flat_cases", HERE / "flat_cases.py")
harness = _load("moe_block_ref_harness", HERE.parent / "ref_harness.py")
weights_py = _load(
    "flat_routed_expert_weights",
    REPO / "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert_weights.py",
)


def _skip_unless_supported(device):
    if ttnn.get_arch_name() != "blackhole":
        pytest.skip("Blackhole only")
    if device.compute_with_storage_grid_size().y < 10:
        pytest.skip("laid out for a 10-row Blackhole grid")


def _bits(t):
    return t.contiguous().view({2: torch.int16, 4: torch.int32}[t.element_size()])


def _rm(t, dt, device):
    return ttnn.from_torch(
        t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def _op(device, model, chunk, weights):
    md, pc = cases.MODELS[model], cases.per_chip(model, "8x4", chunk)
    E = pc["epc"]
    gids = [(md["E"] // E) * e + 1 for e in range(E)]
    return weights_py.FlatRoutedExpert(
        device,
        weights,
        m=chunk,
        H=md["H"],
        I=md["I"],
        gids=[gids],
        n_global=md["E"],
        act=md["act"],
        pin=1,
        layout=dict(rows_hint=round(pc["M"])),
    )


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_flat_routed_expert_pcc(device):
    _skip_unless_supported(device)
    model, chunk = "glm53_flash", 5120
    md, E = cases.MODELS[model], cases.per_chip(model, "8x4", chunk)["epc"]
    wsets = cases.weight_sets(md["H"], md["I"])
    op = _op(device, model, chunk, [[wsets[e % cases.N_W] for e in range(E)]])
    q = lambda w: ttnn.to_torch(ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()  # noqa: E731
    wq = [tuple(q(w) for w in ws) for ws in wsets]
    pinned = []
    entries = None
    # (the dispatch buffer's spec is the program's: both calls' x padded to one row count - only the counts differ)
    calls = [cases.case_inputs(model, chunk, seed) for seed in (1, 2)]
    rows = max(c[0].shape[0] for c in calls)
    for call, (x, c_, r_, _, spans) in enumerate(calls):
        x = torch.nn.functional.pad(x, (0, 0, 0, rows - x.shape[0]))
        y = op(_rm(x, ttnn.bfloat16, device), _rm(c_, ttnn.uint32, device), _rm(r_, ttnn.uint32, device))
        y = ttnn.to_torch(y).float()
        for e, (o, c) in enumerate(spans):
            if not c:
                continue
            sel = list(range(0, c, max(1, c // 48)))[:48] + [c - 1]
            xe = x[o : o + c][sel].to(torch.bfloat16).float()
            wg, wu, wd = wq[e % cases.N_W]
            ref = weights_py.act_ref(xe @ wg, xe @ wu, md["act"]) @ wd
            ok, pcc = comp_pcc(ref, y[o : o + c][sel], 0.99)
            assert ok, (call, e, c, pcc)
        if entries is None:
            entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == entries, "the second call missed the program cache"
        # the next call's per-call arena then sits lower: the cache hit must patch every address
        pinned.append(
            ttnn.from_torch(
                torch.zeros(32 * 110 // 16, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
        )
    op.close()


@pytest.mark.timeout(3600)
@pytest.mark.skipif("MOE_BLOCK_REFS" not in os.environ, reason="reference outputs: set MOE_BLOCK_REFS")
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("case", harness.list_cases(OP) or ["none"])
def test_flat_routed_expert_bit_exact(device, case):
    if case == "none":
        pytest.skip(f"no saved {OP} case under {harness.REFS_ROOT}")
    _skip_unless_supported(device)
    meta, ref = harness.load_case(OP, case)
    cfg, chip = meta["config"], ref[(0, 0)]
    wsets = cases.weight_sets(cfg["H"], cfg["I"], meta["weights"]["seed"])
    assert cases.weights_digest(wsets) == meta["weights"]["sha256"], "the weights did not regenerate bit for bit"
    op = weights_py.FlatRoutedExpert(
        device,
        [[wsets[e % cases.N_W] for e in range(cfg["E"])]],
        m=cfg["m"],
        H=cfg["H"],
        I=cfg["I"],
        gids=[chip["global_expert_ids"].tolist()],
        n_global=cfg["NG"],
        act=cfg["act"],
        pin=cfg["pin"],
        layout=cfg["layout"],
    )
    lay = op.lay
    for k in cases.PLAN_KEYS:
        assert lay[k] == meta["plan"][k], (k, lay[k], meta["plan"][k])
    for k in cases.PLAN_CORE_KEYS:
        assert [list(c) for c in lay[k]] == meta["plan"][k], k
    y = op(
        _rm(chip["x"], ttnn.bfloat16, device),
        _rm(chip["counts"], ttnn.uint32, device),
        _rm(chip["regions"], ttnn.uint32, device),
        y_row_major=cfg["y_row_major"],
    )
    y = ttnn.to_torch(y)
    want = chip["y"]
    assert y.dtype == want.dtype and y.shape == want.shape, (y.dtype, y.shape, want.dtype, want.shape)
    # (only the active experts' rows are written: the rest of y is whatever the allocation held)
    for e, (o, c) in enumerate(cfg["spans"]):
        got, exp = y[o : o + c], want[o : o + c]
        bad = (_bits(got) != _bits(exp)).sum()
        assert int(bad) == 0, f"{case}: expert {e} ({c} rows at {o}): {int(bad)} elements differ"
    op.close()
