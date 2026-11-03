# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""P.28: fill the routed experts' laid-out weight cache (EXPERTS_WEIGHT_CACHE, tt/experts.py) for every layer of the
model in one EXPERTS_IMPL value, without building the rest of the model. Not a gate.

Why: a cold cache costs ~150-195 s a layer (host dequant + the value's rounding + the flat op's layout + the dump), so
a ladder run at rung last on a value with no cache (40 layers, ~2 h) passes pytest's 3600 s limit while still loading
(P.28's first measure stopped at layer 19). Here the layers are loaded ten at a time, one test each, through the same
loader and arguments the MoE block uses (tt/moe_block_adapter.py:_RoutedWeights, chunk = the spec's target chunk,
EXPERTS_LAYOUT from the settings), so the files are the ones the model then hits. Each layer's load logs EXPERTS_BFP4
(changed fraction, worst ||dW|| / ||W||) and EXPERTS_LOAD; a layer already cached costs ~4 s.

    DS41_EXPERTS_IMPL=bfp4_mse PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/perf/test_experts_cache_warm.py -s
"""

from __future__ import annotations

import time

import pytest

from models.demos.common.bringup.testing import harness
from models.demos.common.bringup.testing.harness import spec

S = spec()
GROUP = 10  # layers per test (each under its own timeout)
N_LAYERS = 40  # DeepSeek-V4.1-Flash (asserted against the checkpoint's config below)


@pytest.mark.timeout(3000)
@pytest.mark.parametrize("first", list(range(0, N_LAYERS, GROUP)), ids=lambda a: f"L{a:02d}-{a + GROUP - 1:02d}")
@pytest.mark.parametrize("device_params", [harness.device_params(S)], indirect=True, ids=["box"])
@pytest.mark.parametrize("mesh_device", [tuple(S.mesh)], indirect=True, ids=["x".join(map(str, S.mesh))])
def test_warm(mesh_device, first):
    from models.demos.deepseek_v41_flash_d_p.bringup import hooks
    from models.demos.deepseek_v41_flash_d_p.tt.experts import expert_cache_dir
    from models.demos.deepseek_v41_flash_d_p.tt.moe_block_adapter import _RoutedWeights
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    ck, cfg, chunk = hooks._checkpoint(S), hooks._config(S), hooks._chunk(S)
    assert cfg.n_layers == N_LAYERS, cfg.n_layers
    cdir = expert_cache_dir(S)
    assert settings.get("EXPERTS_WEIGHT_CACHE") == "on", "EXPERTS_WEIGHT_CACHE=off: nothing to warm"
    rows = []
    for layer in range(first, min(first + GROUP, cfg.n_layers)):
        t0 = time.time()
        rw = _RoutedWeights(
            mesh_device,
            ck,
            layer,
            num_experts=cfg.n_experts,
            hidden=cfg.dim,
            inter=cfg.inter,
            top_k=cfg.topk_experts,
            swiglu_limit=cfg.swiglu_limit,
            max_chunk=chunk,
            cache_dir=cdir,
        )
        st = rw.bfp4_stats or {}
        rows.append((layer, "hit" if rw._host_s is None else "miss", round(time.time() - t0, 1), st))
        rw.free()
        print(
            f"EXPERTS_CACHE_WARM impl={settings.get('EXPERTS_IMPL')} layer={layer} {rows[-1][1]} {rows[-1][2]} s "
            f"changed={st.get('changed_frac')} worst_rel={st.get('worst_rel')} worst={st.get('worst_matrix')}"
        )
    assert rows
