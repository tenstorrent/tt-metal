# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import time

import pytest
import torch

import ttnn


@pytest.fixture(scope="module")
def device_1x1():
    """One Blackhole chip as a 1x1 mesh; trace_region_size=0 lets TTNN place traces dynamically."""
    torch.set_num_threads(16)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    yield mesh
    ttnn.close_mesh_device(mesh)


def _real_cases():
    from models.autoports.google_gemma_4_12b_it.tests.hf_reference import text_config

    return [f"real_L{i}" for i in range(text_config().num_hidden_layers)]


def _needed_cases(session):
    out = []
    for item in session.items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        case = params.get("case")
        if case and case not in out:
            out.append(case)
        if params.get("real_group") is not None:
            out += [c for c in _real_cases() if c not in out]
    return out


@pytest.fixture(scope="session")
def qb2(request):
    """Single-chip baselines for every selected case (subprocess, own device open), then the QB2 mesh."""
    from models.autoports.google_gemma_4_12b_it.tests.single_chip_baseline import load as load_baseline

    needed = [c for c in _needed_cases(request.session) if load_baseline(c) is None]
    if needed:
        t0 = time.time()
        cmd = [sys.executable, "-m", "models.autoports.google_gemma_4_12b_it.tests.single_chip_baseline"]
        subprocess.run(cmd + ["--cases", ",".join(needed)], check=True)
        print(f"single-chip baselines for {len(needed)} cases in {time.time() - t0:.0f} s")
    from models.autoports.google_gemma_4_12b_it.tt.multichip_decoder import close_qb2_mesh, open_qb2_mesh

    torch.set_num_threads(16)
    mesh = open_qb2_mesh(trace_region_size=0)
    yield mesh
    close_qb2_mesh(mesh)
