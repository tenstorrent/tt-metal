# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in fixed-vs-paged GLM-5.2 SparseMLA parity contract.

This is deliberately test glue rather than a dependency on a provisional model
API. Set ``TT_PREFILL_PAGING_TEST_BACKEND`` to a Python module exposing::

    run_glm52_sparse_mla_parity(
        scenario, mesh_device, device_params, physical_pages
    ) -> Mapping[str, torch.Tensor | float | Sequence[int]]

Required tensor artifacts are asserted below.  The backend should use identical
weights and inputs for fixed and paged runs, return caches in natural logical
token order, and compile/warm both paths before returning optional timings.
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tests.paged_prefill.support import (
    assert_bank_balance,
    load_sparse_mla_backend,
    require_artifacts,
)
from tests.ttnn.utils_for_testing import assert_with_pcc

QB_PARITY_CASES = [
    pytest.param(
        (4, 1),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_udm_mode": ttnn.FabricUDMMode.ENABLED,
        },
        id="qb-sp4xtp1-ring",
    ),
]

SCENARIOS = [
    pytest.param(
        {
            "name": "full-layer",
            "full_layer": 6,
            "shared_layer": None,
            "output_pcc": 0.98,
            "kvpe_pcc": 0.99,
            "index_pcc": 0.999,
        },
        id="full-layer6",
    ),
    pytest.param(
        {
            "name": "full-to-shared-reuse",
            "full_layer": 6,
            "shared_layer": 7,
            "output_pcc": 0.98,
            "kvpe_pcc": 0.99,
            "index_pcc": 0.999,
            "reuse_pcc": 0.9999,
        },
        id="full6-to-shared7",
    ),
]

QB_OPT_IN = os.environ.get("TT_RUN_PAGED_PREFILL_QB_TESTS") == "1"


@pytest.mark.parametrize(
    "mesh_device,device_params",
    QB_PARITY_CASES,
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.skipif(not QB_OPT_IN, reason="set TT_RUN_PAGED_PREFILL_QB_TESTS=1 on an exclusively assigned QB")
@pytest.mark.skipif(not is_blackhole(), reason="GLM-5.2 sparse prefill is Blackhole-only")
@pytest.mark.timeout(0)
def test_glm52_fixed_vs_paged_sparse_mla(
    mesh_device,
    device_params,
    scenario,
):
    backend = load_sparse_mla_backend()
    if backend is None:
        pytest.skip("set TT_PREFILL_PAGING_TEST_BACKEND to the paging implementation's test glue module")

    physical_pages = (3, 0, 2)
    artifacts = backend.run_glm52_sparse_mla_parity(
        scenario=scenario,
        mesh_device=mesh_device,
        device_params=device_params,
        physical_pages=physical_pages,
    )
    common_keys = [
        "fixed_output",
        "paged_output",
        "fixed_kvpe",
        "paged_kvpe",
        "fixed_index",
        "paged_index",
    ]
    require_artifacts(artifacts, common_keys)

    assert_with_pcc(artifacts["fixed_output"], artifacts["paged_output"], scenario["output_pcc"])
    assert_with_pcc(artifacts["fixed_kvpe"], artifacts["paged_kvpe"], scenario["kvpe_pcc"])
    assert_with_pcc(artifacts["fixed_index"], artifacts["paged_index"], scenario["index_pcc"])

    if scenario["shared_layer"] is not None:
        require_artifacts(artifacts, ["full_topk", "shared_topk"])
        assert_with_pcc(artifacts["full_topk"], artifacts["shared_topk"], scenario["reuse_pcc"])

    if "page_bank_ids" in artifacts:
        num_banks = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).num_banks
        assert_bank_balance(artifacts["page_bank_ids"], num_banks)

    if "fixed_device_seconds" in artifacts or "paged_device_seconds" in artifacts:
        require_artifacts(artifacts, ["fixed_device_seconds", "paged_device_seconds"])
        assert artifacts["fixed_device_seconds"] > 0
        assert artifacts["paged_device_seconds"] > 0

    # Catch a backend accidentally returning the same object for both paths.
    for fixed_key, paged_key in (
        ("fixed_output", "paged_output"),
        ("fixed_kvpe", "paged_kvpe"),
        ("fixed_index", "paged_index"),
    ):
        assert isinstance(artifacts[fixed_key], torch.Tensor)
        assert isinstance(artifacts[paged_key], torch.Tensor)
        assert artifacts[fixed_key] is not artifacts[paged_key]
