# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving-contract step: the model through the prefill engine's adapter/runtime API (see testing/contract.py).
Run with --no-precompile: the precompile pass stubs comp_pcc, which the producer's read-back uses."""

from models.demos.common.bringup.testing.contract import engine_env
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
engine_env(S)  # before any adapter import

from models.demos.common.bringup.testing.contract import run_contract_test  # noqa: E402


@mesh_parametrize
def test_contract(mesh_device):
    failed = run_contract_test(S, mesh_device)
    assert not failed, failed
