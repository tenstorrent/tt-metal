# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Where does building a layer spend its time? Builds layers (default 3, 4, 3) and prints BUILDT lines (DSV41_BUILD_PROFILE=1)."""

import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.model import DSV41DecodeChain


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}, id="ring")],
    indirect=True,
)
@torch.no_grad()
def test_build_profile(mesh_device):
    os.environ["DSV41_BUILD_PROFILE"] = "1"
    CHAIN = "/mnt/tt-data/ssinghal/dsv4-chain-m"
    S = torch.load(os.path.join(CHAIN, "tokens.pt"))["prefill_tokens"].shape[1]
    chain = DSV41DecodeChain(
        mesh_device, log=print
    )  # one chain: layers that read another layer's compressed cache need its source built first
    for L in [int(x) for x in os.environ.get("DSV41_PROFILE_LAYERS", "2,3,4,3").split(",")]:
        ref = torch.load(os.path.join(CHAIN, f"layer_{L}.pt"))
        ref["S"] = S
        t = time.perf_counter()
        chain.build_layer(L, ref)
        ttnn.synchronize_device(mesh_device)
        print(f"BUILDT ==== layer {L} total {time.perf_counter() - t:.2f} s", flush=True)
