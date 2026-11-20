# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device test of the split load on a 4x8 mesh: bytes on all 32 chips and the DRAM layout equal a plain load's.

Set SPLIT_LOAD_OUT=<file> to write the timings as JSON.
"""

from __future__ import annotations

import json
import os
import time

import pytest
import torch

import ttnn
from models.tt_dit.utils import split_load

from ...layers.module import Module, Parameter
from ...parallel.manager import CCLManager

MESH = [
    pytest.param(
        (4, 8),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
        id="mesh4x8",
    )
]

# (name, total shape, dtype, mesh_axes): replicated bf16 and float32, a block-float one (plain), one already
# split over a mesh axis (axis 0 only is free), a tensor too small to split, and a rank-2 weight.
SPECS = [
    ("rep_bf16", (1, 1, 8192, 4096), ttnn.bfloat16, None),
    ("rep_f32", (1, 1, 4096, 2048), ttnn.float32, None),
    ("rep_b8", (1, 1, 4096, 4096), ttnn.bfloat8_b, None),
    ("ax1_bf16", (1, 1, 4096, 8192), ttnn.bfloat16, (None, None, None, 1)),
    ("small", (1, 1, 256, 256), ttnn.bfloat16, None),
    ("rank2", (8192, 2048), ttnn.bfloat16, None),
    ("rep_bf16_b", (1, 1, 4096, 8192), ttnn.bfloat16, None),
]
EXPECT_SPLIT = {"rep_bf16", "rep_f32", "ax1_bf16", "rank2", "rep_bf16_b"}


class Toy(Module):
    def __init__(self, mesh_device) -> None:
        super().__init__()
        for name, shape, dtype, axes in SPECS:
            setattr(self, name, Parameter(total_shape=shape, device=mesh_device, dtype=dtype, mesh_axes=axes))


def _chip_tensors(param: Parameter) -> list[torch.Tensor]:
    return [ttnn.to_torch(t) for t in ttnn.get_device_tensors(param.data)]


@pytest.mark.parametrize(("mesh_device", "device_params"), MESH, indirect=["mesh_device", "device_params"])
def test_split_load_equals_plain_load(mesh_device, tmp_path):
    torch.manual_seed(0)
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    state = {n: torch.randn(*shape) for n, shape, _, _ in SPECS}
    cache = tmp_path / "cache"
    source = Toy(mesh_device)
    source.load_torch_state_dict(state)
    source.save(cache)
    source.deallocate_weights()

    report = {}

    plain = Toy(mesh_device)
    t = time.monotonic()
    with split_load.disabled():
        plain.load(cache)
    ttnn.synchronize_device(mesh_device)
    report["plain_s"] = round(time.monotonic() - t, 3)
    plain_chips = {n: _chip_tensors(p) for n, p in plain.named_parameters()}
    plain_addr = {n: [t.buffer_address() for t in ttnn.get_device_tensors(p.data)][0] for n, p in plain.named_parameters()}
    plain.deallocate_weights()

    loader = split_load.SplitLoad(ccl)
    split = Toy(mesh_device)
    t = time.monotonic()
    with loader.active():
        split.load(cache)
    ttnn.synchronize_device(mesh_device)
    report["split_s"] = round(time.monotonic() - t, 3)
    report["stats"] = dict(loader.stats)
    assert loader.failed is None, loader.failed
    assert loader.stats["split"] == len(EXPECT_SPLIT), loader.stats

    for name, param in split.named_parameters():
        got = _chip_tensors(param)
        assert len(got) == len(plain_chips[name]) == 32
        for chip, (a, b) in enumerate(zip(got, plain_chips[name], strict=True)):
            assert torch.equal(a, b), f"{name}: chip {chip} differs from a plain load"
        param._check_data(param.data)
    # the same allocation order into an empty heap: every buffer sits at the address a plain load gave it
    split_addr = {n: [t.buffer_address() for t in ttnn.get_device_tensors(p.data)][0] for n, p in split.named_parameters()}
    report["addresses_equal"] = split_addr == plain_addr
    report["addresses_plain"] = plain_addr
    report["addresses_split"] = split_addr
    out = os.environ.get("SPLIT_LOAD_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(report, f, indent=1)
    print("SPLITLOAD", json.dumps({k: v for k, v in report.items() if not k.startswith("addresses_p")}))
    assert split_addr == plain_addr, "DRAM layout differs from a plain load"
