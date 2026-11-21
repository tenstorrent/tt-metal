# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Thread-plan tests of sigmoid (op_design.md, Work Distribution, "Tests the plan needs").

A correct result alone does not show that the plan's threads ran: one NEO could have done all the work. These
tests show it, run the op under other plans, and run it in one device session between programs of tt-metal's
own (the tag hand-back of the implicit-sync kernels, target page *Ring protocols*).
"""

import pytest
import torch

import ttnn
from ttnn.operations.sigmoid import sigmoid
from ttnn.operations.sigmoid.sigmoid import _sigmoid
from ttnn.operations.sigmoid.sigmoid_program_artifacts import THREAD_PLAN, split_rounds

ATOL = 2.0**-7  # 2 bf16 ULP at the top of sigmoid's range [0.5, 1)


def _input(device, shape, seed=0, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    return x, t


def _check(got, expected):
    diff = (got.float() - expected.float()).abs().max().item()
    assert diff <= ATOL, diff


def test_each_neo_computes_its_own_tiles(device):
    """With probe_tag_neo, odd NEOs compute sigmoid(-x): core-local page i must come from NEO i mod C."""
    _, C, _ = THREAD_PLAN
    shape = (1, 3, 320, 320)  # 300 pages, 10 x 10 tiles per channel
    num_tiles = 300
    x, t = _input(device, shape)
    _, counts, starts = split_rounds(num_tiles, C, device.compute_with_storage_grid_size())
    local = {starts[core] + i: i for core, n in counts.items() for i in range(n)}
    assert sorted(local) == list(range(num_tiles)), "the split must cover every page exactly once"
    assert max(local.values()) >= C, "every NEO must get a page on some cluster"

    got = ttnn.to_torch(_sigmoid(t, None, probe_tag_neo=True)).float()
    per_neo = [0] * C
    for page in range(num_tiles):
        c, rest = divmod(page, 100)
        th, tw = divmod(rest, 10)
        at = (0, c, slice(th * 32, th * 32 + 32), slice(tw * 32, tw * 32 + 32))
        neo = local[page] % C
        per_neo[neo] += 1
        xs = x[at].double()
        expected = torch.sigmoid(-xs if neo & 1 else xs)
        diff = (got[at].double() - expected).abs().max().item()
        assert diff <= ATOL, (page, local[page], neo, diff)
    assert all(n > 0 for n in per_neo), per_neo


PLANS = [(1, 1, 1), (1, 4, 1), (2, 4, 2), (4, 4, 2)]
# pages: 1, 3 (< C), 39 (YOLOv8 small grid), 789 (YOLOv8 640x640)
SHAPES = [(32, 32), (32, 96), (1, 80, 400), (1, 80, 8400)]


@pytest.mark.parametrize("threads", PLANS, ids=lambda p: "R%dC%dW%d" % p)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_thread_plans(device, shape, threads):
    x, t = _input(device, shape)
    _check(ttnn.to_torch(_sigmoid(t, None, threads=threads)), torch.sigmoid(x.double()))


def test_one_device_session(device):
    """tt-metal program (a tilize) -> sigmoid -> sigmoid -> tt-metal program (an untilize, untagged transfers),
    all in one device session: shows the tag register is handed back at both ends. (ttnn.add is a Metal 1 op
    and does not run on Quasar; to_layout does.)"""
    shape = (1, 3, 320, 320)
    torch.manual_seed(1)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    rm_in = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tiled = ttnn.to_layout(rm_in, ttnn.TILE_LAYOUT)
    y1 = sigmoid(tiled)
    y2 = sigmoid(y1, compute_kernel_config=None)
    rm_out = ttnn.to_layout(y2, ttnn.ROW_MAJOR_LAYOUT)
    got = ttnn.to_torch(ttnn.from_device(rm_out)).float()
    # Reference from the device's own bf16 intermediate, so only the last sigmoid's rounding is compared.
    y1_host = ttnn.to_torch(y1)
    _check(y1_host, torch.sigmoid(x.double()))
    _check(got, torch.sigmoid(y1_host.double()))
