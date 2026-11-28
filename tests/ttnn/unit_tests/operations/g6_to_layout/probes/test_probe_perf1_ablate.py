# Perf 1 probe: device-ns of one kernel-dir variant (an ablated copy of kernels/) on one shape.
# Run under scripts/run_safe_pytest.sh --profile; launch 1 is warm-up, launches 2..N are timings.
#   G6_KDIR     kernel directory to build from; default: the op's own.
#   G6_SHAPE    JSON shape, e.g. [1,1024,2560].
#   G6_IN       TILE | RM (input layout; the target is the other one).
#   G6_BUF      DRAM | L1 (input and output buffer).
#   G6_DTYPE    bf16 | bf8b (input dtype, TILE input only).
#   G6_LAUNCHES launches (default 6).
#   G6_CHECK    1: bit-exact check (off for ablations, which move wrong data).
import json
import os
import sys
import importlib
from pathlib import Path

import pytest
import torch
import ttnn

importlib.import_module("ttnn.operations.g6_to_layout")
op_mod = sys.modules["ttnn.operations.g6_to_layout.g6_to_layout"]
import ttnn.operations.g6_to_layout.g6_to_layout_program_descriptor as pd
from ttnn.operations.g6_to_layout import g6_to_layout


def test_variant(device, monkeypatch):
    if os.environ.get("G6_KDIR"):
        monkeypatch.setattr(pd, "KERNEL_DIR", Path(os.environ["G6_KDIR"]))
    shape = tuple(json.loads(os.environ["G6_SHAPE"]))
    to_rm = os.environ.get("G6_IN", "TILE") == "TILE"
    buf = ttnn.L1_MEMORY_CONFIG if os.environ.get("G6_BUF", "DRAM") == "L1" else ttnn.DRAM_MEMORY_CONFIG
    dtype = ttnn.bfloat8_b if os.environ.get("G6_DTYPE", "bf16") == "bf8b" else ttnn.bfloat16
    op_mod._PLAN_CACHE.clear()
    op_mod._PROGRAM_CACHE.clear()
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.bfloat16)
    tt_x = ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.TILE_LAYOUT if to_rm else ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=buf
    )
    target = ttnn.ROW_MAJOR_LAYOUT if to_rm else ttnn.TILE_LAYOUT
    plan = op_mod.select_plan(tt_x, target)
    print(
        f"\nPLAN {shape} {plan['regime']} cores={plan['num_cores']} rows={plan['block_rows_tiles']} "
        f"cols={plan['block_cols_tiles']} l1={plan['l1_bytes_per_core']}"
    )
    for _ in range(int(os.environ.get("G6_LAUNCHES", "6"))):
        out = g6_to_layout(tt_x, target)
        ttnn.synchronize_device(device)
        if os.environ.get("G6_CHECK") == "1" and dtype == ttnn.bfloat16:
            got = ttnn.to_torch(out).reshape(shape)
            assert torch.equal(got.view(torch.int16), x.view(torch.int16))
        ttnn.deallocate(out)
    op_mod._PLAN_CACHE.clear()
    op_mod._PROGRAM_CACHE.clear()
