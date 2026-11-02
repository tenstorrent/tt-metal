# Refinement 3 ablation harness (perf only, no correctness). Variants are text edits of the production kernels
# copied into a scratch dir; the descriptor's KERNEL_DIR is pointed at it.
import math
from pathlib import Path

import pytest
import torch
import ttnn

import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

SRC = Path(pd.__file__).parent / "kernels"
SCRATCH = Path(__file__).parent / "ablate_kernels"

EDITS = {
    "nogelu": [("compute", "using Gelu = GeluTanhActivation<gelu_fast>;", "using Gelu = NoneActivation;")],
    "skipmm": [("compute", "#include <cstdint>", "#define SKIP_COMPUTE\n#include <cstdint>")],
    "now": [
        (
            "reader",
            "noc_async_read_page(\n                        k * n_tiles",
            "if (false) noc_async_read_page(\n                        k * n_tiles",
        ),
        ("reader", "noc_async_read_page(n_first + w", "if (false) noc_async_read_page(n_first + w"),
    ],
    "nox": [("reader", "noc_async_read_page(x_page + k", "if (false) noc_async_read_page(x_page + k")],
    "fillzone": [
        ("compute", "#include <cstdint>", "#include <cstdint>\n#include <api/debug/kernel_profiler.hpp>"),
        (
            "compute",
            "    const uint32_t num_row_blocks = (core_rows + block_h - 1) / block_h;",
            '    {\n        DeviceZoneScopedN("FILL");\n        act_buf.wait_front(act_block_tiles);\n'
            "        weight_buf.wait_front(w_chunk_tiles);\n    }\n"
            "    const uint32_t num_row_blocks = (core_rows + block_h - 1) / block_h;",
        ),
    ],
    "readerzones": [
        ("reader", "#include <cstdint>", '#include <cstdint>\n#include "api/debug/kernel_profiler.hpp"'),
        (
            "reader",
            "        if (x_mcast.should_send(x_round_offset + row_block_idx)) {\n",
            '        if (x_mcast.should_send(x_round_offset + row_block_idx)) {\n            DeviceZoneScopedN("XSEND");\n',
        ),
        (
            "reader",
            "            x_receiver->receive(x_round_offset + row_block_idx);",
            '            DeviceZoneScopedN("XRECV");\n            x_receiver->receive(x_round_offset + row_block_idx);',
        ),
        (
            "reader",
            "            noc_async_read_barrier();\n            // Bias",
            '            {DeviceZoneScopedN("WBAR"); noc_async_read_barrier();}\n            // Bias',
        ),
        (
            "reader",
            "            w_sender->send(chunk_addr, chunk_addr, w_chunk_tiles * tile_bytes);",
            '            {DeviceZoneScopedN("WSEND"); w_sender->send(chunk_addr, chunk_addr, w_chunk_tiles * tile_bytes);}',
        ),
        (
            "reader",
            "            w_receiver->receive(w_round_offset + n_block_idx);",
            '            if (n_block_idx == 0) {DeviceZoneScopedN("WRECV0"); w_receiver->receive(w_round_offset + n_block_idx);} else w_receiver->receive(w_round_offset + n_block_idx);',
        ),
    ],
    "nowrite": [("writer", "noc_async_write_page(page_row + c", "if (false) noc_async_write_page(page_row + c")],
}

VARIANTS = {
    "fill": ["fillzone"],
    "rz": ["readerzones"],
    "full": [],
    "nogelu": ["nogelu"],
    "nocompute": ["nogelu", "skipmm"],
    "nocompute_now": ["nogelu", "skipmm", "now"],
    "nocompute_now_nox": ["nogelu", "skipmm", "now", "nox"],
    "syncfloor": ["nogelu", "skipmm", "now", "nox", "nowrite"],
    "now_only": ["now"],
    "nox_only": ["nox"],
    "nowrite_only": ["nowrite"],
}


def _make(variant):
    d = SCRATCH / variant
    d.mkdir(parents=True, exist_ok=True)
    for kind in ("reader", "compute", "writer"):
        text = (SRC / f"linear_gelu_{kind}.cpp").read_text()
        for name in VARIANTS[variant]:
            for k, old, new in EDITS[name]:
                if k == kind:
                    assert old in text, (variant, old)
                    text = text.replace(old, new)
        (d / f"linear_gelu_{kind}.cpp").write_text(text)
    return d


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("batch, seq", [(2, 512)])
def test_ablate(device, batch, seq, variant, monkeypatch):
    monkeypatch.setattr(pd, "KERNEL_DIR", _make(variant))
    device.disable_and_clear_program_cache()
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    xd = ttnn.from_torch(x, memory_config=ttnn.L1_MEMORY_CONFIG, **tile)
    wd = ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    bd = ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile)
    out = linear_gelu(xd, wd, bd, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.synchronize_device(device)
    print(f"VARIANT {batch}x{seq} {variant}")
