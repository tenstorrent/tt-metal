# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-kernel-time harness for g6_concat (run under scripts/run_safe_pytest.sh --profile).

Each case runs the op twice so the second dispatch is a program-cache hit; read the generic_op
rows of the profiler CSV. Correctness is the acceptance test's job: values are checked here only
so a fast-but-wrong variant cannot be mistaken for a win. Cases whose cell is outside SUPPORTED
skip, so the perf-focus case (p1, float32) runs as soon as its refinement lands.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_concat import SUPPORTED, g6_concat
from ttnn.operations.g6_concat import g6_concat_program_descriptor as program_descriptor

_TORCH = {ttnn.bfloat16: (torch.bfloat16, torch.int16), ttnn.float32: (torch.float32, torch.int32)}

PERF_CASES = [
    pytest.param([(108, 4096, 128), (84, 4096, 128)], 0, ttnn.float32, id="p1_qwen18_fp32_outer"),
    pytest.param([(1, 6, 4096, 256)] * 16, 0, ttnn.bfloat16, id="p2_qwen17_outer"),
    pytest.param([(1, 4096, 1536)] * 16, 0, ttnn.bfloat16, id="p3_qwen19_outer"),
    pytest.param([(1, 4096, 1536)] * 8, 0, ttnn.bfloat16, id="p4_qwen23_outer"),
    pytest.param([(1, 1, 1, 62080)] * 16, 2, ttnn.bfloat16, id="qwen79_row_gather"),
    pytest.param([(1, 1, 3844, 64), (1, 1, 3844, 192)], -1, ttnn.bfloat16, id="qwen25_w"),
    pytest.param([(1, 1, 16, 16384)] * 3 + [(1, 1, 16, 12928)], -1, ttnn.bfloat16, id="qwen80_w_trim16"),
]


def _run(shapes, dim, dtype, device, repeats=2):
    if dtype not in SUPPORTED["dtype"]:
        pytest.skip(f"{dtype} not in SUPPORTED yet")
    torch_dtype, bits = _TORCH[dtype]
    torch.manual_seed(0)
    torch_inputs = [torch.randn(s, dtype=torch.float32).to(torch_dtype) for s in shapes]
    tt_inputs = [
        ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for t in torch_inputs
    ]
    for _ in range(repeats):
        output = g6_concat(tt_inputs, dim)
    ttnn.synchronize_device(device)
    expected = torch.cat(torch_inputs, dim=dim)
    assert torch.equal(ttnn.to_torch(output).to(torch_dtype).view(bits), expected.view(bits))


@pytest.mark.parametrize("shapes, dim, dtype", PERF_CASES)
def test_g6_concat_device_perf(shapes, dim, dtype, device):
    _run(shapes, dim, dtype, device)


@pytest.mark.parametrize("max_cores", [110, 55, 28])
def test_g6_concat_row_gather_core_scaling(max_cores, device):
    """Qwen 79 at fewer cores: per-core-issue-bound time scales with 1/cores, a DRAM-side bound does not."""
    saved = program_descriptor.MAX_CORES
    program_descriptor.MAX_CORES = max_cores
    try:
        _run([(1, 1, 1, 62080)] * 16, 2, ttnn.bfloat16, device)
    finally:
        program_descriptor.MAX_CORES = saved
