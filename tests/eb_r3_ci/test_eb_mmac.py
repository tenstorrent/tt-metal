# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Round 3 eltwise binary (#58723 review): dit_minimal_matmul_addcmul_fused (minimal_matmul compute_metal2.cpp, its fused
addcmul path) at the per-device shapes of tt_dit's Wan 2.2 and LTX and of Qwen image 2.1, with their fidelity and bias."""
import importlib.util

import pytest
import ttnn

_spec = importlib.util.spec_from_file_location(
    "dit_mmac", "/work/tests/ttnn/nightly/unit_tests/operations/experimental/test_dit_minimal_matmul_addcmul_fused.py"
)
m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m)

H2, LO = ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi
SHAPES = [
    ("wan_4x8", 9472, 5120, 1280, True, H2, (8, 8, 8)),
    ("wan_2x2_480p", 16384, 5120, 2560, True, H2, (8, 8, 8)),
    ("ltx_stage1", 2432, 4096, 2048, True, H2, (8, 8, 8)),
    ("ltx_stage2", 9696, 4096, 2048, True, H2, (8, 8, 8)),
    ("ltx_audio", 64, 2048, 1024, True, H2, (2, 8, 8)),
    ("qwen_wo", 4096, 4096, 4096, False, LO, (4, 8, 16)),
    ("qwen_down", 4096, 12288, 4096, False, LO, (4, 8, 16)),
]


@pytest.mark.parametrize("name, M, K, N, bias, fid, blocks", SHAPES, ids=[s[0] for s in SHAPES])
def test_mmac(device, name, M, K, N, bias, fid, blocks):
    r = m.run_dit_minimal_matmul_addcmul_fused_test(
        device,
        M,
        K,
        N,
        use_bias=bias,
        math_fidelity=fid,
        M_block_size=blocks[0],
        K_block_size=blocks[1],
        N_block_size=blocks[2],
    )
    assert r["pcc"] > 0.99, r
