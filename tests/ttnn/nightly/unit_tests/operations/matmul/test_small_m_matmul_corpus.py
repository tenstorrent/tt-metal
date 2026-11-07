# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Correctness smoke for ttnn.experimental.small_m_matmul with config=None over the ~60-shape production
# corpus (FLUX / LTX transformer projections at Mt = 1..16). Each shape runs once against a Torch reference
# at PCC >= 0.999. Slower than the unit suite (~60 program compiles), so it lives in nightly; run it after
# any change to the picker, planner or kernels.
#
# The corpus is the shape set the lookup table in small_m_matmul_config.cpp was measured on, deduped.

import pytest
import torch
import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc

# --- corpus sources -------------------------------------------------------
_MT8_SHAPES = [
    (32, 2048, 512),
    (32, 2048, 1536),
    (32, 2048, 2048),
    (32, 256, 6144),
    (32, 6080, 4640),
    (32, 6100, 4608),
    (32, 6144, 1536),
    (32, 6144, 2304),
    (32, 6144, 3072),
    (32, 6144, 4600),
    (32, 6144, 4608),
    (32, 6144, 6144),
    (32, 6144, 9216),
    (48, 6144, 4608),
    (64, 15360, 1536),
    (64, 4608, 6144),
    (64, 6080, 4640),
    (64, 6144, 1536),
    (64, 6144, 4608),
    (64, 6144, 9216),
    (128, 15360, 768),
    (128, 2304, 6144),
    (128, 6080, 4640),
    (128, 6144, 2304),
    (128, 6144, 4608),
    (128, 6144, 768),
    (256, 2048, 1024),
    (256, 6080, 4640),
]
_TAIL = [
    (32, 6080, 4640),
    (64, 6080, 4640),
    (128, 6080, 4640),
    (256, 6080, 4640),
    (32, 6144, 4600),
    (32, 6100, 4608),
    (48, 6144, 4608),
]
_MATRIX_M = [32, 64, 128, 256]
_MATRIX_KN = [
    (2048, 512),
    (2048, 1024),
    (2048, 1536),
    (2048, 2048),
    (6144, 768),
    (6144, 1536),
    (6144, 2304),
    (6144, 4608),
    (6144, 6144),
    (15360, 768),
    (15360, 1536),
    (2304, 6144),
]


def _corpus():
    shapes = set(_MT8_SHAPES) | set(_TAIL)
    for m in _MATRIX_M:
        for k, n in _MATRIX_KN:
            shapes.add((m, k, n))
    return sorted(shapes)


_CORPUS = _corpus()


@pytest.mark.skipif(not is_blackhole(), reason="small-M matmul is Blackhole-only")
@pytest.mark.parametrize("M,K,N", _CORPUS, ids=[f"{m}x{k}x{n}" for (m, k, n) in _CORPUS])
def test_small_m_maskzero_corpus_smoke(device, M, K, N):
    # config=None -> auto_select_config + planner. LOGICAL inputs (no manual padding).
    torch.manual_seed(0)
    t0 = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    t1 = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    ref = (t0.float() @ t1.float())[0, 0]
    a0 = ttnn.from_torch(t0, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    wcfg = ttnn.create_small_m_weight_memory_config(list(t1.shape), ttnn.bfloat16, device)
    a1 = ttnn.from_torch(t1, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16, memory_config=wcfg)
    out = ttnn.experimental.small_m_matmul(a0, a1)
    got = ttnn.to_torch(ttnn.from_device(out))[0, 0]
    assert tuple(got.shape) == tuple(ref.shape), f"shape {tuple(got.shape)} != {tuple(ref.shape)}"
    assert torch.isfinite(got.float()).all(), "non-finite output"
    assert_with_pcc(ref, got.float(), 0.999)
