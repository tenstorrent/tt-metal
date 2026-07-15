# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# mm_help profiling harness — isolated single SDPA decode for the main-vs-mm_help5 A/B.
# Underscore-prefixed so ambient `pytest tests/...` does NOT collect it; run EXPLICITLY
# under Tracy on each branch-build:
#
#   SD_B=8 SD_NH=8 SD_NKV=1 SD_S=32768 SD_D=128 SD_GX=8 SD_GY=6 \
#   python -m tracy -r -p -v -m pytest tests/ttnn/unit_tests/operations/sdpa/_sdpa_decode_mmhelp_bench.py
#
# Delegates to run_test_sdpa_decode_single_iter (the repo's decode driver): it builds
# Q/K/V, an SDPAProgramConfig (q_chunk=padded_num_heads, k_chunk=get_chunk_size), a
# compute config, calls ttnn.transformer.scaled_dot_product_attention_decode, and does a
# causal PCC check. Called twice (cold + warm) so the extractor takes the warm
# ScaledDotProductAttentionDecodeDeviceOperation. Why-fields come from the Tracy CSV.
#
# Env: SD_B, SD_NH, SD_NKV, SD_S, SD_D, grid SD_GX/SD_GY,
#      SD_KV_DTYPE (bfloat8_b), SD_Q_DTYPE (bfloat16).
import os
import pytest
import ttnn
from tests.ttnn.unit_tests.operations.sdpa.sdpa_test_utils import run_test_sdpa_decode_single_iter

_DT = {"bfloat16": ttnn.bfloat16, "bfloat8_b": ttnn.bfloat8_b, "bfloat4_b": ttnn.bfloat4_b, "float32": ttnn.float32}


def _e(name, default, cast):
    v = os.environ.get(name)
    return default if v is None or v == "" else cast(v)


CFG = dict(
    b=_e("SD_B", 8, int), nh=_e("SD_NH", 8, int), nkv=_e("SD_NKV", 1, int),
    s=_e("SD_S", 32768, int), d=_e("SD_D", 128, int),
    gx=_e("SD_GX", 8, int), gy=_e("SD_GY", 6, int),
    kv_dtype=_e("SD_KV_DTYPE", ttnn.bfloat8_b, lambda v: _DT[v]),
    q_dtype=_e("SD_Q_DTYPE", ttnn.bfloat16, lambda v: _DT[v]),
)


def test_sdpa_decode_mmhelp_bench(device):
    c = CFG
    if c["nkv"] > 1 and c["q_dtype"] != ttnn.bfloat16:
        pytest.skip("nkv > 1 requires q_dtype to be bfloat16")
    for _ in range(2):  # cold + warm; extractor takes the warm op
        run_test_sdpa_decode_single_iter(
            device, c["b"], c["nh"], c["nkv"], c["s"], c["d"], c["kv_dtype"],
            (c["gx"], c["gy"]), c["q_dtype"], cur_pos_tensor=False,
            sharded_in=False, sharded_out=False,
        )
