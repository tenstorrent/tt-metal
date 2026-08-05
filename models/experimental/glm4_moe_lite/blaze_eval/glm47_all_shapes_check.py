# SPDX-License-Identifier: Apache-2.0
"""Is blaze's DRAMStreamingMatmul correct at EVERY GLM-4.7-Flash decode matmul shape?

o_proj (5120x2048) is already measured at 2.45x. This widens the adoption question: which of
the model's other per-layer matmuls could also move to the streaming path?

Shapes from zai-org/GLM-4.7-Flash config.json (hidden 2048, q_lora 768, kv_lora 512,
20 heads x (qk_nope 192 + qk_rope 64), v_head 256, moe intermediate 1536):

    q_a_proj      2048 ->  768      per layer
    kv_a_proj     2048 ->  576      per layer   (kv_lora 512 + qk_rope 64)
    q_b_proj       768 -> 5120      per layer   (20 heads x 256)
    o_proj        5120 -> 2048      per layer   VALIDATED, 2.45x
    mlp_gate/up   2048 -> 1536      dense layer 0 and shared expert
    mlp_down      1536 -> 2048      dense layer 0 and shared expert

m=1 is the real bs=1 decode case and the only m blaze is correct at for these shapes -- m=32
is broken (F1). bf8 weights, matching DENSE_TT_DTYPE=bf8.

A shape that fails here cannot be adopted; a shape that passes is a candidate for the same
A/B that gave o_proj its number.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path("/home/ttuser/sdawle/tt-blaze/tests/blaze/micro_ops/common")))

import pytest

import ttnn
from test_dram_streaming_matmul import _run_and_compare

GLM_SHAPES = [
    pytest.param(2048, 768, id="q_a_proj_2048x768"),
    pytest.param(2048, 576, id="kv_a_proj_2048x576"),
    pytest.param(768, 5120, id="q_b_proj_768x5120"),
    pytest.param(5120, 2048, id="o_proj_5120x2048"),
    pytest.param(2048, 1536, id="mlp_gate_up_2048x1536"),
    pytest.param(1536, 2048, id="mlp_down_1536x2048"),
]


@pytest.mark.parametrize("k, n", GLM_SHAPES)
def test_glm47_decode_matmul_shapes(device, k, n):
    _run_and_compare(device, k=k, n=n, m=1, weight_dtype=ttnn.bfloat8_b)
