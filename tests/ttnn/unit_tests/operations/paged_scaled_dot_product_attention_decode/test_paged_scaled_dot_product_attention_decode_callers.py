# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""The callers' arguments no golden cell passes (the runner builds a ComputeConfigDescriptor and no program config):

- rule R10: tt-metal's device compute kernel config (`ttnn.WormholeComputeKernelConfig`, what Llama's sixteen calls
  pass: HiFi2, fp32_dest_acc_en=False, packer_l1_acc=False) is taken, read by attribute, not refused for its type;
- rule R11: a `ttnn.SDPAProgramConfig` with any fields (a grid of 64 cores on this device of 32, k_chunk_size 0 or
  32, exp_approx_mode True) is a hint: the result is graded exactly as without it.

Each call is checked by the specification's own check (HOOKS["check"]): values, then placement.
"""

import pytest
import torch
import ttnn

from eval import golden_runner
from eval.specs.paged_scaled_dot_product_attention_decode import spec as S
from ttnn.operations.paged_scaled_dot_product_attention_decode import paged_scaled_dot_product_attention_decode

OP = "paged_scaled_dot_product_attention_decode"
CELL = {"dtype": ttnn.bfloat16, "layout": ttnn.TILE_LAYOUT, "memory_config": "L1", "sharding": "height",
        "output_memory": "DRAM"}


def _program_config(grid, k_chunk_size, max_cores):
    return ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=0, k_chunk_size=k_chunk_size,
                                  exp_approx_mode=True, max_cores_per_head_batch=max_cores)


CASES = [
    # id, inputs, position, device config fields (None: none passed), program config
    ("llama_split_device_config", S.LLAMA_SPLIT, 200,
     dict(math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=False),
     lambda: _program_config((2, 1), 32, 1)),
    ("llama_captured_grid_8x8", S.LLAMA_CAPTURED, 8, None, lambda: _program_config((8, 8), 0, 16)),
    ("qwen_two_kv_heads_program_config", S.QWEN_TWO_KV_HEADS, 8, None, lambda: _program_config((2, 1), 0, 1)),
]


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_callers_arguments(device, case):
    _, inputs, position, device_config, program_config = case
    fp32 = True if device_config is None else device_config["fp32_dest_acc_en"]
    spec = golden_runner.bound(OP).spec
    cell, plan = golden_runner.build_cell(OP, inputs, {**CELL, "fp32_dest_acc_en": fp32}, device,
                                          {"positions": [position], "page_table": "identity"}, seed=3)
    dev = golden_runner.device_tensors(plan, cell.tensors, device)
    call = golden_runner.bind_call(spec, dev, cell.kwargs)
    kwargs = dict(call.kwargs)
    kwargs["program_config"] = program_config()
    if device_config is None:
        kwargs.pop("compute_kernel_config", None)  # the callers pass none: the default factory's config
    else:
        kwargs["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(**device_config)
    output = paged_scaled_dot_product_attention_decode(*call.args, **kwargs)
    expected = spec.reference(**{p: t.to(torch.float64) for p, t in cell.tensors.items()},
                              **{k: v for k, v in cell.kwargs.items() if k == "scale"})
    spec.hooks["check"](cell, output, expected)
