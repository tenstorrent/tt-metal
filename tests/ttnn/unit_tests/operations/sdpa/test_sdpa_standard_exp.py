# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SDPA exponential coverage of the ACCURATE (FP32 state, accurate exp) and STANDARD (BF16 state) recipes."""
import json

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.sdpa.standard_exp_test_utils import (
    TRUE_PCC_MIN,
    make_fixture,
    metrics,
    passes_false_gate,
    reference,
)


# The exp is part of the precision recipe (exp_approx_mode is ignored): ACCURATE must pass the accurate-exp gate (BF16
# quantization only), STANDARD the approximate PCC >= 0.998 gate. One K chunk isolates the main exp; two chunks cover
# explicit -inf and a fully masked later chunk, while every query keeps valid earlier keys. Unit/nonunit scale catch
# missing or duplicate scaling (of the scores and of the attn_mask, which compute multiplies by 1/scale). Repeated
# calls check reused state and program-cache hits.
@pytest.mark.parametrize("k_chunks", [1, 2], ids=["one-k-chunk", "masked-second-k-chunk"])
@pytest.mark.parametrize("scale", [1.0, 128**-0.5], ids=["unit-scale", "nonunit-scale"])
@pytest.mark.parametrize(
    "precision", [ttnn.SDPAPrecision.ACCURATE, ttnn.SDPAPrecision.STANDARD], ids=["accurate", "standard"]
)
def test_standard_sdpa_exp_modes(device, k_chunks, scale, precision, record_property):
    device.enable_program_cache()
    host_inputs = make_fixture(k_chunks)
    golden = reference(*host_inputs, scale)
    # Fail before device execution when a changed fixture makes correlation undefined.
    for head in range(4):
        metrics(golden[:, head], golden[:, head])

    grid = device.compute_with_storage_grid_size()
    program = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=512)
    accurate = precision == ttnn.SDPAPrecision.ACCURATE
    owned, output = [], None
    saved, measured, failures = None, [], []
    cache_entries = []
    try:
        for original in host_inputs:
            tensor = ttnn.from_torch(
                original,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                device=device,
            )
            owned.append(tensor)
            assert torch.equal(original, ttnn.to_torch(tensor)), "Device upload changed a BF16 operand"
        cache_entries_before = device.num_program_cache_entries()
        for iteration in range(3):
            output = ttnn.transformer.scaled_dot_product_attention(
                owned[0],
                owned[1],
                owned[2],
                attn_mask=owned[3],
                is_causal=False,
                scale=scale,
                program_config=program,
                precision=precision,
            )
            cache_entries.append(device.num_program_cache_entries())
            if iteration == 0:
                assert cache_entries[0] == cache_entries_before + 1, "the first call adds exactly one SDPA program"
            else:
                assert cache_entries[-1] == cache_entries[0], f"repeated calls must hit the cache: {cache_entries}"
            actual = ttnn.to_torch(output).clone()
            output.deallocate(True)
            output = None
            assert torch.isfinite(actual).all(), "SDPA returned nonfinite values"
            if saved is not None:
                assert torch.equal(saved, actual), "Repeated output differs"
            saved = actual
            for head in range(4):
                values = metrics(golden[:, head], actual[:, head])
                measured.append({"iteration": iteration, "head": head, **values})
                passed = passes_false_gate(values) if accurate else values["pcc"] >= TRUE_PCC_MIN
                if not passed:
                    failures.append(measured[-1])
        for original, tensor in zip(host_inputs, owned):
            assert torch.equal(original, ttnn.to_torch(tensor)), "SDPA changed an input tensor"
    finally:
        record_property("standard_sdpa_exp_cache_entries", json.dumps(cache_entries))
        record_property("standard_sdpa_exp_metrics", json.dumps(measured, allow_nan=False))
        if output is not None:
            output.deallocate(True)
        while owned:
            owned.pop().deallocate(True)
    assert not failures, json.dumps(failures, indent=2, allow_nan=False)
