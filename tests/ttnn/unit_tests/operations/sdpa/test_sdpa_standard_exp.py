# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SDPA exponential coverage for FP32 standard and BF16 streaming destination paths."""
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


# FP32 destination accumulation selects the standard compute path; BF16 destination
# accumulation selects streaming. Inputs, masks and outputs remain BF16 in both.
# On the standard path, False must honor accurate exp and the full-FP32 attention
# scale. The frozen gate allows BF16 quantization while detecting larger distortion.
# One K chunk isolates the main exp; two chunks cover explicit -inf and a fully
# masked later chunk, while every query keeps valid earlier keys. Unit/nonunit scale
# catch missing or duplicate scaling. Repeated False/True calls check reused state.
@pytest.mark.parametrize("k_chunks", [1, 2], ids=["one-k-chunk", "masked-second-k-chunk"])
@pytest.mark.parametrize("scale", [1.0, 128**-0.5], ids=["unit-scale", "nonunit-scale"])
@pytest.mark.parametrize("fp32_dest_acc_en", [True, False], ids=["fp32-dest", "bf16-dest"])
def test_standard_sdpa_exp_modes(device, k_chunks, scale, fp32_dest_acc_en, record_property):
    device.enable_program_cache()
    host_inputs = make_fixture(k_chunks)
    golden = reference(*host_inputs, scale)
    # Fail before device execution when a changed fixture makes correlation undefined.
    for head in range(4):
        metrics(golden[:, head], golden[:, head])

    compute = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=False,
    )
    grid = device.compute_with_storage_grid_size()
    programs = {
        mode: ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid, q_chunk_size=128, k_chunk_size=512, exp_approx_mode=mode
        )
        for mode in (False, True)
    }
    owned, output = [], None
    saved, measured, failures = {}, [], []
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
        for iteration, mode in enumerate((False, True, False, True)):
            output = ttnn.transformer.scaled_dot_product_attention(
                owned[0],
                owned[1],
                owned[2],
                attn_mask=owned[3],
                is_causal=False,
                scale=scale,
                program_config=programs[mode],
                compute_kernel_config=compute,
            )
            cache_entries.append(device.num_program_cache_entries())
            if iteration == 0:
                # Non-unit scale also caches the public API's mask multiplication.
                assert cache_entries[0] > cache_entries_before, "First SDPA call did not populate the program cache"
            else:
                # The other exp mode adds one SDPA program; both repeated calls must hit.
                expected_entries = cache_entries[0] + 1
                assert cache_entries[-1] == expected_entries, (
                    f"Expected {expected_entries} cache entries after iteration {iteration} "
                    f"(exp_approx_mode={mode}), got {cache_entries[-1]}; history={cache_entries}"
                )
            actual = ttnn.to_torch(output).clone()
            output.deallocate(True)
            output = None
            assert torch.isfinite(actual).all(), "SDPA returned nonfinite values"
            if mode in saved:
                assert torch.equal(saved[mode], actual), "Repeated same-mode output differs"
            else:
                saved[mode] = actual
            for head in range(4):
                values = metrics(golden[:, head], actual[:, head])
                measured.append({"iteration": iteration, "exp_approx_mode": mode, "head": head, **values})
                # Streaming uses approximate main-score exp for both modes, so use the
                # approximate PCC >= 0.998 gate; standard accurate exp keeps its PCC/L2 gate.
                passed = passes_false_gate(values) if fp32_dest_acc_en and not mode else values["pcc"] >= TRUE_PCC_MIN
                if not passed:
                    failures.append(measured[-1])
        for original, tensor in zip(host_inputs, owned):
            assert torch.equal(original, ttnn.to_torch(tensor)), "SDPA changed an input tensor"
        mismatched_elements = int((saved[False] != saved[True]).sum())
        record_property("false_true_mismatched_elements", mismatched_elements)
        if fp32_dest_acc_en:
            # Standard exp must distinguish modes. Streaming can produce identical
            # outputs because its main-score exp stays approximate in both modes.
            assert mismatched_elements > 0, "exp_approx_mode=False and True returned identical outputs"
    finally:
        record_property("standard_sdpa_exp_cache_entries", json.dumps(cache_entries))
        record_property("standard_sdpa_exp_metrics", json.dumps(measured, allow_nan=False))
        if output is not None:
            output.deallocate(True)
        while owned:
            owned.pop().deallocate(True)
    assert not failures, json.dumps(failures, indent=2, allow_nan=False)
