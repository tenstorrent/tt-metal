# SPDX-License-Identifier: Apache-2.0
"""Heterogeneous per-request QSA positions through one reused decode trace.

The cache is built through legal forward calls, without synthetic cache loads:
first prefill 32 tokens in both lanes; then start a fresh 33-token request in
lane zero while lane one continues for 33 tokens from position32. The resulting
request prefixes are33 and65. Decode advances through the end of each four-token
indexer group while permuting slots, page rows, positions and input activations.
"""

import pytest
import torch

from models.autoports.qwen_qwen3_8_flash_next.tests.test_qsa import (
    LAYER,
    load_attention_reference,
    pcc,
    reference_forward,
)


def heterogeneous_case(reference, rotary):
    """CPU reference inputs; each lane uses a separate real-HF attention call."""
    requests = [
        torch.randn(1, 37, 2560, generator=torch.Generator().manual_seed(333)).mul_(0.1).bfloat16(),
        torch.randn(1, 69, 2560, generator=torch.Generator().manual_seed(665)).mul_(0.1).bfloat16(),
    ]
    initial_old_request = torch.randn(1, 32, 2560, generator=torch.Generator().manual_seed(999)).mul_(0.1).bfloat16()
    first_prefill = torch.cat((initial_old_request, requests[1][:, :32]), dim=0)
    second_prefill = torch.cat((requests[0][:, :33], requests[1][:, 32:65]), dim=0)
    expected = [reference_forward(reference, rotary, request) for request in requests]
    assert first_prefill.shape == (2, 32, 2560)
    assert second_prefill.shape == (2, 33, 2560)
    for request, output in zip(requests, expected):
        assert request.shape == output.shape and torch.isfinite(output).all()
    return requests, first_prefill, second_prefill, expected


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [{"trace_region_size": 0}], indirect=True)
def test_qsa_trace_heterogeneous_positions(mesh_device):
    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedQSA as QSA

    config, reference, rotary, state, _ = load_attention_reference(real=True)
    requests, first_prefill, second_prefill, expected = heterogeneous_case(reference, rotary)
    qsa = QSA.from_state_dict(
        state,
        hf_config=config,
        layer_idx=LAYER,
        mesh_device=mesh_device,
        batch_size=2,
        max_context=2048,
        max_requests=2,
    )
    pages_cpu = torch.randperm(128, generator=torch.Generator().manual_seed(113)).reshape(2, 64).int()
    slots_cpu = torch.zeros(2, 32, dtype=torch.int64)
    slots_cpu[:, 0] = torch.arange(2)
    positions_cpu = torch.zeros(2, 32, dtype=torch.int64)

    def upload(value, dtype, layout):
        return ttnn.from_torch(value.contiguous(), device=mesh_device, dtype=dtype, layout=layout)

    def refresh(value, destination, dtype, layout):
        # Explicit test boundary: host tensors update stable device input buffers.
        host = ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout)
        ttnn.copy_host_to_device_tensor(host, destination)

    pages = upload(pages_cpu, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    slots = upload(slots_cpu, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    positions = upload(positions_cpu, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    first_input = upload(first_prefill.unsqueeze(0), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    second_input = upload(second_prefill.unsqueeze(0), ttnn.bfloat16, ttnn.TILE_LAYOUT)

    qsa.prefill_forward(first_input, current_pos=positions, page_table=pages, slots=slots)
    positions_cpu[:, 0] = torch.tensor([0, 32])
    refresh(positions_cpu, positions, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    second_output = qsa.prefill_forward(second_input, current_pos=positions, page_table=pages, slots=slots)
    second_expected = torch.cat((expected[0][:, :33], expected[1][:, 32:65]), dim=0)
    second_pcc = pcc(second_expected, ttnn.to_torch(second_output).squeeze(0))
    assert second_pcc >= 0.995, second_pcc

    base_positions = torch.tensor([33, 65])
    initial_decode = torch.cat((requests[0][:, 33:34], requests[1][:, 65:66]), dim=0)
    decode_input = upload(initial_decode.unsqueeze(0), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    positions_cpu[:, 0] = base_positions
    refresh(positions_cpu, positions, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    qsa.decode_forward(decode_input, current_pos=positions, page_table=pages, slots=slots)
    ttnn.synchronize_device(mesh_device)
    trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    output = qsa.decode_forward(decode_input, current_pos=positions, page_table=pages, slots=slots)
    ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    results = []
    try:
        for step, permutation in enumerate(([1, 0], [0, 1], [1, 0], [0, 1])):
            logical_positions = base_positions + step
            permutation_tensor = torch.tensor(permutation)
            next_inputs = torch.cat(
                [requests[slot][:, logical_positions[slot] : logical_positions[slot] + 1] for slot in permutation],
                dim=0,
            )
            next_expected = torch.cat(
                [expected[slot][:, logical_positions[slot] : logical_positions[slot] + 1] for slot in permutation],
                dim=0,
            )
            positions_cpu[:, 0] = logical_positions[permutation_tensor]
            refresh(next_inputs.unsqueeze(0), decode_input, ttnn.bfloat16, ttnn.TILE_LAYOUT)
            refresh(positions_cpu, positions, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            refresh(pages_cpu[permutation_tensor], pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            refresh(slots_cpu[permutation_tensor], slots, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
            correlation = pcc(next_expected, ttnn.to_torch(output).squeeze(0))
            assert correlation >= 0.995, (step, permutation, positions_cpu[:, 0].tolist(), correlation)
            result = {
                "step": step,
                "slots": permutation,
                "positions": positions_cpu[:, 0].tolist(),
                "trace_id": trace,
                "pcc": correlation,
            }
            results.append(result)
            print(f"QSA_HETEROGENEOUS_TRACE {result}", flush=True)
        assert len({str(row["trace_id"]) for row in results}) == 1
    finally:
        ttnn.release_trace(mesh_device, trace)
    print(f"QSA_HETEROGENEOUS_PREFILL_PCC={second_pcc:.12f}", flush=True)
