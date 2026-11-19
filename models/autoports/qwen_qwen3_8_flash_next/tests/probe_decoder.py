# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real-weight linear decoder prefill and changed-input trace replay probe."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights, pcc
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=33)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(16)
    torch.manual_seed(47)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    weights = load_weights(args.snapshot, 0, "")
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, 0)
    reference.load_state_dict(weights, assign=True)
    reference.eval()
    prompt = (torch.randn(1, args.seq_len, 10240) * 0.02).bfloat16()
    decode_inputs = [(torch.randn(1, 1, 10240) * 0.02).bfloat16() for _ in range(3)]
    cache = DynamicCache(config=config)
    with torch.no_grad():
        expected_prefill = reference(prompt, position_embeddings=(None, None), past_key_values=cache)
        expected_decode = [reference(x, position_embeddings=(None, None), past_key_values=cache) for x in decode_inputs]
    print("REFERENCE_OK", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = {"layer_idx": 0, "seq_len": args.seq_len, "weights": "real", "decode": []}
    trace_id = None
    try:
        model = FunctionalDecoder.from_state_dict(weights, hf_config=config, layer_idx=0, mesh_device=mesh)
        state = model.create_state()

        def host(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT)

        input_prompt = ttnn.to_device(host(prompt.unsqueeze(0)), mesh)
        input_decode = ttnn.to_device(host(decode_inputs[0].unsqueeze(0)), mesh)
        result = model.prefill_forward(input_prompt, state=state)
        actual = ttnn.to_torch(result).reshape(expected_prefill.shape)
        report["prefill_pcc"] = pcc(expected_prefill, actual)
        ttnn.deallocate(result)
        # Explicit test-only prefill/decode boundary. Preserve real prefix state
        # so compile/warmup cannot change the request used by the HF comparison.
        conv_snapshot = host(ttnn.to_torch(state.conv))
        recurrent_snapshot = host(ttnn.to_torch(state.recurrent), ttnn.float32)
        warmed = model.decode_forward(input_decode, state=state)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warmed)
        ttnn.copy_host_to_device_tensor(conv_snapshot, state.conv)
        ttnn.copy_host_to_device_tensor(recurrent_snapshot, state.recurrent)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = model.decode_forward(input_decode, state=state)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        # This exact output allocation is written by this sole live trace and
        # read before the next replay. No view/consumer outlives that ordering.
        trace_allocation_tracker.acknowledge_corruptible(output)
        for step, (x, expected) in enumerate(zip(decode_inputs, expected_decode)):
            ttnn.copy_host_to_device_tensor(host(x.unsqueeze(0)), input_decode)
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            actual = ttnn.to_torch(output).reshape(expected.shape)
            report["decode"].append(
                {"step": step, "pcc": pcc(expected, actual), "finite": bool(torch.isfinite(actual).all())}
            )
        report["trace_id"] = str(trace_id)
        report["captures"] = 1
        report["replays"] = len(expected_decode)
    finally:
        if trace_id is not None:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if report["prefill_pcc"] < 0.995 or any(x["pcc"] < 0.995 or not x["finite"] for x in report["decode"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
