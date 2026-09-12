# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Diagnose the exact real B3/S257 changed-token failure; never replace its HF gate."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.reference import load_config, load_layer_weights, make_reference
from models.autoports.qwen_qwen3_8_27b.tests.run_fused_decoder import device_only, pcc
from models.autoports.qwen_qwen3_8_27b.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_27b.tt.fused_decoder import FusedDecoder


def metrics(expected, actual):
    delta = expected.float() - actual.float()
    return {
        "pcc": pcc(expected, actual),
        "per_user_pcc": [pcc(a, b) for a, b in zip(expected, actual)],
        "rmse": delta.square().mean().sqrt().item(),
        "max_abs": delta.abs().max().item(),
        "bitwise_equal": torch.equal(expected, actual),
    }


def main(args):
    torch.set_num_threads(4)
    config = load_config(args.snapshot)
    weights = load_layer_weights(args.snapshot, 0)
    reference = make_reference(config, 0, weights)
    rope = Qwen3_5TextRotaryEmbedding(config)
    torch.manual_seed(380)
    x = (torch.randn(3, 258, config.hidden_size) * 0.1).bfloat16()
    cos, sin = rope(x, torch.arange(258).unsqueeze(0).expand(3, -1))
    cache = DynamicCache(config=config)
    raw_states = []
    original_update = cache.update_recurrent_state

    def capture(state, *positional, **kwargs):
        raw_states.append(state.detach().clone())
        return original_update(state, *positional, **kwargs)

    with torch.no_grad(), patch.object(cache, "update_recurrent_state", capture):
        expected_prefill = reference(x[:, :-1], position_embeddings=(cos[:, :-1], sin[:, :-1]), past_key_values=cache)
    raw_state = raw_states[-1]
    hf_state = cache.layers[0].recurrent_states.float().clone()
    hf_conv = cache.layers[0].conv_states[..., -3:].transpose(1, 2).contiguous()
    with torch.no_grad():
        expected_original = reference(
            x[:, -1:], position_embeddings=(cos[:, -1:], sin[:, -1:]), past_key_values=copy.deepcopy(cache)
        )
    # Exactly reproduce run_fused_decoder's page permutation before changed_x.
    torch.randperm(30)
    changed = (torch.randn(3, 1, config.hidden_size) * 0.1).bfloat16()
    changed_pos = torch.tensor([257, 256, 255], dtype=torch.int32)
    cc, ss = rope(changed, changed_pos[:, None])
    report = {
        "implementation": "functional" if args.baseline else "fused",
        "variant": args.variant,
        "source_sha256": hashlib.sha256(
            Path("models/autoports/qwen_qwen3_8_27b/tt")
            .joinpath("functional_decoder.py" if args.baseline else "fused_decoder.py")
            .read_bytes()
        ).hexdigest(),
        "case": {"layer": 0, "batch": 3, "length": 257, "seed": 380, "continuation_chunks": [33, 128, 96]},
        "precision_policy": {
            "activations": "BF16",
            "weights": "BF16",
            "matmul": "HiFi4 FP32 destination",
            "recurrent": "FP32",
            "conv": {
                "dtype": "BF16",
                "layout": "tile" if args.baseline else "row major",
                "shape": list(hf_conv.shape),
            },
            "ccl": "none",
            "pages": "unused linear attention; original randperm(30) retained",
            "hf_recurrent": str(cache.layers[0].recurrent_states.dtype),
            "hf_conv": str(cache.layers[0].conv_states.dtype),
            "hf_pre_copy_recurrent": str(raw_state.dtype),
        },
        "hf_cache_rounding": metrics(raw_state, hf_state),
        "trials": {},
    }
    tensors = {
        "x": x,
        "changed_x": changed,
        "hf_raw_recurrent": raw_state,
        "hf_recurrent": hf_state,
        "hf_conv": hf_conv,
    }

    def oracle(recurrent, conv):
        diagnostic_cache = copy.deepcopy(cache)
        diagnostic_cache.layers[0].recurrent_states = recurrent.clone()
        diagnostic_cache.layers[0].conv_states[..., -3:] = conv.transpose(1, 2)
        with torch.no_grad():
            return reference(changed, position_embeddings=(cc, ss), past_key_values=diagnostic_cache)

    # Stock BF16 HF cache is kept intact as the primary acceptance oracle.
    with torch.no_grad():
        stock_expected = reference(changed, position_embeddings=(cc, ss), past_key_values=copy.deepcopy(cache))
    tensors["stock_hf_changed"] = stock_expected
    raw_expected = oracle(raw_state, hf_conv)
    report["hf_stock_vs_raw_fp32_output"] = metrics(stock_expected, raw_expected)
    tensors["raw_fp32_hf_changed"] = raw_expected
    print("HF_READY", json.dumps(report), flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        decoder_cls = FunctionalDecoder if args.baseline else FusedDecoder
        decoder = decoder_cls.from_state_dict(weights, hf_config=config, layer_idx=0, mesh_device=mesh)

        def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                value.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        if args.variant == "fp32_folded_norm":
            for name in ("input_layernorm.weight", "post_attention_layernorm.weight"):
                old = decoder.weights[name]
                decoder.weights[name] = upload((weights[name].float() + 1).reshape(1, 1, -1), ttnn.float32)
                ttnn.deallocate(old)
        elif args.variant in ("norm_compute", "native_default"):

            def explicit_norm(value, name):
                return ttnn.rms_norm(
                    value,
                    weight=decoder.weights[name + ".weight"],
                    epsilon=decoder.eps,
                    compute_kernel_config=decoder.ckc if args.variant == "norm_compute" else None,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            decoder._norm = explicit_norm

        state = decoder.allocate_state(batch_size=3, num_pages=30)
        original_output = decoder.prefill_forward(upload(x[:, :-1]), state=state)
        report["prefill"] = metrics(expected_prefill, ttnn.to_torch(original_output))
        ttnn.deallocate(original_output)
        state = decoder.allocate_state(batch_size=3, num_pages=30)
        first = decoder.prefill_forward(upload(x[:, :33]), state=state)
        tail = decoder.prefill_forward(upload(x[:, 33:-1]), state=state, start_pos=33)
        report["continuation"] = metrics(
            expected_prefill, torch.cat([ttnn.to_torch(first), ttnn.to_torch(tail)], dim=1)
        )
        ttnn.deallocate(first)
        ttnn.deallocate(tail)
        tt_state, tt_conv = ttnn.to_torch(state.recurrent), ttnn.to_torch(state.conv)
        tensors.update(tt_recurrent=tt_state, tt_conv=tt_conv)
        report["prefix_state"] = {
            "tt_vs_hf_stored": metrics(hf_state, tt_state),
            "tt_vs_hf_raw": metrics(raw_state, tt_state),
            "conv": metrics(hf_conv, tt_conv),
        }
        dx = upload(x[:, -1:])
        pos = upload(changed_pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        def restore(recurrent, conv):
            for target, value, dtype, layout in [
                (state.recurrent, recurrent, ttnn.float32, ttnn.TILE_LAYOUT),
                (state.conv, conv, ttnn.bfloat16, state.conv.layout),
            ]:
                host = ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout)
                ttnn.copy_host_to_device_tensor(host, target)

        def decode():
            with device_only():
                return decoder.decode_forward(dx, state=state, current_pos=pos)

        original_output = decode()
        report["original_eager"] = metrics(expected_original, ttnn.to_torch(original_output))
        ttnn.deallocate(original_output)
        restore(tt_state, tt_conv)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        host = ttnn.from_torch(changed.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, dx)
        trials = [
            ("tt_prefix", tt_state, tt_conv),
            ("hf_bf16_prefix", hf_state, hf_conv),
            ("hf_fp32_prefix", raw_state, hf_conv),
            ("hf_conv_only", tt_state, hf_conv),
            ("hf_recurrent_only", hf_state, tt_conv),
        ]
        for name, recurrence, conv in trials:
            expected_aligned = oracle(recurrence, conv)
            restore(recurrence, conv)
            eager = decode()
            actual_eager = ttnn.to_torch(eager)
            eager_state, eager_conv = ttnn.to_torch(state.recurrent), ttnn.to_torch(state.conv)
            ttnn.deallocate(eager)
            restore(recurrence, conv)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            actual_trace = ttnn.to_torch(traced)
            row = {
                "stock_hf_vs_eager": metrics(stock_expected, actual_eager),
                "stock_hf_vs_trace": metrics(stock_expected, actual_trace),
                "aligned_hf_vs_eager": metrics(expected_aligned, actual_eager),
                "stock_hf_vs_aligned_hf": metrics(stock_expected, expected_aligned),
                "eager_vs_trace": metrics(actual_eager, actual_trace),
                "eager_trace_recurrent": metrics(eager_state, ttnn.to_torch(state.recurrent)),
                "eager_trace_conv": metrics(eager_conv, ttnn.to_torch(state.conv)),
            }
            report["trials"][name] = row
            tensors[name + "_changed"] = actual_trace
            tensors[name + "_hf_aligned"] = expected_aligned
            print("TRIAL", name, json.dumps(row), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        report["primary_0995_gate_passed"] = report["trials"]["tt_prefix"]["stock_hf_vs_trace"]["pcc"] >= 0.995
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        torch.save(tensors, args.output.with_suffix(".pt"))
        print("PROBE_DONE", json.dumps(report), flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--variant", choices=("none", "fp32_folded_norm", "norm_compute", "native_default"), default="none"
    )
    main(parser.parse_args())
