# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-context GDN state-path evidence using real target shapes and weights.

Reduced layer-level harness: excludes token-local hyper-connections and MoE,
which have separate full-decoder checks. Every attention output is computed
and compared; this is not a seeded final-position-only test. The longest
nonaligned prefix and its traced final valid decode are checked separately.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights


def metric(expected, actual):
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import row_pcc

    actual = actual.reshape(expected.shape)
    values = row_pcc(expected, actual)
    return {
        "min_row_pcc": values.min().item(),
        "mean_row_pcc": values.mean().item(),
        "worst_row": int(values.argmin().item()),
        "first_eight_row_pcc": values[:8].tolist(),
        "last_eight_row_pcc": values[-8:].tolist(),
        "rows": values.numel(),
        "finite": bool(torch.isfinite(actual).all()),
        "passed": bool(torch.isfinite(values).all() and (values >= 0.995).all()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--length", type=int, default=262144)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument(
        "--device-chunk", type=int, help="Discriminating device chunk size; HF fixture remains unchanged"
    )
    parser.add_argument("--layer", type=int, choices=(0, 1), default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--fixture-cache", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(2194 + args.layer)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    assert 2 <= args.length <= config.max_position_embeddings and args.chunk >= 1
    weights = load_weights(args.snapshot, args.layer, "linear_attn.")
    with torch.device("meta"):
        reference = Qwen4ExpTextGatedDeltaNet(config, args.layer)
    reference.load_state_dict({k.removeprefix("linear_attn."): v for k, v in weights.items()}, assign=True)
    reference.eval()
    identity = {
        "format": "gdn_context_reference_v1",
        "snapshot": str(args.snapshot.resolve()),
        "config_sha256": hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest(),
        "hf_source_sha256": hashlib.sha256(Path(inspect.getfile(Qwen4ExpTextGatedDeltaNet)).read_bytes()).hexdigest(),
        "layer": args.layer,
        "length": args.length,
        "chunk": args.chunk,
        "seed": 2194 + args.layer,
    }
    cached = (
        torch.load(args.fixture_cache, weights_only=True)
        if args.fixture_cache and args.fixture_cache.exists()
        else None
    )
    if cached is not None:
        assert cached["identity"] == identity, "Stale CPU reference cache"
        x, expected = cached["input"], cached["expected"]
    else:
        x = (torch.randn(1, args.length, 2560) * 0.2).bfloat16()
        cache = DynamicCache(config=config)
        parts = []
        with torch.inference_mode():
            for start in range(0, args.length - 1, args.chunk):
                end = min(start + args.chunk, args.length - 1)
                parts.append(reference(x[:, start:end], cache_params=cache))
                if start % (args.chunk * 16) == 0:
                    print("HF_CONTEXT_ROWS", end, flush=True)
            parts.append(reference(x[:, -1:], cache_params=cache))
        expected = torch.cat(parts, dim=1)
        del parts
        if args.fixture_cache:
            partial = args.fixture_cache.with_suffix(".partial.pt")
            torch.save({"identity": identity, "input": x, "expected": expected}, partial)
            partial.replace(args.fixture_cache)
    final = expected[:, -1:]
    report = {
        "scope": "reduced real-weight GDN state path; excludes token-local residual mixing and MoE",
        "layer": args.layer,
        "advertised_context": config.max_position_embeddings,
        "prefill_length": args.length,
        "nonaligned_prefix": args.length - 1,
        "decode_position": args.length - 1,
        "chunk": args.chunk,
        "every_attention_output_compared": True,
        "runtime_audit": True,
        "reference_identity": identity,
    }
    if args.cpu_only:
        report["runtime_audit"] = False
        report["device_executed"] = False
        report["reference_rows"] = args.length
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("GDN_CPU_REFERENCE_READY", flush=True)
        return
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import GatedDeltaNet, WeightLoader

    report["device_executed"] = True
    device_chunk = args.device_chunk or args.chunk
    report["device_chunk"] = device_chunk
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        module = GatedDeltaNet(WeightLoader(weights, mesh), config)
        full = ttnn.from_torch(x.unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        decode = ttnn.from_torch(
            x[:, -1:].unsqueeze(0).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )
        conv = ttnn.zeros((1, 1, 3, 10240), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
        recurrent = ttnn.zeros((1, 48, 128, 128), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        zero_conv, zero_recurrent = ttnn.clone(conv), ttnn.clone(recurrent)
        prefix_conv, prefix_recurrent = ttnn.clone(conv), ttnn.clone(recurrent)

        def reset():
            ttnn.copy(zero_conv, conv)
            ttnn.copy(zero_recurrent, recurrent)

        def prefill(length):
            outputs = []
            with runtime_audit():
                for start in range(0, length, device_chunk):
                    outputs.append(
                        module(
                            full[:, :, start : min(start + device_chunk, length)],
                            conv_state=conv,
                            recurrent_state=recurrent,
                        )
                    )
                return ttnn.concat(outputs, dim=2) if len(outputs) > 1 else outputs[0]

        # The full logical prefill is one device-only call with bounded chunks.
        output = prefill(args.length)
        ttnn.synchronize_device(mesh)
        report["prefill"] = metric(expected, ttnn.to_torch(output))
        ttnn.deallocate(output)
        reset()
        output = prefill(args.length - 1)
        ttnn.synchronize_device(mesh)
        report["nonaligned_prefill"] = metric(expected[:, :-1], ttnn.to_torch(output))
        ttnn.deallocate(output)
        ttnn.copy(conv, prefix_conv)
        ttnn.copy(recurrent, prefix_recurrent)
        warm = module(decode, conv_state=conv, recurrent_state=recurrent)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warm)
        ttnn.copy(prefix_conv, conv)
        ttnn.copy(prefix_recurrent, recurrent)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with runtime_audit():
            output = module(decode, conv_state=conv, recurrent_state=recurrent)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        trace_allocation_tracker.acknowledge_corruptible(output)
        repeated = []
        for _ in range(2):
            ttnn.copy(prefix_conv, conv)
            ttnn.copy(prefix_recurrent, recurrent)
            with runtime_audit():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            repeated.append(ttnn.to_torch(output))
        report["traced_decode"] = metric(final, repeated[0])
        report["deterministic"] = torch.equal(*repeated)
        report["passed"] = (
            all(report[k]["passed"] for k in ("prefill", "nonaligned_prefill", "traced_decode"))
            and report["deterministic"]
        )
    except Exception as exc:
        report["error"] = str(exc).split("backtrace:")[0]
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
    print(json.dumps(report), flush=True)
    assert report["passed"]


if __name__ == "__main__":
    main()
