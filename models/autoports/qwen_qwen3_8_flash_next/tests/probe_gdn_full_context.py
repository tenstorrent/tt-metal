# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Layer0 full-decoder final-token bridge after real streamed GDN context.

The original fixture is a real-weight, 2560-wide attention-input stream. It is
not a stream of full decoder inputs. The entire layer0 decoder stays resident
while its actual attention module advances fresh device state through all
length-1 prefix rows. Every prefix attention output and the original final
attention row are checked. Two independent 10240-wide final inputs then run
through one public full-decoder decode trace, each from the same real device
prefix, with repeated reset, changed-input, and runtime-fallback checks.

HF recomputes that prefix from the original input fixture. Only its final-token
reference cache is saved; no HF state is ever injected into TT. This establishes
full public decode after reduced-prefix context, not full-decoder prefix output
or a full-decoder single-call prefill at that length. --cpu-only never imports
TTNN and never edits the original GDN fixture.
"""

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_context_device import accumulate_metric


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hf_cache(config, prefix):
    cache = DynamicCache(config=config)
    cache.update_conv_state(prefix["conv"].clone(), 0, conv_kernel_size=prefix["conv"].shape[-1])
    cache.update_recurrent_state(prefix["recurrent"].clone(), 0)
    return cache


@torch.inference_mode()
def prepare_reference(reference, config, fixture, identity):
    length, chunk = identity["length"], identity["hf_chunk"]
    cache = DynamicCache(config=config)
    control = {}
    started = time.monotonic()
    for ordinal, start in enumerate(range(0, length - 1, chunk)):
        end = min(start + chunk, length - 1)
        actual = reference.linear_attn(fixture["input"][:, start:end], cache_params=cache)
        accumulate_metric(control, fixture["expected"][:, start:end], actual, start)
        if (ordinal + 1) % 16 == 0 or end == length - 1:
            print(json.dumps({"hf_prefix_rows": end, "seconds": time.monotonic() - started}), flush=True)
    assert control["passed"], control
    prefix = {"conv": cache.get_conv_state(0).clone(), "recurrent": cache.get_recurrent_state(0).clone()}
    original_final = reference.linear_attn(
        fixture["input"][:, length - 1 : length], cache_params=hf_cache(config, prefix)
    )
    original_control = {}
    accumulate_metric(original_control, fixture["expected"][:, length - 1 : length], original_final, length - 1)
    assert original_control["passed"], original_control
    generator = torch.Generator().manual_seed(identity["final_input_seed"])
    inputs = [(torch.randn((1, 1, 10240), generator=generator) * 0.02).bfloat16() for _ in range(2)]
    expected = [
        reference(hidden, position_embeddings=(None, None), past_key_values=hf_cache(config, prefix))
        for hidden in inputs
    ]
    assert all(torch.isfinite(value).all() for value in expected)
    return {
        "identity": identity,
        "prefix_state_hf_only": prefix,
        "final_inputs": inputs,
        "expected_full_decode": expected,
        "expected_original_attention": original_final,
        "hf_prefix_control": control,
        "hf_original_final_control": original_control,
        "preparation_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fixture-cache", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--length", type=int, help="Optional short smoke prefix; default is full original fixture")
    parser.add_argument("--device-chunk", type=int, default=1024)
    parser.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b"), default="bfloat16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cpu_only and args.reference_cache is None:
        parser.error("--cpu-only requires --reference-cache to persist the prepared full-token references")
    torch.set_num_threads(8)
    fixture = torch.load(args.fixture_cache, weights_only=True, mmap=True)
    source_identity = fixture["identity"]
    assert source_identity["format"] == "gdn_context_reference_v1" and source_identity["layer"] == 0
    assert source_identity["snapshot"] == str(args.snapshot.resolve())
    assert source_identity["config_sha256"] == sha256(args.snapshot / "config.json")
    assert source_identity["hf_source_sha256"] == sha256(Path(inspect.getfile(Qwen4ExpTextDecoderLayer)))
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    length = args.length or source_identity["length"]
    assert 2 <= length <= source_identity["length"] <= config.max_position_embeddings
    assert args.device_chunk >= 1
    assert fixture["input"].shape == fixture["expected"].shape == (1, source_identity["length"], 2560)
    identity = {
        "format": "gdn_full_final_context_v1",
        "source_fixture_identity": source_identity,
        "source_fixture": str(args.fixture_cache.resolve()),
        "source_fixture_bytes": args.fixture_cache.stat().st_size,
        "source_sha256": sha256(Path(__file__)),
        "checkpoint_index_sha256": sha256(args.snapshot / "model.safetensors.index.json"),
        "length": length,
        "hf_chunk": source_identity["chunk"],
        "final_input_seed": 21031,
        "final_input_count": 2,
        "hf_expert_dtype": "exact bfloat16",
    }
    weights = load_weights(args.snapshot, 0, "")
    with torch.device("meta"):
        reference = Qwen4ExpTextDecoderLayer(config, 0)
    reference.load_state_dict(weights, assign=True)
    reference.eval()
    if args.reference_cache and args.reference_cache.exists():
        prepared = torch.load(args.reference_cache, weights_only=True)
        assert prepared["identity"] == identity, "Reference cache identity changed; use a fresh path"
    else:
        prepared = prepare_reference(reference, config, fixture, identity)
        if args.reference_cache:
            assert args.reference_cache.resolve() != args.fixture_cache.resolve(), "Original fixture must be preserved"
            args.reference_cache.parent.mkdir(parents=True, exist_ok=True)
            partial = args.reference_cache.with_name(args.reference_cache.name + f".tmp-{os.getpid()}")
            try:
                torch.save(prepared, partial)
                partial.replace(args.reference_cache)
            finally:
                partial.unlink(missing_ok=True)
    report = {
        "scope": __doc__,
        "passed": False,
        "device_executed": False,
        "reference_identity": identity,
        "reference_cache": str(args.reference_cache),
        "hf_prefix_control": prepared["hf_prefix_control"],
        "hf_original_final_control": prepared["hf_original_final_control"],
        "cpu_preparation_seconds": prepared["preparation_seconds"],
        "layer": 0,
        "advertised_context": config.max_position_embeddings,
        "context_length": length,
        "nonaligned_prefix": length - 1,
        "decode_position": length - 1,
        "full_decoder_weights_resident_during_prefix": True,
        "full_decoder_prefill_claim": False,
        "expert_dtype": args.expert_dtype,
        "threshold": 0.995,
        "device_chunk": args.device_chunk,
        "runtime_audit_completed_passes": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cpu_only:
        assert "ttnn" not in sys.modules, "CPU-only preparation imported TTNN"
        report["cpu_reference_passed"] = True
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print("GDN_FULL_FINAL_CPU_REFERENCE_READY", flush=True)
        return

    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import metrics
    from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder

    if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        raise ValueError("TT_METAL_TRACE_ALLOC_TRACKING=1 is required")
    report.update(device_executed=True, trace_allocation_tracking=True, prefix={}, decode=[])
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None

    def device(value):
        return ttnn.from_torch(
            value.unsqueeze(0).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
        )

    try:
        model = FunctionalDecoder.from_state_dict(
            weights, hf_config=config, layer_idx=0, mesh_device=mesh, expert_dtype=getattr(ttnn, args.expert_dtype)
        )
        state = model.create_state()
        started = time.monotonic()
        for ordinal, start in enumerate(range(0, length - 1, args.device_chunk)):
            end = min(start + args.device_chunk, length - 1)
            hidden = device(fixture["input"][:, start:end])
            with runtime_audit():
                output = model.attn(hidden, conv_state=state.conv, recurrent_state=state.recurrent)
            report["runtime_audit_completed_passes"] += 1
            actual = ttnn.to_torch(output).reshape(1, end - start, 2560)
            accumulate_metric(report["prefix"], fixture["expected"][:, start:end], actual, start)
            ttnn.deallocate(output)
            ttnn.deallocate(hidden)
            if (ordinal + 1) % 16 == 0 or end == length - 1:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "device_prefix_rows": end,
                            "seconds": time.monotonic() - started,
                            "minimum_row_pcc": report["prefix"]["minimum_row_pcc"],
                        }
                    ),
                    flush=True,
                )
        report["prefix_wall_seconds"] = time.monotonic() - started
        prefix = {name: ttnn.clone(getattr(state, name)) for name in ("conv", "recurrent")}

        def restore():
            for name, saved in prefix.items():
                ttnn.copy(saved, getattr(state, name))

        original_hidden = device(fixture["input"][:, length - 1 : length])
        restore()
        with runtime_audit():
            original_output = model.attn(original_hidden, conv_state=state.conv, recurrent_state=state.recurrent)
        report["runtime_audit_completed_passes"] += 1
        report["original_final_attention"] = {}
        accumulate_metric(
            report["original_final_attention"],
            fixture["expected"][:, length - 1 : length],
            ttnn.to_torch(original_output).reshape(1, 1, 2560),
            length - 1,
        )
        ttnn.deallocate(original_hidden)
        ttnn.deallocate(original_output)
        restore()
        inputs = prepared["final_inputs"]
        hosts = [
            ttnn.from_torch(value.unsqueeze(0).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            for value in inputs
        ]
        hidden = device(inputs[0])
        for host in hosts:
            restore()
            ttnn.copy_host_to_device_tensor(host, hidden)
            with runtime_audit():
                warmed = model.decode_forward(hidden, state=state)
            report["runtime_audit_completed_passes"] += 1
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(warmed)
        restore()
        ttnn.copy_host_to_device_tensor(hosts[0], hidden)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with runtime_audit():
            output = model.decode_forward(hidden, state=state)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        report["runtime_audit_completed_passes"] += 1
        trace_allocation_tracker.acknowledge_corruptible(output)
        results, states = {}, {}
        for ordinal, index in enumerate((0, 1, 0, 1)):
            restore()
            ttnn.copy_host_to_device_tensor(hosts[index], hidden)
            with runtime_audit():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            report["runtime_audit_completed_passes"] += 1
            actual = ttnn.to_torch(output).reshape(1, 1, 10240).clone()
            current_state = {name: ttnn.to_torch(getattr(state, name)).clone() for name in prefix}
            entry = {
                "input_index": index,
                "replay": ordinal,
                **metrics(prepared["expected_full_decode"][index], actual, inputs[index]),
            }
            if index in results:
                entry["reset_output_exact"] = torch.equal(results[index], actual)
                entry["reset_state_exact"] = {
                    name: torch.equal(states[index][name], current_state[name]) for name in prefix
                }
            else:
                results[index], states[index] = actual, current_state
            report["decode"].append(entry)
        report["changed_inputs_changed_outputs"] = not torch.equal(results[0], results[1])
        report["all_reduced_rows_compared"] = (
            report["prefix"]["rows_compared"] + report["original_final_attention"]["rows_compared"] == length
        )
        report["trace"] = {
            "captures": 1,
            "replays": 4,
            "acknowledged_outputs": 1,
            "public_method": "FunctionalDecoder.decode_forward",
            "restored_device_prefix_before_each_replay": True,
        }
        report["passed"] = (
            report["prefix"]["passed"]
            and report["original_final_attention"]["passed"]
            and report["all_reduced_rows_compared"]
            and report["changed_inputs_changed_outputs"]
            and all(
                row["passed"] and row.get("reset_output_exact", True) and all(row.get("reset_state_exact", {}).values())
                for row in report["decode"]
            )
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
    assert report["passed"], "GDN full-final context bridge failed; threshold remains0.995"


if __name__ == "__main__":
    main()
