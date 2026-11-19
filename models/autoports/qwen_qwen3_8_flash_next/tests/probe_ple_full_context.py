# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full resident layer1 final-decode bridge after a real device prefix.

The entire real decoder (including explicitly selected expert precision) and
complete packed embedding table remain resident. Prefix tokens advance its
actual PLE/hash/GDN modules from empty state, without running the token-local
MoE. Every prefix PLE/GDN row is checked against the exact HF fixture. The final
token executes FunctionalDecoder.decode_forward in one trace, including MoE,
and is compared to a complete HF decoder initialized from the fixture's exact
HF prefix state. No HF prefix state or prepared embeddings are uploaded to TT.

This establishes full-layer decode after streamed recurrent context; it does
not claim full-layer output for all prefix rows or single-call public prefill
residency at the advertised length. Public prefill capacity is a separate test.
"""

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_context import load_reference as reduced_reference
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_context_device import (
    GATES,
    STATE_NAMES,
    compare_outputs,
    token_words,
)


def restore_hf_prefix(config, saved):
    """HF-only initialization; this cache never crosses to the device."""
    cache = DynamicCache(config=config)
    for state_idx, name in enumerate(("gdn_convolution", "ple_convolution", "token_history")):
        value = saved[name].clone()
        cache.update_conv_state(value, 1, state_idx=state_idx, conv_kernel_size=value.shape[-1])
    cache.update_recurrent_state(saved["gdn_recurrent"].clone(), 1, state_idx=0)
    return cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fixture-cache", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b"), required=True)
    parser.add_argument("--length", type=int, help="Short smoke prefix; full fixture length by default")
    parser.add_argument("--outer-chunk", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    fixture = torch.load(args.fixture_cache, weights_only=True, mmap=True)
    identity = fixture["identity"]
    assert identity["format"] == "ple_gdn_reduced_context_v1" and identity["layer"] == 1
    assert identity["snapshot"] == str(args.snapshot.resolve())
    for field, path in (
        ("config_sha256", args.snapshot / "config.json"),
        ("checkpoint_index_sha256", args.snapshot / "model.safetensors.index.json"),
        ("hf_source_sha256", Path(inspect.getfile(Qwen4ExpTextPLELayer))),
        ("fixture_source_sha256", Path(inspect.getfile(reduced_reference))),
    ):
        assert identity[field] == hashlib.sha256(path.read_bytes()).hexdigest(), field
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    config._attn_implementation = "eager"
    length = args.length or identity["length"]
    assert 2 <= length <= identity["length"] <= config.max_position_embeddings
    assert args.outer_chunk >= 32 and args.outer_chunk % 32 == 0
    assert fixture["input"].shape == (1, identity["length"], 10240)
    assert fixture["token_ids"].shape == (1, identity["length"])
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    assert manifest["rows"] == 320001536
    assert manifest.get("completed_rows", manifest["rows"]) == manifest["rows"]
    assert manifest["revision"] == args.snapshot.name
    assert manifest["index_sha256"] == identity["checkpoint_index_sha256"]
    for name, field in (("words.bin", "word_bytes"), ("scales.bin", "scale_bytes")):
        assert (args.cache_dir / name).stat().st_size == manifest[field]

    # Device imports are delayed until fixture identity has been checked.
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import load_reference, metrics
    from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
    from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import PackedEmbedding

    if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        raise ValueError("TT_METAL_TRACE_ALLOC_TRACKING=1 is required")
    reference, weights = load_reference(args.snapshot, config)
    assert not reference.ple.ple_embedding.ngram_embedding.quantized
    with torch.inference_mode():
        if length == identity["length"]:
            hf_cache = restore_hf_prefix(config, fixture["prefix_state"])
            hf_prefix_source = "exact fixture prefix_state, HF only"
        else:
            # Short smoke has no saved intermediate recurrent state. Recompute
            # the real HF state path, still excluding only token-local MoE.
            hf_cache = DynamicCache(config=config)
            for start in range(0, length - 1, 1024):
                end = min(start + 1024, length - 1)
                hidden = fixture["input"][:, start:end]
                after_ple = hidden + reference.ple(hidden, fixture["token_ids"][:, start:end], hf_cache)
                mixed, _, _ = reference.attn_hyper_connection(after_ple)
                reference.linear_attn(mixed, cache_params=hf_cache)
            hf_prefix_source = "exact real HF reduced prefix recomputation for short smoke"
        observed = {}
        hooks = [
            reference.ple.register_forward_hook(lambda module, inputs, output: observed.update(ple_output=output)),
            reference.linear_attn.register_forward_hook(
                lambda module, inputs, output: observed.update(attention=output)
            ),
            reference.attn_hyper_connection.register_forward_pre_hook(
                lambda module, inputs: observed.update(after_ple=inputs[0])
            ),
            reference.mlp_hyper_connection.register_forward_pre_hook(
                lambda module, inputs: observed.update(after_attention=inputs[0])
            ),
            reference.ple.ple_embedding.ngram_embedding.register_forward_pre_hook(
                lambda module, inputs: observed.update(ngram_ids=inputs[0])
            ),
        ]
        final_hidden = fixture["input"][:, length - 1 : length]
        final_ids = fixture["token_ids"][:, length - 1 : length]
        expected = reference(
            final_hidden, position_embeddings=(None, None), past_key_values=hf_cache, ple_input_ids=final_ids
        )
        for hook in hooks:
            hook.remove()
    hf_bridge = {}
    compare_outputs(hf_bridge, fixture, observed, length - 1, length)
    assert all(hf_bridge[name]["passed"] for name in GATES) and hf_bridge["ngram_ids"]["exact"], hf_bridge
    print("FULL_PLE_CONTEXT_HF_FINAL_TOKEN_READY", flush=True)
    report = {
        "scope": __doc__,
        "passed": False,
        "device_executed": True,
        "snapshot": str(args.snapshot),
        "fixture_cache": str(args.fixture_cache),
        "fixture_identity": identity,
        "packed_cache": str(args.cache_dir),
        "packed_manifest": manifest,
        "packed_manifest_sha256": hashlib.sha256((args.cache_dir / "manifest.json").read_bytes()).hexdigest(),
        "expert_dtype": args.expert_dtype,
        "hf_expert_dtype": "exact bfloat16",
        "context_length": length,
        "prefix_rows": length - 1,
        "decode_position": length - 1,
        "advertised_context": config.max_position_embeddings,
        "outer_chunk": args.outer_chunk,
        "full_decoder_weights_resident_during_prefix": True,
        "public_single_call_prefill_claim": False,
        "prefix_execution": "actual full model PLE/hash/attention modules; token-local MoE omitted",
        "hf_prefix_source": hf_prefix_source,
        "hf_final_state_path_control": hf_bridge,
        "device_prefix_source": "fresh state advanced by every raw input/token; no HF state injection",
        "threshold": 0.995,
        "prefix_gated_outputs": list(GATES),
        "prefill": {},
        "runtime_audit_completed_passes": 0,
        "trace_allocation_tracking": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None

    def upload(hidden, ids):
        return (
            ttnn.from_torch(
                hidden.unsqueeze(0).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
            ),
            ttnn.from_torch(token_words(ids), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh),
        )

    def prefix_chunk(hidden, ids):
        count = hidden.shape[-2]
        operation = state.hash_operations[count]
        ttnn.copy(ids, operation.tensors[0])
        operation.forward()
        hashed = ttnn.reshape(operation.tensors[-1], (1, 1, count, 16))
        embeddings = model.embedding(hashed)
        ple = model.ple(hidden, embeddings, conv_state=state.ple_conv)
        after_ple = ttnn.add(hidden, ple)
        mixed, injection = model.attn_mix.read(after_ple)
        attention = model.attn(mixed, conv_state=state.conv, recurrent_state=state.recurrent)
        after_attention = model.attn_mix.write(after_ple, attention, injection)
        return {
            "ngram_ids": ttnn.clone(hashed),
            "ple_output": ple,
            "after_ple": after_ple,
            "attention": attention,
            "after_attention": after_attention,
        }

    def prefix_outer(hidden, ids):
        parts = [
            prefix_chunk(
                hidden[:, :, start : min(start + 32, hidden.shape[-2])],
                ids[:, start : min(start + 32, hidden.shape[-2])],
            )
            for start in range(0, hidden.shape[-2], 32)
        ]
        if len(parts) == 1:
            return parts[0]
        return {name: ttnn.concat([part[name] for part in parts], dim=2) for name in parts[0]}

    def download(outputs):
        return {
            name: ttnn.to_torch(value).reshape(1, value.shape[-2], value.shape[-1]) for name, value in outputs.items()
        }

    try:
        packed = PackedEmbedding.from_checkpoint(args.snapshot, cache_dir=args.cache_dir, mesh_device=mesh)
        model = FunctionalDecoder.from_state_dict(
            weights,
            hf_config=config,
            layer_idx=1,
            mesh_device=mesh,
            packed_embedding=packed,
            expert_dtype=getattr(ttnn, args.expert_dtype),
        )
        state = model.create_state(hash_lengths=tuple(sorted({1, 32, (length - 1) % 32 or 32})))
        started = time.monotonic()
        for ordinal, start in enumerate(range(0, length - 1, args.outer_chunk)):
            end = min(start + args.outer_chunk, length - 1)
            hidden, ids = upload(fixture["input"][:, start:end], fixture["token_ids"][:, start:end])
            with runtime_audit():
                outputs = prefix_outer(hidden, ids)
            report["runtime_audit_completed_passes"] += 1
            actual = download(outputs)
            compare_outputs(report["prefill"], fixture, actual, start, end)
            for value in outputs.values():
                ttnn.deallocate(value)
            ttnn.deallocate(hidden)
            ttnn.deallocate(ids)
            del outputs, actual
            if (ordinal + 1) % 16 == 0 or end == length - 1:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "prefix_rows": end,
                            "target": length,
                            "seconds": time.monotonic() - started,
                            "ple_min_pcc": report["prefill"]["ple_output"]["minimum_row_pcc"],
                            "gdn_min_pcc": report["prefill"]["attention"]["minimum_row_pcc"],
                        }
                    ),
                    flush=True,
                )
        report["streamed_prefix_wall_seconds"] = time.monotonic() - started
        # These are snapshots of the actual device-produced prefix, never HF.
        prefix = {name: ttnn.clone(getattr(state, name)) for name in STATE_NAMES}

        def restore():
            for name, saved in prefix.items():
                ttnn.copy(saved, getattr(state, name))

        hidden, ids = upload(final_hidden, final_ids)
        with runtime_audit():
            warmed = model.decode_forward(hidden, state=state, token_ids=ids)
        report["runtime_audit_completed_passes"] += 1
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warmed)
        restore()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with runtime_audit():
            output = model.decode_forward(hidden, state=state, token_ids=ids)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        report["runtime_audit_completed_passes"] += 1
        # Exactly the sole final output is corruptible; persistent request state,
        # raw inputs, hash buffer and prefix snapshots retain tracker protection.
        trace_allocation_tracker.acknowledge_corruptible(output)
        results, states, hashes = [], [], []
        for repeat in range(2):
            restore()
            with runtime_audit():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            report["runtime_audit_completed_passes"] += 1
            # Consume the output before the next replay can overwrite it.
            results.append(ttnn.to_torch(output).reshape(expected.shape).clone())
            states.append({name: ttnn.to_torch(getattr(state, name)).clone() for name in STATE_NAMES})
            hashes.append(ttnn.to_torch(state.hash_operations[1].tensors[-1]).reshape(1, 1, 16).to(torch.int64))
        report["decode"] = metrics(expected, results[0], final_hidden)
        report["repeat_decode"] = metrics(expected, results[1], final_hidden)
        report["decode_hashes_exact"] = all(
            torch.equal(value, fixture["outputs"]["ngram_ids"][:, length - 1 : length]) for value in hashes
        )
        report["repeated_output_exact"] = torch.equal(results[0], results[1])
        report["repeated_state_exact"] = {name: torch.equal(states[0][name], states[1][name]) for name in STATE_NAMES}
        if length == identity["length"]:
            report["final_token_history_exact"] = torch.equal(
                states[0]["token_history"][:, :2].to(torch.int64), fixture["final_state"]["token_history"]
            )
        report["trace"] = {
            "captures": 1,
            "replays": 2,
            "public_method": "FunctionalDecoder.decode_forward",
            "acknowledged_outputs": 1,
            "restored_device_prefix_before_every_replay": True,
        }
        report["passed"] = (
            all(report["prefill"][name]["passed"] for name in GATES)
            and report["prefill"]["ngram_ids"]["exact"]
            and report["decode"]["passed"]
            and report["repeat_decode"]["passed"]
            and report["decode_hashes_exact"]
            and report["repeated_output_exact"]
            and all(report["repeated_state_exact"].values())
            and report.get("final_token_history_exact", True)
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
    assert report["passed"], "Full resident PLE final-decode bridge failed; threshold remains0.995"


if __name__ == "__main__":
    main()
