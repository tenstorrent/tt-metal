# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident-table layer1 state-path context test; each outer chunk is a pass.

Consumes probe_ple_context's exact-BF16 CPU fixture. Only PLE, attention hyper-
connection and GDN weights are loaded, plus the COMPLETE packed lookup table
(group20, or group10 with a loader-supported scale encoding). Each outer input
upload and output readback is an explicit setup/check boundary. Inside each
audited pass, raw device tokens are hashed and looked up,
PLE and both causal convolutions run, and GDN recurrent state advances without
host fallback. State continues from empty through length-1, then one stable
decode trace executes the final token. This proves the reduced state path's
logical context, NOT public single-call full-decoder prefill residency. MoE and
its hyper-connection are deliberately excluded and separately validated.

Every output row is compared. Acceptance gates use direct PLE contribution,
GDN output, and attention residual contribution, without residual dilution.
Other intermediate PCCs are diagnostics (four-wide injection weights can have
tiny variance); n-gram hash IDs must be exact. No threshold is relaxed for
quantization. The exact CPU fixture is independent of packed-table precision.
"""

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextPLELayer

from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_context import load_reference

GATES = ("ple_output", "attention", "after_attention_contribution")
STATE_NAMES = ("conv", "recurrent", "ple_conv", "token_history")


def token_words(ids):
    words = torch.zeros((*ids.shape, 16), dtype=torch.int32)
    words[..., 0] = ids.to(torch.int32)
    words[..., 1] = 1
    return words


def accumulate_metric(target, expected, actual, start):
    actual = actual.reshape(expected.shape)
    if expected.dtype == torch.int64:
        # Device hash outputs are UINT32; PyTorch does not implicitly promote
        # UInt32 with Long. Both contain exact nonnegative IDs below 2**32.
        actual = actual.to(torch.int64)
        target["rows_compared"] = target.get("rows_compared", 0) + expected.shape[1]
        target["mismatched_elements"] = target.get("mismatched_elements", 0) + int((expected != actual).sum())
        target["exact"] = target["mismatched_elements"] == 0
        return
    expected, actual = expected.float().reshape(-1, expected.shape[-1]), actual.float().reshape(-1, expected.shape[-1])
    left, right = expected - expected.mean(-1, keepdim=True), actual - actual.mean(-1, keepdim=True)
    scores = (left * right).sum(-1) / (left.norm(dim=-1) * right.norm(dim=-1))
    scores = torch.where((expected == actual).all(-1), 1.0, scores)
    finite = torch.isfinite(scores)
    safe = torch.where(finite, scores, -2.0)
    worst = int(safe.argmin())
    minimum = safe[worst].item()
    if minimum < target.get("minimum_row_pcc", 2):
        target.update(minimum_row_pcc=minimum, worst_absolute_token=start + worst)
    count = scores.numel()
    target["rows_compared"] = target.get("rows_compared", 0) + count
    target["pcc_sum"] = target.get("pcc_sum", 0.0) + safe.double().sum().item()
    target["mean_row_pcc"] = target["pcc_sum"] / target["rows_compared"]
    target["rows_below_threshold"] = target.get("rows_below_threshold", 0) + int((safe < 0.995).sum())
    target["invalid_pcc_rows"] = target.get("invalid_pcc_rows", 0) + int((~finite).sum())
    target["nonfinite_elements"] = target.get("nonfinite_elements", 0) + int((~torch.isfinite(actual)).sum())
    target["max_abs_error"] = max(target.get("max_abs_error", 0), (expected - actual).abs().max().item())
    target["passed"] = target["rows_below_threshold"] == target["invalid_pcc_rows"] == target["nonfinite_elements"] == 0


def compare_outputs(target, fixture, actual, start, end):
    expected = fixture["outputs"]
    for name, value in actual.items():
        accumulate_metric(target.setdefault(name, {}), expected[name][:, start:end], value, start)
    expected_contribution = (
        expected["after_attention"][:, start:end].float() - expected["after_ple"][:, start:end].float()
    )
    actual_contribution = actual["after_attention"].float() - actual["after_ple"].float()
    accumulate_metric(
        target.setdefault("after_attention_contribution", {}), expected_contribution, actual_contribution, start
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--fixture-cache", type=Path, required=True)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("/home/ttuser/qwen38-flash/packed_ple_lloyd_lsq_group20")
    )
    parser.add_argument("--length", type=int, help="Optional prefix for initial hardware validation")
    parser.add_argument("--outer-chunk", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    fixture = torch.load(args.fixture_cache, weights_only=True, mmap=True)
    identity = fixture["identity"]
    assert identity["format"] == "ple_gdn_reduced_context_v1" and identity["layer"] == 1
    assert identity["snapshot"] == str(args.snapshot.resolve())
    assert identity["config_sha256"] == hashlib.sha256((args.snapshot / "config.json").read_bytes()).hexdigest()
    assert (
        identity["checkpoint_index_sha256"]
        == hashlib.sha256((args.snapshot / "model.safetensors.index.json").read_bytes()).hexdigest()
    )
    assert (
        identity["hf_source_sha256"]
        == hashlib.sha256(Path(inspect.getfile(Qwen4ExpTextPLELayer)).read_bytes()).hexdigest()
    )
    assert (
        identity["fixture_source_sha256"]
        == hashlib.sha256(Path(inspect.getfile(load_reference)).read_bytes()).hexdigest()
    )
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    length = args.length or identity["length"]
    assert 2 <= length <= identity["length"] <= config.max_position_embeddings
    assert args.outer_chunk >= 32 and args.outer_chunk % 32 == 0
    assert fixture["input"].shape == (1, identity["length"], 10240)
    assert fixture["token_ids"].shape == (1, identity["length"])
    manifest = json.loads((args.cache_dir / "manifest.json").read_text())
    assert manifest["rows"] == 320001536 and manifest["group_size"] in (20, 10)
    assert manifest.get("completed_rows", manifest["rows"]) == manifest["rows"]
    assert (
        manifest["revision"] == args.snapshot.name and manifest["index_sha256"] == identity["checkpoint_index_sha256"]
    )
    assert (args.cache_dir / "words.bin").stat().st_size == manifest["word_bytes"]
    assert (args.cache_dir / "scales.bin").stat().st_size == manifest["scale_bytes"]
    # Construct only these CPU modules to retrieve their real state dict; no
    # embedding rows are queried and no MoE parameters are loaded or allocated.
    reference, _ = load_reference(args.snapshot, config)
    weights = reference.state_dict()
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
    from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import (
        GatedDeltaNet,
        GatedResidual,
        WeightLoader,
    )
    from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
    from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import PackedEmbedding
    from models.autoports.qwen_qwen3_8_flash_next.tt.ple import PLE

    if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        raise ValueError("TT_METAL_TRACE_ALLOC_TRACKING=1 is required for this context contract")
    report = {
        "scope": __doc__,
        "device_executed": True,
        "trace_allocation_tracking": True,
        "fixture_identity": identity,
        "fixture_cache": str(args.fixture_cache),
        "packed_cache": str(args.cache_dir),
        "packed_manifest": manifest,
        "packed_manifest_sha256": hashlib.sha256((args.cache_dir / "manifest.json").read_bytes()).hexdigest(),
        "threshold": 0.995,
        "gated_outputs": list(GATES),
        "context_length": length,
        "prefill_prefix_length": length - 1,
        "decode_position": length - 1,
        "advertised_context": config.max_position_embeddings,
        "outer_chunk": args.outer_chunk,
        "public_single_call_prefill_claim": False,
        "runtime_audit_completed_passes": 0,
        "prefill": {},
        "decode": {},
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None

    class ReducedDevicePath:
        def __init__(self, packed):
            loader = WeightLoader(weights, mesh)
            self.mesh_device, self.kind = mesh, "linear_attention"
            self.ple = PLE(loader, config)
            self.mix = GatedResidual(loader, "attn_hyper_connection.", config)
            self.attention = GatedDeltaNet(loader, config)
            self.embedding = packed

        def chunk(self, hidden, ids, state):
            count = hidden.shape[-2]
            operation = state.hash_operations[count]
            ttnn.copy(ids, operation.tensors[0])
            operation.forward()
            hashed = ttnn.reshape(operation.tensors[-1], (1, 1, count, 16))
            embeddings = self.embedding(hashed)
            ple = self.ple(hidden, embeddings, conv_state=state.ple_conv)
            after_ple = ttnn.add(hidden, ple)
            mixed, injection = self.mix.read(after_ple)
            attention = self.attention(mixed, conv_state=state.conv, recurrent_state=state.recurrent)
            after_attention = self.mix.write(after_ple, attention, injection)
            return {
                # The hash bucket is overwritten on the next chunk: outputs
                # must own their storage rather than alias that request buffer.
                "ngram_ids": ttnn.clone(hashed),
                "embeddings": embeddings,
                "ple_output": ple,
                "after_ple": after_ple,
                "attn_mixed": mixed,
                "attn_injection": ttnn.permute(injection, (0, 3, 2, 1)),
                "attention": attention,
                "after_attention": after_attention,
            }

        def forward(self, hidden, ids, state):
            results = [
                self.chunk(
                    hidden[:, :, start : min(start + 32, hidden.shape[-2])],
                    ids[:, start : min(start + 32, hidden.shape[-2])],
                    state,
                )
                for start in range(0, hidden.shape[-2], 32)
            ]
            if len(results) == 1:
                return results[0]
            return {name: ttnn.concat([part[name] for part in results], dim=2) for name in results[0]}

    def upload(hidden, ids):
        return (
            ttnn.from_torch(
                hidden.unsqueeze(0).contiguous(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh
            ),
            ttnn.from_torch(token_words(ids), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh),
        )

    def download(outputs):
        return {
            name: ttnn.to_torch(value).reshape(1, value.shape[-2], value.shape[-1]) for name, value in outputs.items()
        }

    def free_outputs(outputs):
        addresses = set()
        for value in outputs.values():
            address = value.buffer_address()
            if address not in addresses:
                addresses.add(address)
                ttnn.deallocate(value)

    try:
        packed = PackedEmbedding.from_checkpoint(args.snapshot, cache_dir=args.cache_dir, mesh_device=mesh)
        model = ReducedDevicePath(packed)
        buckets = tuple(sorted({1, 32, (length - 1) % 32 or 32}))
        # Reuse the public decoder's exact setup/state schema without loading
        # its token-local MoE modules.
        state = FunctionalDecoder.create_state(model, hash_lengths=buckets)
        started = time.monotonic()
        for ordinal, start in enumerate(range(0, length - 1, args.outer_chunk)):
            end = min(start + args.outer_chunk, length - 1)
            hidden, ids = upload(fixture["input"][:, start:end], fixture["token_ids"][:, start:end])
            with runtime_audit():
                outputs = model.forward(hidden, ids, state)
            report["runtime_audit_completed_passes"] += 1
            actual = download(outputs)
            compare_outputs(report["prefill"], fixture, actual, start, end)
            free_outputs(outputs)
            ttnn.deallocate(hidden)
            ttnn.deallocate(ids)
            del actual, outputs
            if (ordinal + 1) % 16 == 0 or end == length - 1:
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "state_context_rows": end,
                            "target_rows": length,
                            "elapsed_seconds": time.monotonic() - started,
                            "minimum_ple_pcc": report["prefill"]["ple_output"]["minimum_row_pcc"],
                            "minimum_gdn_pcc": report["prefill"]["attention"]["minimum_row_pcc"],
                        }
                    ),
                    flush=True,
                )
        prefix = {name: ttnn.clone(getattr(state, name)) for name in STATE_NAMES}

        def restore():
            for name, saved in prefix.items():
                ttnn.copy(saved, getattr(state, name))

        hidden, ids = upload(fixture["input"][:, length - 1 : length], fixture["token_ids"][:, length - 1 : length])
        with runtime_audit():
            warmed = model.forward(hidden, ids, state)
        report["runtime_audit_completed_passes"] += 1
        ttnn.synchronize_device(mesh)
        free_outputs(warmed)
        restore()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with runtime_audit():
            outputs = model.forward(hidden, ids, state)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        report["runtime_audit_completed_passes"] += 1
        for output in outputs.values():
            trace_allocation_tracker.acknowledge_corruptible(output)
        repeated, repeated_states = [], []
        for repeat in range(2):
            restore()
            with runtime_audit():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            report["runtime_audit_completed_passes"] += 1
            actual = download(outputs)
            repeated.append(actual)
            repeated_states.append({name: ttnn.to_torch(getattr(state, name)).clone() for name in STATE_NAMES})
            if repeat == 0:
                compare_outputs(report["decode"], fixture, actual, length - 1, length)
        report["repeated_output_exact"] = all(torch.equal(repeated[0][name], repeated[1][name]) for name in outputs)
        report["repeated_state_exact"] = all(
            torch.equal(repeated_states[0][name], repeated_states[1][name]) for name in STATE_NAMES
        )
        if length == identity["length"]:
            report["final_token_history_exact"] = torch.equal(
                repeated_states[0]["token_history"][:, :2].to(torch.int64), fixture["final_state"]["token_history"]
            )
        report["passed"] = (
            all(report[phase][name]["passed"] for phase in ("prefill", "decode") for name in GATES)
            and all(report[phase]["ngram_ids"]["exact"] for phase in ("prefill", "decode"))
            and report["repeated_output_exact"]
            and report["repeated_state_exact"]
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
    assert report["passed"], "Reduced PLE state-context acceptance failed; threshold remains0.995"


if __name__ == "__main__":
    main()
