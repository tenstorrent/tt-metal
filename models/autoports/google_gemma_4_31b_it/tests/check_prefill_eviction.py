# SPDX-License-Identifier: Apache-2.0
"""Device regression for strict continuation after sliding-window page eviction.

Run as a module. Default scope is real layers 0/5; --full selects all layers.
This is correctness evidence, never a serving or full-model performance result.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

import ttnn

from ..tt.generator import Gemma4Generator

START = 1055
CONTEXT = 4096


def prefix_digest(gen, maps):
    """Hash real cached prefix rows on every chip without device-side slices."""
    digest = hashlib.sha256()
    for cache, table in zip(gen.kv_cache, maps):
        ids = table[0, : (START + 31) // 32].long()
        for tensor in cache:
            for chip in ttnn.get_device_tensors(tensor):
                host = ttnn.to_torch(chip)
                # Cache layout is [physical page, heads, token, head dim].
                rows = host[ids].permute(0, 2, 1, 3).reshape(-1, host.shape[1], host.shape[3])[:START]
                digest.update(rows.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def metrics(reference, candidate):
    a, b = reference.reshape(-1).double(), candidate.reshape(-1).double()
    centered_a, centered_b = a - a.mean(), b - b.mean()
    denom = centered_a.norm() * centered_b.norm()
    return {
        "exact_equal": torch.equal(reference, candidate),
        "max_abs_difference": float((a - b).abs().max()),
        "pcc": float(torch.dot(centered_a, centered_b) / denom) if denom else None,
        "top1_equal": int(a.argmax()) == int(b.argmax()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/autoports/google_gemma_4_31b_it/readiness_vllm/prefill_eviction.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    gen = None
    try:
        gen = Gemma4Generator(mesh, layer_indices=None if args.full else (0, 5), batch_size=1, max_seq_len=CONTEXT)
        gen.prepare_decode()
        traces = (gen.model_trace, gen.sample_trace, gen.buffered_sample_trace)
        programs_before = mesh.num_program_cache_entries()
        random = torch.Generator().manual_seed(51055)
        maps = []
        scratch_ids = []
        for layer, cache, original in zip(gen.model.layers, gen.kv_cache, gen.prefill.host_pages):
            permutation = torch.randperm(cache[0].shape[0], generator=random).to(torch.int32)
            table = permutation[original.long()]
            maps.append(table)
            # Use the physical ID originally assigned to logical page 63,
            # not literal physical ID 63 after permutation.
            scratch_ids.append(int(table[0, 63]) if layer.sliding_window else None)
        gen.owns_cache = False  # This test's maps play the external scheduler role.
        text = (
            "The engineer recorded each observation in a notebook. The river curved past a stone bridge, "
            "where travelers compared maps and discussed how rainfall changed the current. "
        )
        ids = gen.tokenizer.encode(text * 128, add_special_tokens=True)
        assert len(ids) > START
        tokens = torch.tensor(ids[: START + 1], dtype=torch.int32)
        results = {}
        logits = {}
        capture_count = gen.counters["captures"]

        for name, removed_page in (("retained", None), ("evicted_page0", 0), ("missing_required_page1", 1)):
            gen.reset()
            gen.refresh_page_tables(maps, revision=f"{name}:prefix")
            for _, _, hidden in gen._prefill_hidden_chunks(tokens, slot=0, start=0, end=START):
                del hidden
            before_digest = prefix_digest(gen, maps)
            candidate_maps = [table.clone() for table in maps]
            for layer, table, scratch in zip(gen.model.layers, candidate_maps, scratch_ids):
                if layer.sliding_window and removed_page is not None:
                    table[0, removed_page] = scratch
            # All full-attention maps stay byte-for-byte equal in every arm.
            assert all(
                layer.sliding_window or torch.equal(old, new)
                for layer, old, new in zip(gen.model.layers, maps, candidate_maps)
            )
            gen.refresh_page_tables(candidate_maps, revision=f"{name}:continuation")
            before_step = mesh.num_program_cache_entries()
            output = None
            for position, valid, hidden in gen._prefill_hidden_chunks(tokens, slot=0, start=START, end=START + 1):
                assert position == START and valid == 1
                output = gen.model.terminal(hidden)
                logits[name] = gen._host_logits(output).reshape(1, -1)
                del hidden, output
            after_digest = prefix_digest(gen, maps)
            results[name] = {
                "prefix_rows_preserved": before_digest == after_digest,
                "prefix_digest_before": before_digest,
                "prefix_digest_after": after_digest,
                "continuation_program_growth": mesh.num_program_cache_entries() - before_step,
                "removed_sliding_logical_page": removed_page,
            }

        eviction = metrics(logits["retained"], logits["evicted_page0"])
        negative = metrics(logits["retained"], logits["missing_required_page1"])
        programs_after = mesh.num_program_cache_entries()
        negative_sensitive = not negative["exact_equal"]
        unchanged_traces = traces == (gen.model_trace, gen.sample_trace, gen.buffered_sample_trace)
        passed = (
            eviction["exact_equal"]
            and negative_sensitive
            and all(arm["prefix_rows_preserved"] for arm in results.values())
            and programs_after == programs_before
            and unchanged_traces
            and gen.counters["captures"] == capture_count
        )
        report = {
            "passed": passed,
            "scope": "full_model" if args.full else "reduced_layers_0_5",
            "layer_indices": list(gen.model.layer_indices),
            "context": CONTEXT,
            "start": START,
            "end": START + 1,
            "sliding_window": 1024,
            "scheduler_skipped_pages": (START - 1024 + 1) // 32,
            "mapping": "deterministic noncontiguous physical permutation; scratch = original map[63]",
            "scratch_physical_ids": scratch_ids,
            "arms": results,
            "eviction_equivalence": eviction,
            "required_history_control": negative,
            "control_status": "sensitive" if negative_sensitive else "weak_control_no_logit_change",
            "program_cache_before": programs_before,
            "program_cache_after": programs_after,
            "trace_ids_unchanged": unchanged_traces,
            "capture_count_before": capture_count,
            "capture_count_after": gen.counters["captures"],
            "limitations": [
                "One-token continuation tests eviction addressing, not the packaged long-resume accuracy gates.",
                "Negative control measures logit sensitivity; it does not require a changed top-1 token.",
                "No serving latency or performance claim.",
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": passed, "output": str(args.output), "control": report["control_status"]}))
        if not passed:
            raise AssertionError("Eviction regression failed or required-history control was weak; inspect JSON")
    finally:
        if gen is not None:
            gen.release_persistent_capture()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
