"""Full-model configured-batch capability and end-to-end workload measurement."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import torch
from readiness_check import run_prefill_check, run_teacher_forcing

import ttnn

from ..tt.generator import Gemma4Generator
from .full_generation_evidence import performance


def memory(mesh):
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return {
        key: int(getattr(view, key))
        for key in (
            "num_banks",
            "total_bytes_per_bank",
            "total_bytes_allocated_per_bank",
            "largest_contiguous_bytes_free_per_bank",
        )
    }


def inventory(gen):
    """Per-device allocator footprint from physical pages, deduplicated by buffer ID."""
    seen = set()

    def size(value):
        if isinstance(value, ttnn.Tensor):
            part = ttnn.get_device_tensors(value)[0]
            uid = part.buffer_unique_id()
            if uid in seen or part.memory_config().buffer_type != ttnn.BufferType.DRAM:
                return 0
            seen.add(uid)
            banks = 8
            return ((part.buffer_num_pages() + banks - 1) // banks) * part.buffer_aligned_page_size() * banks
        if isinstance(value, dict):
            return sum(size(v) for v in value.values())
        if isinstance(value, (list, tuple)):
            return sum(size(v) for v in value)
        return 0

    groups = {
        "decoder_weights": size([layer.weights for layer in gen.model.layers]),
        "decoder_state": size([vars(layer) for layer in gen.model.layers]),
        "embedding": size(gen.model.embedding),
        "lm_head": size(gen.model.lm_head),
        "final_norm": size(gen.model.norm_weight),
        "rope": size(gen.model.rope_tables),
        "kv_cache": size(gen.kv_cache),
        "page_tables": size(gen.page_table),
        "sampler": size(vars(gen.sampler)),
        "prepared_prefill": size(vars(gen.prefill)),
        "generator_state": size(vars(gen)),
    }
    groups["sum_owned_dram_bytes"] = sum(groups.values())
    current = memory(gen.mesh)
    groups["allocator_bytes"] = current["num_banks"] * current["total_bytes_allocated_per_bank"]
    groups["unattributed_allocator_bytes"] = groups["allocator_bytes"] - groups["sum_owned_dram_bytes"]
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--long-context", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = Path("models/autoports/google_gemma_4_31b_it/doc/full_model")
    prompt = json.loads((root / "aime24_prompt.json").read_text())["tokens"]
    ttnn.CONFIG.throw_exception_on_fallback = True
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    report = dict(scope="full60layers", batch=args.batch, memory={"empty": memory(mesh)})
    try:
        gen = Gemma4Generator(mesh, batch_size=args.batch)
        report["memory"]["construction_stages"] = gen.model.allocation_bytes
        report["memory"]["constructed"] = memory(mesh)
        try:
            report["memory"]["inventory"] = inventory(gen)
        except (AttributeError, RuntimeError, TypeError) as exc:
            report["memory"]["inventory_error"] = str(exc)
        gen.prepare_decode()
        report["memory"]["prepared"] = memory(mesh)
        if args.batch == 1:
            reference = json.loads((root / "reference_metadata.json").read_text())
            accuracy = {}
            for phase, module, runner in (
                ("prefill", run_prefill_check, run_prefill_check.run_prefill_check),
                ("decode", run_teacher_forcing, run_teacher_forcing.run_teacher_forcing),
            ):
                with patch.object(module, "_import_build_generator", return_value=lambda **kwargs: gen):
                    accuracy[phase] = runner(
                        model_dir=root.parent.parent, reference_path=Path(reference["reference"]), mesh_device=mesh
                    )
                assert all(row["top5"] >= 0.98 and row["top100"] == 1 for row in accuracy[phase])
            report["accuracy"] = accuracy
            (root / "final_readiness_accuracy.json").write_text(json.dumps(accuracy, indent=2) + "\n")
        first = gen.generate_batch([prompt] * args.batch, 8)
        again = gen.generate_batch([prompt] * args.batch, 8)
        assert all(row == first[0] for row in first) and first == again
        if args.batch == 1:
            canonical = gen.generate(prompt, 8)
            compatibility = gen.generate(prompt, 8, host_sampling=True)
            assert canonical == compatibility
            report["host_sampling_compatibility_equal"] = True
        if args.batch > 1:
            holes = [None if slot % 3 == 0 else prompt for slot in range(args.batch)]
            hole_outputs = gen.generate_batch(holes, 4)
            assert all(row == ([] if holes[slot] is None else first[slot][:4]) for slot, row in enumerate(hole_outputs))
            report["inactive_holes_preserve_active_outputs"] = True
        mixed = [prompt[: 31 + slot] for slot in range(args.batch)]
        if args.batch > 1:
            mixed[-1] = None
        different = gen.generate_batch(mixed, 4)
        assert different == gen.generate_batch(mixed, 4)
        assert args.batch == 1 or different[-1] == []
        report.update(
            repeated_equal=True,
            equal_across_slots=True,
            mixed_repeat_equal=True,
            outputs=first,
            mixed_outputs=different,
            pool_tokens=gen.pool_tokens,
            max_individual_context=gen.context,
        )
        (root / f"batch_capability_b{args.batch}.json").write_text(json.dumps(report, indent=2) + "\n")
        performance(gen, root)
        if args.long_context:
            assert args.batch == 1
            report["long_context"] = []
            for length, generated in ((262143, 1), (262144, 0)):
                ids = (prompt * ((length + len(prompt) - 1) // len(prompt)))[:length]
                if generated:
                    out = gen.generate(ids, generated)
                    boundary = gen.decode_forward(
                        torch.tensor([[out[0]]]),
                        torch.tensor([length]),
                        page_table=gen.page_table,
                        kv_cache=gen.kv_cache,
                    )
                    report["last_input_position_decode"] = dict(
                        position=length,
                        output=boundary,
                        positions_after=ttnn.to_torch(ttnn.get_device_tensors(gen.positions)[0]).reshape(-1).tolist(),
                        scope="Low-level last legal input-position probe; not an extra public-generation capacity claim",
                    )
                else:
                    gen.reset()
                    gen._reserve_requests([length], 0)
                    out = gen.prefill_forward(
                        torch.tensor([ids]), page_table=gen.page_table, kv_cache=gen.kv_cache, prompt_lens=[length]
                    )
                report["long_context"].append(dict(prompt_tokens=length, generated_tokens=generated, output=out))
                print("FULL_CONTEXT", report["long_context"][-1], flush=True)
        report["memory"]["after_workloads"] = memory(mesh)
        report["counters"] = gen.counters
    finally:
        (root / f"batch_capability_b{args.batch}.json").write_text(json.dumps(report, indent=2) + "\n")
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
