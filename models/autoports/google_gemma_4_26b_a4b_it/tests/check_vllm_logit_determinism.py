# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare numeric standalone and adapter logits across runs and batch rows.

This diagnostic explicitly requests host logits. It does not benchmark or
replace the canonical device-sampling path. Device execution is serialized by
the caller with other model/serving work.
"""

import argparse
import json
import os
from pathlib import Path

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator_vllm import AutoportGemma4ForCausalLM

PROMPTS = {"A": [2] + [100] * 30, "B": [2] + [101] * 62}
CONTEXT = 256
WIRE_ROWS = 32


def compare_logits(reference, observed):
    """Report numerical drift without inventing a tolerance for this same model."""
    if reference.shape != observed.shape:
        return {"exact": False, "reference_shape": list(reference.shape), "observed_shape": list(observed.shape)}
    finite = bool(torch.isfinite(reference).all() and torch.isfinite(observed).all())
    exact = finite and torch.equal(reference, observed)
    if not finite:
        return {"exact": False, "finite": False}
    lhs, rhs = reference.double().flatten(), observed.double().flatten()
    difference = (lhs - rhs).abs()
    lhs_centered, rhs_centered = lhs - lhs.mean(), rhs - rhs.mean()
    denominator = lhs_centered.norm() * rhs_centered.norm()
    pcc = float(torch.dot(lhs_centered, rhs_centered) / denominator) if denominator > 0 else None
    return {
        "exact": exact,
        "finite": True,
        "elements": reference.numel(),
        "different_elements": int(torch.count_nonzero(reference != observed)),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "pcc": pcc,
        "top1_equal_by_step": (reference.argmax(-1) == observed.argmax(-1)).tolist(),
        "max_abs_diff_by_step": (reference - observed).abs().amax(-1).tolist(),
    }


def read_logits(generator, device_logits, rows):
    host = generator._read_logits(device_logits)
    return host.reshape(-1, generator.model.config.vocab_size)[:rows].clone()


def standalone_run(generator, prompt, decode_steps, forced_tokens=None):
    generator._release_trace()
    generator.reset()
    table = generator._standalone_cache(CONTEXT)
    logits = generator.prefill_forward(
        torch.tensor([prompt]), page_table=table, kv_cache=generator.cache, prompt_lens=[len(prompt)]
    )
    outputs = [read_logits(generator, logits, 1)[0]]
    del logits
    consumed = []
    try:
        for step in range(decode_steps):
            token = int(outputs[-1].argmax()) if forced_tokens is None else forced_tokens[step]
            consumed.append(token)
            logits = generator.decode_forward(
                torch.tensor([[token]], dtype=torch.int32),
                torch.tensor([len(prompt) + step], dtype=torch.int32),
                page_table=table,
                kv_cache=generator.cache,
                device_feedback=False,
                return_logits=True,
            )
            outputs.append(read_logits(generator, logits, 1)[0])
            del logits
    finally:
        generator._release_trace()
    return torch.stack(outputs), consumed


def serving_cache(adapter):
    """Use the existing hybrid fixture's shared pool and disjoint group pages."""
    model = adapter.generator.model
    specs = [None] * model.config.num_hidden_layers
    sliding = [i for i in model.layer_indices if model.config.layer_types[i] == "sliding_attention"]
    full = [i for i in model.layer_indices if model.config.layer_types[i] != "sliding_attention"]
    for index, layer in zip(model.layer_indices, model.layers):
        config = layer.layer.self_attn.config
        pool = sliding[0] if sliding and full and index == full[0] else index
        specs[index] = ((32, config.num_key_value_heads, 32, config.head_dim), torch.bfloat16, pool)
    return adapter.allocate_kv_cache_per_layer(specs)


def page_tables(generator, row_count):
    sliding = torch.tensor([list(range(row * 16, row * 16 + 8)) for row in range(row_count)], dtype=torch.int32)
    full = sliding + 8
    return [sliding if kind == "sliding_attention" else full for kind in generator.model.config.layer_types]


def adapter_run(adapter, cache, order, teacher_tokens, decode_steps):
    generator = adapter.generator
    generator._release_trace()
    # Aliased hybrid views share storage; clear each physical buffer only once.
    cleared = set()
    for pair in cache:
        for tensor in pair:
            address = tensor.buffer_address()
            if address not in cleared:
                ttnn.mul(tensor, 0.0, output_tensor=tensor)
                cleared.add(address)
    prompts = [PROMPTS[name] for name in order]
    lengths = [len(prompt) for prompt in prompts]
    ids = torch.zeros(len(order), max(lengths), dtype=torch.int32)
    for row, prompt in enumerate(prompts):
        ids[row, : len(prompt)] = torch.tensor(prompt)
    tables = page_tables(generator, len(order))
    prefill = adapter.prefill_forward(
        ids,
        tables[0],
        cache,
        lengths,
        sampling_params=None,
        page_tables_per_layer=tables,
        empty_slots=[5 + 2 * row for row in range(len(order))],
    )
    assert prefill.shape == (len(order), 1, generator.model.config.vocab_size)
    outputs = {name: [prefill[row, 0].clone()] for row, name in enumerate(order)}
    wire_tables = [torch.nn.functional.pad(table, (0, 0, 0, WIRE_ROWS - len(order))) for table in tables]
    trace_id = None
    try:
        for step in range(decode_steps):
            tokens = torch.zeros(WIRE_ROWS, 1, dtype=torch.int32)
            positions = torch.full((WIRE_ROWS,), -1, dtype=torch.int32)
            for row, name in enumerate(order):
                tokens[row, 0] = teacher_tokens[name][step]
                positions[row] = len(PROMPTS[name]) + step
            logits = adapter.decode_forward(
                tokens,
                positions,
                wire_tables[0],
                cache,
                sampling_params=None,
                page_tables_per_layer=wire_tables,
                reset_batch=step == 0,
            )
            assert logits.shape == (len(order), 1, generator.model.config.vocab_size)
            assert generator.batch == len(order)
            if trace_id is None:
                trace_id = generator.trace_id
            else:
                assert generator.trace_id == trace_id, "Unchanged decode layout unexpectedly recaptured"
            for row, name in enumerate(order):
                outputs[name].append(logits[row, 0].clone())
    finally:
        generator._release_trace()
    return {name: torch.stack(rows) for name, rows in outputs.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    layers = parser.add_mutually_exclusive_group()
    layers.add_argument("--layers", default="0,5", help="Comma-separated reduced layer indices (default: 0,5)")
    layers.add_argument("--full-model", action="store_true", help="Load all model layers")
    parser.add_argument("--decode-steps", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 2 or not 2 <= args.decode_steps <= CONTEXT - max(map(len, PROMPTS.values())):
        parser.error("Need at least two runs and two decode steps, within the 256-token fixture context")
    layer_indices = None if args.full_model else tuple(int(value) for value in args.layers.split(","))
    if layer_indices is not None and (not layer_indices or len(set(layer_indices)) != len(layer_indices)):
        parser.error("Reduced layer indices must be nonempty and unique")

    # This diagnostic needs full logits; leave the generator's host_sampling
    # switch disabled and use only the adapter's explicit compatibility API.
    os.environ["GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING"] = "1"
    torch.set_num_threads(8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = args.output.with_suffix(".pt")
    report = {
        "status": "incomplete",
        "scope": "direct adapter and standalone generator; not live vLLM HTTP or performance evidence",
        "requested_layers": "all" if args.full_model else list(layer_indices),
        "exact_equality_required": True,
        "tolerance": "No numerical tolerance: identical selected model and teacher-forced inputs",
        "decode_steps": args.decode_steps,
        "repeats": args.repeats,
        "wire_rows": WIRE_ROWS,
        "prompt_token_ids": PROMPTS,
        "batch_orders": [["A", "B"], ["B", "A"]],
        "step_labels": ["last_prefill"] + [f"decode_{step}" for step in range(args.decode_steps)],
        "logits_artifact": str(tensor_path),
        "ttnn_runtime": ttnn.__file__,
        "comparisons": {},
    }
    vectors, references, teachers, failures = {}, {}, {}, []
    generator = mesh = None

    def record(label, name, values, comparison_reference=None):
        key = f"{label}/{name}"
        vectors[key] = values
        metrics = compare_logits(references[name] if comparison_reference is None else comparison_reference, values)
        report["comparisons"][key] = metrics
        if not metrics["exact"]:
            failures.append(key)

    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100000000)
        generator = Gemma4Generator(mesh, max_seq_len=CONTEXT, layer_indices=layer_indices)
        report["runtime_precision"] = generator.model.precision_summary()
        report["layer_indices"] = list(generator.model.layer_indices)
        for name, prompt in PROMPTS.items():
            references[name], teachers[name] = standalone_run(generator, prompt, args.decode_steps)
            vectors[f"standalone_reference/{name}"] = references[name]
            assert torch.isfinite(references[name]).all(), f"Non-finite baseline logits for {name}"
            for repeat in range(1, args.repeats):
                values, consumed = standalone_run(generator, prompt, args.decode_steps, teachers[name])
                assert consumed == teachers[name]
                record(f"standalone_repeat_{repeat}", name, values)
        report["teacher_forced_decode_tokens"] = teachers
        adapter = AutoportGemma4ForCausalLM(generator, WIRE_ROWS)
        assert adapter.allow_host_sampling and not generator.host_sampling
        cache = serving_cache(adapter)
        for name in PROMPTS:
            first = None
            for repeat in range(args.repeats):
                values = adapter_run(adapter, cache, [name], teachers, args.decode_steps)[name]
                record(f"adapter_isolated_{repeat}", name, values)
                if first is None:
                    first = values
                else:
                    record(f"adapter_isolated_run_to_run_{repeat}", name, values, first)
        first_batch_order = None
        for order in (list(PROMPTS), list(reversed(PROMPTS))):
            first = None
            label = "".join(order)
            for repeat in range(args.repeats):
                results = adapter_run(adapter, cache, order, teachers, args.decode_steps)
                for name, values in results.items():
                    record(f"adapter_batch_{label}_{repeat}", name, values)
                    if first is not None:
                        record(f"adapter_batch_{label}_run_to_run_{repeat}", name, values, first[name])
                    if first_batch_order is not None and repeat == 0:
                        record("adapter_cross_batch_position", name, values, first_batch_order[name])
                first = results if first is None else first
            if first_batch_order is None:
                first_batch_order = first
        report["failures"] = failures
        report["status"] = "passed" if not failures else "failed"
        assert not failures, f"Logit equality failed: {failures}; see {args.output} and {tensor_path}"
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        # Preserve numerical evidence on assertion failures as well as success.
        torch.save(vectors, tensor_path)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if generator is not None:
            generator.teardown()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
