# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-geometry haiku precision/state diagnostic; never a performance ranking."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.tt_qualitative import run
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator, configure_fabric


def compute_summary(config):
    return dict(
        math_fidelity=str(config.math_fidelity),
        fp32_dest_acc_en=config.fp32_dest_acc_en,
        math_approx_mode=config.math_approx_mode,
        packer_l1_acc=config.packer_l1_acc,
    )


def tensor_info(tensor):
    return dict(
        shape=list(tensor.shape),
        padded_shape=list(tensor.padded_shape),
        dtype=str(tensor.dtype),
        layout=str(tensor.layout),
        memory=str(tensor.memory_config()),
    )


def geometry(gen):
    assert gen.cache.capacity == 1088 and gen.cache.batch_size == 1
    assert tuple(gen.page_table.shape) == (1, 34) and gen.history_capacity == 1023
    full = []
    for layer, state in zip(gen.model.layers, gen.cache.layers):
        if layer.kind != "full_attention":
            continue
        pages, policy = gen.page_table.shape[-1], layer.policy
        grid = layer.device.compute_with_storage_grid_size()
        chunk = (32 if policy.get("adaptive_sdpa", False) and pages < 16 else policy.get("sdpa_k", 32)) or 32
        while (pages * layer.PAGE_SIZE) % chunk:
            chunk //= 2
        decode_grid = policy.get("sdpa_grid", [grid.x, grid.y])
        if gen.cache.batch_size == 1 and pages < 16:
            decode_grid = policy.get("sdpa_short_grid", decode_grid)
        full.append(
            dict(
                layer=layer.layer_idx,
                page_size=layer.PAGE_SIZE,
                local_q_heads=layer.config.num_attention_heads,
                local_kv_heads=layer.config.num_key_value_heads,
                head_dim=layer.config.head_dim,
                key=tensor_info(state.key),
                value=tensor_info(state.value),
                sdpa_q=policy.get("sdpa_q", 32),
                sdpa_k=chunk,
                sdpa_grid=decode_grid,
            )
        )
    return dict(
        capacity=gen.cache.capacity,
        num_pages=gen.cache.num_pages,
        batch_size=gen.cache.batch_size,
        page_table=tensor_info(gen.page_table),
        page_table_device=ttnn.to_torch(ttnn.get_device_tensors(gen.page_table)[0]).tolist(),
        history_capacity=gen.history_capacity,
        history=tensor_info(gen.token_history),
        full_attention=full,
    )


def first_difference(left, right):
    return next((i for i in range(max(len(left), len(right))) if left[i : i + 1] != right[i : i + 1]), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host-greedy", action="store_true", help="Add a slower full-logit host-argmax control")
    args = parser.parse_args()
    root = Path("models/autoports/qwen_qwen3_8_27b")
    policy_path = Path(os.environ["QWEN_PRECISION_CONFIG"])
    reference_path = root / "doc/full_model/hf_qualitative_extended.json"
    reference = json.loads(reference_path.read_text())
    prompt = next(row for row in reference["outputs"] if row["prompt_id"] == 0)
    assert len(prompt["prompt_tokens"]) == 60 and reference["max_new_tokens"] == 1024
    assert reference["revision"] == "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    sources = [
        *sorted((root / "tt").glob("*.py")),
        Path(__file__),
        root / "tests/tt_qualitative.py",
        reference_path,
        policy_path,
    ]
    report = dict(
        command=sys.argv,
        diagnostic_only=True,
        performance_ranking=False,
        policy_path=str(policy_path),
        policy_requested=json.loads(policy_path.read_text()),
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        reference=str(reference_path),
        hf_metadata={k: v for k, v in reference.items() if k != "outputs"},
        generation=dict(prompt_length=60, max_new_tokens=1024, top_k=1, top_p=0.0, temperature=1.0, seed=0),
        request_order=["native0", "native1"] + (["host_greedy"] if args.host_greedy else []),
        runs=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    torch.set_num_threads(8)
    configure_fabric()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=200000000)
    gen = None
    try:
        gen = build_generator(root, mesh)
        assert gen.model.layer_indices == list(range(64))
        assert gen.model.precision == report["policy_requested"]
        gen._ensure_cache(1, 1088)
        gen._ensure_history(1023)
        report.update(
            precision_policy=gen.model.precision,
            layer_indices=gen.model.layer_indices,
            checkpoint=str(gen.model.snapshot),
            tokenizer=type(gen.tokenizer).__name__,
            head=tensor_info(gen.model.head_weight),
            head_compute=compute_summary(gen.model.head_compute),
            head_decode_weights=[tensor_info(weight) for weight in gen.model.head_decode_weights],
            norm_compute=compute_summary(gen.model.norm_compute),
            initial_geometry=geometry(gen),
        )
        save()
        for name in report["request_order"]:
            gen.host_sampling = name == "host_greedy"
            before = geometry(gen)
            artifact = run(
                gen,
                root,
                reference_path.name,
                max_new_tokens=1024,
                prompt_ids=[0],
                output_dir=args.output.parent,
                output_name=f"{args.output.stem}_{name}.json",
            )
            result = json.loads(artifact.read_text())
            row = result["outputs"][0]
            after = geometry(gen)
            assert before == after == report["initial_geometry"]
            assert row["prompt_tokens"] == prompt["prompt_tokens"]
            stops = json.loads((gen.model.snapshot / "generation_config.json").read_text())["eos_token_id"]
            stops = stops if isinstance(stops, list) else [stops]
            report["runs"].append(
                dict(
                    name=name,
                    artifact=str(artifact),
                    output=row,
                    first_eos=next((i for i, token in enumerate(row["tokens"]) if token in stops), None),
                    counters_total=dict(gen.counters),
                    geometry=after,
                )
            )
            if len(report["runs"]) >= 2:
                native = report["runs"][0]["output"]["tokens"]
                difference = first_difference(native, row["tokens"])
                report[f"native0_vs_{name}"] = dict(tokens_equal=difference is None, first_divergence=difference)
            save()
        report["status"] = "complete"
        save()
    except BaseException as error:
        report.update(status="error", error=repr(error))
        save()
        raise
    finally:
        try:
            if gen is not None:
                gen.close()
        finally:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
