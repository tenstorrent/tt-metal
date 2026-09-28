# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolate shared BF16 projections on real layer inputs; not serving accuracy."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import MODEL_ID, REVISION, Gemma4Model
from models.common.utility_functions import comp_pcc


def host_shards(value):
    return [ttnn.to_torch(shard).float().clone() for shard in ttnn.get_device_tensors(value)]


def weight_from(project):
    values = [cell.cell_contents for cell in (project.__closure__ or ()) if isinstance(cell.cell_contents, ttnn.Tensor)]
    assert len(values) == 1, "Expected a single original TT weight per shared projection"
    assert values[0].dtype == ttnn.bfloat16
    return values[0]


def program(grid, per_n, block_k):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=block_k,
        out_subblock_h=1,
        out_subblock_w=per_n,
        out_block_h=4,
        out_block_w=per_n,
        per_core_M=4,
        per_core_N=per_n,
        fuse_batch=True,
        mcast_in0=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("samples must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        scope="Isolated shared projections on actual eager layer inputs; layers independently consume S128 embeddings",
        model=MODEL_ID,
        revision=REVISION,
        layers=[0, 5],
        input_length=128,
        precision="Original BF16 weights/input/output; generic defaults vs explicit HiFi2 matching generic resolution",
        raw_tensors=str(args.output.with_suffix(".pt")),
        results=[],
        completed=False,
    )
    raw = {}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    torch.set_num_threads(4)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1024**3)
    trace = None
    try:
        model = Gemma4Model(mesh, max_seq_len=1024, layer_indices=(0, 5))
        cache, host_table = model.allocate_cache(slots=1, context=1024)
        table = model.upload(host_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
        ids = tokenizer.encode(
            "Explain how sunlight, water, and soil help a tree grow, using concrete examples. " * 32
        )[:128]
        assert len(ids) == 128
        report["token_ids"] = ids
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        report["explicit_compute"] = str(compute)
        programs = {
            "gate": [(f"gate{k}", program((9, 2), 2, k)) for k in (22, 44, 88)],
            "down": [("down44", program((11, 4), 2, 17))],
        }
        for number, (index, layer) in enumerate(zip(model.layer_indices, model.layers)):
            shared = layer.layer.shared_mlp
            original = {"gate": shared.gate_up, "down": shared.down}
            captured = {}
            input_memory = {}

            def capture(kind):
                def project(x):
                    assert x.dtype == ttnn.bfloat16 and x.shape[-2] == 128
                    captured[kind] = host_shards(x)
                    input_memory[kind] = x.memory_config()
                    return original[kind](x)

                return project

            tokens = model.upload(torch.tensor(ids).reshape(1, 1, 1, 128).int(), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            embedding = model.embed(tokens)
            try:
                shared.gate_up, shared.down = capture("gate"), capture("down")
                output = layer.prefill_forward(
                    embedding,
                    rope_mats=model.rope_prefill[model.config.layer_types[index]],
                    page_table=table,
                    kv_cache=cache[number],
                    user_id=0,
                )
                ttnn.synchronize_device(mesh)
                output.deallocate(True)
            finally:
                shared.gate_up, shared.down = original["gate"], original["down"]
            for kind in ("gate", "down"):
                prefix = f"layer{index}_{kind}"
                raw[prefix + "_input_shards"] = captured[kind]
                # Preserve each rank's distinct input exactly, including down's
                # column-parallel activations; never replicate rank0 to others.
                x = ttnn.from_torch(
                    torch.cat(captured[kind], dim=-1).to(torch.bfloat16),
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                    memory_config=input_memory[kind],
                )
                assert all(torch.equal(a, b) for a, b in zip(captured[kind], host_shards(x)))
                weight = weight_from(original[kind])
                reference = None
                for name, config in [("generic", None), ("generic_explicit_hifi2", None)] + programs[kind]:
                    assert trace is None

                    def forward():
                        if name == "generic":
                            return original[kind](x)
                        return ttnn.linear(x, weight, program_config=config, compute_kernel_config=compute)

                    output = forward()
                    actual = torch.stack(host_shards(output))
                    output.deallocate(True)
                    if reference is None:
                        reference = actual
                    _, pcc = comp_pcc(reference, actual, 0.999)
                    row = dict(
                        layer=index,
                        projection=kind,
                        candidate=name,
                        program=None if config is None else str(config),
                        compute=None if name == "generic" else str(compute),
                        weight_shape=list(weight.shape),
                        weight_dtype=str(weight.dtype),
                        input_shard_shape=list(captured[kind][0].shape),
                        input_memory=str(input_memory[kind]),
                        pcc=float(pcc),
                        exact_equal=torch.equal(reference, actual),
                        finite=bool(torch.isfinite(actual).all()),
                        max_abs=float((actual - reference).abs().max()),
                        relative_l2=float(
                            torch.linalg.vector_norm(actual - reference)
                            / torch.linalg.vector_norm(reference).clamp_min(1e-12)
                        ),
                    )
                    raw[prefix + "_" + name] = actual
                    report["results"].append(row)
                    ttnn.synchronize_device(mesh)
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    try:
                        output = forward()
                    finally:
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    for _ in range(2):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples = []
                    for _ in range(args.samples):
                        started = time.perf_counter_ns()
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append((time.perf_counter_ns() - started) / 1000)
                    row["replay_exact"] = torch.equal(actual, torch.stack(host_shards(output)))
                    row["trace_us"] = samples
                    row["median_trace_us"] = statistics.median(samples)
                    ttnn.release_trace(mesh, trace)
                    trace = None
                    output.deallocate(True)
                    save()
                x.deallocate(True)
            embedding.deallocate(True)
            tokens.deallocate(True)
        report["completed"] = True
        report["all_exact"] = all(row["exact_equal"] for row in report["results"])
        report["all_replays_exact"] = all(row["replay_exact"] for row in report["results"])
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        save()
        try:
            torch.save(raw, args.output.with_suffix(".pt"))
        finally:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
