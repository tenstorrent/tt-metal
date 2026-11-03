# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized, reproducible full-layer candidate runner. Stops on a timeout."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/optimized_decoder"
BASE = dict(
    experts_dtype="bfloat4_b",
    experts_fidelity="LoFi",
    gate_up_grid=[8, 2],
    gate_up_block=10,
    gate_up_subblock=3,
    down_grid=[8, 2],
    down_block=8,
    down_subblock=5,
    value_grid=[8, 2],
    value_block=10,
    value_subblock=2,
    sparse_l1=True,
)


def run(tag, options, layer=3, baseline=False, real=True, source=None, iterations=30, seq=256):
    command = [
        sys.executable,
        "-m",
        "models.autoports.ifm_k2_horizon_mova_36b_a4b.tests.run_optimized_decoder",
        "--layer",
        str(layer),
        "--seq",
        str(seq),
        "--context",
        str(max(1024, seq * 2)),
        "--trace",
        "--profile",
        "--iterations",
        str(iterations),
        "--tag",
        tag,
    ]
    if baseline:
        command += ["--baseline"]
    else:
        command += ["--options", json.dumps(options)]
    if source:
        command += ["--source", str(ROOT / "tests" / source)]
    if real:
        command += ["--real-inputs"]
    log = DOC / "logs" / f"{tag}_layer{layer}.log"
    print("START", tag, layer, flush=True)
    with log.open("w") as f:
        try:
            result = subprocess.Popen(
                command, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "TT_METAL_TRACE_ALLOC_TRACKING": "0"}
            )
            result.wait(timeout=240)
        except subprocess.TimeoutExpired:
            print(
                f"TIMEOUT: PID {result.pid} is left alive; capture triage before termination or more hardware work",
                flush=True,
            )
            raise
    record = dict(command=command, exit_code=result.returncode, log=str(log), options=options)
    (DOC / "candidates" / f"{tag}_layer{layer}_execution.json").write_text(json.dumps(record, indent=2))
    output = DOC / "candidates" / f"{tag}_layer{layer}.json"
    if output.exists():
        report = json.loads(output.read_text())
        print(
            "RESULT",
            tag,
            report["prefill_pcc"],
            report["traced_decode_pcc"],
            report["warmed_prefill_wall_ms"],
            report["traced_warmed_decode_wall_ms"],
            flush=True,
        )
    else:
        print("FAIL", tag, result.returncode, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="sparse")
    args = ap.parse_args()
    if args.suite == "sparse":
        run("baseline_real", {}, baseline=True)
        run("bfp8_real", {})
        run("bfp4_geom_control", BASE)
        run("bfp8_geom", {**BASE, "experts_dtype": "bfloat8_b", "experts_fidelity": "HiFi2"})
        run("bfp8_lofi_geom", {**BASE, "experts_dtype": "bfloat8_b"})
        for role, grids in [
            ("gate_up", [([8, 1], 6), ([8, 4], 2)]),
            ("down", [([8, 1], 5), ([8, 4], 3)]),
            ("value", [([8, 1], 4), ([8, 4], 1)]),
        ]:
            for grid, sub in grids:
                run(f"{role}_grid{grid[0]*grid[1]}", {**BASE, role + "_grid": grid, role + "_subblock": sub})
        for role, blocks in [("gate_up", [5, 16, 20, 40, 80]), ("down", [3, 6, 12, 24]), ("value", [16, 20, 40, 80])]:
            for block in blocks:
                run(f"{role}_block{block}", {**BASE, role + "_block": block})
        run("sparse_dram_intermediates", {**BASE, "sparse_l1": False})
        run("value4", {**BASE, "value_dtype": "bfloat4_b", "value_fidelity": "LoFi"})
        run("value8_lofi", {**BASE, "value_fidelity": "LoFi"})
        run("attention4", {**BASE, "attention_dtype": "bfloat4_b", "attention_fidelity": "LoFi"})
        run("attention8_lofi", {**BASE, "attention_fidelity": "LoFi"})

    elif args.suite == "dense":
        source = "candidate_optimized_dram.py"
        base = dict(dram_attn_block=5, dram_output_block=8, dram_up_block=5, dram_down_block=12)
        run("baseline_real", {}, layer=0, baseline=True)
        run("dense_bfp8", {}, layer=0)
        candidates = [
            ("dram_control", {}),
            ("dram_mlp4", dict(mlp_dtype="bfloat4_b", mlp_fidelity="LoFi")),
            ("dram_attention4", dict(attention_dtype="bfloat4_b", attention_fidelity="LoFi")),
            ("dram_mlp8_lofi", dict(mlp_fidelity="LoFi")),
            ("dram_attention8_lofi", dict(attention_fidelity="LoFi")),
            ("dram_packed8", dict(packed_dense=True)),
            ("dram_separate_attention8", dict(split_attention=True)),
        ]
        for tag, options in candidates:
            run(tag, {**base, **options}, layer=0, source=source)
        for readers in [2, 3]:
            run(f"dram_readers{readers}", {**base, "readers": readers}, layer=0, source=source)
        for role, blocks in [
            ("attn", [1, 2, 10, 20]),
            ("output", [4, 16, 32]),
            ("up", [2, 10, 20]),
            ("down", [6, 24, 48]),
        ]:
            for block in blocks:
                run(f"dram_{role}_block{block}", {**base, "dram_" + role + "_block": block}, layer=0, source=source)
        for cores in [4, 16]:
            run(f"dram_cores{cores}", {**base, "dense_cores": cores}, layer=0, source=source)
        for grid in [[8, 4], [8, 8], [10, 8]]:
            run(
                f"prefill2d_{grid[0]}x{grid[1]}",
                {**base, "prefill_2d": True, "prefill_grid": grid},
                layer=0,
                source=source,
            )

    elif args.suite == "cumulative":
        base = json.loads((DOC / "candidates/dense_best_options.json").read_text())
        source = "candidate_optimized_cache.py"
        for tag, extra in [
            ("packed4", {"packed_dense": True}),
            ("readers3_padded", {"readers": 3}),
            ("norm_sharded", {"sharded_norm": True}),
            ("norm_bf16", {"sharded_norm": True, "bf16_norm": True}),
            ("residual_sharded", {"sharded_residual": True, "sharded_norm": True}),
            ("cache8", {"cache_dtype": "bfloat8_b"}),
            ("sdpa_k256", {"sdpa_k": 256}),
            ("sdpa_k128", {"sdpa_k": 128}),
            ("sdpa_auto", {"sdpa_k": 0}),
            ("sdpa_8x4", {"sdpa_grid": [8, 4]}),
            ("sdpa_approx", {"sdpa_approx": True}),
        ]:
            run("cum_" + tag, {**base, **extra}, layer=0, source=source)
        for role, blocks in [
            ("attn", [2, 10, 20]),
            ("output", [8, 32, 64]),
            ("up", [5, 20, 40]),
            ("down", [12, 48, 96]),
        ]:
            for block in blocks:
                run(f"cum_{role}_block{block}", {**base, "dram_" + role + "_block": block}, layer=0, source=source)
        for cores in [4, 16, 40]:
            run(f"cum_cores{cores}", {**base, "dense_cores": cores}, layer=0, source=source)
        for role in ["attn", "output", "up", "down"]:
            for readers in [1, 3]:
                run(f"cum_{role}_readers{readers}", {**base, role + "_readers": readers}, layer=0, source=source)

    elif args.suite == "sparse_cumulative":
        base = json.loads((DOC / "candidates/sparse_cumulative_options.json").read_text())
        source = "candidate_optimized_experts.py"
        candidates = [
            ("split", dict(split_experts=True)),
            ("split_fused", dict(split_experts=True, fused_gate=True)),
            ("split24", dict(split_experts=True, split_grid=[8, 3], split_subblock=1)),
            ("split12", dict(split_experts=True, split_grid=[6, 2], split_subblock=2)),
            ("gate48", dict(gate_up_grid=[8, 6], gate_up_subblock=1)),
            ("down40", dict(down_grid=[8, 5], down_subblock=2)),
            ("down80", dict(down_grid=[10, 8], down_subblock=1)),
            ("value32", dict(value_grid=[8, 4], value_subblock=1)),
            ("gate32", dict(gate_up_grid=[8, 4], gate_up_subblock=2)),
            ("experts8", dict(experts_dtype="bfloat8_b", experts_fidelity="HiFi2")),
            ("experts8lofi", dict(experts_dtype="bfloat8_b")),
            ("value4", dict(value_dtype="bfloat4_b", value_fidelity="LoFi")),
            ("value8lofi", dict(value_fidelity="LoFi")),
            ("expert_activation8", dict(expert_activation="bfloat8_b")),
            ("value_activation8", dict(value_activation="bfloat8_b")),
            ("attention8", dict(attention_dtype="bfloat8_b")),
            ("norm_bf16", dict(bf16_norm=True)),
            ("residual", dict(sharded_residual=True)),
            (
                "prefill48_80",
                dict(
                    prefill_gate_up_grid=[8, 6],
                    prefill_gate_up_subblock=1,
                    prefill_down_grid=[10, 8],
                    prefill_down_subblock=1,
                    prefill_value_grid=[8, 4],
                    prefill_value_subblock=1,
                ),
            ),
            ("chunk64_l1", dict(chunk_size=64, l1_groups=2)),
            ("chunk128_l1", dict(chunk_size=128, l1_groups=4)),
        ]
        for tag, extra in candidates:
            run("sparsecum_" + tag, {**base, **extra}, layer=3, source=source)
    elif args.suite == "dense_layout":
        base = json.loads((DOC / "candidates/dense_layout_options.json").read_text())
        source = "candidate_optimized_cache.py"
        candidates = [
            ("control", dict(sharded_norm=False)),
            ("bf16", dict(bf16_norm=True)),
            ("norm4", dict(norm_cores=4)),
            ("norm10", dict(norm_cores=10)),
            ("norm20", dict(norm_cores=20)),
            ("norm40", dict(norm_cores=40)),
            ("residual", dict(sharded_residual=True)),
            ("packed", dict(packed_dense=True)),
            ("sdpa256", dict(sdpa_k=256)),
            ("sdpaauto", dict(sdpa_k=0)),
            ("cachebf16", dict(cache_dtype="bfloat16")),
        ]
        for tag, extra in candidates:
            run("layout_" + tag, {**base, **extra}, layer=0, source=source)

    elif args.suite == "final":
        for layer in (0, 3):
            source = "candidate_optimized_final.py"
            run("final_control", {}, layer=layer, source=source, iterations=100)
            for group in ("attention", "mlp"):
                run(
                    "final_" + group + "_activation8",
                    {group + "_activation": "bfloat8_b"},
                    layer=layer,
                    source=source,
                    iterations=100,
                )
            if layer == 3:
                for tag, options in [
                    ("constant_placeholder", {"constant_placeholder": True}),
                    ("expert_activation8", {"expert_activation": "bfloat8_b"}),
                    ("value_activation8", {"value_activation": "bfloat8_b"}),
                    ("sharded_residual", {"sharded_residual": True, "keep_sharded": True}),
                ]:
                    run("final_" + tag, options, layer=layer, source=source, iterations=100)
    elif args.suite == "final_topology":
        for layer in (0, 3):
            for readers, block in ((1, 5), (1, 10), (2, 5), (2, 10)):
                run(
                    f"final_split_attention_r{readers}_k{block}",
                    dict(split_attention=True, attn_readers=readers, dram_attn_block=block),
                    layer=layer,
                    iterations=100,
                )
        for block in (5, 10):
            run(f"final_separate_mlp_k{block}", dict(packed_dense=False, dram_up_block=block), layer=0, iterations=100)
        import ast

        tree = ast.parse((ROOT / "tt/optimized_decoder.py").read_text())
        policy = next(
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MOE_POLICY" for t in n.targets)
        )
        for grid, sub in (([8, 2], 2), ([8, 3], 1)):
            run(
                f"final_split_experts_{grid[0]*grid[1]}_sub{sub}",
                {**policy, "split_experts": True, "split_grid": grid, "split_subblock": sub, "fused_gate": True},
                layer=3,
                source="candidate_optimized_experts.py",
                iterations=100,
            )
    elif args.suite == "final_combined":
        source = "candidate_optimized_final.py"
        for repeat in range(3):
            run(f"combined_dense_control_{repeat}", {}, layer=0, source=source, iterations=300)
            run(
                f"combined_dense_act8_{repeat}",
                dict(attention_activation="bfloat8_b", mlp_activation="bfloat8_b"),
                layer=0,
                source=source,
                iterations=300,
            )
        base = dict(constant_placeholder=True)
        for tag, extra in [
            ("control", {}),
            ("norm8", dict(attention_activation="bfloat8_b", mlp_activation="bfloat8_b")),
            (
                "expert8",
                dict(attention_activation="bfloat8_b", mlp_activation="bfloat8_b", expert_activation="bfloat8_b"),
            ),
            (
                "residual",
                dict(
                    attention_activation="bfloat8_b",
                    mlp_activation="bfloat8_b",
                    expert_activation="bfloat8_b",
                    sharded_residual=True,
                    keep_sharded=True,
                ),
            ),
        ]:
            run("combined_sparse_" + tag, {**base, **extra}, layer=3, source=source, iterations=300)

    elif args.suite == "final_prefill":
        for layer in (0, 3):
            for chunk in (256, 512, 1024, 2048):
                run(f"long_prefill_chunk{chunk}", {"chunk_size": chunk}, layer=layer, seq=2048, iterations=20)
            run("long_prefill_baseline", {}, layer=layer, seq=2048, iterations=5, baseline=True)

    elif args.suite == "final_geometry":
        for layer, cases in [
            (
                0,
                [
                    ("attn10", {"dram_attn_block": 10}),
                    ("up10", {"dram_up_block": 10}),
                    ("output32", {"dram_output_block": 32}),
                    ("down48", {"dram_down_block": 48}),
                    (
                        "cores16",
                        {
                            "dense_cores": 16,
                            "dram_attn_block": 5,
                            "dram_up_block": 5,
                            "dram_output_block": 8,
                            "dram_down_block": 12,
                        },
                    ),
                ],
            ),
            (
                3,
                [
                    ("gate80", {"gate_up_block": 80}),
                    ("value80", {"value_block": 80}),
                    ("output32", {"dram_output_block": 32}),
                    (
                        "cores16",
                        {
                            "dense_cores": 16,
                            "dram_attn_block": 5,
                            "dram_up_block": 5,
                            "dram_output_block": 8,
                            "dram_down_block": 12,
                        },
                    ),
                ],
            ),
        ]:
            for tag, options in cases:
                run("closure_geometry_" + tag, options, layer=layer, iterations=100)
        for grid, sub in (([8, 2], 2), ([8, 3], 1)):
            run(
                f"closure_split_experts_{grid[0]*grid[1]}",
                dict(split_experts=True, split_grid=grid, split_subblock=sub, fused_gate=True),
                layer=3,
                source="candidate_optimized_separate_experts.py",
                iterations=100,
            )

    elif args.suite == "composite_router":
        run("router_decode", {}, layer=3, source="candidate_optimized_router.py", iterations=100)
        run("router_both", {"prefill_composite": True}, layer=3, source="candidate_optimized_router.py", iterations=100)
        run(
            "closure_cores16_adapted",
            dict(
                dense_cores=16,
                down_cores=8,
                dram_attn_block=5,
                dram_up_block=5,
                dram_output_block=8,
                dram_down_block=24,
            ),
            layer=3,
            iterations=100,
        )
        run(
            "closure_split_experts_12_adapted",
            dict(split_experts=True, split_grid=[6, 2], split_subblock=2, fused_gate=True),
            layer=3,
            source="candidate_optimized_separate_experts.py",
            iterations=100,
        )

    elif args.suite == "norm_prefill":
        src = "candidate_optimized_norm_prefill.py"
        run("norm_output_sharded", {"norm_output_sharded": True}, layer=0, source=src, iterations=100)
        run("attn_reader2_k10", dict(attn_readers=2, dram_attn_block=10), layer=0, iterations=100)
        for seq in (256, 2048):
            for m, k, n in ((4, 16, 8), (8, 16, 16), (4, 40, 8)):
                run(
                    f"minimal_m{m}_k{k}_n{n}_s{seq}",
                    dict(
                        minimal_config=dict(M_block_size=m, K_block_size=k, N_block_size=n, subblock_h=2, subblock_w=4)
                    ),
                    layer=0,
                    source=src,
                    seq=seq,
                    iterations=30,
                )
        for layer in (0, 3):
            for grid in ([8, 8], [10, 10]):
                run(
                    f"prefill_final_grid{grid[0]}x{grid[1]}",
                    dict(prefill_grid=grid),
                    layer=layer,
                    seq=2048,
                    iterations=20,
                )

    elif args.suite == "norm_prefill_adapted":
        src = "candidate_optimized_norm_prefill.py"
        run(
            "norm_sharded_attn10",
            dict(norm_output_sharded=True, dram_attn_block=10),
            layer=0,
            source=src,
            iterations=100,
        )
        run("norm_sharded_sparse", dict(norm_output_sharded=True), layer=3, source=src, iterations=100)
        run("prefill_grid8x8_block4", dict(prefill_grid=[8, 8], prefill_block=4), layer=0, seq=2048, iterations=20)
        for seq in (256, 2048):
            for m, k, n in ((4, 16, 8), (8, 16, 16), (4, 40, 8)):
                run(
                    f"minimal_adapted_m{m}_k{k}_n{n}_s{seq}",
                    dict(
                        minimal_config=dict(M_block_size=m, K_block_size=k, N_block_size=n, subblock_h=2, subblock_w=4)
                    ),
                    layer=0,
                    source=src,
                    seq=seq,
                    iterations=30,
                )

    elif args.suite == "prefill_precision":
        src = "candidate_optimized_prefill_precision.py"
        for tag, options in [
            ("expert8", dict(prefill_expert8=True)),
            ("value8", dict(prefill_value8=True)),
            ("both8", dict(prefill_expert8=True, prefill_value8=True)),
        ]:
            for seq in (32, 256, 2048):
                run(f"prefill_{tag}_s{seq}", options, layer=3, source=src, seq=seq, iterations=30)

    elif args.suite == "sharded_expert_input":
        run(
            "norm_sparse_direct_sharded",
            dict(norm_output_sharded=True, expert_keep_sharded=True),
            layer=3,
            source="candidate_optimized_norm_prefill.py",
            iterations=100,
        )

    elif args.suite == "expert_dram":
        roles = ["value_experts", "experts.gate_up", "experts.down_proj"]
        for names in ([roles[0]], [roles[1]], [roles[2]], roles):
            run(
                "expert_dram_" + str(len(names)) + "_" + names[0].replace(".", "_"),
                dict(sharded_expert_weights=names),
                layer=3,
                source="candidate_optimized_expert_dram.py",
                iterations=100,
            )

    elif args.suite == "attention_compute":
        for layer in (0, 3):
            for fidelity, fp32 in [("HiFi2", True), ("HiFi2", False), ("LoFi", True)]:
                run(
                    f"sdpa_export_adapted_{fidelity}_fp32{fp32}",
                    dict(sdpa_fidelity=fidelity, sdpa_fp32=fp32),
                    layer=layer,
                    source="candidate_optimized_attention_compute.py",
                    iterations=100,
                )

    elif args.suite == "attention_grid":
        src = "candidate_optimized_attention_compute.py"
        for layer in (0, 3):
            for tag, options in [
                ("lofi16", dict(sdpa_fidelity="LoFi", sdpa_fp32=False)),
                ("full_prefill", dict(sdpa_fidelity="HiFi2", sdpa_fp32=False, prefill_sdpa_grid="full")),
                ("full_both", dict(sdpa_fidelity="HiFi2", sdpa_fp32=False, prefill_sdpa_grid="full", sdpa_grid="full")),
            ]:
                run("sdpa_grid_" + tag, options, layer=layer, source=src, iterations=100)
