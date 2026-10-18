# SPDX-License-Identifier: Apache-2.0
"""Standalone real-weight batched linear row-compaction experiment.

No production monkeypatches. Synchronization/readbacks belong only to this
isolated component probe, never the live serving phase collector.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_batch_moe_probe import kernel_identity, measured
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tt import precision
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshOps, _hidden_parts


def compact_linear(ops, x, weight, **kwargs):
    """Compact only independent singleton rows with supported exact layouts."""
    shape = tuple(x.shape)
    rows = math.prod(shape[:-2])
    if (
        shape[-2] != 1
        or rows <= 1
        or math.prod(tuple(weight.shape)[:-2]) != 1
        or x.dtype not in (ttnn.bfloat16, ttnn.float32)
        or kwargs.get("dtype") not in (None, ttnn.bfloat16, ttnn.float32)
    ):
        return ops.linear(x, weight, **kwargs)
    flat = ttnn.to_layout(
        ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (1, 1, rows, shape[-1])), ttnn.TILE_LAYOUT
    )
    out = ops.linear(flat, weight, **kwargs)
    if out.dtype not in (ttnn.bfloat16, ttnn.float32):
        raise ValueError("Candidate must not silently cast a block-float output for row-major restore")
    return ttnn.to_layout(
        ttnn.reshape(ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT), (*shape[:-1], weight.shape[-1])), ttnn.TILE_LAYOUT
    )


def weight_cases(state, kind):
    """Reproduce the immutable checkpoint transforms in multichip_decoder.py."""
    if kind == "linear_attention":
        prefix = "linear_attn."
        qkv = state[prefix + "in_proj_qkv.weight"]
        parts = []
        for r in range(4):
            parts.append(
                torch.cat(
                    (
                        qkv[r * 512 : (r + 1) * 512],
                        qkv[2048 + r * 512 : 2048 + (r + 1) * 512],
                        qkv[4096 + r * 1536 : 4096 + (r + 1) * 1536],
                        state[prefix + "in_proj_z.weight"][r * 1536 : (r + 1) * 1536],
                        torch.nn.functional.pad(
                            state[prefix + "in_proj_a.weight"][r * 12 : (r + 1) * 12], (0, 0, 0, 20)
                        ),
                        torch.nn.functional.pad(
                            state[prefix + "in_proj_b.weight"][r * 12 : (r + 1) * 12], (0, 0, 0, 20)
                        ),
                    ),
                    dim=0,
                ).T
            )
        yield "gdn_input", parts, (32, 1, 2560), "gdn_input", "gdn_input", None, 20
        weight = state[prefix + "out_proj.weight"]
        yield "gdn_output", [weight[:, r * 1536 : (r + 1) * 1536].T for r in range(4)], (
            32,
            1,
            1536,
        ), "gdn_output", "gdn_output", None, 10
    elif kind == "indexed_attention":
        prefix = "self_attn."
        qg = state[prefix + "q_proj.weight"].reshape(24, 512, 2560)
        parts = []
        for r in range(4):
            group = r // 2
            parts.append(
                torch.cat(
                    (
                        qg[r * 6 : (r + 1) * 6, :256].reshape(1536, 2560),
                        state[prefix + "k_proj.weight"][group * 256 : (group + 1) * 256],
                        state[prefix + "v_proj.weight"][group * 256 : (group + 1) * 256],
                        qg[r * 6 : (r + 1) * 6, 256:].reshape(1536, 2560),
                        state[prefix + "indexer.index_qk_proj.weight"],
                    ),
                    dim=0,
                ).T
            )
        # Batched singleton QSA uses the BF8 prefill weight and LoFi singleton
        # fidelity. Flattening here must preserve that already-selected config.
        yield "qsa_input", parts, (32, 1, 1, 2560), "qsa_prefill", "qsa_decode", None, 20
        weight = state[prefix + "o_proj.weight"]
        yield "qsa_output", [weight[:, r * 1536 : (r + 1) * 1536].T for r in range(4)], (
            32,
            1,
            1,
            1536,
        ), "qsa_output", "qsa_output", None, 10
    else:
        raise ValueError(f"Unknown attention kind {kind}")
    prefix = "attn_hyper_connection."
    packed = torch.cat(
        (
            state[prefix + "input_mix_weight_down.weight"].T / 4,
            torch.nn.functional.pad(state[prefix + "block_inject_weight.weight"].T / 4, (0, 28)),
        ),
        dim=-1,
    )
    yield "hc_read", [x.T for x in _hidden_parts(packed.T)], (1, 32, 1, 2560), "hc", "hc", "float32", 5
    yield "hc_up", _hidden_parts(state[prefix + "input_mix_weight_up.weight"].T), (1, 32, 1, 320), "hc", "hc", None, 5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 1, 3])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmup < 1:
        parser.error("Positive repeats and warmup are required")
    if any(os.environ.get(key, "0") not in ("", "0") for key in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER")):
        raise RuntimeError("Run this standalone probe without watcher or device profiler")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    policy = precision.load_policy()
    report = {
        "scope": "isolated projections; not full-model correctness, accuracy, or serving performance",
        "argv": sys.argv,
        "checkpoint": str(args.checkpoint.resolve()),
        "mesh": [1, 4],
        "precision_policy": policy,
        "results": [],
        "source_sha256": {},
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "inputs": "deterministic independent per-chip rows; seeded random BF16 plus row markers",
    }
    for path in (
        Path(__file__),
        precision.ROOT / "tt/multichip_decoder.py",
        Path(precision.__file__),
        Path(sys.modules[measured.__module__].__file__),
    ):
        report["source_sha256"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        (args.output / (path.stem + ".py.txt")).write_bytes(path.read_bytes())
    (args.output / "identity.json").write_text(json.dumps(report, indent=2) + "\n")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=67108864)
    try:
        ops = MeshOps(mesh)
        rng = torch.Generator().manual_seed(2026101601)
        for layer in args.layers:
            selected = precision.for_layer(policy, layer)
            state = CheckpointState(args.checkpoint, f"model.language_model.layers.{layer}.")
            for role, parts, shape, weight_group, compute_group, output_dtype, cores in weight_cases(
                state, cfg.layer_types[layer]
            ):
                weight = ops.shards(parts, ttnn.bfloat8_b)
                desired = getattr(ttnn, selected["weight_groups"][weight_group])
                if weight.dtype != desired:
                    weight = ttnn.typecast(weight, desired)
                inputs = []
                for chip in range(4):
                    rows = torch.randn((32, shape[-1]), generator=rng) * 0.25
                    rows += torch.arange(32).unsqueeze(-1) * 0.01 + chip * 0.03
                    inputs.append(rows.reshape(shape).bfloat16())
                device = ops.shards(inputs)
                kw = {
                    "compute_kernel_config": precision.compute(mesh, selected["compute_fidelities"][compute_group]),
                    "max_cores": cores,
                }
                if output_dtype is not None:
                    kw["dtype"] = getattr(ttnn, output_dtype)
                before, before_replay, baseline_timing = measured(
                    mesh,
                    lambda: ops.linear(device, weight, **kw),
                    args.repeats,
                    args.warmup,
                )
                after, after_replay, candidate_timing = measured(
                    mesh,
                    lambda: compact_linear(ops, device, weight, **kw),
                    args.repeats,
                    args.warmup,
                )
                name = f"layer{layer}_{role}"
                record = {
                    "name": name,
                    "role": role,
                    "layer": layer,
                    "input_shape": list(shape),
                    "input_padded_shape": list(device.padded_shape),
                    "weight_shape": list(weight.shape),
                    "weight_dtype": str(weight.dtype),
                    "input_dtype": str(device.dtype),
                    "output_dtype": str(before.dtype),
                    "compute": kernel_identity(kw["compute_kernel_config"]),
                    "eager_comparison": pcc_metrics(before, after),
                    "baseline_eager_vs_replay": pcc_metrics(before, before_replay),
                    "candidate_eager_vs_replay": pcc_metrics(after, after_replay),
                    "replay_comparison": pcc_metrics(before_replay, after_replay),
                    "per_row_max_abs_error": (before.float() - after.float()).reshape(32, -1).abs().amax(-1).tolist(),
                    "baseline": baseline_timing,
                    "candidate": candidate_timing,
                    "median_speedup": baseline_timing["median_seconds"] / candidate_timing["median_seconds"],
                }
                report["results"].append(record)
                torch.save(
                    {
                        "input_shards": inputs,
                        "baseline": before,
                        "candidate": after,
                        "baseline_replay": before_replay,
                        "candidate_replay": after_replay,
                    },
                    args.output / f"{name}.pt",
                )
                (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(record), flush=True)
                del weight, device
        assert all(row["replay_comparison"]["pcc"] >= 0.995 for row in report["results"])
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
