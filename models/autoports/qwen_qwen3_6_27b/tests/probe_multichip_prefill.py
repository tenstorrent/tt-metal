# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Test-only native SwiGLU/MMRS configuration sweep using the decoder oracle.

One candidate per process. All ordinary run_multichip_decoder flags are forwarded.
Config tuples are M,K,N,subblock_h,subblock_w,grid_x,grid_y, in tile/core units.
No production edits or shared native kernel changes are made.
"""

import argparse
import hashlib
import json
import math
import shlex
import sys
import traceback
from pathlib import Path
from unittest.mock import patch

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests import run_multichip_decoder as runner

FROZEN_SOURCE_SHA256 = "de734a896dc879ef5e4d3ebfb2f41f4734d53e31af1cae52c48d2d8775e42f3e"
CONFIG_KEYS = (
    "M_block_size",
    "K_block_size",
    "N_block_size",
    "subblock_h",
    "subblock_w",
    "grid_x",
    "grid_y",
)


def parse_config(text):
    try:
        values = tuple(int(value) for value in text.split(","))
        assert len(values) == 7
        m, k, n, sh, sw, gx, gy = values
        assert min(values) > 0 and m % sh == 0 and n % sw == 0
        assert n % 2 == 0 and sw % 2 == 0, "keep complete pairs and fused last-K SwiGLU"
        assert sh * sw <= 4, "conservative FP32 DEST subblock area"
        assert 2 <= gx <= 11 and 2 <= gy <= 10
    except (ValueError, AssertionError) as error:
        raise argparse.ArgumentTypeError(f"invalid M,K,N,sh,sw,gx,gy config: {text}: {error}") from error
    return dict(zip(CONFIG_KEYS, values))


def native_config(config):
    fields = {key: value for key, value in config.items() if key not in ("grid_x", "grid_y")}
    fields["compute_with_storage_grid_size"] = (config["grid_x"], config["grid_y"])
    return ttnn.MinimalMatmulConfig(**fields)


def geometry(rows, config):
    """Source-derived work split; a useful-work ratio, not measured utilization."""
    mt, kt, nt = math.ceil(rows / 32), 160, 272
    transposed = mt * 32 > nt * 32
    mcores = config["grid_x"] if transposed else config["grid_y"]
    ncores = config["grid_y"] if transposed else config["grid_x"]
    mpc, npc = math.ceil(mt / mcores), 2 * math.ceil((nt // 2) / ncores)
    mb, kb, nb = (config[key] for key in ("M_block_size", "K_block_size", "N_block_size"))
    padded_mt, padded_nt, padded_kt = mpc * mcores, npc * ncores, math.ceil(kt / kb) * kb
    # Double-buffered A/W, half-width double-buffered BF16 output, and one
    # full-width FP32 intermediate block. Excludes descriptors/kernel scratch.
    dfb_bytes = 2 * mb * kb * 2048 + 2 * kb * nb * 1088 + mb * nb * (2048 + 4096)
    return {
        "rows": rows,
        "transpose_core_grid": transposed,
        "grid_cores": mcores * ncores,
        "m_tiles_per_core": mpc,
        "weight_n_tiles_per_core": npc,
        "m_blocks_per_core": math.ceil(mpc / mb),
        "n_blocks_per_core": math.ceil(npc / nb),
        "k_blocks": padded_kt // kb,
        "useful_tile_volume_fraction": mt * nt * kt / (padded_mt * padded_nt * padded_kt),
        "main_dfb_bytes_per_core_fp32_intermediate": dfb_bytes,
        "note": "partial block loops shrink; uniform grid padding still participates in multicast/compute",
    }


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--swiglu-config", type=parse_config)
    parser.add_argument("--swiglu-tail-config", type=parse_config, help="override for at most32 padded rows")
    parser.add_argument("--swiglu-large-config", type=parse_config, help="override when padded rows exceed8704")
    parser.add_argument("--swiglu-input-l1", action="store_true")
    parser.add_argument("--mmrs-headroom-kib", type=int)
    parser.add_argument("--mmrs-links", type=int, choices=(1, 2))
    parser.add_argument("--mmrs-input-l1", action="store_true")
    parser.add_argument("--mmrs-output-l1", action="store_true")
    parser.add_argument("--mmrs-row-multiple", type=int, default=32)
    parser.add_argument("--mmrs-chunk-rows", type=int)
    parser.add_argument("--mmrs-disable", action="store_true", help="control: ordinary row matmul then reduce scatter")
    parser.add_argument("--mmrs-role", choices=("all", "attention", "mlp"), default="all")
    options, remaining = parser.parse_known_args()
    assert options.mmrs_row_multiple >= 32 and options.mmrs_row_multiple % 32 == 0
    assert options.mmrs_chunk_rows is None or (options.mmrs_chunk_rows >= 32 and options.mmrs_chunk_rows % 32 == 0)
    assert options.mmrs_headroom_kib is None or options.mmrs_headroom_kib >= 0
    if options.mmrs_disable:
        policy_parser = argparse.ArgumentParser(add_help=False)
        policy_parser.add_argument("--policy-json", default="{}")
        policy_parser.add_argument("--policy-file", type=Path)
        policy_args, remaining = policy_parser.parse_known_args(remaining)
        policy = json.loads(policy_args.policy_file.read_text() if policy_args.policy_file else policy_args.policy_json)
        policy["fused_mmrs"] = False
        remaining += ["--policy-json", json.dumps(policy)]
    inspect_args = argparse.ArgumentParser(add_help=False)
    inspect_args.add_argument("--output", type=Path, required=True)
    inspect_args.add_argument("--baseline", action="store_true")
    destination, _ = inspect_args.parse_known_args(remaining)
    assert not destination.baseline, "generate optimized references with the original runner"
    configs = {
        name: (config, native_config(config))
        for name, config in (
            ("default", options.swiglu_config),
            ("tail", options.swiglu_tail_config),
            ("large", options.swiglu_large_config),
        )
        if config is not None
    }
    observations = {"swiglu": {}, "mmrs": {}}
    original_minimal = ttnn.experimental.minimal_matmul
    original_mmrs = ttnn.experimental.matmul_reduce_scatter_prefill
    # Plain nanobind debug functions are not registered onto ttnn.experimental.
    # Match the native MMRS unit-test helper's underlying binding namespace.
    last_mmrs_plan = ttnn._ttnn.operations.experimental.ccl_experimental._matmul_reduce_scatter_prefill_last_plan

    def record(kind, value):
        key = json.dumps(value, sort_keys=True)
        if key not in observations[kind]:
            observations[kind][key] = {"calls": 0, **value}
            print("PREFILL_CANDIDATE", kind, key, flush=True)
        observations[kind][key]["calls"] += 1

    def minimal(input_tensor, weight_tensor, *args, **kwargs):
        if not kwargs.get("fuse_swiglu", False) or tuple(weight_tensor.shape)[-2:] != (5120, 8704):
            return original_minimal(input_tensor, weight_tensor, *args, **kwargs)
        rows = math.prod(tuple(input_tensor.padded_shape)[:-1])
        selection = (
            "tail" if rows <= 32 and "tail" in configs else "large" if rows > 8704 and "large" in configs else "default"
        )
        config = configs.get(selection)
        if config is not None:
            kwargs["config"] = config[1]
        if options.swiglu_input_l1:
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
        value = original_minimal(input_tensor, weight_tensor, *args, **kwargs)
        record(
            "swiglu",
            {
                "input_shape": list(input_tensor.shape),
                "input_padded_shape": list(input_tensor.padded_shape),
                "config_override": None if config is None else config[0],
                "actual_config": str(kwargs.get("config")),
                "input_memory": str(input_tensor.memory_config()),
                "geometry": None if config is None else geometry(rows, config[0]),
            },
        )
        return value

    def mmrs(input_tensor, weight, *args, **kwargs):
        weight_shape = tuple(weight.shape)
        if weight_shape not in ((1536, 5120), (4352, 5120)) or kwargs.get("scatter_dim") != -1:
            return original_mmrs(input_tensor, weight, *args, **kwargs)
        role = "mlp" if weight_shape[0] == 4352 else "attention"
        if options.mmrs_role not in ("all", role):
            return original_mmrs(input_tensor, weight, *args, **kwargs)
        original_rows = input_tensor.shape[-2]
        padded_rows = math.ceil(original_rows / options.mmrs_row_multiple) * options.mmrs_row_multiple
        assert kwargs.get("output_tensor") is None, "candidate row padding assumes native-owned output"
        if padded_rows != original_rows:
            padding = [(0, 0)] * len(input_tensor.shape)
            padding[-2] = (0, padded_rows - original_rows)
            input_tensor = ttnn.pad(input_tensor, tuple(padding), value=0.0)
        if options.mmrs_input_l1:
            input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG)
        if options.mmrs_output_l1:
            kwargs["memory_config"] = ttnn.L1_MEMORY_CONFIG
        if options.mmrs_headroom_kib is not None:
            kwargs["l1_headroom"] = options.mmrs_headroom_kib * 1024
        if options.mmrs_links is not None:
            kwargs["num_links"] = options.mmrs_links
        chunk_rows = options.mmrs_chunk_rows or padded_rows
        outputs = []
        for start_row in range(0, padded_rows, chunk_rows):
            end_row = min(start_row + chunk_rows, padded_rows)
            if start_row == 0 and end_row == padded_rows:
                chunk = input_tensor
            else:
                starts, ends = [0] * len(input_tensor.shape), list(input_tensor.shape)
                starts[-2], ends[-2] = start_row, end_row
                chunk = ttnn.slice(input_tensor, starts, ends, [1] * len(ends))
            output = original_mmrs(chunk, weight, *args, **kwargs)
            plan = dict(last_mmrs_plan())
            resident = plan["xblock"] == "resident"
            thin_wide = resident and plan["orient"] == "m_on_y" and plan["grid"][0] <= 4 and plan["grid"][1] >= 10
            record(
                "mmrs",
                {
                    "role": role,
                    "original_rows": original_rows,
                    "padded_rows": padded_rows,
                    "execution_rows": end_row - start_row,
                    "chunk_count": math.ceil(padded_rows / chunk_rows),
                    "weight_shape": list(weight.shape),
                    "input_memory": str(chunk.memory_config()),
                    "output_memory": str(output.memory_config()),
                    "num_links": kwargs.get("num_links"),
                    "l1_headroom": kwargs.get("l1_headroom"),
                    "native_plan": plan,
                    "inferred_weight_read_noc": 1 if resident and not thin_wide else 0,
                    "thin_wide_branch": thin_wide,
                },
            )
            outputs.append(output)
        value = outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=-2)
        if padded_rows != original_rows:
            ends = list(value.shape)
            ends[-2] = original_rows
            value = ttnn.slice(value, [0] * len(ends), ends, [1] * len(ends))
        return value

    source = Path(__file__).resolve()
    root = source.parents[1]
    production = root / "tt/multichip_decoder.py"
    production_hash = hashlib.sha256(production.read_bytes()).hexdigest()
    command = shlex.join(
        [sys.executable, "-m", "models.autoports.qwen_qwen3_6_27b.tests.probe_multichip_prefill", *sys.argv[1:]]
    )
    previous_mtime = destination.output.stat().st_mtime_ns if destination.output.exists() else None
    argv = sys.argv
    failure = None
    try:
        sys.argv = [argv[0], *remaining]
        with (
            patch.object(ttnn.experimental, "minimal_matmul", minimal),
            patch.object(ttnn.experimental, "matmul_reduce_scatter_prefill", mmrs),
        ):
            runner.main()
    except BaseException:
        failure = traceback.format_exc()
        raise
    finally:
        sys.argv = argv
        produced = destination.output.exists() and destination.output.stat().st_mtime_ns != previous_mtime
        report = json.loads(destination.output.read_text()) if produced else {"pass": False}
        report["prefill_candidate"] = {
            "command": command,
            "options": vars(options),
            "candidate_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "decoder_source_sha256": production_hash,
            "matches_frozen_source": production_hash == FROZEN_SOURCE_SHA256,
            "observations": {kind: list(values.values()) for kind, values in observations.items()},
            "exception": failure,
            "scope": "test-only API arguments/activation placement/row padding/chunking; original runner oracle unchanged",
        }
        if failure is not None:
            report["pass"] = False
        destination.output.parent.mkdir(parents=True, exist_ok=True)
        destination.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({"candidate_artifact": str(destination.output), "pass": report.get("pass", False)}), flush=True
        )


if __name__ == "__main__":
    main()
