# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Read the Kimi-K3 KDA state back through a published KV chunk table and score it against the golden.

    python -m models.demos.deepseek_v3_d_p.tt.runners.kda_state_readback \
        --table /path/kv_chunk_table.pb --device-map /path/device_map[.rank0 ...] \
        --real-len 56320 [--golden DIR] [--num-layers 93] [--slot 0] [--pcc 0.99] [--json out.json]

Walks configs 1 (recurrent) and 2 (convolution) by model layer, reads every segment of every layer this
host can reach with ``read_dram_umd`` (a UMD read is local-PCIe only, so run once per host of a
multi-rank run), reassembles the per-layer state from the segment numbering alone, and compares it to
the head/tail golden snapshot after ``real_len`` tokens. Exit status is non-zero when any layer misses
the PCC bar or when no layer was reachable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import (
    KdaContractGeometry,
    assemble_convolution,
    assemble_recurrent,
    convolution_segment_to_torch,
    recurrent_segment_to_torch,
)
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter

GOLDEN_PERIOD = 1024
GOLDEN_FILES = {"kda_recurrent": "kda_recurrent_state_layer_{}", "kda_convolution": "kda_conv_state_layer_{}"}


def load_device_map(paths: list[str]) -> dict[tuple[int, int], int]:
    merged: dict[tuple[int, int], int] = {}
    for path in paths:
        with open(path) as handle:
            raw = json.load(handle)
        merged.update({tuple(int(x) for x in key.split(":")): int(uid) for key, uid in raw.items()})
    return merged


def resolve_unique_id(fabric_node_ids, device_map) -> int | None:
    for node in fabric_node_ids:
        key = (int(node.mesh_id), int(node.chip_id))
        if key in device_map:
            return device_map[key]
    return None


def load_golden(golden_dir: Path, kind: str, layer: int, row: int) -> torch.Tensor:
    from safetensors import safe_open

    name = GOLDEN_FILES[kind].format(layer)
    with safe_open(golden_dir / "kda" / f"{name}.safetensors", framework="pt") as handle:
        return handle.get_slice(name)[row : row + 1][0]


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.flatten().to(torch.float64)
    y = b.flatten().to(torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    return float((x @ y) / denominator) if denominator > 0 else float("nan")


def read_layer(table, config_id: int, kind: str, layer: int, slot: int, geometry, device_map):
    """All segments of one layer as a host tensor, or None when the layer lives on another host."""
    first = table.lookup(layer, 0, slot, config_id)
    if first.size_bytes == 0:
        return None, "unpublished"
    unique_id = resolve_unique_id(table.get_device_group(first.device_group_index).fabric_node_ids, device_map)
    if unique_id is None:
        return None, "remote"
    segments = {}
    total = table.config(config_id).max_sequence_length
    for segment in range(total):
        loc = table.lookup(layer, segment, slot, config_id)
        uid = resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        if uid is None:
            return None, f"segment {segment} on a remote device"
        raw = ttnn.experimental.disaggregation.read_dram_umd(uid, loc.noc_addr, loc.size_bytes)
        if kind == "kda_recurrent":
            segments[segment] = recurrent_segment_to_torch(raw, geometry)
        else:
            segments[segment] = convolution_segment_to_torch(raw, geometry)
    if kind == "kda_recurrent":
        return assemble_recurrent(segments, geometry), "ok"
    return assemble_convolution(segments, geometry), "ok"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True)
    parser.add_argument("--device-map", required=True, nargs="+")
    parser.add_argument("--real-len", type=int, required=True, help="prefilled tokens (multiple of 1024)")
    parser.add_argument("--golden", default=os.environ.get("PREFILL_KDA_GOLDEN_DIR", KimiK3Adapter.kda_golden_default))
    parser.add_argument(
        "--num-layers", type=int, default=int(os.environ.get("PREFILL_NUM_LAYERS", KimiK3Config.NUM_LAYERS))
    )
    parser.add_argument("--slot", type=int, default=0)
    parser.add_argument("--pcc", type=float, default=0.99)
    parser.add_argument("--sp", type=int, default=int(os.environ.get("PREFILL_SP", 8)))
    parser.add_argument("--tp", type=int, default=int(os.environ.get("PREFILL_TP", 4)))
    parser.add_argument("--json", default=None, help="write the per-layer results here")
    args = parser.parse_args(argv)

    if args.real_len % GOLDEN_PERIOD:
        parser.error(f"--real-len {args.real_len} is not a multiple of the golden period {GOLDEN_PERIOD}")
    row = args.real_len // GOLDEN_PERIOD - 1
    golden_dir = Path(args.golden)
    adapter = KimiK3Adapter()
    geometry = KdaContractGeometry.from_kda_config(
        kimi_k3_kda_config(), mesh_shape=(args.sp, args.tp), sp_axis=0, tp_axis=1
    )

    table = ttnn.experimental.disaggregation.import_from_protobuf_file(args.table)
    device_map = load_device_map(args.device_map)
    logger.info(
        f"table {args.table}: {table.num_configs()} configs; device map {len(device_map)} chips; golden row {row}"
    )

    results = []
    worst = {}
    for config_id in range(table.num_configs()):
        kind = adapter.cache_kind(config_id)
        if kind not in GOLDEN_FILES:
            continue
        for layer in sorted(adapter.cache_layer_rows(config_id, args.num_layers)):
            state, status = read_layer(table, config_id, kind, layer, args.slot, geometry, device_map)
            if state is None:
                results.append({"config": config_id, "kind": kind, "layer": layer, "status": status})
                continue
            golden = load_golden(golden_dir, kind, layer, row)
            if kind == "kda_recurrent":
                golden = golden.transpose(-1, -2)  # golden is [heads, v, k]; the device stores [k, v]
            score = pcc(state, golden)
            results.append({"config": config_id, "kind": kind, "layer": layer, "status": "ok", "pcc": score})
            worst[kind] = min(worst.get(kind, 1.0), score)
            logger.info(f"config {config_id} {kind:16s} layer {layer:3d}: pcc {score:.6f}")

    checked = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] != "ok"]
    logger.info(
        f"checked {len(checked)} layer states, skipped {len(skipped)} ({', '.join(sorted({r['status'] for r in skipped})) or 'none'})"
    )
    for kind, score in worst.items():
        logger.info(f"min pcc {kind}: {score:.6f}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(
                {"real_len": args.real_len, "golden_row": row, "min_pcc": worst, "layers": results}, handle, indent=2
            )
    if not checked:
        logger.error("no KDA layer was reachable from this host")
        return 2
    failed = [r for r in checked if r["pcc"] < args.pcc]
    if failed:
        logger.error(
            f"{len(failed)} layer state(s) below {args.pcc}: {[(r['kind'], r['layer'], round(r['pcc'], 4)) for r in failed]}"
        )
        return 1
    logger.success(f"KDA state read-back PASS: {len(checked)} layer states >= {args.pcc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
