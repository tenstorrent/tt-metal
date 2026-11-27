# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run one prefill stage, layers [first, first + num_layers), from boundary planes saved on the host.

The stage is built exactly as the runner builds a pipeline rank (adapter.build_runtime, allocate_kv_cache, compile),
eager and on one mesh, but its inbound boundary is a directory instead of a D2D socket and its outbound boundary is a
directory too. No producer, H2D/D2H services, layer acks, KV chunk table or MPI: every sequence of the calibration set
is pushed through chunk by chunk in this process, so the stage builds its own KV cache, DSA index cache and KDA state as
it goes (slot 0, each sequence from position 0).

A boundary directory holds `manifest.json` and one raw bf16 file per plane, [rows, hidden] in natural token order, every
sequence padded to whole chunks and stored back to back. Plane 0 is the live stream, planes 1.. the sealed AttnRes
snapshots in seal order (adapter.pipeline_activation_planes). A sealed plane never changes once sealed, so an outbound
manifest points at the inbound file for every plane it passed through and writes only the live stream and the planes
sealed inside the stage.

Configuration is the runner's (PREFILL_MODEL / PREFILL_MANIFEST, PREFILL_SP / PREFILL_TP / PREFILL_SP_AXIS,
PREFILL_CHUNK_SIZE, PREFILL_MAX_SEQ_LEN, PREFILL_TTNN_CACHE, PREFILL_HF_MODEL, MOE_IMPL); set PREFILL_NUM_USERS=1 and
leave PREFILL_USE_TRACE and PREFILL_FORWARD_TOKENS_BUDGET unset.

    python -m models.demos.common.prefill.runners.stream_stage --first 0 --num-layers 12 \
        --tokens calib.pt --out /dev/shm/k31/b012
    python -m models.demos.common.prefill.runners.stream_stage --first 12 --num-layers 7 \
        --tokens calib.pt --inbound /dev/shm/k31/b012 --out /dev/shm/k31/b019

`calib.pt` is a list of 1-D int token tensors (the calibration sequences).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import ttnn
import models.demos.common.prefill.runners.prefill_runner as pr
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import PrefillRunParams


def _plane_file(path: Path, rows: int, hidden: int, mode: str) -> tuple[np.memmap, torch.Tensor]:
    """A [rows, hidden] bf16 plane backed by a raw file, mapped rather than read: (the map, its bf16 view)."""
    mapped = np.memmap(path, dtype=np.uint16, mode=mode, shape=(rows, hidden))
    return mapped, torch.from_numpy(mapped).view(torch.bfloat16)


def _mesh_dims() -> tuple:
    dims = [0, 0]
    dims[pr.SP_AXIS], dims[pr.TP_AXIS] = 2, 3  # rows over SP, hidden over TP, as the D2D activation
    return tuple(dims)


def _upload(mesh_device, host: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=_mesh_dims(), mesh_shape=tuple(mesh_device.shape)),
    )


def _download(mesh_device, tensor: ttnn.Tensor) -> torch.Tensor:
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=_mesh_dims(), mesh_shape=tuple(mesh_device.shape))
    return ttnn.to_torch(tensor, mesh_composer=composer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--first", type=int, required=True, help="global index of the stage's first layer")
    ap.add_argument("--num-layers", type=int, required=True)
    ap.add_argument("--tokens", type=Path, required=True, help="torch.save'd list of 1-D token tensors")
    ap.add_argument("--inbound", type=Path, help="boundary directory at --first (required unless --first is 0)")
    ap.add_argument("--out", type=Path, required=True, help="boundary directory to write at --first + --num-layers")
    args = ap.parse_args()
    # Stages must compose to the undivided forward bit for bit, so a stage that starts on an AttnRes block edge takes
    # the boundary read as the undivided walk does (a stage starting inside a block always does). Export
    # K3_ATTN_RES_EXACT_BOUNDARY=0 to get the pipeline runner's default closed-site read instead.
    os.environ.setdefault("K3_ATTN_RES_EXACT_BOUNDARY", "1")

    first, count = args.first, args.num_layers
    end = first + count
    assert (args.inbound is None) == (first == 0), "--inbound exactly when the stage does not start at layer 0"
    assert pr.FORWARD_CHUNKS == 1 and not pr.USE_TRACE, "the stage runs eager, one chunk per forward"
    chunk, hidden = pr.CHUNK_SIZE, pr.MODEL_CFG.EMB_SIZE
    planes_in = pr.ADAPTER.pipeline_activation_planes(first) if first else 0
    planes_out = pr.ADAPTER.pipeline_activation_planes(end)

    seqs = [t.flatten().tolist() for t in torch.load(args.tokens)]
    padded = [-(-len(s) // chunk) * chunk for s in seqs]
    total_rows = sum(padded)
    assert (
        max(len(s) for s in seqs) <= pr.MAX_SEQ_LEN
    ), f"a sequence is longer than PREFILL_MAX_SEQ_LEN={pr.MAX_SEQ_LEN}"

    inbound = []
    if first:
        manifest = json.loads((args.inbound / "manifest.json").read_text())
        assert (
            manifest["boundary"] == first and manifest["rows"] == total_rows
        ), f"{args.inbound} is not boundary {first}"
        assert (
            len(manifest["planes"]) == planes_in
        ), f"{len(manifest['planes'])} planes, layer {first} takes {planes_in}"
        inbound = [_plane_file(Path(p), total_rows, hidden, "r")[1] for p in manifest["planes"]]

    # Plane j of the outbound for 1 <= j < planes_in is inbound plane j, passed through untouched.
    args.out.mkdir(parents=True, exist_ok=True)
    fresh = [0] + list(range(max(planes_in, 1), planes_out))
    out_paths = [str((args.out / f"plane{j}.bf16").resolve()) for j in fresh]
    outbound = {j: _plane_file(Path(p), total_rows, hidden, "w+") for j, p in zip(fresh, out_paths)}

    mesh_device = pr.open_mesh_device(
        pr.GLOBAL_MESH_SHAPE, pr._FABRIC_PAYLOAD_SIZE, l1_small_size=pr._L1_SMALL_SIZE, sp_axis=pr.SP_AXIS
    )
    hf_config = pr.ADAPTER.load_hf_config()
    hf_config.max_seq_len = pr.MAX_SEQ_LEN
    params = PrefillRunParams(
        mesh_shape=pr.GLOBAL_MESH_SHAPE,
        num_layers=count,
        first_layer_idx=first,
        is_first_rank=first == 0,
        is_last_rank=False,  # always hand the planes off: the model-level read and norm are not wanted
        max_seq_len=pr.MAX_SEQ_LEN,
        chunk_size=chunk,
        num_users=pr.NUM_USERS,
        capacity_factor=pr.CAPACITY_FACTOR,
        num_links=2 if is_blackhole() else 1,
        gate_mode_name=pr._gate_mode_name,
        kv_only_last_layer=False,
        weight_cache_path=pr.ADAPTER.weight_cache_path(pr.GLOBAL_MESH_SHAPE, sp_axis=pr.SP_AXIS),
        tp_shard_kv=pr.TP_SHARD_KV,
        sp_axis=pr.SP_AXIS,
        tp_axis=pr.TP_AXIS,
        sparse_kv_cache_format=pr._sparse_kv_cache_format(),
        overlap_shared_expert_with_dispatch=os.environ.get("PREFILL_OVERLAP_SHARED_EXPERT", "1") == "1",
    )
    t0 = time.perf_counter()
    runtime = pr.ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    kv_caches = pr.ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    t1 = time.perf_counter()
    runtime.compile(kv_caches)
    t2 = time.perf_counter()
    logger.info(f"[stream] layers [{first}, {end}): build {t1 - t0:.1f}s, compile {t2 - t1:.1f}s")

    row = 0
    for s, ids in enumerate(seqs):
        for start in range(0, len(ids), chunk):
            rows = slice(row + start, row + start + chunk)
            if first == 0:
                tokens = ids[start : start + chunk]
                inp = runtime.make_chunk_input(tokens + [0] * (chunk - len(tokens)))
            else:
                inp = _upload(mesh_device, torch.stack([p[rows] for p in inbound]).unsqueeze(0))
            out = runtime.prefill_chunk(
                inp, kv_caches, slot_id=0, actual_start=start, actual_end=min(start + chunk, len(ids))
            )
            assert out.shape[1] == planes_out, f"stage handed off {out.shape[1]} planes, expected {planes_out}"
            for j, (_, dst) in outbound.items():
                plane = ttnn.slice(out, [0, j, 0, 0], [1, j + 1, out.shape[2], out.shape[3]])
                dst[rows] = _download(mesh_device, plane)[0, 0]
                ttnn.deallocate(plane)
            ttnn.deallocate(out)
        row += padded[s]
        logger.info(f"[stream] sequence {s}: {len(ids)} tokens, {time.perf_counter() - t2:.1f}s so far")
    t3 = time.perf_counter()

    for mapped, _ in outbound.values():
        mapped.flush()
    planes = [out_paths[fresh.index(j)] if j in outbound else manifest["planes"][j] for j in range(planes_out)]
    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "boundary": end,
                "rows": total_rows,
                "hidden": hidden,
                "chunk": chunk,
                "seq_lens": [len(s) for s in seqs],
                "planes": planes,
                "made_by": {"first": first, "num_layers": count, "cache": str(params.weight_cache_path)},
            },
            indent=1,
        )
    )
    logger.info(f"[stream] layers [{first}, {end}): forward + I/O {t3 - t2:.1f}s; wrote {args.out}")

    runtime.model.release_sub_device_managers()
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
