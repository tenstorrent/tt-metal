# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The c1 decode MoE (``tt/v41/moe_c1.py``, ``V41_DECODE_C1_MOE=1``) against the batched step's 32-row MoE on one
live row.

One MoE layer (``V41_C1_LAYER``, default 2) from the weight cache, built as the batched decoder builds it (the
decode routing and tail on), fed ``V41_C1_TRIALS`` random tiles (default 8) with a random live (rank, row): both
paths on the same input, the live row's output compared over the full hidden width (bf16 tolerance: the row's bfp8
packing and the fp32 order of the weighted sum are the only differences), the c1 path's idle rows zero. The
per-chip expert counts of every trial are read back so the run covers chips with 0, 1, 2, .. of the row's
experts. Then both paths under trace (``V41_C1_REPLAYS``, default 20): the per-layer device time.

Run with ``V41_DECODE_ROUTING=1 V41_DECODE_TAIL=1 V41_DECODE_C1_MOE=1 TT_V41_EXPERT_DTYPE=bfp4`` (the served
expert dtype)."""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v41 import oracle as orc
from models.demos.deepseek_v3_d_p.reference.deepseek_v41_flash_config import DeepSeekV41FlashConfig as C
from models.demos.deepseek_v3_d_p.tests.v41.galaxy_meshes import galaxy_meshes
from models.demos.deepseek_v3_d_p.tests.v41.weight_cache import weight_cache_dir
from models.demos.deepseek_v3_d_p.tt.v41.batch_decode import TtV41BatchDecoder
from models.demos.deepseek_v3_d_p.tt.v41.decode import ROWS
from models.demos.deepseek_v3_d_p.tt.v41.moe_c1 import C1Moe
from models.demos.deepseek_v3_d_p.tt.v41.transformer import TtV41Transformer
from models.demos.deepseek_v3_d_p.tt.v41.weights import load_layer, load_layer_dense, resolve_checkpoint

try:
    from tracy import signpost
except ModuleNotFoundError:  # not under the tracy wrapper

    def signpost(_name: str) -> None:
        pass


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _row(t, rank: int, u: int, tp: int) -> torch.Tensor:
    """Row ``u`` of rank ``rank``'s tile, the TP shards concatenated -> ``[hidden]`` fp32."""
    chips = ttnn.get_device_tensors(t)
    return torch.cat([ttnn.to_torch(chips[rank * tp + c])[0, 0, u].float() for c in range(tp)])


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "mesh_device, device_params", galaxy_meshes(trace_region_size=256 * 1024 * 1024), indirect=True
)
def test_v41_moe_c1(mesh_device, device_params):
    assert os.environ.get("V41_DECODE_C1_MOE") == "1", "run with V41_DECODE_C1_MOE=1 (and the decode routing)"
    layer = int(os.environ.get("V41_C1_LAYER", 2))
    trials = int(os.environ.get("V41_C1_TRIALS", 8))
    replays = int(os.environ.get("V41_C1_REPLAYS", 20))
    users = 4
    ckpt = resolve_checkpoint()
    if ckpt is None:
        pytest.skip("V4.1 checkpoint shards not downloaded")
    sp, tp = tuple(mesh_device.shape)
    max_seq = 1024
    spec = orc.real_spec((layer,), max_seq, checkpoint=ckpt.root)
    root = weight_cache_dir(spec, mesh_device.shape)
    top = ckpt.read(["embed.weight", "norm.weight", "head.weight"])
    t0 = time.time()
    model = TtV41Transformer(
        mesh_device,
        C,
        [layer],
        lambda l, include_moe: (load_layer if include_moe else load_layer_dense)(ckpt, l),
        top["embed.weight"],
        top["norm.weight"],
        top["head.weight"],
        max_seq_len=max_seq,
        chunk=1024,
        weight_cache_path=root,
    )
    decoder = TtV41BatchDecoder(model, root, users_per_rank=users, trace=False)
    logger.info(f"c1 moe: layer {layer} built in {time.time() - t0:.1f} s")
    # one prefill first, as every served or tested flow has: without it the decode tail's ``ttnn.reduce_scatter``
    # leaves a 512-byte L1 block at 914,176 after its first call and every later routed-expert FFN program (the
    # 32-row one included) is rejected at compile against it
    torch.manual_seed(3)
    prompt = torch.randint(1000, 20000, (C.SLIDING_WINDOW + 7,))
    _, state = model.prefill(prompt, logit_positions=1)
    del state  # its device tensors stay allocated (DRAM); a unit test
    moe, c1 = decoder.moes[0], decoder.c1_moes[0]
    hidden = C.EMB_SIZE
    torch.manual_seed(7)
    result = {"layer": layer, "trials": [], "mesh": [sp, tp]}
    worst = {"max_abs": 0.0, "pcc": 1.0, "rel": 0.0}
    coverage = {}
    for trial in range(trials):
        h = torch.randn(sp, ROWS, hidden) * 1.0
        tt_h = ttnn.from_torch(
            h.view(1, 1, sp * ROWS, hidden),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(2, 3)),
        )
        r, u = int(torch.randint(sp, (1,))), int(torch.randint(users, (1,)))
        rows = {
            k: decoder.pool.per_rank(v.view(sp, 1, 1, ROWS), ttnn.uint32)
            for k, v in C1Moe.host_rows(sp, ROWS, (r, u)).items()
        }
        sel, put = C1Moe.onehots(rows["c1sel"], rows["c1put"], decoder.c1_table)
        c1.debug = {}
        ref = decoder._ffn(moe, tt_h)
        out = c1(tt_h, sel, put)
        if trial == trials - 1 and os.environ.get("V41_C1_PROFILE") == "1":
            # With device profiling enabled: one eager call of each path between signposts (the per-op kernel split)
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            ttnn.deallocate(decoder._ffn(moe, tt_h))
            ttnn.synchronize_device(mesh_device)
            signpost("c1")
            ttnn.deallocate(c1(tt_h, sel, put))
            ttnn.synchronize_device(mesh_device)
            signpost("stop")
        counts = [int(ttnn.to_torch(t).sum()) for t in ttnn.get_device_tensors(c1.debug["counts"])]
        idx = ttnn.to_torch(ttnn.get_device_tensors(c1.debug["indices"])[0])[0, :6].tolist()
        c1.debug = None
        ref_row, out_row = _row(ref, r, u, tp), _row(out, r, u, tp)
        max_abs = float((ref_row - out_row).abs().max())
        rel = max_abs / max(float(ref_row.abs().max()), 1e-6)
        pcc = _pcc(ref_row, out_row)
        # every other row of the c1 output is zero on every chip
        idle_max = 0.0
        for i, chip in enumerate(ttnn.get_device_tensors(out)):
            t = ttnn.to_torch(chip)[0, 0].float()
            if i // tp == r:
                t = torch.cat([t[:u], t[u + 1 :]])
            idle_max = max(idle_max, float(t.abs().max()))
        for n in counts:
            coverage[n] = coverage.get(n, 0) + 1
        entry = {
            "trial": trial,
            "live": [r, u],
            "experts": idx,
            "chip_counts": counts,
            "max_abs": round(max_abs, 5),
            "rel": round(rel, 6),
            "pcc": round(pcc, 6),
            "ref_absmax": round(float(ref_row.abs().max()), 4),
            "idle_max": idle_max,
        }
        result["trials"].append(entry)
        logger.info(f"c1 moe trial {json.dumps(entry)}")
        worst = {
            "max_abs": max(worst["max_abs"], max_abs),
            "pcc": min(worst["pcc"], pcc),
            "rel": max(worst["rel"], rel),
        }
        assert sum(counts) == 6, f"the row's 6 experts over the chips: {counts}"
        assert idle_max == 0.0, f"the c1 output's idle rows are not zero: {idle_max}"
        for t in (ref, out, tt_h, sel, put):
            ttnn.deallocate(t)
    result["worst"] = {k: round(v, 6) for k, v in worst.items()}
    result["coverage"] = {str(k): v for k, v in sorted(coverage.items())}
    # the per-layer device time of each path, under trace
    h = torch.randn(sp, ROWS, hidden)
    tt_h = ttnn.from_torch(
        h.view(1, 1, sp * ROWS, hidden),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (sp, tp), dims=(2, 3)),
    )
    rows = {
        k: decoder.pool.per_rank(v.view(sp, 1, 1, ROWS), ttnn.uint32)
        for k, v in C1Moe.host_rows(sp, ROWS, (1, 2)).items()
    }
    sel, put = C1Moe.onehots(rows["c1sel"], rows["c1put"], decoder.c1_table)
    timing = {}
    paths = (("full", lambda: decoder._ffn(moe, tt_h)), ("c1", lambda: c1(tt_h, sel, put)))
    for name, fn in () if os.environ.get("V41_C1_PROFILE") == "1" else paths:
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        outs = [fn() for _ in range(4)]
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        best = None
        for _ in range(replays):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            dt = (time.perf_counter() - t0) / 4
            best = dt if best is None else min(best, dt)
        ttnn.release_trace(mesh_device, tid)
        for o in outs:
            ttnn.deallocate(o)
        timing[name] = round(best * 1e6, 1)
        logger.info(f"c1 moe traced {name}: {timing[name]} us a layer (min of {replays} replays of 4 calls)")
    result["layer_us"] = timing
    logger.info(f"V41_C1_MOE_RESULT {json.dumps(result)}")
    out_path = os.environ.get("V41_C1_OUT")
    if out_path:
        with open(os.path.expanduser(out_path), "w") as f:
            json.dump(result, f, indent=1)
    decoder.release()
    assert worst["pcc"] > 0.999 and worst["rel"] < 0.02, worst
