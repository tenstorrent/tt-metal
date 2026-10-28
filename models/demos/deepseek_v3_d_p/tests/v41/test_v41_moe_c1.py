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
expert dtype); with the served catalog's ``V41_SHARED_IN_FFN=rs`` the 32-row path folds the
shared expert's partial into its tail and the c1 path takes that partial from the tail's ``pending_shared``
(``C1Moe.shared_fold`` reads ``pending``; ``returned`` with the MoE's own shared expert), and the test checks the
tail holds nothing after either path.

``V41_C1_ROWS=n`` (default 1): the c1 MoE on ``n`` live rows of one user (``C1Moe(rows=n)``, the speculative verify
step's ``1 + K`` rows): rows ``u .. u + n - 1`` of the live rank against the 32-row path, the
idle rows zero, and one traced call replayed against the eager call on the same input (bit for bit)."""

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


def _oracle_row(moe, c1, x: torch.Tensor, experts: list, weights: torch.Tensor, tp: int) -> torch.Tensor:
    """The MoE of one row in fp64 from the device's own weights (bfp4 experts, bfp8 shared expert, dequantized by
    ``to_torch``): ``sum_k w_k expert_k(x) + shared(x)``, ``x`` the experts' input row (after the FP8 QDQ)."""
    routed = c1.routed
    two_d = lambda t: t.reshape(*t.shape[-2:]).double()
    hidden = x.shape[0]
    limit = float(
        moe.moe.shared_expert.clamped_silu_glu_limit
        if hasattr(moe.moe.shared_expert, "clamped_silu_glu_limit")
        else getattr(moe.moe.shared_expert, "_shared").clamped_silu_glu_limit
    )

    def ffn(g, u, d):
        inter = d.shape[0]
        gate, up = x @ g[:, :inter], x @ u[:, :inter]
        act = torch.nn.functional.silu(torch.clamp(gate, max=limit)) * torch.clamp(up, -limit, limit)
        return act @ d[:, :hidden]

    tables = [ttnn.to_torch(t).flatten().tolist() for t in ttnn.get_device_tensors(c1.table)]
    out = torch.zeros(hidden, dtype=torch.float64)
    for k, e in enumerate(experts):
        chip = next(i for i, tab in enumerate(tables) if e in tab)
        l = tables[chip].index(e)
        g = two_d(ttnn.to_torch(ttnn.get_device_tensors(routed.gate_projs[l])[chip]))
        u = two_d(ttnn.to_torch(ttnn.get_device_tensors(routed.up_projs[l])[chip]))
        d = two_d(ttnn.to_torch(ttnn.get_device_tensors(routed.down_projs[l])[chip]))
        out += weights[k] * ffn(g, u, d)
    shared = getattr(moe.moe.shared_expert, "_shared", moe.moe.shared_expert)
    for c in range(tp):  # the shared expert's TP shards of SP rank 0
        g = two_d(ttnn.to_torch(ttnn.get_device_tensors(shared.gate_proj)[c]))
        u = two_d(ttnn.to_torch(ttnn.get_device_tensors(shared.up_proj)[c]))
        d = two_d(ttnn.to_torch(ttnn.get_device_tensors(shared.down_proj)[c]))
        out += ffn(g, u, d)
    return out


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
    n_rows = int(os.environ.get("V41_C1_ROWS", 1))
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
    decoder_cls = type("RowsDecoder", (TtV41BatchDecoder,), {"C1_ROWS": n_rows}) if n_rows > 1 else TtV41BatchDecoder
    decoder = decoder_cls(model, root, users_per_rank=users, trace=False)
    logger.info(f"c1 moe: layer {layer} built in {time.time() - t0:.1f} s; {n_rows} live rows a call")
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
    shared_in_ffn = os.environ.get("V41_SHARED_IN_FFN", "0")
    shared_mode = os.environ.get("V41_DECODE_C1_SHARED", "fold")
    result = {
        "layer": layer,
        "trials": [],
        "mesh": [sp, tp],
        "shared_in_ffn": shared_in_ffn,
        "c1_shared": shared_mode,
        "c1_select": os.environ.get("V41_DECODE_C1_SELECT", "fresh"),
        "c1_ffn": os.environ.get("V41_DECODE_C1_FFN", "composite"),
        "c1_ffn_fidelity": os.environ.get("V41_DECODE_C1_FFN_FIDELITY", "lofi"),
        "rows": n_rows,
    }

    def host_rows(r: int, u: int) -> dict:
        """``C1Moe.host_rows`` for ``n_rows`` rows ``u .. u + n_rows - 1`` of rank ``r`` (rows 0 .. n - 1 of the
        c1 tile and back; the verify decoder's layout)."""
        if n_rows == 1:
            return C1Moe.host_rows(sp, ROWS, (r, u))
        from models.demos.deepseek_v3_d_p.tt.v41.moe_c1 import _SENTINEL

        sel = torch.full((sp, ROWS), _SENTINEL, dtype=torch.int32)
        put = torch.full((sp, ROWS), _SENTINEL, dtype=torch.int32)
        for j in range(n_rows):
            sel[r, j], put[r, u + j] = u + j, j
        return {"c1sel": sel, "c1put": put}

    def rows_of(t, r: int, u: int) -> torch.Tensor:
        return torch.stack([_row(t, r, u + j, tp) for j in range(n_rows)])

    if shared_mode == "split":
        # the split weights: every chip's piece bit for bit the host slice of its TP shard
        split = c1.shared_split
        shared = getattr(moe.moe.shared_expert, "_shared", moe.moe.shared_expert)
        two_d = lambda t: t.reshape(*t.shape[-2:])
        for name, full, piece, axis in (
            ("gate", shared.gate_proj, split.gate, -1),
            ("up", shared.up_proj, split.up, -1),
            ("down", shared.down_proj, split.down, -2),
        ):
            full_chips, piece_chips = ttnn.get_device_tensors(full), ttnn.get_device_tensors(piece)
            for i, (fc, pc) in enumerate(zip(full_chips, piece_chips)):
                s_rank = i // tp
                lo, n = ROWS * split.offsets[s_rank], ROWS * split.counts[s_rank]
                want = torch.zeros((split.hidden, split.width) if axis == -1 else (split.width, split.hidden))
                src = two_d(ttnn.to_torch(fc).float())
                if axis == -1:
                    want[:, :n] = src[:, lo : lo + n]
                else:
                    want[:n] = src[lo : lo + n]
                got = two_d(ttnn.to_torch(pc).float())
                assert torch.equal(want, got), f"{name} split on chip {i}: {int((want != got).sum())} words differ"
        result["split"] = {
            "width": split.width,
            "counts": split.counts,
            "offsets": split.offsets,
            "exact": True,
        }
        logger.info(f"c1 shared split: width {split.width}, tiles per SP rank {split.counts}, bit-exact on 32 chips")
    worst = {"max_abs": 0.0, "pcc": 1.0, "rel": 0.0}
    coverage = {}
    use_oracle = os.environ.get("V41_C1_ORACLE", "1" if result["c1_ffn"] == "matvec" else "0") == "1" and n_rows == 1
    oracle_worst = {"full": 0.0, "c1": 0.0}
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
        r, u = int(torch.randint(sp, (1,))), int(torch.randint(users - n_rows + 1, (1,)))
        rows = {k: decoder.pool.per_rank(v.view(sp, 1, 1, ROWS), ttnn.uint32) for k, v in host_rows(r, u).items()}
        sel, put = C1Moe.onehots(rows["c1sel"], rows["c1put"], decoder.c1_table)
        c1.debug = {}
        ref = decoder._ffn(moe, tt_h)
        assert moe.moe.reduce_module.pending_shared is None, "the 32-row path left a shared partial in the tail"
        out = c1(tt_h, sel, put, n_rows)
        assert moe.moe.reduce_module.pending_shared is None, "the c1 path left a shared partial in the tail"
        expect_fold = "split" if shared_mode == "split" else ("pending" if shared_in_ffn != "0" else "returned")
        assert c1.shared_fold == expect_fold, (c1.shared_fold, expect_fold)
        result["shared_fold"] = c1.shared_fold
        if trial == trials - 1 and os.environ.get("V41_C1_PROFILE") == "1":
            # With device profiling enabled: one eager call of each path between signposts (the per-op kernel split)
            ttnn.synchronize_device(mesh_device)
            signpost("start")
            ttnn.deallocate(decoder._ffn(moe, tt_h))
            ttnn.synchronize_device(mesh_device)
            signpost("c1")
            ttnn.deallocate(c1(tt_h, sel, put, n_rows))
            ttnn.synchronize_device(mesh_device)
            signpost("stop")
        counts = [int(ttnn.to_torch(t).sum()) for t in ttnn.get_device_tensors(c1.debug["counts"])]
        if c1.W0 is not None:
            # the persistent weight tile against a fresh one from the same indices and weights, every chip
            from models.demos.deepseek_v3_d_p.tt.v41.moe_c1 import c1_select

            fresh_counts, fresh_w = c1_select(c1.debug["indices"], c1.debug["weights"], c1.table, c1.E, rows=n_rows)
            for i, (a, b) in enumerate(zip(ttnn.get_device_tensors(fresh_w), ttnn.get_device_tensors(c1.W0))):
                assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), f"persistent W differs on chip {i}"
            for i, (a, b) in enumerate(
                zip(ttnn.get_device_tensors(fresh_counts), ttnn.get_device_tensors(c1.debug["counts"]))
            ):
                assert torch.equal(ttnn.to_torch(a), ttnn.to_torch(b)), f"counts differ on chip {i}"
            ttnn.deallocate(fresh_counts)
            ttnn.deallocate(fresh_w)
        idx = ttnn.to_torch(ttnn.get_device_tensors(c1.debug["indices"])[0])[0, :6].tolist()
        ref_row, out_row = rows_of(ref, r, u), rows_of(out, r, u)
        per_row = {}
        if n_rows > 1:
            # per row: the c1 row against the 32-row path's, and a second eager call on the same input against
            # the first (determinism); the rows' expert ids and the chip counts
            again = c1(tt_h, sel, put, n_rows)
            again_rows = rows_of(again, r, u)
            ttnn.deallocate(again)
            all_idx = ttnn.to_torch(ttnn.get_device_tensors(c1.debug["indices"])[0])[:n_rows, :6].tolist()
            per_row = {
                "pcc": [round(_pcc(ref_row[t], out_row[t]), 5) for t in range(n_rows)],
                "max_abs": [round(float((ref_row[t] - out_row[t]).abs().max()), 4) for t in range(n_rows)],
                "again_max_abs": [round(float((again_rows[t] - out_row[t]).abs().max()), 4) for t in range(n_rows)],
                "experts": all_idx,
            }
            logger.info(f"c1 moe rows {n_rows} per row {json.dumps(per_row)}")
            # the same call captured and replayed: the replay's rows must equal the eager call's (the verify
            # step's rounds replay it)
            ttnn.synchronize_device(mesh_device)
            tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            replayed = c1(tt_h, sel, put, n_rows)
            ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            eager_rows = rows_of(replayed, r, u)  # the capture ran the ops once
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
            replay_rows = rows_of(replayed, r, u)
            ttnn.release_trace(mesh_device, tid)
            ttnn.deallocate(replayed)
            n_diff = int((eager_rows != replay_rows).sum())
            entry_replay = {
                "captured_vs_eager_max_abs": float((eager_rows - out_row).abs().max()),
                "replay_diff_words": n_diff,
            }
            logger.info(f"c1 moe rows {n_rows} replay check {entry_replay}")
            result.setdefault("replay_diff_words", []).append(n_diff)
        max_abs = float((ref_row - out_row).abs().max())
        rel = max_abs / max(float(ref_row.abs().max()), 1e-6)
        pcc = _pcc(ref_row, out_row)
        oracle = {}
        if use_oracle:
            # the fp64 MoE of the live row from the device's own (dequantized) weights: which path is nearer it
            wts = ttnn.to_torch(ttnn.get_device_tensors(c1.debug["weights"])[0])[0, :6].double()
            x_full = ttnn.to_torch(ttnn.get_device_tensors(c1.debug["x_full"])[0])[0, 0, 0].double()
            exact = _oracle_row(moe, c1, x_full, idx, wts, tp)
            for name, row in (("full", ref_row), ("c1", out_row)):
                d = float((exact - row.double()).abs().max())
                oracle[name] = {
                    "rel": round(d / max(float(exact.abs().max()), 1e-6), 6),
                    "pcc": round(_pcc(exact.float(), row), 6),
                }
            ttnn.deallocate(c1.debug["x_full"])
        ttnn.deallocate(c1.debug["weights"])
        c1.debug = None
        # every other row of the c1 output is zero on every chip
        idle_max = 0.0
        for i, chip in enumerate(ttnn.get_device_tensors(out)):
            t = ttnn.to_torch(chip)[0, 0].float()
            if i // tp == r:
                t = torch.cat([t[:u], t[u + n_rows :]])
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
            "oracle": oracle,
            "per_row": per_row,
        }
        result["trials"].append(entry)
        logger.info(f"c1 moe trial {json.dumps(entry)}")
        worst = {
            "max_abs": max(worst["max_abs"], max_abs),
            "pcc": min(worst["pcc"], pcc),
            "rel": max(worst["rel"], rel),
        }
        for name in oracle:
            oracle_worst[name] = max(oracle_worst[name], oracle[name]["rel"])
        if n_rows == 1:
            assert sum(counts) == 6, f"the row's 6 experts over the chips: {counts}"
        else:
            assert sum(counts) % n_rows == 0 and 6 * n_rows <= sum(counts) <= 6 * n_rows * n_rows, counts
        assert idle_max == 0.0, f"the c1 output's idle rows are not zero: {idle_max}"
        for t in (ref, out, tt_h, sel, put):
            ttnn.deallocate(t)
    result["worst"] = {k: round(v, 6) for k, v in worst.items()}
    if use_oracle:
        result["oracle_worst_rel"] = oracle_worst
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
    rows = {k: decoder.pool.per_rank(v.view(sp, 1, 1, ROWS), ttnn.uint32) for k, v in host_rows(1, 0).items()}
    sel, put = C1Moe.onehots(rows["c1sel"], rows["c1put"], decoder.c1_table)
    c1.debug = {}
    ttnn.deallocate(c1(tt_h, sel, put, n_rows))
    timing_counts = [int(ttnn.to_torch(t).sum()) for t in ttnn.get_device_tensors(c1.debug["counts"])]
    for k in ("weights", "x_full"):
        if k in c1.debug:
            ttnn.deallocate(c1.debug[k])
    c1.debug = None
    result["timing_chip_counts"] = timing_counts  # the timed row's experts per chip (the busiest chip is the layer)
    logger.info(f"c1 moe timing row: experts per chip {timing_counts}")
    timing = {}
    paths = (("full", lambda: decoder._ffn(moe, tt_h)), ("c1", lambda: c1(tt_h, sel, put, n_rows)))
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
    assert not any(
        result.get("replay_diff_words", [])
    ), f"replayed calls differ from eager: {result['replay_diff_words']}"
    assert worst["pcc"] > 0.999, worst
    if use_oracle and worst["rel"] >= 0.02:
        # a path nearer the exact MoE than the 32-row path may differ from that path by more than the bf16 bar
        assert oracle_worst["c1"] <= oracle_worst["full"], oracle_worst
    else:
        assert worst["rel"] < 0.02, worst
