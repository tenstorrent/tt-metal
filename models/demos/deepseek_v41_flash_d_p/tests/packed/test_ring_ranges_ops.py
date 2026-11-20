# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7C: ring_indexer_score_dsa's lane_ranges (ONE ring call, a device holding query rows of several segments) on the
model's mesh, op level, against the PB.7B path the glued oracle proved bit-identical to a segment alone.

Random queries, gates and keys (the model's indexer's dtypes, shapes, ring and program settings; the held model is used
only for those objects: no layer runs). Each case is a glued layout on the mesh's chips (R rows a chip): segments back
to back at tile rows, each with its own key buffer (one capacity), start and kv_len. The device side: ONE call with
lane_ranges (q_chunk 32). The reference, per (chip, range): PB.7B's per-chip lanes call (lane_rows = R, the range's
rows rolled to row 0 ON THE HOST, lane start = the position of its first row, q_chunk = the model's), the range's rows
compared bit-exactly over its kv_len columns (and the candidate indexer's block maxes when the layer has them).

Cases: two segments meeting inside one chip at every tile-row offset (the second a fresh request at position 0 and a
continuing one); short (start 0 / 2048) and long histories (up to the capacity); six segments in one ring.

    <the walls env> scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_ring_ranges_ops.py
"""

from __future__ import annotations

import random

import pytest
import torch

from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
BATCH_SHARED = True
BATCH_LAYERS = "executed"
RANGE_STRIDE = 41  # ttnn.bringup ring_indexer_score_dsa lane_ranges: words a device
RANGES_MAX = 8


def _layout(R: int, n: int, segs: list) -> list:
    """segs [(rows, start, kv_len)] back to back from model row 0 -> per chip [(j, a, b, position of row a)]."""
    chips = [[] for _ in range(n)]
    off = 0
    for j, (k, start, _) in enumerate(segs):
        for q in range(off // R, (off + k - 1) // R + 1):
            a, b = max(off, q * R) - q * R, min(off + k, (q + 1) * R) - q * R
            chips[q].append((j, a, b, start + q * R + a - off))
        off += k
    return chips


def _ranges(chips: list, segs: list) -> list:
    words = []
    for ps in chips:
        w = [len(ps)]
        for j, a, b, p in ps:
            w += [a, b, j, p, segs[j][2]]
        words += w + [0] * (RANGE_STRIDE - len(w))
    return words


@mesh_parametrize(timeout_s=3600)
def test_ring_ranges_ops(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tt.common import SP_AXIS, TP_AXIS
    from models.demos.deepseek_v41_flash_d_p.tt.indexer import K_CHUNK, TtCandidateIndexer, TtIndexer, TtReindexer

    full = list(S.layers())
    layers = batch.run_layers(S, full, last_chunk=False)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    mesh = model.mesh
    sp, tp = int(mesh.shape[SP_AXIS]), int(mesh.shape[TP_AXIS])
    n = sp * tp
    picks = {}
    for i in model.layers:
        ix = getattr(model.blocks[i], "indexer", None)
        if ix is None or isinstance(ix, TtReindexer):
            continue
        picks.setdefault(type(ix), ix)
    assert TtIndexer in picks or TtCandidateIndexer in picks, "no indexer layer held"
    rnd = random.Random(7)
    torch.manual_seed(7)
    fails, checked = [], 0
    for kind, ix in picks.items():
        R, nh, hd, ratio = ix.sq, ix.nh, ix.hd, ix.ratio
        assert R * n == model.chunk, (R, n, model.chunk)
        T = 65536  # (keys of a 262144-token slot: the op's geometry is the capacity's; a small one is quick)
        cand = kind is TtCandidateIndexer
        cfg32 = ttnn.bringup.IndexerScoreProgramConfig(
            q_chunk_size=32,
            k_chunk_size=K_CHUNK,
            head_group_size=0,
            max_grid_x=ix.score_cfg.max_grid_x,
            max_grid_y=ix.score_cfg.max_grid_y,
        )
        q_dtype = ttnn.bfloat4_b if ix.q_bfp4 else ttnn.bfloat16

        def up(t, dtype, dims):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(sp, tp), dims=dims),
            )

        nbuf = 6
        k_locals = [up(torch.randn(sp, 1, T // sp, hd), ix.k_dtype, (0, None)) for _ in range(nbuf)]
        kbufs = [
            ttnn.from_torch(
                torch.zeros(1, 1, T, hd),
                dtype=ix.k_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            for _ in range(nbuf)
        ]
        bm_shape = [1, 1, R, -(-(T // ix.block) // 32) * 32] if cand else None
        cap_tokens = T * ratio
        cases = []
        for a in range(32, R, 32):  # two segments meeting inside chip 5 at row a
            for bstart in (0, 4096):
                rows0 = 5 * R + a
                s0 = rnd.randrange(0, (cap_tokens - 8192) // 32) * 32
                cases.append(("meet", a, [(rows0, s0), (model.chunk - rows0, bstart)]))
        cases.append(("short", 0, [(1024, 0), (2048, 2048), (96, 32), (5024, 64)]))
        cases.append(("long", 0, [(3008, cap_tokens - 8192), (2048, 139264), (3136, (cap_tokens // 2) // 64 * 64)]))
        cases.append(("six", 0, [(1600, 139264), (32, 0), (2464, 4096), (800, 8192), (1504, 61440), (1792, 300032)]))

        def scramble():
            """Every key buffer (gathered and local) refilled with new random keys: a read of a buffer the ring did
            not (yet) write, or of the wrong buffer, then differs (the reference's buffers are the call's own)."""
            for b in range(nbuf):
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(
                        torch.randn(1, 1, T, hd),
                        dtype=ix.k_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                    ),
                    kbufs[b],
                )

        for name, a, plan in cases:
            if name != "meet" or a % 96 == 32:
                scramble()
            k_locals_all = k_locals
            k_locals = [k_locals_all[(b + checked) % nbuf] for b in range(nbuf)]  # (another buffer order a case)
            segs = []
            for k, start in plan:
                start = min(start, cap_tokens - 8192)
                kv = min(T, -(-((start + model.chunk) // ratio) // 32) * 32)
                segs.append((k, start, kv))
            used = sum(k for k, _, _ in segs)
            assert used <= model.chunk, (name, used)
            chips = _layout(R, n, segs)
            for q in range(n):  # (the tail pad rows: the first segment's keys at start 0, never compared)
                if not chips[q]:
                    chips[q] = [(0, 0, R, 0)]
                elif chips[q][-1][2] < R:
                    chips[q].append((0, chips[q][-1][2], R, 0))
            words = _ranges(chips, segs)
            kvmax = max(kv for _, _, kv in segs)
            qh = torch.randn(n, nh, R, hd)
            wh = torch.randn(n, 1, R, nh)
            qd = up(qh.view(sp, tp * nh, R, hd), q_dtype, (0, 1))
            wd = up(wh.view(sp, tp, R, nh), ttnn.bfloat16, (0, 1))
            js = list(range(len(segs)))
            ep, epr = {}, {}
            if cand:
                bm = ttnn.allocate_tensor_on_device(
                    ttnn.Shape(bm_shape), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
                )
                ep = dict(block_keys=ix.block, block_max_out=bm, block_max_pin=True)
            common = dict(
                cluster_axis=SP_AXIS,
                topology=ix.topology,
                num_links=ix.num_links,
                key_compression_ratio=ratio,
                compute_kernel_config=ix.score_ckc,
                seq_subshard_axis=TP_AXIS if tp > 1 else None,
                block_cyclic_sp_axis=SP_AXIS,
                block_cyclic_chunk_local=ix.key_rows,
                natural_query_blocks=True,
                natural_query_rows=0 if ix.key_rows == ix.w_rows else ix.w_rows,
            )
            got = ttnn.bringup.ring_indexer_score_dsa(
                qd,
                kbufs[0],
                wd,
                k_locals[js[0]],
                ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
                chunk_start_idx=0,
                program_config=cfg32,
                kv_len=kvmax,
                k_extra=kbufs[1 : len(js)],
                k_local_extra=k_locals[1 : len(js)],
                lane_ranges=words,
                **ep,
                **common,
            )
            gh = [ttnn.to_torch(t).reshape(R, -1) for t in ttnn.get_device_tensors(got)]
            gbm = [ttnn.to_torch(t).reshape(R, -1) for t in ttnn.get_device_tensors(bm)] if cand else None
            ttnn.deallocate(got)
            # the same call again after every key buffer was scrambled: bit-identical (its gathers rewrite them)
            scramble()
            again = ttnn.bringup.ring_indexer_score_dsa(
                qd,
                kbufs[0],
                wd,
                k_locals[js[0]],
                ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
                chunk_start_idx=0,
                program_config=cfg32,
                kv_len=kvmax,
                k_extra=kbufs[1 : len(js)],
                k_local_extra=k_locals[1 : len(js)],
                lane_ranges=words,
                **ep,
                **common,
            )
            ah = [ttnn.to_torch(t).reshape(R, -1) for t in ttnn.get_device_tensors(again)]
            ttnn.deallocate(again)
            for q in range(n):
                for j, a0, b0, p in chips[q]:
                    if q * R + a0 >= used:
                        continue
                    kv = segs[j][2]
                    x, y = gh[q][a0:b0, :kv], ah[q][a0:b0, :kv]
                    if not torch.equal(x.view(torch.int16), y.view(torch.int16)):
                        fails.append(("repeat", kind.__name__, name, a, q, j, a0, b0))
            # reference: PB.7B's passes (pass p: each chip its p-th range's rows rolled to row 0 on the host, a lanes
            # call with each chip its own lane: start = that row's position, its kv_len and key buffer)
            passes = max(len(c) for c in chips)
            for pp in range(passes):
                ent = [c[pp] if pp < len(c) else c[0] for c in chips]
                qr_ = qh.clone()
                wr_ = wh.clone()
                for q, (j, a0, b0, p) in enumerate(ent):
                    qr_[q] = torch.roll(qh[q], -a0, dims=1)
                    wr_[q] = torch.roll(wh[q], -a0, dims=1)
                qrd = up(qr_.view(sp, tp * nh, R, hd), q_dtype, (0, 1))
                wrd = up(wr_.view(sp, tp, R, nh), ttnn.bfloat16, (0, 1))
                if cand:
                    bmr = ttnn.allocate_tensor_on_device(
                        ttnn.Shape(bm_shape), ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
                    )
                    epr = dict(block_keys=ix.block, block_max_out=bmr, block_max_pin=True)
                kvs = [segs[e[0]][2] for e in ent]
                ref = ttnn.bringup.ring_indexer_score_dsa(
                    qrd,
                    kbufs[0],
                    wrd,
                    k_locals[0],
                    ix.ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=SP_AXIS),
                    chunk_start_idx=int(ent[0][3]),
                    program_config=ix.score_cfg,
                    kv_len=max(kvs),
                    lane_rows=R,
                    lane_starts=[int(e[3]) for e in ent],
                    lane_kv_lens=kvs,
                    lane_buffer=[int(e[0]) for e in ent],
                    k_extra=kbufs[1 : len(js)],
                    k_local_extra=k_locals[1 : len(js)],
                    **epr,
                    **common,
                )
                rhs = [ttnn.to_torch(t).reshape(R, -1) for t in ttnn.get_device_tensors(ref)]
                rbs = [ttnn.to_torch(t).reshape(R, -1) for t in ttnn.get_device_tensors(bmr)] if cand else None
                for q, (j, a0, b0, p) in enumerate(ent):
                    if pp >= len(chips[q]) or q * R + a0 >= used:
                        continue  # (a filler chip of this pass, or the tail pad rows)
                    kv, m = segs[j][2], b0 - a0
                    g_, r_ = gh[q][a0:b0, :kv].float(), rhs[q][:m, :kv].float()
                    same_inf = torch.equal(torch.isinf(g_), torch.isinf(r_))
                    fin = ~torch.isinf(r_)
                    d = (g_[fin] - r_[fin]).abs()
                    bad = (not same_inf) or bool((d > 0).any()) or bool(torch.isnan(g_).any())
                    if cand:
                        nb = kv // ix.block
                        bad = bad or not torch.equal(gbm[q][a0:b0, :nb], rbs[q][:m, :nb])
                    checked += 1
                    if bad:
                        fails.append(
                            (
                                kind.__name__,
                                name,
                                a,
                                q,
                                j,
                                a0,
                                b0,
                                p,
                                kv,
                                float(d.max()) if d.numel() else -1.0,
                                same_inf,
                            )
                        )
                ttnn.deallocate(ref)
                ttnn.deallocate(qrd)
                ttnn.deallocate(wrd)
                if cand:
                    ttnn.deallocate(bmr)
            for t in (qd, wd):
                ttnn.deallocate(t)
            if cand:
                ttnn.deallocate(bm)
        for t in k_locals + kbufs:
            ttnn.deallocate(t)
    print(f"RANGES {checked} (chip, range) pairs checked, {len(fails)} differ")
    for f in fails[:40]:
        print("RANGES DIFF", f)
    assert not fails, f"{len(fails)} of {checked} (chip, range) pairs differ from the PB.7B path"
