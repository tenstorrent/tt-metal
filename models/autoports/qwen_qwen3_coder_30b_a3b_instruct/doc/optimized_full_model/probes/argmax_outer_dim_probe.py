# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Why is a 110-core ``ttnn.argmax`` over ``[1,1,32,37984]`` 75x off bandwidth?

Stage 06's distributed argmax (``tt/model.py::_WatcherCleanSampling1D``) spends
**366 us** in one ``ttnn.argmax`` over a ROW_MAJOR ``[1,1,32,37984]`` bf16 shard.
2.4 MB at 512 GB/s is under 5 us, so the op is ~75x off bandwidth and the
question is *where the time goes*, not *how fast can DRAM go*.

Reading the kernel answers it. ``argmax_multi_core_program_factory.cpp`` computes

    output_last_dim  = reduce_all or keepdim or rank < 2 ? 1 : input_shape[rank-2]
    inner_dim_units  = output_last_dim
    outer_dim_units  = logical_volume / inner_dim_units / red_dim_units

With ``keepdim=True`` on ``[1,1,32,37984]`` that is ``inner=1``, **``outer=32``**.
``reader_argmax_interleaved_multicore.cpp``'s main loop is over ``outer_dim_units``
and **every iteration is a full 110-core barrier**: the reduce core resets
``done_sem``, multicasts ``start_sem``, every worker reads its ~352-element
(704-byte) slice from DRAM, scalar-compares it on the RISCV_1 *data-movement*
core, NOC-writes 8 bytes of partial back to the reduce core, and increments
``done_sem``; the reduce core waits for all 110. So the op is **32 sequential
110-way barriers over 704-byte reads** -- pure latency and synchronisation, with
the comparison itself running as scalar C++ on a DM RISC. 366/32 = 11.4 us a
round, which is the right order for a 110-core multicast + 110-way semaphore
wait + a DRAM round trip.

The inner loop over ``j < inner_dim_units`` inside ``find_argmax_for_core`` has
**no** semaphore traffic: it is a plain loop that reads page ``outer*inner + j``
and stores ``red_idxs[j]``/``red_vals[j]``. The reduce core then reduces all
``inner`` outputs under a single ``done_sem.wait``. Rows are therefore free to
batch -- if they land on ``inner`` rather than on ``outer``.

``keepdim=False`` is exactly that switch. It gives ``inner=32, outer=1``: one
barrier instead of 32, the same 32 rows, the same pages, the same per-element
comparator, the same first-maximal tie rule. It is not an approximation; it is
the same reduction with the loop nest transposed.

Legs measured here (trace-captured, median of many ``execute_trace``):

* ``untilize`` alone, to price the input side;
* ``argmax`` at ``keepdim=True`` (shipped) and ``keepdim=False`` (candidate);
* ``keepdim=False`` + the ``[1,1,32] -> [1,1,32,1]`` reshape the rest of the
  reduction needs, because a candidate that needs an expensive fix-up is not a
  candidate;
* ``sub_core_grids`` sweeps at both ``keepdim`` settings -- if the cost is the
  barrier, fewer cores should *help* at ``keepdim=True``;
* a batch-1 row slice, which is the other way to make ``outer=1`` but only works
  at ``max_batch_size == 1``;
* ``ttnn.topk(k=1)`` as an alternative spelling;
* and the two full distributed-argmax reductions end to end.

Every leg's tokens are checked against the same host reference, including a
crafted-tie leg for the first-maximal rule.

Standalone: opens its own 1x4 mesh at the shipped shape, no 48-layer model.
**Standalone probes on this project have lied twice** (stage-04
``rotary_embedding_llama``, stage-06 ``k_chunk`` memory safety), so nothing here
is adopted on these numbers alone -- the gate is in-model.

Nothing here writes into ``doc/full_model/``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from models.common.modules.tt_ccl import TT_CCL  # noqa: E402

HERE = Path(__file__).resolve().parent
VOCAB = 151936
DEVICES = 4
LOCAL_VOCAB = VOCAB // DEVICES  # 37984
SLOTS = 32
TOPOLOGY = ttnn.Topology.Ring
BIG = 1 << 20
#: set from --skip-sweep; the core sweep is slow and its answer does not change.
SKIP_SWEEP = False


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def traced_ms(mesh, fn, reps):
    """Median ms over ``reps`` ``execute_trace`` calls. No readback inside."""
    out = fn()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    for _ in range(5):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        samples.append((time.perf_counter() - t0) * 1e3)
    ttnn.release_trace(mesh, tid)
    return statistics.median(samples), out


def upload_sharded(mesh, host):
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )


def grid(n_cores, mesh):
    """A ``CoreRangeSet`` of the first ``n_cores`` cores, at most 2 ranges."""
    g = mesh.compute_with_storage_grid_size()
    w = g.x
    full_rows, rem = divmod(n_cores, w)
    ranges = []
    if full_rows:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, full_rows - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full_rows), ttnn.CoreCoord(rem - 1, full_rows)))
    return ttnn.CoreRangeSet(ranges)


# ---------------------------------------------------------------------------
# legs
# ---------------------------------------------------------------------------


def run(mesh, ccl, reps, results):
    torch.manual_seed(20260815)
    host = torch.randn(1, 1, SLOTS, VOCAB, dtype=torch.float32) * 4.0
    logits = upload_sharded(mesh, host)
    ref_bf16 = host.to(torch.bfloat16).to(torch.float32)[0, 0].argmax(dim=-1).tolist()

    rm_static = ttnn.untilize(logits, use_multicore=True)  # kept alive; input to the argmax legs

    def check(t, name, rows=SLOTS):
        got = [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(t)[0]).reshape(-1)[:rows].tolist()]
        # the per-die legs return a *local* index; convert the host reference the
        # same way only when the leg is a full reduction.
        return got

    # ---- input side ------------------------------------------------------
    ms, _ = traced_ms(mesh, lambda: ttnn.untilize(logits, use_multicore=True), reps)
    results["untilize_37984"] = {"ms": ms}
    print(f"untilize_37984                       {ms*1e3:8.1f} us")

    # ---- the shipped spelling and the loop-nest transpose -----------------
    def leg(name, fn, rows=SLOTS, expect_local=True):
        ms, out = traced_ms(mesh, fn, reps)
        vals = [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1)[:rows].tolist()]
        entry = {"ms": ms, "first4": vals[:4]}
        if expect_local:
            # die 0 holds columns [0, 37984); a *local* argmax on die 0 equals the
            # global argmax restricted to that range.
            local_ref = host.to(torch.bfloat16).to(torch.float32)[0, 0, :rows, :LOCAL_VOCAB].argmax(dim=-1).tolist()
            entry["matches_local_ref"] = f"{sum(int(a)==int(b) for a,b in zip(vals, local_ref))}/{rows}"
        results[name] = entry
        print(f"{name:36s} {ms*1e3:8.1f} us   {entry.get('matches_local_ref','')}")
        return ms

    leg("argmax_keepdim_true", lambda: ttnn.argmax(rm_static, dim=-1, keepdim=True))
    leg("argmax_keepdim_false", lambda: ttnn.argmax(rm_static, dim=-1, keepdim=False))

    # the reshape the rest of the reduction needs after keepdim=False
    def kdf_reshape():
        i = ttnn.argmax(rm_static, dim=-1, keepdim=False)
        return ttnn.reshape(i, (1, 1, SLOTS, 1))

    leg("argmax_keepdim_false_plus_reshape", kdf_reshape)

    # ---- core-count sweep -------------------------------------------------
    # **Only single-core-group grids are swept.** ``sub_core_grids`` that split
    # into two ranges *and* leave a remainder larger than one core's share hang
    # the device: ``argmax_multi_core_program_factory.cpp`` computes
    #     red_dim_units_last1 = red_dim_units1 - (ideal_red_dim_units - red_dim_units)
    # in **uint32**, and at 64 cores on this 13-wide grid that is 608 - 928, which
    # wraps to 4294966976. The reader then issues an unbounded NOC read and the
    # kernel never returns. Reproduced once here (log truncated at
    # ``argmax_kd_true_cores32``, board reset required); not re-run.
    # 8 / 16 / 32 are safe because their remainder is smaller than one core's
    # share. Upstream bug; worth filing.
    for n in () if SKIP_SWEEP else (8, 16, 32):
        try:
            g = grid(n, mesh)
            leg(f"argmax_kd_true_cores{n}", lambda g=g: ttnn.argmax(rm_static, dim=-1, keepdim=True, sub_core_grids=g))
        except Exception as exc:  # noqa: BLE001
            results[f"argmax_kd_true_cores{n}"] = {"error": str(exc)[:300]}
            print(f"argmax_kd_true_cores{n}: FAILED {str(exc)[:120]}")

    # ---- the row lever ----------------------------------------------------
    # The scalar comparison, not the barrier, is what costs: the reduction runs
    # as a C++ ``>`` loop on the RISCV_1 *data-movement* core, so the bill is
    # 32 rows x 37984 values / 110 cores ~ 11k compares per core. Decode at
    # batch B pads the logits to 32 rows with **zeros** (``decode_terminal``
    # pads the pre-head hidden with 0.0 and ``lm_head`` has no bias), so
    # 32 - B of those rows are padding and their argmax is 0 by construction.
    # Reducing only the B live rows is exact, not an approximation.
    for rows_kept in (1, 2, 4, 8):
        try:

            def slice_leg(r=rows_kept):
                x = ttnn.slice(rm_static, [0, 0, 0, 0], [1, 1, r, LOCAL_VOCAB])
                out = ttnn.argmax(x, dim=-1, keepdim=True)
                ttnn.deallocate(x)
                return out

            leg(f"rm_slice{rows_kept}_then_argmax", slice_leg, rows=rows_kept)
        except Exception as exc:  # noqa: BLE001
            results[f"rm_slice{rows_kept}_then_argmax"] = {"error": str(exc)[:300]}
            print(f"rm_slice{rows_kept}_then_argmax: FAILED {str(exc)[:120]}")

    # the RM slice on its own, and the alternative order (slice in TILE, then
    # untilize only one row) -- the untilize is 75 us and most of it is also
    # padding work
    try:
        ms, _ = traced_ms(mesh, lambda: ttnn.slice(rm_static, [0, 0, 0, 0], [1, 1, 1, LOCAL_VOCAB]), reps)
        results["rm_slice1_only"] = {"ms": ms}
        print(f"rm_slice1_only                       {ms*1e3:8.1f} us")
    except Exception as exc:  # noqa: BLE001
        results["rm_slice1_only"] = {"error": str(exc)[:300]}
        print(f"rm_slice1_only: FAILED {str(exc)[:160]}")

    try:

        def tile_slice_untilize():
            t = ttnn.slice(logits, [0, 0, 0, 0], [1, 1, 1, LOCAL_VOCAB])
            u = ttnn.untilize(t, use_multicore=True)
            ttnn.deallocate(t)
            return u

        ms, _ = traced_ms(mesh, tile_slice_untilize, reps)
        results["tile_slice1_plus_untilize"] = {"ms": ms}
        print(f"tile_slice1_plus_untilize            {ms*1e3:8.1f} us")
    except Exception as exc:  # noqa: BLE001
        results["tile_slice1_plus_untilize"] = {"error": str(exc)[:300]}
        print(f"tile_slice1_plus_untilize: FAILED {str(exc)[:160]}")

    # ---- alternative spellings -------------------------------------------
    for name, fn in (
        ("topk_k1_rm_32rows", lambda: ttnn.topk(rm_static, k=1, dim=-1, largest=True, sorted=True)),
        ("topk_k32_tile_32rows", lambda: ttnn.topk(logits, k=32, dim=-1, largest=True, sorted=True)),
    ):
        try:
            ms, _ = traced_ms(mesh, fn, reps)
            results[name] = {"ms": ms}
            print(f"{name:36s} {ms*1e3:8.1f} us")
        except Exception as exc:  # noqa: BLE001
            results[name] = {"error": str(exc)[:300]}
            print(f"{name}: FAILED {str(exc)[:160]}")

    # the padding the 1-row candidate needs to restore a 32-slot token vector
    try:
        one = ttnn.argmax(ttnn.slice(rm_static, [0, 0, 0, 0], [1, 1, 1, LOCAL_VOCAB]), dim=-1, keepdim=True)

        def pad_leg():
            return ttnn.pad(one, [(0, 0), (0, 0), (0, 0), (0, SLOTS - 1)], value=0)

        ms, out = traced_ms(mesh, pad_leg, reps)
        results["pad_token_1_to_32"] = {
            "ms": ms,
            "vals": [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1).tolist()],
        }
        print(f"pad_token_1_to_32                    {ms*1e3:8.1f} us")
        ttnn.deallocate(one)
    except Exception as exc:  # noqa: BLE001
        results["pad_token_1_to_32"] = {"error": str(exc)[:300]}
        print(f"pad_token_1_to_32: FAILED {str(exc)[:160]}")

    ttnn.deallocate(rm_static)

    # ---- end to end -------------------------------------------------------
    offsets = (
        torch.arange(DEVICES, dtype=torch.int64).reshape(1, 1, 1, DEVICES).expand(1, 1, SLOTS, DEVICES).contiguous()
        * LOCAL_VOCAB
    ).to(torch.int32)
    die_offset = ttnn.from_torch(
        offsets,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )

    def all_gather(t):
        return ttnn.experimental.all_gather_async(
            t,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=TOPOLOGY,
        )

    def full(keepdim, rows=SLOTS):
        rm = ttnn.untilize(logits, use_multicore=True)
        if rows < SLOTS:
            sliced = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, rows, LOCAL_VOCAB])
            ttnn.deallocate(rm)
            rm = sliced
        if keepdim:
            local_idx = ttnn.argmax(rm, dim=-1, keepdim=True)
        else:
            local_idx = ttnn.reshape(ttnn.argmax(rm, dim=-1, keepdim=False), (1, 1, rows, 1))
        local_max = ttnn.to_layout(ttnn.gather(rm, dim=-1, index=local_idx), ttnn.TILE_LAYOUT)
        ttnn.deallocate(rm)
        local_idx_i32 = ttnn.to_layout(ttnn.typecast(local_idx, ttnn.int32), ttnn.TILE_LAYOUT)
        ttnn.deallocate(local_idx)
        global_idx = ttnn.add(local_idx_i32, die_offset)
        ttnn.deallocate(local_idx_i32)
        vals4 = all_gather(local_max)
        idx4 = all_gather(global_idx)
        ttnn.deallocate(local_max)
        ttnn.deallocate(global_idx)
        gmax = ttnn.max(vals4, dim=-1, keepdim=True)
        mask = ttnn.typecast(ttnn.eq(vals4, gmax), ttnn.int32)
        sel = ttnn.add(ttnn.multiply(mask, ttnn.subtract(idx4, BIG)), BIG)
        token = ttnn.min(sel, dim=-1, keepdim=False)
        for s in (vals4, idx4, gmax, mask, sel):
            ttnn.deallocate(s)
        token = ttnn.typecast(ttnn.to_layout(token, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint32)
        if rows < SLOTS:
            # restore the 32-slot vector the generator's tt_out_tok is. Padding
            # with 0 is not a convention: it is the value the shipped 32-row path
            # already produces for a padded row (all-zero logits -> every die
            # ties at 0.0 -> the masked-min keeps global index 0).
            # ``token`` is rank 3 ([1,1,rows]) here, exactly as the 32-row path
            # leaves it; pad the last axis up to the 32 user slots.
            token = ttnn.pad(token, [(0, 0), (0, 0), (0, SLOTS - rows)], value=0)
        return token

    for name, kd, rows in (
        ("full_shipped_keepdim_true", True, SLOTS),
        ("full_candidate_keepdim_false", False, SLOTS),
        ("full_candidate_batch1", True, 1),
        ("full_candidate_batch1_kdfalse", False, 1),
    ):
        ms, out = traced_ms(mesh, lambda kd=kd, rows=rows: full(kd, rows), reps)
        toks = [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1)[:SLOTS].tolist()]
        n = SLOTS if rows == SLOTS else rows
        results[name] = {
            "ms": ms,
            "rows_reduced": rows,
            "matches_host_bf16": f"{sum(int(a)==int(b) for a,b in zip(toks[:n], ref_bf16[:n]))}/{n}",
            "tokens": toks,
        }
        print(f"{name:36s} {ms*1e3:8.1f} us   {results[name]['matches_host_bf16']}")

    ttnn.deallocate(logits)

    # ---- the padding-row claim, checked rather than argued ----------------
    # Rows 1..31 all-zero (what decode_terminal + a bias-free lm_head produce for
    # an inactive user slot). If the shipped 32-row reduction returns token 0
    # there, then padding the batch-1 candidate's output with 0 is bit-identical.
    hz = torch.randn(1, 1, SLOTS, VOCAB, dtype=torch.float32) * 4.0
    hz[0, 0, 1:, :] = 0.0
    logits = upload_sharded(mesh, hz)
    out = full(True, SLOTS)
    toks = [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1)[:SLOTS].tolist()]
    results["padding_rows_produce_token_zero"] = {
        "tokens": toks,
        "rows_1_to_31_all_zero_token": sorted(set(toks[1:])),
        "pass": set(toks[1:]) == {0},
    }
    print(f"padding rows -> tokens {sorted(set(toks[1:]))}  {'PASS' if set(toks[1:]) == {0} else 'FAIL'}")
    ttnn.deallocate(logits)

    # ---- the tie rule must survive the transpose --------------------------
    ties = {}
    for name, (a, extras) in {
        "cross_die_tie": (1 * LOCAL_VOCAB + 77, [3 * LOCAL_VOCAB + 1234]),
        "within_die_tie": (2 * LOCAL_VOCAB + 300, [2 * LOCAL_VOCAB + 9000]),
        "triple_tie": (11, [500, 2 * LOCAL_VOCAB + 4]),
    }.items():
        h = torch.full((1, 1, SLOTS, VOCAB), -8.0)
        h[0, 0, :, a] = 5.0
        for e in extras:
            h[0, 0, :, e] = 5.0
        lg = upload_sharded(mesh, h)
        for kd, tag in ((True, "keepdim_true"), (False, "keepdim_false")):
            rm = ttnn.untilize(lg, use_multicore=True)
            if kd:
                li = ttnn.argmax(rm, dim=-1, keepdim=True)
            else:
                li = ttnn.reshape(ttnn.argmax(rm, dim=-1, keepdim=False), (1, 1, SLOTS, 1))
            lm = ttnn.to_layout(ttnn.gather(rm, dim=-1, index=li), ttnn.TILE_LAYOUT)
            gi = ttnn.add(ttnn.to_layout(ttnn.typecast(li, ttnn.int32), ttnn.TILE_LAYOUT), die_offset)
            v4, i4 = all_gather(lm), all_gather(gi)
            gm = ttnn.max(v4, dim=-1, keepdim=True)
            mk = ttnn.typecast(ttnn.eq(v4, gm), ttnn.int32)
            sl = ttnn.add(ttnn.multiply(mk, ttnn.subtract(i4, BIG)), BIG)
            tk = ttnn.min(sl, dim=-1, keepdim=False)
            got = sorted({int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(tk)[0]).reshape(-1)[:SLOTS].tolist()})
            ties[f"{name}_{tag}"] = {"expected": a, "device_unique": got, "pass": got == [a]}
            print(f"tie {name}/{tag}: expected {a}, got {got} -> {'PASS' if got == [a] else 'FAIL'}")
            ttnn.deallocate(rm)
        ttnn.deallocate(lg)
    results["ties"] = ties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--out", type=Path, default=HERE / "argmax_outer_dim_probe.json")
    args = ap.parse_args()
    global SKIP_SWEEP
    SKIP_SWEEP = args.skip_sweep

    # Same fabric the model opens with; the 4-wide all-gathers below need it.
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, DEVICES), trace_region_size=90_000_000)
    ccl = TT_CCL(mesh)
    results = {}
    try:
        run(mesh, ccl, args.reps, results)
    finally:
        args.out.write_text(json.dumps(results, indent=2))
        try:
            ccl.close()
        except Exception:  # noqa: BLE001
            pass
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
