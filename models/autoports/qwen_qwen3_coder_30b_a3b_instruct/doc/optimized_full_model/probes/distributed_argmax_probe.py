# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Can greedy decode stop all-gathering the whole vocabulary?

Stage 05 ships greedy on ``Sampling1D``'s force-argmax path
(``sampling_1d.py::_sample_argmax``): all-gather the column-parallel logits
shard up to the full 151936 on every die, ``ttnn.untilize``, ``ttnn.argmax``.
``doc/full_model/tt_perf_report_full_model_decode.txt`` prices that at
**AllGatherAsync 889 us + ArgMax 859 us = ~1.80 ms**, which is essentially all
of the 1.87 ms of non-layer work inside a 22.079 ms token-out decode.

The candidate measured here reduces first and gathers second:

    rm        = untilize(shard)               # bf16  [1,1,32,37984]
    local_idx = argmax(rm, dim=-1)            # uint32[1,1,32,1]
    local_max = gather(rm, -1, local_idx)     # bf16  [1,1,32,1]  -- NOT ttnn.max
    global_idx = local_idx + die_rank*37984   # int32, per-die sharded constant
    all_gather(local_max), all_gather(global_idx)   -> [1,1,32,4] each
    gmax  = max(vals4, -1, keepdim)
    mask  = (vals4 == gmax)                   # int32 0/1
    sel   = BIG + mask*(idx4 - BIG)           # non-winners -> BIG
    token = min(sel, -1)                      # lowest global index among ties

Two things this file establishes empirically rather than by assumption:

* **``ttnn.argmax`` returns the first occurrence on an exact tie.** The kernel
  comparator is a strict ``>`` and the cross-core combine documents "tie-breaking
  favors the smaller index"
  (``ttnn/cpp/ttnn/operations/reduction/argmax/device/kernels/reader_argmax_interleaved_multicore.cpp``),
  but ``--tie-test`` checks it on the device with a crafted tie.
* **The cross-die reduction never sums two tied indices.** It is a ``min`` over
  masked indices, not ``sum(mask*idx)``: on a tie both lanes survive the mask and
  ``min`` keeps the lower global index, which -- because the dies hold contiguous
  ascending vocabulary ranges -- is exactly ``torch.argmax``'s first-maximal rule.

Three device facts found while building it, all load-bearing:

* **The obvious spelling of the per-die maximum is the expensive one.**
  ``ttnn.max`` over ``[1,1,32,37984]`` bf16 costs **0.494 ms** -- more than
  ``ttnn.argmax`` over the same tensor (0.373 ms), and 6.6x the ``untilize``
  that feeds it (0.075 ms). Reshaping, L1, ``keepdim``, a 2-way slice-and-
  ``maximum`` tree: none of it moved below 0.434 ms. ``ttnn.gather`` at the index
  the argmax already produced costs **0.058 ms**, and that single substitution is
  the difference between 1.10x and 1.6x over the baseline.

* ``ttnn.where`` on INT32 operands returns bit-garbage
  (``where(mask, [36885,...], 151937)`` -> ``[-1877638913, 164095, ...]``), so the
  select is spelled as int32 arithmetic instead.
* FLOAT32 elementwise loses the index: ``typecast(uint32->float32)`` + ``add``
  turned index 36885 into 36864 (bf16 rounding, ulp 128 at 2^15). Indices are
  carried as **INT32 end to end**, which is exact.

Standalone: opens its own 1x4 mesh, synthetic logits at the shipped shape
``[1,1,32,37984]`` bf16 per die, no 48-layer model. Every leg is trace-captured
and timed as a median over many ``execute_trace`` reps; no readback happens
inside a timed region.

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

from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig  # noqa: E402
from models.common.modules.tt_ccl import TT_CCL  # noqa: E402

HERE = Path(__file__).resolve().parent
VOCAB = 151936
DEVICES = 4
LOCAL_VOCAB = VOCAB // DEVICES  # 37984
SLOTS = 32
TOPOLOGY = ttnn.Topology.Ring  # what tt/multichip_decoder.py uses on this mesh
BIG = 1 << 20  # > VOCAB; the "not a winner" sentinel for the min-reduction


# ---------------------------------------------------------------------------
# Baseline: the shipped force-argmax path, spelled the way the model spells it
# ---------------------------------------------------------------------------


class _WatcherCleanSampling1D(Sampling1D):
    """Same override the model carries (``tt/model.py::_WatcherCleanSampling1D``).

    ``Sampling1D._argmax_all_gather``'s fallback pins ``num_workers_per_link=1``
    with ``Topology::Linear``, which trips a BRISC ASSERT in
    ``minimal_default_writer.cpp``. This spells the same gather with no tuning
    knobs pinned. Copied rather than imported so the probe does not drag in the
    model module; **no shared file is edited**.
    """

    def _argmax_all_gather(self, logits):
        cfg = self.config
        return ttnn.experimental.all_gather_async(
            logits,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=cfg.num_argmax_gather_links,
            memory_config=logits.memory_config(),
            topology=cfg.ag_topology,
        )


def build_sampler(mesh, ccl):
    sampler = _WatcherCleanSampling1D.from_config(
        Sampling1DConfig(
            vocab_size=VOCAB,
            valid_vocab_size=VOCAB,
            mesh_device=mesh,
            tt_ccl=ccl,
            max_batch_size=SLOTS,
            max_top_k=32,
            num_gather_links=1,
            sampling_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            allow_force_argmax=True,
            num_argmax_gather_links=1,
            ag_topology=TOPOLOGY,
            pad_to_power_of_2=False,
        )
    )
    sampler.load_device_buffers()
    return sampler


# ---------------------------------------------------------------------------
# The candidate
# ---------------------------------------------------------------------------


class DistributedArgmax:
    """Per-die argmax, then a 4-wide candidate gather.

    ``valid_vocab_size == vocab_size`` for this model (151936 is the real vocab
    and the LM head is not padded past it), so the masking/slicing that
    ``sampling_1d.py`` does for a padded vocabulary
    (``_can_slice_valid_vocab_for_argmax`` / ``_mask_invalid_vocab_logits``) is a
    no-op here and is not reproduced. ``valid_vocab_size < vocab_size`` would need
    the invalid tail masked **before** the local argmax -- asserted in the
    constructor rather than silently ignored.
    """

    def __init__(
        self,
        mesh,
        ccl,
        *,
        num_links=1,
        untilize_local=True,
        u32_rm_out=False,
        local_value="gather",
        valid_vocab_size=VOCAB,
    ):
        assert valid_vocab_size == VOCAB, (
            "a padded vocabulary would need the invalid tail masked before the local argmax; "
            "this probe only covers the shipped valid_vocab_size == vocab_size case"
        )
        assert local_value in ("gather", "reduce")
        assert untilize_local or local_value == "reduce", "the gather path reads the ROW_MAJOR tensor"
        self.mesh = mesh
        self.ccl = ccl
        self.num_links = num_links
        self.untilize_local = untilize_local
        self.u32_rm_out = u32_rm_out
        self.local_value = local_value
        offsets = (
            torch.arange(DEVICES, dtype=torch.int64).reshape(1, 1, 1, DEVICES).expand(1, 1, SLOTS, DEVICES).contiguous()
            * LOCAL_VOCAB
        ).to(torch.int32)
        # Sharded on the last dim: die d gets the single column d*37984.
        self.die_offset = ttnn.from_torch(
            offsets,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )

    def _gather(self, tensor):
        return ttnn.experimental.all_gather_async(
            tensor,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=TOPOLOGY,
            # No num_workers_per_link / chunks_per_sync pinned -- see the
            # _WatcherCleanSampling1D docstring: Linear + num_workers_per_link=1
            # is the pair that trips the watcher, and pinning nothing avoids it.
        )

    def __call__(self, logits):
        # -- per-die reduction, over this die's own 37984 columns --------------
        if self.untilize_local:
            # ttnn.argmax's multicore path needs ROW_MAJOR; the TILE path is
            # single-core (see the support table in argmax_nanobind.cpp).
            x = ttnn.untilize(logits, use_multicore=True)
            local_idx = ttnn.argmax(x, dim=-1, keepdim=True)  # uint32 RM [1,1,32,1]
        else:
            x = None
            local_idx = ttnn.argmax(logits, dim=-1, keepdim=True)

        if self.local_value == "gather":
            # **The whole difference between 1.10x and 1.6x.** ttnn.max over the
            # 37984-wide shard costs 0.494 ms -- more than the argmax over the same
            # tensor (0.373 ms) and 6.6x the untilize that feeds it (0.075 ms).
            # ttnn.gather reads the single value the argmax already located, in
            # 0.058 ms. ROW_MAJOR input and a UINT32 index are both required: the
            # TILE spelling of the same gather runs 19.05 ms, and an INT32 index is
            # rejected outright.
            local_max = ttnn.to_layout(ttnn.gather(x, dim=-1, index=local_idx), ttnn.TILE_LAYOUT)
        else:
            local_max = ttnn.max(logits, dim=-1, keepdim=True)  # bf16 [1,1,32,1] TILE
        if x is not None:
            ttnn.deallocate(x)
        # INT32, not FLOAT32: fp32 elementwise rounds the index through bf16.
        local_idx_i32 = ttnn.to_layout(ttnn.typecast(local_idx, ttnn.int32), ttnn.TILE_LAYOUT)
        global_idx = ttnn.add(local_idx_i32, self.die_offset)

        # -- gather the 4 candidates, not the 151936 logits --------------------
        vals4 = self._gather(local_max)  # bf16  [1,1,32,4]
        idx4 = self._gather(global_idx)  # int32 [1,1,32,4]

        # -- cross-die reduction ----------------------------------------------
        gmax = ttnn.max(vals4, dim=-1, keepdim=True)
        mask = ttnn.typecast(ttnn.eq(vals4, gmax), ttnn.int32)  # 0/1
        # NOT sum(mask*idx): on a tie that would add the two indices together.
        # BIG + mask*(idx-BIG) sends losers to BIG and leaves every tied winner
        # at its own global index, so min() keeps the lowest -- which is the
        # first-maximal index, because die ranges ascend.
        sel = ttnn.add(ttnn.multiply(mask, ttnn.subtract(idx4, BIG)), BIG)
        token = ttnn.min(sel, dim=-1, keepdim=False)  # int32 [1,1,32] TILE

        if self.u32_rm_out:
            # Match the force-argmax path's output contract exactly: ttnn.argmax
            # emits UINT32 / ROW_MAJOR, which is what the generator's tt_out_tok is.
            token = ttnn.typecast(ttnn.to_layout(token, ttnn.ROW_MAJOR_LAYOUT), ttnn.uint32)
        return token


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def read_tokens(tensor):
    return [int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).reshape(-1)[:SLOTS].tolist()]


def traced_ms(mesh, fn, reps):
    """Median ms over ``reps`` ``execute_trace`` calls. Nothing reads back inside."""
    fn()
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
    return statistics.median(samples)


def upload(mesh, host):
    return ttnn.from_torch(
        host,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
    )


def tie_tests(mesh, ccl, results):
    """Crafted exact ties: across two dies, and within one die."""
    dist = DistributedArgmax(mesh, ccl)
    cases = {}

    # (a) cross-die tie: identical maxima on die 1 and die 3. torch.argmax takes
    #     the lower global index, i.e. the die-1 one.
    host = torch.full((1, 1, SLOTS, VOCAB), -8.0)
    a, b = 1 * LOCAL_VOCAB + 77, 3 * LOCAL_VOCAB + 1234
    host[0, 0, :, a] = 5.0
    host[0, 0, :, b] = 5.0
    cases["cross_die_tie"] = (host, a)

    # (b) within-die tie: two identical maxima inside die 2.
    host = torch.full((1, 1, SLOTS, VOCAB), -8.0)
    a2, b2 = 2 * LOCAL_VOCAB + 300, 2 * LOCAL_VOCAB + 9000
    host[0, 0, :, a2] = 5.0
    host[0, 0, :, b2] = 5.0
    cases["within_die_tie"] = (host, a2)

    # (c) tie in the *first* die against a later die, plus a within-die tie in
    #     the same row -- the lowest index of all must win.
    host = torch.full((1, 1, SLOTS, VOCAB), -8.0)
    a3 = 11
    host[0, 0, :, a3] = 5.0
    host[0, 0, :, 500] = 5.0
    host[0, 0, :, 2 * LOCAL_VOCAB + 4] = 5.0
    cases["triple_tie"] = (host, a3)

    for name, (host, expect) in cases.items():
        logits = upload(mesh, host)
        torch_ref = host[0, 0].argmax(dim=-1).tolist()
        assert all(int(t) == expect for t in torch_ref), (name, torch_ref[:4], expect)
        tokens = read_tokens(dist(logits))
        ok = all(t == expect for t in tokens)
        cases_result = {
            "expected_torch_argmax": expect,
            "device_tokens_unique": sorted(set(tokens)),
            "pass": ok,
        }
        results[f"tie_{name}"] = cases_result
        print(
            f"tie:{name:<18} expect {expect:<8} device {sorted(set(tokens))} -> {'PASS' if ok else 'FAIL'}", flush=True
        )
        ttnn.deallocate(logits)

    # (d) does ttnn.argmax itself return the first occurrence? Direct check, per
    #     die, with two equal maxima in one shard.
    host = torch.full((1, 1, SLOTS, VOCAB), -8.0)
    for d in range(DEVICES):
        host[0, 0, :, d * LOCAL_VOCAB + 40] = 3.0
        host[0, 0, :, d * LOCAL_VOCAB + 41] = 3.0  # exact tie, adjacent
        host[0, 0, :, d * LOCAL_VOCAB + 20000] = 3.0  # and one far away, same value
    logits = upload(mesh, host)
    idx = ttnn.argmax(ttnn.untilize(logits, use_multicore=True), dim=-1, keepdim=True)
    per_die = {
        f"die{d}": sorted(
            set(int(v) for v in ttnn.to_torch(ttnn.get_device_tensors(idx)[d]).reshape(-1)[:SLOTS].tolist())
        )
        for d in range(DEVICES)
    }
    first_occurrence = all(v == [40] for v in per_die.values())
    results["ttnn_argmax_returns_first_occurrence"] = {"per_die_indices": per_die, "pass": first_occurrence}
    print(
        f"tie:{'ttnn.argmax_first':<18} expect 40 per die, got {per_die} -> "
        f"{'PASS' if first_occurrence else 'FAIL'}",
        flush=True,
    )
    ttnn.deallocate(logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--legs", type=str, default="", help="comma-separated leg names; default all")
    parser.add_argument("--skip-ties", action="store_true")
    parser.add_argument("--skip-components", action="store_true")
    parser.add_argument("--json", type=str, default=str(HERE / "distributed_argmax_probe.json"))
    args = parser.parse_args()
    wanted = {s for s in args.legs.split(",") if s}

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, DEVICES), trace_region_size=90_000_000)
    results = {}
    try:
        ccl = TT_CCL(mesh)
        torch.manual_seed(args.seed)
        host = torch.randn(1, 1, SLOTS, VOCAB) * 4.0
        # Two references. fp32 is what the task asks for; bf16 is what the device
        # actually ranks, because the logits are uploaded as bfloat16. Stage 05 hit
        # exactly this: they disagree on slot 4 at seed 0 (an exact bf16 tie).
        expect_fp32 = [int(v) for v in host[0, 0].argmax(dim=-1).tolist()]
        expect_bf16 = [int(v) for v in host[0, 0].to(torch.bfloat16).float().argmax(dim=-1).tolist()]
        ref_disagree = [i for i in range(SLOTS) if expect_fp32[i] != expect_bf16[i]]
        results["reference"] = {
            "fp32_vs_bf16_host_argmax_disagree_slots": ref_disagree,
            "note": "the device ranks a bf16 tensor; fp32/bf16 host argmax disagreement is the probe's "
            "own reference, not a leg's error",
        }
        print(f"host fp32 vs bf16 argmax disagree on slots {ref_disagree}", flush=True)

        logits = upload(mesh, host)
        assert int(logits.shape[-1]) == LOCAL_VOCAB, logits.shape

        sampler = build_sampler(mesh, ccl)
        legs = {
            "baseline_force_argmax": lambda: sampler.decode_forward(logits, enable_log_probs=False)[0],
            "distributed_argmax": DistributedArgmax(mesh, ccl),
            "distributed_argmax_u32_rm_out": DistributedArgmax(mesh, ccl, u32_rm_out=True),
            "distributed_argmax_links2": DistributedArgmax(mesh, ccl, num_links=2),
            "distributed_argmax_reduce_value": DistributedArgmax(mesh, ccl, local_value="reduce"),
            "distributed_argmax_no_untilize": DistributedArgmax(mesh, ccl, untilize_local=False, local_value="reduce"),
        }

        for name, leg in legs.items():
            if wanted and name not in wanted:
                continue
            fn = leg if name == "baseline_force_argmax" else (lambda leg=leg: leg(logits))
            try:
                tokens = read_tokens(fn())
                ttnn.synchronize_device(mesh)
                results[name] = {
                    "ms": traced_ms(mesh, fn, args.reps),
                    "matches_host_fp32": f"{sum(int(a) == int(b) for a, b in zip(tokens, expect_fp32))}/{SLOTS}",
                    "matches_host_bf16": f"{sum(int(a) == int(b) for a, b in zip(tokens, expect_bf16))}/{SLOTS}",
                    "disagree_slots_vs_fp32": [i for i in range(SLOTS) if tokens[i] != expect_fp32[i]],
                    "tokens": tokens,
                }
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                results[name] = {"error": repr(exc)}
            printable = {k: v for k, v in results[name].items() if k != "tokens"}
            print(f"{name:<32} {printable}", flush=True)

        if not args.skip_components:
            comp = {}
            rm_local = ttnn.untilize(logits, use_multicore=True)
            idx_local = ttnn.argmax(rm_local, dim=-1, keepdim=True)
            local_max = ttnn.to_layout(ttnn.gather(rm_local, dim=-1, index=idx_local), ttnn.TILE_LAYOUT)
            dist = DistributedArgmax(mesh, ccl)
            gidx = ttnn.add(
                ttnn.to_layout(ttnn.typecast(idx_local, ttnn.int32), ttnn.TILE_LAYOUT),
                dist.die_offset,
            )
            vals4 = dist._gather(local_max)
            idx4 = dist._gather(gidx)

            def tail():
                gmax = ttnn.max(vals4, dim=-1, keepdim=True)
                mask = ttnn.typecast(ttnn.eq(vals4, gmax), ttnn.int32)
                return ttnn.min(ttnn.add(ttnn.multiply(mask, ttnn.subtract(idx4, BIG)), BIG), dim=-1, keepdim=False)

            pieces = {
                "comp_local_max_via_ttnn_max_37984": lambda: ttnn.max(logits, dim=-1, keepdim=True),
                "comp_local_max_via_gather_rm": lambda: ttnn.gather(rm_local, dim=-1, index=idx_local),
                "comp_local_max_via_gather_tile": lambda: ttnn.gather(
                    logits, dim=-1, index=ttnn.to_layout(idx_local, ttnn.TILE_LAYOUT)
                ),
                "comp_untilize_local_37984": lambda: ttnn.untilize(logits, use_multicore=True),
                "comp_argmax_local_rm_37984": lambda: ttnn.argmax(rm_local, dim=-1, keepdim=True),
                "comp_gather_1lane_bf16": lambda: dist._gather(local_max),
                "comp_gather_1lane_int32": lambda: dist._gather(gidx),
                "comp_cross_die_tail": tail,
                "comp_full_vocab_gather_151936": lambda: dist._gather(logits),
            }
            for key, fn in pieces.items():
                try:
                    comp[key] = traced_ms(mesh, fn, max(20, args.reps // 4))
                    print(f"{key:<32} {comp[key]:.4f} ms", flush=True)
                except Exception as exc:  # noqa: BLE001
                    comp[key] = repr(exc)
                    print(f"{key:<32} FAILED {exc}", flush=True)
            results["components"] = comp

        if not args.skip_ties:
            print("--- tie tests ---", flush=True)
            tie_tests(mesh, ccl, results)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    Path(args.json).write_text(json.dumps(results, indent=2))
    base = results.get("baseline_force_argmax", {}).get("ms")
    cand = results.get("distributed_argmax", {}).get("ms")
    if base and cand:
        print(
            f"\nspeedup distributed_argmax over baseline: {base / cand:.2f}x ({base:.4f} -> {cand:.4f} ms)", flush=True
        )
    print(json.dumps({k: v for k, v in results.items() if k != "components"}, indent=2))


if __name__ == "__main__":
    main()
