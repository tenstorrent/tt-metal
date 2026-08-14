# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Does ``Sampling1D._argmax_all_gather`` trip the watcher assert, and does a
barrier semaphore fix it?

Stage 05 shipped with an unlocalized watcher abort in
``all_gather_async/.../minimal_default_writer.cpp``. The full-model watcher run
(`../pytest_full_model_watcher_FAILS.log`) reached
``test_split_sampling_feeds_its_own_token_back_on_device`` and aborted inside
it, so the assert was in fact already localized to the sampling test; the
README's "a two-layer build did not finish inside a ten-minute budget" was
wrong about its own log.

This probe removes the model entirely. It builds the shipped ``Sampling1D`` over
synthetic ``[1, 1, 32, 37984]``-per-die logits -- the column-parallel LM head's
real output shape -- and drives the two collective spellings the sampler can
take, A/B:

* ``argmax_nobarrier``  -- ``Sampling1D``'s force-argmax path **unmodified**.
                           The name records the hypothesis this probe was built
                           to test, not what the code does: on this 1x4 mesh
                           ``_argmax_all_gather`` does **not** take its Ring /
                           no-barrier branch at all (see ``ccl_watcher_ab.py``),
                           so what actually runs is the Linear + barrier
                           fallback;
* ``argmax_barrier``    -- ``_argmax_all_gather`` replaced by the Ring +
                           barrier-semaphore spelling, byte for byte;
* ``split_k32``         -- the top-k/top-p path, whose candidate-block gather is
                           a different call site, as a control.

Why this matters even though the model passes without the watcher: a device
``ASSERT`` **compiles out** when the watcher is off. "Does not reproduce without
the watcher" therefore means the invariant is *unchecked*, not that it holds. A
tripped assert inside a shipped, traced, every-token collective is silent
undefined behaviour on the delivered path.

Run it under the watcher, one leg per process (the watcher aborts the process on
the first trip, so legs cannot share one):

    TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 \
      python .../probes/sampler_watcher_ab.py --leg argmax_nobarrier

``run_watcher_ab.sh`` drives every leg; ``--leg list`` is on the sibling probe.

**Result (archived in ../watcher_ab.log).** Force-argmax trips on its first
eager call, with zero layers, **with and without** the barrier semaphore -- so
the barrier is neither cause nor cure. Split top-k is clean. Stopping after the
gather, the vocab slice or the untilize all trip identically, so it is the
gather. But the same gather spelled out by hand is clean, because
``default_topology`` is ``Linear`` on this 1x4 mesh and the Ring branch above is
dead code here; the fallback's ``num_workers_per_link=1`` is what matters. See
``ccl_watcher_ab.py`` for the two-parameter minimal trigger.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))

from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.model import _WatcherCleanSampling1D  # noqa: E402
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig  # noqa: E402
from models.common.modules.tt_ccl import TT_CCL  # noqa: E402

VOCAB = 151936
DEVICES = 4
LOCAL_VOCAB = VOCAB // DEVICES  # 37984
SLOTS = 32


def build_sampler(mesh, ccl, cls=Sampling1D):
    sampler = cls.from_config(
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
            ag_topology=ttnn.Topology.Ring,
            pad_to_power_of_2=False,
        )
    )
    sampler.load_device_buffers()
    return sampler


def barrier_argmax_all_gather(sampler, logits):
    """``_argmax_all_gather``, byte for byte, plus ``barrier_semaphore=``.

    The layer's own two all-reduces pass a barrier semaphore on this same
    ``all_gather_async`` op and are watcher-clean on this tree
    (``../pytest_stage04_watcher.log.gz``); the sampler's spelling is the one
    collective in the model that omits it.
    """
    cfg = sampler.config
    return ttnn.experimental.all_gather_async(
        logits,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=cfg.tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=1,
        memory_config=logits.memory_config(),
        topology=ttnn.Topology.Ring,
        barrier_semaphore=cfg.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=24,
        num_workers_per_link=4,
        num_buffers_per_channel=2,
    )


def run_leg(mesh, ccl, logits, leg, *, reps, traced):
    # ``argmax_shipped`` is the class the model actually instantiates -- the
    # local workaround -- as opposed to the upstream ``Sampling1D``.
    cls = _WatcherCleanSampling1D if leg == "argmax_shipped" else Sampling1D
    sampler = build_sampler(mesh, ccl, cls)
    original = type(sampler)._argmax_all_gather
    if leg == "argmax_barrier":
        type(sampler)._argmax_all_gather = barrier_argmax_all_gather

    k = ttnn.from_torch(
        torch.full((SLOTS,), 5 if leg == "split_k32" else 1, dtype=torch.int32),
        device=mesh,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    p = ttnn.from_torch(
        torch.full((SLOTS,), 0.9 if leg == "split_k32" else 0.0, dtype=torch.bfloat16),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    t = ttnn.from_torch(
        torch.ones(SLOTS, dtype=torch.bfloat16),
        device=mesh,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    def argmax_stage(stop):
        """``Sampling1D._sample_argmax`` re-spelled so it can be stopped early.

        Byte-for-byte the same op sequence, so a leg that stops after the
        gather isolates the collective from the ``untilize``/``argmax`` that
        normally follow it in the same dispatch.
        """
        x = logits
        if not sampler._can_slice_valid_vocab_for_argmax():
            x = sampler._mask_invalid_vocab_logits(x)
        x = sampler._pre_argmax_gather(x)
        if stop == "gather":
            return x
        if sampler._can_slice_valid_vocab_for_argmax():
            x = sampler._slice_valid_vocab_for_argmax(x)
        if stop == "slice":
            return x
        x = ttnn.untilize(x, use_multicore=True)
        if stop == "untilize":
            return x
        return ttnn.argmax(x, dim=-1, keepdim=False)

    def raw_gather():
        """``_argmax_all_gather``'s exact arguments, called directly.

        ``ccl_watcher_ab.py`` shows this call is watcher-clean when nothing else
        has been built on the mesh, so this leg asks whether merely having the
        sampler's device buffers resident changes the outcome.
        """
        return ttnn.experimental.all_gather_async(
            logits,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=sampler.config.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            memory_config=logits.memory_config(),
            topology=ttnn.Topology.Ring,
            chunks_per_sync=24,
            num_workers_per_link=4,
            num_buffers_per_channel=2,
        )

    def call():
        if leg == "raw_gather":
            return raw_gather()
        if leg == "argmax_shipped":
            return sampler.decode_forward(logits, enable_log_probs=False)[0]
        if leg == "split_k32":
            return sampler.decode_forward(logits, k=k, p=p, temp=t, enable_log_probs=False)[0]
        if leg.startswith("argmax_stage_"):
            return argmax_stage(leg[len("argmax_stage_") :])
        return sampler.decode_forward(logits, enable_log_probs=False)[0]

    try:
        print(f"[{leg}] eager warm-up", flush=True)
        call()
        ttnn.synchronize_device(mesh)
        print(f"[{leg}] eager warm-up OK", flush=True)

        for i in range(reps):
            call()
            ttnn.synchronize_device(mesh)
        print(f"[{leg}] {reps} eager calls OK", flush=True)

        if traced:
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = call()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            print(f"[{leg}] trace captured", flush=True)
            for i in range(reps):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                if i % 10 == 0:
                    print(f"[{leg}] replay {i}", flush=True)
            ttnn.synchronize_device(mesh)
            token = int(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1)[0].item())
            ttnn.release_trace(mesh, tid)
            print(f"[{leg}] {reps} traced replays OK, token={token}", flush=True)
        print(f"[{leg}] RESULT clean", flush=True)
    except Exception:
        traceback.print_exc()
        print(f"[{leg}] RESULT exception", flush=True)
    finally:
        type(sampler)._argmax_all_gather = original


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--leg",
        default="all",
        choices=[
            "argmax_nobarrier",
            "argmax_barrier",
            "split_k32",
            "argmax_stage_gather",
            "argmax_stage_slice",
            "argmax_stage_untilize",
            "argmax_stage_argmax",
            "raw_gather",
            "argmax_shipped",
            "all",
        ],
    )
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, DEVICES), trace_region_size=50_000_000)
    try:
        ccl = TT_CCL(mesh)
        torch.manual_seed(0)
        host = torch.randn(1, 1, SLOTS, VOCAB) * 4.0
        logits = ttnn.from_torch(
            host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        assert int(logits.shape[-1]) == LOCAL_VOCAB, logits.shape
        legs = ["argmax_nobarrier", "argmax_barrier", "split_k32"] if args.leg == "all" else [args.leg]
        for leg in legs:
            run_leg(mesh, ccl, logits, leg, reps=args.reps, traced=not args.no_trace)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
