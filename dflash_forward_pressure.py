# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SCRATCH -- untracked on purpose; do not commit, do not add to a PR.
#
"""Find the user count a prefill pipeline can serve: caches resident AND a real forward running.

Fitting and serving are different limits. The MoE routed-expert op takes a ~300 MB DRAM buffer per
chip for the duration of a chunk, so the last user whose caches fit is not the last user that can be
prefilled. Only a forward run against resident caches separates the two.

The exploration holds one built stack and grows the resident cache set, running a real chunk at
every step:

    for n in baseline, baseline+1, ...:
        allocate (n - baseline) more users' caches
        run a chunk at positions across the whole sequence

  max_fit_users      last n whose caches allocate
  max_served_users   last n whose caches allocate and whose forward still runs

Growing the resident set stands in for relaunching at n because the transient DRAM a chunk takes is
set by chunk size and model dims, not by the user count -- the caches are the only term that moves.
Confirm mode relaunches at one n with the whole stack built at that count, which is the true
configuration, and is what validates the substitution.

Each rank fabricates its own chunk input rather than receiving the previous rank's activation, so
the outputs are meaningless. The op sequence and its allocations are real, which is what is measured.

  PREFILL_PRESSURE_MODE=confirm  verdict for the configured count only, no walk
  PREFILL_PRESSURE_MODE=fit      allocation verdict only, no forward -- the cheap half of confirm
  PREFILL_PRESSURE_POSITIONS=N   sweep N positions across the sequence (default 6)
  PREFILL_PRESSURE_FROM/_TO=N    the candidate counts to walk (default baseline+1 .. +41)
"""

from __future__ import annotations

import os
import time

import ttnn
from loguru import logger

from dflash_capacity_probe import _allocate_extra, _dram, _free, allocate_cache, build_model, open_stack, shutdown


def _positions(max_seq_len: int, chunk: int, count: int) -> list[int]:
    last = ((max_seq_len // chunk) - 1) * chunk
    if count <= 1:
        return [last]
    step = max(chunk, last // (count - 1))
    return sorted({0, last} | {(i * step // chunk) * chunk for i in range(count)})


def _run_chunk(runtime, kv_caches, *, slot_id: int, start: int, chunk: int) -> float:
    inp = runtime.make_chunk_input([0] * chunk)
    # The real ack transport is wired by the serving driver, and `zero_pad_and_ack` is gated on it.
    # A no-op sink keeps that branch compiled without needing a producer attached.
    prev_sink = runtime._layer_completion_sink
    runtime._layer_completion_sink = lambda *_args, **_kwargs: None
    t0 = time.perf_counter()
    try:
        runtime.prefill_chunk(inp, kv_caches, slot_id=slot_id, actual_start=start, actual_end=start + chunk)
        ttnn.synchronize_device(runtime.mesh_device)
    finally:
        runtime._layer_completion_sink = prev_sink
    return (time.perf_counter() - t0) * 1000.0


def sweep_positions(*, runtime, kv_caches, params, mesh_device, rank, count, label):
    """Run a chunk at positions across the sequence.

    Attention reads the cache up to the chunk's start, so the transient peak grows with position and
    a warm-up at chunk zero says nothing about the tail. Returns the last position that ran, whether
    all of them ran, and why the first failure failed."""
    chunk = params.chunk_size
    worst = 0
    for start in _positions(params.max_seq_len, chunk, count):
        view = _dram(mesh_device)
        try:
            ms = _run_chunk(runtime, kv_caches, slot_id=0, start=start, chunk=chunk)
            detail = f"{ms:.0f} ms"
            ok = True
        except Exception as exc:
            detail = str(exc).strip().splitlines()[0][:160]
            ok = False
        logger.info(
            f"[pressure rank {rank}] {label} pos={start} ok={ok} {detail} | "
            f"free/bank={view.total_bytes_free_per_bank} contig/bank={view.largest_contiguous_bytes_free_per_bank}"
        )
        if not ok:
            return worst, False, f"pos={start}: {detail}"
        worst = start
    return worst, True, ""


def explore(*, mesh_device, hf_config, params, runtime, kv_caches, rank, num_ranks, positions, candidates):
    """Grow the resident cache set one user at a time, running a real forward at every step.

    Every rank walks the same candidates behind a barrier, and a rank past its own ceiling keeps
    walking instead of leaving. Ranks bind at different counts, and the ones still below their
    ceiling need the rest of the pipeline alive to keep running chunks."""
    view = _dram(mesh_device)
    banks = view.num_banks
    free = view.total_bytes_free_per_bank * banks

    before = _dram(mesh_device).total_bytes_allocated_per_bank
    one = _allocate_extra(
        mesh_device=mesh_device, hf_config=hf_config, params=params, runtime=runtime, extra_users=1
    )
    per_user = (_dram(mesh_device).total_bytes_allocated_per_bank - before) * banks
    _free(one)

    max_fit = max_served = params.num_users
    free_at_served = free
    worst = 0
    first_fit_fail = first_served_fail = ""
    for n in candidates:
        ttnn.distributed_context_barrier()
        tensors = []
        fit = True
        try:
            tensors = _allocate_extra(
                mesh_device=mesh_device,
                hf_config=hf_config,
                params=params,
                runtime=runtime,
                extra_users=n - params.num_users,
            )
        except Exception as exc:
            fit = False
            first_fit_fail = first_fit_fail or f"{n}: {str(exc).strip().splitlines()[0][:120]}"

        remaining = _dram(mesh_device).total_bytes_free_per_bank * banks
        try:
            pos, served, why = sweep_positions(
                runtime=runtime,
                kv_caches=kv_caches,
                params=params,
                mesh_device=mesh_device,
                rank=rank,
                count=positions,
                label=f"users={n} fit={fit}",
            )
        finally:
            _free(tensors)

        if fit and n > max_fit:
            max_fit = n
        if fit and served and n > max_served:
            max_served, free_at_served, worst = n, remaining, pos
        elif fit and not served:
            first_served_fail = first_served_fail or f"{n}: {why}"

    logger.info(
        f"COMBINED rank={rank} num_ranks={num_ranks} is_last={params.is_last_rank} "
        f"layers={params.num_layers} drafter={getattr(runtime, 'drafter', None) is not None} "
        f"baseline_users={params.num_users} worst_pos={worst} "
        f"free_bytes_per_chip={free} per_user_bytes_per_chip={per_user} "
        f"max_fit_users={max_fit} max_served_users={max_served} "
        f"free_at_served_bytes_per_chip={free_at_served} "
        f"first_fit_fail={first_fit_fail!r} first_forward_fail={first_served_fail!r}"
    )


def main() -> None:
    mesh_device, hf_config, params, rank, num_ranks = open_stack()
    positions = int(os.environ.get("PREFILL_PRESSURE_POSITIONS", 6))
    mode = os.environ.get("PREFILL_PRESSURE_MODE", "explore")

    # Fitting the caches and running against them are separate verdicts about the same count, and a
    # launch that dies inside the bring-up cannot say which of the two it lost. Score them apart.
    # Both bring-up stages allocate per user: the drafter's context cache comes up with the model on
    # the last rank, the verifier's after it, so either can be the one that runs out.
    runtime = kv_caches = None
    fit, why = True, ""
    try:
        runtime = build_model(mesh_device=mesh_device, hf_config=hf_config, params=params)
        kv_caches = allocate_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    except Exception as exc:
        stage = "build_runtime" if runtime is None else "allocate_kv_cache"
        fit, why = False, f"{stage}: {str(exc).strip().splitlines()[0][:160]}"

    # A forward is a pipeline operation, so a rank whose caches did not fit would leave the rest
    # waiting on activations that never arrive. Agree on the fit verdict before anyone starts one.
    pipeline_fits = all(ttnn.distributed_context_allgather_int(int(fit)))

    worst, served = 0, False
    if pipeline_fits and mode != "fit":
        # `compile` runs a chunk itself, so a count whose caches fit but whose forward does not can
        # only fail here. That is a verdict about the count, not a crash.
        try:
            runtime.compile(kv_caches)
            worst, served, why = sweep_positions(
                runtime=runtime,
                kv_caches=kv_caches,
                params=params,
                mesh_device=mesh_device,
                rank=rank,
                count=positions,
                label="configured",
            )
        except Exception as exc:
            worst, served, why = 0, False, f"compile: {str(exc).strip().splitlines()[0][:160]}"
    elif fit and not pipeline_fits:
        why = "another rank could not allocate"

    view = _dram(mesh_device)
    logger.info(
        f"SERVED rank={rank} num_ranks={num_ranks} users={params.num_users} fit={int(fit)} "
        f"ok={int(served)} worst_pos={worst} "
        f"free_bytes_per_chip={view.total_bytes_free_per_bank * view.num_banks} why={why!r}"
    )

    if served and mode == "explore":
        lo = int(os.environ.get("PREFILL_PRESSURE_FROM", params.num_users + 1))
        hi = int(os.environ.get("PREFILL_PRESSURE_TO", lo + 40))
        explore(
            mesh_device=mesh_device,
            hf_config=hf_config,
            params=params,
            runtime=runtime,
            kv_caches=kv_caches,
            rank=rank,
            num_ranks=num_ranks,
            positions=positions,
            candidates=range(max(lo, params.num_users + 1), hi + 1),
        )

    shutdown(mesh_device, rank)


if __name__ == "__main__":
    main()
