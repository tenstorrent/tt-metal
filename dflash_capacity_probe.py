# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SCRATCH -- issue #56487 experiment. Untracked on purpose; do not commit, do not add to a PR.
#
"""Measure how many concurrent users the prefill KV caches fit in DRAM, with and without DFlash.

Stands in for ``prefill_runner`` and reuses its env-resolved config, so the mesh, adapter, pipeline
split and weights are identical to a real run. Instead of serving, it allocates *additional* users'
caches on top of the resident baseline until the allocator refuses.

Weight load dominates a launch by an order of magnitude, so a sweep that relaunches per candidate
costs hours; holding the weights and reallocating only the caches costs seconds per candidate.

The answer is per rank and the pipeline is bound by its worst one -- DFlash builds the drafter cache
on the LAST rank only, which is the whole point of measuring per rank. Each rank prints one
``CAPACITY`` line; take the minimum across them.

  PREFILL_CAPACITY_PROBE_COMPILE=0   skip runtime.compile (faster, ignores program-time DRAM)
"""

from __future__ import annotations

import os

import ttnn
from loguru import logger

from models.demos.common.prefill.adapter import PrefillRunParams
from models.demos.common.prefill.runners import prefill_runner as R
from models.demos.common.prefill.runners.runner_utils import compute_layer_split, open_mesh_device
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_dflash_kv_cache, allocate_mla_kvpe_cache

# A prediction that misses by more than this many users means per-user cost is not linear the way
# this assumes. Bound the walk rather than let a bad model become an unbounded allocate/free loop.
_MAX_WALK_STEPS = 24


def _dram(mesh_device):
    """DRAM allocator state, per bank. Every chip in the mesh runs the same program and holds the
    same cache shards, so one view describes them all."""
    return ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)


def _allocate_extra(*, mesh_device, hf_config, params, runtime, extra_users):
    """Allocate ``extra_users`` more users' caches alongside the resident ones; return them to free.

    The drafter cache exists only where the drafter was built, so a non-last rank measures the
    verifier cost alone."""
    tensors = []
    if extra_users <= 0:
        return tensors
    tensors.append(
        allocate_mla_kvpe_cache(
            mesh_device=mesh_device,
            hf_config=hf_config,
            max_seq_len=params.max_seq_len,
            mesh_shape=params.mesh_shape,
            sp_axis=params.sp_axis,
            num_layers=params.num_layers,
            num_users=extra_users,
        ).storage
    )
    drafter = getattr(runtime, "drafter", None)
    if drafter is not None and params.is_last_rank:
        k, v = allocate_dflash_kv_cache(
            mesh_device,
            drafter.config,
            params.max_seq_len,
            sp_axis=params.sp_axis,
            tp_axis=params.tp_axis,
            num_users=extra_users,
        )
        tensors += [k, v]
    return tensors


def _free(tensors):
    for t in reversed(tensors):
        try:
            ttnn.deallocate(t)
        except Exception as exc:
            logger.warning(f"[capacity] deallocate failed: {exc}")


def _try(*, mesh_device, hf_config, params, runtime, total_users):
    tensors = []
    try:
        tensors = _allocate_extra(
            mesh_device=mesh_device,
            hf_config=hf_config,
            params=params,
            runtime=runtime,
            extra_users=total_users - params.num_users,
        )
        return True, f"dram_allocated_per_bank={_dram(mesh_device).total_bytes_allocated_per_bank}"
    except Exception as exc:  # the allocator reports exhaustion as a plain RuntimeError
        return False, str(exc).strip().splitlines()[0][:200]
    finally:
        _free(tensors)


def probe(*, mesh_device, hf_config, params, runtime, rank, num_ranks):
    has_drafter = getattr(runtime, "drafter", None) is not None
    view = _dram(mesh_device)
    banks, free_per_bank = view.num_banks, view.total_bytes_free_per_bank
    logger.info(
        f"[capacity rank {rank}/{num_ranks}] baseline_users={params.num_users} layers={params.num_layers} "
        f"is_last={params.is_last_rank} drafter={has_drafter} | dram banks={banks} "
        f"total/bank={view.total_bytes_per_bank} allocated/bank={view.total_bytes_allocated_per_bank} "
        f"free/bank={free_per_bank} largest_contig/bank={view.largest_contiguous_bytes_free_per_bank}"
    )

    # One user's marginal cost, measured rather than derived: the tile and shard rounding the cache
    # layout applies is not recoverable from the model config alone.
    before = _dram(mesh_device).total_bytes_allocated_per_bank
    one = _allocate_extra(
        mesh_device=mesh_device, hf_config=hf_config, params=params, runtime=runtime, extra_users=1
    )
    per_user = _dram(mesh_device).total_bytes_allocated_per_bank - before
    _free(one)
    if per_user <= 0:
        raise RuntimeError(f"[capacity rank {rank}] per-user cost measured as {per_user} B/bank")

    predicted = params.num_users + free_per_bank // per_user
    logger.info(
        f"[capacity rank {rank}] per_user={per_user} B/bank ({per_user * banks} B/chip) "
        f"-> predicted max_users={predicted}"
    )

    # Walk from the prediction to the true boundary: the allocator enforces contiguity, not total
    # free bytes, so the prediction can sit on either side of it.
    users = predicted
    fits, detail = _try(mesh_device=mesh_device, hf_config=hf_config, params=params, runtime=runtime, total_users=users)
    logger.info(f"[capacity rank {rank}] users={users} fits={fits} {detail}")
    step = 1 if fits else -1
    confirmed = users if fits else 0
    for _ in range(_MAX_WALK_STEPS):
        users += step
        if users < params.num_users:
            break
        fits, detail = _try(
            mesh_device=mesh_device, hf_config=hf_config, params=params, runtime=runtime, total_users=users
        )
        logger.info(f"[capacity rank {rank}] users={users} fits={fits} {detail}")
        if step > 0:
            if not fits:
                break
            confirmed = users
        elif fits:
            confirmed = users
            break
    else:
        logger.warning(f"[capacity rank {rank}] walk did not converge in {_MAX_WALK_STEPS} steps")

    logger.info(
        f"CAPACITY rank={rank} num_ranks={num_ranks} is_last={params.is_last_rank} "
        f"layers={params.num_layers} drafter={has_drafter} dflash_env={R.DFLASH_ENABLED} "
        f"predicted={predicted} max_users={confirmed} "
        f"per_user_bytes_per_chip={per_user * banks} free_bytes_per_chip={free_per_bank * banks}"
    )
    return confirmed


def open_stack():
    """Bring the mesh up and resolve this rank's slice of the pipeline, with nothing allocated yet.

    Split out of ``build_stack`` because the two allocating stages after it -- the drafter cache in
    ``build_runtime`` and the verifier cache in ``allocate_kv_cache`` -- are what a capacity run is
    scoring, and a caller cannot score a stage it cannot reach."""
    R._print_config()
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    num_ranks = int(ttnn.distributed_context_get_size())
    R._assert_ranks_agree_on_config(rank, num_ranks)

    layer_split = compute_layer_split(R.NUM_LAYERS, num_ranks, R.ADAPTER.layer_split_boundaries(R.NUM_LAYERS))
    first_layer_idx, num_my_layers = layer_split[rank]
    is_last_rank = rank == num_ranks - 1

    mesh_device = open_mesh_device(
        R.GLOBAL_MESH_SHAPE, R.MODEL_CFG, l1_small_size=R._L1_SMALL_SIZE, trace_region_size=R._TRACE_REGION_SIZE
    )
    hf_config = R.ADAPTER.load_hf_config()
    hf_config.max_seq_len = R.MAX_SEQ_LEN

    params = PrefillRunParams(
        mesh_shape=R.GLOBAL_MESH_SHAPE,
        num_layers=num_my_layers,
        first_layer_idx=first_layer_idx,
        is_first_rank=rank == 0,
        is_last_rank=is_last_rank,
        max_seq_len=R.MAX_SEQ_LEN,
        chunk_size=R.CHUNK_SIZE,
        num_users=R.NUM_USERS,
        capacity_factor=R.CAPACITY_FACTOR,
        num_links=2 if R.is_blackhole() else 1,
        gate_mode_name=R._gate_mode_name,
        kv_only_last_layer=is_last_rank,
        dflash_enabled=R.DFLASH_ENABLED,
        dflash_checkpoint_path=R.DFLASH_MODEL,
        weight_cache_path=R.ADAPTER.weight_cache_path(R.GLOBAL_MESH_SHAPE),
        sparse_kv_cache_format=R.ADAPTER.default_sparse_kv_cache_format,
        use_trace=R.USE_TRACE,
        overlap_shared_expert_with_dispatch=os.environ.get("PREFILL_OVERLAP_SHARED_EXPERT", "1") == "1",
    )
    return mesh_device, hf_config, params, rank, num_ranks


def build_model(*, mesh_device, hf_config, params):
    """Load the weights and, under DFlash on the last rank, the drafter's own context KV cache."""
    return R.ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)


def allocate_cache(*, mesh_device, hf_config, params):
    return R.ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)


def build_stack():
    """Bring up the same stack a real prefill run does, and hand back its pieces.

    Shared with the forward-pressure probe so both measure the identical mesh, split and weights."""
    mesh_device, hf_config, params, rank, num_ranks = open_stack()
    runtime = build_model(mesh_device=mesh_device, hf_config=hf_config, params=params)
    kv_caches = allocate_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    return mesh_device, hf_config, params, runtime, kv_caches, rank, num_ranks


def shutdown(mesh_device, rank):
    ttnn.distributed_context_barrier()
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[capacity rank {rank}] shutdown complete")


def main() -> None:
    mesh_device, hf_config, params, runtime, kv_caches, rank, num_ranks = build_stack()
    if os.environ.get("PREFILL_CAPACITY_PROBE_COMPILE", "1") == "1":
        runtime.compile(kv_caches)

    probe(
        mesh_device=mesh_device, hf_config=hf_config, params=params, runtime=runtime, rank=rank, num_ranks=num_ranks
    )
    shutdown(mesh_device, rank)


if __name__ == "__main__":
    main()
