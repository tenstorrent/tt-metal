# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Pure capacity declarations for one TP4 Nemotron lane; no device imports.

The aggregate budget constrains simultaneous resident tokens, not the logical
context of any request. It does not promise 32 simultaneous 262144-token
requests. Full-stack measurements must separately establish that the resolved
allocation, recurrent state, scratch and traces fit the target hardware.

A caller using these constrained pools must enforce aggregate admission and
provide nonoverlapping physical page tables. The generator's default modulo
page mappings do not provide that scheduling policy.
"""

import operator
import os

from .precision import load_precision

MAX_CONTEXT = 262144
MAX_SLOTS = 32
BLOCK_SIZE = 32
AGGREGATE_TOKEN_BUDGET = 262144
OUTPUT_RESERVATION_PER_SLOT = 32
ATTENTION_LAYER_INDICES = (5, 12, 19, 26, 33, 42)


def _integer(name, value):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _validate(*, num_devices, tt_data_parallel, max_model_len, max_num_seqs):
    num_devices = _integer("num_devices", num_devices)
    tt_data_parallel = _integer("tt_data_parallel", tt_data_parallel)
    max_model_len = _integer("max_model_len", max_model_len)
    max_num_seqs = _integer("max_num_seqs", max_num_seqs)
    if num_devices != 4 or tt_data_parallel != 1:
        raise ValueError("Capacity declaration requires one TP4 lane: num_devices=4, tt_data_parallel=1")
    if not 1 <= max_model_len <= MAX_CONTEXT:
        raise ValueError(f"max_model_len must be between 1 and {MAX_CONTEXT}")
    if not 1 <= max_num_seqs <= MAX_SLOTS:
        raise ValueError(f"max_num_seqs must be between 1 and {MAX_SLOTS}")
    return max_model_len, max_num_seqs


def get_max_tokens_all_users(
    *, model_name=None, num_devices=4, tt_data_parallel=1, max_model_len=MAX_CONTEXT, max_num_seqs=1
):
    """Return aggregate resident-token budget per lane, BEFORE output reserve.

    Accept the preload query's keyword names without importing its caller.
    ``model_name`` is opaque because the caller already resolved this model.
    This is a declared allocation policy, not a runtime-validation verdict.
    A caller adding output headroom must add it exactly once; do not return
    ``physical_token_capacity`` from :func:`cache_capacity_plan` here.
    """
    _validate(
        num_devices=num_devices,
        tt_data_parallel=tt_data_parallel,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
    )
    # perf/multiuser: the attention KV per token is tiny here (6 layers, 1 KV head per chip, BFP8
    # ~1.6 KB/token/chip), so a pool sized for one max-length request starves concurrency.
    # NEMOTRON_KV_POOL_TOKENS overrides the aggregate resident-token budget.
    override = os.environ.get("NEMOTRON_KV_POOL_TOKENS")
    if override:
        return max(int(override), max_model_len)
    return AGGREGATE_TOKEN_BUDGET


def cache_capacity_plan(
    *,
    max_seq_len=MAX_CONTEXT,
    max_batch_size=1,
    num_devices=4,
    tt_data_parallel=1,
    model_name=None,
    precision_config=None,
):
    """Describe six independent pools including one output block per slot.

    The returned ``per_layer_specs`` use the existing cache_specs string dtype
    convention, so resolving them needs no new allocator or scheduler adapter.
    Each tuple describes one K/V pair; K and V remain separate allocations.
    Input-validation limits below are not claims that all workloads were run.
    """
    max_seq_len, max_batch_size = _validate(
        num_devices=num_devices,
        tt_data_parallel=tt_data_parallel,
        max_model_len=max_seq_len,
        max_num_seqs=max_batch_size,
    )
    budget = get_max_tokens_all_users(
        model_name=model_name,
        num_devices=num_devices,
        tt_data_parallel=tt_data_parallel,
        max_model_len=max_seq_len,
        max_num_seqs=max_batch_size,
    )
    reservation = OUTPUT_RESERVATION_PER_SLOT * max_batch_size
    blocks = (budget + reservation + BLOCK_SIZE - 1) // BLOCK_SIZE
    precision = load_precision(precision_config)
    return {
        "logical_context_per_request": max_seq_len,
        "num_slots": max_batch_size,
        "mesh_shape": (1, 4),
        "tt_data_parallel": 1,
        "block_size": BLOCK_SIZE,
        "aggregate_token_budget": budget,
        "output_reservation_per_slot": OUTPUT_RESERVATION_PER_SLOT,
        "output_reservation_tokens": reservation,
        "physical_blocks_per_pool": blocks,
        "physical_token_capacity": blocks * BLOCK_SIZE,
        "attention_layer_indices": ATTENTION_LAYER_INDICES,
        "per_layer_specs": tuple(((blocks, 1, BLOCK_SIZE, 128), precision["kv_cache_dtype"], i) for i in range(6)),
        "precision_config_id": precision["config_id"],
        "input_validation_limits": {
            "logical_context": (1, MAX_CONTEXT),
            "slots": (1, MAX_SLOTS),
            "num_devices": 4,
            "tt_data_parallel": 1,
        },
        "admission_requirement": "Enforce aggregate residency and supply nonoverlapping physical page tables.",
        "evidence_scope": "Allocation policy and arithmetic only; requires separate full-stack capacity evidence.",
    }
