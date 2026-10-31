# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Prefix-caching (APC) diagnostics for MiMo under vLLM lane-DP, on with MIMO_APC_DEBUG=1.

The TT lanes run in the EngineCore process, so this wraps the plugin's ``TTLaneCoordinator`` from the model side and
logs, per request: the lane it is bound to and every lane's load at admission, its block hashes, the prefix-cache hit
each lane's KV cache manager would give it (total and per KV cache group), the scheduler's num_computed_tokens and
first block ids when it is scheduled, and when it leaves its lane. ``prefill_forward`` logs each row's lane, slot,
start and page tables (``log_prefill_row``). Every hook is best effort: a failure logs once and never reaches the
engine.
"""

import functools
import hashlib
import os
import traceback

from loguru import logger

ENABLED = os.environ.get("MIMO_APC_DEBUG", "0") not in ("", "0")
_installed = False
_failed = set()


def _guard(name, fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:  # diagnostics never break serving
        if name not in _failed:
            _failed.add(name)
            logger.warning(f"MiMo APC debug: {name} failed: {e!r}\n{traceback.format_exc()}")
        return None


def _hex(h):
    try:
        return bytes(h).hex()[:16]
    except Exception:
        return str(h)[:16]


def _loads(coord):
    return [(len(s.waiting), len(s.running), len(s.requests)) for s in coord.lanes]


def _cache_state(sched):
    pool = sched.kv_cache_manager.block_pool
    cached = getattr(pool, "cached_block_hash_to_block", None)
    n_cached = len(cached) if cached is not None and hasattr(cached, "__len__") else -1
    return pool.get_num_free_blocks(), n_cached


def _lane_hit(sched, req):
    """(hit tokens, per-group hit tokens) the lane's prefix cache would give ``req`` now (no allocation, no stats)."""
    kvm = sched.kv_cache_manager
    if not kvm.enable_caching:
        return "caching off"
    coord = kvm.coordinator
    max_len = req.num_tokens - 1
    res = coord.find_longest_cache_hit(req.block_hashes, max_len)
    total = res[1]
    per_group = None
    if hasattr(coord, "find_longest_cache_hit_per_group"):
        per_group = _guard(
            "per-group hit", lambda: list(coord.find_longest_cache_hit_per_group(req.block_hashes, max_len)[1])
        )
    return total, per_group


def _admission(coord, request, loads_before):
    lane = coord._req_to_lane.get(request.request_id)
    hashes = request.block_hashes
    hits = []
    for i, sched in enumerate(coord.lanes):
        hits.append((i, _guard("lane hit", _lane_hit, sched, request), _guard("cache state", _cache_state, sched)))
    toks = list(request.prompt_token_ids or [])
    logger.info(
        f"MiMo APC debug: admit {request.request_id} -> lane {lane}; loads before (waiting, running, requests) "
        f"{loads_before}; {request.num_prompt_tokens} prompt tokens (sha {_sha(toks)}, head {_sha(toks[:22016])}), "
        f"{len(hashes)} block hashes [0] {_hex(hashes[0]) if hashes else None} [-1] {_hex(hashes[-1]) if hashes else None}, "
        f"cache_salt {getattr(request, 'cache_salt', None)!r}, skip_read "
        f"{getattr(request, 'skip_reading_prefix_cache', None)}; per lane (hit tokens, per group), (free blocks, cached "
        f"hashes): {hits}"
    )


def _sha(toks):
    return hashlib.sha1(",".join(map(str, toks)).encode()).hexdigest()[:12]


def _scheduled(coord, merged):
    from vllm_tt_plugin.lane_scheduler import _get_tt_step_state

    state = _get_tt_step_state(merged)
    if state is None:
        return
    for lane, out in enumerate(state.lane_outputs):
        sched = coord.lanes[lane]
        for nr in out.scheduled_new_reqs:
            req = sched.requests.get(nr.req_id)
            blocks = [list(g[:3]) for g in (nr.block_ids or ())]
            logger.info(
                f"MiMo APC debug: lane {lane} schedules new {nr.req_id}: num_computed_tokens {nr.num_computed_tokens} "
                f"(cache hit), {out.num_scheduled_tokens.get(nr.req_id)} tokens this step, prompt "
                f"{len(nr.prompt_token_ids or ())}, num_cached_tokens {getattr(req, 'num_cached_tokens', None)}, first "
                f"block ids per group {blocks}"
            )


def _left(coord, before):
    for lane, sched in enumerate(coord.lanes):
        gone = before[lane] - set(sched.requests)
        for rid in sorted(gone):
            logger.info(
                f"MiMo APC debug: {rid} left lane {lane}; loads now {_loads(coord)}, lane (free blocks, cached hashes) "
                f"{_guard('cache state', _cache_state, sched)}"
            )


def install():
    """Wrap the plugin's TTLaneCoordinator (idempotent; no-op unless MIMO_APC_DEBUG)."""
    global _installed
    if not ENABLED or _installed:
        return
    try:
        from vllm_tt_plugin.lane_scheduler import TTLaneCoordinator as C
    except Exception as e:
        logger.warning(f"MiMo APC debug: no TTLaneCoordinator to wrap ({e!r})")
        return
    _installed = True

    add_request, schedule, update, finish, reset = (
        C.add_request,
        C.schedule,
        C.update_from_output,
        C.finish_requests,
        C.reset_prefix_cache,
    )

    @functools.wraps(add_request)
    def add_request_(self, request):
        loads = _guard("loads", _loads, self)
        out = add_request(self, request)
        _guard("admission", _admission, self, request, loads)
        return out

    @functools.wraps(schedule)
    def schedule_(self, *args, **kwargs):
        merged = schedule(self, *args, **kwargs)
        _guard("scheduled", _scheduled, self, merged)
        return merged

    @functools.wraps(update)
    def update_(self, *args, **kwargs):
        before = _guard("snapshot", lambda: [set(s.requests) for s in self.lanes])
        out = update(self, *args, **kwargs)
        if before is not None:
            _guard("left", _left, self, before)
        return out

    @functools.wraps(finish)
    def finish_(self, request_ids, finished_status):
        before = _guard("snapshot", lambda: [set(s.requests) for s in self.lanes])
        out = finish(self, request_ids, finished_status)
        logger.info(f"MiMo APC debug: finish_requests({request_ids!r}, {finished_status})")
        if before is not None:
            _guard("left", _left, self, before)
        return out

    @functools.wraps(reset)
    def reset_(self, *args, **kwargs):
        logger.warning(
            f"MiMo APC debug: reset_prefix_cache{args or ''}{kwargs or ''} from\n{''.join(traceback.format_stack(limit=12))}"
        )
        return reset(self, *args, **kwargs)

    C.add_request, C.schedule, C.update_from_output, C.finish_requests, C.reset_prefix_cache = (
        add_request_,
        schedule_,
        update_,
        finish_,
        reset_,
    )
    logger.info("MiMo APC debug: TTLaneCoordinator wrapped (admission, schedule, finish, reset)")


def log_prefill_row(i, lane, slot, scheduled_start, start, end, cached, tokens_row, group_rows, block_sizes):
    """One prefill row as the model runs it: ``scheduled_start`` is the plugin's start_pos (the scheduler's
    num_computed_tokens), ``start`` where the model starts (aligned, or 0 for a recomputed short prefix)."""
    toks = tokens_row[:end].tolist() if hasattr(tokens_row, "tolist") else list(tokens_row[:end])
    tables = {}
    for g, t in group_rows.items():
        t = [int(b) for b in t]
        last = (end - 1) // block_sizes.get(g, 64)
        tables[g] = (t[:4], t[last] if 0 <= last < len(t) else None)
    logger.info(
        f"MiMo APC debug: prefill row {i} lane {lane} slot {slot}: start_pos {scheduled_start} -> {start}, end {end} "
        f"({end - start} tokens), cached blocks {cached}, tokens sha {_sha(toks[:22016])} (first 22016 of the row), "
        f"page tables per group (first 4 ids, id at end - 1) {tables}"
    )
