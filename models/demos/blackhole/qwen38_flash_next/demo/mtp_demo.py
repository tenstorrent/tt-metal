# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP speculative decode for the text demo: the chat server's chain, session and lane scheduler without the server.

Batch 1: ``open_mtp_chain`` builds the greedy traced chain with K drafts and ``run_mtp_single`` generates one request
through ``Qwen38ChatSession.complete``.  Batch B (2..8): ``open_mtp_chain`` with a ``Qwen38LanesSession`` allocates the
lane states in the chain's warm hook, ``run_mtp_lanes`` admits every prompt and runs the lane scheduler to the end.
"""

import queue
import time
from types import SimpleNamespace
from typing import Any, Callable, Sequence

from loguru import logger

import ttnn
from models.demos.blackhole.qwen38_flash_next.tools import hardware_profiles
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    LONG_CHUNKS_BYTES_PER_BANK_AFTER_CAPTURES,
    RESIDENT_POST_BUILD_BYTES_PER_BANK_UPPER_BOUND,
    Qwen38ChatChainError,
    Qwen38ChatSession,
    construct_chain,
    lanes_capacity_admission,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lane_scheduler import Qwen38LaneScheduler, Qwen38LaneTicket
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lanes_session import Qwen38LanesSession
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import (
    TRACE_ALLOC_TRACKING,
    prefill_form,
    prepare_vllm_construction,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import fused as fused_module
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import gdn_rows_scan as gdn_rows_scan_module


def _lanes_warm_hook(mesh_device, lanes_session: Qwen38LanesSession, allocated_context: int, long_chunks: bool):
    """The lanes' admission on the live allocator, then their allocation and compile; runs after the resident build, the
    MTP states and the warm pass, before any capture, so what the chain still allocates is reserved on the free side."""

    route = tuple(
        int(mesh_device.get_device_id(ttnn.MeshCoordinate(0, column))) for column in range(mesh_device.shape[1])
    )

    def warm_hook(opened_chain) -> None:
        live = hardware_profiles.symmetric_mesh_dram_memory(mesh_device, route)
        reserved = RESIDENT_POST_BUILD_BYTES_PER_BANK_UPPER_BOUND + int(
            opened_chain.mtp.admission["mtp_growth_estimate_bytes_per_bank"]["traces"]
        )
        if long_chunks:
            reserved += LONG_CHUNKS_BYTES_PER_BANK_AFTER_CAPTURES
        admission = lanes_capacity_admission(
            allocated_context,
            lanes=lanes_session.lanes,
            drafts=lanes_session.drafts,
            live=live,
            reserved_bytes_per_bank=reserved,
            gdn_rows_scan=fused_module.enabled(gdn_rows_scan_module.NAME),
        )
        lanes_session.admission = admission
        if not admission["fits"]:
            raise Qwen38ChatChainError(
                f"{lanes_session.lanes} lanes with {lanes_session.drafts} drafts refused at allocated context "
                f"{allocated_context}: {admission['decided_by']['shortfalls']} (required "
                f"{admission['required_free_bytes_per_bank']} B per bank against {admission['free_bytes_per_bank']} "
                f"free, contiguous {admission['largest_contiguous_bytes_free_per_bank']} against "
                f"{admission['required_largest_contiguous_bytes_per_bank']})"
            )
        lanes_session.prepare(opened_chain)

    return warm_hook


def open_mtp_chain(
    mesh_device,
    checkpoint,
    max_seq_len: int,
    drafts: int,
    *,
    lanes_session: Qwen38LanesSession | None = None,
    marker: Callable[[str], Any] = logger.info,
):
    """The greedy MTP chain at ``max_seq_len``'s resident context, built from the vLLM adapter's preparation and
    prefill form.  The single-stream form reads the verify rows on the second command queue (needs the 2-queue mesh); the
    lanes form does not.  Raises ``Qwen38ChatChainError`` when the live admission refuses the MTP states or the lanes.
    """

    if not TRACE_ALLOC_TRACKING:
        raise ValueError("the trace chain needs TT_METAL_TRACE_ALLOC_TRACKING=1 exported before ttnn is imported")
    prepared, allocated_context = prepare_vllm_construction(
        SimpleNamespace(_name_or_path=str(checkpoint)), mesh_device, max_seq_len
    )
    long_chunks, slab_rows = prefill_form()
    warm_hook = None
    if lanes_session is not None:
        warm_hook = _lanes_warm_hook(
            mesh_device, lanes_session, allocated_context, long_chunks or slab_rows is not None
        )
    return construct_chain(
        prepared,
        mesh_device,
        marker=marker,
        chunked_prefill=True,
        sampling=False,
        long_chunks=long_chunks,
        slab_rows=slab_rows,
        mtp=drafts,
        mtp_ple_early=lanes_session is None,
        warm_hook=warm_hook,
    )


def run_mtp_single(
    session: Qwen38ChatSession, prompt_ids: Sequence[int], max_new_tokens: int, stop_ids: Sequence[int] = ()
) -> dict[str, Any]:
    """One request through the pass loop, from a reset session (a kept prefix would make the TTFT warm).

    The chain's pass counters include the bootstrap pass (the pass after the prefill that has placeholder drafts and
    emits one token); ``passes`` and ``tokens_per_pass`` leave it out: the bootstrap pass emits one token and accepts
    none, so ``passes = chain passes - 1`` and ``tokens_per_pass = (generated - 2) / passes`` (the first token comes
    from the prefill, the second from the bootstrap pass; the last pass is cut at ``max_new_tokens``)."""

    session.reset()
    completion = session.complete(list(prompt_ids), max_new_tokens, stop_ids=tuple(stop_ids), prefill_mode="chunked")
    tokens = list(completion.token_ids)
    mtp = completion.mtp
    passes = mtp["passes"] - 1
    return {
        "tokens": tokens,
        "finish": completion.finish_reason,
        "ttft_s": completion.ttft_seconds,
        "prefill_s": completion.prefill_seconds,
        "decode_s": completion.decode_seconds,
        "decode_ms_per_token": 1000.0 * completion.decode_seconds / (len(tokens) - 1),
        "generated_tokens": len(tokens),
        "chain_passes": mtp["passes"],
        "chain_accepted_drafts": mtp["accepted_drafts"],
        "chain_tokens_per_pass": mtp["tokens_per_pass"],
        "passes": passes,
        "accepted_drafts": mtp["accepted_drafts"],
        "tokens_per_pass": (len(tokens) - 2) / passes if passes > 0 else None,
        "k": mtp["k"],
    }


def _drain_ticket(ticket: Qwen38LaneTicket) -> list[int]:
    """Every delivered id of a finished ticket."""

    tokens: list[int] = []
    while True:
        try:
            token, _finish = ticket.tokens.get_nowait()
        except queue.Empty:
            return tokens
        if token is None:
            return tokens
        tokens.append(token)


def run_mtp_lanes(
    lanes_session: Qwen38LanesSession,
    prompts: Sequence[Sequence[int]],
    drafts: int,
    max_new_tokens: int,
    stop_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """Every prompt submitted at once to the lane scheduler (no stall budget: each admission runs whole, so a lane's
    TTFT is clean), run until all tickets finish.  Token 1 of a lane is delivered when its admission ends; the
    scheduler's passes then carry every lane at once, so the decode window starts at the last admission."""

    lanes = len(prompts)
    scheduler = Qwen38LaneScheduler(lanes=lanes, drafts=drafts, queue_limit=lanes, stall_budget_seconds=None)
    tickets = [
        scheduler.submit(
            Qwen38LaneTicket(
                request_id=f"u{user}",
                prompt_ids=list(prompt),
                max_tokens=max_new_tokens,
                stop_ids=tuple(stop_ids),
                think_budget=None,
            )
        )
        for user, prompt in enumerate(prompts)
    ]
    started = time.perf_counter()
    scheduler.run(lanes_session, forever=False)
    ended = time.perf_counter()
    for ticket in tickets:
        if not ticket.done.is_set():
            raise Qwen38ChatChainError(f"lane ticket {ticket.request_id} did not finish")
    last_admitted = max(ticket.admitted_at for ticket in tickets)
    last_finished = max(ticket.finished_at for ticket in tickets)
    window = last_finished - last_admitted
    rows = []
    for ticket in tickets:
        tokens = _drain_ticket(ticket)
        passes = ticket.passes
        after_first = ticket.committed - 1  # token 1 came from the admission
        rows.append(
            {
                "request_id": ticket.request_id,
                "lane": ticket.lane,
                "finish": ticket.finish,
                "tokens": tokens,
                "ttft_own_prefill_s": ticket.admission_wall_seconds,
                "ttft_from_batch_start_s": ticket.admitted_at - started,
                "decode_s": ticket.finished_at - last_admitted,  # the passes start at the last admission
                "decode_s_from_own_admission": ticket.finished_at - ticket.admitted_at,
                "generated_tokens": ticket.committed,
                "queue_wait_s": ticket.queue_wait,
                "admission_segments_s": dict(ticket.admission_segments),
                "passes": passes,
                "committed": ticket.committed,
                "accepted_drafts": after_first - passes,
                "tokens_per_pass": after_first / passes if passes else None,
            }
        )
    pass_ms = [1000.0 * seconds for seconds in scheduler.pass_seconds]
    pass_ms_sorted = sorted(pass_ms)
    total_after_first = sum(row["committed"] - 1 for row in rows)
    return {
        "lanes": lanes,
        "drafts": drafts,
        "wall_s": ended - started,
        "decode_window_s": window,
        "scheduler_passes": scheduler.passes,
        "pass_ms": pass_ms,
        "pass_ms_mean": sum(pass_ms) / len(pass_ms),
        "pass_ms_median": pass_ms_sorted[len(pass_ms_sorted) // 2],
        "decode_ms_per_token_per_user": 1000.0 * window / max(row["committed"] - 1 for row in rows),
        "aggregate_tok_s": total_after_first / window,
        "aggregate_all_tokens_tok_s": sum(row["committed"] for row in rows) / window,  # token 1 of each lane included
        "total_generated_tokens": sum(row["committed"] for row in rows),
        "mean_per_user_tok_s": sum((row["committed"] - 1) / row["decode_s"] for row in rows) / lanes,
        "per_user_tok_s": total_after_first / window / lanes,
        "tokens_per_pass_mean": sum(row["tokens_per_pass"] for row in rows) / lanes,
        "lanes_rows": rows,
    }
