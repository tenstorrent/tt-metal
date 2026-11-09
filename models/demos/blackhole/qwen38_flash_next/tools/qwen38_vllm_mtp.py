# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""MTP speculative decoding through vllm-tt-plugin's runner-model contract (method ``custom_class``); greedy; batch 1
(the chain's pass loop) or batch 2..8 (the lane chain, ``B x (K + 1) <= 32``).

``Qwen38MTPForCausalLM`` is the plain adapter (``qwen38_vllm.Qwen38ForCausalLM``) over the MTP chain: the chain opens
with ``mtp=K, sampling=False`` and ``decode_forward`` serves the plugin's ``[1, 1+K]`` verify block with the chain's pass
loop (``Qwen38TracedChain.mtp_enter`` / ``mtp_step`` / ``mtp_leave``, as ``Qwen38ChatSession._generate_mtp`` runs them).
One chain pass verifies the block, decides the acceptance on the device, drafts the next pass and reads one pass row
(the ``K + 1`` argmaxes and the next pass's ``[t', d_1 .. d_K]``).  The adapter keeps the drafts and hands them to
``propose_draft_tokens`` (no device call).  Select the class with ``TT_MODEL_CLASS_OVERRIDES`` (see the README of the
vLLM bundle); without ``speculative_config`` the plain class is the one served.
Batch 1 runs on the plain adapter's paged pools: ``allocate_kv_cache`` opens the chain on vLLM's blocks, ``prefill_forward``
writes the request's block ids (page-table row 0) into the chain's page table after the reset, and every verify writes them
before its device work, with the blocks up to row ``position + K`` of the pass.

The contract's split of work: ``spec_plan`` (host), ``prefill_forward`` (prefill and the first token), ``decode_forward``
(verify, returns ``VerifyOutput(argmax_ids)``), ``propose_draft_tokens`` (the drafts the last verify left),
``release_request``.  ``Qwen38VLLMMTPDriver`` holds the pass-loop state and the resync rule and takes the chain and the
chat session as arguments, so the host tests run it on fakes.

Batch 2..8 (``max_num_seqs`` B, K from ``QWEN38_VLLM_MTP_K``, the lanes rule ``B x (K + 1) <= 32``): ``allocate_kv_cache``
opens the chain with a ``Qwen38LanesSession`` (the text demo's batch-8 form: the lane states in the chain's warm hook, three
lane traces after the chain's) and ``Qwen38VLLMMTPLanesDriver`` runs the plugin's calls over its lane protocol
(``admit``, ``write_counts``, ``set_blocks``, ``set_active``, ``step``, ``override_commit``, the scheduler's boundary
rules without its queue).  A request is a state slot of the plugin; its lane is ``lane_of_slot[slot]`` (the plain batch
adapter's map: the lanes never move on the device, ``slot_remap`` only permutes the map).  One ``decode_forward`` is one
lane pass for every live row; requests join (a prefill into a free lane) and leave (``release_request`` parks the lane)
between passes.
The lanes run on vLLM's blocks like batch 1: ``allocate_kv_cache`` builds the paging from vLLM's cache shape and opens the chain
with it (the lanes share the chain's pools, page id = vLLM block id, page 0 the scratch).  A prefill takes the request's block
ids from its page-table row; every verify sets each live lane's row from the plugin's table (blocks up to row
``position + K``), zeroes the rows of the other lanes and uploads the tables before the pass; ``release_request`` zeroes the
lane's row so its junk writes land in block 0, never in blocks vLLM hands to another request.  The pools are never freed.

Hybrid single (``QWEN38_MTP_HYBRID_SINGLE``, default on, batch mode only): a lone request, prefilled when no lane holds a
request, runs on the chain's own pass loop (the batch-1 path above, same K) in a lane-free state; the lane chain is not
passed.  When a second request arrives (or a step carries another live row) the lone request is settled at the rows its last
pass committed and imported into its lane without a new prefill (``Qwen38VLLMMTPLanesDriver.admit_from_chain``); from then on
everything runs on the lanes.  There is no move back to the chain.

``QWEN38_MAMBA_SPEC=1`` (the plain adapter's MambaSpec contract: block size >= 1024, GDN state saved in and restored from the
plugin's state blocks, ``num_speculative_blocks`` K) adds prefix caching and chunked prefill.  A prefill row is a chunk
``[start_pos, prompt_lens)`` run alone on the chain (``Qwen38VLLMMTPDriver.prefill_range``): it restores the state from the
row's block at a start past 0 (the MTP layer's alignment state is reset with it), saves it at every checkpoint and at a
block-aligned end.  An intermediate chunk (``intermediate_prefill_mask``) takes no lane; the last chunk imports the chain's
state into its lane (or stays on the chain as the hybrid single).  The MTP layer's pools are not state blocks: they hold the
rows of the shared block ids, so a prefix hit reads the earlier request's MTP rows (the row before the hit consumed that
request's next token: draft quality only, the target verifies every token).

No narrow decode (``SpecPlan.supports_narrow_decode`` False): the MTP chain has no sampled tail, so every step is a verify
and the plugin refuses a request with a temperature or a penalty when it arrives.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import torch
from loguru import logger

from models.demos.blackhole.qwen38_flash_next.chat import TOKENIZER_SIZE, VOCAB_SIZE
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_chat_session import (
    CHUNK_PREFILL_MIN_ROWS,
    MTP_BOOTSTRAP_DRAFT_TOKEN,
    MTP_DRAFTS,
    SEED_TOKEN_ID,
    Qwen38ChatChainError,
    Qwen38ChatSession,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lane_scheduler import (
    MAX_LANES,
    MIN_LANES,
    PLACEHOLDER_DRAFT_TOKEN,
    lane_geometry,
)
from models.demos.blackhole.qwen38_flash_next.tools.qwen38_vllm import (
    WARMUP_PROMPT_TOKEN_IDS,
    Qwen38ForCausalLM,
    _ModelCapabilities,
    check_paged_block_size,
    device_greedy_contract,
    mamba_spec_enabled,
    parse_slot_remap,
    prefill_form,
    resident_allocated_context,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.qsa import Qwen38LanePaging, paged_table_width, validate_index_paging

MTP_DRAFTS_VARIABLE = "QWEN38_VLLM_MTP_K"  # the draft count the chain is built for (3, 4 or 5); default 4
MTP_PLE_EARLY_VARIABLE = (
    "QWEN38_VLLM_MTP_PLE_EARLY"  # "1": read the verify lanes on command queue 1 (needs a 2-queue mesh)
)
MTP_HYBRID_SINGLE_VARIABLE = (
    "QWEN38_MTP_HYBRID_SINGLE"  # batch mode: a lone request decodes on the B=1 chain; "0" turns it off
)
MTP_LANE_BUCKETS_VARIABLE = (
    "QWEN38_MTP_LANE_BUCKETS"  # lane widths of an 8-lane session: unset "2,4,8", "off"/"0" one width
)
MTP_CHAIN_VARIABLE = "QWEN38_MTP_CHAIN"  # batch mode: "1" serves the plugin's chain mode (fused_sample verifies submitted over the pending one)
DEFAULT_DRAFTS = 4
PLACEHOLDER_TOKEN_ID = -1  # the plugin's candidate block pad (vllm_tt_plugin.spec_decode.PLACEHOLDER_TOKEN_ID)
PADDING_ROW_TOKEN_ID = (
    0  # a fused readout's padding row: count 1 with a real column-0 id (SPEC_DECODE_CONTRACT: "its column-0 argmax")
)


def mtp_drafts_from_env(environ: Mapping[str, str] | None = None) -> int:
    """``QWEN38_VLLM_MTP_K`` as a draft count in ``MTP_DRAFTS``; default 4; anything else is refused by name."""

    environ = os.environ if environ is None else environ
    text = environ.get(MTP_DRAFTS_VARIABLE, str(DEFAULT_DRAFTS))
    try:
        drafts = int(text)
    except ValueError:
        drafts = -1
    if drafts not in MTP_DRAFTS:
        raise ValueError(f"{MTP_DRAFTS_VARIABLE} must be one of {MTP_DRAFTS}, got {text!r}")
    return drafts


def mtp_ple_early_from_env(environ: Mapping[str, str] | None = None) -> bool:
    """``QWEN38_VLLM_MTP_PLE_EARLY``: "1" reads the verify lanes on the second command queue, "0" (default) does not.  The
    plugin's mesh opens one command queue, so the default is off; "1" needs a mesh opened with two."""

    environ = os.environ if environ is None else environ
    text = environ.get(MTP_PLE_EARLY_VARIABLE, "0")
    if text not in ("0", "1"):
        raise ValueError(f"{MTP_PLE_EARLY_VARIABLE} must be 0 or 1, got {text!r}")
    return text == "1"


def mtp_hybrid_single_from_env(environ: Mapping[str, str] | None = None) -> bool:
    """``QWEN38_MTP_HYBRID_SINGLE``: unset or "1" keeps a lone request of a batch-mode server on the chain's pass loop, "0"
    runs every request on the lanes; anything else is refused by name."""

    environ = os.environ if environ is None else environ
    text = environ.get(MTP_HYBRID_SINGLE_VARIABLE, "1")
    if text not in ("0", "1"):
        raise ValueError(f"{MTP_HYBRID_SINGLE_VARIABLE} must be 0 or 1, got {text!r}")
    return text == "1"


def mtp_chain_from_env(environ: Mapping[str, str] | None = None) -> bool:
    """``QWEN38_MTP_CHAIN``: "1" declares the async and chain-mode capabilities and the ``fused_sample`` accept mode (batch
    mode), unset or "0" (default) leaves the adapter as it is; anything else is refused by name."""

    environ = os.environ if environ is None else environ
    text = environ.get(MTP_CHAIN_VARIABLE, "0")
    if text not in ("0", "1"):
        raise ValueError(f"{MTP_CHAIN_VARIABLE} must be 0 or 1, got {text!r}")
    return text == "1"


class _MTPModelCapabilities:
    """The plain adapter's capabilities with the speculative keys; computed on first access like the plain ones."""

    def __get__(self, instance, owner) -> dict:
        capabilities = dict(_ModelCapabilities().__get__(instance, owner))
        mamba_on = bool(owner._mamba_spec_capable) and mamba_spec_enabled()
        chain_on = mtp_chain_from_env()
        capabilities.update(
            {
                "supports_spec_decode": True,
                # the drafter proposes on the device and reads no hidden state the runner feeds it
                "spec_requirements": ["device_propose"],
                "spec_hidden_handoff": [],
                "output_tokens_per_step": 1,
                # every verify is synchronous (the platform turns async scheduling off with a warning), unless
                # QWEN38_MTP_CHAIN=1: the lane verifies then run as the plugin's chain mode
                "supports_async_decode": chain_on,
                "supports_async_spec_decode": chain_on,
                # the chain has no sampled tail: prefill returns a logits row and the decode steps are verifies
                "supports_sample_on_device": False,
                "supports_device_penalties": False,
                # its prefill takes position 0 only, unless the GDN state travels in MambaSpec blocks
                "supports_chunked_prefill": mamba_on,
            }
        )
        if mamba_on:
            capabilities["accepts_intermediate_prefill_mask"] = True
        return capabilities


class _Pass:
    """One chain pass as the driver keeps it until the next pass or the leave commits it."""

    __slots__ = ("position", "tokens", "accepted", "argmaxes", "next_tokens")

    def __init__(self, record: Any, next_tokens: Sequence[int]) -> None:
        self.position = int(record.position)  # the device position the pass started at
        self.tokens = tuple(int(token) for token in record.tokens)  # the rows it fed: [t_P, d_1 .. d_K]
        self.accepted = int(record.accepted)  # drafts the device took; the pass commits accepted + 1 rows
        self.argmaxes = tuple(int(token) for token in record.argmaxes)  # K + 1 ids
        self.next_tokens = tuple(int(token) for token in next_tokens)  # the next pass's [t', d_1 .. d_K]


class Qwen38VLLMMTPDriver:
    """The pass-loop state of one request on a chain and a chat session (batch 1).  No ttnn, no vLLM: the chain's
    ``mtp_enter`` / ``mtp_step`` / ``mtp_leave`` / ``mtp_pass_fits`` / ``read_token_row`` / ``loop_guard`` and the session's
    ``reset`` / ``_prefill_chunked`` / ``_prefill`` / ``committed`` / ``ple_context`` are all it calls."""

    def __init__(self, chain: Any, session: Qwen38ChatSession, drafts: int) -> None:
        self.chain = chain
        self.session = session
        self.drafts = drafts
        self.run: _Pass | None = None  # the last pass while the chain is in verify mode; None: the 1-row state
        self.passes = 0
        self.resyncs = 0
        self.accepted_drafts = 0

    # -- prefill ---------------------------------------------------------------------------------------------------

    def prefill(self, ids: Sequence[int], page_ids: Sequence[int] | None = None) -> int:
        """The prompt into the reset state as ``Qwen38ChatSession.complete`` prefills a chunked prompt; returns the model's
        first token (the token row after the last prompt token's TAIL).  Leaves an open pass loop first.  ``page_ids`` (a
        paged chain): the request's vLLM block ids, written into the chain's page table after the reset, before any chunk.
        """

        self.close()
        session = self.session
        session.reset()
        if page_ids is not None:
            self.chain.write_page_table(page_ids)
        suffix = [int(token) for token in ids]
        if session.chunk_prefill_rows(len(suffix) - 1) >= CHUNK_PREFILL_MIN_ROWS:
            session.resolve_prefill_mode("chunked")  # refused by name when the chain has no chunk trace
            session._prefill_chunked(suffix[:-1], suffix[-1], None)
            suffix = suffix[-1:]
        with self.chain.loop_guard():
            session._prefill(suffix, None, None)
        token = int(self.chain.read_token_row())
        if not 0 <= token < VOCAB_SIZE:
            raise Qwen38ChatChainError(f"token row holds {token}, outside the vocabulary [0, {VOCAB_SIZE})")
        return token

    def prefill_range(
        self,
        ids: Sequence[int],
        *,
        start: int,
        page_ids: Sequence[int] | None,
        layer_blocks: dict[int, tuple[Any, int]] | None,
        checkpoints: Sequence[tuple[int, dict[int, tuple[Any, int]]]],
        block_size: int,
        final: bool,
    ) -> int | None:
        """``ids[start:]`` (``ids`` end at the chunk end) as ``prefill`` prefills a prompt, on the state of ``start``: the reset
        at 0, else the restore from ``layer_blocks`` (a block boundary).  ``checkpoints`` (``start < n < len(ids)``,
        ascending): the prefill stops at each ``n`` and saves the state into that entry's blocks.  A chunk end on a block
        boundary saves into ``layer_blocks``.  ``final``: returns the model's first token; else the chunk must end on a
        block boundary and nothing is read (None)."""

        ids = [int(token) for token in ids]
        end = len(ids)
        if start < 0 or start >= end or (start and (start % block_size or layer_blocks is None)):
            raise Qwen38ChatChainError(
                f"prefill range [{start}, {end}) must start at 0 or at a block boundary with the state blocks "
                f"(block size {block_size})"
            )
        save_end = layer_blocks is not None and end % block_size == 0
        if not final and not save_end:
            raise Qwen38ChatChainError(
                f"an intermediate chunk must end on a block boundary with the state blocks, ends at {end}"
            )
        self.close()
        session, chain = self.session, self.chain
        if start:
            session.restore_prefix(start, ids, layer_blocks, block_size)
        else:
            session.reset()
        if page_ids is not None:
            chain.write_page_table(page_ids)
        stops = dict(checkpoints)
        at = start
        for target in [*stops, end]:
            head_end = target - 1
            # the MTP row of the part's last token consumes ids[target]; at the chunk end that token is not in ``ids``
            # (the runner's prefill tokens end there), so the step takes the model's own argmax
            following = ids[target] if target < end else None
            if session.chunk_prefill_rows(head_end - at) >= CHUNK_PREFILL_MIN_ROWS:
                session.resolve_prefill_mode("chunked")  # refused by name when the chain has no chunk trace
                session._prefill_chunked(ids[at:head_end], ids[head_end], None)
                tail = ids[head_end:target]
            else:
                tail = ids[at:target]
            with chain.loop_guard():
                session._prefill(tail, None, None, last_next=following)
            if target < end:
                chain.save_prefix_state(target, stops[target], block_size)
            elif save_end:
                chain.save_prefix_state(end, layer_blocks, block_size)
            at = target
        if not final:
            return None
        token = int(chain.read_token_row())
        if not 0 <= token < VOCAB_SIZE:
            raise Qwen38ChatChainError(f"token row holds {token}, outside the vocabulary [0, {VOCAB_SIZE})")
        return token

    # -- verify ----------------------------------------------------------------------------------------------------

    def verify(
        self,
        tokens: Sequence[int],
        positions: Sequence[int],
        num_valid: int,
        accepted: int,
        page_ids: Sequence[int] | None = None,
    ) -> list[int]:
        """One verify block of the plugin: ``tokens`` ``[t, d_1 .. d_K]`` at ``positions``, ``num_valid`` real drafts,
        ``accepted`` the tokens the plugin's previous step committed for the request (1 after a prefill).  Returns the
        pass's ``K + 1`` argmaxes (``argmax_ids``).  ``page_ids`` (a paged chain): the request's vLLM block ids, written
        into the chain's page table (a no-op when unchanged) before the pass's device work, the resync's leave included.
        """

        k = self.drafts
        tokens = [int(token) for token in tokens]
        if len(tokens) != k + 1 or len(positions) != k + 1:
            raise ValueError(
                f"the verify block must be {k + 1} wide, got {len(tokens)} tokens, {len(positions)} positions"
            )
        if page_ids is not None:
            self.chain.write_page_table(page_ids)
        first, position = tokens[0], int(positions[0])
        run = self.run
        if run is not None:
            device_rows = run.accepted + 1
            if accepted > device_rows:
                raise Qwen38ChatChainError(
                    f"the plugin committed {accepted} tokens of the last pass, the device accepted {device_rows - 1} drafts"
                )
            follows = (
                accepted == device_rows
                and num_valid == k
                and position == run.position + device_rows
                and tuple(tokens) == run.next_tokens
            )
            if follows:
                self._require_room(position)
                self.session.committed.extend(
                    run.tokens[:device_rows]
                )  # the device commits them at the start of this pass
                record = self.chain.mtp_step()
                return self._record(record)
            self.resyncs += 1
            logger.info(
                f"mtp resync: plugin committed {accepted} of {device_rows} rows, block {tokens} (valid drafts {num_valid}) "
                f"vs the device's {run.next_tokens} at {run.position + device_rows}, position {position}"
            )
            self._settle(accepted)
        if accepted != 1 and run is None:
            raise Qwen38ChatChainError(
                f"a verify with no pass loop open must follow a prefill (accepted_counts 1), got {accepted}"
            )
        if position != len(self.session.committed):
            raise Qwen38ChatChainError(
                f"the plugin's position {position} differs from the committed sequence's {len(self.session.committed)}"
            )
        self._require_room(position)
        if num_valid:
            # Real drafts the device did not draft itself (the scheduler trimmed a proposal): the entry pass feeds exactly the
            # plugin's rows, so its argmaxes answer the plugin's walk; the columns past num_valid are the placeholder id.
            bootstrap = [first, *tokens[1 : 1 + num_valid]] + [MTP_BOOTSTRAP_DRAFT_TOKEN] * (k - num_valid)
            record = self.chain.mtp_enter(first, self.session.ple_context, bootstrap_tokens=bootstrap)
        else:
            record = self.chain.mtp_enter(first, self.session.ple_context)
        return self._record(record)

    def _require_room(self, position: int) -> None:
        if not self.chain.mtp_pass_fits(position):
            raise Qwen38ChatChainError(f"position {position} has no room for a verify pass in the resident context")

    def _record(self, record: Any) -> list[int]:
        next_tokens = self.chain.mtp.chain.next_tokens
        run = _Pass(record, next_tokens)
        if len(run.argmaxes) != self.drafts + 1 or len(run.next_tokens) != self.drafts + 1:
            raise Qwen38ChatChainError(
                f"pass row widths {len(run.argmaxes)}, {len(run.next_tokens)}, expected {self.drafts + 1}"
            )
        for token in run.argmaxes[: run.accepted + 1]:
            if not 0 <= token < TOKENIZER_SIZE:
                raise Qwen38ChatChainError(
                    f"the pass committed token {token}, outside the tokenizer [0, {TOKENIZER_SIZE})"
                )
        self.run = run
        self.passes += 1
        self.accepted_drafts += run.accepted
        return list(run.argmaxes)

    # -- propose ---------------------------------------------------------------------------------------------------

    def propose(self, committed: Sequence[int], accepted: int, run: _Pass | None = None) -> tuple[list[int], int]:
        """The drafts the last pass left, with their count: ``(d_1 .. d_K, K)`` when the plugin committed what the device
        accepted, else ``([0] * K, 0)`` (a decline: no pass is open, or the plugin cut the pass).  ``run``: the pass of a
        step already followed by another one (a chained step's handle keeps it); default the last pass."""

        k = self.drafts
        run = self.run if run is None else run
        if run is None or accepted != run.accepted + 1:
            return [0] * k, 0
        if tuple(int(token) for token in committed[:accepted]) != run.argmaxes[:accepted]:
            return [0] * k, 0
        return list(run.next_tokens[1:]), k

    # -- leave -----------------------------------------------------------------------------------------------------

    def _settle(self, rows: int) -> None:
        """Commit ``rows`` rows of the last pass and rebuild the 1-row buffers (``Qwen38ChatSession._mtp_settle``)."""

        run = self.run
        self.session.ple_context = self.chain.mtp_leave(position=run.position + rows, committed_rows=rows)
        self.session.committed.extend(run.tokens[:rows])
        self.run = None

    def close(self) -> None:
        """Leave verify mode with every row of the last pass committed; the next prefill resets the state."""

        if self.run is not None:
            self._settle(self.run.accepted + 1)


class _LaneRun:
    """One request's lane as the driver keeps it.  ``record`` None: the lane was admitted and no pass ran (``pending`` is the
    model's first token, the bootstrap block is queued); else the last pass of the lane."""

    __slots__ = ("position", "pending", "settled", "record", "tokens", "accepted", "argmaxes", "next_tokens")

    def __init__(self, position: int, pending: int | None = None, settled: int = 1) -> None:
        self.position = int(position)  # fresh: the prompt length; after a pass: the device position the pass started at
        self.pending = pending
        self.settled = int(
            settled
        )  # fresh: the tokens the plugin's last step commits (1 after a prefill, more after a chain move)
        self.record = None
        self.tokens: tuple[int, ...] = ()
        self.accepted = 0
        self.argmaxes: tuple[int, ...] = ()
        self.next_tokens: tuple[int, ...] = ()

    @property
    def fresh(self) -> bool:
        return self.record is None

    @property
    def device_rows(self) -> int:
        return self.accepted + 1

    @property
    def next_position(self) -> int:
        return self.position if self.fresh else self.position + self.device_rows


_LIVE = object()  # ``Qwen38VLLMMTPLanesDriver.propose``: read the lane's live run


class _LanePending:
    """A lane pass enqueued and not yet landed: the device session's pending pass and the lanes of the live rows."""

    __slots__ = ("device", "live")

    def __init__(self, device: Any, live: frozenset[int]) -> None:
        self.device = device
        self.live = live


class VerifyRow:
    """One live row of a verify call: the lane it runs in, the plugin's block, its position, its valid drafts and the tokens
    the plugin committed for the request in its previous step (1 after a prefill), and (paged lanes) the request's vLLM block
    ids for the pass."""

    __slots__ = ("lane", "tokens", "position", "num_valid", "accepted", "pages")

    def __init__(
        self,
        lane: int,
        tokens: Sequence[int],
        position: int,
        num_valid: int,
        accepted: int,
        pages: Sequence[int] | None = None,
    ) -> None:
        self.lane = int(lane)
        self.tokens = [int(token) for token in tokens]
        self.position = int(position)
        self.num_valid = int(num_valid)
        self.accepted = int(accepted)
        self.pages = None if pages is None else [int(block) for block in pages]


class SimpleTicket:
    """What ``Qwen38LanesSession.admit`` reads of a ticket: the prompt, no images."""

    def __init__(self, ids: Sequence[int]) -> None:
        self.prompt_ids = [int(token) for token in ids]
        self.images = None
        self.vision_positions = None


class Qwen38VLLMMTPLanesDriver:
    """The lane chain's pass loop under the plugin's verify calls (batch 2..8).  No ttnn, no vLLM: ``device`` is a
    ``Qwen38LanesSession`` (``admit``, ``write_counts``, ``set_blocks``, ``set_active``, ``step``, ``override_commit``,
    ``room``), so the host tests run it on a fake.

    The host writes of a pass boundary are the scheduler's (``qwen38_lane_scheduler``): the accept counts the coming commit
    takes (the device lands them itself; the host rewrites the vector only with a write: -1 on an admitted lane, fewer rows
    on a lane the plugin cut), the next token blocks (only for an admitted lane or a block that differs from the device's) and
    the active mask (the live rows).  The decision of acceptance stays on the device (greedy, fused); the plugin walks the same
    argmaxes, so both agree, and a disagreement (the plugin committed less, or trimmed drafts) is a per-lane resync."""

    def __init__(self, device: Any, lanes: int, drafts: int, paged: bool = False, hold_orphans: bool = False) -> None:
        lane_geometry(lanes, drafts)
        self.device = device
        self.paged = paged  # the lanes run on the chain's shared paged pools: the device owns lane page tables
        # Async scheduling builds a step before the previous one is processed, so a lane's request can be missing from
        # a step (its prefill's first token not yet applied, or it is finishing): such a lane is held, not an error.
        self.hold_orphans = hold_orphans
        self.lanes = lanes
        self.drafts = drafts
        self.runs: list[_LaneRun | None] = [None] * lanes
        self.counts = [-1] * lanes
        self.blocks: list[list[int] | None] = [None] * lanes
        self.writes = False
        self.passes = 0
        self.lane_passes = 0
        self.resyncs = 0
        self.accepted_drafts = 0

    @property
    def holders(self) -> list[int]:
        return [lane for lane, run in enumerate(self.runs) if run is not None]

    def _placeholders(self, *head: int) -> list[int]:
        return [*head, *([PLACEHOLDER_DRAFT_TOKEN] * (self.drafts + 1 - len(head)))]

    # -- admission and release -------------------------------------------------------------------------------------

    def admit(self, lane: int, ids: Sequence[int], block_ids: Sequence[int] | None = None) -> int:
        """The prompt prefilled into ``lane`` (the lanes session's admission); returns the model's first token.  The lane's
        first pass is the bootstrap pass on ``[pending, placeholders]`` committing nothing (count -1).  ``block_ids`` (paged
        lanes): the request's vLLM blocks, the prefill's and the lane's."""

        if self.runs[lane] is not None:
            raise Qwen38ChatChainError(f"admission into lane {lane}, which still holds a request")
        if self.paged:
            if block_ids is None:
                raise Qwen38ChatChainError(f"admission into lane {lane} on paged lanes needs the request's block ids")
            admitted = self.device.admit(lane, SimpleTicket(ids), block_ids=list(block_ids))
        else:
            admitted = self.device.admit(lane, SimpleTicket(ids))
        pending, position = int(admitted.pending), int(admitted.position)
        if position != len(ids):
            raise Qwen38ChatChainError(f"lane {lane}: admitted position {position} for a prompt of {len(ids)} tokens")
        if not 0 <= pending < VOCAB_SIZE:
            raise Qwen38ChatChainError(f"token row holds {pending}, outside the vocabulary [0, {VOCAB_SIZE})")
        self.runs[lane] = _LaneRun(position, pending)
        self.counts[lane] = -1
        self.blocks[lane] = self._placeholders(pending)
        self.writes = True
        return pending

    def admit_from_chain(
        self,
        lane: int,
        position: int,
        pending: int,
        settled: int,
        ple_context: Any,
        block_ids: Sequence[int] | None = None,
    ) -> None:
        """``lane`` takes the request the chain holds in its settled 1-row state (``admit`` without the reset and the prefill):
        ``position`` rows committed, ``pending`` the model's next token, ``settled`` the rows the chain's last pass committed
        (the plugin's next ``accepted_counts``).  The lane's first pass is the bootstrap pass on the plugin's block.  Paged
        lanes: the KV is already in the shared pools, the import is state only and the lane's row takes ``block_ids``, the
        blocks the chain's table holds."""

        if self.runs[lane] is not None:
            raise Qwen38ChatChainError(f"chain move into lane {lane}, which still holds a request")
        if not 0 <= pending < VOCAB_SIZE:
            raise Qwen38ChatChainError(f"token row holds {pending}, outside the vocabulary [0, {VOCAB_SIZE})")
        if self.paged and block_ids is None:
            raise Qwen38ChatChainError(f"chain move into lane {lane} on paged lanes needs the chain's block ids")
        self._import_chain_state(lane, position, ple_context)
        if self.paged:
            self.device.set_lane_pages(lane, list(block_ids))
            self.device.sync_page_tables()
        self.runs[lane] = _LaneRun(position, pending, settled)
        self.counts[lane] = -1
        self.blocks[lane] = self._placeholders(pending)
        self.writes = True

    def _import_chain_state(self, lane: int, position: int, ple_context: Any) -> None:
        """The lanes session's eviction and import (``Qwen38LanesSession.admit`` after its prefill) on the state the chain
        holds now."""

        import ttnn
        from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
        from models.demos.blackhole.qwen38_flash_next.ttnn import mtp_lanes

        device = self.device
        device.backbone_pager.evict(0, device.backbone_pool.slots[0])
        device.alignment_pager.evict(0, device.alignment_pool.slots[0])
        gdn_layers = sum(1 for layer in device.model.layers if isinstance(layer.attention, gdn_module.Qwen38TTNNGDN))
        mtp_lanes.import_lane_state(
            device.model,
            device.verify,
            lane,
            position=position,
            ple_context=ple_context,
            gdn_phases=[position % gdn_module.CONV_KERNEL_SIZE] * gdn_layers,
            backbone_slot=device.backbone_pool.slots[0],
            backbone_pager=device.backbone_pager,
            alignment_slot=device.alignment_pool.slots[0],
            alignment_pager=device.alignment_pager,
            rope_shift=0,
        )
        ttnn.synchronize_device(device.mesh)

    def release(self, lane: int) -> None:
        """The request left (finished, preempted): the lane parks at the next pass (it still commits its last real pass).
        Paged lanes: the lane's page row is zeroed and uploaded now (between passes), so the parked lane's junk writes land
        in block 0 and never in blocks vLLM reassigns."""

        self.runs[lane] = None
        self.counts[lane] = -1
        self.blocks[lane] = None
        if self.paged:
            self.device.clear_lane_pages(lane)
            self.device.sync_page_tables()

    # -- verify ----------------------------------------------------------------------------------------------------

    def verify(self, rows: Sequence[VerifyRow]) -> dict[int, list[int]]:
        """One lane pass for the live rows of the plugin's verify call; returns lane -> its ``K + 1`` argmaxes."""

        return self.verify_finish(self.verify_enqueue(rows, defer_read=False))

    def verify_enqueue(
        self, rows: Sequence[VerifyRow], chain_rows: Collection[int] | None = None, *, defer_read: bool = True
    ) -> _LanePending:
        """The launch half of :meth:`verify`: the checks, the resync writes and the lane pass enqueued (the device session's
        ``step_enqueue``); the pass lands in :meth:`verify_finish`.  ``chain_rows``: the lanes of the rows the plugin chained
        onto the pass in flight (placeholder tokens, accepted -1, a provisional position that is an upper bound of the real
        one): they take the lane's own next block, position and accepted count (the last :meth:`verify_finish` must have
        run).  ``defer_read`` False: the pass row is read blocking in :meth:`verify_finish`."""

        k = self.drafts
        live = {row.lane: row for row in rows}
        if not live or len(live) != len(rows):
            raise Qwen38ChatChainError(f"a verify needs distinct live rows, got lanes {[row.lane for row in rows]}")
        orphans = sorted(set(self.holders) - set(live))
        if orphans and not self.hold_orphans:
            raise Qwen38ChatChainError(
                f"contract error: lanes {orphans} hold requests but no live row of this verify maps to them "
                f"(live lanes {sorted(live)}); a lane pass advances every active lane"
            )
        strangers = sorted(lane for lane in live if self.runs[lane] is None)
        if strangers:
            raise Qwen38ChatChainError(f"contract error: live rows map to lanes {strangers} with no prefilled request")
        chained = sorted(set(chain_rows or ()))
        outside = [lane for lane in chained if lane not in live]
        if outside:
            raise Qwen38ChatChainError(
                f"contract error: chain rows {outside} are not live rows (live lanes {sorted(live)})"
            )
        for lane in chained:
            live[lane] = self.chain_row(lane, live[lane])
        for lane, row in sorted(live.items()):
            self._plan(lane, row)
        if self.paged:
            self._sync_pages(live)
        if self.writes:
            self.device.write_counts(list(self.counts))
            self.device.set_blocks(self.blocks)
            self.blocks = [None] * self.lanes
            self.writes = False
        set_held = getattr(self.device, "set_held", None)
        if set_held is not None:  # the width pick covers every lane that holds a request
            set_held([1 if run is not None else 0 for run in self.runs])
        self.device.set_active([1 if lane in live else 0 for lane in range(self.lanes)])
        return _LanePending(self.device.step_enqueue(defer_read=defer_read), frozenset(live))

    def verify_finish(self, pending: _LanePending) -> dict[int, list[int]]:
        """The landing half of :meth:`verify`: the pass's readback, the lanes' runs and counts; returns lane -> its ``K + 1``
        argmaxes."""

        k, live = self.drafts, pending.live
        record = self.device.finish(pending.device)
        self.passes += 1
        out: dict[int, list[int]] = {}
        for lane in range(self.lanes):
            if lane not in live:
                self.counts[lane] = -1
                continue
            run = _LaneRun(record.positions[lane])
            run.record = record
            run.tokens = tuple(int(token) for token in record.tokens[lane])
            run.accepted = int(record.accepted[lane])
            run.argmaxes = tuple(int(token) for token in record.argmaxes[lane])
            run.next_tokens = tuple(int(token) for token in record.next_tokens[lane])
            if len(run.argmaxes) != k + 1 or len(run.next_tokens) != k + 1 or not 0 <= run.accepted <= k:
                raise Qwen38ChatChainError(
                    f"lane {lane}: pass row widths {len(run.argmaxes)}, {len(run.next_tokens)}, accepted {run.accepted}, "
                    f"expected {k + 1} wide"
                )
            for token in run.argmaxes[: run.accepted + 1]:
                if not 0 <= token < TOKENIZER_SIZE:
                    raise Qwen38ChatChainError(
                        f"lane {lane}: the pass committed token {token}, outside the tokenizer [0, {TOKENIZER_SIZE})"
                    )
            self.runs[lane] = run
            self.counts[lane] = run.accepted
            self.lane_passes += 1
            self.accepted_drafts += run.accepted
            out[lane] = list(run.argmaxes)
        return out

    def chain_position(self, lane: int, provisional: int) -> int:
        """The real column-0 position of a chained row on ``lane`` (where its last pass leaves it); the plugin's
        ``provisional`` one is an upper bound of it, a lower one is refused."""

        run = self.runs[lane]
        if run is None or run.fresh:
            raise Qwen38ChatChainError(f"chain row on lane {lane}, which has no pass to chain onto")
        if provisional < run.next_position:
            raise ValueError(
                f"chain row on lane {lane}: provisional position {provisional} is below the real one {run.next_position}"
            )
        return run.next_position

    def chain_row(self, lane: int, row: VerifyRow) -> VerifyRow:
        """A chained row as the lane's own state makes it: the placeholders are refused, the block, the position and the
        accepted count come from the lane's last pass (the plugin took what the device took), the valid drafts are the
        smaller of the plugin's bound and K."""

        if any(token != PLACEHOLDER_TOKEN_ID for token in row.tokens):
            raise ValueError(f"chain row on lane {lane} must carry placeholder tokens, got {row.tokens}")
        if row.accepted != -1:
            raise ValueError(f"chain row on lane {lane} must carry accepted_counts -1, got {row.accepted}")
        position = self.chain_position(lane, row.position)
        run = self.runs[lane]
        return VerifyRow(lane, run.next_tokens, position, min(row.num_valid, self.drafts), run.device_rows, row.pages)

    def _sync_pages(self, live: Mapping[int, VerifyRow]) -> None:
        """Paged lanes: each live lane's page row from its verify row, the other lanes' rows zero, uploaded before the pass
        (a no-op for what did not change)."""

        missing = sorted(lane for lane, row in live.items() if row.pages is None)
        if missing:
            raise Qwen38ChatChainError(f"paged lanes: the live rows of lanes {missing} carry no block ids")
        for lane in range(self.lanes):
            if lane in live:
                self.device.set_lane_pages(lane, live[lane].pages)
            else:
                self.device.clear_lane_pages(lane)
        self.device.sync_page_tables()

    def _plan(self, lane: int, row: VerifyRow) -> None:
        """Check the row against the lane's state; queue the resync writes when the plugin's block is not the device's."""

        k = self.drafts
        if len(row.tokens) != k + 1:
            raise ValueError(f"the verify block must be {k + 1} wide, got {len(row.tokens)} tokens")
        run = self.runs[lane]
        if not 0 <= row.num_valid <= k:
            raise ValueError(f"num_valid_drafts must be in [0, {k}], got {row.num_valid}")
        if not self.device.room(row.position):
            raise Qwen38ChatChainError(f"lane {lane}: position {row.position} has no room for a verify pass")
        block = self._placeholders(row.tokens[0], *row.tokens[1 : 1 + row.num_valid])
        if run.fresh:
            if row.accepted != run.settled or row.position != run.position or row.tokens[0] != run.pending:
                raise Qwen38ChatChainError(
                    f"lane {lane}: the first verify after a prefill must carry the first token {run.pending} at position "
                    f"{run.position} with accepted_counts {run.settled}, got token {row.tokens[0]}, position {row.position}, "
                    f"accepted_counts {row.accepted}"
                )
            if row.num_valid:  # drafts the device did not make: the entry pass feeds exactly the plugin's rows
                self.blocks[lane] = block
                self.writes = True
            return
        device_rows = run.device_rows
        if row.accepted > device_rows:
            raise Qwen38ChatChainError(
                f"lane {lane}: the plugin committed {row.accepted} tokens of the last pass, the device accepted "
                f"{run.accepted} drafts"
            )
        if (
            row.accepted == device_rows
            and row.num_valid == k
            and row.position == run.next_position
            and tuple(row.tokens) == run.next_tokens
        ):
            return  # the plugin took what the device took: the device's next block is the plugin's
        self.resyncs += 1
        rows = row.accepted
        logger.info(
            f"mtp lanes resync: lane {lane} plugin committed {rows} of {device_rows} rows, block {row.tokens} (valid "
            f"drafts {row.num_valid}) vs the device's {run.next_tokens} at {run.next_position}, position {row.position}"
        )
        if row.position != run.position + rows:
            raise Qwen38ChatChainError(
                f"lane {lane}: the plugin's position {row.position} differs from the pass start {run.position} plus the "
                f"{rows} committed rows"
            )
        if row.tokens[0] != run.argmaxes[rows - 1]:
            raise Qwen38ChatChainError(
                f"lane {lane}: the block's first token {row.tokens[0]} is not the pass's argmax {run.argmaxes[rows - 1]} "
                f"of the last committed row"
            )
        if rows < device_rows:
            self.device.override_commit(lane, run.record, rows)
            self.counts[lane] = rows - 1
        self.blocks[lane] = block
        self.writes = True

    # -- propose ---------------------------------------------------------------------------------------------------

    def propose(self, lane: int, committed: Sequence[int], accepted: int, run: Any = _LIVE) -> tuple[list[int], int]:
        """The drafts the lane's last pass left, with their count; ``([0] * K, 0)`` (decline) with no pass open or when the
        plugin committed something else than the device accepted.  ``run``: the lane's pass of a step already followed by
        another one (a chained step's handle keeps it; None: the lane held no row); default the lane's live run."""

        k = self.drafts
        run = self.runs[lane] if run is _LIVE else run
        if run is None or run.fresh or accepted != run.accepted + 1:
            return [0] * k, 0
        if tuple(int(token) for token in committed[:accepted]) != run.argmaxes[:accepted]:
            return [0] * k, 0
        return list(run.next_tokens[1:]), k


class Qwen38MTPFusedHandle:
    """The opaque handle of a deferred ``fused_sample`` verify (the plugin's chain mode): ``tip_id`` names the step for the
    next step's ``chain_tip``; :meth:`wait` lands the step (the reader thread's ``read_decode_output``, or the engine thread's
    ``_finish_pending``) and returns its ``FusedReadout``.  The landing runs once under the handle's lock, whichever thread
    gets there first; the others wait for it and read the same readout (or the same error).  The snapshots (``runs``,
    ``lane_of_slot``, ``hybrid_slot``, ``single_run``) are what ``propose_draft_tokens`` reads when a later step has already
    changed the live state."""

    def __init__(self, tip_id: int, finish: Callable[[], Any]) -> None:
        self.tip_id = int(tip_id)
        self._finish = finish
        self._lock = threading.Lock()
        self._readout: Any = None
        self._error: BaseException | None = None
        self.runs: dict[int, Any] = {}  # lane -> its _LaneRun of this step (set by the landing)
        self.lane_of_slot: list[int] | None = None  # the slot -> lane map after this step's remap
        self.hybrid_slot: int | None = None  # the chain's request slot when this step ran on the pass loop
        self.single_run: Any = None  # that step's _Pass

    @property
    def finished(self) -> bool:
        return self._readout is not None or self._error is not None

    def wait(self, timeout: float | None = None):
        """Land the step if no thread has, then return its ``FusedReadout`` (the landing's error is raised again)."""

        del timeout  # the device event wait has no deadline; the plugin's own watchdog covers a hang
        with self._lock:
            if not self.finished:
                try:
                    self._readout = self._finish()
                except BaseException as error:
                    self._error = error
            if self._error is not None:
                raise self._error
            return self._readout


class Qwen38MTPForCausalLM(Qwen38ForCausalLM):
    """The plain adapter over the MTP chain, serving the plugin's speculative contract (``custom_class``): batch 1 over the
    chain's pass loop, batch 2..8 over the lane chain (see the module docstring)."""

    model_capabilities = _MTPModelCapabilities()
    decode_input_update_contract = 0  # decode_forward is overridden and runs no reload command
    _paged_single_slot = True  # batch 1: the chain opens in allocate_kv_cache on vLLM's pools (lanes mode: flat caches)
    _mamba_spec_capable = True  # QWEN38_MAMBA_SPEC=1 turns it on
    _open_lanes = 1  # the lane count the chain opens for (set by initialize_vllm_model before the open)
    _last_handle: Qwen38MTPFusedHandle | None = None  # the newest deferred fused step (chain mode)
    _tip_id: int | None = (
        None  # its tip: what the next chained step's chain_tip must name (None: nothing to chain onto)
    )
    _tip_count = 0  # the tips handed out

    def __init__(self, chain: Any, *, allocated_context: int, **kwargs: Any) -> None:
        lanes = kwargs.get("batch_lanes", 1)
        self.lanes_mode = lanes != 1
        if self.lanes_mode:
            if kwargs.get("chain_opener") is None:
                raise ValueError("the MTP adapter in batch mode needs the chain opener initialize_vllm_model hands in")
            self.drafts = mtp_drafts_from_env()
            lane_geometry(lanes, self.drafts)
            super().__init__(chain, allocated_context=allocated_context, **kwargs)
            self.lanes_session = None  # allocate_kv_cache opens the chain and the lanes
            self.lane_driver: Qwen38VLLMMTPLanesDriver | None = None
            # hybrid single: the pass loop (``driver``) of a lone request lives on the chain; ``mtp_hybrid_slot`` is its state
            # slot (None: no such request), ``mtp_hybrid_pending`` the model's first token while no pass has run
            self.mtp_hybrid_single = mtp_hybrid_single_from_env()
            self.mtp_hybrid_slot: int | None = None
            self.mtp_hybrid_pending = 0
            self.mtp_hybrid_pages: list[
                int
            ] | None = None  # paged: the block ids the chain's table holds for that request
            self.driver: Qwen38VLLMMTPDriver | None = None
            return
        super().__init__(chain, allocated_context=allocated_context, **kwargs)
        self.drafts = mtp_drafts_from_env()
        self.session: Qwen38ChatSession | None = None
        self.driver: Qwen38VLLMMTPDriver | None = None
        if chain is not None:  # a chain opened by the caller (flat); else allocate_kv_cache opens it, paged
            self._attach_chain()

    @classmethod
    def _mamba_spec_speculative_blocks(cls) -> int:
        return mtp_drafts_from_env()

    def _attach_chain(self) -> None:
        """The pass-loop driver and the chat session over ``self.chain``."""

        mtp = getattr(self.chain, "mtp", None)
        if mtp is None:
            raise ValueError("the MTP adapter needs a chain opened with mtp=K")
        self.drafts = int(mtp.drafts)
        self.session = Qwen38ChatSession(self.chain, None, allocated_context=self.allocated_context)
        self.driver = Qwen38VLLMMTPDriver(self.chain, self.session, self.drafts)

    def _open_paged_single(self, kv_cache_shape: Any) -> None:
        super()._open_paged_single(kv_cache_shape)  # the opener applies ``_chain_open_options`` (the MTP chain)
        self._attach_chain()

    def _pass_blocks(self, position: int) -> int:
        """The blocks a verify pass at ``position`` needs.  The pass writes rows ``position .. position + K`` (the entry pass
        and every following pass feed K + 1 rows), so the last row's block ``(position + K) // block_size`` must be in the
        row; rows past the resident context are not written."""

        return min(position + self.drafts, self.allocated_context - 1) // self.block_size + 1

    def _verify_pages(self, page_table: Any, position: int) -> list[int] | None:
        """The request's block ids for a verify pass at ``position`` (None: flat chain): row 0 of the table, the blocks of
        ``_pass_blocks``."""

        if self.paging is None:
            return None
        return self._slot_page_ids(page_table, needed=self._pass_blocks(position), label="verify")

    def _row_page_ids(self, page_table: Any, row: int, *, needed: int, label: str) -> list[int]:
        """The vLLM block ids of row ``row`` of the plugin's page table (``_page_row``'s checks), for the paged lanes."""

        if page_table is None:
            raise ValueError(f"{label}: the paged lanes need the plugin's page_table")
        table = torch.as_tensor(page_table)
        if table.ndim != 2 or not 0 <= row < table.shape[0]:
            raise ValueError(f"{label}: page_table must be [rows, blocks] with row {row}, got {tuple(table.shape)}")
        return self._page_row(table[row], needed=needed, label=label)

    def _warm_page_table(self, rows: int, used: int, tokens: int) -> torch.Tensor | None:
        """The warm-up's page table (None: flat chain): ``used`` rows each hold the blocks of ``tokens`` rows, distinct per
        row and counted from block 1 (block 0 is the scratch); the other rows are zero."""

        if self.paging is None:
            return None
        blocks = (tokens - 1) // self.block_size + 1
        if 1 + used * blocks > self.num_blocks:
            raise ValueError(f"the warm-up needs {1 + used * blocks} blocks, the pool has {self.num_blocks}")
        table = torch.zeros(rows, self.paging.max_blocks, dtype=torch.int32)
        for row in range(used):
            table[row, :blocks] = torch.arange(1 + row * blocks, 1 + (row + 1) * blocks, dtype=torch.int32)
        return table

    # -- construction ----------------------------------------------------------------------------------------------

    @classmethod
    def _chain_open_options(cls, mesh_device: Any) -> dict[str, Any]:
        # the lanes form reads no verify lanes on a second command queue (``open_mtp_chain``: ``mtp_ple_early`` off)
        ple_early = mtp_ple_early_from_env() and cls._open_lanes == 1
        return {"sampling": False, "mtp": mtp_drafts_from_env(), "mtp_ple_early": ple_early}

    @classmethod
    def initialize_vllm_model(cls, hf_config, mesh_device, max_batch_size, max_seq_len, *args: Any, **kwargs: Any):
        drafts = mtp_drafts_from_env()  # refused by name before any library call
        mtp_ple_early_from_env()
        mtp_hybrid_single_from_env()
        mtp_chain_from_env()
        if max_batch_size != 1:
            if not MIN_LANES <= max_batch_size <= MAX_LANES:
                raise ValueError(
                    f"the MTP adapter serves max_num_seqs 1 or {MIN_LANES}..{MAX_LANES}, got {max_batch_size}"
                )
            lane_geometry(max_batch_size, drafts)  # B x (K + 1) <= 32, refused by name
        cls._open_lanes = max_batch_size
        return super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, max_seq_len, *args, **kwargs)

    @classmethod
    def get_max_tokens_all_users(
        cls, *, max_model_len: int | None = None, max_num_seqs: int | None = None, **kwargs: Any
    ) -> int:
        """The plugin's block-table pool.  The lanes run on vLLM-sized shared paged pools like the plain batch mode, so the
        plain sizing applies: ``max_model_len * max_num_seqs`` capped by the pool's DRAM budget (``batch_pool_token_cap``);
        batch 1 is one context.  ``QWEN38_MAX_TOKENS_ALL_USERS`` overrides."""

        return super().get_max_tokens_all_users(max_model_len=max_model_len, max_num_seqs=max_num_seqs, **kwargs)

    # -- config time -----------------------------------------------------------------------------------------------

    @classmethod
    def spec_plan(cls, vllm_config: Any, max_num_seqs: int, requested_k: int):
        """The plugin's config-time plan: host only, never raises, never touches a device."""

        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        try:
            k = mtp_drafts_from_env()
        except ValueError as error:
            return SpecReject(reason=str(error), supported_k=())
        if max_num_seqs != 1:
            try:
                lane_geometry(max_num_seqs, k)
            except ValueError as error:
                return SpecReject(
                    reason=f"the MTP adapter serves max_num_seqs 1 or {MIN_LANES}..{MAX_LANES} with B x (K + 1) <= 32: {error}",
                    supported_k=(),
                )
        if requested_k < k:
            return SpecReject(
                reason=f"requested_k={requested_k} is below the chain's draft count K={k} ({MTP_DRAFTS_VARIABLE})",
                supported_k=(k,),
            )
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is not None:
            try:
                resident_allocated_context(int(model_config.max_model_len))
            except ValueError as error:
                return SpecReject(reason=str(error), supported_k=())
            text_config = getattr(model_config, "hf_config", None)
            if text_config is not None and hasattr(text_config, "get_text_config"):
                text_config = text_config.get_text_config()
            mtp_layers = getattr(text_config, "mtp_num_hidden_layers", None)
            if mtp_layers is not None and int(mtp_layers) < 1:
                return SpecReject(reason="the checkpoint has no MTP layer (mtp_num_hidden_layers < 1)", supported_k=())
        return SpecPlan(
            effective_k=k,
            lanes_per_request=k + 1,
            extra_bytes_per_seq=0,  # not budgeted by the plugin yet; the chain's live admission checks the DRAM at open
            extra_bytes_per_token=0,
            accept_modes=("argmax_ids", "fused_sample")
            if mtp_chain_from_env() and max_num_seqs != 1
            else ("argmax_ids",),
            drafter_state="internal",
            supports_narrow_decode=False,
        )

    # -- the forward calls -----------------------------------------------------------------------------------------

    def prefill_forward(
        self,
        *,
        tokens: torch.Tensor,
        page_table: Any,
        kv_cache: Any,
        enable_trace: bool,
        prompt_lens: Any,
        start_pos: Any,
        empty_slots: Any,
        sampling_params: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The prompt into the reset state and the MTP layer's rows; returns the first token as a one-hot float32 logits
        row ``[1, 1, VOCAB_SIZE]`` (the plugin's host argmax gives it; sampled requests never reach here) or, for a
        plain greedy ``sampling_params``, as int32 ``[1, 1]``; and the zero mrope delta."""

        self._finish_pending()
        greedy_token = sampling_params is not None
        rows = int(tokens.shape[0]) if self.lanes_mode else None
        if greedy_token and not (
            all(device_greedy_contract(sampling_params, row) for row in range(rows))
            if self.lanes_mode
            else device_greedy_contract(sampling_params)
        ):
            raise ValueError("prefill returns logits: sample_on_device_mode must be decode_only, not all")
        if self.lanes_mode:
            return self._prefill_lanes(
                tokens=tokens,
                page_table=page_table,
                enable_trace=enable_trace,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                empty_slots=empty_slots,
                greedy_token=greedy_token,
                mamba_state=kwargs.get("mamba_state"),
                intermediate=kwargs.get("intermediate_prefill_mask"),
            )
        if tokens.shape[0] != 1:
            raise ValueError(f"one resident slot: tokens must be [1, P], got {tuple(tokens.shape)}")
        if self._mamba_spec_on():
            return self._prefill_single_ranged(
                tokens=tokens,
                page_table=page_table,
                enable_trace=enable_trace,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                empty_slots=empty_slots,
                greedy_token=greedy_token,
                mamba_state=kwargs.get("mamba_state"),
                intermediate=kwargs.get("intermediate_prefill_mask"),
            )
        if int(start_pos[0]) != 0:
            raise ValueError(
                f"prefill must start at position 0 (no prefix caching, no chunking), got {int(start_pos[0])}"
            )
        length = int(prompt_lens[0])
        if not 1 <= length <= min(self.context_limit, tokens.shape[1]):
            raise ValueError(
                f"prompt length {length} must be in [1, {self.context_limit}] and within tokens {tuple(tokens.shape)}"
            )
        if list(empty_slots) != [0]:
            raise ValueError(f"one resident slot: empty_slots must be [0], got {list(empty_slots)}")
        ids = tokens[0, :length].tolist()
        if min(ids) < 0 or max(ids) >= TOKENIZER_SIZE:
            raise ValueError(f"prompt ids must be in [0, {TOKENIZER_SIZE}), got {min(ids)}..{max(ids)}")
        self._require_serviceable(enable_trace, recovering=True)
        page_ids = None
        if self.paging is not None:
            page_ids = self._slot_page_ids(page_table, needed=-(-length // self.block_size), label="prefill")
        return self._chain_prefill(ids, greedy_token, page_ids)[:2]

    def _chain_prefill(
        self, ids: list[int], greedy_token: bool, page_ids: list[int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """The prompt on the chain's pass-loop driver (``page_ids``: the paged chain's request blocks); the first token as
        ``prefill_forward`` returns it (and as an int)."""

        try:
            token = self.driver.prefill(ids, page_ids)
        except BaseException:
            self.slot.failed = True
            raise
        self.slot.failed = False
        if token >= TOKENIZER_SIZE:
            self.slot.failed = True
            raise Qwen38ChatChainError(f"the first token {token} is a padding row of the LM head")
        if greedy_token:
            out = torch.tensor([[token]], dtype=torch.int32)
        else:
            out = torch.zeros(1, 1, VOCAB_SIZE, dtype=torch.float32)
            out[0, 0, TOKENIZER_SIZE:] = float("-inf")
            out[0, 0, token] = 1.0
        return out, torch.zeros(1, dtype=torch.long), token

    def _ranged_row(
        self,
        tokens: torch.Tensor,
        row: int,
        *,
        page_table: Any,
        prompt_lens: Any,
        start_pos: Any,
        mamba_state: Any,
        intermediate: Any,
    ) -> dict[str, Any]:
        """MambaSpec mode: row ``row``'s chunk ``[start, end)`` of the plugin's prefill as the arguments of
        ``Qwen38VLLMMTPDriver.prefill_range`` (``ids``, ``start``, ``final``, the request's ``page_ids`` up to the chunk end, the
        state blocks and the checkpoints).  Refused by name before any device call."""

        start, end = int(start_pos[row]), int(prompt_lens[row])
        if not 1 <= end <= min(self.context_limit, tokens.shape[1]):
            raise ValueError(
                f"prompt length {end} must be in [1, {self.context_limit}] and within tokens {tuple(tokens.shape)}"
            )
        if not 0 <= start < end:
            raise ValueError(f"prefill start {start} must be in [0, {end})")
        final = not (intermediate is not None and bool(intermediate[row]))
        if start % self.block_size:
            raise ValueError(
                f"MambaSpec mode: a prefill that continues a request starts at a block boundary, got {start} with "
                f"block size {self.block_size}"
            )
        if not final and end % self.block_size:
            raise ValueError(
                f"MambaSpec mode: an intermediate chunk ends at a block boundary, got {end} with block size "
                f"{self.block_size}"
            )
        ids = tokens[row, :end].tolist()
        if min(ids) < 0 or max(ids) >= TOKENIZER_SIZE:
            raise ValueError(f"prompt ids must be in [0, {TOKENIZER_SIZE}), got {min(ids)}..{max(ids)}")
        page_ids = None
        if self.paging is not None:
            needed = -(-end // self.block_size)
            if self.lanes_mode:
                page_ids = self._row_page_ids(page_table, row, needed=needed, label=f"prefill row {row}")
            else:
                page_ids = self._slot_page_ids(page_table, needed=needed, label="prefill")
        layer_blocks = self._mamba_layer_blocks(mamba_state, row) if self._needs_state_blocks(end, start) else None
        return dict(
            ids=ids,
            start=start,
            final=final,
            page_ids=page_ids,
            layer_blocks=layer_blocks,
            checkpoints=self._mamba_checkpoints(mamba_state, row, start, end),
        )

    def _run_range(self, row: dict[str, Any]) -> int | None:
        """``prefill_range`` on the chain's pass-loop driver for one ``_ranged_row``; the first token (None: an intermediate
        chunk)."""

        return self.driver.prefill_range(
            row["ids"],
            start=row["start"],
            page_ids=row["page_ids"],
            layer_blocks=row["layer_blocks"],
            checkpoints=row["checkpoints"],
            block_size=self.block_size,
            final=row["final"],
        )

    @staticmethod
    def _first_token_rows(firsts: Sequence[int | None], greedy_token: bool) -> torch.Tensor:
        """The prefill output of the rows' first tokens, as ``prefill_forward`` returns it; an intermediate chunk (None) is
        a row the plugin drops: token 0 or the logits row of token 0."""

        if greedy_token:
            return torch.tensor([[0 if token is None else token] for token in firsts], dtype=torch.int32)
        out = torch.zeros(len(firsts), 1, VOCAB_SIZE, dtype=torch.float32)
        out[:, 0, TOKENIZER_SIZE:] = float("-inf")
        for i, token in enumerate(firsts):
            out[i, 0, 0 if token is None else token] = 1.0
        return out

    def _prefill_single_ranged(
        self,
        *,
        tokens: torch.Tensor,
        page_table: Any,
        enable_trace: bool,
        prompt_lens: Any,
        start_pos: Any,
        empty_slots: Any,
        greedy_token: bool,
        mamba_state: Any,
        intermediate: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch 1 in MambaSpec mode: the chunk ``[start_pos, prompt_lens)`` on the chain (a prefix-cache hit or a later chunk
        restores its state from the blocks); an intermediate chunk returns a dummy row."""

        if list(empty_slots) != [0]:
            raise ValueError(f"one resident slot: empty_slots must be [0], got {list(empty_slots)}")
        row = self._ranged_row(
            tokens,
            0,
            page_table=page_table,
            prompt_lens=prompt_lens,
            start_pos=start_pos,
            mamba_state=mamba_state,
            intermediate=intermediate,
        )
        self._require_serviceable(enable_trace, recovering=True)
        try:
            token = self._run_range(row)
        except BaseException:
            self.slot.failed = True
            raise
        self.slot.failed = False
        if token is not None and token >= TOKENIZER_SIZE:
            self.slot.failed = True
            raise Qwen38ChatChainError(f"the first token {token} is a padding row of the LM head")
        return self._first_token_rows([token], greedy_token), torch.zeros(1, dtype=torch.long)

    def decode_forward(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: Any,
        spec_mode: str | None = None,
        num_valid_drafts: Any = None,
        accepted_counts: Any = None,
        slot_remap: Any = None,
        page_table: Any = None,
        **kwargs: Any,
    ):
        """The verify of the plugin's ``[1, 1+K]`` block: ``VerifyOutput(argmax_ids [1, 1+K])``.  Every decode step of a
        speculative launch is a verify (no narrow decode), so a call without ``spec_mode`` is refused.  Batch mode with
        ``QWEN38_MTP_CHAIN=1`` also answers ``fused_sample`` (see :meth:`_decode_lanes`)."""

        from vllm_tt_plugin.spec_decode import VerifyOutput, check_spec_side_tensors

        if self.lanes_mode:
            chain = {name: kwargs.get(name) for name in ("defer_read", "chain_rows", "chain_tip")}
            return self._decode_lanes(
                tokens=tokens,
                start_pos=start_pos,
                spec_mode=spec_mode,
                num_valid_drafts=num_valid_drafts,
                accepted_counts=accepted_counts,
                slot_remap=slot_remap,
                page_table=page_table,
                **chain,
            )
        if spec_mode is None:
            raise ValueError(
                "ordinary decode reached the MTP adapter: it declares no narrow decode (every step is a verify); "
                "sampled and penalized requests are refused by the plugin at admission"
            )
        if spec_mode != "argmax_ids":
            raise ValueError(f"the MTP adapter answers spec_mode 'argmax_ids', got {spec_mode!r}")
        if self.slot.failed:
            raise Qwen38ChatChainError("a chain call failed earlier; the device state is not trusted")
        k = self.drafts
        tokens = torch.as_tensor(tokens)
        positions = torch.as_tensor(start_pos)
        if tuple(tokens.shape) != (1, k + 1) or tuple(positions.shape) != (1, k + 1):
            raise ValueError(f"one resident slot: the verify block is [1, {k + 1}], got {tuple(tokens.shape)}")
        if num_valid_drafts is None or accepted_counts is None:
            raise ValueError("a verify carries num_valid_drafts and accepted_counts")
        check_spec_side_tensors(num_valid_drafts, accepted_counts, 1, k)
        if slot_remap is not None and [int(slot) for slot in torch.as_tensor(slot_remap).reshape(-1).tolist()] != [0]:
            raise ValueError(f"one resident slot: slot_remap must be the identity, got {slot_remap}")
        if int(positions[0, 0]) < 0:
            raise ValueError("the verify's only row is a padding row (position -1): nothing to verify")
        page_ids = self._verify_pages(page_table, int(positions[0, 0]))  # a refused table touches no device state
        try:
            ids = self.driver.verify(
                tokens[0].tolist(), positions[0].tolist(), int(num_valid_drafts[0]), int(accepted_counts[0]), page_ids
            )
        except BaseException:
            self.slot.failed = True
            raise
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=torch.tensor([ids], dtype=torch.int32), hidden=None)

    def propose_draft_tokens(
        self, num_drafts: int, committed_tokens: Any, committed_positions: Any, accepted_counts: Any, hidden: Any = None
    ):
        """The drafts the last verify left on the host (no device call); ``num_valid`` 0 when the plugin's commit is not
        what the device accepted, or no pass is open."""

        from vllm_tt_plugin.spec_decode import DraftOutput, check_spec_side_tensors

        k = self.drafts
        if self.lanes_mode:
            return self._propose_lanes(num_drafts, committed_tokens, accepted_counts, hidden)
        if num_drafts != k:
            raise ValueError(f"the chain drafts {k} tokens per pass, the plugin asked for {num_drafts}")
        committed = torch.as_tensor(committed_tokens)
        if tuple(committed.shape) != (1, k + 1):
            raise ValueError(f"one resident slot: the committed block is [1, {k + 1}], got {tuple(committed.shape)}")
        check_spec_side_tensors(torch.zeros(1, dtype=torch.int32), accepted_counts, 1, k, call="propose")
        drafts, valid = self.driver.propose(committed[0].tolist(), int(accepted_counts[0]))
        return DraftOutput(
            draft_token_ids=torch.tensor([drafts], dtype=torch.int32),
            num_valid=torch.tensor([valid], dtype=torch.int32),
        )

    def read_decode_output(self, tt_out: Any, async_read: bool = False, sample_rows: Any = None):
        """The plugin's read of ``decode_forward``'s result: a deferred fused step's handle lands (blocking on its device
        event) and returns its ``FusedReadout``; anything else is the plain adapter's read."""

        if isinstance(tt_out, Qwen38MTPFusedHandle):
            if async_read:
                raise ValueError("a fused handle is read with read_decode_output(handle), not with async_read")
            return tt_out.wait()
        return super().read_decode_output(tt_out, async_read=async_read, sample_rows=sample_rows)

    def _finish_pending(self) -> None:
        """Land the deferred fused step in flight, if any (idempotent; the reader thread may have landed it).  Every entry
        point that reads the host mirrors it updates or changes lane state outside a decode step calls it first."""

        handle = self._last_handle
        if handle is not None:
            handle.wait()

    # -- lifecycle -------------------------------------------------------------------------------------------------

    def release_request(self, slot: int) -> None:
        """The request's slot is free.  The pass loop stays open: the next prefill leaves it (``Qwen38VLLMMTPDriver.close``)
        before it resets the state."""

        if self.lanes_mode:
            self._release_lane(slot)
            return
        if slot != 0:
            raise ValueError(f"one resident slot: slot must be 0, got {slot}")

    def release_persistent_capture(self) -> None:
        if self.lanes_mode:
            self._close_lanes()
            return
        if not self.slot.failed and self.driver is not None:
            try:
                self.driver.close()
            except Exception as error:  # the mesh close below frees the traces either way
                logger.warning(f"leaving the MTP pass loop at shutdown failed: {error}")
        super().release_persistent_capture()

    def warmup_model_decode(
        self,
        *,
        enable_trace: bool,
        kv_cache: Any,
        max_batch_size: int,
        num_blocks: int,
        can_sample_on_device: bool,
        **kwargs: Any,
    ) -> None:
        """Phase 1 (``enable_trace`` False): checks.  Phase 2: a prefill, the bootstrap verify and one following verify on
        the drafts it left, under the closed miss guard, then the state back at the reset."""

        self._finish_pending()
        self.warmup_model_prefill(enable_trace=enable_trace, kv_cache=kv_cache, can_sample_on_device=False)
        if self.lanes_mode:
            if max_batch_size != self.lanes:
                raise ValueError(f"batch mode: max_batch_size must be the {self.lanes} lanes, got {max_batch_size}")
            if enable_trace:
                self._warmup_lanes()
            return
        if max_batch_size != 1:
            raise ValueError(f"one resident slot: max_batch_size must be 1, got {max_batch_size}")
        if not enable_trace:
            return
        k = self.drafts
        ids = list(WARMUP_PROMPT_TOKEN_IDS)
        warm_table = None
        if (
            self.paging is not None
        ):  # the warm-up request's blocks: 1.. (block 0 is the scratch), up to the last verify row
            warm_blocks = (len(ids) + 1 + k) // self.block_size + 1
            warm_table = torch.zeros(1, self.paging.max_blocks, dtype=torch.int32)
            warm_table[0, :warm_blocks] = torch.arange(1, warm_blocks + 1, dtype=torch.int32)
        warm_ids = None if warm_table is None else self._slot_page_ids(warm_table, needed=warm_blocks, label="warmup")
        first = self.driver.prefill(ids, warm_ids)
        block = torch.full((1, k + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        block[0, 0] = first
        positions = torch.full((1, k + 1), -1, dtype=torch.int32)
        positions[0, 0] = len(ids)
        committed = self.driver.verify(block[0].tolist(), positions[0].tolist(), 0, 1, warm_ids)[:1]
        drafts, valid = self.driver.propose(committed, 1)
        if valid == k:  # a second pass on the drafts the first one left
            self.driver.verify(
                [committed[0], *drafts], [len(ids) + 1 + column for column in range(k + 1)], k, 1, warm_ids
            )
        self.driver.close()
        try:
            self.session.reset()
        except BaseException:
            self.slot.failed = True
            raise
        self.chain.reset_and_seed(SEED_TOKEN_ID)

    # -- batch 2..8: the lane chain --------------------------------------------------------------------------------

    def allocate_kv_cache(self, kv_cache_shape: Any, dtype: Any, num_layers: int) -> None:
        """Batch mode: open the chain on vLLM's pools (the paging from ``kv_cache_shape``, as the plain adapter's
        ``_open_paged_single``) with the lanes' allocation as its warm hook (the text demo's form), capture the lane traces;
        the lanes share the chain's pools, page id = vLLM block id.  Batch 1: the parent opens the paged chain (vLLM's block
        count and size) through the chain opener, ``_chain_open_options`` applied."""

        if not self.lanes_mode:
            return super().allocate_kv_cache(kv_cache_shape, dtype, num_layers)
        shape = tuple(int(extent) for extent in kv_cache_shape)
        if len(shape) != 4:
            raise ValueError(f"kv_cache_shape must be (num_blocks, heads, block_size, head_size), got {shape}")
        self._open_lanes_chain(num_blocks=shape[0], block_size=shape[2])
        return None

    def _open_batch(self, *, num_blocks: int, block_size: int) -> None:
        self._open_lanes_chain(num_blocks=num_blocks, block_size=block_size)  # MambaSpec mode's per-layer allocation

    def _open_lanes_chain(self, *, num_blocks: int, block_size: int) -> None:
        """Open the chain on ``num_blocks`` pages of ``block_size`` rows (page id = vLLM block id, page 0 the scratch) with the
        lanes' allocation as its warm hook (MambaSpec mode: then the block programs' warm), capture the lane traces."""

        if self.lanes_session is not None:
            raise RuntimeError("the lanes are already open: allocate_kv_cache runs once")
        from models.demos.blackhole.qwen38_flash_next.demo.mtp_demo import _lanes_warm_hook
        from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lanes_session import Qwen38LanesSession

        block_size = check_paged_block_size(block_size)
        if num_blocks < 2:
            raise ValueError(f"vLLM num_blocks must be >= 2 (block 0 is the scratch), got {num_blocks}")
        paging = validate_index_paging(
            Qwen38LanePaging(
                page_size=block_size,
                num_pages=num_blocks,
                max_blocks=paged_table_width(self.allocated_context, block_size),
            ),
            self.allocated_context,
        )
        long_chunks, slab_rows = prefill_form()
        lanes_session = Qwen38LanesSession(lanes=self.lanes, drafts=self.drafts, marker=logger.info)
        self.paging, self.block_size, self.num_blocks = paging, block_size, num_blocks
        block_warm = self._block_warm_hook()  # None when the mode is off

        def warm_hook(chain: Any) -> None:
            _lanes_warm_hook(chain.mesh, lanes_session, self.allocated_context, long_chunks or slab_rows is not None)(
                chain
            )
            if block_warm is not None:
                block_warm(chain)

        logger.info(
            f"mtp lanes: {self.lanes} lanes x {self.drafts} drafts, context {self.allocated_context}, block size "
            f"{block_size}, {num_blocks} blocks, page table width {paging.max_blocks}"
        )
        self.chain = self.chain_opener(warm_hook, paging=paging)
        self.session = Qwen38ChatSession(self.chain, None, allocated_context=self.allocated_context)
        self.session.reset()
        lanes_session.capture(self.chain, self.session)
        self.lanes_session = lanes_session
        self.lane_driver = Qwen38VLLMMTPLanesDriver(
            lanes_session, self.lanes, self.drafts, paged=True, hold_orphans=mtp_chain_from_env()
        )
        self.driver = Qwen38VLLMMTPDriver(self.chain, self.session, self.drafts)  # the hybrid single's pass loop
        logger.info(f"mtp lanes ready: {lanes_session.summary()}")
        summary = lanes_session.summary()
        logger.info(
            f"mtp lanes widths {summary['buckets']} ({MTP_LANE_BUCKETS_VARIABLE}), per-width DRAM growth per bank "
            f"{summary['dram_growth_bytes_per_bank_by_width']}"
        )

    def _take_lane(self, slot: int) -> int:
        """The lane ``slot`` admits into: the LOWEST lane that holds no request, so the live lanes stay low and a narrow
        lane width serves them.  ``lane_of_slot`` stays a permutation: the slot that mapped to the chosen lane takes the
        lane ``slot`` gave up (that slot holds no request on the lanes: its lane is free).  Returns the lane."""

        self._finish_pending()
        driver = self._require_lanes()
        lane = next((candidate for candidate in range(self.lanes) if driver.runs[candidate] is None), None)
        if lane is None:
            raise Qwen38ChatChainError(f"no free lane for slot {slot}: all {self.lanes} lanes hold requests")
        current = self.lane_of_slot[slot]
        if lane != current:
            other = self.lane_of_slot.index(lane)
            self.lane_of_slot[slot], self.lane_of_slot[other] = lane, current
        return lane

    def _require_lanes(self) -> Qwen38VLLMMTPLanesDriver:
        if self.lane_driver is None:
            raise Qwen38ChatChainError("batch mode: allocate_kv_cache opens the chain and the lanes first")
        if self.batch_failed:
            raise Qwen38ChatChainError("a lane call failed earlier; the lane state is not trusted")
        return self.lane_driver

    def _prefill_lanes(
        self,
        *,
        tokens: torch.Tensor,
        page_table: Any = None,
        enable_trace: bool,
        prompt_lens: Any,
        start_pos: Any,
        empty_slots: Any,
        greedy_token: bool,
        mamba_state: Any = None,
        intermediate: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each row alone into the lane of its state slot (the lanes session's admission); the first tokens as int32
        ``[rows, 1]`` for a greedy contract or one-hot float32 logits ``[rows, 1, VOCAB_SIZE]``.  Paged lanes: row ``i``'s
        blocks are the ``ceil(length / block_size)`` leading entries of ``page_table[i]``."""

        driver = self._require_lanes()
        rows = int(tokens.shape[0])
        if tokens.dim() != 2 or not 1 <= rows <= self.lanes:
            raise ValueError(f"batch mode: tokens must be [1..{self.lanes}, P], got {tuple(tokens.shape)}")
        if empty_slots is None or not (len(empty_slots) == len(prompt_lens) == len(start_pos) == rows):
            raise ValueError("batch mode: empty_slots, prompt_lens and start_pos must hold one entry per row")
        slots = [int(slot) for slot in empty_slots]
        if len(set(slots)) != rows or any(not 0 <= slot < self.lanes for slot in slots):
            raise ValueError(f"batch mode: empty_slots must be distinct slots in [0, {self.lanes}), got {slots}")
        if self._mamba_spec_on():
            return self._prefill_lanes_ranged(
                tokens=tokens,
                page_table=page_table,
                enable_trace=enable_trace,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                slots=slots,
                greedy_token=greedy_token,
                mamba_state=mamba_state,
                intermediate=intermediate,
            )
        prompts, pages = [], []
        for i in range(rows):
            if int(start_pos[i]) != 0:
                raise ValueError(
                    f"prefill must start at position 0 (no prefix caching, no chunking), got {int(start_pos[i])}"
                )
            length = int(prompt_lens[i])
            if not 1 <= length <= min(self.context_limit, tokens.shape[1]):
                raise ValueError(
                    f"prompt length {length} must be in [1, {self.context_limit}] and within tokens {tuple(tokens.shape)}"
                )
            ids = tokens[i, :length].tolist()
            if min(ids) < 0 or max(ids) >= TOKENIZER_SIZE:
                raise ValueError(f"prompt ids must be in [0, {TOKENIZER_SIZE}), got {min(ids)}..{max(ids)}")
            prompts.append(ids)
            if self.paging is not None:
                pages.append(
                    self._row_page_ids(page_table, i, needed=-(-length // self.block_size), label=f"prefill row {i}")
                )
        self._require_serviceable(enable_trace, recovering=True)
        if self.mtp_hybrid_slot in slots:
            logger.warning(f"prefill into slot {self.mtp_hybrid_slot} whose request still runs on the chain: released")
            self._release_hybrid()
        if self._hybrid_starts(rows):
            try:
                out, rope, first = self._chain_prefill(prompts[0], greedy_token, pages[0] if pages else None)
            except BaseException:
                self.batch_failed = True
                raise
            self.mtp_hybrid_slot, self.mtp_hybrid_pending = slots[0], first
            self.mtp_hybrid_pages = pages[0] if pages else None
            return out, rope
        if self.mtp_hybrid_slot is not None:
            self._move_hybrid_to_lane()  # before the admission below resets the chain
        firsts = []
        try:
            for index, (slot, ids) in enumerate(zip(slots, prompts)):
                lane = self.lane_of_slot[slot]
                if driver.runs[lane] is not None:
                    logger.warning(f"prefill into slot {slot} whose lane {lane} still holds a request: released")
                    driver.release(lane)
                lane = self._take_lane(slot)
                firsts.append(driver.admit(lane, ids, pages[index] if pages else None))
        except BaseException:
            self.batch_failed = True
            raise
        for token in firsts:
            if token >= TOKENIZER_SIZE:
                self.batch_failed = True
                raise Qwen38ChatChainError(f"the first token {token} is a padding row of the LM head")
        if greedy_token:
            out = torch.tensor([[token] for token in firsts], dtype=torch.int32)
        else:
            out = torch.zeros(rows, 1, VOCAB_SIZE, dtype=torch.float32)
            out[:, 0, TOKENIZER_SIZE:] = float("-inf")
            for i, token in enumerate(firsts):
                out[i, 0, token] = 1.0
        return out, torch.zeros(rows, dtype=torch.long)

    def _prefill_lanes_ranged(
        self,
        *,
        tokens: torch.Tensor,
        page_table: Any,
        enable_trace: bool,
        prompt_lens: Any,
        start_pos: Any,
        slots: list[int],
        greedy_token: bool,
        mamba_state: Any,
        intermediate: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """MambaSpec mode: each row's chunk ``[start_pos, prompt_lens)`` alone on the chain (``prefill_range``: the restore from
        the state blocks at a start past 0, the saves at the checkpoints and a block-aligned end).  An intermediate chunk
        takes no lane (a dummy row).  The last chunk imports the chain's state into the lane of its slot (no second prefill),
        or, for a lone request with nothing on the lanes or the chain, stays on the chain's pass loop."""

        driver = self._require_lanes()
        rows = [
            self._ranged_row(
                tokens,
                i,
                page_table=page_table,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                mamba_state=mamba_state,
                intermediate=intermediate,
            )
            for i in range(len(slots))
        ]
        self._require_serviceable(enable_trace, recovering=True)
        if self.mtp_hybrid_slot in slots:
            logger.warning(f"prefill into slot {self.mtp_hybrid_slot} whose request still runs on the chain: released")
            self._release_hybrid()
        hybrid = self._hybrid_starts(len(rows)) and rows[0]["final"]
        if self.mtp_hybrid_slot is not None:
            self._move_hybrid_to_lane()  # before the chain is reset or restored
        firsts: list[int | None] = []
        try:
            for slot, row in zip(slots, rows):
                lane = self.lane_of_slot[slot]
                if driver.runs[lane] is not None:
                    logger.warning(f"prefill into slot {slot} whose lane {lane} still holds a request: released")
                    driver.release(lane)
                token = self._run_range(row)
                firsts.append(token)
                if token is None or hybrid:
                    continue
                if token >= TOKENIZER_SIZE:
                    raise Qwen38ChatChainError(f"the first token {token} is a padding row of the LM head")
                end = len(row["ids"])
                if len(self.session.committed) != end or int(self.chain.position()) != end:
                    raise Qwen38ChatChainError(
                        f"chain prefill: committed {len(self.session.committed)} rows, device position "
                        f"{int(self.chain.position())}, expected {end}"
                    )
                driver.admit_from_chain(self._take_lane(slot), end, token, 1, self.session.ple_context, row["page_ids"])
        except BaseException:
            self.batch_failed = self.slot.failed = True
            raise
        self.slot.failed = False
        if hybrid:
            if firsts[0] >= TOKENIZER_SIZE:
                self.batch_failed = self.slot.failed = True
                raise Qwen38ChatChainError(f"the first token {firsts[0]} is a padding row of the LM head")
            self.mtp_hybrid_slot, self.mtp_hybrid_pending = slots[0], firsts[0]
            self.mtp_hybrid_pages = rows[0]["page_ids"]
        return self._first_token_rows(firsts, greedy_token), torch.zeros(len(rows), dtype=torch.long)

    def _decode_lanes(
        self,
        *,
        tokens: Any,
        start_pos: Any,
        spec_mode: str | None,
        num_valid_drafts: Any,
        accepted_counts: Any,
        slot_remap: Any,
        page_table: Any = None,
        defer_read: Any = None,
        chain_rows: Any = None,
        chain_tip: Any = None,
    ):
        """One lane pass for the live rows of the plugin's ``[lanes, 1+K]`` verify block.  Row ``i`` is live when
        ``start_pos[i, 0] >= 0`` and runs in lane ``lane_of_slot[slot_remap[i]]`` (the map the plain batch adapter keeps:
        ``slot_remap`` is applied to it once, no device move).  Returns ``VerifyOutput(argmax_ids [lanes, 1+K])``, padding
        rows 0.  Paged lanes: row ``i``'s lane takes the blocks of ``_pass_blocks`` from ``page_table[i]``.

        ``QWEN38_MTP_CHAIN=1``: ``spec_mode="fused_sample"`` answers ``VerifyOutput(accepted_token_ids [lanes, 1+K]
        (-1 past the count), accepted_counts [lanes])``, or with ``defer_read`` a ``VerifyOutput`` holding a
        :class:`Qwen38MTPFusedHandle` whose read returns the ``FusedReadout``.  The deferred pass is enqueued and not landed:
        the next decode call (and every other entry point that touches lane state) lands it first
        (:meth:`_finish_pending`).  ``chain_rows`` (bool ``[lanes]``) marks the rows the plugin chained onto that pass: their
        tokens, position and accepted count come from the lane's own state; ``chain_tip`` must name the pending step."""

        from vllm_tt_plugin.spec_decode import VerifyOutput, check_spec_side_tensors

        if spec_mode is None:
            raise ValueError(
                "ordinary decode reached the MTP adapter: it declares no narrow decode (every step is a verify); "
                "sampled and penalized requests are refused by the plugin at admission"
            )
        fused = spec_mode == "fused_sample"
        if spec_mode != "argmax_ids" and not (fused and mtp_chain_from_env()):
            raise ValueError(f"the MTP adapter answers spec_mode 'argmax_ids', got {spec_mode!r}")
        driver = self._require_lanes()
        self._finish_pending()  # this step's host rows need the last step's tokens
        k, lanes = self.drafts, self.lanes
        tokens = torch.as_tensor(tokens)
        positions = torch.as_tensor(start_pos)
        if tuple(tokens.shape) != (lanes, k + 1) or tuple(positions.shape) != (lanes, k + 1):
            raise ValueError(
                f"batch mode: the verify block is [{lanes}, {k + 1}], got tokens {tuple(tokens.shape)}, positions "
                f"{tuple(positions.shape)}"
            )
        if num_valid_drafts is None:
            raise ValueError("a verify carries num_valid_drafts and accepted_counts")
        chained = self._chain_mask(chain_rows, chain_tip, fused, lanes)
        if chain_rows is not None and (chain_tip is None or chain_tip != self._tip_id):
            raise ValueError(f"chain_tip {chain_tip} is not the newest fused step {self._tip_id}")
        if accepted_counts is None:
            if not chained.any():
                raise ValueError("a verify carries num_valid_drafts and accepted_counts")
            accepted_counts = torch.full((lanes,), -1, dtype=torch.int32)
        counts = accepted_counts.clone()
        counts[chained] = 1  # a chained row's count is the lane's own (the plugin sends -1)
        check_spec_side_tensors(num_valid_drafts, counts, lanes, k)
        remap = parse_slot_remap(slot_remap, lanes)
        new_map = [self.lane_of_slot[remap[row]] for row in range(lanes)]
        live_rows = [row for row in range(lanes) if int(positions[row, 0]) >= 0]
        if not live_rows:
            raise ValueError("batch mode: a verify needs a live row (start_pos >= 0)")
        if any(bool(chained[row]) and row not in live_rows for row in range(lanes)):
            raise ValueError(f"chain_rows marks a padding row: {chained.tolist()}")
        if self.mtp_hybrid_slot is not None:
            if [remap[row] for row in live_rows] == [self.mtp_hybrid_slot]:
                return self._decode_hybrid(
                    live_rows[0],
                    tokens,
                    positions,
                    num_valid_drafts,
                    accepted_counts,
                    remap,
                    new_map,
                    page_table,
                    fused=fused,
                    defer_read=bool(defer_read),
                    chained=bool(chained[live_rows[0]]),
                )
        pages = {}
        if self.paging is not None:  # a refused table touches no device state
            pages = {
                row: self._row_page_ids(
                    page_table,
                    row,
                    needed=self._pass_blocks(
                        driver.chain_position(new_map[row], int(positions[row, 0]))
                        if chained[row]
                        else int(positions[row, 0])
                    ),
                    label=f"verify row {row}",
                )
                for row in live_rows
            }
        if self.mtp_hybrid_slot is not None:
            self._move_hybrid_to_lane()
        rows = [
            VerifyRow(
                new_map[row],
                tokens[row].tolist(),
                int(positions[row, 0]),
                int(num_valid_drafts[row]),
                -1 if chained[row] else int(accepted_counts[row]),
                pages.get(row),
            )
            for row in live_rows
        ]
        deferred = fused and bool(defer_read)
        try:
            if fused:
                pending = driver.verify_enqueue(
                    rows, {new_map[row] for row in live_rows if chained[row]}, defer_read=deferred
                )
            else:
                by_lane = driver.verify(rows)
        except BaseException:
            self.batch_failed = True
            raise
        self._pending_note = (
            {remap[row]: row for row in range(lanes) if remap[row] != row} if remap != list(range(lanes)) else None
        )
        self.lane_of_slot = new_map
        if not fused:
            self._tip_id = None  # a synchronous verify: nothing chains onto it or onto the step before it
            out = torch.zeros(lanes, k + 1, dtype=torch.int32)
            for row in live_rows:
                out[row] = torch.tensor(by_lane[new_map[row]], dtype=torch.int32)
            return VerifyOutput(spec_mode="argmax_ids", argmax_ids=out, hidden=None)
        snapshot_map = list(new_map)  # a later admission may swap lanes in ``lane_of_slot`` in place

        def land():
            try:
                driver.verify_finish(pending)
            except BaseException:
                self.batch_failed = True
                raise
            handle.runs = {snapshot_map[row]: driver.runs[snapshot_map[row]] for row in live_rows}
            return self._lane_readout(handle.runs, live_rows, snapshot_map)

        if not deferred:
            try:
                driver.verify_finish(pending)
            except BaseException:
                self.batch_failed = True
                raise
            runs = {new_map[row]: driver.runs[new_map[row]] for row in live_rows}
            readout = self._lane_readout(runs, live_rows, new_map)
            self._tip_id = None  # no handle: the plugin cannot chain onto this step
            return VerifyOutput(
                spec_mode="fused_sample",
                accepted_token_ids=readout.accepted_token_ids,
                accepted_counts=readout.accepted_counts,
            )
        handle = self._new_handle(land)
        handle.lane_of_slot = snapshot_map
        return VerifyOutput(spec_mode="fused_sample", handle=handle)

    @staticmethod
    def _chain_mask(chain_rows: Any, chain_tip: Any, fused: bool, rows: int) -> torch.Tensor:
        """``chain_rows`` as a bool ``[rows]`` tensor (all False without it); ``chain_tip`` goes with it (the caller checks it
        against the newest step)."""

        if chain_rows is None:
            if chain_tip is not None:
                raise ValueError("chain_tip without chain_rows")
            return torch.zeros(rows, dtype=torch.bool)
        if not fused:
            raise ValueError("chain_rows on a verify that is not fused_sample")
        mask = torch.as_tensor(chain_rows)
        if mask.dtype != torch.bool or tuple(mask.shape) != (rows,):
            raise ValueError(f"chain_rows must be bool [{rows}], got {chain_rows}")
        return mask

    def _new_handle(self, land: Callable[[], Any]) -> Qwen38MTPFusedHandle:
        self._tip_count += 1
        handle = Qwen38MTPFusedHandle(self._tip_count, land)
        self._last_handle = handle
        self._tip_id = handle.tip_id
        return handle

    def _lane_readout(self, runs: Mapping[int, Any], live_rows: Sequence[int], new_map: Sequence[int]):
        """The ``FusedReadout`` of a landed lane pass in the plugin's row order: row ``i`` is the lane ``new_map[i]``'s pass
        (its committed ids, ``accepted + 1`` of them, -1 past the count; the column-0 position the pass really started at);
        a padding row has count 1, p0 -1 and a real column-0 id (token 0, the contract's "its column-0 argmax": the
        plugin's AcceptResult refuses a placeholder inside any row's committed prefix), -1 past it."""

        from vllm_tt_plugin.spec_decode import FusedReadout

        k, lanes = self.drafts, self.lanes
        ids = torch.full((lanes, k + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        ids[:, 0] = PADDING_ROW_TOKEN_ID
        counts = torch.ones(lanes, dtype=torch.int32)
        p0 = torch.full((lanes,), -1, dtype=torch.int32)
        for row in live_rows:
            run = runs[new_map[row]]
            count = run.accepted + 1
            ids[row, :count] = torch.tensor(run.argmaxes[:count], dtype=torch.int32)
            counts[row] = count
            p0[row] = run.position
        return FusedReadout(ids, counts, p0)

    def _propose_lanes(self, num_drafts: int, committed_tokens: Any, accepted_counts: Any, hidden: Any = None):
        """The drafts of the step ``hidden`` names (a fused handle: its own snapshots, a later step may already have changed
        the live state), else of the last pass."""

        from vllm_tt_plugin.spec_decode import DraftOutput, check_spec_side_tensors

        driver = self._require_lanes()
        k, lanes = self.drafts, self.lanes
        if num_drafts != k:
            raise ValueError(f"the chain drafts {k} tokens per pass, the plugin asked for {num_drafts}")
        committed = torch.as_tensor(committed_tokens)
        if tuple(committed.shape) != (lanes, k + 1):
            raise ValueError(f"batch mode: the committed block is [{lanes}, {k + 1}], got {tuple(committed.shape)}")
        check_spec_side_tensors(torch.zeros(lanes, dtype=torch.int32), accepted_counts, lanes, k, call="propose")
        handle = hidden if isinstance(hidden, Qwen38MTPFusedHandle) else None
        if handle is not None:
            handle.wait()
        hybrid_slot = self.mtp_hybrid_slot if handle is None else handle.hybrid_slot
        lane_of_slot = self.lane_of_slot if handle is None else handle.lane_of_slot
        drafts = torch.zeros(lanes, k, dtype=torch.int32)
        valid = torch.zeros(lanes, dtype=torch.int32)
        for row in range(lanes):  # row ``row`` is state slot ``row`` after the verify's remap
            if hybrid_slot is not None:
                if row == hybrid_slot:
                    row_drafts, count = self.driver.propose(
                        committed[row].tolist(),
                        int(accepted_counts[row]),
                        None if handle is None else handle.single_run,
                    )
                    drafts[row] = torch.tensor(row_drafts, dtype=torch.int32)
                    valid[row] = count
                continue
            lane = lane_of_slot[row]
            row_drafts, count = driver.propose(
                lane,
                committed[row].tolist(),
                int(accepted_counts[row]),
                _LIVE if handle is None else handle.runs.get(lane),
            )
            drafts[row] = torch.tensor(row_drafts, dtype=torch.int32)
            valid[row] = count
        return DraftOutput(draft_token_ids=drafts, num_valid=valid)

    def _hybrid_starts(self, rows: int) -> bool:
        """A lone prompt with no request on the lanes and none on the chain runs on the chain's pass loop."""

        return (
            self.mtp_hybrid_single and rows == 1 and self.mtp_hybrid_slot is None and not self._require_lanes().holders
        )

    def _decode_hybrid(
        self,
        row: int,
        tokens,
        positions,
        num_valid_drafts,
        accepted_counts,
        remap,
        new_map,
        page_table=None,
        *,
        fused: bool = False,
        defer_read: bool = False,
        chained: bool = False,
    ):
        """The only live row is the chain's request: one chain pass (``Qwen38VLLMMTPDriver.verify``); the other rows are
        padding (0), as the lane path pads its idle rows.  The request's slot follows the row (``slot_remap``).  Paged: the
        chain's table takes the row's blocks of ``_pass_blocks`` (as batch 1) before the pass.  A chained row takes its block,
        position and accepted count from the pass loop's last pass.  ``fused``: the pass stays synchronous; with
        ``defer_read`` the result is a :class:`Qwen38MTPFusedHandle` that is already landed."""

        from vllm_tt_plugin.spec_decode import FusedReadout, VerifyOutput

        k, lanes = self.drafts, self.lanes
        block, position_row = tokens[row].tolist(), positions[row].tolist()
        accepted, num_valid = int(accepted_counts[row]), int(num_valid_drafts[row])
        if chained:
            run = self.driver.run
            if run is None:
                raise Qwen38ChatChainError("chain row on the hybrid single with no pass to chain onto")
            real = run.position + run.accepted + 1
            if any(token != PLACEHOLDER_TOKEN_ID for token in block) or accepted != -1:
                raise ValueError(
                    f"chain row {row} must carry placeholder tokens and accepted_counts -1, got {block}, {accepted}"
                )
            if position_row[0] < real:
                raise ValueError(
                    f"chain row {row}: provisional position {position_row[0]} is below the real one {real}"
                )
            block, position_row = list(run.next_tokens), [real + column for column in range(k + 1)]
            accepted, num_valid = run.accepted + 1, min(num_valid, k)
        page_ids = None
        if self.paging is not None:  # a refused table touches no device state
            page_ids = self._row_page_ids(
                page_table, row, needed=self._pass_blocks(position_row[0]), label=f"verify row {row}"
            )
        try:
            ids = self.driver.verify(block, position_row, num_valid, accepted, page_ids)
        except BaseException:
            self.batch_failed = self.slot.failed = True
            raise
        self.mtp_hybrid_pages = page_ids
        self._pending_note = (
            {remap[r]: r for r in range(lanes) if remap[r] != r} if remap != list(range(lanes)) else None
        )
        self.lane_of_slot = new_map
        self.mtp_hybrid_slot = row
        if not fused:
            self._tip_id = None
            out = torch.zeros(lanes, k + 1, dtype=torch.int32)
            out[row] = torch.tensor(ids, dtype=torch.int32)
            return VerifyOutput(spec_mode="argmax_ids", argmax_ids=out, hidden=None)
        run = self.driver.run
        count = run.accepted + 1
        out = torch.full((lanes, k + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        out[:, 0] = PADDING_ROW_TOKEN_ID  # the padding rows' committed column (see _lane_readout)
        out[row, :count] = torch.tensor(ids[:count], dtype=torch.int32)
        counts = torch.ones(lanes, dtype=torch.int32)
        counts[row] = count
        p0 = torch.full((lanes,), -1, dtype=torch.int32)
        p0[row] = run.position
        if not defer_read:
            self._tip_id = None
            return VerifyOutput(spec_mode="fused_sample", accepted_token_ids=out, accepted_counts=counts)
        readout = FusedReadout(out, counts, p0)
        handle = self._new_handle(lambda: readout)
        handle.lane_of_slot, handle.hybrid_slot, handle.single_run = list(new_map), row, run
        handle.wait()
        return VerifyOutput(spec_mode="fused_sample", handle=handle)

    def _move_hybrid_to_lane(self) -> None:
        """The chain's request into its lane (no reverse move exists): the chain leaves verify mode at the rows its last pass
        committed (the rows ``propose`` answered for), the lane imports that state and starts on the plugin's block
        ``[t', d_1 .. d_K]``.  The chain is free for the next admission, which resets it."""

        self._finish_pending()
        driver, single = self._require_lanes(), self.driver
        lane = self._take_lane(self.mtp_hybrid_slot)
        run = single.run
        try:
            if run is None:  # no pass ran: the prompt's 1-row state and the first token, as an admission leaves them
                position, pending, rows = len(single.session.committed), self.mtp_hybrid_pending, 1
            else:
                position, pending, rows = run.position + run.accepted + 1, run.next_tokens[0], run.accepted + 1
                single.close()
            if len(single.session.committed) != position or int(self.chain.position()) != position:
                raise Qwen38ChatChainError(
                    f"chain move: committed {len(single.session.committed)} rows, device position "
                    f"{int(self.chain.position())}, expected {position}"
                )
            driver.admit_from_chain(lane, position, pending, rows, single.session.ple_context, self.mtp_hybrid_pages)
        except BaseException:
            self.batch_failed = self.slot.failed = True
            raise
        logger.info(f"mtp hybrid single: slot {self.mtp_hybrid_slot} moved to lane {lane} at position {position}")
        self.mtp_hybrid_slot = None
        self.mtp_hybrid_pages = None

    def _release_hybrid(self) -> None:
        """The chain's request is gone.  Its pass loop is left now: an admission's reset needs the 1-row state."""

        self.mtp_hybrid_slot = None
        self.mtp_hybrid_pages = None
        try:
            self.driver.close()
        except BaseException:
            self.batch_failed = self.slot.failed = True
            raise

    def _release_lane(self, slot: int) -> None:
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < self.lanes:
            raise ValueError(f"slot must be in [0, {self.lanes}), got {slot!r}")
        self._finish_pending()
        driver = self._require_lanes()
        if slot == self.mtp_hybrid_slot:
            self._release_hybrid()
            return
        driver.release(self.lane_of_slot[slot])

    def _close_lanes(self) -> None:
        """Worker shutdown: the lane traces and states, then the chain."""

        if self.batch_failed:
            logger.warning("a lane call failed; the traces and states are left to the mesh close")
            return
        try:
            self._finish_pending()
        except Exception as error:  # the mesh close frees the traces either way
            logger.warning(f"landing the pending lane pass at shutdown failed: {error}")
        if self.mtp_hybrid_slot is not None:
            self.mtp_hybrid_slot = None
            try:
                self.driver.close()
            except Exception as error:  # the mesh close frees the traces either way
                logger.warning(f"leaving the MTP pass loop at shutdown failed: {error}")
        if self.lanes_session is not None:
            try:
                self.lanes_session.close()
            except Exception as error:  # the mesh close frees them either way
                logger.warning(f"closing the lanes failed: {error}")
        if self.chain is not None:
            self.chain.close()

    def _warmup_lanes(self) -> None:
        """Two requests through the serving calls: both prefills, the bootstrap verify, a following verify on the drafts it
        left, both releases.  Runs under the closed miss guard in the vLLM process, before the first request."""

        k, lanes = self.drafts, self.lanes
        driver = self._require_lanes()
        slots = [0, 1]  # two prompts never start a hybrid single: the warmup runs the lane path only
        prompt = torch.tensor([list(WARMUP_PROMPT_TOKEN_IDS)] * len(slots), dtype=torch.int32)
        length = int(prompt.shape[1])
        warm_table = self._warm_page_table(lanes, len(slots), length + k + 2)  # rows up to the second pass's last
        out, _ = self.prefill_forward(
            tokens=prompt,
            page_table=None if warm_table is None else warm_table[: len(slots)],
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[prompt.shape[1]] * len(slots),
            start_pos=[0] * len(slots),
            empty_slots=slots,
        )
        firsts = [int(out[i, 0].argmax()) for i in range(len(slots))]
        block = torch.full((lanes, k + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        positions = torch.full((lanes, k + 1), -1, dtype=torch.int32)
        for row, first in zip(slots, firsts):
            block[row, 0] = first
            positions[row, 0] = length
        valid = torch.zeros(lanes, dtype=torch.int32)
        counts = torch.ones(lanes, dtype=torch.int32)
        verified = self.decode_forward(
            tokens=block,
            start_pos=positions,
            spec_mode="argmax_ids",
            num_valid_drafts=valid,
            accepted_counts=counts,
            page_table=warm_table,
        )
        committed = torch.zeros(lanes, k + 1, dtype=torch.int32)
        for row in slots:
            committed[row, 0] = verified.argmax_ids[row, 0]
        offered = self.propose_draft_tokens(k, committed, positions, counts)
        if all(int(offered.num_valid[row]) == k for row in slots):  # a second pass on the drafts the first one left
            block2 = torch.full((lanes, k + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
            positions2 = torch.full((lanes, k + 1), -1, dtype=torch.int32)
            for row in slots:
                block2[row, 0] = committed[row, 0]
                block2[row, 1:] = offered.draft_token_ids[row]
                positions2[row] = torch.arange(length + 1, length + 2 + k, dtype=torch.int32)
            self.decode_forward(
                tokens=block2,
                start_pos=positions2,
                spec_mode="argmax_ids",
                num_valid_drafts=torch.tensor([k if row in slots else 0 for row in range(lanes)], dtype=torch.int32),
                accepted_counts=counts,
                page_table=warm_table,
            )
        warm_widths = getattr(self.lanes_session, "warm_widths", None)
        if warm_widths is not None:  # every lane width's traces replay once here (the two warm lanes are still live)
            logger.info(f"mtp lanes: warm replays at widths {warm_widths()}")
        if self._mamba_spec_on():
            self.release_request(slots[1])  # its lane takes the continuation chunk; the other warm lane stays live
            self._warmup_prefix(slots[1])
        for slot in slots:
            self.release_request(slot)
        if self.mtp_hybrid_single:
            self._warmup_hybrid()

    def _warmup_prefix(self, slot: int) -> None:
        """MambaSpec mode, under the closed miss guard: an intermediate chunk of one block that saves its state block, then the
        last chunk from that boundary that restores it (the chunk trace at a start past 0, the restore, the lane import).  The
        state blocks follow the KV blocks of the warm table: state saved into a KV block would put float32 bytes into KV
        rows."""

        block_size = self.block_size
        prompt = list(WARMUP_PROMPT_TOKEN_IDS)
        length = block_size + len(prompt)
        ids = (prompt * (-(-length // len(prompt))))[:length]
        kv_blocks = -(-length // block_size)
        by_slot: dict[int, list[int]] = {}
        for layer, gdn in sorted(self.gdn_slot.items()):
            by_slot.setdefault(gdn, []).append(layer)
        groups = [
            [layers[g] for layers in by_slot.values() if len(layers) > g]
            for g in range(max(len(layers) for layers in by_slot.values()))
        ]  # a block of a slot holds one state: the groups' layers of a slot take different blocks
        first_state = 1 + kv_blocks
        if first_state + len(groups) > self.num_blocks:
            raise ValueError(
                f"the prefix warm-up needs {first_state + len(groups)} blocks, the pool has {self.num_blocks}"
            )
        warm_table = self._warm_page_table(1, 1, length)
        mamba_state = SimpleNamespace(
            groups=tuple(
                SimpleNamespace(
                    layer_names=tuple(f"model.layers.{layer}.linear_attn" for layer in layers),
                    windows=torch.tensor([[first_state + g]], dtype=torch.int32),
                    checkpoint_blocks=None,
                )
                for g, layers in enumerate(groups)
            ),
            checkpoint_positions=None,
        )
        for start, end, last in ((0, block_size, False), (block_size, length, True)):
            self.prefill_forward(
                tokens=torch.tensor([ids[:end]], dtype=torch.int32),
                page_table=warm_table,
                kv_cache=None,
                enable_trace=True,
                prompt_lens=[end],
                start_pos=[start],
                empty_slots=[slot],
                mamba_state=mamba_state,
                intermediate_prefill_mask=[not last],
            )

    def _warmup_hybrid(self) -> None:
        """One lone request through the hybrid path with the serving calls, under the closed miss guard: the prefill on the
        chain, a bootstrap pass and two following passes of the chain's pass loop, the move into lane 0, one lane pass on the
        plugin's block, the release.  Leaves the adapter as a fresh server."""

        k, lanes = self.drafts, self.lanes
        ids = list(WARMUP_PROMPT_TOKEN_IDS)
        length = len(ids)
        warm_table = self._warm_page_table(lanes, 1, length + 4 * (k + 1) + 1)  # rows up to the lane pass's last
        clock, marks = time.perf_counter, []
        mark = clock()

        def lap() -> None:
            nonlocal mark
            now = clock()
            marks.append(now - mark)
            mark = now

        def block_of(token: int, drafts: Sequence[int], position: int, accepted: int):
            tokens = torch.full((lanes, k + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
            positions = torch.full((lanes, k + 1), -1, dtype=torch.int32)
            valid = torch.zeros(lanes, dtype=torch.int32)
            counts = torch.ones(lanes, dtype=torch.int32)
            tokens[0, : 1 + len(drafts)] = torch.tensor([token, *drafts], dtype=torch.int32)
            positions[0] = torch.arange(position, position + k + 1, dtype=torch.int32)
            valid[0], counts[0] = len(drafts), accepted
            return dict(
                tokens=tokens,
                start_pos=positions,
                spec_mode="argmax_ids",
                num_valid_drafts=valid,
                accepted_counts=counts,
                page_table=warm_table,
            )

        out, _ = self.prefill_forward(
            tokens=torch.tensor([ids], dtype=torch.int32),
            page_table=None if warm_table is None else warm_table[:1],
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[length],
            start_pos=[0],
            empty_slots=[0],
        )
        if self.mtp_hybrid_slot != 0:
            raise Qwen38ChatChainError("the hybrid warmup prefill did not start on the chain")
        lap()
        committed, drafts, accepted = [int(out[0, 0].argmax())], [], 1
        for _ in range(3):  # the bootstrap pass, then passes on the drafts the proposals return
            position = length + len(committed) - 1
            verified = self.decode_forward(**block_of(committed[-1], drafts, position, accepted))
            accepted = self.driver.run.accepted + 1  # what the plugin's walk commits of a pass the device decided
            new = verified.argmax_ids[0, :accepted].tolist()
            block = torch.zeros(lanes, k + 1, dtype=torch.int32)
            block[0, :accepted] = torch.tensor(new, dtype=torch.int32)
            counts = torch.ones(lanes, dtype=torch.int32)
            counts[0] = accepted
            offered = self.propose_draft_tokens(k, block, torch.zeros(lanes, k + 1, dtype=torch.int32), counts)
            committed += new
            drafts = offered.draft_token_ids[0, : int(offered.num_valid[0])].tolist()
        lap()
        self._move_hybrid_to_lane()
        lap()
        position = length + len(committed) - 1
        self.decode_forward(**block_of(committed[-1], drafts, position, accepted))
        lap()
        self.release_request(0)
        self.lane_of_slot = list(range(lanes))
        logger.info(
            "mtp hybrid warmup: prefill {:.3f} s, 3 chain passes {:.3f} s, move {:.3f} s, lane pass {:.3f} s".format(
                *marks
            )
        )
