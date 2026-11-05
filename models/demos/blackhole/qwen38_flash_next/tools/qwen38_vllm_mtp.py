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

Hybrid single (``QWEN38_MTP_HYBRID_SINGLE``, default on, batch mode only): a lone request, prefilled when no lane holds a
request, runs on the chain's own pass loop (the batch-1 path above, same K) in a lane-free state; the lane chain is not
passed.  When a second request arrives (or a step carries another live row) the lone request is settled at the rows its last
pass committed and imported into its lane without a new prefill (``Qwen38VLLMMTPLanesDriver.admit_from_chain``); from then on
everything runs on the lanes.  There is no move back to the chain.

No narrow decode (``SpecPlan.supports_narrow_decode`` False): the MTP chain has no sampled tail, so every step is a verify
and the plugin refuses a request with a temperature or a penalty when it arrives.
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
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
    device_greedy_contract,
    parse_slot_remap,
    prefill_form,
    resident_allocated_context,
)

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
DEFAULT_DRAFTS = 4
PLACEHOLDER_TOKEN_ID = -1  # the plugin's candidate block pad (vllm_tt_plugin.spec_decode.PLACEHOLDER_TOKEN_ID)


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


class _MTPModelCapabilities:
    """The plain adapter's capabilities with the speculative keys; computed on first access like the plain ones."""

    def __get__(self, instance, owner) -> dict:
        capabilities = dict(_ModelCapabilities().__get__(instance, owner))
        capabilities.update(
            {
                "supports_spec_decode": True,
                # the drafter proposes on the device and reads no hidden state the runner feeds it
                "spec_requirements": ["device_propose"],
                "spec_hidden_handoff": [],
                "output_tokens_per_step": 1,
                # every verify is synchronous: the platform turns async scheduling off with a warning
                "supports_async_decode": False,
                "supports_async_spec_decode": False,
                # the chain has no sampled tail: prefill returns a logits row and the decode steps are verifies
                "supports_sample_on_device": False,
                "supports_device_penalties": False,
            }
        )
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

    def prefill(self, ids: Sequence[int]) -> int:
        """The prompt into the reset state as ``Qwen38ChatSession.complete`` prefills a chunked prompt; returns the model's
        first token (the token row after the last prompt token's TAIL).  Leaves an open pass loop first."""

        self.close()
        session = self.session
        session.reset()
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

    # -- verify ----------------------------------------------------------------------------------------------------

    def verify(self, tokens: Sequence[int], positions: Sequence[int], num_valid: int, accepted: int) -> list[int]:
        """One verify block of the plugin: ``tokens`` ``[t, d_1 .. d_K]`` at ``positions``, ``num_valid`` real drafts,
        ``accepted`` the tokens the plugin's previous step committed for the request (1 after a prefill).  Returns the
        pass's ``K + 1`` argmaxes (``argmax_ids``)."""

        k = self.drafts
        tokens = [int(token) for token in tokens]
        if len(tokens) != k + 1 or len(positions) != k + 1:
            raise ValueError(
                f"the verify block must be {k + 1} wide, got {len(tokens)} tokens, {len(positions)} positions"
            )
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

    def propose(self, committed: Sequence[int], accepted: int) -> tuple[list[int], int]:
        """The drafts the last pass left, with their count: ``(d_1 .. d_K, K)`` when the plugin committed what the device
        accepted, else ``([0] * K, 0)`` (a decline: no pass is open, or the plugin cut the pass)."""

        k = self.drafts
        run = self.run
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


class VerifyRow:
    """One live row of a verify call: the lane it runs in, the plugin's block, its position, its valid drafts and the tokens
    the plugin committed for the request in its previous step (1 after a prefill)."""

    __slots__ = ("lane", "tokens", "position", "num_valid", "accepted")

    def __init__(self, lane: int, tokens: Sequence[int], position: int, num_valid: int, accepted: int) -> None:
        self.lane = int(lane)
        self.tokens = [int(token) for token in tokens]
        self.position = int(position)
        self.num_valid = int(num_valid)
        self.accepted = int(accepted)


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

    def __init__(self, device: Any, lanes: int, drafts: int) -> None:
        lane_geometry(lanes, drafts)
        self.device = device
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

    def admit(self, lane: int, ids: Sequence[int]) -> int:
        """The prompt prefilled into ``lane`` (the lanes session's admission); returns the model's first token.  The lane's
        first pass is the bootstrap pass on ``[pending, placeholders]`` committing nothing (count -1)."""

        if self.runs[lane] is not None:
            raise Qwen38ChatChainError(f"admission into lane {lane}, which still holds a request")
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

    def admit_from_chain(self, lane: int, position: int, pending: int, settled: int, ple_context: Any) -> None:
        """``lane`` takes the request the chain holds in its settled 1-row state (``admit`` without the reset and the prefill):
        ``position`` rows committed, ``pending`` the model's next token, ``settled`` the rows the chain's last pass committed
        (the plugin's next ``accepted_counts``).  The lane's first pass is the bootstrap pass on the plugin's block."""

        if self.runs[lane] is not None:
            raise Qwen38ChatChainError(f"chain move into lane {lane}, which still holds a request")
        if not 0 <= pending < VOCAB_SIZE:
            raise Qwen38ChatChainError(f"token row holds {pending}, outside the vocabulary [0, {VOCAB_SIZE})")
        self._import_chain_state(lane, position, ple_context)
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
        """The request left (finished, preempted): the lane parks at the next pass (it still commits its last real pass)."""

        self.runs[lane] = None
        self.counts[lane] = -1
        self.blocks[lane] = None

    # -- verify ----------------------------------------------------------------------------------------------------

    def verify(self, rows: Sequence[VerifyRow]) -> dict[int, list[int]]:
        """One lane pass for the live rows of the plugin's verify call; returns lane -> its ``K + 1`` argmaxes."""

        k = self.drafts
        live = {row.lane: row for row in rows}
        if not live or len(live) != len(rows):
            raise Qwen38ChatChainError(f"a verify needs distinct live rows, got lanes {[row.lane for row in rows]}")
        orphans = sorted(set(self.holders) - set(live))
        if orphans:
            raise Qwen38ChatChainError(
                f"contract error: lanes {orphans} hold requests but no live row of this verify maps to them "
                f"(live lanes {sorted(live)}); a lane pass advances every active lane"
            )
        strangers = sorted(lane for lane in live if self.runs[lane] is None)
        if strangers:
            raise Qwen38ChatChainError(f"contract error: live rows map to lanes {strangers} with no prefilled request")
        for lane, row in sorted(live.items()):
            self._plan(lane, row)
        if self.writes:
            self.device.write_counts(list(self.counts))
            self.device.set_blocks(self.blocks)
            self.blocks = [None] * self.lanes
            self.writes = False
        set_held = getattr(self.device, "set_held", None)
        if set_held is not None:  # the width pick covers every lane that holds a request
            set_held([1 if run is not None else 0 for run in self.runs])
        self.device.set_active([1 if lane in live else 0 for lane in range(self.lanes)])
        record = self.device.step()
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

    def propose(self, lane: int, committed: Sequence[int], accepted: int) -> tuple[list[int], int]:
        """The drafts the lane's last pass left, with their count; ``([0] * K, 0)`` (decline) with no pass open or when the
        plugin committed something else than the device accepted."""

        k = self.drafts
        run = self.runs[lane]
        if run is None or run.fresh or accepted != run.accepted + 1:
            return [0] * k, 0
        if tuple(int(token) for token in committed[:accepted]) != run.argmaxes[:accepted]:
            return [0] * k, 0
        return list(run.next_tokens[1:]), k


class Qwen38MTPForCausalLM(Qwen38ForCausalLM):
    """The plain adapter over the MTP chain, serving the plugin's speculative contract (``custom_class``): batch 1 over the
    chain's pass loop, batch 2..8 over the lane chain (see the module docstring)."""

    model_capabilities = _MTPModelCapabilities()
    decode_input_update_contract = 0  # decode_forward is overridden and runs no reload command
    _open_lanes = 1  # the lane count the chain opens for (set by initialize_vllm_model before the open)

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
            self.driver: Qwen38VLLMMTPDriver | None = None
            return
        super().__init__(chain, allocated_context=allocated_context, **kwargs)
        mtp = getattr(chain, "mtp", None)
        if mtp is None:
            raise ValueError("the MTP adapter needs a chain opened with mtp=K")
        self.drafts = int(mtp.drafts)
        self.session = Qwen38ChatSession(chain, None, allocated_context=allocated_context)
        self.driver = Qwen38VLLMMTPDriver(chain, self.session, self.drafts)

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
        """The plugin's block-table pool: one context per sequence (the lanes hold their own resident contexts, the paged
        pool of the plain batch mode is not allocated); ``QWEN38_MAX_TOKENS_ALL_USERS`` overrides."""

        override = os.environ.get("QWEN38_MAX_TOKENS_ALL_USERS", "").strip()
        if override:
            if not override.isascii() or not override.isdigit() or int(override) < 1:
                raise ValueError(f"QWEN38_MAX_TOKENS_ALL_USERS must be a positive integer, got {override!r}")
            return int(override)
        return int(max_model_len) * int(max_num_seqs)

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
            accept_modes=("argmax_ids",),
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
                enable_trace=enable_trace,
                prompt_lens=prompt_lens,
                start_pos=start_pos,
                empty_slots=empty_slots,
                greedy_token=greedy_token,
            )
        if tokens.shape[0] != 1:
            raise ValueError(f"one resident slot: tokens must be [1, P], got {tuple(tokens.shape)}")
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
        return self._chain_prefill(ids, greedy_token)[:2]

    def _chain_prefill(self, ids: list[int], greedy_token: bool) -> tuple[torch.Tensor, torch.Tensor, int]:
        """The prompt on the chain's pass-loop driver; the first token as ``prefill_forward`` returns it (and as an int)."""

        try:
            token = self.driver.prefill(ids)
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

    def decode_forward(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: Any,
        spec_mode: str | None = None,
        num_valid_drafts: Any = None,
        accepted_counts: Any = None,
        slot_remap: Any = None,
        **kwargs: Any,
    ):
        """The verify of the plugin's ``[1, 1+K]`` block: ``VerifyOutput(argmax_ids [1, 1+K])``.  Every decode step of a
        speculative launch is a verify (no narrow decode), so a call without ``spec_mode`` is refused."""

        from vllm_tt_plugin.spec_decode import VerifyOutput, check_spec_side_tensors

        if self.lanes_mode:
            return self._decode_lanes(
                tokens=tokens,
                start_pos=start_pos,
                spec_mode=spec_mode,
                num_valid_drafts=num_valid_drafts,
                accepted_counts=accepted_counts,
                slot_remap=slot_remap,
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
        try:
            ids = self.driver.verify(
                tokens[0].tolist(), positions[0].tolist(), int(num_valid_drafts[0]), int(accepted_counts[0])
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
            return self._propose_lanes(num_drafts, committed_tokens, accepted_counts)
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
        if not self.slot.failed:
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
        first = self.driver.prefill(ids)
        block = torch.full((1, k + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        block[0, 0] = first
        positions = torch.full((1, k + 1), -1, dtype=torch.int32)
        positions[0, 0] = len(ids)
        committed = self.driver.verify(block[0].tolist(), positions[0].tolist(), 0, 1)[:1]
        drafts, valid = self.driver.propose(committed, 1)
        if valid == k:  # a second pass on the drafts the first one left
            self.driver.verify([committed[0], *drafts], [len(ids) + 1 + column for column in range(k + 1)], k, 1)
        self.driver.close()
        try:
            self.session.reset()
        except BaseException:
            self.slot.failed = True
            raise
        self.chain.reset_and_seed(SEED_TOKEN_ID)

    # -- batch 2..8: the lane chain --------------------------------------------------------------------------------

    def allocate_kv_cache(self, kv_cache_shape: Any, dtype: Any, num_layers: int) -> None:
        """Batch mode: open the chain with the lanes' allocation as its warm hook (the text demo's form), capture the lane
        traces.  The vLLM cache is not used (the lanes hold their resident contexts).  Batch 1: nothing to do."""

        if not self.lanes_mode:
            return super().allocate_kv_cache(kv_cache_shape, dtype, num_layers)
        if self.lanes_session is not None:
            raise RuntimeError("the lanes are already open: allocate_kv_cache runs once")
        from models.demos.blackhole.qwen38_flash_next.demo.mtp_demo import _lanes_warm_hook
        from models.demos.blackhole.qwen38_flash_next.tools.qwen38_lanes_session import Qwen38LanesSession

        long_chunks, slab_rows = prefill_form()
        lanes_session = Qwen38LanesSession(lanes=self.lanes, drafts=self.drafts, marker=logger.info)

        def warm_hook(chain: Any) -> None:
            _lanes_warm_hook(chain.mesh, lanes_session, self.allocated_context, long_chunks or slab_rows is not None)(
                chain
            )

        logger.info(f"mtp lanes: {self.lanes} lanes x {self.drafts} drafts, context {self.allocated_context}")
        self.chain = self.chain_opener(warm_hook)
        self.session = Qwen38ChatSession(self.chain, None, allocated_context=self.allocated_context)
        self.session.reset()
        lanes_session.capture(self.chain, self.session)
        self.lanes_session = lanes_session
        self.lane_driver = Qwen38VLLMMTPLanesDriver(lanes_session, self.lanes, self.drafts)
        self.driver = Qwen38VLLMMTPDriver(self.chain, self.session, self.drafts)  # the hybrid single's pass loop
        logger.info(f"mtp lanes ready: {lanes_session.summary()}")
        summary = lanes_session.summary()
        logger.info(
            f"mtp lanes widths {summary['buckets']} ({MTP_LANE_BUCKETS_VARIABLE}), per-width DRAM growth per bank "
            f"{summary['dram_growth_bytes_per_bank_by_width']}"
        )
        return None

    def _take_lane(self, slot: int) -> int:
        """The lane ``slot`` admits into: the LOWEST lane that holds no request, so the live lanes stay low and a narrow
        lane width serves them.  ``lane_of_slot`` stays a permutation: the slot that mapped to the chosen lane takes the
        lane ``slot`` gave up (that slot holds no request on the lanes: its lane is free).  Returns the lane."""

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
        enable_trace: bool,
        prompt_lens: Any,
        start_pos: Any,
        empty_slots: Any,
        greedy_token: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Each row alone into the lane of its state slot (the lanes session's admission); the first tokens as int32
        ``[rows, 1]`` for a greedy contract or one-hot float32 logits ``[rows, 1, VOCAB_SIZE]``."""

        driver = self._require_lanes()
        rows = int(tokens.shape[0])
        if tokens.dim() != 2 or not 1 <= rows <= self.lanes:
            raise ValueError(f"batch mode: tokens must be [1..{self.lanes}, P], got {tuple(tokens.shape)}")
        if empty_slots is None or not (len(empty_slots) == len(prompt_lens) == len(start_pos) == rows):
            raise ValueError("batch mode: empty_slots, prompt_lens and start_pos must hold one entry per row")
        slots = [int(slot) for slot in empty_slots]
        if len(set(slots)) != rows or any(not 0 <= slot < self.lanes for slot in slots):
            raise ValueError(f"batch mode: empty_slots must be distinct slots in [0, {self.lanes}), got {slots}")
        prompts = []
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
        self._require_serviceable(enable_trace, recovering=True)
        if self.mtp_hybrid_slot in slots:
            logger.warning(f"prefill into slot {self.mtp_hybrid_slot} whose request still runs on the chain: released")
            self._release_hybrid()
        if self._hybrid_starts(rows):
            try:
                out, rope, first = self._chain_prefill(prompts[0], greedy_token)
            except BaseException:
                self.batch_failed = True
                raise
            self.mtp_hybrid_slot, self.mtp_hybrid_pending = slots[0], first
            return out, rope
        if self.mtp_hybrid_slot is not None:
            self._move_hybrid_to_lane()  # before the admission below resets the chain
        firsts = []
        try:
            for slot, ids in zip(slots, prompts):
                lane = self.lane_of_slot[slot]
                if driver.runs[lane] is not None:
                    logger.warning(f"prefill into slot {slot} whose lane {lane} still holds a request: released")
                    driver.release(lane)
                lane = self._take_lane(slot)
                firsts.append(driver.admit(lane, ids))
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

    def _decode_lanes(
        self,
        *,
        tokens: Any,
        start_pos: Any,
        spec_mode: str | None,
        num_valid_drafts: Any,
        accepted_counts: Any,
        slot_remap: Any,
    ):
        """One lane pass for the live rows of the plugin's ``[lanes, 1+K]`` verify block.  Row ``i`` is live when
        ``start_pos[i, 0] >= 0`` and runs in lane ``lane_of_slot[slot_remap[i]]`` (the map the plain batch adapter keeps:
        ``slot_remap`` is applied to it once, no device move).  Returns ``VerifyOutput(argmax_ids [lanes, 1+K])``, padding
        rows 0."""

        from vllm_tt_plugin.spec_decode import VerifyOutput, check_spec_side_tensors

        if spec_mode is None:
            raise ValueError(
                "ordinary decode reached the MTP adapter: it declares no narrow decode (every step is a verify); "
                "sampled and penalized requests are refused by the plugin at admission"
            )
        if spec_mode != "argmax_ids":
            raise ValueError(f"the MTP adapter answers spec_mode 'argmax_ids', got {spec_mode!r}")
        driver = self._require_lanes()
        k, lanes = self.drafts, self.lanes
        tokens = torch.as_tensor(tokens)
        positions = torch.as_tensor(start_pos)
        if tuple(tokens.shape) != (lanes, k + 1) or tuple(positions.shape) != (lanes, k + 1):
            raise ValueError(
                f"batch mode: the verify block is [{lanes}, {k + 1}], got tokens {tuple(tokens.shape)}, positions "
                f"{tuple(positions.shape)}"
            )
        if num_valid_drafts is None or accepted_counts is None:
            raise ValueError("a verify carries num_valid_drafts and accepted_counts")
        check_spec_side_tensors(num_valid_drafts, accepted_counts, lanes, k)
        remap = parse_slot_remap(slot_remap, lanes)
        new_map = [self.lane_of_slot[remap[row]] for row in range(lanes)]
        live_rows = [row for row in range(lanes) if int(positions[row, 0]) >= 0]
        if not live_rows:
            raise ValueError("batch mode: a verify needs a live row (start_pos >= 0)")
        if self.mtp_hybrid_slot is not None:
            if [remap[row] for row in live_rows] == [self.mtp_hybrid_slot]:
                return self._decode_hybrid(
                    live_rows[0], tokens, positions, num_valid_drafts, accepted_counts, remap, new_map
                )
            self._move_hybrid_to_lane()
        rows = [
            VerifyRow(
                new_map[row],
                tokens[row].tolist(),
                int(positions[row, 0]),
                int(num_valid_drafts[row]),
                int(accepted_counts[row]),
            )
            for row in live_rows
        ]
        try:
            by_lane = driver.verify(rows)
        except BaseException:
            self.batch_failed = True
            raise
        self._pending_note = (
            {remap[row]: row for row in range(lanes) if remap[row] != row} if remap != list(range(lanes)) else None
        )
        self.lane_of_slot = new_map
        out = torch.zeros(lanes, k + 1, dtype=torch.int32)
        for row in live_rows:
            out[row] = torch.tensor(by_lane[new_map[row]], dtype=torch.int32)
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=out, hidden=None)

    def _propose_lanes(self, num_drafts: int, committed_tokens: Any, accepted_counts: Any):
        from vllm_tt_plugin.spec_decode import DraftOutput, check_spec_side_tensors

        driver = self._require_lanes()
        k, lanes = self.drafts, self.lanes
        if num_drafts != k:
            raise ValueError(f"the chain drafts {k} tokens per pass, the plugin asked for {num_drafts}")
        committed = torch.as_tensor(committed_tokens)
        if tuple(committed.shape) != (lanes, k + 1):
            raise ValueError(f"batch mode: the committed block is [{lanes}, {k + 1}], got {tuple(committed.shape)}")
        check_spec_side_tensors(torch.zeros(lanes, dtype=torch.int32), accepted_counts, lanes, k, call="propose")
        drafts = torch.zeros(lanes, k, dtype=torch.int32)
        valid = torch.zeros(lanes, dtype=torch.int32)
        for row in range(lanes):  # row ``row`` is state slot ``row`` after the verify's remap
            if self.mtp_hybrid_slot is not None:
                if row == self.mtp_hybrid_slot:
                    row_drafts, count = self.driver.propose(committed[row].tolist(), int(accepted_counts[row]))
                    drafts[row] = torch.tensor(row_drafts, dtype=torch.int32)
                    valid[row] = count
                continue
            row_drafts, count = driver.propose(
                self.lane_of_slot[row], committed[row].tolist(), int(accepted_counts[row])
            )
            drafts[row] = torch.tensor(row_drafts, dtype=torch.int32)
            valid[row] = count
        return DraftOutput(draft_token_ids=drafts, num_valid=valid)

    def _hybrid_starts(self, rows: int) -> bool:
        """A lone prompt with no request on the lanes and none on the chain runs on the chain's pass loop."""

        return (
            self.mtp_hybrid_single and rows == 1 and self.mtp_hybrid_slot is None and not self._require_lanes().holders
        )

    def _decode_hybrid(self, row: int, tokens, positions, num_valid_drafts, accepted_counts, remap, new_map):
        """The only live row is the chain's request: one chain pass (``Qwen38VLLMMTPDriver.verify``); the other rows are
        padding (0), as the lane path pads its idle rows.  The request's slot follows the row (``slot_remap``)."""

        from vllm_tt_plugin.spec_decode import VerifyOutput

        k, lanes = self.drafts, self.lanes
        try:
            ids = self.driver.verify(
                tokens[row].tolist(), positions[row].tolist(), int(num_valid_drafts[row]), int(accepted_counts[row])
            )
        except BaseException:
            self.batch_failed = self.slot.failed = True
            raise
        self._pending_note = (
            {remap[r]: r for r in range(lanes) if remap[r] != r} if remap != list(range(lanes)) else None
        )
        self.lane_of_slot = new_map
        self.mtp_hybrid_slot = row
        out = torch.zeros(lanes, k + 1, dtype=torch.int32)
        out[row] = torch.tensor(ids, dtype=torch.int32)
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=out, hidden=None)

    def _move_hybrid_to_lane(self) -> None:
        """The chain's request into its lane (no reverse move exists): the chain leaves verify mode at the rows its last pass
        committed (the rows ``propose`` answered for), the lane imports that state and starts on the plugin's block
        ``[t', d_1 .. d_K]``.  The chain is free for the next admission, which resets it."""

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
            driver.admit_from_chain(lane, position, pending, rows, single.session.ple_context)
        except BaseException:
            self.batch_failed = self.slot.failed = True
            raise
        logger.info(f"mtp hybrid single: slot {self.mtp_hybrid_slot} moved to lane {lane} at position {position}")
        self.mtp_hybrid_slot = None

    def _release_hybrid(self) -> None:
        """The chain's request is gone.  Its pass loop is left now: an admission's reset needs the 1-row state."""

        self.mtp_hybrid_slot = None
        try:
            self.driver.close()
        except BaseException:
            self.batch_failed = self.slot.failed = True
            raise

    def _release_lane(self, slot: int) -> None:
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < self.lanes:
            raise ValueError(f"slot must be in [0, {self.lanes}), got {slot!r}")
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
        out, _ = self.prefill_forward(
            tokens=prompt,
            page_table=None,
            kv_cache=None,
            enable_trace=True,
            prompt_lens=[prompt.shape[1]] * len(slots),
            start_pos=[0] * len(slots),
            empty_slots=slots,
        )
        length = int(prompt.shape[1])
        firsts = [int(out[i, 0].argmax()) for i in range(len(slots))]
        block = torch.full((lanes, k + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        positions = torch.full((lanes, k + 1), -1, dtype=torch.int32)
        for row, first in zip(slots, firsts):
            block[row, 0] = first
            positions[row, 0] = length
        valid = torch.zeros(lanes, dtype=torch.int32)
        counts = torch.ones(lanes, dtype=torch.int32)
        verified = self.decode_forward(
            tokens=block, start_pos=positions, spec_mode="argmax_ids", num_valid_drafts=valid, accepted_counts=counts
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
            )
        warm_widths = getattr(self.lanes_session, "warm_widths", None)
        if warm_widths is not None:  # every lane width's traces replay once here (the two warm lanes are still live)
            logger.info(f"mtp lanes: warm replays at widths {warm_widths()}")
        for slot in slots:
            self.release_request(slot)
        if self.mtp_hybrid_single:
            self._warmup_hybrid()

    def _warmup_hybrid(self) -> None:
        """One lone request through the hybrid path with the serving calls, under the closed miss guard: the prefill on the
        chain, a bootstrap pass and two following passes of the chain's pass loop, the move into lane 0, one lane pass on the
        plugin's block, the release.  Leaves the adapter as a fresh server."""

        k, lanes = self.drafts, self.lanes
        ids = list(WARMUP_PROMPT_TOKEN_IDS)
        length = len(ids)
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
            )

        out, _ = self.prefill_forward(
            tokens=torch.tensor([ids], dtype=torch.int32),
            page_table=None,
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
