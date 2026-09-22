# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Request and completion state for Gemma's model-owned dFlash adapter.

Ordinary decode keeps its native device output and readback events. The adapter
retains the committed token prefix so a request can rebuild its drafter context
after ordinary batched decoding. Reconstruction runs only after every earlier
ordinary completion has reached its proposal callback.
"""

import os
from collections import deque
from dataclasses import dataclass

import torch


@dataclass(eq=False)
class ContractRequest:
    identity: int
    slot: int
    tokens: list[int]
    live: bool = True


@dataclass(eq=False)
class ContractStep:
    owners: tuple[ContractRequest | None, ...]
    page_table: torch.Tensor | None
    page_tables_per_layer: object
    kv_cache: object
    verified: ContractRequest | None = None
    consumed: bool = False


@dataclass
class ContractDecodeOutput:
    payload: object
    step: ContractStep
    on_host: bool = False


@dataclass
class ContractProposal:
    owner: ContractRequest
    position: int
    anchor: int
    drafts: list[int]
    posterior: list[int]


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    return value


def _row_tables(value, row):
    if isinstance(value, torch.Tensor):
        return value[row : row + 1] if value.ndim > 1 else value.unsqueeze(0)
    if isinstance(value, list):
        return [_row_tables(item, row) for item in value]
    if isinstance(value, tuple):
        return tuple(_row_tables(item, row) for item in value)
    if isinstance(value, dict):
        return {key: _row_tables(item, row) for key, item in value.items()}
    return value


class DFlashContractMixin:
    """Completion ownership shared by ordinary and speculative decode.

    The target hooks call Gemma4ForCausalLM directly, bypassing the block-output
    adapter. A ContractRequest object is a request generation: release invalidates
    that object even if a later request reuses its state slot and physical pages.
    """

    def _contract_init(self):
        self._ct_requests = {}
        self._ct_ordinary = deque()
        self._ct_proposal = None
        self._ct_decoder_owner = None
        self._ct_submitted_moves = None
        self._ct_force_reload = False
        self._ct_warmup_depth = 0

    def warmup_model_prefill(self, *args, **kwargs):
        self._ct_warmup_depth += 1
        try:
            return super().warmup_model_prefill(*args, **kwargs)
        finally:
            self._ct_warmup_depth -= 1

    def warmup_model_decode(self, *args, **kwargs):
        self._ct_warmup_depth += 1
        try:
            return super().warmup_model_decode(*args, **kwargs)
        finally:
            self._ct_warmup_depth -= 1

    def _contract_disarm(self):
        self.model[0].dflash_capture_taps(None)
        decoder = self._spec_decoder
        if decoder is not None:
            decoder.restore_model_logits_mode()

    def _contract_prefill_tables(self, page_table, page_tables_per_layer, lengths, kv_cache):
        from models.demos.gemma4.tt.attention.operations import effective_block_size

        if page_table is None and page_tables_per_layer is None:
            return None, None
        per_layer = self._build_per_layer_page_tables(page_tables_per_layer, page_table)
        kv_layers = kv_cache
        if isinstance(kv_layers[0][0], (list, tuple)):
            kv_layers = kv_layers[0]
        target = self.model[0]
        tp = target.mesh_config.tp or 1

        def sanitize(table, layer_index):
            if table is None:
                return None
            clean = table.clone()
            attention = target.layers[layer_index].self_attn
            config = attention.config
            # Ring columns name fixed physical slots, not absolute positions.
            if getattr(config, "cache_position_modulo", None) is not None:
                return clean
            local_heads = 1 if attention.weights.kv_replicated else config.num_key_value_heads // tp
            block_size = int(effective_block_size(kv_layers[layer_index][0], config.head_dim, local_heads))
            if clean.ndim != 2 or clean.shape[0] < len(lengths):
                raise ValueError("Gemma4 dFlash prefill page-table rows do not match prompt lengths")
            for row, length in enumerate(lengths):
                # Traced prefill writes its padded bucket; stale tail columns
                # can otherwise alias and overwrite this request's live KV.
                clean[row, (int(length) + block_size - 1) // block_size :] = 0
            clean[len(lengths) :] = 0
            return clean

        return sanitize(page_table, 0), [sanitize(table, index) for index, table in enumerate(per_layer)]

    def prefill_forward(self, *args, page_tables_per_layer=None, **kwargs):
        tokens = kwargs.get("tokens", args[0] if args else None)
        lengths = kwargs.get("prompt_lens", [tokens.shape[1]] * tokens.shape[0])
        if not self._ct_warmup_depth and not kwargs.get("warmup_prefill"):
            kwargs["page_table"], page_tables_per_layer = self._contract_prefill_tables(
                kwargs.get("page_table"), page_tables_per_layer, lengths, kwargs.get("kv_cache")
            )
        self._contract_disarm()
        out = self._contract_target_prefill(*args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        if self._ct_warmup_depth or kwargs.get("warmup_prefill"):
            return out
        slots = kwargs.get("empty_slots", range(tokens.shape[0]))
        tables = kwargs.get("page_table")
        if tables is None:
            return out
        for row, (length, slot) in enumerate(zip(lengths, slots)):
            identity = self._spec_pt_identity(tables[row : row + 1])
            old = self._ct_requests.get(identity)
            if old is not None:
                old.live = False
            # A resumed prefill supplies the complete prefix and supersedes any
            # completion captured before the preemption.
            self._ct_requests[identity] = ContractRequest(identity, int(slot), tokens[row, : int(length)].tolist())
            if self._ct_proposal is not None and self._ct_proposal.owner is old:
                self._ct_proposal = None
        return out

    def note_state_slots_moved(self, moves):
        submitted = self._ct_submitted_moves
        if submitted is not None and all(submitted.get(int(old), int(old)) == int(new) for old, new in moves.items()):
            self._ct_submitted_moves = None
            return
        self._contract_move_owners(moves)

    def _contract_move_owners(self, moves):
        for owner in self._ct_requests.values():
            if owner.slot in moves:
                owner.slot = int(moves[owner.slot])
        if self._ct_decoder_owner is not None:
            self._spec_owner_slot = self._ct_decoder_owner.slot

    def release_request(self, row):
        for identity, owner in list(self._ct_requests.items()):
            if row is not None and owner.slot != int(row):
                continue
            owner.live = False
            del self._ct_requests[identity]
            if self._ct_proposal is not None and self._ct_proposal.owner is owner:
                self._ct_proposal = None
            if self._ct_decoder_owner is owner:
                self._ct_decoder_owner = None
                self._spec_active = False
                self._spec_active_owner = None
                self._spec_pending = None
                self._spec_pending_owner = None
                self._spec_owner_slot = None
                self._spec_last_pt = None

    def _contract_step(self, tokens, positions, page_table, page_tables_per_layer, kv_cache):
        rows = tokens.shape[0]
        positions = positions.reshape(rows, -1)[:, 0]
        owners = []
        for row in range(rows):
            identity = self._spec_pt_identity(page_table[row : row + 1]) if page_table is not None else None
            owner = self._ct_requests.get(identity) if int(positions[row]) >= 0 else None
            owners.append(owner)
            # The first decode input supplies the token sampled by prefill.
            # Async-ahead input can be stale; it must never replace committed
            # history or fill a gap before an older completion supplies it.
            if owner is not None and int(positions[row]) == len(owner.tokens):
                owner.tokens.append(int(tokens.reshape(rows, -1)[row, 0]))
        step = ContractStep(tuple(owners), _snapshot(page_table), _snapshot(page_tables_per_layer), kv_cache)
        return step

    def decode_forward(self, *args, page_tables_per_layer=None, **kwargs):
        from vllm_tt_plugin.spec_decode import PLACEHOLDER_TOKEN_ID, VerifyOutput

        spec_mode = kwargs.pop("spec_mode", None)
        num_valid = kwargs.pop("num_valid_drafts", None)
        kwargs.pop("accepted_counts", None)
        kwargs.pop("draft_token_ids", None)
        tokens = kwargs.get("tokens", args[0] if args else None)
        positions = kwargs.get("start_pos", args[1] if len(args) > 1 else None)
        if positions is None or self._ct_warmup_depth or not self._ct_requests:
            return self._contract_target_decode(*args, page_tables_per_layer=page_tables_per_layer, **kwargs)
        rows = int(tokens.shape[0])
        step = self._contract_step(
            tokens,
            positions,
            kwargs.get("page_table"),
            page_tables_per_layer,
            kwargs.get("kv_cache"),
        )
        # Row identity is settled by the accepted submission. Off-batch owners
        # follow the simultaneous permutation as well.
        remap = kwargs.get("slot_remap")
        self._ct_submitted_moves = None
        if remap is not None:
            moves = {int(old): new for new, old in enumerate(remap) if int(old) >= 0}
            self._contract_move_owners(moves)
            # Newer runners acknowledge the same accepted gather through the
            # lifecycle hook. Older runners send only slot_remap.
            self._ct_submitted_moves = moves
        for row, owner in enumerate(step.owners):
            if owner is not None:
                owner.slot = row
        if spec_mode is None:
            self._contract_disarm()
            if self._ct_force_reload:
                # Mixed verification disables the speculative row in the plain
                # trace. Its next ordinary step must load the committed anchor.
                kwargs["reset_batch"] = True
                self._slots_prefilled_since_decode = {row for row, owner in enumerate(step.owners) if owner is not None}
            payload = self._contract_target_decode(*args, page_tables_per_layer=page_tables_per_layer, **kwargs)
            self._ct_force_reload = False
            self._ct_ordinary.append(step)
            return ContractDecodeOutput(payload, step, self._contract_is_host(payload))

        valid = num_valid.reshape(-1) if num_valid is not None else torch.zeros(rows, dtype=torch.int32)
        proposal = self._ct_proposal
        verified_row = None
        for row in range(rows):
            count = int(valid[row])
            if not count:
                continue
            owner = step.owners[row]
            if proposal is None or owner is not proposal.owner or not owner.live:
                raise RuntimeError("Gemma4 dFlash holds no device posterior for this request")
            anchor = int(tokens[row, 0])
            position = int(positions.reshape(rows, -1)[row, 0])
            if (position, anchor) != (proposal.position, proposal.anchor) or tokens[
                row, 1 : count + 1
            ].tolist() != proposal.drafts[:count]:
                raise ValueError("Gemma4 dFlash received candidate tokens or positions it did not propose")
            if count > len(proposal.drafts):
                raise ValueError("Gemma4 dFlash received more drafts than it proposed")
            verified_row = row
            step.verified = owner

        ids = torch.full((rows, self._SPEC_CONTRACT_K + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32)
        plain_positions = positions.reshape(rows, -1)[:, 0].clone()
        if verified_row is not None:
            ids[verified_row, : len(proposal.posterior)] = torch.tensor(proposal.posterior, dtype=torch.int32)
            plain_positions[verified_row] = -1
        if bool((plain_positions >= 0).any()):
            self._contract_disarm()
            plain_kwargs = dict(kwargs, tokens=tokens.reshape(rows, -1)[:, 0].clone(), start_pos=plain_positions)
            plain_kwargs["reset_batch"] = True
            plain_kwargs["read_from_device"] = False
            payload = self._contract_target_decode(page_tables_per_layer=page_tables_per_layer, **plain_kwargs)
            self._ct_force_reload = True
            host = self._contract_materialize(payload, is_tokens=kwargs.get("sampling_params") is not None)
            host = host[0] if isinstance(host, tuple) else host
            sampled = (
                host.reshape(-1)
                if kwargs.get("sampling_params") is not None
                else host.reshape(-1, host.shape[-1]).argmax(-1)
            )
            for row in range(rows):
                if int(plain_positions[row]) >= 0:
                    ids[row, 0] = int(sampled[row])
        return VerifyOutput(spec_mode="argmax_ids", argmax_ids=ids, hidden=step)

    @staticmethod
    def _contract_is_host(payload):
        return isinstance(payload, torch.Tensor) or (
            isinstance(payload, tuple) and all(item is None or isinstance(item, torch.Tensor) for item in payload)
        )

    def _contract_materialize(self, payload, is_tokens):
        if self._contract_is_host(payload):
            return payload
        host = self._contract_target_read(payload, async_read=False)
        return self._contract_target_process(host, is_tokens=is_tokens)

    def read_decode_output(self, output, async_read=False, *args, **kwargs):
        if not isinstance(output, ContractDecodeOutput):
            if self._contract_is_host(output):
                return (output, []) if async_read else output
            return self._contract_target_read(output, async_read=async_read)
        if output.on_host:
            return (output, []) if async_read else output
        if async_read:
            payload, events = self._contract_target_read(output.payload, async_read=True)
            return ContractDecodeOutput(payload, output.step, True), events
        payload = self._contract_target_read(output.payload, async_read=False)
        return ContractDecodeOutput(payload, output.step, True)

    def process_decode_output_host(self, output, is_tokens=False):
        if not isinstance(output, ContractDecodeOutput):
            return self._contract_target_process(output, is_tokens=is_tokens)
        if self._contract_is_host(output.payload):
            return output.payload
        if output.on_host:
            return self._contract_target_process(output.payload, is_tokens=is_tokens)
        return self._contract_materialize(output.payload, is_tokens)

    @staticmethod
    def _contract_no_drafts(rows, k):
        from vllm_tt_plugin.spec_decode import DraftOutput

        return DraftOutput(torch.zeros((rows, k), dtype=torch.int32), num_valid=torch.zeros(rows, dtype=torch.int32))

    def _contract_pending_widths(self):
        decoder = self._spec_decoder
        if not self._spec_width_set or decoder is None:
            return ()
        if any(record["trace"] is not None for record in decoder._pv_widths.values()):
            return ()
        return getattr(decoder, "_prepared_widths", ())

    def _contract_covers_position(self, owner, position):
        decoder = self._spec_decoder
        physical_width = decoder.P_v if decoder is not None else self._SPEC_N
        max_position = int(getattr(self.model_args[0], "max_seq_len", 0))
        if max_position and position + physical_width > max_position:
            return False
        if decoder is None:
            return True
        if self._spec_width_set:
            return decoder.width_for(position) is not None or any(
                width >= position + physical_width + 64 for width in self._contract_pending_widths()
            )
        # A new request gets its own capture horizon. Retaining the exhausted
        # owner's decoder prevents ordinary completions from restarting it.
        if self._ct_decoder_owner is owner:
            return position < self._spec_budget_end and (
                not decoder.use_packed or position + physical_width <= decoder.pv_sk
            )
        return True

    def propose_draft_tokens(self, num_drafts, committed, positions, counts, hidden=None):
        from vllm_tt_plugin.spec_decode import DraftOutput

        rows, k = int(committed.shape[0]), int(num_drafts)
        step = (
            hidden if isinstance(hidden, ContractStep) else (self._ct_ordinary.popleft() if self._ct_ordinary else None)
        )
        if step is None:
            return self._contract_no_drafts(rows, k)
        if step.consumed:
            raise RuntimeError("Gemma4 dFlash completion was proposed twice")
        step.consumed = True
        if len(step.owners) != rows:
            raise RuntimeError("Gemma4 dFlash completion row count changed")
        live_rows = []
        for row, owner in enumerate(step.owners):
            if owner is None or not owner.live:
                continue
            n = int(counts[row])
            if n < 1 or int(positions[row, 0]) < 0:
                continue
            start = int(positions[row, 0])
            if start > len(owner.tokens):
                raise RuntimeError("Gemma4 dFlash committed prefix has a gap")
            expected = list(range(start, start + n))
            if positions[row, :n].tolist() != expected:
                raise RuntimeError("Gemma4 dFlash committed positions are not consecutive")
            owner.tokens[start:] = committed[row, :n].tolist()
            live_rows.append(row)
        if step.verified is not None and self._ct_proposal is not None and self._ct_proposal.owner is step.verified:
            self._ct_proposal = None
        # The plugin drains initial and changed-layout ordinary submissions.
        # An older steady completion must not rebuild KV over a newer decode.
        if self._ct_ordinary or len(live_rows) != 1:
            return self._contract_no_drafts(rows, k)
        row = live_rows[0]
        owner = step.owners[row]
        if self._ct_proposal is not None and self._ct_proposal.owner is not owner:
            return self._contract_no_drafts(rows, k)
        ceiling = int(os.environ.get("GEMMA4_DFLASH_MAX_SPEC_ISL", "0"))
        if ceiling and len(owner.tokens) - 1 > ceiling:
            return self._contract_no_drafts(rows, k)
        position = int(positions[row, int(counts[row]) - 1])
        anchor = int(committed[row, int(counts[row]) - 1])
        if not self._contract_covers_position(owner, position):
            return self._contract_no_drafts(rows, k)
        if step.verified is owner and self._ct_decoder_owner is owner and self._spec_active:
            self._contract_refresh(step, row)
            self._spec_decoder.contract_commit(int(counts[row]), anchor)
        else:
            self._contract_rebuild(owner, step, row, position, anchor)
        decoder = self._spec_decoder
        self._contract_refresh(step, row)
        if self._spec_width_set:
            decoder.select_width(decoder.start)
        drafts, posterior = decoder.contract_replay(first=self._spec_first_step)
        self._spec_first_step = False
        drafts = [int(token) for token in drafts[:k]]
        self._ct_proposal = ContractProposal(
            owner, position, anchor, drafts, [int(token) for token in posterior[: k + 1]]
        )
        output = torch.zeros((rows, k), dtype=torch.int32)
        output[row, : len(drafts)] = torch.tensor(drafts, dtype=torch.int32)
        valid = torch.zeros(rows, dtype=torch.int32)
        valid[row] = len(drafts)
        return DraftOutput(output, num_valid=valid)

    def _contract_refresh(self, step, row):
        tables = _row_tables(step.page_table, row)
        per_layer = self._build_per_layer_page_tables(_row_tables(step.page_tables_per_layer, row), tables)
        per_layer = self._pad_sliding_page_tables_for_bounded(per_layer, step.kv_cache, authoritative=True)
        self.model[0]._active_page_tables_per_layer = per_layer
        self._spec_decoder.refresh_page_tables(tables)

    def _contract_rebuild(self, owner, step, row, position, anchor):
        if len(owner.tokens) != position + 1:
            raise RuntimeError("Gemma4 dFlash anchor does not end the committed prefix")
        self._contract_synchronize()
        self._contract_disarm()
        pending_widths = self._contract_pending_widths()
        if pending_widths:
            # Eager-only ordinary execution leaves prepared widths uncaptured.
            # Capture before reconstructing the request's target KV and taps.
            self._spec_decoder.capture_widths(pending_widths)
        target = self.model[0]
        drafter = self._spec_get_drafter()
        tables = _row_tables(step.page_table, row)
        per_layer = _row_tables(step.page_tables_per_layer, row)
        prefill_table, prefill_per_layer = self._contract_prefill_tables(tables, per_layer, [position], step.kv_cache)
        target.dflash_capture_taps(drafter.target_layer_ids)
        try:
            self._contract_target_prefill(
                tokens=torch.tensor([owner.tokens[:position]], dtype=torch.int32),
                prompt_lens=[position],
                start_pos=[0],
                empty_slots=[owner.slot],
                page_table=prefill_table,
                page_tables_per_layer=prefill_per_layer,
                kv_cache=step.kv_cache,
                enable_trace=False,
                sampling_params=None,
                warmup_prefill=False,
            )
        finally:
            taps = target.pop_dflash_taps()
            target.dflash_capture_taps(None)
        self._spec_pending = (taps, position)
        self._spec_pending_owner = owner.identity
        self._spec_owner_slot = owner.slot
        self._spec_bootstrap(anchor, position, tables, step.kv_cache, page_tables_per_layer=per_layer)
        self._ct_decoder_owner = owner
        self._ct_force_reload = True
