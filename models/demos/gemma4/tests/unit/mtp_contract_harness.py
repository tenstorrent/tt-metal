# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stubs and step helpers for the Gemma4 MTP (assistant drafter) contract host tests.

The pattern is ``dflash_contract_harness``: the production
``Gemma4MTPContractForCausalLM`` runs over a recording target stub and a
scripted batched decoder that implements the ``spec_decode.py`` serving API
of the P1 interface (``serving_prepare`` / ``serving_capture`` /
``serving_reseed`` / ``serving_step`` / ``serving_refresh_page_tables`` /
``serving_release``) on the host. What a replay computes on the device is not
something a host test can attest; what the adapter sends it and how it
answers the runner from the result is.
"""

from __future__ import annotations

import dataclasses
import sys
from types import SimpleNamespace

import torch

from models.demos.gemma4.tests.unit.conftest import install_base_stubs
from models.demos.gemma4.tests.unit.dflash_contract_harness import DeviceResult, _sampling, _table, _tensor
from models.demos.gemma4.tests.unit.dflash_sampling_harness import attach_sampling_state

#: Host stand-in for the 262144-wide vocabulary.
VOCAB = 600

#: The env the MTP contract reads at call time; cleared for every model so the shell cannot leak in.
SPEC_ENV = (
    "GEMMA4_SPEC_K_BY_ROWS",
    "GEMMA4_SPEC_MAX_ROWS",
    "GEMMA4_SPEC_MAX_ISL",
    "GEMMA4_SPEC_WIDTH_RUNGS",
    "GEMMA4_ASSISTANT_MODEL",
    "HF_MODEL",
)

DEFAULT_SHAPES = ((1, 5), (5, 5), (8, 3), (16, 1))


class Target:
    """``model[0]``: layers with attention configs, the output converter and the vocabulary."""

    def __init__(self, events, ring=None, window=1024):
        self.events = events
        self.layers = [
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    config=SimpleNamespace(
                        cache_position_modulo=ring if kind == "sliding" else None, sliding_window=window
                    )
                )
            )
            for kind in ("sliding", "full")
        ]
        self.vocab_size = VOCAB
        self.mesh_config = SimpleNamespace(tp=8)
        self.ccl_manager = SimpleNamespace(name="ccl")
        self.convert_calls = []

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        self.convert_calls.append((int(B), bool(is_tokens)))
        value = tt_out.value if isinstance(tt_out, DeviceResult) else tt_out
        if is_tokens:
            return value.reshape(-1)[:B]
        return value.reshape(-1, value.shape[-1])[:B].reshape(B, S, -1)


@dataclasses.dataclass
class StepResult:
    """Host stand-in for ``spec_decode.ServingStepResult``.

    ``logits`` is the decoder's ONE verify-logits buffer, shared by every
    record and rewritten by every ``serving_step`` and ``serving_reseed``;
    ``logits_to_host`` reads whatever that buffer holds NOW (its live rows),
    as the device readback would, and counts the readbacks.
    """

    shape: tuple
    width: int
    live: int
    drafts: torch.Tensor
    argmax_ids: torch.Tensor
    logits: object
    decoder: object = dataclasses.field(default=None, repr=False)
    readbacks: int = 0

    def logits_to_host(self):
        self.readbacks += 1
        self.decoder.readbacks += 1
        return self.decoder.buffer[: self.live]


def _ladder(max_seq_len, k, bucket=1024):
    """``mtp_pv_width_ladder`` for the stubbed decoder: doubling rungs up to the context."""

    def _r(n):
        return ((int(n) + bucket - 1) // bucket) * bucket

    tail = int(k) + 2
    ladder, rung = [], _r(tail)
    largest = _r(int(max_seq_len) + tail)
    while rung < largest:
        ladder.append(rung)
        rung = _r(rung * 2)
    ladder.append(largest)
    return ladder


class Decoder:
    """Scripted batched decoder: records every serving call, answers ``serving_step`` per row.

    ``script`` holds, per upcoming replay, a list of ``(drafts, argmax_ids)``
    pairs, one per live row; an empty script yields deterministic values
    (``100 * replay + 10 * row + j``). The logits of each replay put the
    argmax id at +1.0 and everything else at -1.0.
    """

    def __init__(self, events):
        self.events = events
        self.shapes = None
        self.widths = None
        self.max_rows = None
        self.captured = False
        self.released = False
        self.script = []
        self.replays = 0
        self.results = []
        self.buffer = None  # the one verify-logits buffer; rewritten by every step and reseed
        self.readbacks = 0
        self.reseed_tables = []  # the [B_v, blocks] table of every serving_reseed call

    # -- warmup --------------------------------------------------------------
    def serving_prepare(self, shapes, widths_by_shape, max_rows, max_context=None, max_rungs=None):
        if self.shapes is not None:
            raise RuntimeError("serving_prepare called twice")
        self.shapes = [tuple(s) for s in shapes]
        if widths_by_shape is None:
            # The decoder resolves its budgeted default ladder; the stub keeps its doubling
            # rungs and applies max_rungs the same way (the largest rungs survive).
            if max_context is None:
                raise ValueError("serving_prepare without widths needs max_context")
            widths_by_shape = {}
            for s_ in self.shapes:
                rungs = _ladder(int(max_context), int(s_[1]))
                if max_rungs is not None and len(rungs) > int(max_rungs):
                    rungs = rungs[-int(max_rungs) :]
                widths_by_shape[s_] = rungs
        self.widths = {tuple(s): list(w) for s, w in widths_by_shape.items()}
        self.max_rows = int(max_rows)
        self.max_context = max_context
        self.max_rungs = max_rungs
        self.events.append(("prepare", list(self.shapes), self.max_rows))

    def serving_widths(self):
        if self.shapes is None:
            raise RuntimeError("serving_widths before serving_prepare")
        return {shape: list(ws) for shape, ws in self.widths.items()}

    def serving_capture(self):
        if self.shapes is None:
            raise RuntimeError("serving_capture before serving_prepare")
        if self.captured:
            raise RuntimeError("serving_capture called twice")
        self.captured = True
        self.events.append(("capture", list(self.shapes)))

    def serving_release(self):
        self.released = True
        self.events.append(("release",))

    # -- serving -------------------------------------------------------------
    def _check(self, shape):
        if not self.captured:
            raise RuntimeError("serving call before serving_capture")
        if tuple(shape) not in self.shapes:
            raise ValueError(f"record {shape} was never prepared ({self.shapes})")

    def serving_refresh_page_tables(self, page_tables_per_layer):
        self.events.append(("refresh", [t.tolist() for t in page_tables_per_layer]))

    def _check_tables(self, shape, rows, page_tables):
        """spec_decode's ``_srv_upload_tables`` / ``_srv_check_rows`` contract on one call's rows and table."""
        rows = [int(r) for r in rows]
        b_v = int(shape[0])
        if not 1 <= len(rows) <= b_v:
            raise ValueError(f"shape {shape}: {len(rows)} live rows, expected 1..{b_v}")
        if len(set(rows)) != len(rows) or min(rows) < 0:
            raise ValueError(f"shape {shape}: bad rows {rows}")
        if tuple(page_tables.shape[:1]) != (b_v,):
            raise ValueError(f"shape {shape}: page_tables must be [{b_v}, blocks], got {tuple(page_tables.shape)}")
        if len(rows) < b_v and bool(page_tables[len(rows) :].any()):
            raise ValueError(
                f"shape {shape}: padding rows {len(rows)}..{b_v - 1} of page_tables must be zero (block 0)"
            )

    def serving_reseed(self, shape, rows, anchors, positions, page_tables):
        self._check(shape)
        self._check_tables(shape, rows, page_tables)
        self.reseed_tables.append(page_tables.tolist())
        self.events.append(("reseed", tuple(shape), list(rows), list(anchors), list(positions)))
        if self.buffer is not None:
            self.buffer.fill_(-1.0)  # the seed forward writes through the shared buffer

    def width_for(self, shape, positions):
        need = max(positions) + int(shape[1]) + 2
        fits = [w for w in self.widths[tuple(shape)] if w >= need]
        return min(fits) if fits else None

    def serving_step(self, shape, rows, anchors, positions, page_tables, seed_cols):
        self._check(shape)
        shape = tuple(shape)
        rows, anchors, positions, seed_cols = list(rows), list(anchors), list(positions), list(seed_cols)
        if not (len(rows) == len(anchors) == len(positions) == len(seed_cols)):
            raise ValueError("serving_step row lists disagree")
        self._check_tables(shape, rows, page_tables)
        width = self.width_for(shape, positions)
        if width is None:
            raise ValueError(f"no rung of {shape} covers positions {positions}")
        self.replays += 1
        self.events.append(("step", shape, rows, anchors, positions, seed_cols, width, page_tables.tolist()))
        b_v, k_v = shape
        p_v = k_v + 1
        live = len(rows)
        scripted = self.script.pop(0) if self.script else None
        drafts = torch.zeros((live, k_v), dtype=torch.int64)
        argmax = torch.zeros((live, p_v), dtype=torch.int64)
        for i in range(live):
            if scripted is not None:
                row_drafts, row_argmax = scripted[i]
            else:
                base = 100 * self.replays + 10 * i
                row_drafts = [base + j for j in range(1, k_v + 1)]
                row_argmax = [base + 50 + j for j in range(p_v)]
            drafts[i] = torch.tensor(row_drafts[:k_v], dtype=torch.int64)
            argmax[i] = torch.tensor(row_argmax[:p_v], dtype=torch.int64)
        self.buffer = torch.full((b_v, p_v, VOCAB), -1.0)
        for i in range(live):
            for j in range(p_v):
                self.buffer[i, j, int(argmax[i, j]) % VOCAB] = 1.0
        result = StepResult(
            shape=shape, width=width, live=live, drafts=drafts, argmax_ids=argmax, logits=self.buffer, decoder=self
        )
        self.results.append(result)
        return result


def install_decoder(model, shapes=DEFAULT_SHAPES, max_rows=None, widths=None):
    """Give ``model`` a prepared and captured decoder without running warmup."""
    shapes = tuple(tuple(s) for s in shapes)
    max_seq_len = int(model.model_args[0].max_seq_len)
    widths = widths or {s: _ladder(max_seq_len, s[1]) for s in shapes}
    decoder = Decoder(model.events)
    decoder.serving_prepare(list(shapes), widths, max_rows or max(s[0] for s in shapes))
    decoder.serving_capture()
    model._mtp_decoder = decoder
    model._mtp_shapes = shapes
    model._mtp_widths = {s: list(w) for s, w in widths.items()}
    model._mtp_max_rows = decoder.max_rows
    model._mtp_captured = True
    return decoder


def build_mtp_model(
    adapter, monkeypatch, *, ring=None, max_batch_size=32, max_seq_len=1024, env=None, warm=True, resident=None
):
    """A contract-rail model over recording stubs; ``warm`` installs a captured decoder.

    ``resident`` narrows the instance's ``supports_async_decode`` (the runner's resident-decode
    mode, which ``plan_decode_reload`` reads off the instance) the way the base narrows it for
    PLI models; None keeps the class declaration (GEMMA4_SUPPORTS_ASYNC_DECODE, default on).
    """
    for name in SPEC_ENV:
        monkeypatch.delenv(name, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    cls = adapter.Gemma4MTPContractForCausalLM
    model = cls.__new__(cls)
    if resident is not None:
        model.model_capabilities = {**cls.model_capabilities, "supports_async_decode": bool(resident)}
    model.events = []
    model.model = [Target(model.events, ring=ring)]
    attach_sampling_state(model)
    model.model_args = [SimpleNamespace(max_seq_len=max_seq_len, max_batch_size=max_batch_size)]
    model.kv_cache = [[("k", "v"), ("k", "v")]]
    model.mesh_device = object()
    model._bounded_sliding_kv_cache = ring is not None
    model.results = []
    model.reloads = []
    model.decode_commands = []
    model.decode_kwargs = []
    model._spec_init_state()
    model._mtp_init_state()
    model._build_per_layer_page_tables = lambda per_layer, table: [table, table]
    model._pad_sliding_page_tables_for_bounded = lambda per_layer, kv_cache, authoritative=False: per_layer

    def prefill(self, *args, page_tables_per_layer=None, **kwargs):
        tokens = kwargs["tokens"]
        model.events.append(("prefill", int(tokens.shape[0]), kwargs.get("prompt_lens"), list(kwargs["empty_slots"])))
        return torch.zeros(int(tokens.shape[0]), 1, dtype=torch.int32)

    def decode(
        self,
        *args,
        page_tables_per_layer=None,
        reload_inputs,
        reload_page_table,
        reload_sampling_params,
        reset_sampling_state,
        **kwargs,
    ):
        model.events.append(("decode", kwargs["tokens"].reshape(-1).tolist(), kwargs["start_pos"].reshape(-1).tolist()))
        model.reloads.append(reload_inputs)
        model.decode_commands.append(
            dict(
                reload_inputs=reload_inputs,
                reload_page_table=reload_page_table,
                reload_sampling_params=reload_sampling_params,
                reset_sampling_state=reset_sampling_state,
            )
        )
        model.decode_kwargs.append(sorted(kwargs))
        sampling_params = kwargs.get("sampling_params")
        if sampling_params is not None:
            active = model._prepare_decode_sampling_state(
                sampling_params,
                start_pos=torch.chunk(kwargs["start_pos"], model.data_parallel, 0),
                prompt_tokens=kwargs.get("prompt_tokens"),
                output_tokens=kwargs.get("output_tokens"),
                slot_remap=kwargs.get("slot_remap"),
                reload_inputs=reload_inputs,
                reload_sampling_params=reload_sampling_params,
                reset_sampling_state=reset_sampling_state,
            )
            for target, slots in zip(model.model, active):
                target.sampling.seed_manager.get_new_values(slots)
        else:
            model._apply_sampling_slot_remap(kwargs.get("slot_remap"))
        if not model.results:
            raise AssertionError("Supply a plain decode result before submitting the step")
        return model.results.pop(0)

    def read(self, tt_out, async_read=False):
        model.events.append(("read", bool(async_read)))
        return [tt_out]

    def base_warmup(self, *args, **kwargs):
        model.events.append(("base_warmup", kwargs.get("enable_trace")))

    install_base_stubs(adapter, monkeypatch)
    model._harness_prefill_forward = prefill
    model._harness_decode_forward = decode
    model._harness_read_decode_output = read
    model._harness_warmup_model_decode = base_warmup
    if warm:
        install_decoder(model)
    return model


def stub_drafter_modules(monkeypatch, events):
    """Replace ``create_assistant_model`` and ``SpeculativeDecoder`` with recording builders.

    Returns the list the builders append their construction arguments to.
    """
    built = []

    def create_assistant_model(**kwargs):
        built.append(("assistant", kwargs))
        return SimpleNamespace(name="assistant_args"), SimpleNamespace(name="assistant")

    class SpeculativeDecoder(Decoder):
        def __init__(self, **kwargs):
            super().__init__(events)
            built.append(("decoder", kwargs))

    common = sys.modules["models.demos.gemma4.tt.common"]
    monkeypatch.setattr(common, "create_assistant_model", create_assistant_model, raising=False)
    spec_decode = SimpleNamespace(SpeculativeDecoder=SpeculativeDecoder)
    monkeypatch.setitem(sys.modules, "models.demos.gemma4.tt.spec_decode", spec_decode)
    return built


def logits_result(rows, hit):
    """A plain decode's device logits ``[1, 1, rows, VOCAB]``: row r's argmax is ``hit + r``."""
    logits = torch.full((1, 1, rows, VOCAB), -1.0)
    for r in range(rows):
        logits[0, 0, r, hit + r] = 1.0
    return DeviceResult(logits)


def _prefill(model, prompt_len=2, slot=0, key=10, rows=1):
    """One prefill call as the runner makes it for ``rows`` prompts."""
    tokens = torch.full((rows, prompt_len), 7, dtype=torch.int32)
    return model.prefill_forward(
        tokens=tokens,
        prompt_lens=[prompt_len] * rows,
        empty_slots=[slot + r for r in range(rows)],
        page_table=_table([key + 2 * r for r in range(rows)]),
        kv_cache=model.kv_cache,
        warmup_prefill=False,
    )


def _commands(**overrides):
    commands = dict(reload_inputs=True, reload_page_table=False, reload_sampling_params=True, reset_sampling_state=True)
    commands.update(overrides)
    return commands


def _ordinary(model, anchors, positions, keys, result=None, sampling=True, **kwargs):
    """An ordinary decode step: ``[rows, 1]`` tokens, ``[rows]`` positions, -1 pads."""
    if result is not None:
        model.results.append(result)
    if sampling:
        kwargs.setdefault("sampling_params", _sampling(len(anchors)))
    kwargs = dict(_commands(), **kwargs)
    page_table = kwargs.pop("page_table", _table(keys))
    return model.decode_forward(
        tokens=_tensor(anchors).reshape(-1, 1),
        start_pos=_tensor(positions),
        page_table=page_table,
        kv_cache=model.kv_cache,
        **kwargs,
    )


def _propose(model, committed, positions, counts=None, k=5):
    committed = _tensor(committed)
    positions = _tensor(positions)
    if counts is None:
        counts = [1] * int(committed.shape[0])
    return model.propose_draft_tokens(k, committed, positions, _tensor(counts))


def _verify(
    model, blocks, positions, valid, keys, result=None, sampling=None, mode="argmax_ids", counts=None, **kwargs
):
    """A verify step; ``sampling`` defaults to device sampling for argmax_ids and none for logits."""
    if result is not None:
        model.results.append(result)
    if sampling is None:
        sampling = mode == "argmax_ids"
    if sampling:
        kwargs.setdefault("sampling_params", _sampling(len(blocks)))
    kwargs = dict(_commands(), **kwargs)
    page_table = kwargs.pop("page_table", _table(keys))
    return model.decode_forward(
        tokens=_tensor(blocks),
        start_pos=_tensor(positions),
        spec_mode=mode,
        num_valid_drafts=_tensor(valid),
        accepted_counts=_tensor(counts if counts is not None else [1] * len(blocks)),
        page_table=page_table,
        kv_cache=model.kv_cache,
        **kwargs,
    )


def _block(anchor, drafts, width=6):
    """A verify row: the anchor, the drafts, -1 padding to ``width``."""
    row = [int(anchor)] + [int(d) for d in drafts]
    return row + [-1] * (width - len(row))


def _positions(start, count, width=6):
    return list(range(int(start), int(start) + int(count))) + [-1] * (width - int(count))


def _start_solo(model, prompt_len=2, key=10, first=150):
    """Prefill alone, take the first ordinary step (plain decode), then the first proposal."""
    _prefill(model, prompt_len=prompt_len, key=key)
    out = _ordinary(model, [3], [prompt_len], [key], result=_tensor([first]))
    assert out.tolist() == [first]
    proposal = _propose(model, [[first, -1, -1, -1, -1, -1]], [[prompt_len + 1] + [-1] * 5])
    assert proposal.num_valid.tolist() == [5]
    return proposal


def _names(model, kind):
    return [event for event in model.events if event[0] == kind]


def _steps(model):
    return _names(model, "step")


def _wire(model, live_rows, anchors, positions, keys, total, result, **kwargs):
    """An ordinary step over ``total`` wire rows with the given live rows (-1 positions elsewhere)."""
    toks = [0] * total
    poss = [-1] * total
    ks = [0] * total
    for r, a, p, key in zip(live_rows, anchors, positions, keys):
        toks[r], poss[r], ks[r] = a, p, key
    return _ordinary(model, toks, poss, ks, result=result, **kwargs)


def _committed(total, live_rows, anchors, positions):
    """The committed-token and position blocks of a proposal: one anchor per live row, -1 elsewhere."""
    committed = [[-1] * 6 for _ in range(total)]
    posits = [[-1] * 6 for _ in range(total)]
    for r, a, p in zip(live_rows, anchors, positions):
        committed[r] = [a] + [-1] * 5
        posits[r] = [p] + [-1] * 5
    return committed, posits


def _warmup(model, enable_trace):
    model.warmup_model_decode(
        kv_cache=model.kv_cache, enable_trace=enable_trace, max_batch_size=32, num_blocks=8, can_sample_on_device=True
    )


# -- asynchronous scheduling: the deferred verify readback walked on the host ------------------


def _read(model, out):
    """The deferred read of a verify: the mode's tensor, unwrapped from the VerifyOutput as the runner does,
    handed to ``read_decode_output`` and waited on later (the host tensor carries no events)."""
    tensor = out.argmax_ids if out.spec_mode == "argmax_ids" else out.logits
    host, events = model.read_decode_output(tensor, async_read=True)
    assert events == []
    return host


def _greedy_walk(block, valid, ids):
    """The plugin's greedy walk (``accept_greedy_drafts``): every draft the target argmax agrees with, then the
    argmax at the first disagreeing column (the correction, or the bonus past the last draft)."""
    accepted = 0
    while accepted < valid and int(ids[accepted]) == int(block[1 + accepted]):
        accepted += 1
    return [int(ids[j]) for j in range(accepted + 1)], accepted + 1


def _script_two_replays(model):
    """Replay 1 agrees with three drafts, replay 2 with two (every id below the stub vocabulary)."""
    model._mtp_decoder.script = [
        [([101, 102, 103, 104, 105], [101, 102, 103, 477, 478, 479])],
        [([201, 202, 203, 204, 205], [201, 202, 488, 489, 490, 491])],
    ]


def _verify_walk_propose(model, anchor, position, drafts, mode, deferred):
    """One verify of ``drafts`` after ``anchor`` at ``position``, read now (synchronous order) or through the
    deferred read, walked on the host, committed and proposed from."""
    block = _block(anchor, drafts)
    out = _verify(model, [block], [_positions(position, 6)], [5], [10], mode=mode)
    host = _read(model, out) if deferred else (out.argmax_ids if mode == "argmax_ids" else out.logits)
    ids = host[0] if mode == "argmax_ids" else host[0].float().argmax(dim=-1)
    committed, count = _greedy_walk(block, 5, ids)
    proposal = _propose(model, [committed + [-1] * (6 - count)], [_positions(position + 1, count)], counts=[count])
    return committed, count, proposal.draft_token_ids[0].tolist()


__all__ = [
    "DEFAULT_SHAPES",
    "Decoder",
    "DeviceResult",
    "SPEC_ENV",
    "StepResult",
    "Target",
    "VOCAB",
    "_block",
    "_commands",
    "_committed",
    "_greedy_walk",
    "_ladder",
    "_names",
    "_ordinary",
    "_positions",
    "_prefill",
    "_propose",
    "_read",
    "_sampling",
    "_script_two_replays",
    "_start_solo",
    "_steps",
    "_table",
    "_tensor",
    "_verify",
    "_verify_walk_propose",
    "_warmup",
    "_wire",
    "build_mtp_model",
    "install_decoder",
    "logits_result",
    "stub_drafter_modules",
]
