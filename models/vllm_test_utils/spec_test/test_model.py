# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from loguru import logger

from models.vllm_test_utils.no_op_test.test_model import DummyNoOpModel

# Which target this model stands in for, selected by ``TT_SPEC_TARGET``.
#
# ``depth`` accepts the first ``TT_SPEC_ACCEPT_DEPTH`` drafts of every row by
# returning them unchanged, which makes the acceptance rate a knob and is what
# every acceptance-accounting measurement wants.
#
# ``fixed`` chooses by a rule the drafts never enter, which makes the output
# sequence independent of whether anything was drafted at all, and is what an
# end-to-end losslessness comparison needs.
TARGET_DEPTH = "depth"
TARGET_FIXED = "fixed"

# How the drafter decides how much to offer. ``always`` offers the full draft
# length on every step, which is what a fixed-width measurement needs.
# ``solo`` offers it only while one request is live and nothing otherwise,
# which is the adaptive shape a real deployment has: speculation pays for a
# lone request and loses to batching for a full one.
DRAFT_POLICY_ALWAYS = "always"
DRAFT_POLICY_SOLO = "solo"


def _draft_policy():
    """The configured draft policy, read from the environment.

    Read here rather than on the instance because the plugin reads
    ``model_capabilities`` and calls ``spec_plan`` off the class, at
    configuration time, before any instance exists. A launch sets this in the
    server process's environment, so an import-time read sees it.
    """
    return os.environ.get("TT_SPEC_DRAFT_POLICY", DRAFT_POLICY_ALWAYS)


def _declared_spec_requirements():
    """What this model's drafter reads, which the policy decides.

    Under ``always`` the drafter is only ever asked after a verify, so it can
    require the target hidden state and does, to keep the handoff under test.

    Under ``solo`` it is asked after ordinary decode steps too, and those
    return no hidden handle at all. A drafter requiring a fed hidden state
    cannot serve them, and the plugin keeps such a model on the verify path
    for exactly that reason, which would defeat the policy. So this policy's
    drafter reads the committed block and nothing else, and says so.
    """
    if _draft_policy() == DRAFT_POLICY_SOLO:
        return ["device_propose"]
    return ["device_propose", "hidden_feed"]


class DummySpecDecodeModel(DummyNoOpModel):
    """Dummy model implementing the speculative-decoding contract.

    Does no work: it accepts the drafts it is handed and returns arithmetic for
    everything else. That makes it two things at once.

    First, an instrument. Its sibling ``DummyNoOpModel`` exists to measure what
    vLLM costs per step with the model removed; this one measures what vLLM
    costs per *speculative* step. With ``TT_SPEC_ACCEPT_DEPTH=0`` nothing is
    accepted, so each step commits one token and its wall clock is the host
    cost of one whole verify-then-propose iteration: the scheduler, the
    candidate-block build, this model's own arithmetic, the plugin's acceptance
    walk, the commit and the n-gram proposal, with no device work anywhere
    under it. That is what a plugin-driven speculation loop can be compared
    against a model-internal one with.

    It is the cost of the loop and not of vLLM alone, because this model does
    host work of its own on every step: ``_verified_ids`` builds a ``[B, 1+K]``
    answer whatever the accept depth. That work measures about 17 microseconds
    at every shape from ``[1, 6]`` to ``[32, 17]``, against a step whose other
    costs are milliseconds, so it does not move the comparison. Subtract it
    only if the comparison ever turns on a margin that small.

    Second, the only way to exercise the plugin's speculative path end to end.
    Everything between the runner and the engine needs a real server: the
    ``take_draft_token_ids`` handshake, the scheduler's lookahead budget and
    its grammar truncation of drafts, placeholder accounting, and KV cache
    allocation. None of it is reachable from a host test.

    It serves both halves of the contract. ``decode_forward`` verifies a
    candidate block, and ``propose_draft_tokens`` drafts the next one, which is
    what a launch asking for the model's own drafter calls instead of an n-gram
    proposer. The drafter proposes exactly what the verify accepts, so with
    every draft accepted every step after the first commits ``1+K`` tokens. The
    first decode step of a request commits one: the drafter runs after a
    commit, so nothing is in flight yet. From there no step is ever draftless,
    which an n-gram drafter cannot promise, because it stalls whenever the
    generated text stops repeating and makes a fixed committed width impossible
    to measure.

    ``README.md`` beside this file carries the server command, the two modes
    and what the measurement mode measures.

    The contract is
    https://github.com/tenstorrent/vllm-tt-plugin/issues/110 and the model side
    of it is ``docs/SPEC_DECODE_CONTRACT.md`` in that repository. This class
    imports the plugin's value types, which is how a model declares a plan; the
    plugin is therefore required to import this module, as it is to serve it.

    Environment:
        ``TT_SPEC_ACCEPT_DEPTH``  how many of each row's drafts to accept.
            Default ``-1``, meaning all of them, which exercises the loop
            hardest. ``0`` accepts none, which is the measurement mode.
    """

    # The explicit decode-input update contract, version 1: this model honors
    # the four commands independently rather than inferring what to reload
    # from the shapes it was handed. ``supports_async_decode`` below is only
    # readable by the plugin for an adapter that declares this.
    decode_input_update_contract = 1

    model_capabilities = {
        **DummyNoOpModel.model_capabilities,
        # The master gate. Everything below it is read only when a launch
        # carries a speculative_config.
        "supports_spec_decode": True,
        # This model drafts as well as verifies, so it declares both halves of
        # a device drafter. An n-gram launch requires neither and is unaffected:
        # admission checks that the method's requirements are a subset of what
        # the model declares, so declaring more never refuses a launch.
        "spec_requirements": _declared_spec_requirements(),
        # How the target hidden state reaches the drafter. ``roundtrip`` and
        # not ``on_device``, because what this model returns is a host Python
        # object carrying no hidden state at all: it exists so the handoff's
        # one checkable property, that the runner hands back the object the
        # verify produced, can be asserted. A model that keeps real hidden
        # state in device memory declares ``on_device``.
        "spec_hidden_handoff": (["roundtrip"] if _draft_policy() != DRAFT_POLICY_SOLO else []),
        # The deferred speculative path asks two things of a model: that its
        # readback serve a ``[B, 1+K]`` verify whose committed length the host
        # decides after the forward, and that the verify's hidden handle stay
        # valid across that readback until the next step's propose call. This
        # model satisfies both trivially, and the word is exact: it returns
        # host tensors, so the plugin reads no device buffer for it, and its
        # handle is a host Python object with no lifetime to lose. What is not
        # trivial, and is what the two order-independence tests beside this
        # model assert, is that it keeps no state between steps: every answer
        # is computed from the ``tokens``, ``start_pos`` and
        # ``accepted_counts`` of the step being served.
        #
        "supports_async_spec_decode": True,
        # And the ordinary half, which the plugin reads first: absent, it
        # disables asynchronous scheduling for the model whatever else is
        # declared. Backed by ``decode_forward``'s split submission, by
        # ``read_decode_output``, and by the resident forward inputs below.
        "supports_async_decode": True,
        # Left at 1 by omission. Any value above 1 selects the block-output
        # rail, which owns the committed width per step and cannot be combined
        # with speculation.
    }

    def __init__(self, mesh_device, max_batch_size, vocab_size, **kwargs):
        super().__init__(mesh_device, max_batch_size, vocab_size, **kwargs)
        depth = int(os.environ.get("TT_SPEC_ACCEPT_DEPTH", "-1"))
        self.accept_depth = None if depth < 0 else depth
        self.target = os.environ.get("TT_SPEC_TARGET", TARGET_DEPTH)
        if self.target not in (TARGET_DEPTH, TARGET_FIXED):
            raise ValueError(f"TT_SPEC_TARGET must be {TARGET_DEPTH!r} or {TARGET_FIXED!r}, " f"got {self.target!r}")
        self.draft_policy = os.environ.get("TT_SPEC_DRAFT_POLICY", DRAFT_POLICY_ALWAYS)
        if self.draft_policy not in (DRAFT_POLICY_ALWAYS, DRAFT_POLICY_SOLO):
            raise ValueError(
                f"TT_SPEC_DRAFT_POLICY must be {DRAFT_POLICY_ALWAYS!r} or "
                f"{DRAFT_POLICY_SOLO!r}, got {self.draft_policy!r}"
            )
        # Set by every verify and checked by the drafter. None before the first
        # verify, which is also the state a propose arriving before any verify
        # would be caught by.
        self._verify_hidden = None
        # The resident forward inputs, which back
        # ``supports_async_decode``: a step submitted before the previous
        # step's token came back cannot have been handed that token on the
        # host, so the model holds it. None until the first step reloads.
        self._resident_tokens = None
        self._resident_positions = None
        self._resident_page_table = None
        # Counters, so a test can assert the commands arrived and the position
        # advanced once per forward rather than infer either from an output.
        self._position_advances = 0
        self._readbacks = 0
        self._sampling_param_uploads = 0
        self._sampling_state_resets = 0
        logger.info(
            f"DummySpecDecodeModel: target={self.target} accept_depth="
            f"{'all' if self.accept_depth is None else self.accept_depth} "
            f"draft_policy={self.draft_policy}"
        )

    def _fixed_choice(self, tokens, positions):
        """``fixed`` target: the choice that follows each input, by one rule.

        The rule reads the token and its position and nothing else, so what
        this model chooses at a candidate position does not depend on what was
        drafted there. That is what makes an end-to-end losslessness check
        possible: the same prompt run with and without speculation has to emit
        the same sequence, because the sequence is a property of the rule.

        The ``depth`` target cannot answer that question. It returns each draft
        unchanged up to its accept depth, so its output is a function of what
        was drafted, and a speculated run and a plain run of it are meant to
        differ.
        """
        ids = tokens.to(torch.int64) * 31 + positions.to(torch.int64) * 7 + 11
        return (ids % self.vocab_size).to(torch.int32)

    @classmethod
    def get_max_tokens_all_users(cls, **kwargs):
        """The KV budget this model declares, which sizes the block pool.

        A real model computes this from its own device memory. This one has no
        KV cache at all, so the number is free, and the plugin's default of
        131072 tokens is large enough that no reachable number of requests can
        exhaust it. That makes preemption unreachable, and a preemption test
        that cannot reach its subject is worth nothing.

        ``TT_SPEC_MAX_TOKENS_ALL_USERS`` is therefore how a launch asks for a
        pool small enough to force one. ``--num-gpu-blocks-override`` cannot:
        the plugin writes that field itself from this value, so an operator's
        setting is replaced.
        """
        del kwargs  # nothing here scales with the device or the batch
        return int(os.environ.get("TT_SPEC_MAX_TOKENS_ALL_USERS", "131072"))

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """Which ``(max_num_seqs, K)`` points this model can serve.

        Every point whose candidate block fits the context, because this model
        allocates nothing per candidate and holds no state a candidate could
        invalidate, so no draft length costs it more than another.

        It deliberately imposes no ceiling below that, so a measurement runs at
        the K it was asked for rather than at a limit invented here, and
        nothing else imposes one either: ``MAX_SPEC_LEN`` is asserted inside
        vLLM's ``RejectionSampler``, which the TT path never calls, and
        ``SpeculativeConfig`` checks only that the draft length is positive. A
        large K therefore reaches the runner, whose ``[B, 1+K]`` candidate
        block and ``[B, K]`` draft block scale with it. That is the intended
        behaviour for an instrument; a real model answers from its own lane
        arithmetic and L1 budget and returns ``SpecReject`` for a point it
        cannot fit.
        """
        from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, DRAFTER_STATE_INTERNAL, SpecPlan, SpecReject

        del max_num_seqs  # no cost scales with concurrency here
        if requested_k < 1:
            return SpecReject(reason=f"a draft length of {requested_k} speculates nothing")
        # A candidate block of 1+K positions cannot be verified past the
        # context, so this is a real limit rather than an invented one.
        max_model_len = int(vllm_config.model_config.max_model_len)
        if requested_k + 1 > max_model_len:
            return SpecReject(
                reason=(
                    f"a draft length of {requested_k} needs a {requested_k + 1} position "
                    f"candidate block, past max_model_len {max_model_len}"
                ),
                supported_k=(max_model_len - 1,),
            )
        return SpecPlan(
            effective_k=requested_k,
            # One decode row per request: this model has no physical candidate
            # layout, so a request costs it the rows a plain decode costs.
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
            drafter_state=DRAFTER_STATE_INTERNAL,
            # Under the adaptive policy this model is asked to decode steps
            # that verify nothing, which is the whole point of that policy:
            # those steps are ordinary decodes and can overlap. It serves them
            # with the decode its base class already implements, so it grows no
            # input shape for them. Under ``always`` every step carries drafts
            # and the question never arises.
            supports_narrow_decode=_draft_policy() == DRAFT_POLICY_SOLO,
        )

    def decode_forward(self, *args, **kwargs):
        """One decode step, which is a verify when the runner asks for one.

        Verify is not a separate entry point: the runner passes the same
        ``tokens`` and ``start_pos`` a plain decode gets, ``1+K`` wide, plus
        ``num_valid_drafts``, ``accepted_counts`` and ``spec_mode``. Their
        absence is what makes a step an ordinary decode, and then the resident
        path below answers.

        ``read_from_device=False`` submits and returns a handle the runner
        reads later through ``read_decode_output``. There is no device here, so
        the answer is computed now and held; what is split is the interface,
        which is what lets a server run this model with asynchronous
        scheduling at all. Ordering tests belong in the plugin's host suite,
        where a completion event can be held open; a model with no device
        cannot hold one honestly.
        """
        spec_mode = kwargs.get("spec_mode")
        if spec_mode is None:
            return self._plain_decode(*args, **kwargs)

        from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, VerifyOutput, check_spec_side_tensors

        if spec_mode != ACCEPT_MODE_ARGMAX_IDS:
            # Declared in accept_modes, so the runner should never ask; raise by
            # name rather than returning a field the mode does not carry.
            raise NotImplementedError(
                f"DummySpecDecodeModel serves {ACCEPT_MODE_ARGMAX_IDS!r}, " f"asked for {spec_mode!r}"
            )

        tokens = kwargs.get("tokens")
        if tokens is None and args:
            tokens = args[0]
        num_valid_drafts = kwargs["num_valid_drafts"]
        # Both side tensors, against the contract module's own validator rather
        # than a private copy of its rules: this model is the plugin's
        # conformance witness, and a witness that checks a weaker domain than
        # the contract states is worth little. The shared validator is also why
        # this cannot drift from the plugin's own stand-in.
        check_spec_side_tensors(
            num_valid_drafts,
            kwargs["accepted_counts"],
            int(tokens.shape[0]),
            int(tokens.shape[1]) - 1,
        )
        # A fresh handle per verify, of no useful type on purpose. The contract
        # is that the runner carries it back to the drafter without
        # interpreting its dtype, layout or tensor-parallel fracturing, so this
        # model can check the one thing that has to hold, that the object it
        # receives is the object it produced, and nothing else can be checked
        # by a stand-in.
        self._verify_hidden = object()
        positions = kwargs.get("start_pos")
        if positions is None and len(args) > 1:
            positions = args[1]
        if self.target == TARGET_FIXED:
            # Every column at once, and no reference to the drafts or to the
            # accept depth: a draft is accepted exactly when it already equals
            # what this rule would have chosen.
            verified = self._fixed_choice(tokens, positions)
        else:
            verified = self._verified_ids(tokens, num_valid_drafts)
        return VerifyOutput(
            spec_mode=ACCEPT_MODE_ARGMAX_IDS,
            argmax_ids=verified,
            hidden=self._verify_hidden,
        )

    def propose_draft_tokens(
        self,
        num_drafts,
        committed_tokens,
        committed_positions,
        accepted_counts,
        hidden=None,
    ):
        """Draft the next ``num_drafts`` tokens per row, on the host.

        This is the device-drafter half of the contract, and what it exists to
        exercise is the loop rather than any drafting quality: it proposes
        exactly what its own verify accepts, ``last + 1 + j`` from each row's
        last committed token, so with every draft accepted every step that has
        drafts in flight commits ``1+K`` tokens. A request's first decode step
        has none, because this call runs after a commit. An n-gram drafter
        cannot promise even the steps after that, because it stalls whenever
        the text stops repeating, which makes this the only way to measure a
        speculative step's cost at a fixed committed width.

        Which entry of the committed block is a row's last token is
        ``accepted_counts - 1``, the same arithmetic the verify uses to pick a
        candidate state slot. Reading a fixed column instead would continue
        every row from the same place and quietly lose the rows that accepted
        less.
        """
        from vllm_tt_plugin.spec_decode import DraftOutput, check_spec_side_tensors

        rows, width = committed_tokens.shape
        if committed_positions.shape != committed_tokens.shape:
            raise ValueError(
                f"propose committed_tokens {tuple(committed_tokens.shape)} and "
                f"committed_positions {tuple(committed_positions.shape)} must "
                "agree"
            )
        if width != num_drafts + 1:
            raise ValueError(
                f"propose committed_tokens is {width} wide for "
                f"num_drafts {num_drafts}; the committed block is 1+K wide"
            )
        # Against the contract module's own validator rather than a private
        # copy of its rules: a row's count is what selects its last token here,
        # so a count outside [1, 1+K] would read the wrong entry or pad.
        check_spec_side_tensors(
            torch.zeros(rows, dtype=torch.int32),
            accepted_counts,
            rows,
            num_drafts,
            call="propose",
        )
        if self.draft_policy == DRAFT_POLICY_SOLO and hidden is None:
            # A step with nothing to verify returns no hidden handle, and this
            # policy exists to be asked on exactly those steps. It drafts from
            # the committed block alone, which is why it declares no
            # ``hidden_feed`` requirement.
            pass
        elif hidden is not self._verify_hidden:
            raise ValueError(
                "propose_draft_tokens received a hidden handle that is not the "
                "one this model's verify returned, so the runner replaced or "
                "dropped it; a device drafter cannot run against another "
                "step's hidden state"
            )

        index = (accepted_counts.to(torch.int64) - 1).unsqueeze(1)
        last = committed_tokens.to(torch.int64).gather(1, index)
        if self.target == TARGET_FIXED:
            # Walk the rule forward from each row's last committed token, so
            # the drafts are what the verify is going to choose, and bend the
            # ones past the accept depth so that a partial-acceptance run is
            # still reachable in this target. Wrongness lives in the drafter
            # here, which is where a real drafter's wrongness lives.
            position = committed_positions.to(torch.int64).gather(1, index)
            depth = num_drafts if self.accept_depth is None else self.accept_depth
            columns = []
            token, pos = last, position
            for column in range(num_drafts):
                token = self._fixed_choice(token, pos).to(torch.int64)
                pos = pos + 1
                offered = token if column < depth else (token + 1) % self.vocab_size
                columns.append(offered)
            drafts = torch.cat(columns, dim=1)
        else:
            offsets = torch.arange(1, num_drafts + 1, dtype=torch.int64)
            drafts = (last + offsets.unsqueeze(0)) % self.vocab_size
        return DraftOutput(
            draft_token_ids=drafts.to(torch.int32), num_valid=self._num_valid(committed_positions, num_drafts)
        )

    def _num_valid(self, committed_positions, num_drafts):
        """How many drafts each row is offered, under this draft policy.

        ``None`` under ``always``, which means every row is offered the full
        draft length: the runner reads that as "all of it" and a drafter that
        always drafts needs to say nothing.

        Under ``solo`` the offer is the full length while one request is live
        and nothing while more are. Live requests are counted from the
        committed positions, not from the number of rows: the rows are padded
        to the wire batch size, and a padding row's position is negative
        precisely so that it can be told from a request. Counting rows would
        make a one-request batch look like a full one and this policy would
        never offer anything.
        """
        if self.draft_policy == DRAFT_POLICY_ALWAYS:
            return None
        rows = int(committed_positions.shape[0])
        live = int((committed_positions[:, 0] >= 0).sum())
        return torch.full((rows,), num_drafts if live == 1 else 0, dtype=torch.int32)

    def prefill_forward(self, *args, **kwargs):
        """One prefill, following the same rule the verify follows.

        Only under ``fixed``, and it matters most where a prefill is not the
        first thing a request does. A preempted request resumes with a prefill
        that replays its saved history, and a wholesale prefix-cache reset
        makes every running request do that. The base class answers a prefill
        with zero logits whatever the history was, so under that answer a
        resumed request emits a 0 in the middle of its output and the whole
        point of this target is lost: its output is meant to be a property of
        the rule, not of how many times the request was replayed.

        The rule reads the token and its position, and a prefill's are the
        last prompt token of each row and that token's position. ``prompt_lens``
        gives each row's length, which is where the padded ``tokens`` row ends.
        """
        if self.target != TARGET_FIXED:
            return super().prefill_forward(*args, **kwargs)
        tokens = kwargs.get("tokens")
        if tokens is None and args:
            tokens = args[0]
        prompt_lens = kwargs.get("prompt_lens")
        rows = int(tokens.shape[0])
        lengths = (
            torch.as_tensor(prompt_lens, dtype=torch.int64).reshape(-1)
            if prompt_lens is not None
            else torch.full((rows,), int(tokens.shape[1]), dtype=torch.int64)
        )
        last_index = (lengths[:rows] - 1).clamp(min=0)
        last_token = tokens.to(torch.int64).gather(1, last_index.unsqueeze(1))
        choice = self._fixed_choice(last_token, last_index.unsqueeze(1))
        if kwargs.get("sampling_params") is not None:
            return choice.reshape(rows).to(torch.int64)
        logits = torch.zeros(rows, 1, self.vocab_size, dtype=torch.float32)
        logits.scatter_(2, choice.to(torch.int64).unsqueeze(2), 1.0)
        return logits

    def _plain_decode(self, *args, **kwargs):
        """An ordinary decode, from whichever inputs the commands make current.

        The four update commands decide what this step reads. ``reload_inputs``
        copies the host's token, position and page table into the resident
        buffers; ``reload_page_table`` copies the page table alone; neither
        means the host's copies are stale by design and the resident ones are
        what this step decodes from. That is the whole content of
        ``supports_async_decode`` for a model with no device: a step that was
        submitted before the previous step's token came back cannot have been
        given that token on the host, so the model has to hold it.

        The token this step chooses is written back into the resident buffer,
        which is the persistent token feedback the contract asks for, and the
        resident position advances exactly once per forward.
        """
        tokens, positions = self._resident_decode_inputs(*args, **kwargs)
        rows = tokens.shape[0]
        if self.target != TARGET_FIXED:
            # The measurement target: the base class answers with zero logits,
            # and the resident bookkeeping above is what makes the answer
            # available asynchronously.
            answer = super().decode_forward(*args, **kwargs)
            self._advance_resident(self._first_column(answer, rows, kwargs))
            return answer
        answer = self._fixed_plain_answer(tokens, positions, rows, kwargs)
        self._advance_resident(self._first_column(answer, rows, kwargs))
        return answer

    def _resident_decode_inputs(self, *args, **kwargs):
        """Apply this step's update commands and return what it decodes from.

        The commands are commands, not hints: a version-1 adapter must honor
        each independently and refuse a combination it cannot serve rather
        than guessing. ``reload_inputs`` already covers page tables, so the
        legal forward-input modes are everything, page tables only, and
        nothing.
        """
        tokens = kwargs.get("tokens")
        if tokens is None and args:
            tokens = args[0]
        positions = kwargs.get("start_pos")
        if positions is None and len(args) > 1:
            positions = args[1]
        reload_inputs = bool(kwargs.get("reload_inputs", True))
        reload_page_table = bool(kwargs.get("reload_page_table", False))
        reset_sampling_state = bool(kwargs.get("reset_sampling_state", False))
        if reload_inputs and reload_page_table:
            raise ValueError(
                "DummySpecDecodeModel was told to reload every forward input "
                "and, separately, only the page table; reload_inputs already "
                "covers page tables, so the two are not a legal combination"
            )
        if reset_sampling_state and not reload_inputs:
            raise ValueError(
                "DummySpecDecodeModel was told to rebuild its sampling state "
                "without reloading forward inputs; a sampler aligns its seed "
                "counters from host positions, which are only authoritative "
                "on a step that reloads them"
            )
        if kwargs.get("reload_sampling_params"):
            # Nothing to upload: this model samples by arithmetic and holds no
            # temperature, top-k or seed state. Recorded so a test can see the
            # command arrived rather than infer it from behaviour.
            self._sampling_param_uploads += 1
        if reset_sampling_state:
            self._sampling_state_resets += 1
        if reload_inputs:
            self._resident_tokens = tokens.clone()
            self._resident_positions = positions.clone()
            self._resident_page_table = kwargs.get("page_table")
        elif reload_page_table:
            self._resident_page_table = kwargs.get("page_table")
        if self._resident_tokens is None:
            raise ValueError(
                "DummySpecDecodeModel was asked to decode from resident "
                "inputs before any step reloaded them; the first decode of a "
                "chain reloads, and every reset starts a new chain"
            )
        return self._resident_tokens, self._resident_positions

    def _first_column(self, answer, rows, kwargs):
        """The token this step chose, whichever way it was asked to answer."""
        if kwargs.get("sampling_params") is not None:
            return answer.reshape(rows)[:rows].to(torch.int64)
        return answer.reshape(rows, 1, -1).argmax(dim=-1).reshape(rows).to(torch.int64)

    def _advance_resident(self, chosen):
        """Feed the chosen token back and advance the position once.

        This is what a resident decode means: the next step's input is what
        this step produced, held here rather than sent down from the host,
        because the host does not have it yet when that step is submitted.
        """
        rows = int(self._resident_tokens.shape[0])
        self._resident_tokens = chosen.reshape(rows, *([1] * (self._resident_tokens.dim() - 1))).to(
            self._resident_tokens.dtype
        )
        self._resident_positions = self._resident_positions + 1
        self._position_advances += 1

    def read_decode_output(self, tt_out, async_read: bool = False):
        """Read back a submitted decode.

        Split from submission, which is the first thing
        ``supports_async_decode`` asks for, and the reason a server can run
        this model with asynchronous scheduling. It samples nothing, advances
        no position and mutates no decode state: the answer was produced by
        the forward and is handed over unchanged.

        No completion events, because there is no device to signal one. The
        plugin's ``finalize_decode`` waits on each event it is given and an
        empty list is therefore already complete, which is the truthful
        description of a model whose forward runs on the host.
        """
        del async_read
        self._readbacks += 1
        return tt_out, []

    def _fixed_plain_answer(self, tokens, positions, rows, kwargs):
        """An ordinary decode under the ``fixed`` target.

        The base class answers a decode with zero logits, whose argmax is token
        0 at every step, so an unspeculated run of it emits one token forever
        and cannot be compared with anything. This follows the same rule the
        verify follows, which is what makes the two arms comparable.
        """
        choice = self._fixed_choice(tokens.reshape(rows, -1)[:, :1], positions.reshape(rows, -1)[:, :1])
        if kwargs.get("sampling_params") is not None:
            # Device sampling asks for ids rather than logits.
            return choice.reshape(rows).to(torch.int64)
        logits = torch.zeros(rows, 1, self.vocab_size, dtype=torch.float32)
        logits.scatter_(2, choice.to(torch.int64).unsqueeze(2), 1.0)
        return logits

    def _verified_ids(self, tokens, num_valid_drafts):
        """What this model claims at each candidate position.

        Column ``j`` is the choice draft ``j`` has to match, and each row's
        bonus sits at that row's own ``num_valid_drafts``. That is the
        contract's layout and upstream's: the committed block is this return
        truncated at the accepted count, with no column spent echoing an input
        the runner already holds.

        Accepting means returning the draft unchanged, so whatever the drafter
        proposed is what commits; that is what makes an n-gram drafter's
        acceptance rate configurable here rather than a property of the text.
        Past ``accept_depth`` the return differs from the draft, which stops
        that row. The depth is per row and never reduced across the batch: one
        row's short draft list must not shorten another's speculation.
        """
        rows, width = tokens.shape
        num_drafts = width - 1
        depth = num_drafts if self.accept_depth is None else self.accept_depth
        cap = torch.clamp(num_valid_drafts.to(torch.int64), max=depth)

        drafted = tokens[:, 1:].to(torch.int64)
        columns = torch.arange(num_drafts, dtype=torch.int64)
        diverge = columns.unsqueeze(0) >= cap.unsqueeze(1)
        verified = torch.empty((rows, width), dtype=torch.int64)
        verified[:, :num_drafts] = torch.where(diverge, (drafted + 1) % self.vocab_size, drafted)

        # The bonus, at each row's own count. Arithmetic on the row's last
        # committed token, so a step's output is predictable from its input.
        valid = num_valid_drafts.to(torch.int64)
        bonus = (tokens[:, 0].to(torch.int64) + valid + 1) % self.vocab_size
        verified.scatter_(1, valid.unsqueeze(1), bonus.unsqueeze(1))
        return verified.to(torch.int32)
