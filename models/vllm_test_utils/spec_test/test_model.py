# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import torch
from loguru import logger

from models.vllm_test_utils.no_op_test.test_model import DummyNoOpModel


class DummySpecDecodeModel(DummyNoOpModel):
    """Dummy model implementing the speculative-decoding contract.

    Does no work: it accepts the drafts it is handed and returns arithmetic for
    everything else. That makes it two things at once.

    First, an instrument. Its sibling ``DummyNoOpModel`` exists to measure what
    vLLM costs per step with the model removed; this one measures what vLLM
    costs per *speculative* step. With ``TT_SPEC_ACCEPT_DEPTH=0`` nothing is
    accepted, so each step commits one token and the step time is the host cost
    of one verify-then-propose iteration with no device work under it. That
    number decides whether a plugin-driven speculation loop can compete with a
    model-internal one, and it has only ever been estimated.

    Second, the only way to exercise the plugin's speculative path end to end.
    Everything between the runner and the engine needs a real server: the
    ``take_draft_token_ids`` handshake, the scheduler's lookahead budget and
    its grammar truncation of drafts, placeholder accounting, and KV cache
    allocation. None of it is reachable from a host test.

    The contract is
    https://github.com/tenstorrent/vllm-tt-plugin/issues/110 and the model side
    of it is ``docs/SPEC_DECODE_CONTRACT.md`` in that repository. This class
    imports the plugin's value types, which is how a model declares a plan; the
    plugin is therefore required to import this module, as it is to serve it.

    Environment:
        ``TT_SPEC_ACCEPT_DEPTH``  how many of each row's drafts to accept.
            Default ``-1``, meaning all of them, which exercises the loop
            hardest. ``0`` accepts none, which is the measurement mode.
        ``TT_SPEC_MAX_DRAFTS``    the largest draft length to admit, default 16.
            A launch asking for more is served at this length.
    """

    model_capabilities = {
        **DummyNoOpModel.model_capabilities,
        # The master gate. Everything below it is read only when a launch
        # carries a speculative_config.
        "supports_spec_decode": True,
        # Nothing: an n-gram drafter runs on the host and asks the model for no
        # drafting of its own. A device drafter would declare device_propose
        # and hidden_feed here.
        "spec_requirements": [],
        # Left at 1 by omission. Any value above 1 selects the block-output
        # rail, which owns the committed width per step and cannot be combined
        # with speculation.
    }

    _MAX_DRAFTS = int(os.environ.get("TT_SPEC_MAX_DRAFTS", "16"))

    def __init__(self, mesh_device, max_batch_size, vocab_size, **kwargs):
        super().__init__(mesh_device, max_batch_size, vocab_size, **kwargs)
        depth = int(os.environ.get("TT_SPEC_ACCEPT_DEPTH", "-1"))
        self.accept_depth = None if depth < 0 else depth
        logger.info(
            f"DummySpecDecodeModel: accept_depth="
            f"{'all' if self.accept_depth is None else self.accept_depth}, "
            f"max_drafts={self._MAX_DRAFTS}"
        )

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        """Which ``(max_num_seqs, K)`` points this model can serve.

        Everything, up to a draft length ceiling, because it allocates nothing
        per candidate and holds no state a candidate could invalidate. A real
        model answers from its own lane arithmetic and L1 budget, and returns
        ``SpecReject`` for a point it cannot fit.
        """
        from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, DRAFTER_STATE_INTERNAL, SpecPlan, SpecReject

        del vllm_config, max_num_seqs  # no cost scales with either here
        if requested_k < 1:
            return SpecReject(
                reason=f"a draft length of {requested_k} speculates nothing",
                supported_k=tuple(range(1, cls._MAX_DRAFTS + 1)),
            )
        return SpecPlan(
            effective_k=min(requested_k, cls._MAX_DRAFTS),
            # One decode row per request: this model has no physical candidate
            # layout, so a request costs it the rows a plain decode costs.
            lanes_per_request=1,
            extra_bytes_per_seq=0,
            extra_bytes_per_token=0,
            accept_modes=(ACCEPT_MODE_ARGMAX_IDS,),
            drafter_state=DRAFTER_STATE_INTERNAL,
            supports_narrow_decode=False,
        )

    def decode_forward(self, *args, **kwargs):
        """One decode step, which is a verify when the runner asks for one.

        Verify is not a separate entry point: the runner passes the same
        ``tokens`` and ``start_pos`` a plain decode gets, ``1+K`` wide, plus
        ``num_valid_drafts``, ``accepted_counts`` and ``spec_mode``. Their
        absence is what makes a step an ordinary decode, and then the base
        class answers.
        """
        spec_mode = kwargs.get("spec_mode")
        if spec_mode is None:
            return super().decode_forward(*args, **kwargs)

        from vllm_tt_plugin.spec_decode import ACCEPT_MODE_ARGMAX_IDS, VerifyOutput

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
        self._check_accepted_counts(kwargs["accepted_counts"], tokens.shape[0])
        return VerifyOutput(
            spec_mode=ACCEPT_MODE_ARGMAX_IDS,
            argmax_ids=self._verified_ids(tokens, num_valid_drafts),
        )

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

    def _check_accepted_counts(self, accepted_counts, rows):
        """Refuse a count outside the contract's domain.

        A count is never 0 and never exceeds the block: a model reading
        ``accepted_counts - 1`` to select a candidate state would index -1.
        Checked here because this model is also the plugin's conformance
        witness, and a silent acceptance would make it a poor one.
        """
        if accepted_counts is None:
            raise ValueError(
                "accepted_counts may be None only after a fused_sample step, "
                "which DummySpecDecodeModel does not serve"
            )
        if accepted_counts.shape != (rows,):
            raise ValueError(f"accepted_counts must be [{rows}], got {tuple(accepted_counts.shape)}")
        bad = accepted_counts[accepted_counts < 1]
        if bad.numel():
            raise ValueError(f"accepted_counts entries must be at least 1, got {bad.tolist()}")
