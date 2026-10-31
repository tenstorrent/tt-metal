# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MTP draft chained behind a verify pass on one TP=8 replica, and the pass's packed result page (the flow and
the layouts are in reference/spec_chain.py, which mirrors kernels/spec_chain.cpp).

Three one-core steps on persistent buffers only (and two more with every MTP layer, below):

* ``chain``, at the end of the accept trace (TtMiMoModel.capture_verify_trace), after SampleVerify: writes the MTP
  draft's packed upload (SpecInputs.mtp_packed) from the accept page and the verify upload's chain row, and the result
  page's header (the accept page, the verify's argmax ids and positions, the chain's status);
* ``drafts``, at the end of the MTP draft trace (TtMiMoMTP._body): writes the anchor slot's drafts into the result
  page, whichever wrote ``mtp_packed`` (the chain step, or the host for a host-planned draft);
* ``relay``, first in the verify trace (TtMiMoModel.capture_verify_trace), before SpecInputs.derive_verify: for a
  chained verify (the host has not read the lane's last pass) writes the source's tokens and positions into
  ``verify_packed`` and its drafts into VerifySampler's control page from the result page that pass left; a no-op
  for an upload the host built.

With every MTP layer (``layers`` > 1, TtMiMoMTP with MIMO_MTP_LAYERS=3) ``mtp_packed`` holds the commit's description
and the draft trace runs two more steps (reference/spec_chain.py ``plan_step``, ``tokens_step``):

* ``plan_layers``, first: the passes' inputs (TtMiMoMTP's persistent ``LayerInputs``) and the new stash page from the
  description and the stash;
* ``tokens(i)``, before pass i >= 1: that pass's tokens from the layer-0 tokens and the heads of the passes before it.

The host reads the result page, one page from the row's first chip (``read``), or copies it to the host behind the
pass without blocking (``read_async``, ``unpack``: every pass of a verify is enqueued before the host waits, and a
deferred verify is read a step later). Every buffer a step reads or writes is persistent (made before any trace
capture: TtMiMoMTP.setup_cache, VerifySampler's control page in TtMiMoModel.setup_vllm_kv_cache), so every step
compiles in the decode trace's warm-up (``warm``) with the runtime arguments their captures use, and nothing compiles
after a capture.

The relay reads the result page in place: everything the next verify needs (the accept page, the bonus, the drafts)
is there, at a fixed address, until the next pass of the lane rewrites it. Steps on a pass the host built leave the
relay nothing to do, so one verify trace serves both.

With draft groups (MIMO_SPEC_GROUPS=2: two requests of a lane in one pass, reference/spec_decode.py) every step runs per
group: the chain row and the result page have a block per group, the chain step writes each group's commit, the
drafts step each group's drafts, the plan and tokens steps each group's slots, and the relay each relayed group.

With DFlash drafting (``dflash``, tt/dflash.py) the chain step writes the DFlash draft's packed upload instead of
``mtp_packed`` (reference.spec_chain.dflash_chain_step), and the drafts step, at the end of the DFlash draft trace,
reads its anchor from that upload and the drafts from the drafter's GreedyHead (dflash_drafts_step). The result page
and the relay are the same.
"""

import torch

import ttnn

from ..reference.sample_verify import GROUP_WORDS, GROUPS, OUT_WORDS
from ..reference.spec_chain import (
    ACCEPT_AT,
    ACCEPT_WORDS,
    ARGMAX_AT,
    CHAIN_COLUMNS,
    CHAIN_LIMIT,
    CHAIN_RELAY,
    CHAIN_ROW,
    CHAIN_SLOTS,
    CHAIN_SOURCE,
    CTRL_DRAFTS,
    DRAFTS_AT,
    POSITIONS_AT,
    RESULT_WORDS,
    STASH_POSITIONS,
    STASH_TOKENS,
    STATUS_AT,
    unpack_result,
)
from ..reference.spec_decode import DRAFT_GROUPS, GROUP_SLOTS, MAX_DRAFTS, MTP_RING_TOKENS, SERIAL_WRITES, SPEC_SLOTS
from ..reference.spec_inputs import DESC_A, DESC_COUNT, DESC_DRAFTING, DESC_SLOTS, DESC_TOKENS, MTP_ROWS, VERIFY_ROWS
from .device_sampler import CB_SCRATCH, _cb, _is_dram, _kernel, _ranges

# kernels/spec_chain.cpp addresses the chain row's words at these offsets, and the accept page at the result's start.
assert (CHAIN_SOURCE, CHAIN_COLUMNS, CHAIN_LIMIT, CHAIN_SLOTS) == (0, 1, 2, 3) and ACCEPT_AT == 0
# ... and the description's and the stash page's words at these.
assert (DESC_DRAFTING, DESC_COUNT, DESC_A, DESC_SLOTS, DESC_TOKENS) == (0, 1, 2, 3, 3 + GROUP_SLOTS)
assert (STASH_POSITIONS, STASH_TOKENS) == (0, SPEC_SLOTS)
CHAIN, DRAFTS, RELAY, PLAN, TOKENS = 0, 1, 2, 3, 4  # the kernel's modes
CTRL_WORDS = GROUPS * GROUP_WORDS  # VerifySampler's control page


def _align64(nbytes):
    return (nbytes + 63) // 64 * 64


class SpecChain:
    def __init__(self, plan, spec_inputs, accept, argmax, heads, layer_inputs=None, dflash=None):
        """``spec_inputs``: the replica's SpecInputs (``verify_packed``, ``mtp_packed``). ``accept``: VerifySampler's
        output page. ``argmax``: the verify step's greedy tokens (GreedyHead's ``next_tokens``). ``heads``: the MTP
        passes' greedy token rows (TtMiMoMTP's heads). ``layer_inputs``: with every MTP layer, TtMiMoMTP's
        LayerInputs (the plan and tokens steps' outputs, the stash page, ``table_rows``); None with one layer.
        ``dflash``: with DFlash drafting, (TtDFlash's packed upload, its reference.spec_chain.DFlashLayout), and
        ``heads`` its GreedyHead's token row alone (row k: draft k). All persistent DRAM buffers; allocates the result
        page."""
        self.plan, self.max_batch, self.spec_inputs = plan, spec_inputs.max_batch, spec_inputs
        self.verify, self.mtp = spec_inputs.verify_packed, spec_inputs.mtp_packed
        self.accept, self.argmax, self.heads = accept, argmax, list(heads)
        self.layer_inputs = layer_inputs
        self.layers = MAX_DRAFTS if layer_inputs is not None else 1
        self.dflash = None if dflash is None else dflash[1]
        if dflash is not None:
            # The chain step writes the DFlash draft's upload, which the drafts step reads.
            assert layer_inputs is None and dflash[0].shape[-1] == self.dflash.words
            self.mtp = dflash[0]
        assert len(self.heads) == (1 if dflash is not None else MAX_DRAFTS)
        assert accept.shape[-1] == ACCEPT_WORDS and ACCEPT_WORDS % OUT_WORDS == 0
        steps = []
        if layer_inputs is not None:
            li = layer_inputs
            steps = [li.pos, li.rope, li.hidden, *li.tokens, *li.serial, li.stash]
            assert self.max_batch <= li.table_rows and len(li.tokens) == MAX_DRAFTS and len(li.serial) == SERIAL_WRITES
        for t in (self.verify, self.mtp, accept, argmax, *self.heads, *steps):
            assert _is_dram(t) and t.layout == ttnn.ROW_MAJOR_LAYOUT, "the chain reads one-page DRAM buffers"
        self.result = ttnn.from_torch(
            torch.zeros(1, RESULT_WORDS, dtype=torch.int32),
            device=plan.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=plan.replicate(),
        )
        self.core = ttnn.CoreCoord(0, 0)

    def _program(self, mode, args, scratch, layer=0):
        cores = [self.core]
        compile_args = [
            mode,
            self.max_batch,
            SPEC_SLOTS,
            MAX_DRAFTS,
            VERIFY_ROWS,
            CHAIN_ROW,
            MTP_ROWS,
            ACCEPT_WORDS,
            MTP_RING_TOKENS,
            ARGMAX_AT,
            STATUS_AT,
            DRAFTS_AT,
            RESULT_WORDS,
            CB_SCRATCH,
            POSITIONS_AT,
            CHAIN_RELAY,
            CTRL_WORDS,
            CTRL_DRAFTS,
            self.layers,
            self.layer_inputs.table_rows if self.layer_inputs is not None else self.max_batch,
            layer,
            *self._dflash_args(),
            DRAFT_GROUPS,
            OUT_WORDS,
            GROUP_WORDS,
            int(self.layer_inputs is not None),  # planned: the MTP drafter's planned program (LayerInputs)
            SERIAL_WRITES,
            GROUP_SLOTS,
        ]
        kernel = _kernel("spec_chain.cpp", cores, _ranges(cores), compile_args, [args], ttnn.ReaderConfigDescriptor())
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(CB_SCRATCH, scratch, _ranges(cores))])

    def _dflash_args(self):
        """The kernel's drafter and DFlash layout arguments (zeros with MTP drafting)."""
        d = self.dflash
        if d is None:
            return [0] * 6
        assert d.groups == DRAFT_GROUPS
        return [1, d.words, d.ring, d.block, d.heads, d.last]

    def _draft_bytes(self):
        """The draft's upload that the chain step writes and the drafts step reads."""
        return self.dflash.words * 4 if self.dflash is not None else MTP_ROWS * self.max_batch * 4

    # ------------------------------------------------------------------ device (trace-capturable)
    def chain(self):
        """The chain step (after SampleVerify wrote the accept page)."""
        b = self.max_batch
        args = [t.buffer_address() for t in (self.verify, self.accept, self.argmax, self.mtp, self.result)]
        scratch = (
            _align64(VERIFY_ROWS * b * 4)
            + _align64(ACCEPT_WORDS * 4)
            + _align64(b * 4)
            + _align64(self._draft_bytes())
            + _align64(DRAFTS_AT * 4)
        )
        prog = self._program(CHAIN, args, scratch)
        ttnn.generic_op([self.verify, self.accept, self.argmax, self.mtp, self.result], prog)

    def drafts(self):
        """The drafts step (after the drafter's passes wrote their heads' tokens)."""
        b = self.max_batch
        args = [self.mtp.buffer_address(), self.result.buffer_address()] + [h.buffer_address() for h in self.heads]
        scratch = (
            _align64(self._draft_bytes()) + MAX_DRAFTS * _align64(b * 4) + _align64((RESULT_WORDS - DRAFTS_AT) * 4)
        )
        prog = self._program(DRAFTS, args, scratch)
        ttnn.generic_op([self.mtp, *self.heads, self.result], prog)

    def relay(self, ctrl):
        """The relay step (first in the verify trace): a relayed source's tokens and positions into ``verify_packed``
        and its drafts into ``ctrl``, VerifySampler's control page, from the result page the lane's last pass left
        (reference.spec_chain.relay_step). Writes nothing unless the upload's chain row says relay."""
        b = self.max_batch
        args = [self.verify.buffer_address(), self.result.buffer_address(), ctrl.buffer_address()]
        scratch = _align64(VERIFY_ROWS * b * 4) + _align64(RESULT_WORDS * 4) + _align64(CTRL_WORDS * 4)
        prog = self._program(RELAY, args, scratch)
        ttnn.generic_op([self.result, ctrl, self.verify], prog)

    def plan_layers(self):
        """The plan step (every layer; the start of the MTP draft trace)."""
        li = self.layer_inputs
        outs = [li.pos, li.rope, li.hidden, li.tokens[0], *li.serial]
        args = [self.mtp.buffer_address(), li.stash.buffer_address()] + [t.buffer_address() for t in outs]
        row = _align64(self.max_batch * 4)
        prog = self._program(PLAN, args, (DRAFT_GROUPS + 2 + len(outs)) * row)
        ttnn.generic_op([self.mtp, li.stash, *outs], prog)

    def tokens(self, layer):
        """The tokens step of pass ``layer`` >= 1 (every layer): its tokens from the layer-0 tokens and the earlier
        passes' heads, in each draft group's width (the descriptions' drafting words)."""
        li = self.layer_inputs
        assert 1 <= layer < MAX_DRAFTS
        ins = [li.tokens[0], *self.heads[:layer], self.mtp]
        args = [li.tokens[0].buffer_address(), li.tokens[layer].buffer_address()]
        args += [h.buffer_address() for h in self.heads[:layer]] + [self.mtp.buffer_address()]
        rows = 2 + layer + DRAFT_GROUPS
        prog = self._program(TOKENS, args, rows * _align64(self.max_batch * 4), layer=layer)
        ttnn.generic_op([*ins, li.tokens[layer]], prog)

    def warm(self, ctrl=None):
        """Compile every step before any trace capture (the relay with ``ctrl``, VerifySampler's control page), on
        idle uploads (SpecInputs.reset: a recapture finds the last served step's there). The relay writes nothing on
        an idle upload (its chain row is zeros); the chain and drafts steps write ``mtp_packed`` and the result page
        only, which no step reads before rewriting them; the plan and tokens steps write the draft trace's inputs,
        which it rewrites the same way, and on the idle description the chain step writes they leave the stash page
        as it was (on the last served commit the plan step would re-plan it into a stash a prompt just emptied)."""
        self.spec_inputs.reset()
        if ctrl is not None:
            self.relay(ctrl)
        self.chain()
        self.drafts()
        if self.layer_inputs is not None:
            self.plan_layers()
            for layer in range(1, MAX_DRAFTS):
                self.tokens(layer)

    # ------------------------------------------------------------------ host
    def read(self):
        """The result page (reference.spec_chain.ChainResult; blocks until the steps writing it finished)."""
        words = ttnn.to_torch(ttnn.get_device_tensors(self.result)[0]).reshape(-1).to(torch.int64) & 0xFFFFFFFF
        return unpack_result(words.tolist())

    def read_async(self):
        """Enqueue a copy of the result page to the host behind the work already enqueued, without blocking (the
        decode step's token read, generator_vllm.read_decode_output, does the same). Returns the host tensor it
        fills, valid once an event recorded after it completed (``unpack``); the page itself may be rewritten by the
        next pass meanwhile."""
        return self.result.cpu(blocking=False)

    @staticmethod
    def unpack(host):
        """The ChainResult of a host copy made by ``read_async`` (after its event completed; touches no device)."""
        words = ttnn.to_torch(ttnn.get_device_tensors(host)[0]).reshape(-1).to(torch.int64) & 0xFFFFFFFF
        return unpack_result(words.tolist())
