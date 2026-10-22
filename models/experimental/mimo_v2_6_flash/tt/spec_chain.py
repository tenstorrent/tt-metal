# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The MTP draft chained behind a verify pass on one TP=8 replica, and the pass's packed result page (the flow and
the layouts are in reference/spec_chain.py, which mirrors kernels/spec_chain.cpp).

Two one-core steps on persistent buffers only:

* ``chain``, at the end of the accept trace (TtMiMoModel.capture_verify_trace), after SampleVerify: writes the MTP
  draft's packed upload (SpecInputs.mtp_packed) from the accept page and the verify upload's chain row, and the result
  page's header (the accept page, the verify's argmax ids, the chain's status);
* ``drafts``, at the end of the MTP draft trace (TtMiMoMTP._body): writes the anchor slot's drafts into the result
  page, whichever wrote ``mtp_packed`` (the chain step, or the host for a host-planned draft).

The host reads the result page, one page from the row's first chip (``read``), or copies it to the host behind the
pass without blocking (``read_async``, ``unpack``: every pass of a verify is enqueued before the host waits, and a
deferred verify is read a step later). Every buffer either step reads or
writes is persistent (made before any trace capture: TtMiMoMTP.setup_cache), so both steps compile in the decode
trace's warm-up (``warm``) with the runtime arguments their captures use, and nothing compiles after a capture.

For a later chain program (the next verify's inputs derived on device from this result, the host reading one step
behind) everything it needs is resident: the accept page (VerifySampler.out), the drafts (the MTP heads' token rows)
and the result page, all at fixed addresses.
"""

import torch

import ttnn

from ..reference.sample_verify import OUT_WORDS
from ..reference.spec_chain import (
    ACCEPT_WORDS,
    ARGMAX_AT,
    CHAIN_COLUMNS,
    CHAIN_LIMIT,
    CHAIN_ROW,
    CHAIN_SLOTS,
    CHAIN_SOURCE,
    DRAFTS_AT,
    RESULT_WORDS,
    STATUS_AT,
    unpack_result,
)
from ..reference.spec_decode import MAX_DRAFTS, MTP_RING_TOKENS, SPEC_SLOTS
from ..reference.spec_inputs import MTP_ROWS, VERIFY_ROWS
from .device_sampler import CB_SCRATCH, _cb, _is_dram, _kernel, _ranges

# kernels/spec_chain.cpp addresses the chain row's words at these offsets.
assert (CHAIN_SOURCE, CHAIN_COLUMNS, CHAIN_LIMIT, CHAIN_SLOTS) == (0, 1, 2, 3)
CHAIN, DRAFTS = 0, 1  # the kernel's modes


def _align64(nbytes):
    return (nbytes + 63) // 64 * 64


class SpecChain:
    def __init__(self, plan, spec_inputs, accept, argmax, heads):
        """``spec_inputs``: the replica's SpecInputs (``verify_packed``, ``mtp_packed``). ``accept``: VerifySampler's
        output page. ``argmax``: the verify step's greedy tokens (GreedyHead's ``next_tokens``). ``heads``: the MTP
        passes' greedy token rows (TtMiMoMTP's heads). All persistent DRAM buffers; allocates the result page."""
        self.plan, self.max_batch = plan, spec_inputs.max_batch
        self.verify, self.mtp = spec_inputs.verify_packed, spec_inputs.mtp_packed
        self.accept, self.argmax, self.heads = accept, argmax, list(heads)
        assert len(self.heads) == MAX_DRAFTS and accept.shape[-1] == ACCEPT_WORDS and ACCEPT_WORDS % OUT_WORDS == 0
        for t in (self.verify, self.mtp, accept, argmax, *self.heads):
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

    def _program(self, mode, args, scratch):
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
        ]
        kernel = _kernel("spec_chain.cpp", cores, _ranges(cores), compile_args, [args], ttnn.ReaderConfigDescriptor())
        return ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(CB_SCRATCH, scratch, _ranges(cores))])

    # ------------------------------------------------------------------ device (trace-capturable)
    def chain(self):
        """The chain step (after SampleVerify wrote the accept page)."""
        b = self.max_batch
        args = [t.buffer_address() for t in (self.verify, self.accept, self.argmax, self.mtp, self.result)]
        scratch = (
            _align64(VERIFY_ROWS * b * 4)
            + _align64(ACCEPT_WORDS * 4)
            + _align64(b * 4)
            + _align64(MTP_ROWS * b * 4)
            + _align64(DRAFTS_AT * 4)
        )
        prog = self._program(CHAIN, args, scratch)
        ttnn.generic_op([self.verify, self.accept, self.argmax, self.mtp, self.result], prog)

    def drafts(self):
        """The drafts step (after the MTP passes wrote their heads' tokens)."""
        b = self.max_batch
        args = [self.mtp.buffer_address(), self.result.buffer_address()] + [h.buffer_address() for h in self.heads]
        scratch = _align64(MTP_ROWS * b * 4) + MAX_DRAFTS * _align64(b * 4) + _align64((RESULT_WORDS - DRAFTS_AT) * 4)
        prog = self._program(DRAFTS, args, scratch)
        ttnn.generic_op([self.mtp, *self.heads, self.result], prog)

    def warm(self):
        """Compile both steps before any trace capture. They write ``mtp_packed`` and the result page only, which no
        step reads before rewriting them (a capture idles ``mtp_packed`` first: SpecInputs.reset)."""
        self.chain()
        self.drafts()

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
