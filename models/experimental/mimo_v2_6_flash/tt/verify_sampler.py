# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Speculative acceptance on device: a verify pass's committed tokens instead of its logits (vLLM ``fused_sample``).

A verify pass (reference/spec_decode.py) runs each request's last committed token and drafts in consecutive decode
slots. This head runs on the pass's logits, in a small trace replayed after the verify trace (TtMiMoModel.
capture_verify_trace):

1. DeviceSampler's SampleScan and SampleChipMerge over every slot, with the slots' own parameters (``params``, the
   decode sampler's layout: a slot takes its request's temperature / top_k / top_p), then the row all_gather;
2. SampleVerify (kernels/sample_verify.cpp), one core: per request of the pass, the accept walk of the plugin's host
   walk with the uniforms the plugin drew on the host (``ctrl``), greedy requests on the verify's argmax ids. It writes
   one page: per request the committed count and tokens, and whether its candidates were truncated.

The algorithm, its layouts and its exactness are in reference/sample_verify.py, which mirrors the kernel on the host.
The host reads 128 bytes per pass instead of 4 slots x 152k fp32 logits, and walks nothing. In the vLLM adapter the
accept trace goes on with the chain step (tt/spec_chain.py), which copies this page into the pass's result page and
derives the MTP draft from it; an ``argmax_ids`` step runs it too, every request greedy.

GreedyVerify is the accept step of a verify whose slots the device sampler does not serve (more slots than the
replica's decode users, which it serves up to 16): every request greedy on the argmax ids (kernels/greedy_verify.cpp),
the same pages.
"""

import numpy as np
import torch

import ttnn

from ..reference.sample_verify import GROUP_WORDS, GROUPS, OUT_WORDS, PARAM_WORDS, control_words
from ..reference.sample_verify import param_words, unpack_out
from ..reference.spec_decode import MAX_DRAFTS
from .device_sampler import CB_SCRATCH, _cb, _is_dram, _kernel, _ranges


def _align64(nbytes):
    return (nbytes + 63) // 64 * 64


def _words(words):
    """uint32 words as the int32 [1, n] torch tensor ttnn uploads as uint32."""
    return torch.from_numpy(np.asarray(words, dtype=np.uint32).view(np.int32).reshape(1, -1).copy())


class GreedyVerify:
    def __init__(self, plan, max_batch):
        """The control page and the output page (persistent: call before any trace capture)."""
        self.plan, self.max_batch = plan, max_batch
        self.ctrl = self._persistent(control_words([]))
        self.out = self._persistent([0] * (GROUPS * OUT_WORDS))
        self.groups = 0  # requests accepted, and how many of them on truncated candidates (``tally``)
        self.truncated = 0

    def _persistent(self, words):
        return ttnn.from_torch(
            _words(words),
            device=self.plan.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.plan.replicate(),
        )

    def _copy_ctrl(self, groups):
        rep = self.plan.replicate()
        host = ttnn.from_torch(
            _words(control_words(groups)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
        )
        ttnn.copy_host_to_device_tensor(host, self.ctrl)

    def stage(self, slots, groups):
        """Copies a pass's control page (reference.sample_verify.pass_inputs' (slots, groups)); raises for a sampled
        request (the plugin routes one here only when the model offers it ``fused_sample``)."""
        if slots or any(g is not None and not g[2] for g in groups):
            raise ValueError(
                f"the greedy accept step takes greedy requests only (the device sampler does not serve this "
                f"replica's {GROUPS}-slot verify); got sampled slots {sorted(slots)}"
            )
        self._copy_ctrl(groups)

    def tally(self, out):
        """Counts a fused pass's accepted requests and those accepted on truncated candidates (``out``: unpack_out)."""
        used = [g for g in out if g[0] > 0]
        self.groups += len(used)
        self.truncated += sum(g[2] for g in used)

    def read(self):
        """The last pass's [(count, committed tokens, truncated)] per group (blocks); counts the groups."""
        words = ttnn.to_torch(ttnn.get_device_tensors(self.out)[0]).reshape(-1).to(torch.int64) & 0xFFFFFFFF
        out = unpack_out(words.tolist())
        self.tally(out)
        return out

    def __call__(self, logits, argmax):
        """The accept step on the verify's greedy tokens ``argmax`` (uint32 [1, max_batch] row-major DRAM, GreedyHead's
        ``next_tokens``; ``logits`` unused). Returns the persistent output page (reference.sample_verify.unpack_out)."""
        assert _is_dram(argmax), "the greedy accept step reads the argmax ids from DRAM"
        cores = [ttnn.CoreCoord(0, 0)]
        args = [argmax.buffer_address(), self.ctrl.buffer_address(), self.out.buffer_address()]
        scratch = _align64(self.max_batch * 4) + _align64(GROUPS * GROUP_WORDS * 4) + _align64(GROUPS * OUT_WORDS * 4)
        kernel = _kernel(
            "greedy_verify.cpp",
            cores,
            _ranges(cores),
            [self.max_batch, GROUPS, MAX_DRAFTS, GROUP_WORDS, OUT_WORDS, CB_SCRATCH],
            [args],
            ttnn.ReaderConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(CB_SCRATCH, scratch, _ranges(cores))])
        ttnn.generic_op([argmax, self.ctrl, self.out], prog)
        return self.out


class VerifySampler(GreedyVerify):
    def __init__(self, sampler, vocab):
        """``sampler``: the model's DeviceSampler (its scan cores, scratch buffers and candidate page are reused; the
        verify replays never overlap a decode replay). ``vocab``: the full vocabulary (top_k at or past it is none).
        Allocates the persistent inputs and output: call before any trace capture."""
        assert sampler.users >= GROUPS, f"a verify pass has {GROUPS} slots, the sampler serves {sampler.users} users"
        super().__init__(sampler.plan, sampler.max_batch)
        self.sampler, self.vocab = sampler, vocab
        self.params = self._persistent(self._param_words({}))
        self.staged = {}  # the sampled slots ``params`` holds (``stage``; None: unknown)

    # ------------------------------------------------------------------ host side
    def _param_words(self, slots):
        words = []
        for s in range(self.max_batch):
            temperature, top_k, top_p = slots.get(s, (0.0, 0, 1.0))
            words += param_words(temperature, top_k, top_p, self.vocab)
        assert len(words) == self.max_batch * PARAM_WORDS
        return words

    def host_inputs(self, slots, groups):
        """Host tensors for ``set_inputs``. ``slots``: {slot: (temperature, top_k, top_p)} of the pass's sampled
        slots (every other slot greedy). ``groups``: one (first slot, drafts, greedy, u_bonus, u_accept, u_corr) per
        request of the pass (reference.sample_verify.group_words)."""
        rep = self.plan.replicate()
        return [
            ttnn.from_torch(
                _words(self._param_words(slots)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
            ),
            ttnn.from_torch(
                _words(control_words(groups)), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
            ),
        ]

    def set_inputs(self, hosts):
        """Copies a pass's parameters and control page, in queue order with the work already enqueued."""
        ttnn.copy_host_to_device_tensor(hosts[0], self.params)
        ttnn.copy_host_to_device_tensor(hosts[1], self.ctrl)
        self.staged = None

    def stage(self, slots, groups):
        """``set_inputs(host_inputs(slots, groups))``, without the parameters when ``params`` already holds them: a
        pass's parameters are those of its sampled slots, the same for every all-greedy pass and for a sampled
        request's passes in a row, so a step usually copies the control page only."""
        if slots != self.staged:
            words = _words(self._param_words(slots))
            rep = self.plan.replicate()
            host = ttnn.from_torch(words, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)
            ttnn.copy_host_to_device_tensor(host, self.params)
            self.staged = dict(slots)
        self._copy_ctrl(groups)

    # ------------------------------------------------------------------ device side (trace-capturable)
    def _verify(self, cands, argmax):
        s, tp = self.sampler, self.plan.tp
        cores = [s.final_core]
        args = [
            cands.buffer_address(),
            _is_dram(cands),
            self.params.buffer_address(),
            argmax.buffer_address(),
            _is_dram(argmax),
            self.ctrl.buffer_address(),
            self.out.buffer_address(),
        ]
        scratch = (
            _align64(tp * s.users * s.k * 8)
            + _align64(tp * s.slots * 16)
            + _align64(self.max_batch * PARAM_WORDS * 4)
            + _align64(self.max_batch * 4)
            + _align64(GROUPS * GROUP_WORDS * 4)
            + _align64(GROUPS * OUT_WORDS * 4)
            + 3 * s.candidates * 4
        )
        kernel = _kernel(
            "sample_verify.cpp",
            cores,
            _ranges(cores),
            [s.users, tp, s.k, s.vocab_local, self.max_batch, GROUPS, MAX_DRAFTS, CB_SCRATCH, GROUP_WORDS, OUT_WORDS],
            [args],
            ttnn.ReaderConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(CB_SCRATCH, scratch, _ranges(cores))])
        ttnn.generic_op([cands, self.params, argmax, self.ctrl, self.out], prog)

    def __call__(self, logits, argmax):
        """logits: the verify step's [1, 1, 32, vocab_local] fp32 TILE logits (slots in rows 0..users-1); argmax: its
        greedy tokens, uint32 [1, max_batch] row-major (GreedyHead's ``next_tokens``). Returns the persistent output
        page, uint32 [1, GROUPS * OUT_WORDS] (reference.sample_verify.unpack_out)."""
        s = self.sampler
        assert logits.layout == ttnn.TILE_LAYOUT and logits.dtype == ttnn.float32 and _is_dram(logits)
        s._scan(logits, params=self.params)
        s._chip_merge(params=self.params)
        cands = self.plan.all_gather(s.chip_cands, dim=-1)
        self._verify(cands, argmax)
        cands.deallocate(True)
        return self.out
