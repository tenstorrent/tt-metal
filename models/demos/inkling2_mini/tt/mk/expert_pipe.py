# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Guessed experts streamed ahead by the DRAM cores: the tensor prefetcher (``start_tensor_prefetcher``) reads the
experts a guess mask marks from PF-layout expert weights (tt/fused/moe_bank.py pf_weights) into an L1 ring on the
expert step's ring workers while other programs run; the step then reads those experts from L1 and the rest from
DRAM.

    ep = ExpertPipe(device, [experts, ...], ring_bytes)   # every BankMoeExperts on the same cores (PF layout)
    ep.start()
    ep.request(experts, mask, capture=...)                  # one request per expert step, in call order
    y = experts(x, weights, indices, guess=mask, signal=ep.signal)  # the same mask until this step has run
    ep.stop()

Paced: every request after the first waits for the op signal the previous expert step raises at its end, so a step's
ring slabs are read from DRAM after the previous step's reads, not during them (the step pops its gate_up slabs before
it reads its down slabs, which would otherwise let the next request start mid-step). Waits and signals pair one to
one: ExpertPipe counts the unmatched signals (``signaled``) and close() queues a wait for each.

The mask is [1, 1, 1, E] uint16 ROW_MAJOR in DRAM; the prefetcher reads it on device when it reaches the request and
streams the marked experts in ascending id order: gate_up as 2 entries per expert (half a column each), then down as
1 entry per expert. The step reads the same mask, so both agree on what is in the ring. The ring must hold one
step's gate_up entries and, separately, its down entries (each entry size wraps on its own).
"""

import ttnn
from models.demos.inkling2_mini.tt.mk.pipe_stage import _core_list


def _crs(cs):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(*c), ttnn.CoreCoord(*c)) for c in cs])


class ExpertPipe:
    def __init__(self, device, experts, ring_bytes):
        self.device = device
        ref = experts[0]
        rc = ref.ring_cores()
        for ex in experts[1:]:
            assert ex.ring_cores() == rc, "every expert step on the same ring cores"
        self.space = ttnn.experimental.create_prefetcher_pipe_space(
            device, sender_cores=ttnn.CoreRangeSet(set()), receiver_domain=_crs([c for cs in rc.values() for c in cs]),
            ring_size=ring_bytes, max_receivers_per_pipe=ref.n_ring(), num_dram_senders=2 * ref.nb,
        )  # fmt: skip
        self.pipes = ttnn.experimental.create_prefetcher_pipes_for_tensor_prefetcher(
            self.space, [(b, _crs(cs)) for b, cs in rc.items()], support_multi_receiver_shards=True
        )
        self.ring_bytes = self.space.ring_size()
        # receiver r of a bank, counted over that bank's pipes in order, gets the bank's slab r
        order = {b: [] for b in rc}
        for p in self.pipes:
            cs = _core_list(p.receiver_cores())
            b = next(b for b, v in rc.items() if set(cs) <= set(v))
            order[b] += cs
        for ex in experts:
            ex.set_ring(order, self.space.config_address(), self.space.buffer_address(), self.ring_bytes)
        self.running = False
        self.signal = None
        self.signaled = 0  # expert steps run with signal=self.signal whose signal no wait has taken yet

    def max_guesses(self, ex):
        """Most marked experts one step's ring slabs hold (gate_up entries, then down entries, each on its own)."""
        t = lambda w: ttnn.Tile([32, 32]).get_tile_size(w.dtype)  # noqa: E731
        from models.demos.inkling2_mini.tt.fused.moe_expert import KT_DN, KT_GU

        gu, dn = KT_GU // 2 * t(ex.w_gate_up), KT_DN * t(ex.w_down)
        return min((self.ring_bytes - self.ring_bytes % gu) // (2 * gu), (self.ring_bytes - self.ring_bytes % dn) // dn)

    def start(self):
        if not self.running:
            ttnn.experimental.start_tensor_prefetcher(self.device)
            self.running = True
            self.signal = ttnn.experimental.get_tensor_prefetcher_signal_address(self.device, 0)

    def stop(self):
        if self.running:
            ttnn.experimental.stop_tensor_prefetcher(self.device)
            ttnn.synchronize_device(self.device)
            self.running = False

    def release(self):
        """Free the ring's L1 (after stop())."""
        assert not self.running
        import gc

        self.pipes = self.space = None
        gc.collect()

    def wait(self, capture=False):
        """Queue a wait for the last expert step's signal (requests queued after it start once that step ended)."""
        assert self.signaled > 0
        ttnn.experimental.queue_tensor_prefetcher_wait_for_signal(self.device, 0, capture_into_trace=capture)
        self.signaled -= 1

    def stepped(self):
        """An expert step ran with signal=self.signal."""
        self.signaled += 1

    def close(self, capture=False):
        """Take every unmatched signal (before stop(), or before a trace capture whose replays must start even)."""
        while self.signaled:
            self.wait(capture)

    def request(self, ex, mask, capture=False):
        """Queue one expert step's slabs: the experts ``mask`` marks, of ``ex``'s weights. Waits for the previous
        step's signal first when there is one outstanding."""
        if self.signaled:
            self.wait(capture)
        ttnn.experimental.queue_tensor_prefetcher_request(
            self.device, [(ex.w_gate_up, 2, [], mask), (ex.w_down, 1, [], mask)], prefetcher_pipes=self.pipes,
            capture_into_trace=capture,
        )  # fmt: skip
