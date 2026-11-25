# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""Decode: guessed experts of the next MoE step streamed into L1 while attention runs (tt/mk/expert_pipe.py).

Needs a PF-layout cache (convert/pf_layout.py). Every expert step writes a mask of its n_guess highest-weight routed
experts; the next app's step is guessed from the mask of the same layer's previous loop pass (passes 2, 3) or of
the same app one token earlier (pass 1 and the non-loop layers). Right after app a's expert step the request for
app a + 1's guessed experts is queued (captured into the trace with the step); it waits for app a's op signal, so
its DRAM reads run during app a + 1's attention, not during app a's expert step.

    ov = ExpertOverlap(model)   # before capture, then eager steps as usual; ov.uninstall(model) between steps

release="moe" (default): the request waits for app a's expert step to end (its collector raises the signal).
release="w_all": it waits for a signal raised right after app a + 1's attention w_all projection instead, so the
stream skips the residual / gate programs between the two and runs during SDPA, w_o and the DRAM-light programs
before the expert step. Since the sdpa fill pieces (tt/mk/overlap.py) use that window, "w_all" lost its gain.
Measured, 2K / 120K ms: off 15.86 / 21.09, "w_all" 15.91 / 21.22, "moe" 15.52 / 20.85, "w_o" 17.29 / 22.55.
"""

import os

import torch

import ttnn
from models.demos.inkling2_mini.tt.mk.expert_pipe import ExpertPipe


class _Proxy:
    def __init__(self, ov, L, ex):
        self.ov, self.L, self.ex = ov, L, ex

    def __getattr__(self, name):
        return getattr(self.ex, name)

    def __call__(self, x, weights, indices):
        return self.ov.on_call(self.L, x, weights, indices)


class _Signal:
    """Calls the wrapped projection, then raises the tensor prefetcher op signal."""

    def __init__(self, ov, lin):
        self.ov, self.lin = ov, lin

    def __getattr__(self, name):
        return getattr(self.lin, name)

    def __call__(self, *a, **k):
        y = self.lin(*a, **k)
        ttnn.experimental.signal_tensor_prefetcher(self.ov.device, 0)
        return y


class ExpertOverlap:
    def __init__(self, model, ring_kb=240, guesses=None, release="moe", dynamic=True, mpfe=None):
        """ring_kb: per ring core; 240 holds 2 BFP8 guesses (230 KB). 300 no longer fits beside the staging and the
        router stage (CB clash at capture). guesses: {gate_up dtype: n_guess} override, e.g. {"BFP4": 3}; default 2 (4K: 1 -> 18.01 ms, 2 -> 17.81,
        BFP4 3 -> 17.86, BFP4 4 -> 18.39; with release "moe": 1 -> 15.90, 2 -> 15.52, BFP4 3 -> 15.83).
        release: see the module docstring."""
        from models.demos.inkling2_mini.tt import model as MD

        self.apps = list(MD.APPS)
        dev = self.device = model.device
        assert release in ("w_all", "w_o", "moe"), release
        self.release = release
        self.staged = []  # the staged w_all linears that raise the signal in their reader instead
        self.proj = {"w_all": "w_all", "w_o": "wo"}.get(release)  # the attention projection that releases a request
        self.dynamic = dynamic  # False: the seed masks stay (no guess_out writes; for tests)
        layers = sorted({L for L, _ in self.apps})
        self.orig = {L: model.layers[L].moe.fused for L in layers}
        assert all(getattr(e, "pf", False) for e in self.orig.values()), "needs a PF-layout cache"
        self.ring_bytes, self.mpfe = ring_kb << 10, dict(mpfe or {})
        self._at = None  # (ring, config) L1 addresses while suspended
        self.ep = ExpertPipe(dev, list(self.orig.values()), self.ring_bytes)
        b8 = ttnn.bfloat8_b
        g = {"BFP8": 2, "BFP4": 2, **(guesses or {})}
        self.n_guess = {}
        for L, e in self.orig.items():
            self.n_guess[L] = min(g["BFP8" if e.w_gate_up.dtype == b8 else "BFP4"], self.ep.max_guesses(e))
        n = self.orig[layers[0]].w_gate_up.shape[1]
        seed = torch.zeros(1, 1, 1, n, dtype=torch.int32)
        self.masks = []
        for L, _ in self.apps:
            m = seed.clone()
            m[..., : self.n_guess[L]] = 1
            self.masks.append(ttnn.from_torch(m, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                                              memory_config=ttnn.DRAM_MEMORY_CONFIG))  # fmt: skip
        mode = os.environ.get("INKLING_GUESS_SRC", "pass")  # "self": always the same app's last token
        self.src = []
        for a, (L, _) in enumerate(self.apps):
            prev = [b for b in range(a) if self.apps[b][0] == L]
            self.src.append(prev[-1] if prev and mode == "pass" else a)
        self.i, self.n = 0, 0
        self.ep.start(**self.mpfe)
        ttnn.synchronize_device(dev)
        self._request(0)
        for L in layers:
            model.layers[L].moe.fused = _Proxy(self, L, self.orig[L])
        self.wrapped = {}  # L -> the projection wrapped with the release signal
        if self.proj == "w_all":
            from models.demos.inkling2_mini.tt.mk.overlap import _Proxy as M3Proxy

            banks = [model.layers[L].attn.w_all.bank for L in layers]
            if all(isinstance(b, M3Proxy) for b in banks):
                for b in banks:
                    b.st.signal = self.ep.signal
                    self.staged.append(b.st)
                self.proj = None if self.staged else self.proj
        if self.proj:
            for L in layers:
                a = model.layers[L].attn
                self.wrapped[L] = getattr(a, self.proj)
                setattr(a, self.proj, _Signal(self, self.wrapped[L]))

    def _request(self, a):
        L = self.apps[a][0]
        if self.proj or self.staged:  # pairs with app a's w_all / w_o signal, after this request in program order
            ttnn.experimental.queue_tensor_prefetcher_wait_for_signal(self.device, 0, capture_into_trace=True)
        self.ep.request(self.orig[L], self.masks[self.src[a]], capture=True)

    def on_call(self, L, x, weights, indices):
        a = self.i
        assert self.apps[a][0] == L, ("call order out of step", a, L)
        moe = self.release == "moe"
        y = self.orig[L](x, weights, indices, guess=self.masks[self.src[a]], guess_out=self.masks[a] if self.dynamic else None,
                         n_guess=self.n_guess[L], signal=self.ep.signal if moe else None)  # fmt: skip
        if moe:
            self.ep.stepped()
        self.i = (a + 1) % len(self.apps)
        self._request(self.i)
        self.n += 1
        return y

    def uninstall(self, model):
        assert self.i == 0, "uninstall between decode steps only"
        if self.ep is not None:
            self._drain(model)
        for L, e in self.orig.items():
            model.layers[L].moe.fused = e
            e.ring = None
        for L, lin in self.wrapped.items():
            setattr(model.layers[L].attn, self.proj, lin)
        for st in self.staged:
            st.signal = None
        for t in self.masks:
            ttnn.deallocate(t)
        if self.ep is not None:
            self.ep.release()

    # Between two decode steps, around a chunked prefill (which needs the ring's L1): suspend() takes the queued
    # request and frees the ring, resume() makes the pipes again at the same L1 addresses (the traces' expert
    # steps and prefetch requests carry them), starts the prefetcher and queues app 0's request as __init__ does.
    def suspend(self, model):
        assert self.i == 0, "suspend between decode steps only"
        self._drain(model)
        self._at = (self.ep.space.buffer_address(), self.ep.space.config_address())
        for e in self.orig.values():
            e.ring = None
        self.ep.release()
        self.ep = None

    def resume(self, retake):
        """retake: see tt/mk/overlap.Overlap.resume. False: the ring is not back (still suspended)."""
        make = lambda: ExpertPipe(self.device, list(self.orig.values()), self.ring_bytes)  # noqa: E731
        at = lambda p: (p.space.buffer_address(), p.space.config_address())  # noqa: E731
        ep = retake(make, at, lambda p: p.release(), self._at)
        if ep is None:
            for e in self.orig.values():
                e.ring = None
            return False
        self.ep = ep
        ep.start(**self.mpfe)
        ttnn.synchronize_device(self.device)
        self._request(0)
        return True

    def _drain(self, model):
        """Take the queued request for app 0 and stop the prefetcher."""
        # the queued request for app 0: run app 0's step once more (output dropped) so its slabs are popped
        L = self.apps[0][0]
        ex = self.orig[L]
        dev = model.device
        l1 = ttnn.L1_MEMORY_CONFIG
        x = ttnn.from_torch(torch.zeros(1, 1, 1, ex.w_gate_up.shape[2]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev, memory_config=l1)  # fmt: skip
        w = ttnn.from_torch(torch.zeros(1, 1, 1, 8), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=l1)  # fmt: skip
        ids = ttnn.from_torch(torch.arange(8, dtype=torch.int32).reshape(1, 1, 1, 8), dtype=ttnn.uint16,
                              layout=ttnn.TILE_LAYOUT, device=dev, memory_config=l1)  # fmt: skip
        if self.proj or self.staged:
            ttnn.experimental.signal_tensor_prefetcher(dev, 0)  # the queued request waits for app 0's w_all / w_o
            ttnn.deallocate(ex(x, w, ids, guess=self.masks[self.src[0]]))
        else:
            ttnn.deallocate(ex(x, w, ids, guess=self.masks[self.src[0]], signal=self.ep.signal))
            self.ep.stepped()
        self.ep.close()
        self.ep.stop()
        for t in (x, w, ids):
            ttnn.deallocate(t)

    def abort(self, model):
        """After an error mid-step: try to drain the queued request and stop the prefetcher, so the device can close
        (a close with a queued request hangs). Best effort; restores the model either way."""
        self.i = 0
        try:
            self.uninstall(model)
        except Exception:
            for L, e in self.orig.items():
                model.layers[L].moe.fused = e
            for L, lin in self.wrapped.items():
                setattr(model.layers[L].attn, self.proj, lin)
            for st in self.staged:
                st.signal = None
            raise

    def stats(self):
        return {
            "expert_calls": self.n,
            "ring_kb": self.ep.ring_bytes >> 10,
            "n_guess": sorted(set(self.n_guess.values())),
            "src": "pass" if self.src != list(range(len(self.src))) else "self",
            "release": self.release,
        }
