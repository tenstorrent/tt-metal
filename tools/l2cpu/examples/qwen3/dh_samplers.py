# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""Pluggable samplers for the x280 decode harness (decode_harness.py).

Every sampler works on the ROW-MAJOR bf16 logits tensor that the decode trace leaves in DRAM
([1, 1, 32, Vpad], interleaved, one page per user row, page b in bank b % 8 at
buffer_address + (b // 8) * page_size). The harness calls, once per decode session:

    sampler.bind(device, logits_tensor, batch, vocab, params_per_user, seeds)

and then, once per decode step, after the device has finished the step:

    tokens = sampler.sample(step)      # list of `batch` ints

`params_per_user` is a list of dicts {"temperature", "top_k", "top_p"} (temperature 0 = greedy);
`seeds` is a list of per-user uint64 seeds. `step` is the x280s step index (the harness passes
decode step s as step s + 1; step 0 is the prefill token).

Samplers that read the logits on the host share one LogitsReader so that a CompareSampler reads
the device tensor once per step. Timings of the last call are in `sampler.last` (seconds):
{"read": readback, "sample": sampling}.
"""

import hashlib
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import deps  # noqa: E402,F401  (bindings to the other l2cpu components, see deps.py)
from x280s_ref import X280S  # noqa: E402


def bf16_rows_to_f32(rows_u16):
    return (rows_u16.astype(np.uint32) << 16).view(np.float32)


def row_hash(row_u16):
    return hashlib.blake2b(np.ascontiguousarray(row_u16).tobytes(), digest_size=8).hexdigest()


class LogitsReader:
    """Host copy of the device row-major bf16 logits, cached per step.

    mode "full": `logits.cpu()` of the whole tensor (32 rows, 32 x page bytes) through the
                 command queue (ordered after the trace, no explicit sync needed).
    mode "rows": raw NoC reads (ttnn.cluster.read_from_core) of only the `batch` needed pages,
                 using the DRAM bank table; requires the device to be idle (the reader calls
                 ttnn.synchronize_device first; the harness has normally synced already).
    """

    def __init__(self, device, logits_tensor, batch, vocab, mode="rows", device_id=0):
        import ttnn

        self.ttnn = ttnn
        self.device = device
        self.t = logits_tensor
        self.batch = batch
        self.vocab = vocab
        self.mode = mode
        self.device_id = device_id
        shape = list(logits_tensor.shape)
        self.vpad = shape[-1]
        self.nrows = shape[-2]
        assert logits_tensor.layout == ttnn.ROW_MAJOR_LAYOUT, logits_tensor.layout
        assert logits_tensor.dtype == ttnn.bfloat16, logits_tensor.dtype
        self.addr = logits_tensor.buffer_address()
        self.page = logits_tensor.buffer_aligned_page_size()
        assert self.page == self.vpad * 2, (self.page, self.vpad)
        self.banks = None
        if mode == "rows":
            self.banks = ttnn.cluster.get_dram_bank_table(device_id)
        self._step = None
        self._rows = None
        self.last_read_s = 0.0
        self.bytes_read = 0

    def page_location(self, b):
        nb = len(self.banks)
        e = self.banks[b % nb]
        return e["noc_x"], e["noc_y"], e["base_addr"] + self.addr + (b // nb) * self.page

    def _read_full(self):
        ttnn = self.ttnn
        host = self.t.cpu(blocking=True)
        import torch

        tt = ttnn.to_torch(host)  # bf16 [1,1,32,Vpad]
        u16 = tt.reshape(-1, self.vpad)[: self.batch].contiguous().view(torch.int16).numpy().view(np.uint16)
        self.bytes_read = self.nrows * self.page
        return u16

    def _read_rows(self):
        ttnn = self.ttnn
        ttnn.synchronize_device(self.device)
        out = np.empty((self.batch, self.vpad), np.uint16)
        for b in range(self.batch):
            x, y, a = self.page_location(b)
            buf = ttnn.cluster.read_from_core(self.device_id, x, y, a, self.page)
            out[b] = np.frombuffer(buf, np.uint16)
        self.bytes_read = self.batch * self.page
        return out

    def rows(self, step):
        """[batch, Vpad] uint16 (bf16 bits) for this step (read once per step)."""
        if self._step != step:
            t0 = time.perf_counter()
            self._rows = self._read_full() if self.mode == "full" else self._read_rows()
            self.last_read_s = time.perf_counter() - t0
            self._step = step
        return self._rows

    def invalidate(self):
        self._step = None


class Sampler:
    name = "base"

    def __init__(self, reader_mode="rows"):
        self.reader_mode = reader_mode
        self.reader = None
        self.last = {"read": 0.0, "sample": 0.0}

    def bind(self, device, logits_tensor, batch, vocab, params_per_user, seeds, reader=None):
        self.device = device
        self.logits = logits_tensor
        self.batch = batch
        self.vocab = vocab
        self.params = list(params_per_user)
        self.seeds = [int(s) & 0xFFFFFFFFFFFFFFFF for s in seeds]
        assert len(self.params) == batch and len(self.seeds) == batch
        self.reader = reader or LogitsReader(device, logits_tensor, batch, vocab, mode=self.reader_mode)

    def sample(self, step):
        raise NotImplementedError


class HostRefSampler(Sampler):
    """The bit-exact host reference: x280s_sample_row (libx280s_host.so) on the bf16 row of each user."""

    name = "hostref"

    def __init__(self, reader_mode="rows", lib=None):
        super().__init__(reader_mode)
        self.lib = lib or X280S()

    def sample_rows(self, rows, step):
        toks = []
        for b in range(self.batch):
            p = self.params[b]
            tok, st = self.lib.sample(
                rows[b], p["temperature"], p["top_k"], p["top_p"], self.seeds[b], user=b, step=step, vocab=self.vocab
            )
            if tok < 0:
                raise RuntimeError("x280s_sample_row error %d (user %d step %d)" % (tok, b, step))
            if st.nan_count:
                print("hostref: user %d step %d: %d NaN logits" % (b, step, st.nan_count))
            toks.append(int(tok))
        return toks

    def sample(self, step):
        rows = self.reader.rows(step)
        t0 = time.perf_counter()
        toks = self.sample_rows(rows, step)
        self.last = {"read": self.reader.last_read_s, "sample": time.perf_counter() - t0}
        return toks


class ThreadedHostRefSampler(HostRefSampler):
    """HostRefSampler with the users split over a thread pool (ctypes releases the GIL during
    x280s_sample_row). One X280S instance (own workspace) per thread; results are identical."""

    name = "hostref-mt"

    def __init__(self, reader_mode="full", threads=8):
        super().__init__(reader_mode)
        from concurrent.futures import ThreadPoolExecutor

        self.threads = threads
        self.libs = [X280S() for _ in range(threads)]
        self.pool = ThreadPoolExecutor(max_workers=threads)

    def _chunk(self, rows, step, lib, users):
        out = []
        for b in users:
            p = self.params[b]
            tok, _ = lib.sample(
                rows[b], p["temperature"], p["top_k"], p["top_p"], self.seeds[b], user=b, step=step, vocab=self.vocab
            )
            if tok < 0:
                raise RuntimeError("x280s_sample_row error %d (user %d step %d)" % (tok, b, step))
            out.append((b, int(tok)))
        return out

    def sample_rows(self, rows, step):
        n = min(self.threads, self.batch)
        parts = [list(range(i, self.batch, n)) for i in range(n)]
        futs = [self.pool.submit(self._chunk, rows, step, self.libs[i], parts[i]) for i in range(n)]
        toks = [0] * self.batch
        for f in futs:
            for b, t in f.result():
                toks[b] = t
        return toks


def argmax_lowest(row_u16, vocab):
    """Greedy over bf16 bits: NaN -> -inf, ties -> lowest index (explicit, not relying on torch)."""
    f = bf16_rows_to_f32(np.asarray(row_u16[:vocab]))
    f = np.where(np.isnan(f), np.float32(-np.inf), f)
    m = f.max()
    return int(np.flatnonzero(f == m)[0])


class TorchArgmaxSampler(Sampler):
    """Independent greedy check: torch.argmax over the same host copy.

    torch.argmax documents "the first maximal value" on ties and does so on CPU (checked:
    argmax([1,3,3,2]) == 1); we still compute the lowest maximal index explicitly when the row
    has a tie at the maximum, so the semantics do not depend on the torch version/backend.
    Only valid for greedy parameters (temperature <= 0).
    """

    name = "torchargmax"

    def bind(self, *a, **kw):
        super().bind(*a, **kw)
        for p in self.params:
            if p["temperature"] > 0:
                raise ValueError("torchargmax is greedy only; got temperature %r" % p["temperature"])
        self.ties = 0

    def sample(self, step):
        import torch

        rows = self.reader.rows(step)
        t0 = time.perf_counter()
        toks = []
        for b in range(self.batch):
            u = torch.from_numpy(rows[b, : self.vocab].view(np.int16)).view(torch.bfloat16).float()
            u = torch.nan_to_num(u, nan=float("-inf"))
            i = int(torch.argmax(u))
            if int((u == u[i]).sum()) > 1:
                self.ties += 1
                i = int(torch.nonzero(u == u[i])[0, 0])
            toks.append(i)
        self.last = {"read": self.reader.last_read_s, "sample": time.perf_counter() - t0}
        return toks


class CompareSampler(Sampler):
    """Runs `primary` and `reference` on the same step (same host copy of the logits when both read
    on the host), records every mismatch (the first one with full context) and returns the
    reference's tokens, so the decode trajectory follows the reference."""

    def __init__(self, primary, reference, strict=False, topn=8):
        super().__init__(getattr(reference, "reader_mode", "rows"))
        self.primary = primary
        self.reference = reference
        self.strict = strict
        self.topn = topn
        self.name = "compare:%s,%s" % (primary.name, reference.name)

    def bind(self, device, logits_tensor, batch, vocab, params_per_user, seeds, reader=None):
        super().bind(device, logits_tensor, batch, vocab, params_per_user, seeds, reader=reader)
        self.primary.bind(device, logits_tensor, batch, vocab, params_per_user, seeds, reader=self.reader)
        self.reference.bind(device, logits_tensor, batch, vocab, params_per_user, seeds, reader=self.reader)
        self.n_compared = 0
        self.n_mismatch = 0
        self.first_mismatch = None
        self.context = {}

    def _context(self, step, b, tp, tr):
        rows = self.reader.rows(step)
        f = bf16_rows_to_f32(rows[b, : self.vocab])
        order = np.lexsort((np.arange(f.size), -f))[: self.topn]
        return {
            "step": step,
            "user": b,
            "primary": tp,
            "reference": tr,
            "primary_logit": float(f[tp]) if 0 <= tp < f.size else None,
            "reference_logit": float(f[tr]) if 0 <= tr < f.size else None,
            "top": [[int(i), float(f[i])] for i in order],
            "row_hash": row_hash(rows[b]),
            **self.context,
        }

    def sample(self, step):
        tp = self.primary.sample(step)
        tr = self.reference.sample(step)
        self.last = {
            "read": self.primary.last["read"],
            "sample": self.primary.last["sample"],
            "sample_ref": self.reference.last["sample"],
        }
        for b in range(self.batch):
            self.n_compared += 1
            if tp[b] != tr[b]:
                self.n_mismatch += 1
                if self.first_mismatch is None:
                    self.first_mismatch = self._context(step, b, tp[b], tr[b])
                    print("COMPARE MISMATCH:", self.first_mismatch)
                if self.strict:
                    raise AssertionError("sampler mismatch at step %d user %d: %d vs %d" % (step, b, tp[b], tr[b]))
        return tr

    def summary(self):
        return {
            "compared": self.n_compared,
            "mismatches": self.n_mismatch,
            "identical": self.n_mismatch == 0 and self.n_compared > 0,
            "first_mismatch": self.first_mismatch,
        }


def make_sampler(spec, reader_mode="full"):
    """'hostref' | 'torchargmax' | 'x280' | 'compare:<primary>,<reference>'"""
    if spec.startswith("compare:"):
        a, b = spec[len("compare:") :].split(",")
        return CompareSampler(make_sampler(a, reader_mode), make_sampler(b, reader_mode))
    if spec == "hostref":
        return HostRefSampler(reader_mode)
    if spec.startswith("hostref-mt"):
        n = int(spec.split(":")[1]) if ":" in spec else 8
        return ThreadedHostRefSampler(reader_mode, threads=n)
    if spec == "torchargmax":
        return TorchArgmaxSampler(reader_mode)
    if spec == "x280":
        from l2cpu_sampler import L2cpuHostSampler  # needs the --x280 firmware session

        return L2cpuHostSampler(reader_mode)
    raise ValueError("unknown sampler %r" % spec)
