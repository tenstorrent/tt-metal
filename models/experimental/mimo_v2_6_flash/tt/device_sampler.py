# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Temperature / top-k / top-p sampling of the decode users on device, inside the decode trace (vLLM mode).

vLLM's sampler (``apply_top_k_top_p``) keeps the top_k largest logits, then the smallest set of the most likely tokens
whose probability reaches top_p, and draws from it. Here, per user (rows 0..users-1 of the logit tile):

1. SampleScan, on 32 cores per chip, each over a range of the chip's vocab shard: the compute kernel sums
   exp((logit - the range's largest) / T) over the range in fp32 (the SFPU's exact exp) while the reader and the
   writer keep half of the user slots' K largest fp32 logits each (two of 4 slots, or four of 8 past 4 users; exact
   values, integer keys); the writer then adds up the sums per user in fixed point;
2. SampleChipMerge, one core per user: the chip's K largest, sorted, and its normalizer, the sum of
   exp((logit - the chip's largest) / T) over the shard;
3. after one row all-gather, SampleMerge walks the row's candidates (the chips' lists, merged) in descending
   order: probabilities are softmax(logits / T) over the vocabulary, or over the first top_k candidates when
   top_k is set, as vLLM normalizes; the nucleus keeps a candidate while the probability ranked above it is below
   top_p; the token is drawn by inverse CDF with one uniform per row from a splitmix64 stream on device (its state
   persistent, seeded by the host with the parameters), so a sampled step copies nothing either. A greedy row (temperature 0 or top_k 1) takes the first candidate: the lowest id among the
   largest logits, as GreedyHead and the host's argmax.

The data movement cores have no FPU: steps 2 and 3 run in fixed point (kernels/sample_fixed.hpp), probabilities to
~1e-9 of the largest. The candidates hold the row's K most likely tokens and, past that rank, each chip's K most
likely, so the draw equals vLLM's whenever the nucleus has at most K tokens (99.7% of the decode rows of GPQA
reasoning at temperature 1, top_p 0.95; 99.9% fit in the tp * K candidates) or lies inside the candidates;
otherwise it is the nucleus's tokens among the candidates. top_k beyond tp * K is not supported
(``max_device_top_k``). A seeded row's stream is keyed by (seed, decode position) (``stream_states``). Requests for
logprobs stay on host sampling (the plugin routes them).

Grammar (vLLM structured output, ``set_grammar``): a row whose parameters flag it reads its row of the packed mask,
and the scan reader sets the logits it disallows to a large negative value in L1 before anything reads them, so the
row samples, greedy or not, from the allowed tokens only, as vLLM's sampler after its bitmask.
"""

import numpy as np
import torch

import ttnn

from .common import TILE

KERNEL_DIR = "models/experimental/mimo_v2_6_flash/tt/kernels"
SCAN_CORES = 32
CANDIDATES_PER_CHIP = 32  # K: each scan core's and each chip's list per user
MAX_USERS = 8  # decode users of the scan's user slots; more sample on the host
# Per row: log2(e) / T (fp32 bits), top_p (Q24), top_k, greedy, 1 / T (fp32 bits), grammar (1: apply the row's mask),
# unused.
PARAM_WORDS = 8
MASK_ALIGN = 16  # words: a chip's mask row is padded to 64 bytes (whole DRAM reads)
# Circular buffers: the scan's logit tiles, its shift and scale tiles, the compute kernel's sum tile, the reader's and
# writer's scratch, the tiles' address from the reader to the writer.
CB_X, CB_M, CB_A, CB_S, CB_SCRATCH, CB_OUT, CB_READY = 0, 1, 2, 3, 4, 5, 6


MASK64 = (1 << 64) - 1
GAMMA = 0x9E3779B97F4A7C15  # splitmix64's state increment
# How close (in cumulative probability) a uniform may land to a bucket edge before a mismatch counts as rounding.
EDGE = 2e-5


def _mix64(z):
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def splitmix_next(state):
    """(next state, uniform in [0, 1)) of the device's splitmix64 stream (kernels/sample_fixed.hpp)."""
    state = (state + GAMMA) & MASK64
    return state, (_mix64(state) >> 40) / float(1 << 24)


def stream_states(seeds, positions, generator=None):
    """Per-row splitmix64 states (int64 [len(seeds)]) for draws starting at the rows' decode positions. A seeded row's
    state at position p is mix(seed) + p * GAMMA, and each sampled step advances it by GAMMA, so its draw at every
    position depends only on (seed, position), not on which steps reloaded it or what it was batched with. Unseeded
    rows (seed None) get random states from ``generator`` (torch's default when None)."""
    states = torch.randint(-(2**63), 2**63 - 1, (len(seeds),), dtype=torch.int64, generator=generator)
    for r, (seed, pos) in enumerate(zip(seeds, positions)):
        if seed is not None:
            state = (_mix64((int(seed) + GAMMA) & MASK64) + max(int(pos), 0) * GAMMA) & MASK64
            states[r] = state - (1 << 64) if state >= 1 << 63 else state
    return states


def reference_draw(logits, temperature, top_k, top_p, u):
    """vLLM's draw over the whole vocabulary (logits: torch [vocab]): softmax(logits / T), or over the top_k largest
    (top_k < 1 or >= vocab: none); keep a token while the mass ranked above it is below top_p; the first kept token
    whose cumulative probability exceeds u * the kept mass. Greedy (T 0 or top_k 1): the first maximum. Returns
    (token, nucleus size, whether u lies within EDGE of a bucket edge)."""
    row = logits.double()
    vocab = row.numel()
    if temperature == 0.0 or top_k == 1:
        return int(torch.argmax(row)), 1, False
    vals, ids = torch.sort(row / temperature, descending=True, stable=True)
    probs = torch.softmax(vals, dim=0)
    if 1 <= top_k < vocab:
        probs = probs[:top_k] / probs[:top_k].sum()
        ids = ids[:top_k]
    keep = (probs.cumsum(0) - probs) < top_p
    keep[0] = True
    kept = probs[keep]
    cum = kept.cumsum(0)
    threshold = u * cum[-1]
    pick = min(int((cum <= threshold).sum()), kept.numel() - 1)
    near = bool(((cum - threshold).abs() < EDGE).any())
    return int(ids[keep][pick]), int(keep.sum()), near


def _is_dram(t):
    return int(t.memory_config().buffer_type == ttnn.BufferType.DRAM)


def _cb(index, nbytes, cores, dtype=ttnn.uint32, page=None):
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page or nbytes)
    return ttnn.CBDescriptor(total_size=nbytes, core_ranges=cores, format_descriptors=[fmt])


def _kernel(name, cores, crs, compile_args, runtime_args, config):
    rt = ttnn.RuntimeArgs()
    for core, args in zip(cores, runtime_args):
        rt[core.x][core.y] = args
    return ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/{name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=crs,
        compile_time_args=compile_args,
        runtime_args=rt,
        config=config,
    )


def _ranges(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


class DeviceSampler:
    def __init__(self, plan, vocab_local, users, max_batch, scan_cores=SCAN_CORES):
        assert vocab_local % TILE == 0 and 1 <= users <= MAX_USERS and max_batch * 4 <= 1024
        self.plan, self.vocab_local, self.users, self.max_batch = plan, vocab_local, users, max_batch
        self.slots = 4 if users <= 4 else 8  # user slots of a scan core's partials block
        self.k = CANDIDATES_PER_CHIP
        self.candidates = self.k * plan.tp
        device = plan.mesh_device
        grid = device.compute_with_storage_grid_size()
        assert scan_cores % 4 == 0 and scan_cores <= min(grid.x * grid.y, 128)
        self.num_scan = scan_cores
        self.scan_cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(scan_cores)]
        self.merge_cores = self.scan_cores[:users]
        self.final_core = ttnn.CoreCoord(0, 0)
        tiles = vocab_local // TILE
        bounds = [tiles * i // scan_cores for i in range(scan_cores + 1)]
        self.ranges = list(zip(bounds[:-1], bounds[1:]))
        self.max_tiles = max(e - b for b, e in self.ranges)
        rep = plan.replicate()

        def persistent(shape, dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, value=None):
            torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.int32
            t = torch.zeros(shape, dtype=torch_dtype) if value is None else value
            return ttnn.from_torch(
                t, device=device, dtype=dtype, layout=layout, memory_config=memory_config, mesh_mapper=rep
            )

        # Scan partials: ``slots`` user slots of K (key, index) pairs per scan core, and ``slots`` uint64 sums per scan
        # core, as raw 4096-byte pages. The chip candidates page: the users' sorted lists, then their normalizers at
        # byte 2048.
        assert scan_cores * self.slots * 8 <= 4096 and self.users * self.k * 8 <= 2048
        partial_pages = scan_cores * self.slots * self.k * 8 // 4096
        self.partials = persistent((1, 1, TILE, TILE * partial_pages), ttnn.uint32)
        self.sums = persistent((1, 1, TILE, TILE), ttnn.uint32)
        self.chip_cands = persistent((1, 1, TILE, TILE), ttnn.uint32)
        # Per-row parameters, rewritten in place (``set_params``); rows default to greedy.
        dram_rm = {"layout": ttnn.ROW_MAJOR_LAYOUT, "memory_config": ttnn.DRAM_MEMORY_CONFIG}
        self.params = persistent(
            (1, max_batch * PARAM_WORDS), ttnn.uint32, value=self._params(*self._greedy()), **dram_rm
        )
        self.rng = persistent((1, max_batch * 2), ttnn.uint32, **dram_rm)
        self.rows = self._greedy()  # (temperature, top_k, top_p) of ``params``, for ``check`` and ``set_grammar``
        # The grammar mask of the users' rows, one row-major row per user on every chip: the words of the chip's vocab
        # shard (a tile column of 32 tokens per word), padded to MASK_ALIGN; rewritten in place (``set_grammar``).
        self.mask_words = vocab_local // TILE
        self.mask_stride = -(-self.mask_words // MASK_ALIGN) * MASK_ALIGN
        self.mask = ttnn.from_torch(
            torch.full((users, plan.tp * self.mask_stride), -1, dtype=torch.int32),
            device=device,
            dtype=ttnn.uint32,
            mesh_mapper=plan.shard(-1),
            **dram_rm,
        )
        # The tokens of the last step, persistent (the trace writes them; reads and the feedback use them).
        self.next_tokens = persistent((1, max_batch), ttnn.uint32, **dram_rm)
        self.compute_config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        for cb in (CB_X, CB_M, CB_A):
            modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
        self.compute_config.unpack_to_dest_mode = modes

    # ------------------------------------------------------------------ host side
    def _greedy(self):
        n = self.max_batch
        return torch.zeros(n), torch.zeros(n), torch.ones(n)

    def _params(self, temperature, top_k, top_p, grammar=None):
        """int32 [1, max_batch * PARAM_WORDS] of per-row temperature, top_k (0: none), top_p (each [max_batch]) and
        whether the row's grammar mask applies (``grammar``, bool [max_batch]; default none)."""
        greedy = (temperature == 0) | (top_k == 1)
        inv = torch.where(greedy, torch.ones_like(temperature), 1.0 / temperature.double().clamp(min=1e-2).float())
        scale = (inv.double() * 1.4426950408889634).float()
        zeros = torch.zeros(self.max_batch, dtype=torch.int32)
        words = [
            torch.from_numpy(scale.numpy().view(np.int32).copy()),
            (top_p.double().clamp(max=1.0) * (1 << 24)).round().to(torch.int32),
            top_k.to(torch.int32),
            greedy.to(torch.int32),
            torch.from_numpy(inv.float().numpy().view(np.int32).copy()),
            zeros if grammar is None else grammar.to(torch.int32),
            zeros,
            zeros,
        ]
        return torch.stack(words, dim=1).reshape(1, -1)

    def host_params(self, temperature, top_k, top_p, states=None):
        """torch [rows] each (rows <= max_batch; the rest greedy) -> (host tensors for ``set_params``, whether any row
        samples). top_k < 1 or at least the vocabulary means no top_k; top_k 1 or temperature 0 is greedy. ``states``:
        int64 [max_batch] splitmix64 states for the rows' draws (``stream_states``; default: random)."""
        rows = temperature.numel()
        temp, k, p = self._greedy()
        temp[:rows] = temperature.float()
        kk = top_k.float()
        k[:rows] = torch.where((kk >= 1) & (kk < self.vocab_local * self.plan.tp), kk, torch.zeros_like(kk))
        p[:rows] = top_p.float()
        greedy = (temp == 0) | (k == 1)
        if states is None:
            states = stream_states([None] * self.max_batch, [0] * self.max_batch)
        self.pending_rows = (temp.clone(), k.clone(), p.clone())
        words = states.numpy().astype(np.int64).view(np.int32).reshape(1, -1)  # low word first
        rep = self.plan.replicate()
        hosts = [
            ttnn.from_torch(self._params(temp, k, p), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep),
            ttnn.from_torch(
                torch.from_numpy(words.copy()), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep
            ),
        ]
        return hosts, bool((~greedy[:rows]).any())

    def set_params(self, hosts):
        """Uploads ``host_params``' tensors; the rows' grammar flags are off until ``set_grammar``."""
        ttnn.copy_host_to_device_tensor(hosts[0], self.params)
        ttnn.copy_host_to_device_tensor(hosts[1], self.rng)
        self.rows = self.pending_rows

    def set_grammar(self, mask=None, rows=None):
        """Constrains the next draws of the users ``rows`` (bool [n], n <= users) to the tokens ``mask`` allows: int32
        [n, words] in vLLM's packing (bit j of word w allows token 32 * w + j; the tokens of words past ``words`` are
        disallowed). The rows keep the parameters and streams ``set_params`` gave them. ``mask`` None lifts every row's
        constraint. Uploads the parameters again (with each row's flag) and the mask rows of the flagged users."""
        flags = torch.zeros(self.max_batch, dtype=torch.bool)
        if mask is not None:
            n = mask.shape[0]
            assert n <= self.users and rows.numel() == n
            flags[:n] = rows
            words = self.plan.tp * self.mask_words
            given = min(words, mask.shape[1])
            full = torch.full((self.users, words), -1, dtype=torch.int32)
            full[:n] = 0
            full[:n, :given] = mask[:, :given]
            padded = torch.full((self.users, self.plan.tp, self.mask_stride), -1, dtype=torch.int32)
            padded[:, :, : self.mask_words] = full.reshape(self.users, self.plan.tp, self.mask_words)
            host = ttnn.from_torch(
                padded.reshape(self.users, -1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=self.plan.shard(-1),
            )
            ttnn.copy_host_to_device_tensor(host, self.mask)
        params = self._params(*self.rows, grammar=flags)
        host = ttnn.from_torch(
            params, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=self.plan.replicate()
        )
        ttnn.copy_host_to_device_tensor(host, self.params)

    def read_rng(self):
        """The rows' stream states (Python ints), blocking."""
        words = ttnn.to_torch(ttnn.get_device_tensors(self.rng)[0]).reshape(-1).to(torch.int64) & 0xFFFFFFFF
        return [int(words[2 * r]) | int(words[2 * r + 1]) << 32 for r in range(self.max_batch)]

    def check(self, logits, tokens, rng, active):
        """Checks a step's tokens (torch [>= users]) against ``reference_draw`` on its logit rows (torch [users, vocab])
        with each row's uniform from its stream state before the step (``read_rng``). Returns (rows checked, rows the
        draw is exact for (nucleus and top_k within K), [(row, device token, reference token)] mismatches among those
        away from a bucket edge)."""
        temp, k, p = self.rows
        checked, exact, bad = 0, 0, []
        for r in range(self.users):
            if not active[r]:
                continue
            t, kk, pp = float(temp[r]), int(k[r]), float(p[r])
            greedy = t == 0.0 or kk == 1
            u = 0.0 if greedy else splitmix_next(rng[r])[1]
            token, nucleus, near = reference_draw(logits[r], t, kk if kk >= 1 else -1, pp, u)
            checked += 1
            if nucleus > self.k or kk > self.k:
                continue
            exact += 1
            got = int(tokens[r])
            if got != token and float(logits[r, got]) != float(logits[r, token]) and not near:
                bad.append((r, got, token))
        return checked, exact, bad

    # ------------------------------------------------------------------ device side (trace-capturable)
    def _scan(self, logits, params=None):
        """Step 1. ``params``: the per-row parameters to read (default ``self.params``; a verify passes its own,
        tt/verify_sampler.py)."""
        params = self.params if params is None else params
        cores, crs = self.scan_cores, _ranges(self.scan_cores)
        params_bytes = self.max_batch * PARAM_WORDS * 4
        block_bytes = self.slots * self.k * 8
        mask_bytes = self.users * self.mask_stride * 4
        reader_args = [
            [
                logits.buffer_address(),
                self.partials.buffer_address(),
                params.buffer_address(),
                b,
                e,
                i,
                self.mask.buffer_address(),
            ]
            for i, (b, e) in enumerate(self.ranges)
        ]
        reader = _kernel(
            "sample_scan.cpp",
            cores,
            crs,
            [self.users, self.k, self.max_batch, CB_X, CB_M, CB_A, CB_SCRATCH, CB_READY, self.slots, self.mask_stride],
            reader_args,
            ttnn.ReaderConfigDescriptor(),
        )
        compute = _kernel(
            "sample_norm_compute.cpp",
            cores,
            crs,
            [CB_X, CB_M, CB_A, CB_S],
            [[e - b] for b, e in self.ranges],
            self.compute_config,
        )
        writer = _kernel(
            "sample_scan_writer.cpp",
            cores,
            crs,
            [self.users, self.k, CB_S, CB_OUT, CB_READY, self.slots],
            [
                [self.partials.buffer_address(), self.sums.buffer_address(), b, e, i]
                for i, (b, e) in enumerate(self.ranges)
            ],
            ttnn.WriterConfigDescriptor(),
        )
        tile = 4096
        cbs = [
            _cb(CB_X, self.max_tiles * tile, crs, ttnn.float32, tile),
            _cb(CB_M, tile, crs, ttnn.float32),
            _cb(CB_A, tile, crs, ttnn.float32),
            _cb(CB_S, tile, crs, ttnn.float32),
            _cb(CB_SCRATCH, block_bytes // 2 + params_bytes + mask_bytes, crs),
            _cb(CB_OUT, block_bytes // 2 + self.slots * 8, crs),
            _cb(CB_READY, 16, crs),
        ]
        prog = ttnn.ProgramDescriptor(kernels=[reader, compute, writer], semaphores=[], cbs=cbs)
        ttnn.generic_op([logits, self.partials, self.sums, params, self.mask], prog)

    def _chip_merge(self, params=None):
        """Step 2, with ``params`` as ``_scan``."""
        params = self.params if params is None else params
        cores, crs = self.merge_cores, _ranges(self.merge_cores)
        addrs = [
            self.partials.buffer_address(),
            self.sums.buffer_address(),
            params.buffer_address(),
            self.chip_cands.buffer_address(),
        ]
        scratch = self.num_scan * (self.k * 8 + self.slots * 8) + self.max_batch * PARAM_WORDS * 4 + self.k * 8 + 16
        kernel = _kernel(
            "sample_chip_merge.cpp",
            cores,
            crs,
            [self.num_scan, self.k, CB_SCRATCH, self.max_batch, self.slots],
            [addrs + [u] for u in range(self.users)],
            ttnn.ReaderConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(CB_SCRATCH, scratch, crs)])
        ttnn.generic_op([self.partials, self.sums, params, self.chip_cands], prog)

    def _merge(self, cands, token_input):
        tp = self.plan.tp
        cores, crs = [self.final_core], _ranges([self.final_core])
        args = [
            cands.buffer_address(),
            _is_dram(cands),
            self.params.buffer_address(),
            self.rng.buffer_address(),
            self.next_tokens.buffer_address(),
            _is_dram(self.next_tokens),
            token_input.buffer_address() if token_input is not None else 0,
            _is_dram(token_input) if token_input is not None else 0,
        ]
        scratch = tp * (self.users * self.k * 8 + self.slots * 16) + self.max_batch * (PARAM_WORDS * 4 + 8)
        scratch += 2 * self.candidates * 4 + self.max_batch * 4
        kernel = _kernel(
            "sample_merge.cpp",
            cores,
            crs,
            [self.users, tp, self.k, self.vocab_local, self.max_batch, CB_SCRATCH],
            [args],
            ttnn.ReaderConfigDescriptor(),
        )
        prog = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[_cb(CB_SCRATCH, scratch, crs)])
        io = [cands, self.params, self.rng, self.next_tokens] + ([token_input] if token_input is not None else [])
        ttnn.generic_op(io, prog)

    def __call__(self, logits, token_input=None):
        """logits [1, 1, 32, vocab_local] fp32 TILE (DRAM interleaved; users in rows 0..users-1).

        Returns the persistent ``next_tokens`` [1, max_batch] uint32 row-major (zeros past ``users``); also writes it
        into ``token_input`` (a [1, max_batch] uint32 row-major tensor) when given."""
        assert logits.layout == ttnn.TILE_LAYOUT and logits.dtype == ttnn.float32 and _is_dram(logits)
        assert logits.shape[-1] == self.vocab_local and logits.shape[-2] == TILE
        self._scan(logits)
        self._chip_merge()
        cands = self.plan.all_gather(self.chip_cands, dim=-1)
        self._merge(cands, token_input)
        cands.deallocate(True)
        return self.next_tokens
