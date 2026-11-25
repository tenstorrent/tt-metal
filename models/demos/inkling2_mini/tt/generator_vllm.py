# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Inkling-2-Mini behind vLLM: the model class the vLLM TT plugin (tenstorrent/vllm-tt-plugin) loads.

    TT_MODEL_CLASS_OVERRIDES=Inkling2ForCausalLM=models.demos.inkling2_mini.tt.generator_vllm:Inkling2ForCausalLM \
    vllm serve models/demos/inkling2_mini/vllm --tokenizer $INKLING_CKPT --served-model-name Inkling-2-Mini \
        --max-model-len 196608 --max-num-seqs 1 --shutdown-timeout 60 \
        --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
        --additional-config '{"tt": {"worker_l1_size": 1394752, "trace_region_size": 134217728,
                                     "sample_on_device_mode": "all"}}'

One request at a time on one Blackhole p100a (the model is batch 1 and has one sequence of state on the card).
``models/demos/inkling2_mini/vllm/config.json`` stands in for the checkpoint's config.json, whose model type the
transformers release vLLM pins does not know; the tokenizer and chat template are the checkpoint's own.

What the plugin's calls do here:
- prefill_forward: the prompt goes through the traced decode step token by token (tt/generator.py), as in the demo.
  A prompt that starts with the tokens already on the card (the conversation so far) feeds only the new ones; any
  other prompt resets the state and starts at position 0. vLLM's own prefix cache and chunked prefill stay off:
  the short-conv histories can not be rewound to an earlier position.
- decode_forward: one traced step on the token vLLM hands in, at the position after the last one.
- The KV caches are the model's own (position-addressed, sized by ``--max-model-len``); vLLM's block tables are
  not used. ``get_max_tokens_all_users`` gives vLLM a pool of one context.
- Sampling without ``sample_on_device_mode`` in the tt config: vLLM's host sampler on the step's logits (one fp32
  row, read through a small tail trace). It sorts the 200K vocabulary for every token of a request with top-k or
  top-p, about 13 ms a token on the host this was measured on.
- With ``"sample_on_device_mode": "all"`` (or ``"decode_only"``) the adapter returns the token itself, through the
  plugin's device-sampling interface: a greedy step returns the head's on-device argmax (4 bytes to read); a
  sampled step is drawn here, on the host, from the same logits row, with the request's temperature, top-k, top-p
  and seed and no sort (`_sample`). Penalties, logprobs, min_p, logit_bias and structured output stay with vLLM's
  host sampler (the plugin routes them there).

Weights: the run mode's decode cache (modes.py; ``$INKLING_MODE``, ``$INKLING_CACHE``). The tt config's
``"optimizations": "performance"`` picks fast mode.

Chunked prefill (opt-in, ``INKLING_VLLM_PREFILL=<tokens>``, Inkling2ForCausalLM only): a new prompt of at least
that many tokens runs through tt/prefill_model.py in chunks instead of token by token. The process keeps the one
(PF-layout) decode cache: the prefill rebuilds each layer's plain expert tensors on the card for the layer's expert
step (tt/pf_tiles.py). The decode's L1 weight staging and the prefill's programs do not fit together: the
staging gives its L1 to the prefill and takes the same addresses back after it (Generator.suspend / resume), so
the decode traces stay (if an address does not come back, the reply's first six steps run eagerly and capture
them again: Generator.begin_live). It needs
- DRAM: about 1.05 GiB for the prefill's own tensors, 2 GiB of room for a layer's rebuilt experts and 0.26 GiB per
  1024 tokens of a pass, so a ``--max-model-len`` below the largest. With a 128 MiB trace region: passes of 8192
  tokens up to about 94K, of 4096 up to about 151K, of 2048 up to about 180K; at 196608 about 0.3 GiB is short
  (`_prefill_chunk` picks, or says how much);
- L1: a 48 KB kernel-config ring, ``"worker_l1_size": 1411136`` (the prefill's largest program leaves 57 KB of
  L1 free under the 56 KB ring, and the decode modules keep 58 KB).
"""

from __future__ import annotations

import gc
import os
import time

import torch
from loguru import logger

import ttnn


class Inkling2ForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_chunked_prefill": False,
        "supports_async_decode": False,
        # with "sample_on_device_mode" the adapter returns tokens: the head's argmax for a greedy row, its own
        # draw from the logits row for a sampled one (see the module docstring). No penalties there.
        "supports_sample_on_device": True,
        "supports_device_penalties": False,
    }

    # the tail's circular buffers must fit under the decode's L1 staging: the logits go to rows in this many
    # pieces, the first count that fits
    TAIL_PIECES = (1, 2, 4, 8, 16)

    def __init__(self, device, max_ctx, vocab, fast=False, *, vllm_config=None):
        from models.demos.inkling2_mini import modes
        from models.demos.inkling2_mini.tt import device as tt_device
        from models.demos.inkling2_mini.tt.model import VOCAB_REAL

        if vocab < VOCAB_REAL:
            raise ValueError(f"the served config has {vocab} tokens, the model {VOCAB_REAL}")
        self.device, self.max_ctx, self.vocab, self.vocab_real = device, max_ctx, vocab, VOCAB_REAL
        want = tt_device.DEFAULT_L1_BASE + (tt_device.RING_KB << 10)
        ring = ttnn._ttnn.device.GetAllocatorBaseAddress(device, ttnn.BufferType.L1) - tt_device.DEFAULT_L1_BASE
        if ring != want - tt_device.DEFAULT_L1_BASE and not (
            os.environ.get("INKLING_VLLM_PREFILL") and ring == 48 << 10
        ):
            logger.warning(
                f'decode runs slower without the bigger kernel-config ring: set "worker_l1_size": '
                f'{tt_device.worker_l1_size()} in --additional-config\'s "tt" (tt/device.py)'
            )
        cache = modes.roots(fast)[0]
        t0 = time.time()
        self.model = self._load(cache)
        logger.info(f"Inkling-2-Mini loaded from {cache} in {time.time() - t0:.0f} s, context {max_ctx}")
        self.gen = None  # tt/generator.Generator once the traces are captured
        self.tail, self.rows, self.tail_pieces = None, None, 0  # trace: TILE logits -> fp32 rows; its outputs
        self.fed = []  # the tokens the card's state holds, in order (position i = fed[i])
        self._out = None  # output dict of the last step run
        self._pos_ready = None  # position whose inputs are already written
        self._t = [0, 0.0, 0.0, None]  # decode steps of the request, s inside them, s between them, last exit
        self._pf_logits = None  # last-position logits of a chunked prefill, until the first decode step
        self.pf = self._prefill_init(cache)  # tt/prefill_model.Inkling2Prefill, or None (before any capture)
        self._rng, self._seeded = torch.Generator(), False  # the adapter's sampler: one stream per request

    def _load(self, cache):
        from models.demos.inkling2_mini.tt.model import Inkling2Model

        return Inkling2Model(self.device, root=cache, max_ctx=self.max_ctx, log=lambda s: None)

    # DRAM a chunked prefill needs in a decode process, GiB (a p100a's allocator, measured): its own tensors; free
    # room it must find before a layer's expert rebuild (the rebuilt BFP8 tensors with the pieces of the last one
    # take 1.43, in blocks of up to 0.36 that a fragmented pool does not have below about 2.0 free); its working
    # set per 1024 tokens of a pass. PF_CHUNKS: tokens per pass, the first that fits. PF_ONE: a prompt of up to this
    # many tokens goes in one pass when the largest chunk fits (every pass rebuilds each layer's experts).
    PF_GIB = (1.05, 2.0, 0.26)
    PF_CHUNKS = (8192, 4096, 2048)
    PF_ONE = 8448

    def _prefill_chunk(self):
        """(tokens per prefill pass that fit in the DRAM free now or None, GiB left over for that pass or short).
        The rebuild room only when the build's expert op can not read the PF tensors in place."""
        from models.demos.inkling2_mini.tt.prefill_model import pf_inplace_op

        v = ttnn.get_memory_view(self.device, ttnn.BufferType.DRAM)
        room = v.num_banks * v.total_bytes_free_per_bank / 2**30 - self.PF_GIB[0]
        room -= 0.0 if pf_inplace_op() else self.PF_GIB[1]
        chunk = next((c for c in self.PF_CHUNKS if self.PF_GIB[2] * c / 1024 <= room), None)
        return chunk, room - self.PF_GIB[2] * (chunk or self.PF_CHUNKS[-1]) / 1024

    def _prefill_init(self, cache):
        """The chunked prefill for prompts of at least $INKLING_VLLM_PREFILL tokens (unset or 0: off)."""
        self.pf_min = int(os.environ.get("INKLING_VLLM_PREFILL") or 0)
        if not self.pf_min:
            return None
        from models.demos.inkling2_mini.tt import device as tt_device
        from models.demos.inkling2_mini.tt.prefill_model import Inkling2Prefill

        why = None
        base = ttnn._ttnn.device.GetAllocatorBaseAddress(self.device, ttnn.BufferType.L1)
        if base > tt_device.DEFAULT_L1_BASE + (48 << 10):
            why = f'its programs need "worker_l1_size": {tt_device.worker_l1_size(48)} (a 48 KB ring) or more L1'
        chunk, spare = self._prefill_chunk()
        if why is None and chunk is None:
            why = (
                f"{-spare:.2f} GiB of DRAM short at --max-model-len {self.max_ctx} (a smaller context frees 19,584 B "
                "per token)"
            )
        if why is not None:
            logger.warning(f"chunked prefill off: {why}")
            return None
        t0 = time.time()
        self.pf_chunk = chunk  # tokens per pass; the prefill is built for one pass of PF_ONE when that fits too
        one = spare >= self.PF_GIB[2] * (self.PF_ONE - chunk) / 1024 > 0
        pf = Inkling2Prefill(self.model, root=cache, chunk=-(-self.PF_ONE // 1024) * 1024 if one else chunk)
        pf.drafter = getattr(self, "drafter", None)  # (the speculative class: the prompt's drafter context too)
        logger.info(
            f"chunked prefill on for prompts of {self.pf_min}+ tokens: passes of {chunk}"
            f"{f', one pass up to {self.PF_ONE}' if one else ''}; built in {time.time() - t0:.0f} s"
        )
        return pf

    def _make_gen(self):
        """(the generator with its traces captured, the output dicts of its step traces)."""
        from models.demos.inkling2_mini.tt.generator import Generator

        gen = Generator(self.model)
        gen.capture()
        return gen, gen.out

    def _after_tail(self, lg):
        pass

    def _reset(self):
        """A new sequence at position 0."""
        self.model.reset()

    # ------------------------------------------------------------ what vLLM's model registry looks for
    def embed_input_ids(self, input_ids):
        raise NotImplementedError("TT serving uses prefill_forward / decode_forward")

    def forward(self, input_ids, positions, **kwargs):
        raise NotImplementedError("TT serving uses prefill_forward / decode_forward")

    def compute_logits(self, hidden_states):
        raise NotImplementedError("TT serving uses prefill_forward / decode_forward")

    # ------------------------------------------------------------ setup
    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        if max_batch_size != 1 or tt_data_parallel != 1:
            raise ValueError("Inkling-2-Mini on one card serves one request at a time: --max-num-seqs 1")
        if mesh_device.get_num_devices() != 1:
            raise ValueError("Inkling-2-Mini runs on one Blackhole card (a 1x1 mesh)")
        # the per-token host work (vLLM's sampler over one 200K row) is small: one thread, not a pool
        torch.set_num_threads(int(os.environ.get("INKLING_VLLM_THREADS") or 1))
        text = getattr(hf_config, "text_config", hf_config)
        return cls(mesh_device, int(max_seq_len), int(text.vocab_size), fast=optimizations == "performance")

    @classmethod
    def get_max_tokens_all_users(cls, max_model_len=None, **kwargs):
        """vLLM's (logical) KV pool: one request of the whole context. The card's caches are the model's own."""
        return max_model_len or 131072

    def allocate_kv_cache(self, shape, dtype, num_layers):
        return None  # Inkling2Model allocated its caches for max_ctx positions when it loaded

    def _rows(self, lg, k):
        """[1, 1, 1, W] TILE -> k ROW_MAJOR pieces of whole tiles, in order."""
        tiles, out, a = lg.shape[-1] // 32, [], 0
        for j in range(k):
            b = tiles * (j + 1) // k * 32
            piece = lg if k == 1 else ttnn.slice(lg, [0, 0, 0, a], [1, 1, 1, b])
            out.append(ttnn.to_layout(piece, ttnn.ROW_MAJOR_LAYOUT))
            a = b
        return out

    def _capture_tail(self, lg, check=True):
        """Trace: TILE logits -> fp32 rows (800 KB to read instead of the 25 MB padded TILE tensor). Checked once
        against the full TILE read; None if no piece count fits or the rows are not bit-equal. check=False (in
        the middle of a sequence: no step may run): the piece count found before, unchecked."""
        for k in self.TAIL_PIECES if check else (self.tail_pieces,):
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            try:
                rows = self._rows(lg, k)
            except RuntimeError:  # circular buffers clash with L1 buffers
                ttnn.end_trace_capture(self.device, tid, cq_id=0)
                ttnn.release_trace(self.device, tid)
                continue
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            if not check:
                self.rows = rows
                return tid
            self.gen.decode(0, 0)
            ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
            want = ttnn.to_torch(lg).float().reshape(-1)
            got = torch.cat([ttnn.to_torch(t).float().reshape(-1) for t in rows])
            if torch.equal(want.view(torch.int32), got.view(torch.int32)):
                self.rows, self.tail_pieces = rows, k
                return tid
            ttnn.release_trace(self.device, tid)
        return None

    def _capture(self):
        """Compile the step and every tail variant, then capture the decode traces (once)."""
        if self.gen is not None:
            return
        t0, m = time.time(), self.model
        m.set_inputs(0, 0)
        lg = m.step()["logits"]  # the head's logits are a module buffer, the same one in the traces
        tail = lg.layout == ttnn.TILE_LAYOUT and m.head_tail is not None
        if tail:
            for k in self.TAIL_PIECES:  # every program exists before any trace does
                for t in self._rows(lg, k):
                    ttnn.deallocate(t)
        self.gen, outs = self._make_gen()
        self._lg = lg
        if tail and all(o["logits"].buffer_address() == lg.buffer_address() for o in outs):
            self.tail = self._capture_tail(lg)
        self._after_tail(lg)
        self._reset()
        self.fed, self._out, self._pos_ready = [], None, None
        v = ttnn.get_memory_view(self.device, ttnn.BufferType.TRACE)
        logger.info(
            f"decode traces captured in {time.time() - t0:.0f} s, "
            f"{v.num_banks * v.total_bytes_allocated_per_bank / 2**20:.0f} MiB of trace region, "
            f"logits row tail {f'in {self.tail_pieces} pieces' if self.tail is not None else 'off'}"
        )

    def warmup_model_prefill(self, **kwargs):
        self._capture()

    def warmup_model_decode(self, **kwargs):
        self._capture()

    def release_persistent_capture(self):
        if self.gen is None:
            return
        if self.tail is not None:
            ttnn.release_trace(self.device, self.tail)
            self.tail = None
        self.gen.release()
        self.gen = None
        if self.pf is not None:
            self.pf.release_traces()

    # ------------------------------------------------------------ one step
    def _step(self, token, pos):
        """Inputs, then the traced step (not waited for), then the next position's inputs: the host writes them
        while this step runs (the queue keeps them behind it)."""
        from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv

        m, ph = self.model, pos % FusedShortConv.PHASES
        if self.gen.live:  # after a chunked prefill: the generator runs the step and captures it (tt/generator.py)
            self._out, self._pos_ready = self.gen.step_live(token, pos), None
            return
        m.phase = ph
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[token]], dtype=torch.int32), dtype=ttnn.uint32), m.tokens
        )
        if pos != self._pos_ready:
            m.positions.update(pos)
        ttnn.execute_trace(self.device, self.gen.trace[ph], cq_id=0, blocking=False)
        self._out = self.gen.out[ph]
        self._pos_ready = pos + 1 if pos + 1 < self.max_ctx else None
        if self._pos_ready is not None:
            m.positions.update(self._pos_ready)

    def _logits(self):
        """fp32 logits [vocab of the model] of the last step."""
        if self.tail is None:
            return ttnn.to_torch(self._out["logits"]).float().reshape(-1)[: self.vocab_real]
        ttnn.execute_trace(self.device, self.tail, cq_id=0, blocking=False)
        return torch.cat([ttnn.to_torch(t).float().reshape(-1) for t in self.rows])[: self.vocab_real]

    NUCLEUS_K = 1024  # top-p candidates looked at first

    @classmethod
    def _candidates(cls, lg, t, top_k, top_p):
        """(probabilities, token ids or None = all, in order) of softmax(lg / t) cut to the top_k logits and then
        to the top_p nucleus, as vLLM's sampler cuts them, without sorting the vocabulary: the nucleus is a prefix
        of the sorted probabilities, so the NUCLEUS_K largest hold it (everything is sorted only when they do not:
        a flat distribution). The probabilities are not renormalized."""
        x, idx = lg / t, None
        if 0 < top_k < x.numel():
            x, idx = torch.topk(x, top_k)  # sorted, descending
        p = torch.softmax(x, dim=-1)
        if top_p < 1.0:
            if idx is None:
                p, idx = torch.topk(p, min(cls.NUCLEUS_K, p.numel()))
                if float(p.sum()) < top_p:
                    p, idx = torch.sort(torch.softmax(x, dim=-1), descending=True)
            keep = int((torch.cumsum(p, 0) - p < top_p).sum())  # >= 1: the top token always stays
            p, idx = p[:keep], idx[:keep]
        return p, idx

    def _sample(self, lg, t, top_k, top_p):
        """One draw from `_candidates`, by inverse CDF (torch.multinomial is slow over 200K entries)."""
        p, idx = self._candidates(lg, t, top_k, top_p)
        c = torch.cumsum(p, 0)
        i = min(int(torch.searchsorted(c, torch.rand(1, generator=self._rng) * c[-1])), p.numel() - 1)
        return int(idx[i]) if idx is not None else i

    def _output(self, sampling_params):
        """The last step's result as vLLM wants it: fp32 logits [1, 1, vocab] for its host sampler, or, when the
        plugin picked device sampling, the token [1, 1]: the head's argmax (greedy), else the adapter's draw."""
        pf_logits = self._pf_logits  # after a chunked prefill: its last position, not a step's output
        if sampling_params is None:
            lg = self._logits() if pf_logits is None else pf_logits
            if self.vocab == self.vocab_real:
                return lg.reshape(1, 1, -1)
            out = torch.full((1, 1, self.vocab), float("-inf"))  # the config's padded vocabulary
            out[0, 0, : self.vocab_real] = lg
            return out
        t = float(sampling_params.temperature[0])
        if t < 1e-5:
            if pf_logits is not None:
                return pf_logits.argmax().reshape(1, 1).to(torch.int32)
            return ttnn.to_torch(self._out["token"]).reshape(1, 1).to(torch.int32)
        self._seed(sampling_params)
        lg = self._logits() if pf_logits is None else pf_logits
        token = self._sample(lg, t, int(sampling_params.top_k[0]), float(sampling_params.top_p[0]))
        return torch.tensor([[token]], dtype=torch.int32)

    def _seed(self, sampling_params):
        """Before a request's first draw: its seed, or a fresh one."""
        if self._seeded:
            return
        seed = sampling_params.seed[0] if isinstance(sampling_params.seed, (list, tuple)) else sampling_params.seed
        if seed is None:
            self._rng.seed()
        else:
            self._rng.manual_seed(int(seed))
        self._seeded = True

    # ------------------------------------------------------------ the plugin's calls
    def prefill_forward(
        self, tokens, page_table, kv_cache, prompt_lens, start_pos=None, sampling_params=None, **kwargs
    ):
        self._capture()
        if tokens.shape[0] != 1:
            raise ValueError("one request at a time")
        if start_pos is not None and int(torch.as_tensor(start_pos).reshape(-1)[0]) != 0:
            raise ValueError("a prompt starts at position 0 (no prefix cache, no chunked prefill)")
        n = int(torch.as_tensor(prompt_lens).reshape(-1)[0])
        if not 0 < n <= self.max_ctx:
            raise ValueError(f"the prompt is {n} tokens, the context is {self.max_ctx}")
        ids = [int(t) for t in tokens[0, :n].tolist()]
        fed, self.fed = self.fed, []  # (not valid while the steps run)
        k = len(fed)
        # the conversation on the card goes on (when all of it is the prompt, the last step's output is the answer)
        if not (0 < k <= n and ids[:k] == fed and (k < n or self._out is not None)):
            self._reset()
            k = 0
        last = self._request_log()
        t0, self._seeded, self._pf_logits = time.perf_counter(), False, None
        how = "fed"
        if self.pf is not None and k == 0 and self.pf_min <= n <= self.max_ctx - 8:
            how = self._prefill_chunked(ids)
        else:
            for i in range(k, n):
                self._step(ids[i], i)
        out = self._output(sampling_params)
        self.fed = ids
        dt = time.perf_counter() - t0
        logger.info(f"prompt {n} tokens, {k} already on the card, {n - k} {how} in {dt:.2f} s{last}")
        return out

    def _prefill_chunked(self, ids):
        """The whole prompt through the chunked prefill, which hands its state to the decode step. The decode's
        L1 weight staging gives way to it: Generator.suspend() frees that L1 between two steps and resume() takes
        the same addresses again, so the decode traces stay. If an address does not come back, the traces are
        released and captured again from the reply's own first steps (Generator.begin_live)."""
        t0 = time.perf_counter()
        self.gen.suspend()
        t1 = time.perf_counter()
        self.pf.n_rebuilt = 0
        one = len(ids) <= self.PF_ONE and self.pf.chunk > self.pf_chunk
        self._pf_logits = self.pf.prefill(ids, chunk=self.pf.chunk if one else self.pf_chunk)[0].float()
        self.pf.drop_buffers()  # (nothing of the prefill stays where a decode replay writes)
        t2 = time.perf_counter()
        kept = self.gen.resume()
        if not kept:
            if self.tail is not None:  # (its row tensors go too; captured again below)
                ttnn.release_trace(self.device, self.tail)
                for t in self.rows:
                    ttnn.deallocate(t)
            self.gen.release()
            gc.collect()
            self.gen.begin_live()
            if self.tail is not None:
                self.tail = self._capture_tail(self._lg, check=False)
        self._out, self._pos_ready = None, None
        return (
            f"prefilled in chunks (decode L1 freed in {t1 - t0:.2f} s, prefill {t2 - t1:.2f} s with "
            f"{self.pf.n_rebuilt} expert rebuilds; decode traces "
            f"{'kept' if kept else 'released, the next steps capture them again'})"
        )

    def _request_log(self):
        """What the request before did (and its counters start over)."""
        (steps, inside, between, _), self._t = self._t, [0, 0.0, 0.0, None]
        last = f"; the request before: {steps} decode calls, {inside / steps * 1e3:.2f} ms each" if steps else ""
        return last + (f" + {between / (steps - 1) * 1e3:.2f} ms between two" if steps > 1 else "")

    def decode_forward(self, tokens, start_pos, page_table, kv_cache, sampling_params=None, **kwargs):
        self._capture()
        pos = int(torch.as_tensor(start_pos).reshape(-1)[0])
        if pos < 0:  # no live request in the batch
            return (
                torch.zeros(1, 1, dtype=torch.int32) if sampling_params is not None else torch.zeros(1, 1, self.vocab)
            )
        if pos != len(self.fed):
            raise RuntimeError(f"decode at position {pos}, the state on the card ends at {len(self.fed)}")
        token = int(torch.as_tensor(tokens).reshape(-1)[0])
        t0, t = time.perf_counter(), self._t
        if t[3] is not None:
            t[2] += t0 - t[3]
        fed, self.fed = self.fed, []
        self._pf_logits = None
        self._step(token, pos)
        out = self._output(sampling_params)
        fed.append(token)
        self.fed = fed
        t[3] = time.perf_counter()
        t[0], t[1] = t[0] + 1, t[1] + t[3] - t0
        return out


class Inkling2SpecForCausalLM(Inkling2ForCausalLM):
    """Inkling2ForCausalLM with speculative decode, through the plugin's model-owned drafter contract
    (vllm-tt-plugin docs/SPEC_DECODE_CONTRACT.md). Select this class in TT_MODEL_CLASS_OVERRIDES and add

        --speculative-config '{"method": "custom_class", "model": "vllm_tt_plugin.model_owned_drafter",
                               "num_speculative_tokens": 2}' --no-async-scheduling

    and a 512 MiB "trace_region_size" (the draft, verify, commit and stash traces take about 270 MiB).

    The drafter (tt/mtp.py) proposes up to 2 tokens, one verify pass (tt/verify.py) runs the model on the token
    and the drafts, and the rows the plugin accepts are committed. The tokens are those of plain steps.
      propose_draft_tokens   commits the pass before (the call carries its accepted count), then runs the draft
                             pass on the last committed token. num_valid 0 declines: the drafter covers the
                             positions below modes.spec_max_ctx(), and the cost model below verifies a draft only
                             when it is likely to pay.
      decode_forward, wide   the verify pass over [token, drafts]: VerifyOutput with the model's argmax per row.
      decode_forward, [1, 1] the plain step (SpecPlan.supports_narrow_decode), which also keeps the drafter's
                             context (stashed in groups of eight positions: TappedModel.stash).
    Greedy requests speculate, and so do sampled ones that the adapter samples itself ("sample_on_device_mode";
    temperature, top-k, top-p, seed). Sampled requests: the model's token of row i of a pass is one draw of the
    adapter's sampler on that row's logits, rows in order, and the plugin keeps draft i only when the draw is
    that very token (the capability spec_ids_are_sampled). The drafter's proposal is one fixed token d per
    position, a proposal distribution with q(d) = 1; speculative sampling accepts it with probability p(d) and
    otherwise draws from p without d, and "draw t ~ p, keep d iff t == d, else the token is t" is that rule. Each
    row's logits are the plain step's and the sampler makes one draw per token in the same order, so a seeded
    request gives the tokens of plain steps. A request vLLM's host sampler samples (penalties, logprobs, ...)
    runs plain steps: the plan offers argmax_ids only.

    When the drafter can not be built (no weights from convert.mtp, a context under SPEC_MIN_CTX, DRAM short at
    this --max-model-len) the reason is logged and every proposal declines: plain decode at plain speed.

    Cost model of a pass, in plain steps (measured on a p100a): a draft alone costs
    SPEC_COST[0], a pass with k verified drafts 1 + SPEC_COST[k]; SPEC_FAR scales both with the position (a
    plain step grows with the far attention, a pass's extra rows hardly do). A draft with top probability p is
    right about SPEC_ACC(p) of the time, scaled per request by the drafts kept so far (SPEC_PRIOR: weight of the
    prior). The row count of a pass is the one with the most (expected tokens) - (cost). Drafting may not cost a
    request more than SPEC_DEBT[0] + SPEC_DEBT[1] x (its plain steps), not counted before SPEC_DEBT[2] drafts;
    below that it runs plain steps and tries one draft every SPEC_PROBE[0] steps, doubling to SPEC_PROBE[1]
    while the probes do not gain; a probe that gains gives SPEC_PROBE[2] steps of new credit.
    """

    model_capabilities = {
        **Inkling2ForCausalLM.model_capabilities,
        "supports_spec_decode": True,
        "spec_requirements": ["device_propose"],
        # a verify of a step the adapter samples returns its draws, one per row (see "Sampled requests")
        "spec_ids_are_sampled": True,
    }

    SPEC_K = 2  # drafts per pass: verify passes of 2 and 3 rows
    SPEC_MIN_CTX = 4096  # the verify traces are captured around position 1536
    SPEC_COST = (0.35, 0.75, 1.05)
    SPEC_FAR = (0.1, 0.38)
    SPEC_PRIOR = (6.0, 1.0, 0.95)  # (weight, scale for greedy requests, for sampled ones)
    SPEC_DEBT = (1.5, 0.002, 6)
    SPEC_PROBE = (8, 32, 1.0)

    @staticmethod
    def SPEC_ACC(p):
        return 0.89 if p >= 0.9 else 0.95 * p * p

    @classmethod
    def spec_plan(cls, vllm_config, max_num_seqs, requested_k):
        from vllm_tt_plugin.spec_decode import SpecPlan, SpecReject

        if max_num_seqs != 1:
            return SpecReject(reason="Inkling-2-Mini verifies one request at a time: --max-num-seqs 1")
        if requested_k < 1:
            return SpecReject(reason="num_speculative_tokens must be 1 or 2", supported_k=(1, cls.SPEC_K))
        return SpecPlan(
            effective_k=min(int(requested_k), cls.SPEC_K),
            lanes_per_request=1,
            extra_bytes_per_seq=int(1.6 * 2**30),  # drafter weights, its head copy, the verify steps
            extra_bytes_per_token=4608,  # the drafter's context, per covered position (tt/spec_generator.py)
            accept_modes=("argmax_ids",),
            drafter_state="internal",
            supports_narrow_decode=True,
        )

    # ------------------------------------------------------------ setup
    def _load(self, cache):
        from models.demos.inkling2_mini import modes
        from models.demos.inkling2_mini.tt.mtp import DSparkDrafter, mtp_root
        from models.demos.inkling2_mini.tt.spec_generator import TappedModel, dram_free_gib, spec_fits

        self.drafter, self.spec_ctx = None, self.max_ctx
        self._draft = None  # the proposal on the card: {pos, token, drafts, k, pr}
        self._pend = None  # a verify pass whose accepted count is not known yet: {pos, rows, cand, pr, fed}
        # stash steps (TappedModel.stash): on = the group under way is stashed; pending = its positions so far
        self._stash, self._pending, self._declined, self._next = False, 0, None, 0
        self._sampled = False  # the adapter samples the request (sampling_params of its last step)
        self.vtail = {}  # verify row -> (tail trace, row pieces): the row's logits as fp32 rows (sampled passes)
        self._sp = [0, 0, 0]  # of the request: verify passes, the tokens they gave, drafts not verified
        self._begin()
        why = None
        if self.max_ctx < self.SPEC_MIN_CTX:
            why = f"it needs --max-model-len >= {self.SPEC_MIN_CTX}"
        elif not (mtp_root() / "manifest.json").is_file():
            why = f"no drafter weights in {mtp_root()} (python -m models.demos.inkling2_mini.convert.mtp)"
        if why is not None:
            logger.warning(f"speculative decode off: {why}")
            return super()._load(cache)
        m = TappedModel(self.device, max_ctx=self.max_ctx, log=lambda s: None, root=cache)
        ctx = min(self.max_ctx, max(modes.spec_max_ctx(), 2048))  # (not below the verify traces' positions)
        free = dram_free_gib(self.device)
        if not spec_fits(free, ctx):
            logger.warning(
                f"speculative decode off: {free:.2f} GiB DRAM free beside the model at --max-model-len "
                f"{self.max_ctx}, too little for the drafter and a verify step"
            )
            m.tapping = False  # the plain model from here on
            return m
        d = self.drafter = m.drafter = DSparkDrafter(m, max_ctx=ctx, k=self.SPEC_K)
        d.head = d.make_head(ttnn.bfloat4_b)  # the drafts' own head copy (BFP4: 3.3 ms per draft pass)
        d.top_probs = True
        self.spec_ctx = ctx
        return m

    def _make_gen(self):
        if self.drafter is None:
            return super()._make_gen()
        from models.demos.inkling2_mini.tt.spec_generator import SpecGenerator, dram_free_gib, verify_rows
        from models.demos.inkling2_mini.tt.verify import VerifyStep

        free = dram_free_gib(self.device)
        rows = verify_rows(self.device, self.max_ctx)  # 3-row passes need 0.28 GiB more than 2-row ones
        gen = SpecGenerator(self.model, self.drafter, verify=[VerifyStep(self.model, r) for r in rows], stash=True)
        ht = self.model.head_tail

        def row_tails():  # the tails of the verify rows' logits are compiled before any capture, as every program
            for x in ht.rows_tiles[1 : max(rows)] if ht is not None else []:
                for k in self.TAIL_PIECES:
                    try:
                        for t in self._rows(x, k):
                            ttnn.deallocate(t)
                    except RuntimeError:  # circular buffers clash with L1 buffers (as in _capture_tail)
                        pass

        gen.capture_rows(1536, before_capture=row_tails)
        logger.info(
            f"speculative decode on (greedy requests) below position {self.spec_ctx}: passes of up to {max(rows)} "
            f"rows, {free:.2f} GiB DRAM free before the verify steps, {dram_free_gib(self.device):.2f} now"
        )
        return gen, gen.out + (gen.sout or []) + (gen.qout or [])

    def _after_tail(self, lg):
        """self.vtail: the tail of each verify row's logits (row 0 is the plain step's buffer and tail). The rows
        of a pass are the fused head's row buffers in every verify trace; else no sampled passes."""
        if self.drafter is None or self.tail is None:
            return
        n, same = max(self.gen.verifies), lambda a, b: a.buffer_address() == b.buffer_address()  # noqa: E731
        tiles = self.model.head_tail.rows_tiles[:n]
        if len(tiles) < n or not same(tiles[0], lg):
            return
        if not all(same(x, t) for _, o in self.gen.vtrace.values() for x, t in zip(o["logits"], tiles)):
            return
        vt = {0: (self.tail, self.rows)}
        for r in range(1, n):
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            rows = self._rows(tiles[r], self.tail_pieces)
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            vt[r] = (tid, rows)
            ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
            want = ttnn.to_torch(tiles[r]).float().reshape(-1)
            got = torch.cat([ttnn.to_torch(t).float().reshape(-1) for t in rows])
            if not torch.equal(want.view(torch.int32), got.view(torch.int32)):
                for t, _ in list(vt.values())[1:]:
                    ttnn.release_trace(self.device, t)
                return
        self.vtail = vt
        logger.info(f"sampled requests speculate too: logits tails for {n} verify rows")

    def release_persistent_capture(self):
        if self.gen is not None:
            for r, (tid, _) in self.vtail.items():
                if r:
                    ttnn.release_trace(self.device, tid)
            self.vtail = {}
        super().release_persistent_capture()

    def _reset(self):
        self.model.reset()
        self._stash, self._pending, self._next = False, 0, 0

    def release_request(self, slot):
        self._draft = None  # the proposal after a request's last step is never verified

    # ------------------------------------------------------------ the drafting account of a request
    def _begin(self):
        self._bal, self._steps, self._n, self._mark = 0.0, 0, 0, None
        self._gap, self._wait = self.SPEC_PROBE[0], 0
        self._hit, self._exp = self.SPEC_PRIOR[0] * self.SPEC_PRIOR[2 if self._sampled else 1], self.SPEC_PRIOR[0]

    def _room(self):
        """Steps the account may still lose before drafting stops."""
        return self._bal + self.SPEC_DEBT[0] + self.SPEC_DEBT[1] * self._steps

    def _ready(self):
        if self._next >= self.spec_ctx:  # the drafter does not cover the next position: plain steps
            return False
        return self._room() >= 0 or self._n < self.SPEC_DEBT[2] or self._wait <= 0

    def _acc(self, p):
        return min(0.97, self.SPEC_ACC(p) * self._hit / self._exp)

    def _cost(self, pos):
        x = pos / 131072
        return [c * (1 + self.SPEC_FAR[0] * x) / (1 + self.SPEC_FAR[1] * x) for c in self.SPEC_COST]

    def _note(self, k, a, pr, pos):
        """One draft at pos done: k of its tokens verified (0: none), the first a of them kept."""
        v = a - self._cost(pos)[k]
        probe = self._room() < 0 and self._n >= self.SPEC_DEBT[2]
        self._bal, self._n = self._bal + v, self._n + 1
        if probe and v > 0:  # new credit; _mark: the account at which this run has gained a step
            self._bal = max(self._bal, self._bal - self._room() + self.SPEC_PROBE[2])
            self._mark = self._bal + 1.0
        elif self._mark is not None and self._bal >= self._mark:
            self._gap, self._mark = self.SPEC_PROBE[0], None
        if self._room() < 0 and self._n >= self.SPEC_DEBT[2]:  # stopped (again): the next probe after the gap
            if probe or self._mark is not None:
                self._gap = min(2 * self._gap, self.SPEC_PROBE[1])
            self._wait, self._mark = self._gap, None
        for i in range(min(k, a + 1)):
            self._exp += self.SPEC_ACC(pr[i])
            self._hit += i < a
        if k == 0:
            self._sp[2] += 1

    def _prefill_chunked(self, ids):
        """As Inkling2ForCausalLM._prefill_chunked; the prefill also writes the drafter's context of the prompt
        (Inkling2Prefill.drafter: DSparkDrafter.commit_chunk) while the prompt lies inside the drafter's positions."""
        d = self.drafter
        self.pf.drafter = d if d is not None and len(ids) + self.pf.chunk <= d.S else None
        how = super()._prefill_chunked(ids)
        self._stash, self._pending, self._next = False, 0, len(ids)
        return how

    def _request_log(self):
        (passes, tokens, declined), self._sp, last = self._sp, [0, 0, 0], super()._request_log()
        if last and (passes or declined):
            last += f"; {passes} verify passes gave {tokens} tokens, {declined} drafts not verified"
        return last

    # ------------------------------------------------------------ steps
    def _seq(self, pos, n=1):
        """The step or pass at pos feeds n positions. A position that does not follow the last one drops the stash
        group under way: its slots belong to the other sequence."""
        if pos != self._next:
            self._stash, self._pending = False, 0
        self._next = pos + n

    def _flush(self):
        """Commit the stashed positions of a group cut short (before a draft, or a committing step)."""
        if self._pending:
            ttnn.execute_trace(self.device, self.gen.ftrace[self._pending], cq_id=0, blocking=False)
        self._stash, self._pending = False, 0

    def _positions(self, pos):
        self.model.positions.update(pos)
        if pos < self.spec_ctx:  # the step also commits the drafter's context row
            self.drafter.set_commit_pos(pos)

    def _step(self, token, pos):
        """The plain step, as Inkling2ForCausalLM._step, in the form the drafter needs at this position."""
        if self.drafter is None:
            return super()._step(token, pos)
        from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv

        m, g, ph = self.model, self.gen, pos % FusedShortConv.PHASES
        m.phase = ph
        tr, outs = g.trace, g.out
        self._seq(pos)
        if pos >= self.spec_ctx:  # past the drafter's positions: the step that leaves it alone
            tr, outs = g.qtrace, g.qout
            self._stash, self._pending = False, 0
        elif g.strace:
            # a plain step that is not the fallback of a draft is one of a run (prompt feed, drafting off): it
            # stashes its drafter taps, from the next group start on
            if self._declined == pos:
                self._flush()
            elif self._pending == 0:
                self._stash = pos % len(g.strace) == 0
                if self._stash:
                    self.drafter.set_stash_pos(pos)
            if self._stash:
                slot = pos % len(g.strace)
                tr, outs = g.strace[slot - ph :], g.sout[slot - ph :]  # [ph] = this slot's step
                self._pending = (slot + 1) % len(g.strace)  # the last slot's step commits the group
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[token]], dtype=torch.int32), dtype=ttnn.uint32), m.tokens
        )
        if pos != self._pos_ready:
            self._positions(pos)
        ttnn.execute_trace(self.device, tr[ph], cq_id=0, blocking=False)
        self._out = outs[ph]
        self._steps, self._wait = self._steps + 1, self._wait - 1
        self._pos_ready = pos + 1 if pos + 1 < self.max_ctx else None
        if self._pos_ready is not None:
            self._positions(self._pos_ready)

    def _settle(self, counts):
        """Commit the verify pass before, of which the plugin accepted counts[0] tokens (rows fed)."""
        p = self._pend
        if p is None:
            return
        if counts is None:
            raise RuntimeError("a verify pass is not committed and the call carries no accepted_counts")
        acc, rows, pos = int(torch.as_tensor(counts).reshape(-1)[0]), p["rows"], p["pos"]
        if not 1 <= acc <= rows:
            raise RuntimeError(f"accepted count {acc} for a verify pass of {rows} rows")
        ttnn.execute_trace(self.device, self.gen.ctrace[(rows, pos % 4, acc)], cq_id=0, blocking=False)
        self._pend, self.fed, self._next = None, p["fed"] + p["cand"][:acc], pos + acc
        self._note(rows - 1, acc - 1, p["pr"], pos)
        self._sp[0], self._sp[1] = self._sp[0] + 1, self._sp[1] + acc

    def _unused(self, pos):
        """The step at pos runs without the proposal made for it (its drafts were not taken)."""
        d, self._draft = self._draft, None
        if d is not None and d["pos"] == pos and d["k"]:
            self._note(0, 0, d["pr"], pos)
            self._declined = pos

    # ------------------------------------------------------------ the plugin's calls
    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, start_pos=None, sampling_params=None, **kw):
        self._draft, self._pend = None, None  # (a pass never committed: self.fed is empty, the prompt starts over)
        self._sampled = sampling_params is not None and float(sampling_params.temperature[0]) >= 1e-5
        self._begin()
        return super().prefill_forward(tokens, page_table, kv_cache, prompt_lens, start_pos, sampling_params, **kw)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,
        kv_cache,
        sampling_params=None,
        spec_mode=None,
        num_valid_drafts=None,
        accepted_counts=None,
        **kwargs,
    ):
        if spec_mode is not None:
            return self._verify(
                torch.as_tensor(tokens), torch.as_tensor(start_pos), num_valid_drafts, spec_mode, sampling_params
            )
        pos = int(torch.as_tensor(start_pos).reshape(-1)[0])
        if pos >= 0 and self.drafter is not None:
            self._settle(accepted_counts)
            self._unused(pos)
            self._sampled = sampling_params is not None and float(sampling_params.temperature[0]) >= 1e-5
        return super().decode_forward(tokens, start_pos, page_table, kv_cache, sampling_params, **kwargs)

    def _vrow_logits(self, r):
        return torch.cat([ttnn.to_torch(t).float().reshape(-1) for t in self.vtail[r][1]])[: self.vocab_real]

    def _verify(self, tokens, start_pos, num_valid_drafts, spec_mode, sampling_params=None):
        """tokens [1, 1+K]: the token to feed and its drafts. Returns the model's token after each of the
        1 + num_valid_drafts rows (the argmax; for a request the adapter samples, its draws up to the first one
        that is not the draft); the rows are committed when the plugin says how many it accepted."""
        from vllm_tt_plugin.spec_decode import VerifyOutput

        if spec_mode != "argmax_ids":
            raise ValueError(f"spec_mode {spec_mode!r}: this adapter verifies argmax_ids")
        self._capture()
        ids = torch.zeros(1, tokens.shape[1], dtype=torch.int32)
        pos = int(start_pos.reshape(1, -1)[0, 0])
        if pos < 0:  # no live request in the batch
            return VerifyOutput(spec_mode=spec_mode, argmax_ids=ids)
        if self._pend is not None:
            raise RuntimeError("a verify pass before this one is not committed")
        if pos != len(self.fed):
            raise RuntimeError(f"verify at position {pos}, the state on the card ends at {len(self.fed)}")
        t0, t = time.perf_counter(), self._t
        if t[3] is not None:
            t[2] += t0 - t[3]
        n, token = int(torch.as_tensor(num_valid_drafts).reshape(-1)[0]), int(tokens[0, 0])
        temp = float(sampling_params.temperature[0]) if sampling_params is not None else 0.0
        self._sampled = temp >= 1e-5
        if self._sampled:
            self._seed(sampling_params)
            top_k, top_p = int(sampling_params.top_k[0]), float(sampling_params.top_p[0])
        if n == 0 or self.drafter is None:
            # nothing to verify (the step after a pass that committed several tokens, with no new draft): a plain
            # step in a verify's clothes
            self._unused(pos)
            fed, self.fed = self.fed, []
            self._step(token, pos)
            if self._sampled:
                ids[0, 0] = self._sample(self._logits(), temp, top_k, top_p)
            else:
                ids[0, 0] = int(ttnn.to_torch(self._out["token"]).reshape(-1)[0])
            fed.append(token)
            self.fed = fed
        else:
            d, self._draft = self._draft, None
            cand = [token] + [int(x) for x in tokens[0, 1 : 1 + n].tolist()]
            if d is None or d["pos"] != pos or d["token"] != token or d["drafts"][:n] != cand[1:]:
                raise RuntimeError(f"the candidate block at position {pos} is not the proposal on the card")
            g, rows = self.gen, n + 1
            # the pass reads its tokens on the card: the draft's anchor and its drafts (SpecGenerator._feed)
            ttnn.execute_trace(self.device, g.vtrace[(rows, pos % 4, g._shared(pos, rows))][0], cq_id=0, blocking=False)
            if not self._sampled:
                ids[0, :rows] = ttnn.to_torch(g.vtok[rows]).reshape(-1)[:rows].to(torch.int32)
            else:
                # row r's token = one draw on row r's logits, rows in order, while the draws are the drafts (the
                # plugin's walk reads no column after the first miss). The tails run one row ahead of the draws.
                if rows > len(self.vtail):
                    raise RuntimeError(f"a sampled pass of {rows} rows, logits tails for {len(self.vtail)}")
                for r in range(min(2, rows)):
                    ttnn.execute_trace(self.device, self.vtail[r][0], cq_id=0, blocking=False)
                a = 0
                while True:
                    ids[0, a] = self._sample(self._vrow_logits(a), temp, top_k, top_p)
                    if a == n or int(ids[0, a]) != cand[a + 1]:
                        break
                    a += 1
                    if a + 1 < rows:
                        ttnn.execute_trace(self.device, self.vtail[a + 1][0], cq_id=0, blocking=False)
            self._pend = {"pos": pos, "rows": rows, "cand": cand, "pr": d["pr"], "fed": self.fed}
            self.fed, self._out = [], None  # (until the commit)
        t[3] = time.perf_counter()
        t[0], t[1] = t[0] + 1, t[1] + t[3] - t0
        return VerifyOutput(spec_mode=spec_mode, argmax_ids=ids)

    def propose_draft_tokens(self, num_drafts, committed_tokens, committed_positions, accepted_counts, hidden=None):
        """After every step: commit a verify pass, then draft on the last committed token (the next one to feed)."""
        from vllm_tt_plugin.spec_decode import DraftOutput

        K, rows_b = int(num_drafts), int(committed_tokens.shape[0])
        ids, valid = torch.zeros(rows_b, K, dtype=torch.int32), torch.zeros(rows_b, dtype=torch.int32)
        no = DraftOutput(draft_token_ids=ids, num_valid=valid)
        acc = int(torch.as_tensor(accepted_counts).reshape(-1)[0])
        pos = int(committed_positions[0, acc - 1]) if acc >= 1 else -1
        if pos < 0 or self.drafter is None:  # no live request, or no drafter
            return no
        self._settle(accepted_counts)
        self._draft = None
        if pos != len(self.fed):
            raise RuntimeError(f"proposal for position {pos}, the state on the card ends at {len(self.fed)}")
        g, d, token = self.gen, self.drafter, int(committed_tokens[0, acc - 1])
        kmax = min(max(g.verifies) - 1, K, self.max_ctx - 1 - pos)
        if self._sampled:  # a sampled pass reads each row's logits through its tail
            kmax = min(kmax, len(self.vtail) - 1)
        if kmax < 1 or pos < 1 or pos >= self.spec_ctx or not self._ready():
            return no
        t0 = time.perf_counter()
        self._seq(pos, 0)
        self._flush()
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor([[token]], dtype=torch.int32), dtype=ttnn.uint32), d.anchor
        )
        d.set_draft_pos(pos)
        ttnn.execute_trace(self.device, d.trace, cq_id=0, blocking=False)
        for v in g.verifies.values():
            v.set_positions(pos)  # host writes while the draft runs
        pr = ttnn.to_torch(d.probs).float()[0, 0, 0, :kmax].tolist()
        a1 = self._acc(pr[0])
        exp = [1.0, 1.0 + a1] + ([1.0 + a1 + a1 * self._acc(pr[1])] if kmax > 1 else [])
        cost = self._cost(pos)
        k = max(range(len(exp)), key=lambda i: exp[i] - cost[i])
        self._pos_ready = None  # the verify steps wrote their own position inputs
        drafts = [int(x) for x in ttnn.to_torch(d.tokens).reshape(-1)[:k].tolist()] if k else []
        if any(not 0 <= x < self.vocab for x in drafts):
            k, drafts = 0, []
        self._draft = {"pos": pos, "token": token, "drafts": drafts, "k": k, "pr": pr}
        if self._t[3] is not None:  # (the draft is time between two decode calls: count it inside them)
            self._t[1], self._t[3] = self._t[1] + time.perf_counter() - t0, self._t[3] + time.perf_counter() - t0
        if k == 0:
            self._note(0, 0, pr, pos)
            self._declined = pos
            return no
        ids[0, :k] = torch.tensor(drafts, dtype=torch.int32)
        valid[0] = k
        return DraftOutput(draft_token_ids=ids, num_valid=valid)
