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
- Sampling is vLLM's host sampler on the step's logits (one fp32 row, read through a small tail trace). With
  ``"sample_on_device_mode": "all"`` (or ``"decode_only"``) in the tt config, greedy steps return the head's
  on-device argmax instead (4 bytes); steps with a temperature, a penalty or logprobs still sample on the host.

Weights: the run mode's decode cache (modes.py; ``$INKLING_MODE``, ``$INKLING_CACHE``). The tt config's
``"optimizations": "performance"`` picks fast mode.
"""

from __future__ import annotations

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
        # greedy steps only: the argmax the head computes anyway. A sampled row has no device sampler
        # (max_device_top_k 0 sends it to the host), and neither has a penalty.
        "supports_sample_on_device": True,
        "max_device_top_k": 0,
        "supports_device_penalties": False,
    }

    # the tail's circular buffers must fit under the decode's L1 staging: the logits go to rows in this many
    # pieces, the first count that fits
    TAIL_PIECES = (1, 2, 4, 8, 16)

    def __init__(self, device, max_ctx, vocab, fast=False, *, vllm_config=None):
        from models.demos.inkling2_mini import modes
        from models.demos.inkling2_mini.tt import device as tt_device
        from models.demos.inkling2_mini.tt.model import VOCAB_REAL, Inkling2Model

        if vocab < VOCAB_REAL:
            raise ValueError(f"the served config has {vocab} tokens, the model {VOCAB_REAL}")
        self.device, self.max_ctx, self.vocab, self.vocab_real = device, max_ctx, vocab, VOCAB_REAL
        want = tt_device.DEFAULT_L1_BASE + (tt_device.RING_KB << 10)
        if ttnn._ttnn.device.GetAllocatorBaseAddress(device, ttnn.BufferType.L1) != want:
            logger.warning(
                f'decode runs slower without the bigger kernel-config ring: set "worker_l1_size": '
                f'{tt_device.worker_l1_size()} in --additional-config\'s "tt" (tt/device.py)'
            )
        cache = modes.roots(fast)[0]
        t0 = time.time()
        self.model = Inkling2Model(device, root=cache, max_ctx=max_ctx, log=lambda s: None)
        logger.info(f"Inkling-2-Mini loaded from {cache} in {time.time() - t0:.0f} s, context {max_ctx}")
        self.gen = None  # tt/generator.Generator once the traces are captured
        self.tail, self.rows = None, None  # trace: the head's TILE logits -> fp32 rows; its output pieces
        self.fed = []  # the tokens the card's state holds, in order (position i = fed[i])
        self._out = None  # output dict of the last step run
        self._pos_ready = None  # position whose inputs are already written
        self._t = [0, 0.0, 0.0, None]  # decode steps of the request, s inside them, s between them, last exit

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

    def _capture_tail(self, lg):
        """Trace: TILE logits -> fp32 rows (800 KB to read instead of the 25 MB padded TILE tensor). Checked once
        against the full TILE read; None if no piece count fits or the rows are not bit-equal."""
        for k in self.TAIL_PIECES:
            tid = ttnn.begin_trace_capture(self.device, cq_id=0)
            try:
                rows = self._rows(lg, k)
            except RuntimeError:  # circular buffers clash with L1 buffers
                ttnn.end_trace_capture(self.device, tid, cq_id=0)
                ttnn.release_trace(self.device, tid)
                continue
            ttnn.end_trace_capture(self.device, tid, cq_id=0)
            self.gen.decode(0, 0)
            ttnn.execute_trace(self.device, tid, cq_id=0, blocking=False)
            want = ttnn.to_torch(lg).float().reshape(-1)
            got = torch.cat([ttnn.to_torch(t).float().reshape(-1) for t in rows])
            if torch.equal(want.view(torch.int32), got.view(torch.int32)):
                self.rows = rows
                return tid
            ttnn.release_trace(self.device, tid)
        return None

    def _capture(self):
        """Compile the step and every tail variant, then capture the decode traces (once)."""
        if self.gen is not None:
            return
        from models.demos.inkling2_mini.tt.generator import Generator

        t0, m = time.time(), self.model
        m.set_inputs(0, 0)
        lg = m.step()["logits"]  # the head's logits are a module buffer, the same one in the traces
        tail = lg.layout == ttnn.TILE_LAYOUT and m.head_tail is not None
        if tail:
            for k in self.TAIL_PIECES:  # every program exists before any trace does
                for t in self._rows(lg, k):
                    ttnn.deallocate(t)
        gen = Generator(m)
        gen.capture()
        self.gen = gen
        if tail and all(o["logits"].buffer_address() == lg.buffer_address() for o in gen.out):
            self.tail = self._capture_tail(lg)
        m.reset()
        self.fed, self._out, self._pos_ready = [], None, None
        v = ttnn.get_memory_view(self.device, ttnn.BufferType.TRACE)
        logger.info(
            f"decode traces captured in {time.time() - t0:.0f} s, "
            f"{v.num_banks * v.total_bytes_allocated_per_bank / 2**20:.0f} MiB of trace region, "
            f"logits row tail {'on' if self.tail is not None else 'off'}"
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

    # ------------------------------------------------------------ one step
    def _step(self, token, pos):
        """Inputs, then the traced step (not waited for), then the next position's inputs: the host writes them
        while this step runs (the queue keeps them behind it)."""
        from models.demos.inkling2_mini.tt.fused.conv import FusedShortConv

        m, ph = self.model, pos % FusedShortConv.PHASES
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

    def _output(self, sampling_params):
        """The last step's result as vLLM wants it: the argmax [1, 1] when the plugin picked device sampling
        (greedy rows only), else fp32 logits [1, 1, vocab] for the host sampler."""
        if sampling_params is not None:
            if any(float(t) != 0.0 for t in sampling_params.temperature):
                raise ValueError("device sampling is the argmax: a sampled request samples on the host")
            return ttnn.to_torch(self._out["token"]).reshape(1, 1).to(torch.int32)
        if self.tail is None:
            lg = ttnn.to_torch(self._out["logits"]).float().reshape(-1)
        else:
            ttnn.execute_trace(self.device, self.tail, cq_id=0, blocking=False)
            lg = torch.cat([ttnn.to_torch(t).float().reshape(-1) for t in self.rows])
        if self.vocab == self.vocab_real:
            return lg[: self.vocab].reshape(1, 1, -1)
        out = torch.full((1, 1, self.vocab), float("-inf"))  # the config's padded vocabulary
        out[0, 0, : self.vocab_real] = lg[: self.vocab_real]
        return out

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
        if not (0 < k <= n and ids[:k] == fed):  # not the conversation on the card: a new sequence
            self.model.reset()
            k = 0
        t0 = time.perf_counter()
        for i in range(k, n):
            self._step(ids[i], i)
        out = self._output(sampling_params)
        self.fed = ids
        dt = time.perf_counter() - t0
        steps, inside, between, _ = self._t
        last = f"; the request before: {steps} decode steps, {inside / steps * 1e3:.2f} ms each" if steps else ""
        last += f" + {between / (steps - 1) * 1e3:.2f} ms between two" if steps > 1 else ""
        logger.info(f"prompt {n} tokens, {k} already on the card, {n - k} fed in {dt:.2f} s{last}")
        self._t = [0, 0.0, 0.0, None]
        return out

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
        self._step(token, pos)
        out = self._output(sampling_params)
        fed.append(token)
        self.fed = fed
        t[3] = time.perf_counter()
        t[0], t[1] = t[0] + 1, t[1] + t[3] - t0
        return out
