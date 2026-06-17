# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM / tt-inference-server Generator wrapper for NemotronH-30B (TP=4 QB).

Maps the HF architecture class name ``NemotronHForCausalLM`` to this TT
generator so the tt-inference-server can discover and serve the model.

Architecture
------------
NemotronH-30B is a 52-layer hybrid model (MEMEM*EMEMEM* pattern) mixing:
  - 23 Mamba2 SSM layers — stateful, no KV cache
  - 23 sparse MoE layers — stateless
  - 6 Dense-attention layers — paged KV cache

Because vLLM's KV-cache allocator is built for pure-attention models, we
bypass it entirely and manage ALL state (SSM states + KV caches) inside a
``DecoderState`` object.  This means:

  - Paged attention with prefix caching is NOT supported (B=1 only).
  - The 6 KV caches live in our ``DecoderState``; vLLM never allocates them.
  - SSM states also live in ``DecoderState``, invisible to vLLM.

Traced decode
-------------
A single-token decode trace is captured during ``warmup_model_decode`` and
replayed for every ``decode_forward`` call.  The trace gives ~18 tok/s on
TP=4 QB.  ``prefill_forward`` is always eager (no trace).

Batch size
----------
Currently B=1 only.  Multi-sequence batching would require separate
``DecoderState`` objects and a batched forward path (future work).

vLLM integration
----------------
Register this class in the tt-inference-server model registry, e.g.::

    # vllm_tt_backend/model_classes.py
    from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.nemotron_generator import NemotronHForCausalLM

    MODEL_ARCHITECTURES = {
        "NemotronHForCausalLM": NemotronHForCausalLM,
        ...
    }
"""
from __future__ import annotations

import time
from typing import Optional

import torch

import ttnn
from models.tt_transformers.tt.generator import Generator

from .kv_cache import DEFAULT_BLOCK_SIZE, DEFAULT_MAX_SEQ_LEN, MODEL_MAX_SEQ_LEN, DecoderState, allocate_decoder_state
from .model import WeightCache, nemotron_h_forward_stateful
from .tp import _R, _host_rep

# ---------------------------------------------------------------------------
# Minimal model-args proxy — satisfies any Generator base that calls
# self.model_args[0].<attr> without crashing when we don't use those paths.
# ---------------------------------------------------------------------------


class _NemotronHArgs:
    """Proxy object returned by self.model_args[0] inside Generator base."""

    def __init__(self, max_seq_len: int):
        self.max_seq_len = max_seq_len
        self.max_prefill_chunk_size = max_seq_len
        self.is_multimodal = False

    def get_warmup_prefill_supported_seq_lens(self):
        # We override warmup_model_prefill entirely; this should never be called.
        return [128]

    @property
    def trace_prefill_supported_seq_lens(self):
        return []


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------


def _load_tokenizer():
    from .generate import _load_tokenizer as _lt

    return _lt()


# ---------------------------------------------------------------------------
# NemotronHForCausalLM — the Generator subclass
# ---------------------------------------------------------------------------


class NemotronHForCausalLM(Generator):
    """vLLM Generator wrapper for NemotronH-30B (TP=4 QB, B=1).

    The HF architecture name ``NemotronHForCausalLM`` must match the class
    name for vLLM auto-discovery.
    """

    model_capabilities = {
        "supports_prefix_caching": False,  # SSM state is not prefix-cacheable
        "supports_async_decode": False,  # traced sync decode only
        "supports_sample_on_device": False,  # sampling done on CPU
        "max_batch_size": 1,
    }

    # -----------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------

    def __init__(
        self,
        mesh_device,
        wc: WeightCache,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        block_size: int = DEFAULT_BLOCK_SIZE,
        tokenizer=None,
    ):
        """Direct constructor — use ``initialize_vllm_model`` for vLLM."""
        # Satisfy Generator.__init__ with a dummy model list (B=1, no DP).
        # We override every method that touches self.model[i], so the dummy
        # is never used at runtime.
        dummy_model_args = [_NemotronHArgs(max_seq_len)]
        super().__init__(
            model=[None],
            model_args=dummy_model_args,
            mesh_device=mesh_device,
            tokenizer=tokenizer or _load_tokenizer(),
        )

        self._wc = wc
        self._max_seq_len = max_seq_len
        self._block_size = block_size
        self._state: Optional[DecoderState] = None
        self._trace_id: Optional[int] = None
        self._trace_logits_tt = None  # persistent output tensor captured in trace
        self._ids_tt = None  # persistent input token tensor (updated between replays)
        self._prefill_pos: int = 0  # tracks current sequence position across calls
        self._decode_pos: int = 0

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size: int = 1,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        n_layers: int = None,
        **kwargs,
    ) -> "NemotronHForCausalLM":
        """Factory classmethod called by the tt-inference-server to load the model.

        Args:
            hf_config:    HuggingFace config object (used for logging only).
            mesh_device:  Already-opened TTNN MeshDevice (TP=4 QB).
            max_batch_size: Must be 1 (NemotronH current limitation).
            max_seq_len:  Maximum sequence length (prompt + generated tokens).
                          Capped at MODEL_MAX_SEQ_LEN (262144).
            n_layers:     Optional layer count override (for unit tests).
        """
        if max_batch_size > 1:
            raise ValueError(
                f"NemotronHForCausalLM currently supports only batch_size=1 "
                f"(requested {max_batch_size}). Multi-sequence batching is future work."
            )
        if max_seq_len > MODEL_MAX_SEQ_LEN:
            raise ValueError(f"max_seq_len={max_seq_len} exceeds MODEL_MAX_SEQ_LEN={MODEL_MAX_SEQ_LEN}.")

        print(
            f"[NemotronHForCausalLM] Loading weight cache " f"(max_seq_len={max_seq_len})...",
            flush=True,
        )
        wc = WeightCache()
        tok = _load_tokenizer()
        instance = cls(mesh_device, wc, max_seq_len=max_seq_len, tokenizer=tok)
        print("[NemotronHForCausalLM] Ready.", flush=True)
        return instance

    # -----------------------------------------------------------------
    # KV cache / state allocation (called by vLLM after initialize_vllm_model)
    # -----------------------------------------------------------------

    def allocate_kv_cache(self, max_batch_size: int = 1, max_seq_len: int = None, **kwargs):
        """Allocate the DecoderState (SSM states + KV caches).

        vLLM calls this after ``initialize_vllm_model``.  We allocate our
        custom ``DecoderState`` (not vLLM's paged cache).

        Returns a handle that vLLM stores and passes back to warmup/forward
        calls — we ignore the handle and track state internally.
        """
        if max_seq_len is None:
            max_seq_len = self._max_seq_len
        self._state = allocate_decoder_state(
            self.mesh_device,
            B=1,
            max_seq_len=max_seq_len,
            block_size=self._block_size,
        )
        # Allocate the persistent token tensor (updated between decode steps).
        cpu_tok = torch.tensor([[0]], dtype=torch.int32)
        self._ids_tt = ttnn.from_torch(
            cpu_tok,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=_R(self.mesh_device),
        )
        self._prefill_pos = 0
        self._decode_pos = 0
        # Return a minimal handle (1-element list matching Generator convention).
        return [self._state]

    # -----------------------------------------------------------------
    # Warmup
    # -----------------------------------------------------------------

    def warmup_model_prefill(self, kv_cache, enable_trace: bool = False, can_sample_on_device: bool = False, **kwargs):
        """Run one prefill step to prime JIT kernel compilation.

        Overrides Generator.warmup_model_prefill to skip the tt_transformers
        transformer-specific warmup sweep and run a single NemotronH forward.
        """
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True

        if self._state is None:
            self.allocate_kv_cache()

        print("[NemotronHForCausalLM] Warming up prefill (compiling kernels)...", flush=True)
        t0 = time.time()

        # One prefill forward at position 0 (dummy token = 0).
        self._update_ids(0)
        self._update_pos(0)
        nemotron_h_forward_stateful(self.mesh_device, self._ids_tt, self._wc, self._state, cpu_gate=True)
        ttnn.synchronize_device(self.mesh_device)
        self._state.advance()
        self._prefill_pos = 1

        print(f"[NemotronHForCausalLM] Prefill warmup done ({time.time() - t0:.1f}s).", flush=True)

    def warmup_model_decode(self, kv_cache, enable_trace: bool = True, **kwargs):
        """Capture the decode trace for fast single-token generation.

        Called by vLLM after warmup_model_prefill.  Sets up the traced
        decode path (``ttnn.begin_trace_capture`` / ``ttnn.end_trace_capture``)
        so subsequent ``decode_forward`` calls use ``ttnn.execute_trace``.
        """
        if self._trace_id is not None:
            return  # already captured

        if self._state is None:
            self.allocate_kv_cache()

        print("[NemotronHForCausalLM] Capturing decode trace...", flush=True)
        t0 = time.time()

        # Warmup pass outside trace (primes caches for cpu_gate=False path).
        self._update_ids(0)
        self._update_pos(self._decode_pos)
        nemotron_h_forward_stateful(self.mesh_device, self._ids_tt, self._wc, self._state, cpu_gate=False)
        ttnn.synchronize_device(self.mesh_device)

        # Trace capture.
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._trace_logits_tt = nemotron_h_forward_stateful(
            self.mesh_device, self._ids_tt, self._wc, self._state, cpu_gate=False
        )
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self._trace_id = trace_id

        print(
            f"[NemotronHForCausalLM] Decode trace captured ({time.time() - t0:.1f}s).",
            flush=True,
        )

    # -----------------------------------------------------------------
    # Forward passes
    # -----------------------------------------------------------------

    def prefill_forward(
        self,
        tokens: torch.Tensor,  # [B, S] int64 — prompt token ids
        current_pos: torch.Tensor = None,  # [B] int64 starting position
        kv_cache=None,
        prompt_lens: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Process a prompt sequence token-by-token (S=1 decode steps).

        vLLM calls this once per request at the start of a new sequence.
        Returns logits for the LAST prompt position [B, 1, vocab].

        Note: NemotronH processes S=1 at a time; S>1 prompts are processed
        sequentially (no batched prefill kernel).
        """
        if self._state is None:
            self.allocate_kv_cache()
        assert tokens.shape[0] == 1, f"NemotronHForCausalLM: batch_size must be 1, got {tokens.shape[0]}"
        seq = tokens[0].tolist()  # [S]
        start_pos = int(current_pos[0]) if current_pos is not None else self._prefill_pos

        logits_tt = None
        for i, tok in enumerate(seq):
            pos = start_pos + i
            self._update_ids(tok)
            self._update_pos(pos)
            logits_tt = nemotron_h_forward_stateful(
                self.mesh_device, self._ids_tt, self._wc, self._state, cpu_gate=True
            )
            ttnn.synchronize_device(self.mesh_device)
            self._state.advance()

        self._prefill_pos = start_pos + len(seq)
        self._decode_pos = self._prefill_pos

        # Return last-position logits on CPU: [1, 1, vocab] → [1, vocab]
        logits_cpu = _host_rep(logits_tt, self.mesh_device, 1)  # [1, 1, vocab]
        return logits_cpu[0]  # [1, vocab]

    def decode_forward(
        self,
        tokens: torch.Tensor,  # [B, 1] int64 — next input token
        current_pos: torch.Tensor = None,  # [B] int64 current position
        kv_cache=None,
        **kwargs,
    ) -> torch.Tensor:
        """Single-token decode step using the captured trace.

        vLLM calls this once per token per active sequence.
        Returns logits [B, 1, vocab] on CPU.

        Falls back to eager (cpu_gate=True) if the trace has not been captured.
        """
        if self._state is None:
            self.allocate_kv_cache()
        assert tokens.shape[0] == 1, f"NemotronHForCausalLM: batch_size must be 1, got {tokens.shape[0]}"
        tok = int(tokens[0, 0])
        pos = int(current_pos[0]) if current_pos is not None else self._decode_pos

        self._update_ids(tok)
        self._update_pos(pos)

        if self._trace_id is not None:
            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=True)
            logits_tt = self._trace_logits_tt
        else:
            logits_tt = nemotron_h_forward_stateful(
                self.mesh_device, self._ids_tt, self._wc, self._state, cpu_gate=True
            )
            ttnn.synchronize_device(self.mesh_device)

        self._state.advance()
        self._decode_pos = pos + 1

        logits_cpu = _host_rep(logits_tt, self.mesh_device, 1)  # [1, 1, vocab]
        return logits_cpu  # [1, 1, vocab]

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    def reset_state(self):
        """Reset all SSM / conv / KV state to zeros for a new sequence.

        Call this between independent generation requests when reusing the
        same generator instance (e.g. in server mode between requests).
        Reallocates the DecoderState so all persistent tensors start at zero.
        """
        max_seq_len = self._max_seq_len
        self._state = allocate_decoder_state(
            self.mesh_device, B=1, max_seq_len=max_seq_len, block_size=self._block_size
        )
        self._prefill_pos = 0
        self._decode_pos = 0
        # Note: trace is still valid (it captures ops, not data), so we do NOT
        # release it here.  The new zero state will be updated by the trace on
        # the next execute_trace call.

    # -----------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------

    def _update_ids(self, token_id: int):
        cpu = torch.tensor([[token_id]], dtype=torch.int32)
        host = ttnn.from_torch(cpu, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self._ids_tt)

    def _update_pos(self, pos: int):
        cpu = torch.tensor([pos], dtype=torch.int32)
        host = ttnn.from_torch(cpu, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self._state.current_pos)

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    def __del__(self):
        if self._trace_id is not None:
            try:
                ttnn.release_trace(self.mesh_device, self._trace_id)
            except Exception:
                pass
            self._trace_id = None
