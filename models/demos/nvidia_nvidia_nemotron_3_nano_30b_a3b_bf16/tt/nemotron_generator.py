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
import torch.nn.functional as F

import ttnn
from models.tt_transformers.tt.generator import Generator

from .kv_cache import DEFAULT_BLOCK_SIZE, DEFAULT_MAX_SEQ_LEN, MODEL_MAX_SEQ_LEN, DecoderState, allocate_decoder_state
from .model import WeightCache, nemotron_h_forward_stateful, nemotron_h_prefill_stateful, nemotron_h_prefill_stateful_tt
from .tp import _R, _host_rep, probe_dram_defect_for_shape


def _sample_from_logits(logits_1d: torch.Tensor, sampling_params) -> int:
    """Host-side top-k → top-p → temperature sampling from a 1-D logits tensor.

    Called when the vllm runner is in sample_on_device_mode='decode_only' —
    the runner expects a token ID back, not raw logits.
    sampling_params is a TTSamplingParams(temperature, top_k, top_p).
    temperature=0.0 uses greedy argmax (avoids division-by-zero in softmax).
    """
    temperature = float(getattr(sampling_params, "temperature", 1.0))
    top_k = int(getattr(sampling_params, "top_k", 0))
    top_p = float(getattr(sampling_params, "top_p", 1.0))

    logits = logits_1d.float()

    if temperature == 0.0:
        return int(torch.argmax(logits).item())

    if top_k > 0:
        k = min(top_k, logits.shape[0])
        top_k_values, top_k_indices = torch.topk(logits, k=k)
    else:
        top_k_values = logits
        top_k_indices = torch.arange(logits.shape[0], dtype=torch.long)

    # top-p nucleus filtering
    sorted_probs = F.softmax(top_k_values / temperature, dim=-1)
    cumulative = sorted_probs.cumsum(-1)
    mask = (cumulative - sorted_probs) > top_p
    top_k_values = top_k_values.masked_fill(mask, float("-inf"))

    probs = F.softmax(top_k_values / temperature, dim=-1)
    probs = torch.nan_to_num(probs)
    selected = torch.multinomial(probs, num_samples=1).item()
    return int(top_k_indices[selected])


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
        self.max_batch_size = 1
        self.vocab_size = 131_072

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
        self._greedy_tok_tt = None  # persistent [1,1] uint32: on-device argmax output
        self._pos_one_tt = None  # constant [1] int32 = 1 for in-trace position increment
        self._first_decode: bool = True  # True until first decode_forward is called
        self._prefill_pos: int = 0  # tracks current sequence position across calls
        self._decode_pos: int = 0
        # Prefill trace support: separate trace per ISL.
        self._prefill_trace_ids: dict = {}  # ISL → trace_id
        self._prefill_ids_tt: dict = {}  # ISL → [1, ISL] uint32 device tensor
        self._prefill_logits_tt: dict = {}  # ISL → [1, 1, vocab] device tensor

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
                          MODEL_MAX_SEQ_LEN (262144) is the trained context window; values
                          above it are allowed for generation beyond a full-context prefill
                          (Mamba2 layers are position-independent; attention RoPE extrapolates).
            n_layers:     Optional layer count override (for unit tests).
        """
        if max_batch_size > 1:
            raise ValueError(
                f"NemotronHForCausalLM currently supports only batch_size=1 "
                f"(requested {max_batch_size}). Multi-sequence batching is future work."
            )
        if max_seq_len > MODEL_MAX_SEQ_LEN:
            import warnings

            warnings.warn(
                f"max_seq_len={max_seq_len} exceeds MODEL_MAX_SEQ_LEN={MODEL_MAX_SEQ_LEN}. "
                f"Mamba2 layers are position-independent; attention RoPE extrapolates beyond "
                f"the trained range for the extra {max_seq_len - MODEL_MAX_SEQ_LEN} positions.",
                stacklevel=2,
            )

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

    def allocate_kv_cache(self, *args, max_batch_size: int = 1, max_seq_len: int = None, **kwargs):
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
        # Pre-allocate the on-device greedy argmax output ([1,1] uint32).
        # Must exist before trace capture so no allocation occurs during replay.
        cpu_greedy = torch.zeros(1, 1, dtype=torch.int32)
        self._greedy_tok_tt = ttnn.from_torch(
            cpu_greedy,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=_R(self.mesh_device),
        )
        # Constant [1] int32 = 1 used for in-trace position increment.
        self._pos_one_tt = ttnn.from_torch(
            torch.ones(1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=_R(self.mesh_device),
        )
        self._prefill_pos = 0
        self._decode_pos = 0
        self._first_decode = True
        # Return a minimal handle (1-element list matching Generator convention).
        return [self._state]

    # -----------------------------------------------------------------
    # Warmup
    # -----------------------------------------------------------------

    def warmup_model_prefill(self, kv_cache, enable_trace: bool = False, can_sample_on_device: bool = False, **kwargs):
        """Warm up the chunked prefill path to compile kernels before the first request.

        Runs a 128-token dummy sequence through nemotron_h_prefill_stateful so the
        chunked SSD scan, dense-attention causal SDPA, and paged_fill_cache kernels
        are compiled.  Without this, the first real prefill pays a large compilation
        penalty.  State is reset after warmup so decode warmup starts clean.
        """
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True

        if self._state is None:
            self.allocate_kv_cache()

        print("[NemotronHForCausalLM] Warming up prefill (compiling kernels)...", flush=True)
        t0 = time.time()

        # When using the tt-lang SSD path, pre-compile ELFs for all powers-of-2
        # n_chunks up to max_seq_len so inference never pays JIT latency.
        from models.demos.nvidia_nvidia_nemotron_3_nano_30b_a3b_bf16.tt.mamba2_prefill import warmup_ttlang_kernels

        warmup_ttlang_kernels(self._max_seq_len, mesh_device=self.mesh_device)

        # 128-token dummy sequence: compiles the chunked SSD scan + paged_fill_cache.
        dummy_ids = torch.zeros(1, 128, dtype=torch.int64)
        self._update_pos(0)
        nemotron_h_prefill_stateful(self.mesh_device, dummy_ids, self._wc, self._state)
        ttnn.synchronize_device(self.mesh_device)
        self._state.advance()

        # Reset in-place so decode warmup and the first real request start from zero state.
        self._state.reset_inplace(self.mesh_device)
        self._prefill_pos = 0
        self._decode_pos = 0

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

        # Warmup pass outside trace: primes caches + compiles all kernels used in trace.
        self._update_ids(0)
        self._update_pos(self._decode_pos)
        logits_warmup = nemotron_h_forward_stateful(
            self.mesh_device, self._ids_tt, self._wc, self._state, cpu_gate=False
        )
        ttnn.argmax(logits_warmup, dim=-1, output_tensor=self._greedy_tok_tt)
        self._state.advance()
        ttnn.assign(ttnn.add(self._state.current_pos, self._pos_one_tt), self._state.current_pos)
        ttnn.synchronize_device(self.mesh_device)

        # Trace capture.  advance() and pos-increment are folded in so decode_forward
        # needs no Python calls between steps for state management.
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._trace_logits_tt = nemotron_h_forward_stateful(
            self.mesh_device, self._ids_tt, self._wc, self._state, cpu_gate=False
        )
        ttnn.argmax(self._trace_logits_tt, dim=-1, output_tensor=self._greedy_tok_tt)
        self._state.advance()
        ttnn.assign(ttnn.add(self._state.current_pos, self._pos_one_tt), self._state.current_pos)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self._trace_id = trace_id

        print(
            f"[NemotronHForCausalLM] Decode trace captured ({time.time() - t0:.1f}s).",
            flush=True,
        )

    def capture_prefill_trace(self, isl: int) -> None:
        """Capture a TTNN trace for prefill at a fixed sequence length.

        After capture, prefill_forward called with exactly `isl` tokens will
        bypass Python dispatch overhead and execute the compiled trace directly.
        For other ISLs the existing eager path is used.

        Call this after warmup_model_prefill has already compiled all kernels.
        Multiple ISLs can be traced independently (each needs ~1 forward pass).
        """
        if isl in self._prefill_trace_ids:
            return  # already captured

        if self._state is None:
            self.allocate_kv_cache()

        print(f"[NemotronHForCausalLM] Capturing prefill trace ISL={isl}...", flush=True)
        t0 = time.time()

        # Pre-allocate the fixed-ISL input ids tensor on device.
        ids_cpu = torch.zeros(1, isl, dtype=torch.int32)
        ids_tt = ttnn.from_torch(
            ids_cpu,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._prefill_ids_tt[isl] = ids_tt

        # Warmup pass outside trace (compile all ISL-specific kernels).
        self.reset_state()
        nemotron_h_prefill_stateful_tt(self.mesh_device, ids_tt, self._wc, self._state)
        ttnn.synchronize_device(self.mesh_device)

        # Trace capture.
        self.reset_state()
        trace_id = ttnn.begin_trace_capture(self.mesh_device, cq_id=0)
        self._prefill_logits_tt[isl] = nemotron_h_prefill_stateful_tt(self.mesh_device, ids_tt, self._wc, self._state)
        ttnn.end_trace_capture(self.mesh_device, trace_id, cq_id=0)
        ttnn.synchronize_device(self.mesh_device)
        self._prefill_trace_ids[isl] = trace_id

        print(
            f"[NemotronHForCausalLM] Prefill trace ISL={isl} captured ({time.time() - t0:.1f}s).",
            flush=True,
        )

    # -----------------------------------------------------------------
    # Forward passes
    # -----------------------------------------------------------------

    def prefill_forward_text(
        self,
        tokens: torch.Tensor,  # [B, S] int64 — prompt token ids
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        enable_trace=True,
        model_id_warmup=None,
        sampling_params=None,
        start_pos=None,
        return_hidden_states=False,
        warmup_prefill=False,
        **kwargs,
    ) -> torch.Tensor:
        """vLLM text-model prefill interface (called by the tt-inference-server).

        Routes to our chunked prefill path and returns [B, 1, vocab_size] to
        match the shape contract of the Generator base class.
        """
        current_pos_val = int(start_pos[0]) if start_pos is not None else None
        current_pos_t = torch.tensor([current_pos_val], dtype=torch.int64) if current_pos_val is not None else None
        logits_1v = self.prefill_forward(
            tokens,
            current_pos=current_pos_t,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
        )
        # prefill_forward returns [1, vocab]; add the sequence dim → [1, 1, vocab]
        return logits_1v.unsqueeze(1)

    def prefill_forward(
        self,
        tokens: torch.Tensor,  # [B, S] int64 — prompt token ids
        current_pos: torch.Tensor = None,  # [B] int64 starting position
        kv_cache=None,
        prompt_lens: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Process all S prompt tokens in one chunked-prefill forward pass.

        Calls nemotron_h_prefill_stateful which runs:
          - Mamba2 layers: bulk chunked SSD scan (fast O(S) rather than O(S²))
          - Dense-attention layers: causal SDPA + paged_fill_cache (fills KV cache)
          - MoE layers: per-token fallback (sparse_matmul is S=1 only)

        vLLM calls this once per request at the start of a new sequence.
        Returns logits for the LAST prompt position [B, 1, vocab].
        """
        if self._state is None:
            self.allocate_kv_cache()
        assert tokens.shape[0] == 1, f"NemotronHForCausalLM: batch_size must be 1, got {tokens.shape[0]}"
        # When current_pos is None (vllm server path), every prefill call is a
        # new sequence starting at position 0 — always reset.  When current_pos
        # is provided (demo / test path), reset only when it says 0.
        # In-place reset preserves DRAM buffer addresses and is trace-safe.
        start_pos = int(current_pos[0]) if current_pos is not None else 0
        if start_pos == 0:
            # switch_mode("prefill") mirrors the Qwen3.6 pattern: synchronize
            # the device and reset SSM/conv states for the new sequence.
            # It also documents the decode→prefill boundary for future extensions
            # (e.g. rebuilding persistent prefill-specific CCL buffers if NemotronH
            # ever gains separate prefill/decode sub-device managers).
            self.switch_mode("prefill")
        S = tokens.shape[1]

        # Set current_pos on device (used by dense_attention decode path; harmless for prefill).
        self._update_pos(start_pos)

        # Bulk chunked prefill: all S tokens in one forward pass.
        # Returns [B, 1, vocab_size] ttnn.Tensor for the last prompt position only.
        if S in self._prefill_trace_ids:
            # Traced path: write actual token ids into the pre-allocated device tensor,
            # then replay the compiled trace (zero Python dispatch overhead).
            ids_host = ttnn.from_torch(
                tokens.int(),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(ids_host, self._prefill_ids_tt[S])
            ttnn.execute_trace(self.mesh_device, self._prefill_trace_ids[S], cq_id=0, blocking=True)
            logits_tt = self._prefill_logits_tt[S]
        else:
            import os as _os

            _verbose = _os.environ.get("NEMOTRON_PREFILL_VERBOSE", "0") == "1"
            logits_tt = nemotron_h_prefill_stateful(self.mesh_device, tokens, self._wc, self._state, verbose=_verbose)
            ttnn.synchronize_device(self.mesh_device)
        self._state.advance()

        self._prefill_pos = start_pos + S
        self._decode_pos = self._prefill_pos

        # Return last-position logits on CPU: [1, 1, vocab] → [1, vocab]
        logits_cpu = _host_rep(logits_tt, self.mesh_device, 1)  # [1, 1, vocab]
        return logits_cpu[0]  # [1, vocab]

    def decode_forward(
        self,
        tokens: torch.Tensor,  # [B, 1] int64 — next input token
        start_pos: torch.Tensor = None,  # [B] int64 current position (matches Generator base)
        kv_cache=None,
        greedy: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """Single-token decode step using the captured trace.

        vLLM calls this once per token per active sequence.
        Returns (logits [B, 1, vocab], None) matching the Generator base class
        contract.  Falls back to eager (cpu_gate=True) if the trace has not
        been captured.

        greedy=True enables the on-device sampling fast path:
          - The argmax (captured inside the trace) writes the next token to
            _greedy_tok_tt on-device.
          - After the first step, _greedy_tok_tt is copied to _ids_tt via D2D
            (no host involvement for the token).
          - Only 1 int is read back from device (vs 131K floats for full logits).
          - Returns (torch.Tensor [1, 1] int64 containing the greedy token, None).

        When the vllm runner uses sample_on_device_mode='decode_only', it passes
        sampling_params (TTSamplingParams) via kwargs and expects a token ID back,
        not full logits.  temperature=0 maps to greedy (fast path); temperature>0
        samples on host inside this function before returning.
        """
        if self._state is None:
            self.allocate_kv_cache()
        assert tokens.shape[0] == 1, f"NemotronHForCausalLM: batch_size must be 1, got {tokens.shape[0]}"
        tok = int(tokens[0, 0])
        pos = int(start_pos[0]) if start_pos is not None else self._decode_pos

        # When vllm runner operates in sample_on_device_mode='decode_only' it passes
        # sampling_params and always expects a token (not logits) in return.
        sampling_params = kwargs.get("sampling_params", None)
        if sampling_params is not None and float(getattr(sampling_params, "temperature", 1.0)) == 0.0:
            greedy = True
        if self._first_decode:  # log once per sequence
            print(
                f"decode_forward: sampling_params={sampling_params}, greedy={greedy}, is_first={self._first_decode}",
                flush=True,
            )

        is_first = self._first_decode
        self._first_decode = False

        if greedy and self._trace_id is not None and not is_first:
            # Autoregressive greedy: previous trace wrote _greedy_tok_tt; copy D2D.
            ttnn.assign(self._greedy_tok_tt, self._ids_tt)
        else:
            self._update_ids(tok)

        # Position: trace increments current_pos on-device, so H2D write is only
        # needed on the first decode step (to set the initial position from prefill).
        if not (greedy and self._trace_id is not None and not is_first):
            self._update_pos(pos)

        if self._trace_id is not None:
            # advance() and pos-increment are folded into the trace.
            ttnn.execute_trace(self.mesh_device, self._trace_id, cq_id=0, blocking=True)
            logits_tt = self._trace_logits_tt
        else:
            logits_tt = nemotron_h_forward_stateful(
                self.mesh_device, self._ids_tt, self._wc, self._state, cpu_gate=True
            )
            ttnn.synchronize_device(self.mesh_device)
            self._state.advance()

        self._decode_pos = pos + 1

        if greedy and self._trace_id is not None:
            # Read back only the greedy token (4 bytes, not 262 KB of logits).
            greedy_full = ttnn.to_torch(
                self._greedy_tok_tt,
                mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
            )
            next_tok = int(greedy_full[0, 0])
            return torch.tensor([[next_tok]], dtype=torch.int64), None

        logits_cpu = _host_rep(logits_tt, self.mesh_device, 1)  # [1, 1, vocab]

        if sampling_params is not None:
            # Runner expects a sampled token when sample_on_device_mode is active —
            # returning logits would corrupt the runner's token-ID interpretation.
            next_tok = _sample_from_logits(logits_cpu[0, 0, :], sampling_params)
            return torch.tensor([[next_tok]], dtype=torch.int64), None

        return logits_cpu, None  # (logits [1, 1, vocab], log_probs) matches Generator base

    # -----------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------

    def switch_mode(self, mode: str) -> None:
        """Switch between prefill and decode modes.

        Mirrors the Qwen3.6 switch_mode() pattern.  For NemotronH:

        "prefill": drain the dispatch queue, then reset SSM/conv states
            in-place for the new sequence.  The in-place reset preserves
            DRAM buffer addresses — reallocation would break the decode trace
            which holds address references, not tensor values.  Persistent
            prefill-specific DRAM tensors (e.g. the causal_mask in
            _mamba2_ssd_chunk, cached via _rep_keyed) are not touched; they
            live at stable addresses first assigned during warmup and are safe
            to reuse across requests.

        "decode": no-op — the decode trace is already captured at init and
            stays valid across requests.  NemotronH has no prefetch-side CCL
            state that needs tearing down on decode entry.
        """
        if mode == "prefill":
            ttnn.synchronize_device(self.mesh_device)
            # Probe DRAM for defective pages at the shape of the largest
            # per-chunk intermediates in _mamba2_ssd_chunk (log_L, L_raw:
            # [B, NUM_HEADS, CHUNK_SIZE, CHUNK_SIZE] = [1, 64, 64, 64] = 512 KB).
            # After a prior request frees ~180 MB of chunk intermediates, the
            # DRAM allocator may coalesce those freed blocks back to a defective
            # page address.  This probe permanently blocks any defective pages
            # before the actual prefill allocation sequence begins.
            probe_dram_defect_for_shape(self.mesh_device, [1, 64, 64, 64], n_probes=4500)
            self.reset_state()

    def reset_state(self):
        """Reset all SSM / conv / KV state to zeros for a new sequence.

        Uses reset_inplace() which zeros SSM/conv states and current_pos via
        copy_host_to_device_tensor, preserving all DRAM buffer addresses.
        This is safe to call between requests when a decode trace is active —
        the trace holds buffer address references, not tensor values, so
        in-place zeroing does not invalidate it.  Reallocation (the prior
        approach) would break the trace on the next execute_trace call.
        """
        if self._state is None:
            self.allocate_kv_cache()
            return
        self._state.reset_inplace(self.mesh_device)
        self._prefill_pos = 0
        self._decode_pos = 0
        self._first_decode = True

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
