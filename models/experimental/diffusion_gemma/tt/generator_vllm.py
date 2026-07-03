# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma vLLM adapter for the tenstorrent/vllm TT plugin (#47466 / #47488).

DiffusionGemma is a **block-diffusion** model: a single decode step denoises a
256-token canvas and commits it, so the model emits a **256-token BLOCK per decode
step**, not one token. This adapter is written to that *block-granular* contract.
The whole denoise loop (bidirectional canvas attention, three-phase KV, on-device
Gumbel-max / entropy-budget / renoise sampling, self-conditioning) lives inside
``prefill_forward`` / ``decode_forward`` via the existing ``tt.generate`` engine —
the runner passes only tokens / page_table / kv_cache / start_pos / prompt_lens /
sampling; the tt-metal model owns forward + attention + KV.

Structure
---------
The block-emission state machine is the vLLM-free
:class:`~models.experimental.diffusion_gemma.tt.serving.BlockDiffusionServingSession`;
this file is the thin vLLM interface wrapper over it:

- ``prefill_forward`` → :meth:`BlockDiffusionServingSession.prefill` (write prompt
  K/V, build the stateful denoise logits fn) followed by the first
  :meth:`decode_block` (block 0), mirroring the autoregressive contract where
  prefill returns the first token — here it returns the first 256-token block.
- ``decode_forward`` → one :meth:`decode_block` per active request (block N).

Because the block-emission core has no vLLM import, the reduced-surface serving
driver drives the identical contract on device without the (container-gated) vLLM
stack. See ``doc/vllm_integration/README.md``.

Contract gaps handled here vs deferred to #47488 (upstream tenstorrent/vllm)
--------------------------------------------------------------------------
The current TT runner assumes **one committed token per decode step** — hard
``assert num_out_tokens == 1`` at ``model_runner.py:2471``, ``[sz, 1]`` sampled-id
shape (``:2378``, ``:1878``), single-token ``_build_runner_output`` (``:2437``),
and a ``+1`` host position advance (``_apply_sampled_tokens_to_state`` ``:2479`` /
``:2508``). Emitting a 256-token block therefore needs the runner/scheduler to (a)
accept a ``[num_reqs, 256]`` block output, (b) advance ``num_computed_tokens`` /
``num_tokens`` by ``canvas_length`` per decode step, and (c) bound-check
``start_idx + 256 <= max_model_len``. That runner+scheduler change is **#47488**;
this adapter is written to that block contract so it works once #47488 lands.

Cache ownership
---------------
The diffusion denoise-read path reads the frozen prompt prefix from the
**model-owned contiguous** ``tt_model.tt_kv_cache`` via ``ttnn.slice`` (not from a
vLLM paged block pool). Serving therefore runs in the generator/standalone
cache-ownership mode: the model owns its ``max_model_len`` cache and is driven with
``page_table=None``; :meth:`allocate_kv_cache` returns those existing handles (no
double allocation). Routing the frozen-prefix read through a vLLM paged cache +
per-request block tables (for concurrent batched serving) is part of #47488 and
the batched-canvas-decode work (#47557). Until then one contiguous cache backs one
active sequence.

**Do not edit ``models/demos/gemma4/``.** The backbone is imported and reused
unchanged; the ``get_kv_cache_spec`` hybrid layer-type logic is copied (not
imported) so this adapter stays self-contained.
"""

from __future__ import annotations

import os

import torch
from loguru import logger

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession
from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM, allocate_vllm_kv_cache


def _resolve_checkpoint_dir(hf_config):
    """Locate the DiffusionGemma checkpoint from the vLLM hf_config / env."""
    for attr in ("_name_or_path", "name_or_path"):
        path = getattr(hf_config, attr, None)
        if path:
            return path
    env_path = os.environ.get("DG_CKPT")
    if env_path:
        return env_path
    raise ValueError("DiffusionGemma checkpoint path not found on hf_config (_name_or_path) or DG_CKPT env var")


class DiffusionGemmaForCausalLM(HybridAttentionForCausalLM):
    """Block-diffusion TT bridge for the tenstorrent/vllm TT plugin.

    Registered as ``TTDiffusionGemmaForBlockDiffusion`` (HF arch
    ``DiffusionGemmaForBlockDiffusion`` → plugin ``TT`` prefix). Inherits the
    hybrid KV-cache scaffolding and per-layer page-table plumbing from
    :class:`HybridAttentionForCausalLM`; overrides the forward path to run the
    diffusion block engine instead of the autoregressive one.
    """

    # Serving-feature reality on the TT path (documented in the stage evidence):
    #  * prefix caching: force-disabled for sliding-window models (platform.py:512),
    #    and block-diffusion recomputes canvas K/V every step → declare False.
    #  * async decode: the per-BLOCK async contract is unproven without the #47488
    #    runner; never advertise async without proof → declare False (safe default).
    #  * on-device sampling: the canvas Gumbel-max / entropy-budget / renoise path
    #    runs on device (no host argmax, no full-logits readback) → True.
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
    }

    def __init__(self, *args, dg_state_dict=None, tokenizer=None, config=None, gumbel_mode="argmax", **kwargs):
        super().__init__(*args, **kwargs)
        self._dg_state_dict = dg_state_dict
        self._tokenizer = tokenizer
        self._config = DiffusionConfig() if config is None else config
        self.canvas_length = self._config.canvas_length
        # Sampler memory strategy at the served context. "argmax" (RUN-first) and
        # "chunked" both fit full-depth 256K; the full-vocab Gumbel materialization
        # OOMs at 256K (see doc/context_contract.json).
        self._gumbel_mode = os.environ.get("DG_VLLM_GUMBEL_MODE", gumbel_mode)
        # One active session per batch row. A single contiguous model cache backs
        # one active sequence today (see module docstring); the dict is keyed by
        # row so output formatting never assumes batch size 1.
        self._sessions: dict[int, BlockDiffusionServingSession] = {}

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=262144,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        if optimizations not in (None, "performance"):
            raise ValueError("DiffusionGemma TT serving uses the full-model bf16 policy; no custom profiles")
        if tt_data_parallel != 1:
            # The 26B-A4B backbone is tensor-parallel (TP=4) on the (1,4) QB2 mesh;
            # attention data-parallel replicas are not part of the block-diffusion
            # serving path today.
            raise ValueError("DiffusionGemma TT serving is TP=4 single-replica (tt_data_parallel must be 1)")

        checkpoint_dir = _resolve_checkpoint_dir(hf_config)
        model_kwargs = dict(
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            dtype=ttnn.bfloat16,  # full-model policy: bf16 weights + bf16 KV cache
            create_kv_cache=True,  # model owns its contiguous KV cache (see docstring)
        )
        if n_layers is not None:
            model_kwargs["num_layers"] = n_layers

        bundle = build_tt_model_from_checkpoint_dir(mesh_device, checkpoint_dir, **model_kwargs)
        logger.info(
            f"[DiffusionGemma vLLM] built model: max_seq_len={max_seq_len} "
            f"n_layers={n_layers or 'full'} gumbel_mode={os.environ.get('DG_VLLM_GUMBEL_MODE', 'argmax')}"
        )
        return cls(
            [bundle.tt_model],
            [bundle.model_args],
            mesh_device,
            dg_state_dict=bundle.state_dict,
            tokenizer=bundle.tokenizer,
        )

    @property
    def cache_path(self):
        return self.model_args[0].weight_cache_path(ttnn.bfloat16)

    # ── vLLM VllmModelForTextGeneration protocol shims ──────────────────
    # vLLM's is_text_generation_model predicate inspects the resolved class for
    # embed_input_ids / forward / compute_logits. DiffusionGemma has no upstream
    # vLLM impl, so inspection lands here. Execution goes through prefill_forward /
    # decode_forward; these are never invoked.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError("DiffusionGemma is a TT bridge; embeddings happen on TT in decode_forward.")

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("DiffusionGemma is a TT bridge; the TT runner calls prefill_forward/decode_forward.")

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("DiffusionGemma keeps logits on device; canvas sampling runs in decode_forward.")

    # ── KV cache ────────────────────────────────────────────────────────
    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Per-layer hybrid KV spec (copied from the gemma4 bridge geometry).

        DiffusionGemma's text backbone == Gemma-4 26B-A4B: sliding layers use
        ``head_dim`` (256) / ``num_key_value_heads``; full-attention layers use
        ``global_head_dim`` (512) / ``num_global_key_value_heads``. Emitting the
        correct per-type spec keeps vLLM's hybrid manager from crashing; note the
        diffusion forward reads the model-owned contiguous cache, so this spec is
        the manager's bookkeeping, not the physical cache (see #47488).
        """
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config

        hf_config = model_config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            raise ValueError(f"{cls.__name__}.get_kv_cache_spec requires text_config.layer_types")

        sliding_kv_heads = text_config.num_key_value_heads
        sliding_head_dim = text_config.head_dim
        sliding_window = getattr(text_config, "sliding_window", None)
        full_kv_heads = getattr(text_config, "num_global_key_value_heads", None) or sliding_kv_heads
        full_head_dim = getattr(text_config, "global_head_dim", None) or sliding_head_dim

        tp = parallel_config.tensor_parallel_size
        sliding_kv_heads_per_dev = sliding_kv_heads // tp
        full_kv_heads_per_dev = full_kv_heads // tp

        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )
        block_size = cache_config.block_size

        spec_per_layer = {}
        for i, lt in enumerate(layer_types):
            name = f"model.layers.{i}.self_attn"
            if lt == "sliding_attention":
                if sliding_window is None:
                    raise ValueError(f"layer_types[{i}] is sliding but sliding_window is None")
                spec_per_layer[name] = SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=sliding_kv_heads_per_dev,
                    head_size=sliding_head_dim,
                    dtype=dtype,
                    sliding_window=sliding_window,
                )
            elif lt == "full_attention":
                spec_per_layer[name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=full_kv_heads_per_dev,
                    head_size=full_head_dim,
                    dtype=dtype,
                )
            else:
                raise ValueError(f"Unsupported layer_type {lt!r} at layer {i}")
        return spec_per_layer

    def allocate_kv_cache(self, *args, **kwargs):
        # Legacy uniform path; hybrid uses allocate_kv_cache_per_layer. Serving
        # runs on the model-owned contiguous cache, so this exists to satisfy the
        # interface — the returned handles are the model's own (no new DRAM).
        return allocate_vllm_kv_cache(*args, **kwargs, dp_model=self.model, tt_cache_path=self.cache_path)

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        del per_layer_specs  # sizing bookkeeping only; physical cache is model-owned
        # Return the model's own per-layer [k, v] handles so vLLM's kv_cache arg
        # points at the physical cache the diffusion forward actually reads/writes.
        return [[[k, v] for (k, v) in model.tt_kv_cache] for model in self.model]

    # ── warmup ──────────────────────────────────────────────────────────
    def warmup_model_prefill(self, kv_cache, enable_trace, can_sample_on_device, greedy_only: bool = False):
        # The block-diffusion path builds its per-request denoise logits fn lazily
        # in prefill_forward and does not use gemma4's AR prefill trace warmup.
        # Program-cache warm-up happens naturally on the first prefill/decode.
        del kv_cache, enable_trace, can_sample_on_device, greedy_only
        logger.info("[DiffusionGemma vLLM] warmup is a no-op; block-diffusion warms on first prefill/decode")

    # ── block-granular forward ──────────────────────────────────────────
    def _prompt_tokens_for_row(self, tokens, prompt_lens, row):
        length = int(prompt_lens[row]) if prompt_lens is not None else tokens.shape[1]
        ids = tokens[row, :length].reshape(1, length).to(torch.long)
        return ids

    def _make_session(self, seed: int = 0) -> BlockDiffusionServingSession:
        return BlockDiffusionServingSession(
            self.model[0],
            self._dg_state_dict,
            config=self._config,
            tokenizer=self._tokenizer,
            gumbel_mode=self._gumbel_mode,
            seed=seed,
        )

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        start_pos=None,
        page_tables_per_layer=None,
        sampling_params=None,
        empty_slots=None,
        **kwargs,
    ):
        """Write prompt K/V and emit block 0 for each request.

        Prompt length may be any value up to ``max_model_len`` — there is no
        divisibility requirement (the intrinsic 256-token *output* block
        granularity is not an input constraint). ``prefill_prompt_tokens`` pads to
        a 32-tile multiple internally. Returns ``[num_reqs, canvas_length]``
        committed block-0 token ids (block-granular output; see #47488).
        """
        del kv_cache, start_pos, page_tables_per_layer, sampling_params  # model-owned cache path
        num_reqs = tokens.shape[0]
        if num_reqs > 1:
            # One contiguous model cache backs one active sequence: a second
            # request's prefill would overwrite the first's frozen prompt K/V.
            # Concurrent batched serving needs the vLLM paged-cache ownership
            # change (#47488) + batched canvas decode (#47557). Fail loud rather
            # than silently corrupt — this is the recorded hard limit, not a
            # hardcoded batch-1 assumption in shapes/formatting.
            raise NotImplementedError(
                f"DiffusionGemma serving is single active sequence (got {num_reqs}); "
                "concurrent batched serving is #47488 (paged-cache ownership) + #47557 "
                "(batched canvas decode). Set --max-num-seqs 1."
            )
        blocks = []
        for row in range(num_reqs):
            session = self._make_session()
            prompt_tokens = self._prompt_tokens_for_row(tokens, prompt_lens, row)
            cache_len = session.prefill(prompt_tokens)
            emission = session.decode_block()
            logger.info(
                f"[DiffusionGemma vLLM] prefill row={row} prompt_len={session.prompt_len} "
                f"cache_len={cache_len} block0 next_pos={emission.next_pos} "
                f"steps={emission.num_denoise_steps} latency={emission.latency_s:.3f}s"
            )
            self._sessions[row] = session
            blocks.append(emission.tokens.reshape(1, self.canvas_length))
        return torch.cat(blocks, dim=0)

    def decode_forward(
        self,
        tokens=None,
        start_pos=None,
        page_table=None,
        kv_cache=None,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params=None,
        page_tables_per_layer=None,
        reset_batch: bool = False,
        slot_remap=None,
        **kwargs,
    ):
        """Emit the next 256-token block for each active request.

        One call == one denoise+commit block per active session. ``start_pos``,
        page-table refresh, and any stale-input handling are per-BLOCK (not
        per-token): the model already holds each request's absolute position and
        committed K/V in its cache, so the runner-supplied per-token ``tokens`` /
        ``start_pos`` are advisory only on this path. ``read_from_device`` /
        async semantics are per-block; the committed block returns on host (only
        per-step [B,L] decision tensors are read back — the [B,L,vocab] logits stay
        on device).
        """
        del tokens, start_pos, page_table, kv_cache, enable_trace, read_from_device
        del sampling_params, page_tables_per_layer, reset_batch, slot_remap
        if not self._sessions:
            raise RuntimeError("decode_forward called with no active sessions (prefill_forward first)")
        rows = sorted(self._sessions)
        blocks = []
        for row in rows:
            session = self._sessions[row]
            if session.finished:
                # Request already emitted a stop token; pad with the stop id.
                stop_id = 0
                if session.stop_token_ids is not None:
                    ids = (
                        session.stop_token_ids
                        if isinstance(session.stop_token_ids, (list, tuple))
                        else [session.stop_token_ids]
                    )
                    stop_id = int(ids[0])
                blocks.append(torch.full((1, self.canvas_length), stop_id, dtype=torch.long))
                continue
            emission = session.decode_block()
            logger.info(
                f"[DiffusionGemma vLLM] decode row={row} block={emission.block_idx} "
                f"start_pos={emission.start_pos} next_pos={emission.next_pos} "
                f"steps={emission.num_denoise_steps} halted={emission.halted} "
                f"stop={emission.stop} latency={emission.latency_s:.3f}s"
            )
            blocks.append(emission.tokens.reshape(1, self.canvas_length))
        return torch.cat(blocks, dim=0)

    def release_request(self, row: int) -> None:
        """Drop a finished request's session and release its logits-fn state."""
        session = self._sessions.pop(row, None)
        if session is not None:
            session.reset()
