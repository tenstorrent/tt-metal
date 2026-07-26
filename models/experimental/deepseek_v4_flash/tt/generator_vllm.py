# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""vLLM generator wrapper for DeepSeek-V4-Flash (Phase 1: functional bringup).

A thin adapter from vLLM's batched forward contract onto
:class:`~models.experimental.deepseek_v4_flash.tt.generator.DeepSeekV4Generator`,
which owns the model, the caches and the decode step. Each vLLM slot is one paged
session on the generator: the batch is served by looping over the slots, activating
each one's session and replaying the shared decode trace. Prefill replays decode one
token at a time. Sampling is done on host by vLLM.

The checkpoint declares ``architectures: ["DeepseekV4ForCausalLM"]`` and
``model_type: "deepseek_v4"``. Register that name in the plugin clone
(``plugins/vllm-tt-plugin/src/vllm_tt_plugin/model_registry.py``), alongside the
DeepSeek-V3 entry it mirrors::

    "DeepseekV4ForCausalLM": (
        "models.experimental.deepseek_v4_flash.tt.generator_vllm",
        "DeepseekV4FlashForCausalLM",
    ),

V4 is pipeline-parallel across the whole mesh, so it runs with ``tt_data_parallel=1``
and no TP/PP splitting; prefix caching, async decode, on-device sampling, speculative
decode and LoRA are all unsupported (see :attr:`model_capabilities`).
"""

from __future__ import annotations

import os

import torch
from loguru import logger

import ttnn
from models.experimental.deepseek_v4_flash.tt.generator import DeepSeekV4Generator
from models.experimental.deepseek_v4_flash.tt.paged_cache import build_groups, min_tokens_per_block


class DeepseekV4FlashForCausalLM:
    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": False,
    }

    def __init__(self, generator: DeepSeekV4Generator, max_batch_size: int, slots: list[int]):
        self.generator = generator
        self.max_batch_size = max_batch_size
        self.slots = slots  # vLLM slot index -> generator session id
        self._warmed_up = False

    @property
    def model(self):
        return self.generator.model

    def _session(self, slot: int) -> int:
        if not self.slots:
            raise RuntimeError("allocate_kv_cache must run before a forward: it owns the pools and the sessions")
        return self.slots[slot]

    @classmethod
    def get_max_tokens_all_users(cls, model_name: str = "", num_devices: int = 1, tt_data_parallel: int = 1, **kwargs):
        """KV-cache token budget shared by all users, which sizes vLLM's block pool.

        V4's caches are compressed, so a token is cheap: a CSA layer keeps one
        512-wide row per 4 tokens and an HCA layer one per 128, i.e. under 6 KB per
        token across the whole 43-layer stack, plus a bounded 128-token ring per
        session. 128K tokens is therefore a few hundred MB rather than the tens of GB
        the same figure would cost a dense model.

        This is still a per-device tuning knob and not a model constant -- override it
        with ``DEEPSEEK_V4_MAX_TOKENS_ALL_USERS`` (the tt-inference-server model spec's
        ``env_vars`` block is the right place) rather than editing the default.
        """
        override = os.environ.get("DEEPSEEK_V4_MAX_TOKENS_ALL_USERS")
        return int(override) if override else 131072

    def warmup_model_prefill(self, kv_cache=None, enable_trace=True, can_sample_on_device=False, greedy_only=False):
        """Compile the kernels and capture the decode traces before the server reports
        healthy, so the first real request does not pay for minutes of capture.

        There is no separate prefill program to warm: a prompt is replayed through the
        decode path, so this is the same throwaway step as the decode warmup below.
        """
        self._warmup(can_sample_on_device)

    def warmup_model_decode(
        self,
        kv_cache=None,
        enable_trace=True,
        max_batch_size=None,
        num_blocks=None,
        can_sample_on_device=False,
        read_from_device=True,
        greedy_only=False,
    ):
        """Counterpart of :meth:`warmup_model_prefill`; the plugin calls both.

        ``max_batch_size`` and ``num_blocks`` describe geometry this model fixed in
        :meth:`allocate_kv_cache` (its sessions and page tables are model-owned), so
        they are only checked for agreement, not used to shape anything.
        """
        if max_batch_size is not None and max_batch_size != self.max_batch_size:
            raise ValueError(
                f"warmup asks for a batch of {max_batch_size} but the sessions were opened for "
                f"{self.max_batch_size}; decode indexes one session per vLLM slot, so the two must match"
            )
        self._warmup(can_sample_on_device)

    def _warmup(self, can_sample_on_device: bool) -> None:
        """One throwaway decode step on slot 0's session, rewound afterwards.

        The plugin warms up in two phases -- compile with ``enable_trace=False``, then
        capture with ``enable_trace=True`` -- and calls both the prefill and the decode
        hook in each. All four collapse into this single step for V4: there is only a
        traced decode path, and ``decode_traced`` compiles *and* captures every variant
        on its first call, so every later call would just replay them. Hence the flag,
        which is deliberately not the ``already_warmed_up_prefill`` the plugin resets
        between its phases.
        """
        if can_sample_on_device:
            raise ValueError(
                "on-device sampling is not supported (model_capabilities declares it off); "
                "this model only ever returns logits"
            )
        if self._warmed_up:
            return
        logger.info("warming up DeepSeek-V4-Flash: compiling kernels and capturing decode traces")
        self.generator.warmup(self._session(0))
        self._warmed_up = True

    @property
    def tokenizer(self):
        return self.generator.tokenizer

    @classmethod
    def initialize_vllm_model(
        cls, hf_config, mesh_device, max_batch_size, max_seq_len, tt_data_parallel=1, optimizations=None
    ):
        if tt_data_parallel != 1:
            raise ValueError(
                "DeepSeek-V4-Flash is pipeline-parallel across the whole mesh; only tt_data_parallel=1 is supported, "
                f"got {tt_data_parallel}"
            )
        # The pools are allocated in ``allocate_kv_cache`` (vLLM sizes them), which is
        # also where the decode state is prepared -- it must all exist before the first
        # forward captures a trace.
        generator = DeepSeekV4Generator.from_pretrained(mesh_device, max_seq_len=max_seq_len, prepare=False)
        return cls(generator, max_batch_size, slots=[])

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """One uniform MLA spec for the whole stack, in place of the plugin's default.

        Without this hook the plugin builds the spec itself, and that path asserts
        ``not sliding_window`` whenever ``model_config.use_mla`` -- which holds for V4
        (``model_type: deepseek_v4`` carrying a ``head_dim``) while its config also sets
        ``sliding_window: 128``, so it fails before the model is ever loaded. The assert
        does not apply here: the window is a ring *inside* each layer's own pool (see
        :meth:`allocate_kv_cache`), never a vLLM-visible block budget, so the spec must
        leave ``sliding_window`` unset -- declaring it would have vLLM hand out a
        window's worth of blocks per request instead of a context's worth.

        The spec is deliberately uniform. ``allocate_kv_cache`` scales each layer's
        *rows* per block by its compress rate so that one block is ``block_size`` tokens
        of context in every layer, which is exactly what a single spec claims. A
        per-layer spec would instead split the stack into several kv cache groups, and
        the plugin then routes per-layer page tables that this wrapper does not take.
        """
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        model_config, cache_config = vllm_config.model_config, vllm_config.cache_config
        # MLA keeps one latent row per position rather than a K and a V tensor, which is
        # also what the pools hold: [blocks, 1, rows, head_dim]. num_kv_heads is 1 for
        # the same reason (MLA decodes as MQA) and head_size is config.head_dim.
        spec = MLAAttentionSpec(
            block_size=cache_config.block_size,
            num_kv_heads=model_config.get_num_kv_heads(vllm_config.parallel_config),
            head_size=model_config.get_head_size(),
            dtype=(
                model_config.dtype
                if cache_config.cache_dtype == "auto"
                else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
            ),
        )
        # Same dummy layer name the plugin's single-spec default uses: the TT forward
        # context is empty, so nothing resolves this back to a real layer.
        return {"foo": spec}

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate the per-layer KV block pools and hand them to the model.

        vLLM asks for ``(max_num_blocks, num_kv_heads, block_size, head_size)`` with one
        geometry for every layer, which V4 cannot honour directly: a block is a fixed
        number of *rows* of a layer's KV axis, and a compressor layer stores one row per
        ``compress_rate`` tokens. Scaling the rows by the rate instead --
        ``tokens_per_block / compress_rate``, so CSA gets 1024-row blocks and HCA 32-row
        ones -- makes every layer type consume one block per ``tokens_per_block`` tokens
        of context, which is the uniformity vLLM actually depends on. Only the block
        *count* is taken from ``kv_cache_shape``; the per-layer shapes are derived here.

        The compressor *window* state (a few KB per session of not-yet-pooled
        projections) has no per-token block representation at all and stays model-owned,
        as do the page tables: ``PagedKVManager`` hands out block ids from these pools,
        so the ``page_table`` argument vLLM passes into forward is still ignored.
        """
        model, config = self.generator.model, self.generator.config
        max_num_blocks = int(kv_cache_shape[0])
        layer_types = config.layer_types[: model.num_layers]
        tokens_per_block = min_tokens_per_block(config.compress_rates.values())
        groups = build_groups(
            layer_types,
            config.compress_rates,
            model.sliding_window,
            self.generator.max_seq,
            tokens_per_block=tokens_per_block,
        )
        # Block 0 of every pool is the shared zero block, and a session's ring blocks are
        # held for its lifetime, so this many blocks are gone before a single token is
        # decoded. Fail here rather than with a PagedCacheFull on the first prefill.
        for name, group in groups.items():
            needed = 1 + self.max_batch_size * group.ring_blocks
            if max_num_blocks < needed + self.max_batch_size:
                raise ValueError(
                    f"{max_num_blocks} blocks is too few for {self.max_batch_size} sessions of {name}: "
                    f"{needed} are taken by the zero block and the sliding rings alone. Raise the KV cache "
                    f"budget (get_max_tokens_all_users) or lower max_num_seqs."
                )

        pools = {
            li: ttnn.from_torch(
                torch.zeros(max_num_blocks, 1, groups[layer_types[li]].block_size, config.head_dim),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=model.layer_devices[li],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for li in range(model.num_layers)
        }
        logger.info(
            f"allocated {max_num_blocks} KV blocks per layer of {tokens_per_block} tokens each: "
            + ", ".join(f"{name} {g.block_size} rows" for name, g in groups.items())
        )

        self.generator.prepare_decode(
            num_sessions=self.max_batch_size,
            tokens_per_block=tokens_per_block,
            pools=pools,
        )
        self.slots = [self.generator.open_session() for _ in range(self.max_batch_size)]
        # vLLM keeps this and hands it back on every forward; it is indexed by layer, so
        # pad out the layers a capped stack (DEEPSEEK_V4_DECODE_LAYERS) does not have.
        return [pools.get(li) for li in range(num_layers)]

    def prefill_forward(self, *args, **kwargs):
        """Seed each user's cache with its prompt and return the last position's logits.

        There is no prefill kernel: a prompt is replayed one decode step per token at
        ascending positions, so this costs O(prompt_len) steps per user. vLLM's
        ``page_table`` and ``kv_cache`` are ignored (see :meth:`allocate_kv_cache`).

        Returns ``[B, 1, V]`` on host -- only the final position, since that is the one
        vLLM samples. If the plugin's ``_prepare_model_inputs`` turns out to want the
        full ``[B, S, V]``, widen it here; nothing else depends on the shape.
        """
        tokens = kwargs["tokens"]  # [B, max_prompt_len], zero-padded
        prompt_lens = kwargs["prompt_lens"]  # [B]
        empty_slots = kwargs.get("empty_slots")
        assert kwargs.get("sampling_params") is None, "on-device sampling is not supported"

        out = torch.zeros(tokens.shape[0], 1, self.generator.vocab_size)
        for i in range(tokens.shape[0]):
            slot = int(empty_slots[i]) if empty_slots is not None else i
            sid = self._session(slot)
            # A slot handed back by vLLM carries the previous sequence's cache; the
            # prefill of a new one starts at position 0, so rewind it first.
            self.generator.reset_session(sid)
            prompt_len = int(prompt_lens[i])
            out[i, 0] = self.generator.prefill(sid, [int(t) for t in tokens[i, :prompt_len]])
        return out

    def decode_forward(self, *args, **kwargs):
        """Advance every active user by one token and return ``[B, 1, V]`` host logits.

        ``enable_trace`` is moot -- each per-user step is already a trace replay -- and
        the batch is served by looping over the slots, so a step costs B replays rather
        than one. Folding the loop into a single B>1 traced step is Phase 2.
        ``read_from_device`` is likewise ignored: async decode is not supported, so the
        result is always read back here.
        """
        tokens = kwargs["tokens"].squeeze(1)  # [max_batch_size]
        start_pos = kwargs["start_pos"]  # [max_batch_size]
        assert kwargs.get("sampling_params") is None, "on-device sampling is not supported"

        out = torch.zeros(tokens.shape[0], 1, self.generator.vocab_size)
        for slot in range(tokens.shape[0]):
            pos = int(start_pos[slot])
            # Rows past the real batch are padding, zero-filled to the traced shape.
            if pos <= 0 or pos >= self.generator.max_seq:
                continue
            out[slot, 0] = self.generator.logits(self._session(slot), int(tokens[slot]), pos)
        return out

    def read_decode_output(self, tt_out, async_read=False):
        return (tt_out, []) if async_read else tt_out
