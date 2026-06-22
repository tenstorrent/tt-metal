# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os

import torch

import ttnn
from models.demos.gemma4.tt.common import create_tt_model
from models.demos.gemma4.tt.generator_trace import (
    maybe_disable_pli_prefill_trace,
    patch_gemma4_trace_model_args,
    resolve_gemma4_prefill_trace_enable,
    warmup_gemma4_model_prefill,
)
from models.tt_transformers.tt.common import get_padded_prefill_len
from models.tt_transformers.tt.generator import create_submeshes
from models.tt_transformers.tt.generator_vllm import HybridAttentionForCausalLM, allocate_vllm_kv_cache


class _Gemma4VllmOptimizations:
    @staticmethod
    def get_tensor_dtype(decoder_id, tensor, prefetcher=False):
        del decoder_id, tensor, prefetcher
        return ttnn.bfloat16


def _patch_model_args(model_args, mesh_device, max_batch_size, max_seq_len, model_path):
    model_args.max_batch_size = max_batch_size
    model_args.max_seq_len = max_seq_len
    model_args.max_prefill_chunk_size = max_seq_len
    patch_gemma4_trace_model_args(model_args, prefill_trace_enabled=True)
    model_args.optimizations = _Gemma4VllmOptimizations()
    model_args.mesh_device = mesh_device
    model_args._gemma4_model_path = model_path
    model_args.is_llama_vision = lambda: False


class Gemma4ForCausalLM(HybridAttentionForCausalLM):
    """Gemma4 — hybrid attention (sliding-window + full).

    Gemma4's decoder alternates ``sliding_attention`` and ``full_attention``
    layers per ``hf_config.layer_types``, so the bridge inherits from
    :class:`HybridAttentionForCausalLM` to opt into vLLM's hybrid kv cache
    manager. ``get_kv_cache_spec`` is inherited; layer-routed page tables
    flow through the model's ``_active_page_tables_per_layer`` stash and
    are picked up inside ``Gemma4Model.{ttnn_prefill_forward,
    ttnn_decode_forward}`` (mirrors the gpt-oss bridge).
    """

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
        "supports_sample_on_device": True,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mirrors the env-var check in :meth:`initialize_vllm_model` so the
        # forward-side padding logic doesn't need to re-read the environment
        # on every step. Defaults to on; flip ``GEMMA4_BOUNDED_SLIDING_KV_CACHE=0``
        # to fall back to the legacy unbounded path through the paged ops.
        self._bounded_sliding_kv_cache = os.environ.get("GEMMA4_BOUNDED_SLIDING_KV_CACHE", "1") != "0"

    def _maybe_disable_pli_prefill_trace(self, enable_trace: bool, batch_size: int = 1) -> bool:
        return maybe_disable_pli_prefill_trace(enable_trace, self.model[0], batch_size=batch_size)

    def warmup_model_prefill(
        self,
        kv_cache,
        enable_trace,
        can_sample_on_device,
        greedy_only: bool = False,
    ):
        warmup_gemma4_model_prefill(
            self,
            kv_cache,
            enable_trace=enable_trace,
            can_sample_on_device=can_sample_on_device,
            greedy_only=greedy_only,
        )

    def prefill_forward_text(self, *args, enable_trace=True, **kwargs):
        tokens = args[0] if args else kwargs.get("tokens")
        batch_size = tokens.shape[0] if tokens is not None else 1
        enable_trace = self._maybe_disable_pli_prefill_trace(enable_trace, batch_size=batch_size)
        if tokens is not None:
            batch_seq_len = tokens.shape[1]
            prompt_lens = kwargs.get("prompt_lens")
            start_pos = kwargs.get("start_pos")
            prompt_lens_list = prompt_lens if prompt_lens is not None else [batch_seq_len] * batch_size
            if not isinstance(prompt_lens_list, list):
                prompt_lens_list = prompt_lens_list.tolist()
            num_cached_per_user = [int(n) for n in start_pos] if start_pos is not None else [0] * len(prompt_lens_list)
            prefill_seq_lens = [
                get_padded_prefill_len(seq_len - num_cached)
                for seq_len, num_cached in zip(prompt_lens_list, num_cached_per_user)
            ]
            page_table = kwargs.get("page_table")
            can_batch_prefill = (
                page_table is not None
                and batch_size > 1
                and len(set(prefill_seq_lens)) == 1
                and self.data_parallel == 1
                and not getattr(self.model_args[0], "disable_batched_prefill", False)
                and all(n == 0 for n in num_cached_per_user)
            )
            enable_trace = resolve_gemma4_prefill_trace_enable(
                enable_trace,
                self.model[0],
                self.model_args[0],
                batch_size=batch_size,
                prefill_seq_lens=prefill_seq_lens,
                can_batch_prefill=can_batch_prefill,
            )
        return super().prefill_forward_text(*args, enable_trace=enable_trace, **kwargs)

    def _get_prefill_user_page_table(
        self,
        page_table,
        kv_cache,
        prefill_len,
        trace_enabled=False,
        prefill_seq_len=None,
        use_batched_prefill=False,
        user_id=None,
        padded_batch_size=None,
        use_full_prompt_len=False,
    ):
        """Override the shared Generator helper to size/slice the
        per-user page table to the *smallest* effective block_size in
        the model, not the cache's declared block_size.

        Background: ``Generator._get_prefill_user_page_table`` slices
        the page_table to ``cdiv(prefill_seq_len, get_block_size(kv_cache))``
        columns, where ``get_block_size`` reads ``kv_cache[0][0].shape[2]``
        — the declared block_size of layer 0's K cache. Under vLLM's
        hybrid kv-cache-groups manager that's the buffer's *allocation*
        block_size, which for Gemma4-E2B is sliding's 128 (sliding
        layers come first in the layer order and their spec wins the
        shared buffer's shape). But full-attention layers operate
        through a view with effective block_size=64 (see
        ``attention/{prefill,decode}.py``), and ``paged_fill_cache``
        validates ``input_seq_len <= max_num_blocks_per_seq *
        effective_block_size``. With the legacy slice the full layer
        sees too few blocks and the validator fires.

        Use the smallest effective block_size across all attention
        layers — same invariant the warmup ``_mock_tokens`` override
        uses — so the slice covers every layer's needs.
        """
        import torch

        if use_batched_prefill:
            return super()._get_prefill_user_page_table(
                page_table,
                kv_cache,
                prefill_len,
                trace_enabled=trace_enabled,
                prefill_seq_len=prefill_seq_len,
                use_batched_prefill=use_batched_prefill,
                user_id=user_id,
                padded_batch_size=padded_batch_size,
                use_full_prompt_len=use_full_prompt_len,
            )

        # Per-user (non-batched) path: replicate the base behavior but
        # with effective block_size instead of ``get_block_size``.
        from models.tt_transformers.tt.common import num_blocks_in_seq

        cache = kv_cache[0][0]  # layer 0, K (HMA-shared across specs)
        cache_block_size = cache.shape[2]
        cache_head_dim = cache.shape[-1]
        # Layer 0's model_args isn't accessible from here, so locate
        # the matching submodel by identity (the same trick used in
        # ``_mock_tokens``). For batch_size=1 (the only path we
        # exercise here) the submodel is index 0.
        head_dims = {layer.self_attn.config.head_dim for layer in self.model[0].layers}
        max_head_dim = max(head_dims)
        effective_block_size = cache_block_size * cache_head_dim // max_head_dim

        # Mirror the base Generator semantics: when ``use_full_prompt_len`` is
        # set (vLLM warmup runs prefill kernels on the padded prompt length, e.g.
        # a 32-token prompt becomes a 128-token kernel), size the page table to
        # the full ``prefill_len`` so it exposes blocks for the padded length.
        if use_full_prompt_len:
            target_prefill_len = prefill_len
        else:
            target_prefill_len = prefill_seq_len if prefill_seq_len is not None else prefill_len
        num_blocks = num_blocks_in_seq(target_prefill_len, effective_block_size)
        # Bounded sliding-window cache: ``paged_fill_cache`` /
        # ``paged_update_cache`` with ``cache_position_modulo`` set require
        # ``page_table.shape[1] >= sliding_window / effective_block_size``
        # so the modulo wrap can address every position in
        # ``[0, sliding_window)``. The sliced page_table needs to satisfy
        # this even when the warmup or early-prefill ``target_prefill_len``
        # is smaller than the sliding window. Unused slots are never
        # referenced at runtime; the ``-1`` padding below makes any stray
        # read produce an obvious out-of-range block ID rather than
        # silently clobbering block 0.
        if self._bounded_sliding_kv_cache:
            text_config = getattr(self.model[0].hf_config, "text_config", self.model[0].hf_config)
            sliding_window = getattr(text_config, "sliding_window", None)
            if sliding_window is not None and effective_block_size > 0:
                min_blocks_for_bounded = num_blocks_in_seq(sliding_window, effective_block_size)
                if min_blocks_for_bounded > num_blocks:
                    num_blocks = min_blocks_for_bounded
        if page_table.shape[1] < num_blocks:
            padding = torch.ones(1, num_blocks - page_table.shape[1], dtype=torch.int32) * -1
            page_table = torch.cat([page_table, padding], dim=1)
        return page_table[:, :num_blocks]

    def _mock_tokens(self, batch_size, seq_len, kv_cache, model_id):
        """Override warmup page_table sizing for the hybrid-kv-cache-groups
        path.

        Warmup must produce a page_table whose shape matches the *runtime*
        legacy page_table shape (``model_input.block_tables`` =
        ``block_tables_per_group[0]`` in the plugin), because the decode
        trace captures device tensors at warmup shapes and ``copy_host_to_device``
        asserts shape-equality on every replay. The runtime per-group
        block_table for layer 0's group has width
        ``cdiv(max_model_len, group_block_size_after_unification)``; for
        Gemma4-E2B layer 0 is sliding and the unifier doubled sliding's
        block_size from ``cache_config.block_size`` to match the larger
        full-attn page size (sliding head_dim=256 → 128 block_size;
        full head_dim=512 → 64 block_size). The cache tensor's declared
        ``shape[2]`` is that post-unification block_size, so reading
        directly from layer 0's K-cache shape gives the right value.

        The full-attention layers operate through a view with the smaller
        effective block_size (= ``cache.shape[2] * cache.shape[-1] // full_head_dim``
        = 64), but their per-layer block_table is *padded* to the same width
        by the plugin's ``_block_tables_per_layer`` — so a single warmup
        width still aligns every layer's persistent buffer. The smaller
        effective block_size narrows the kernel's ``input_seq_len <=
        max_num_blocks_per_seq * block_size`` validation budget; warmup
        chunks stay well under that limit, and full-coverage of
        ``max_model_len`` for full-attn would require sizing the per-layer
        table separately (separate work).
        """
        import torch

        from models.tt_transformers.tt.common import num_blocks_in_seq

        ret = {
            "tokens": torch.zeros(batch_size, seq_len, dtype=torch.long),
            "prompt_lens": torch.tensor([seq_len] * batch_size, dtype=torch.long),
            "empty_slots": list(range(batch_size)),
        }

        page_table_warmup = None
        if kv_cache is not None and kv_cache[model_id] is not None:
            cache = kv_cache[model_id][0][0]  # layer 0, K
            cache_block_size = cache.shape[2]
            # Match the plugin's runtime page_table width for layer 0's
            # group: ``cdiv(max_seq_len, declared_block_size)``.
            max_seq_len = self.model_args[model_id].max_seq_len
            num_blocks = num_blocks_in_seq(max_seq_len, cache_block_size)
            page_table_warmup = torch.zeros(batch_size, num_blocks, dtype=torch.int32)

        ret["page_table"] = page_table_warmup
        return ret

    # ── vLLM ``VllmModelForTextGeneration`` protocol shim ────────────────
    #
    # vLLM's ``is_text_generation_model`` predicate checks for
    # ``embed_input_ids``, ``forward(input_ids, positions)``, and
    # ``compute_logits`` on the resolved model class — that's how upstream
    # ``runner_type=="generate"`` validates a model is generative. Other TT
    # models (Gemma3, GptOss, etc.) get away without these because vLLM
    # finds an upstream torch implementation in its registry first and uses
    # *that* class for inspection, while the plugin's ``TT``-prefix logic
    # routes execution to the TT class. Gemma4 has no upstream vLLM impl,
    # so the inspection has to land on this class.
    #
    # Actual execution on the TT path goes through ``prefill_forward`` /
    # ``decode_forward`` (called by the TT runner via the
    # ``HybridAttentionForCausalLM`` overrides above), so these stubs are
    # never invoked. They exist purely to satisfy the protocol check.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "Gemma4ForCausalLM is a TT bridge; embeddings happen on TT via "
            "prefill_forward / decode_forward, not through this method."
        )

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "Gemma4ForCausalLM is a TT bridge; the TT runner invokes "
            "prefill_forward / decode_forward, not forward()."
        )

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError(
            "Gemma4ForCausalLM is a TT bridge; logits are produced on TT "
            "and surfaced through prefill_forward / decode_forward."
        )

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Build per-layer KVCacheSpec, honoring Gemma4's per-layer-type
        differences in ``head_dim`` and ``num_kv_heads``.

        The base ``HybridAttentionForCausalLM.get_kv_cache_spec`` assumes
        all layers share one ``head_size`` / ``num_kv_heads`` (only the
        sliding-vs-full *spec class* changes). That's true for Gemma3 but
        not Gemma4: sliding layers use ``head_dim`` (256 on E2B/E4B),
        full layers use ``global_head_dim`` (512). Sliding and full also
        each have their own ``num_key_value_heads`` (with the full count
        falling back to the sliding count when ``num_global_key_value_heads``
        is unset). Emitting one uniform spec made the K tensor produced
        by full-attention layers mismatch the cache shape and trip
        ``Last dim of input tensor must match last dim of cache tensor``
        in ``paged_update_cache``.

        vLLM's hybrid kv cache manager handles the resulting
        non-uniform-shape grouping fine (sliding layers form one group,
        full layers another), so the only thing that needs to differ
        between groups is the spec — block_size stays uniform.
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
            raise ValueError(
                f"{cls.__name__}.get_kv_cache_spec requires "
                "hf_config.text_config.layer_types (one of 'full_attention' / "
                "'sliding_attention' per layer); none found on this model"
            )

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
                    raise ValueError(
                        f"layer_types[{i}] is 'sliding_attention' but "
                        f"hf_config.sliding_window is None on {cls.__name__}"
                    )
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
                raise ValueError(
                    f"Unsupported layer_type {lt!r} at layer {i} on "
                    f"{cls.__name__}; expected 'full_attention' or "
                    "'sliding_attention'"
                )
        return spec_per_layer

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str = None,
    ):
        if optimizations not in (None, "performance"):
            raise ValueError("Gemma4 TT does not support custom optimization profiles")

        model_path = hf_config._name_or_path
        submesh_devices = create_submeshes(mesh_device, tt_data_parallel)

        # Opt into the bounded sliding-window KV cache path on sliding layers.
        # vLLM's hybrid manager already produces SlidingWindowSpec-shaped page
        # tables for sliding layers; the flag makes Gemma4Attention pass
        # ``cache_position_modulo=sliding_window`` to the three paged ops so
        # they correctly address the bounded physical pool. Default-off in
        # ``create_tt_model``; flipped on here because the bridge is the
        # path that actually receives bounded buffers from vLLM. Env-var
        # override ``GEMMA4_BOUNDED_SLIDING_KV_CACHE=0`` lets the legacy
        # unbounded path back in if a regression appears.
        bounded_sliding_kv_cache = os.environ.get("GEMMA4_BOUNDED_SLIDING_KV_CACHE", "1") != "0"

        model_args = []
        model = []
        state_dict = None
        for submesh in submesh_devices:
            model_args_i, model_i, _, state_dict = create_tt_model(
                mesh_device=submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                dtype=ttnn.bfloat16,
                state_dict=state_dict,
                num_layers=n_layers,
                mesh_config=None,
                paged_attention_config=None,
                create_kv_cache=False,
                model_path=model_path,
                bounded_sliding_kv_cache=bounded_sliding_kv_cache,
            )
            _patch_model_args(
                model_args_i,
                submesh,
                max_batch_size=max_batch_size // tt_data_parallel,
                max_seq_len=max_seq_len,
                model_path=model_path,
            )
            # The shared TT vLLM cache allocator reads ``model.args.optimizations``;
            # mirror the text-transformer wrappers by exposing model_args here.
            model_i.args = model_args_i
            model_args.append(model_args_i)
            model.append(model_i)

        return cls(model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args[0].weight_cache_path(ttnn.bfloat16)

    def prefill_forward(self, *args, page_tables_per_layer=None, **kwargs):
        page_tables_per_layer = self._build_per_layer_page_tables(page_tables_per_layer, kwargs.get("page_table"))
        page_tables_per_layer = self._pad_sliding_page_tables_for_bounded(page_tables_per_layer, kwargs.get("kv_cache"))
        per_submesh = self._chunk_page_tables_per_dp(page_tables_per_layer)
        # Push the per-layer block IDs into the persistent device buffers
        # *before* entering ``Generator.prefill_forward_text`` — that path
        # may execute a captured trace, which reads block IDs from the
        # persistent addresses and forbids in-trace writes. Allocation
        # happens lazily inside ``Gemma4Model._page_tables_to_ttnn`` the
        # first time the inner forward runs.
        if per_submesh is not None:
            for m, pt_for_submesh in zip(self.model, per_submesh):
                m.update_persistent_per_layer_page_tables(pt_for_submesh)
        with self._route_per_layer_page_tables(per_submesh):
            return super().prefill_forward_text(**kwargs)

    def decode_forward(self, *args, page_tables_per_layer=None, **kwargs):
        page_tables_per_layer = self._build_per_layer_page_tables(page_tables_per_layer, kwargs.get("page_table"))
        page_tables_per_layer = self._pad_sliding_page_tables_for_bounded(page_tables_per_layer, kwargs.get("kv_cache"))
        per_submesh = self._chunk_page_tables_per_dp(page_tables_per_layer)
        if per_submesh is not None:
            for m, pt_for_submesh in zip(self.model, per_submesh):
                m.update_persistent_per_layer_page_tables(pt_for_submesh)
        with self._route_per_layer_page_tables(per_submesh):
            # Skip ``HybridAttentionForCausalLM.decode_forward``, which is a
            # NotImplementedError placeholder; route to ``Generator``'s
            # actual decode implementation. ``decode_forward_text`` was
            # renamed to ``decode_forward`` in tt_transformers/generator.py
            # (commit 72217c1af4f, 2026-03-26); calling the old name now
            # raises AttributeError as soon as decode warmup runs. Use
            # the same skip pattern as the GptOssForCausalLM sibling.
            return super(HybridAttentionForCausalLM, self).decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        # Legacy uniform path (vLLM falls back here when ``get_kv_cache_spec``
        # isn't consulted). The hybrid path uses ``allocate_kv_cache_per_layer``
        # inherited from :class:`HybridAttentionForCausalLM`.
        return allocate_vllm_kv_cache(
            *args,
            **kwargs,
            dp_model=self.model,
            tt_cache_path=self.cache_path,
        )

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        """Allocate per-layer KV cache, then alias KV-shared layers to
        their source layer's buffer.

        Gemma4-E2B / -E4B have a Gemma3n-style "num_kv_shared_layers"
        optimization where the last N layers reuse an earlier layer's
        K/V instead of computing+storing their own. The model side
        encodes this via ``self.kv_shared_layer_map`` (layer_idx →
        source_idx) and ``attention/{prefill,decode}.py`` skips
        ``paged_{fill,update}_cache`` whenever a layer is flagged as
        shared. vLLM's hybrid kv-cache manager is unaware of this
        TT-specific reuse and allocates a distinct buffer for every
        layer; without the post-allocator alias the shared layers'
        SDPA reads land on zero-initialized buffers.

        Important: aliasing the buffer is *necessary but not sufficient*.
        Source and shared layers share an attention *type* (sliding or
        full), but vLLM's hybrid manager constructs more groups than
        just "one per type" — for Gemma4-E2B with 35 layers in the
        4-sliding-then-1-full pattern, vLLM produces 5 groups of 7
        layers each (4 sliding sub-groups + 1 full group), and each
        physical tensor is shared by one layer from each group. That
        means layer 13 (sliding, in group[3]) and layer 15 (sliding,
        in group[0]) have *different* per-layer page_tables, so
        aliasing only the buffer leaves layer 15 reading the wrong
        slot of the shared tensor — whatever group[0]'s layer 10 wrote
        there, not what layer 13 wrote. The buffer alias must be
        paired with a per-layer-page-table alias in
        :meth:`_block_tables_per_layer_with_kv_share` so the shared
        layer indexes the buffer the same way the source did.
        """
        kv_cache = super().allocate_kv_cache_per_layer(per_layer_specs)
        for submesh_idx, submesh_kv in enumerate(kv_cache):
            kv_shared_map = getattr(self.model[submesh_idx], "kv_shared_layer_map", None)
            if not kv_shared_map:
                continue
            for layer_idx, source_idx in kv_shared_map.items():
                submesh_kv[layer_idx] = submesh_kv[source_idx]
        return kv_cache

    def _build_per_layer_page_tables(self, page_tables_per_layer, legacy_page_table):
        """Compose the inherited per-layer broadcast/passthrough with
        the Gemma4-specific kv-share alias.

        Composition logic, kept in one place so
        :meth:`prefill_forward` / :meth:`decode_forward` each take one
        call instead of remembering to chain two helpers:

        1. :meth:`HybridAttentionForCausalLM._ensure_page_tables_per_layer`
           — broadcast a legacy single ``page_table`` to per-layer when
           the plugin only sent the legacy view (warmup, tests).
        2. :meth:`_apply_kv_share_to_per_layer_page_tables` — for every
           ``(shared_idx, source_idx)`` in the model's
           ``kv_shared_layer_map``, re-point the shared layer's
           page_table at the source's. See the
           :meth:`allocate_kv_cache_per_layer` docstring for why
           aliasing the buffer alone leaves the shared layer reading
           a different layer's slot of the shared HMA tensor.
        """
        page_tables_per_layer = self._ensure_page_tables_per_layer(page_tables_per_layer, legacy_page_table)
        return self._apply_kv_share_to_per_layer_page_tables(page_tables_per_layer)

    def _apply_kv_share_to_per_layer_page_tables(self, page_tables_per_layer):
        """Replace every kv-shared layer's per-layer page_table with
        its source layer's per-layer page_table.

        The buffer alias in :meth:`allocate_kv_cache_per_layer` makes
        ``caches[shared] is caches[source]``; this method makes
        ``page_tables[shared] is page_tables[source]``. Together they
        ensure the shared layer reads exactly the (buffer, block IDs)
        the source layer wrote — without this, the shared layer reads
        the slot in the HMA-shared buffer that the layer in its own
        kv-cache sub-group wrote, which is some other layer's K/V.
        See [[gemma4-kv-share-page-table-alias]] for the diagnosis
        path.
        """
        if not page_tables_per_layer:
            return page_tables_per_layer
        kv_shared_map = getattr(self.model[0], "kv_shared_layer_map", None) or {}
        if not kv_shared_map:
            return page_tables_per_layer
        out = list(page_tables_per_layer)
        for layer_idx, source_idx in kv_shared_map.items():
            if 0 <= layer_idx < len(out) and 0 <= source_idx < len(out):
                out[layer_idx] = out[source_idx]
        return out

    def _pad_sliding_page_tables_for_bounded(self, page_tables_per_layer, kv_cache):
        """Pad sliding-layer page tables out to ``sliding_window/block_size``
        columns so the bounded paged ops' static shape check passes on short
        prompts.

        The paged-cache kernels with ``cache_position_modulo`` set require
        ``page_table_shape[1] >= modulo / effective_block_size`` so the
        modulo wrap can address every position in ``[0, sliding_window)``.
        vLLM's hybrid kv-cache manager allocates block columns *lazily* as
        the prompt fills, so for a short prompt (e.g. 256 tokens at
        block_size=64 → 4 columns) the per-layer page_table is smaller than
        the bounded path needs and the kernel correctly TT_FATALs.

        Pad each sliding layer's page_table by repeating the last valid
        column. The padded slots are never accessed at runtime: every
        active position is < ``current_pos``, and vLLM has already
        allocated real blocks for positions in ``[0, current_pos]``. The
        repeat (vs. zeros) is purely defensive — if a future code path
        dereferenced a padded slot it would land on a real allocated block
        instead of clobbering block 0.

        Full-attention layers are left alone; ``cache_position_modulo`` is
        only set on sliding layers, so their page-table sizing is governed
        by the legacy ``input_shape[2]`` check that doesn't have this
        problem.

        No-op when:

          - bounded mode is off (``GEMMA4_BOUNDED_SLIDING_KV_CACHE=0``)
          - the model has no sliding layers / no ``sliding_window``
          - ``kv_cache`` is not available (warmup pathways before
            allocation, or non-paged callers); the check below will catch
            any genuine size mismatch
        """
        if not self._bounded_sliding_kv_cache:
            return page_tables_per_layer
        if not page_tables_per_layer:
            return page_tables_per_layer
        model = self.model[0]
        text_config = getattr(model.hf_config, "text_config", model.hf_config)
        sliding_window = getattr(text_config, "sliding_window", None)
        layer_types = getattr(text_config, "layer_types", None)
        if sliding_window is None or layer_types is None:
            return page_tables_per_layer

        # Read block_size off the first available K-cache. Layout per the
        # vLLM bridge's allocator: ``kv_cache[submesh_idx][layer_idx][0]``
        # is the K cache for that layer; its declared ``shape[2]`` is the
        # block_size vLLM used when sizing the page tables it just gave us.
        block_size = None
        if kv_cache is not None:
            try:
                block_size = int(kv_cache[0][0].shape[2])
            except (TypeError, IndexError, AttributeError):
                block_size = None
        if block_size is None:
            # Fallback: read off whatever layer-attached cache exists.
            try:
                block_size = int(model.layers[0].self_attn.kv_cache[0].shape[2])
            except (TypeError, IndexError, AttributeError):
                return page_tables_per_layer  # Truly can't determine; let the kernel surface the size mismatch.

        if sliding_window % block_size != 0:
            return page_tables_per_layer  # Misalignment — bounded path can't be used; let the kernel fail loudly.
        target_cols = sliding_window // block_size

        out = []
        for i, pt in enumerate(page_tables_per_layer):
            if (
                pt is None
                or i >= len(layer_types)
                or layer_types[i] != "sliding_attention"
                or not hasattr(pt, "shape")
                or pt.shape[-1] >= target_cols
            ):
                out.append(pt)
                continue
            pad_amount = target_cols - pt.shape[-1]
            last_col = pt[..., -1:].repeat(*([1] * (pt.dim() - 1)), pad_amount)
            out.append(torch.cat([pt, last_col], dim=-1))
        return out
