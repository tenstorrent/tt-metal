# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import ttnn

import ttml
from ttml.common.config import DeviceConfig


class LlamaGRPOCompleter:
    """Llama completer backed by a tt-transformers ``Transformer``.

    Public surface is intentionally minimal:

      * :meth:`__init__` opens the mesh device, loads the tokenizer, and
        builds the ``ModelArgs`` describing the architecture. No weights
        are loaded yet.
      * :meth:`load_weights` builds (or rebuilds) the ``Transformer`` and
        ``Generator`` from a Meta-style state dict. Call this at least
        once before using the underlying model.
      * :meth:`generate` runs prefill + decode for a batch of tokenised
        prompts, modelled on
        ``models/tt_transformers/demo/simple_text_demo.py``.
    """

    def setup_device(self, device_config: DeviceConfig) -> Any:
        """Enable fabric (multi-device) and open the AutoContext mesh device.

        Returns the resulting ``ttnn.MeshDevice``. Tests may override this
        method (e.g. via ``monkeypatch.setattr``) to reuse a device that was
        already opened earlier in the process by returning
        ``ttml.autograd.AutoContext.get_instance().get_device()`` without
        calling ``open_device`` again.
        """
        if device_config.total_devices() > 1:
            ttml.core.distributed.enable_fabric(device_config.total_devices())
        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)
        return autograd_ctx.get_device()

    def __init__(
        self,
        device_config: DeviceConfig,
        model_source: str,
        max_batch_size: int,
        *,
        max_seq_len: int = 2048,
        instruct: bool = True,
        dtype: Any = None,
    ) -> None:
        """Open the device and build the tt-transformers ``ModelArgs``.

        Args:
            device_config: Mesh / device configuration.
            model_source: HuggingFace model id (also used as ``HF_MODEL``
                so ``ModelArgs`` can resolve the architecture).
            max_batch_size: Max batch size the ``Transformer`` will support.
                Required: this is the ceiling for :meth:`generate` and is
                baked into the model's compiled traces, so there is no
                meaningful default.
            max_seq_len: Max sequence length the ``Transformer`` will support.
            instruct: Whether to use the instruct chat template in
                ``ModelArgs``.
            dtype: ttnn weight dtype. Defaults to ``ttnn.bfloat8_b``.
        """
        import torch

        from models.tt_transformers.tt.common import PagedAttentionConfig
        from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

        if dtype is None:
            dtype = ttnn.bfloat8_b

        self._mesh_device: Any = self.setup_device(device_config)
        self._model_source: str = model_source

        # ``ModelArgs`` reads the model id from the env var.
        os.environ["HF_MODEL"] = model_source

        self._dtype: Any = dtype
        self._max_batch_size: int = max_batch_size

        self._model_args = ModelArgs(
            self._mesh_device,
            instruct=instruct,
            max_batch_size=max_batch_size,
            optimizations=lambda ma: DecodersPrecision.accuracy(ma.n_layers, ma.model_name),
            max_seq_len=max_seq_len,
            cache_hf=True,
        )
        # Reuse the tokenizer created by ModelArgs so stop tokens and
        # fallbacks match simple_text_demo / tt-transformers behavior.
        self._tokenizer: Any = self._model_args.tokenizer

        # Paged attention KV cache: kept simple (single contiguous identity
        # mapping). The page_table is a host torch tensor — that's what the
        # ``Generator`` API expects, and re-using the same tensor identity
        # across calls keeps the on-device sampling-feedback chain intact.
        #
        # Size the global pool to ``max(MIN_NUM_BLOCKS, max_batch_size *
        # blocks_per_user_for_max_seq_len)`` and split it evenly across
        # users. ``MIN_NUM_BLOCKS = 1024`` matches the demo / PCC test
        # configs that are known to work end-to-end on this model
        # (``simple_text_demo.py`` and ``gen_hf_ttt.py`` both pin
        # ``max_num_blocks=1024`` regardless of how short the actual
        # sequence is). Tighter sizing has empirically produced incoherent
        # output and may trip undocumented kernel assumptions, so we
        # over-provision rather than minimise.
        block_size = 32
        MIN_NUM_BLOCKS = 1024
        required_blocks_per_user = (max_seq_len + block_size - 1) // block_size
        max_num_blocks = max(MIN_NUM_BLOCKS, max_batch_size * required_blocks_per_user)
        blocks_per_user = max_num_blocks // max_batch_size
        # Re-snap so ``max_num_blocks`` is an exact multiple of
        # ``max_batch_size`` (the page_table reshape requires this).
        max_num_blocks = blocks_per_user * max_batch_size
        self._paged_attention_config = PagedAttentionConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )
        self._paged_cache_max_seq_len = block_size * blocks_per_user
        self._page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(max_batch_size, blocks_per_user)

        self._model: Any = None
        self._generator: Any = None
        self._kv_cache: Any = None
        self._cache_dir_tmp: Optional[tempfile.TemporaryDirectory] = None

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model(self) -> Any:
        """The tt-transformers ``Transformer`` (or ``None`` until weights are loaded)."""
        return self._model

    @property
    def generator(self) -> Any:
        """The tt-transformers ``Generator`` (or ``None`` until weights are loaded)."""
        return self._generator

    @property
    def model_args(self) -> Any:
        return self._model_args

    @property
    def page_table(self) -> Any:
        return self._page_table

    @property
    def kv_cache(self) -> Any:
        """Per-layer (k_cache, v_cache) list used by paged attention."""
        return self._kv_cache

    def load_weights(self, state_dict: Dict[str, Any]) -> None:
        """Build (or rebuild) the ``Transformer`` mirror from a Meta-style state dict.

        The state dict must use Meta naming (post HF -> Meta conversion), i.e.
        the format produced by
        ``models.tt_transformers.tt.model_config.ModelArgs.load_state_dict()``.

        Idempotent: if a previous mirror exists it is dropped (along with
        its temporary weight-cache directory) before a new one is built.
        """
        from models.tt_transformers.tt.generator import Generator
        from models.tt_transformers.tt.model import Transformer

        # Drop the previous Transformer/Generator/KV-cache references first
        # so device memory is freed before we allocate the next Transformer
        # — otherwise we'd briefly hold two copies on device, which would
        # OOM on small parts in a GRPO weight-update loop. Reassigning the
        # ``TemporaryDirectory`` similarly retires the old weight-cache dir
        # via its weakref finalizer.
        self._model = None
        self._generator = None
        self._kv_cache = None

        # Per-instance throwaway weight cache. ``weight_cache_path=None`` is
        # the eventual goal but currently crashes inside ``mlp.py`` and
        # ``embedding.py`` (their ``cache_name`` lambdas do ``None / "..."``
        # when ``args.dummy_weights == False``). Using a fresh directory
        # per call also guarantees stale tilized weights from a previous
        # ``load_weights`` never get reused — important when GRPO updates
        # weights between training steps.
        cache_dir_tmp = tempfile.TemporaryDirectory(prefix="ttt_grpo_completer_")
        cache_dir = Path(cache_dir_tmp.name)

        model = Transformer(
            args=self._model_args,
            mesh_device=self._mesh_device,
            dtype=self._dtype,
            state_dict=state_dict,
            weight_cache_path=cache_dir,
            paged_attention_config=self._paged_attention_config,
        )

        # We deliberately do NOT require ``_supports_on_device_sampling``.
        # On non-Galaxy / non-Llama-3.1-8B configs the on-device sampler runs
        # the multinomial path even when we ask for greedy
        # (``_allow_force_argmax_sampling=False``); empirically this produces
        # incoherent output and is non-reproducible run-to-run on Llama-3.2-1B
        # — same class of bug as ``simple_text_demo.py``'s
        # ``models_not_supported_for_device_sampling`` workaround
        # (https://github.com/tenstorrent/tt-metal/issues/34763). :meth:`generate`
        # therefore always sources the next token via host sampling, which is
        # bit-exact deterministic for greedy and matches what
        # ``gen_hf_ttt.py`` already validates for this model.

        # Build paged-attention KV-cache handles the same way ``create_tt_model``
        # does in ``models/tt_transformers/tt/common.py``: one (k, v) tuple per
        # decoder layer, looked up via the public ``layer.attention.layer_past``
        # attribute.
        kv_cache = [l.attention.layer_past for l in model.layers]

        generator = Generator(
            model=[model],
            model_args=[self._model_args],
            mesh_device=self._mesh_device,
            tokenizer=self._tokenizer,
        )

        self._model = model
        self._generator = generator
        self._kv_cache = kv_cache
        self._cache_dir_tmp = cache_dir_tmp

    def _reset_kv_cache(self) -> None:
        """Zero the paged-attention KV cache before a new prompt batch.

        Mirrors ``simple_text_demo.py``: each layer's
        ``attention.layer_past`` is a ``(k_cache, v_cache)`` tuple of device
        tensors; we overwrite their contents in place so subsequent attention
        kernels read zeros at unfilled positions. ``Generator.prev_page_table``
        is also cleared so the next prefill rebinds trace inputs.
        """
        for layer in self._model.layers:
            k_cache, v_cache = layer.attention.layer_past
            ttnn.mul(k_cache, 0, output_tensor=k_cache)
            ttnn.mul(v_cache, 0, output_tensor=v_cache)
        self._generator.prev_page_table = None

    def _stop_token_ids(self) -> set:
        """Stop tokens used to truncate completions.

        Includes ``eos_token_id``, ``pad_token_id``, and the common Llama-3
        chat-template terminators (``<|eot_id|>``, ``<|end_of_text|>``,
        ``<|eom_id|>``) when the tokenizer recognises them.
        """
        tok = self._tokenizer
        ids: set = set()
        if tok.eos_token_id is not None:
            ids.add(int(tok.eos_token_id))
        if tok.pad_token_id is not None:
            ids.add(int(tok.pad_token_id))
        for s in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"):
            tid = tok.convert_tokens_to_ids(s)
            if tid is not None and tid >= 0 and tid != tok.unk_token_id:
                ids.add(int(tid))
        return ids

    def _prepare_prompt_batch(
        self, prompts: List[List[int]], max_new_tokens: int
    ) -> tuple[List[List[int]], List[int], int]:
        """Normalize/clip prompts and pad to ``max_batch_size`` for decode.

        ``Transformer.prepare_decode_inputs_host`` requires decode batch size
        to exactly match ``ModelArgs.max_batch_size``. We therefore pad with
        inert one-token prompts when ``len(prompts) < max_batch_size`` and
        ignore those padded slots in the returned completions.
        """
        assert max_new_tokens >= 0, "max_new_tokens must be non-negative"

        active_batch_size = len(prompts)
        assert 0 < active_batch_size <= self._max_batch_size, (
            f"generate() got {active_batch_size} prompts but completer was built with "
            f"max_batch_size={self._max_batch_size}"
        )

        normalized_prompts = [[int(tok) for tok in prompt] for prompt in prompts]
        prompt_lens = [len(p) for p in normalized_prompts]
        assert min(prompt_lens) > 0, "empty prompts are not supported"

        max_prefill_len = self._model_args.max_seq_len - max_new_tokens
        assert (
            max_prefill_len > 0
        ), f"max_new_tokens ({max_new_tokens}) must be smaller than max_seq_len ({self._model_args.max_seq_len})"

        if max(prompt_lens) > max_prefill_len:
            normalized_prompts = [p[-max_prefill_len:] for p in normalized_prompts]
            prompt_lens = [len(p) for p in normalized_prompts]

        max_prompt_len = max(prompt_lens)
        assert max_prompt_len + max_new_tokens <= self._model_args.max_seq_len, (
            f"prompt prefill tokens ({max_prompt_len}) + decode tokens ({max_new_tokens}) "
            f"must be <= max_seq_len ({self._model_args.max_seq_len})"
        )
        assert max_prompt_len + max_new_tokens <= self._paged_cache_max_seq_len, (
            f"prompt prefill tokens ({max_prompt_len}) + decode tokens ({max_new_tokens}) "
            f"must be <= paged-cache capacity ({self._paged_cache_max_seq_len})"
        )

        if active_batch_size < self._max_batch_size:
            filler_token = self._tokenizer.pad_token_id
            if filler_token is None:
                filler_token = self._tokenizer.eos_token_id
            if filler_token is None or filler_token < 0:
                filler_token = 0
            filler_prompt = [int(filler_token)]
            pad_slots = self._max_batch_size - active_batch_size
            normalized_prompts.extend([filler_prompt] * pad_slots)
            prompt_lens.extend([1] * pad_slots)

        return normalized_prompts, prompt_lens, active_batch_size

    def generate(
        self,
        prompts: List[List[int]],
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: Optional[int] = None,
        enable_trace: bool = True,
        stop_at_eos: bool = True,
    ) -> List[List[int]]:
        """Generate continuations for a batch of tokenised prompts.

        The structural flow follows
        ``models/tt_transformers/demo/simple_text_demo.py`` (zero the KV
        cache, pad prompts to a common length, prefill in one shot via
        :meth:`Generator.prefill_forward_text`, then loop
        :meth:`Generator.decode_forward` with ``reset_batch=True`` on the
        first step), but **sampling is always done on host**. We pass
        ``sampling_params=None`` to both prefill and decode so they return
        logits, then take ``torch.argmax`` (greedy) or top-p multinomial
        on host via :func:`models.tt_transformers.tt.common.sample_host`.
        See the comment in :meth:`load_weights` for why the on-device
        sampling path is unsuitable on this model/hardware combo.

        Args:
            prompts: Tokenised prompts; ``len(prompts)`` must not exceed the
                ``max_batch_size`` passed to :meth:`__init__`. Empty prompts
                are not supported.
            max_new_tokens: Hard cap on generated tokens per user.
            temperature: Sampling temperature. ``0.0`` means greedy
                (host argmax); otherwise top-p multinomial on host.
            top_p: Nucleus sampling cutoff (ignored when ``temperature == 0``).
            seed: Optional torch RNG seed for reproducible non-greedy
                sampling. Greedy is deterministic regardless of ``seed``.
            enable_trace: Whether prefill/decode capture and execute traces
                (matches the demo default; only affects performance, not
                correctness).
            stop_at_eos: When ``True``, truncate each user's completion at
                the first stop token (and stop generating once all users
                are done). When ``False``, run for exactly ``max_new_tokens``
                steps and return the raw output.

        Returns:
            One list of generated token IDs per prompt (prompt tokens are
            not included).
        """
        assert self._model is not None and self._generator is not None, "Call load_weights(state_dict) first"

        import torch

        from models.tt_transformers.tt.common import sample_host

        if max_new_tokens == 0:
            return [[] for _ in prompts]

        prompts, prompt_lens, active_batch_size = self._prepare_prompt_batch(prompts, max_new_tokens)
        batch_size = len(prompts)
        max_prompt_len = max(prompt_lens)

        # Pad prompts to a common length. Padding token is irrelevant for
        # attention because ``prompt_lens`` tells the model exactly how many
        # real tokens each user has.
        pad_id = int(self._tokenizer.pad_token_id) if self._tokenizer.pad_token_id is not None else 0
        input_tokens_prefill_pt = torch.full((batch_size, max_prompt_len), pad_id, dtype=torch.int32)
        for i, p in enumerate(prompts):
            input_tokens_prefill_pt[i, : len(p)] = torch.tensor(p, dtype=torch.int32)

        # Reproducible non-greedy sampling: ``sample_host`` uses
        # ``torch.multinomial`` which reads the global torch RNG.
        if seed is not None:
            torch.manual_seed(seed)

        kv_cache = [self._kv_cache]

        self._reset_kv_cache()

        _ = self._generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self._page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            sampling_params=None,
            warmup_prefill=True,
            enable_trace=enable_trace,
        )
        # Second call: no warmup, clean prefill (matching demo pattern)
        prefill_logits = self._generator.prefill_forward_text(
            input_tokens_prefill_pt,
            page_table=self._page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            sampling_params=None,
            warmup_prefill=False,
            enable_trace=enable_trace,
        )

        # ``sample_host`` → next-token tensor of shape ``[B, 1]``. For
        # ``temperature == 0`` this is ``torch.argmax(logits, dim=-1)``,
        # which is bit-exact deterministic across runs.
        _, prefilled_token = sample_host(prefill_logits, temperature=temperature, top_p=top_p, on_host=True)

        completions: List[List[int]] = [[] for _ in range(batch_size)]
        user_done = [False] * batch_size
        for u in range(active_batch_size, batch_size):
            user_done[u] = True
        stop_ids = self._stop_token_ids() if stop_at_eos else set()

        for u in range(batch_size):
            if user_done[u]:
                continue
            tok = int(prefilled_token[u].item())
            if stop_at_eos and tok in stop_ids:
                user_done[u] = True
            else:
                completions[u].append(tok)

        if all(user_done) or max_new_tokens <= 1:
            return completions[:active_batch_size]

        current_pos = torch.tensor(prompt_lens, dtype=torch.int32)
        out_tok = prefilled_token  # shape [B, 1]

        # decode loop: -1 because the prefill already produced the first
        # generated token above.
        for step in range(max_new_tokens - 1):
            # ``decode_forward`` always returns ``(first, log_probs)``; with
            # ``sampling_params=None`` ``first`` is the logits tensor of
            # shape ``[B, 1, vocab_size]`` (see
            # ``Transformer.process_output_decode``).
            decode_logits, _ = self._generator.decode_forward(
                out_tok,
                current_pos,
                page_table=self._page_table,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                sampling_params=None,
                reset_batch=(step == 0),
            )
            _, sampled = sample_host(decode_logits, temperature=temperature, top_p=top_p, on_host=True)
            out_tok = sampled  # already shape [B, 1]
            current_pos = current_pos + 1

            for u in range(batch_size):
                if user_done[u]:
                    continue
                tok = int(out_tok[u].item())
                if stop_at_eos and tok in stop_ids:
                    user_done[u] = True
                else:
                    completions[u].append(tok)

            if stop_at_eos and all(user_done):
                break

        return completions[:active_batch_size]
