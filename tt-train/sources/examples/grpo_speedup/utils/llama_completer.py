# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ttnn

import ttml
from ttml.common.config import DeviceConfig, TransformerConfig
from ttml.common.utils import no_grad
from ttml.models import RunnerType, WeightTyingType
from ttml.models.llama import LlamaConfig, LlamaRopeScalingConfig, load_from_safetensors
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from ttml.trainers.grpo_trainer import GRPOCompleter
from .llama_overrides import LlamaCompositeKV


TILE_SIZE = 32
SAMPLE_SEED = 42

# Chunked async readback: every CHUNK decode steps the loop fires a
# non-blocking d2h for the just-finished chunk, then on the next chunk
# boundary host-synchronises that read and checks for stop tokens. The check
# thus lags compute by CHUNK steps, which is the cost of avoiding any
# per-step host sync.
CHUNK = 32


@dataclass
class LlamaCompletionCtx:
    max_tokens_to_complete: int
    temperature: float
    completions_per_prompt: int = 1
    _tokenizer: Any = None
    _pad_token: Optional[int] = None


def deallocate_tensors(tensors: Any) -> None:
    if tensors is None:
        return
    if not isinstance(tensors, (list, tuple)):
        tensors = [tensors]
    for t in tensors:
        if t is None:
            continue
        if isinstance(t, ttml.autograd.Tensor):
            ttnn.deallocate(t.get_value(), force=True)
        elif isinstance(t, ttnn.Tensor):
            ttnn.deallocate(t, force=True)


def load_checkpoint(model: Any, checkpoint_path: str, dp_mapper: Any = None) -> None:
    from safetensors.numpy import load_file
    import ml_dtypes

    checkpoint = load_file(checkpoint_path)
    parameters = model.parameters()
    loaded, missing = 0, []

    for name, param in parameters.items():
        if name in checkpoint:
            arr = checkpoint[name].astype(ml_dtypes.bfloat16)
            if arr.ndim == 1:
                arr = arr.reshape(1, 1, 1, -1)
            elif arr.ndim == 2:
                arr = arr.reshape(1, 1, arr.shape[0], arr.shape[1])
            restored = ttml.autograd.Tensor.from_numpy(arr, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, dp_mapper)
            param.assign(restored)
            loaded += 1
        else:
            missing.append(name)

    print(f"Loaded {loaded}/{len(parameters)} parameters from {checkpoint_path}")
    if missing:
        print(f"Warning: {len(missing)} parameters not found in checkpoint:")
        for n in missing:
            print(f"  - {n}")


def _async_read_to_host(tensors: List[Any], mesh_device: Any) -> Tuple[List[Any], Any]:
    """Issue non-blocking d2h reads for ``tensors`` on the single command queue.

    Returns ``(host_tensors, event)``. The caller must call
    ``event_synchronize(event)`` before consuming ``host_tensors``; deallocating
    the source ``tensors`` before then races with the in-flight DMA.
    """
    hosts = [t.cpu(blocking=False) for t in tensors]
    done = ttnn.record_event(mesh_device=mesh_device, cq_id=0)
    return hosts, done


def _round_up(x: int) -> int:
    return ((x + TILE_SIZE - 1) // TILE_SIZE) * TILE_SIZE


class LlamaGRPOCompleter(GRPOCompleter):
    """Llama-specific completion engine.

    Handles autoregressive generation and forward passes for Llama models.
    All mutable generation state (KV cache, batch dims) lives here, not
    on :class:`LlamaCompletionCtx`.

    Args:
        ctx: Generation parameters. ``_tokenizer`` and ``_pad_token`` are set
            automatically from ``model_source``.
        transformer_config: Model architecture config (a
            :class:`ttml.common.config.TransformerConfig` instance, typically
            built via ``get_model_config``).
        device_config: Device mesh config (a
            :class:`ttml.common.config.DeviceConfig` instance). Device
            initialisation (``enable_fabric``, ``open_device``) is delegated
            to :meth:`setup_device`, which the constructor calls. Subclasses
            and tests may override ``setup_device`` to swap that behaviour
            (e.g. reuse an already-open device).
        model_source: HuggingFace model ID or path to a local directory
            containing ``model.safetensors``.
    """

    def setup_device(self, device_config: DeviceConfig) -> Any:
        """Enable fabric (multi-device) and open the AutoContext mesh device.

        Returns the resulting ``ttnn.MeshDevice``. Tests may override this
        method (e.g. via ``monkeypatch.setattr``) to reuse a device that was
        already opened earlier in the process by returning
        ``ttml.autograd.AutoContext.get_instance().get_device()`` without
        calling ``open_device`` again. Overrides must return a mesh whose
        topology matches ``device_config``; otherwise the cached
        ``_mesh_device``/``_num_devices`` will be inconsistent with the rest
        of the completer.
        """
        if device_config.total_devices() > 1:
            ttml.core.distributed.enable_fabric(device_config.total_devices())
        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        autograd_ctx.open_device(device_config.mesh_shape, device_config.device_ids)
        return autograd_ctx.get_device()

    def __init__(
        self,
        ctx: LlamaCompletionCtx,
        transformer_config: TransformerConfig,
        device_config: DeviceConfig,
        model_source: str,
    ) -> None:
        tf_config = transformer_config
        dev_config = device_config

        # Cache the device + parallelism state on ``self`` rather than going
        # through ``AutoContext`` on every tensor upload. The completer (and
        # the trainer that drives it) only handles single-device or DDP today;
        # tensor parallelism is intentionally not supported here.
        mesh_device: Any = self.setup_device(dev_config)
        autograd_ctx = ttml.autograd.AutoContext.get_instance()
        self._mesh_device: Any = mesh_device
        self._num_devices: int = mesh_device.get_num_devices()

        tokenizer = AutoTokenizer.from_pretrained(model_source)
        tf_config.vocab_size = len(tokenizer)

        rope_scaling = LlamaRopeScalingConfig(
            scaling_factor=getattr(tf_config, "scaling_factor", 0.0) or 0.0,
            high_freq_factor=getattr(tf_config, "high_freq_factor", 4.0) or 4.0,
            low_freq_factor=getattr(tf_config, "low_freq_factor", 1.0) or 1.0,
            original_context_length=getattr(tf_config, "original_context_length", 0) or 0,
        )

        runner_type = RunnerType.from_string(str(tf_config.runner_type))
        weight_tying = WeightTyingType.Disabled
        if tf_config.weight_tying:
            weight_tying = WeightTyingType.from_string(str(tf_config.weight_tying))

        llama_cfg = LlamaConfig(
            hidden_size=tf_config.embedding_dim,
            intermediate_size=tf_config.intermediate_dim,
            num_hidden_layers=tf_config.num_blocks,
            num_attention_heads=tf_config.num_heads,
            num_key_value_heads=tf_config.num_groups,
            vocab_size=len(tokenizer),
            max_position_embeddings=tf_config.max_sequence_length,
            rope_theta=tf_config.theta or 10000.0,
            attention_dropout=tf_config.dropout_prob,
            mlp_dropout=tf_config.dropout_prob,
            runner_type=runner_type,
            weight_tying=weight_tying,
            rope_scaling=rope_scaling,
        )

        tt_model = LlamaCompositeKV(llama_cfg)

        if dev_config.enable_ddp:
            autograd_ctx.initialize_parallelism_context(
                ttml.autograd.DistributedConfig(enable_ddp=True, enable_tp=False)
            )

        # Resolve DDP state once after the parallelism context is (maybe)
        # initialised so callers can use the cached mapper/composer/flags
        # without re-querying ``AutoContext``.
        self._ddp_enabled: bool = (
            autograd_ctx.is_parallelism_context_initialized()
            and autograd_ctx.get_parallelism_context().is_ddp_enabled()
        )
        self._dp_mapper: Any = (
            ttml.core.distributed.shard_tensor_to_mesh_mapper(mesh_device, 0) if self._ddp_enabled else None
        )
        self._dp_composer: Any = (
            ttml.core.distributed.concat_mesh_to_tensor_composer(mesh_device, 0) if self._ddp_enabled else None
        )

        local_safetensors = os.path.isdir(model_source) and any(
            f == "model.safetensors" for f in os.listdir(model_source)
        )
        if local_safetensors:
            logging.info("Loading model from local safetensors: %s", model_source)
            load_checkpoint(tt_model, model_source, dp_mapper=self._dp_mapper)
        else:
            logging.info("Downloading model from HuggingFace: %s", model_source)
            model_repo_path = snapshot_download(
                repo_id=model_source,
                allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"],
            )
            load_from_safetensors(tt_model, model_repo_path, llama_cfg)

        ctx._tokenizer = tokenizer
        if ctx._pad_token is None:
            ctx._pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        self._ctx = ctx
        self._model = tt_model
        self.transformer_config = tf_config
        self._model_source: str = model_source

        self._kv_cache: Any = None
        self._kv_cache_B: int = 0

        # tt-transformers mirror state (populated by ``init_model``). Held
        # alongside the ttml model so the completer can hold both copies on the
        # same mesh device (single-device, two-copies bringup; see step 1 of
        # the ttml -> tt-transformers bridge).
        #
        # ``_tt_generator`` owns the prefill/decode trace lifecycle and on-device
        # sampling. ``_tt_kv_cache`` is the per-layer (k_cache, v_cache) list
        # used by paged attention (built via the public
        # ``[l.attention.layer_past for l in model.layers]`` pattern from
        # ``models/tt_transformers/tt/common.py``). ``_tt_page_table`` is kept
        # as a host torch tensor — that's what the Generator API expects, and
        # passing the same identity each call avoids tripping the Generator's
        # ``prev_page_table`` content check (which would otherwise refresh the
        # trace inputs from host and clobber the on-device sampling-feedback
        # chain).
        self._tt_model: Any = None
        self._tt_model_args: Any = None
        self._tt_generator: Any = None
        self._tt_kv_cache: Any = None
        self._tt_page_table: Any = None
        self._tt_model_cache_dir: Optional[Path] = None

    @property
    def tokenizer(self) -> Any:
        return self._ctx._tokenizer

    @property
    def model(self) -> Any:
        """The underlying ttml Llama model used for training."""
        return self._model

    @property
    def tt_model(self) -> Any:
        """The tt-transformers ``Transformer`` mirror (or ``None`` if not initialized).

        Populated by :meth:`init_model`; once set, :meth:`generate` (and
        therefore :meth:`generate_str`) dispatches to the tt-transformers
        path instead of the ttml one. Distinct from :attr:`model`, which is
        the ttml model used for training.
        """
        return self._tt_model

    # ------------------------------------------------------------------
    # tt-transformers bridge (step 1: model init only)
    # ------------------------------------------------------------------

    def init_model(
        self,
        state_dict: Dict[str, Any],
        *,
        max_seq_len: int = 2048,
        dtype: Any = None,
        instruct: bool = True,
    ) -> None:
        """Initialize a tt-transformers ``Transformer`` mirror from a Meta-style state dict.

        The state dict must use Meta naming (post HF -> Meta conversion), i.e.
        the format produced by
        ``models.tt_transformers.tt.model_config.ModelArgs.load_state_dict()``.
        Eventually the dict will come from a ttml weight dump so the mirror
        tracks the ttml model. For step 1 of the bridge this method only
        builds the model; weight transfer ttml -> tt-transformers lands in a
        later step (see :meth:`update_weights`).

        After this returns, calls to :meth:`generate` (and therefore
        :meth:`generate_str`) automatically dispatch to the tt-transformers
        path instead of the ttml one.

        Idempotent: if a previous mirror exists it is torn down (including
        its temporary weight-cache directory) before a new one is built.

        Args:
            state_dict: Meta-style host state dict (torch tensors).
            max_seq_len: Max sequence length the ``Transformer`` will support.
            dtype: ttnn weight dtype. Defaults to ``ttnn.bfloat8_b`` to match
                ``pcc_hf_ttml_ttt.py``.
            instruct: Whether to use the instruct chat template in
                ``ModelArgs``.
        """
        import torch

        from models.tt_transformers.tt.common import PagedAttentionConfig
        from models.tt_transformers.tt.generator import Generator
        from models.tt_transformers.tt.model import Transformer
        from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs

        if dtype is None:
            dtype = ttnn.bfloat8_b

        if self._tt_model is not None:
            self._teardown_tt_model()

        # ``ModelArgs`` reads the model id from the env var.
        os.environ["HF_MODEL"] = self._model_source

        batch_size = 1  # step 1 supports batch_size=1 only

        args = ModelArgs(
            self._mesh_device,
            instruct=instruct,
            max_batch_size=batch_size,
            optimizations=lambda ma: DecodersPrecision.accuracy(ma.n_layers, ma.model_name),
            max_seq_len=max_seq_len,
            cache_hf=True,
        )

        # Paged attention KV cache: kept simple (single contiguous identity
        # mapping) because we only run one user at a time. The page_table is a
        # host torch tensor — that's what ``Generator.prefill_forward_text``
        # and ``decode_forward`` expect, and the ``Generator`` content-checks
        # it at every decode step to decide whether to refresh trace inputs.
        # Re-using the same tensor identity across calls keeps the on-device
        # sampling-feedback chain intact.
        paged_attention_config = PagedAttentionConfig(block_size=32, max_num_blocks=1024)
        page_table = torch.arange(paged_attention_config.max_num_blocks, dtype=torch.int32).reshape(
            batch_size, paged_attention_config.max_num_blocks // batch_size
        )

        # Per-instance throwaway weight cache. ``weight_cache_path=None`` is
        # the eventual goal but currently crashes inside ``mlp.py`` and
        # ``embedding.py`` (their ``cache_name`` lambdas do ``None / "..."``
        # when ``args.dummy_weights == False``). Until that's fixed upstream,
        # we use a unique directory and clean it up in :meth:`_teardown_tt_model`
        # so stale tilized weights from a prior init never get reused.
        cache_dir = Path(tempfile.mkdtemp(prefix="ttt_grpo_completer_"))

        tt_model = Transformer(
            args=args,
            mesh_device=self._mesh_device,
            dtype=dtype,
            state_dict=state_dict,
            weight_cache_path=cache_dir,
            paged_attention_config=paged_attention_config,
        )

        # On-device sampling is required for the fast decode path that keeps
        # tokens on device between steps. If the mesh shape / vocab size /
        # prefetcher combination doesn't satisfy
        # ``Transformer._supports_on_device_sampling`` (see ``model.py``),
        # ``decode_forward(sampling_params=...)`` would silently fall back to
        # logits-on-host and our async pattern would not save anything.
        # Fail loudly here instead.
        if not getattr(tt_model, "_supports_on_device_sampling", False):
            shutil.rmtree(cache_dir, ignore_errors=True)
            raise RuntimeError(
                "tt-transformers Transformer was built without on-device sampling support; "
                "the fast decode path requires it. Check mesh shape, vocab size, and prefetcher "
                "settings (see Transformer._supports_on_device_sampling in model.py)."
            )

        # Build paged-attention KV-cache handles the same way ``create_tt_model``
        # does in ``models/tt_transformers/tt/common.py``: one (k, v) tuple per
        # decoder layer, looked up via the public ``layer.attention.layer_past``
        # attribute. ``Generator`` expects a list per data-parallel rank, so we
        # wrap in another list for the single-rank case.
        tt_kv_cache = [l.attention.layer_past for l in tt_model.layers]

        generator = Generator(
            model=[tt_model],
            model_args=[args],
            mesh_device=self._mesh_device,
            tokenizer=self.tokenizer,
        )

        self._tt_model = tt_model
        self._tt_model_args = args
        self._tt_generator = generator
        self._tt_kv_cache = tt_kv_cache
        self._tt_page_table = page_table
        self._tt_model_cache_dir = cache_dir

    def update_weights(self, state_dict: Dict[str, Any]) -> None:
        """Update the tt-transformers mirror from a Meta-style state dict.

        Step 1 of the bridge: not yet implemented. Until weight transfer
        lands, callers must rebuild via :meth:`init_model` (full reconstruction;
        slow because the program cache is rebuilt and weights re-tilized).
        """
        raise NotImplementedError("update_weights is not yet implemented; rebuild via init_model(state_dict)")

    def _teardown_tt_model(self) -> None:
        """Drop the tt-transformers mirror and remove its weight-cache dir."""
        self._tt_model = None
        self._tt_model_args = None
        self._tt_generator = None
        self._tt_kv_cache = None
        self._tt_page_table = None
        if self._tt_model_cache_dir is not None:
            shutil.rmtree(self._tt_model_cache_dir, ignore_errors=True)
            self._tt_model_cache_dir = None

    def generate(self, prompts: List[List[int]]) -> List[List[int]]:
        """Generate completions for a batch of tokenised prompts.

        For N prompts, returns N * completions_per_prompt completions.

        Dispatches to the tt-transformers mirror when :meth:`init_model` has
        populated it; otherwise runs the original ttml batched path. Both
        paths return tokens-only completions in the same shape.
        """
        if self._tt_model is not None:
            return self._completion_tt_impl(prompts)

        ctx = self._ctx
        max_len = max(len(row) for row in prompts)
        pad_lengths = [max_len - len(row) for row in prompts for _ in range(ctx.completions_per_prompt)]
        B = ctx.completions_per_prompt * len(prompts)
        N = max_len

        prompt_tokens_np = np.full((B, max_len), ctx._pad_token)
        for i, row in enumerate(prompts):
            start = i * ctx.completions_per_prompt
            end = start + ctx.completions_per_prompt
            prompt_tokens_np[start:end, max_len - len(row) :] = np.asarray(row)

        self._model.eval()
        with no_grad():
            return self._completion_batched_impl(prompt_tokens_np, pad_lengths, B, N)

    def generate_str(self, prompt_strs: List[str]) -> List[str]:
        """Generate completions from string prompts, returning decoded strings.

        Backend-agnostic — :meth:`generate` does the dispatch.
        """
        prompts = [self._ctx._tokenizer.encode(s) for s in prompt_strs]
        completions = self.generate(prompts)
        return [self._ctx._tokenizer.decode(c, skip_special_tokens=False) for c in completions]

    def _reset_tt_kv_cache(self) -> None:
        """Zero the tt-transformers paged-attention KV cache before a new prompt.

        Mirrors the public-attribute pattern used in
        ``models/tt_transformers/demo/simple_text_demo.py``: each layer's
        ``attention.layer_past`` is a ``(k_cache, v_cache)`` tuple of device
        tensors; we overwrite their contents in place so subsequent attention
        kernels read zeros at unfilled positions.
        """
        for layer in self._tt_model.layers:
            k_cache, v_cache = layer.attention.layer_past
            ttnn.mul(k_cache, 0, output_tensor=k_cache)
            ttnn.mul(v_cache, 0, output_tensor=v_cache)

    def _completion_tt_impl(self, prompts: List[List[int]]) -> List[List[int]]:
        """tt-transformers generation loop using the public Generator API.

        Mirrors the structure of :meth:`_completion_batched_impl` (the ttml
        loop) so the on-device flow is the same in spirit:

          * Prefill the prompt in one shot via ``Generator.prefill_forward_text``.
            With ``sampling_params`` set, on-device sampling returns the first
            generated token directly (already on host).
          * Decode loop calls ``Generator.decode_forward(read_from_device=False)``
            so each step's sampled token stays on device and is fed back into
            the next step's trace input slot via the on-device feedback chain
            (``ttnn_decode_forward`` writes the sampled token into the
            trace-input ``x`` and increments ``current_pos`` on device — see
            ``models/tt_transformers/tt/model.py:638-642`` and
            ``model.py:603-605``).
          * Stop-token detection runs on chunked async d2h reads (every
            ``CHUNK`` steps) instead of per-step host syncs, matching ttml's
            ``_async_read_to_host`` pattern. Detection thus lags compute by up
            to ``CHUNK`` tokens, which we trim out at the end.

        Single-prompt only for now (``init_model`` builds with
        ``max_batch_size=1`` and ``completions_per_prompt`` must be 1).
        Multiple prompts are processed sequentially with the KV cache reset
        between each.
        """
        import torch

        from models.common.sampling import SamplingParams

        ctx = self._ctx
        args = self._tt_model_args
        generator = self._tt_generator
        page_table = self._tt_page_table
        kv_cache = [self._tt_kv_cache]
        mesh_device = self._mesh_device

        if args.max_batch_size != 1:
            raise NotImplementedError(
                "tt-transformers path currently supports max_batch_size=1; " f"got max_batch_size={args.max_batch_size}"
            )
        if ctx.completions_per_prompt != 1:
            raise NotImplementedError("tt-transformers path currently supports completions_per_prompt=1")

        # Sampling: temperature=0 means greedy (top_k=1 with neutral
        # temperature). Otherwise multinomial at the requested temperature.
        if ctx.temperature == 0.0:
            sampling_params = SamplingParams(temperature=1.0, top_k=1, top_p=1.0)
        else:
            sampling_params = SamplingParams(temperature=ctx.temperature, top_k=0, top_p=1.0)

        stop_ids = self._get_stop_ids()
        stop_arr = np.fromiter(stop_ids, dtype=np.int32) if stop_ids else np.empty(0, dtype=np.int32)

        completions: List[List[int]] = []
        for prompt_ids in prompts:
            prompt_len = len(prompt_ids)
            if prompt_len == 0:
                completions.append([])
                continue

            # Reset paged KV cache between prompts so context never leaks.
            self._reset_tt_kv_cache()

            # ---- Prefill ----
            prompt_t = torch.tensor([list(prompt_ids)], dtype=torch.long)
            prefilled, _ = generator.prefill_forward_text(
                prompt_t,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=[prompt_len],
                sampling_params=sampling_params,
                enable_trace=True,
                warmup_prefill=True,
            )
            first_tok = int(prefilled[0, 0].item())

            completion: List[int] = [first_tok]
            if first_tok in stop_ids:
                completions.append(completion)
                continue

            max_new = min(
                ctx.max_tokens_to_complete - 1,
                args.max_seq_len - prompt_len - 1,
            )
            if max_new <= 0:
                completions.append(completion)
                continue

            # ---- Decode loop with chunked async readback ----
            #
            # ``out_tok`` and ``current_pos`` are only used by the FIRST
            # decode_forward call (when ``reset_batch=True`` populates trace
            # inputs from host). After that the trace updates current_pos and
            # the token slot on device, and these args are ignored — see
            # ``Generator._decode_forward_trace_text`` reset_inputs gating.
            out_tok = prefilled
            current_pos = torch.tensor([prompt_len])

            chunk_columns: List[Any] = []
            pending_hosts: List[Any] = []
            pending_event: Any = None
            early_stop = False

            for step in range(max_new):
                tt_dec = generator.decode_forward(
                    out_tok,
                    current_pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    enable_trace=True,
                    sampling_params=sampling_params,
                    reset_batch=(step == 0),
                    read_from_device=False,
                )
                # ``tt_dec`` is one (token_dev, log_probs_dev) per data-parallel
                # rank; data_parallel == 1 for our completer.
                device_tok = tt_dec[0][0]
                chunk_columns.append(device_tok)
                # Host shadow for current_pos. The trace updates the device-side
                # current_pos itself, but we keep this in sync so any future
                # decode_forward call that hits ``reset_inputs=True`` (e.g. on a
                # later prompt boundary) writes the right value.
                current_pos = current_pos + 1

                if (step + 1) % CHUNK == 0:
                    # Sync the previous chunk and promote its tokens to the
                    # completion. Detection lags by CHUNK steps; we break as
                    # soon as a stop token is found and trim correctly.
                    if pending_event is not None:
                        ttnn.event_synchronize(mesh_event=pending_event)
                        for h in pending_hosts:
                            tok_torch, _ = generator.process_decode_output_host([(h, None)], is_tokens=True)
                            tok = int(tok_torch[0].item())
                            completion.append(tok)
                            if tok in stop_ids:
                                early_stop = True
                                break
                        if early_stop:
                            break
                    # Fire async d2h for this chunk; the event marks all
                    # in-flight reads on cq=0.
                    pending_hosts = [t.cpu(blocking=False) for t in chunk_columns]
                    pending_event = ttnn.record_event(mesh_device=mesh_device, cq_id=0)
                    chunk_columns = []

            # Drain any in-flight chunk we fired but never synced inside the loop.
            if not early_stop and pending_event is not None:
                ttnn.event_synchronize(mesh_event=pending_event)
                for h in pending_hosts:
                    tok_torch, _ = generator.process_decode_output_host([(h, None)], is_tokens=True)
                    tok = int(tok_torch[0].item())
                    completion.append(tok)
                    if tok in stop_ids:
                        early_stop = True
                        break

            # Drain any remaining device tokens that never hit a chunk boundary.
            if not early_stop and chunk_columns:
                tail_hosts = [t.cpu(blocking=False) for t in chunk_columns]
                tail_event = ttnn.record_event(mesh_device=mesh_device, cq_id=0)
                ttnn.event_synchronize(mesh_event=tail_event)
                for h in tail_hosts:
                    tok_torch, _ = generator.process_decode_output_host([(h, None)], is_tokens=True)
                    tok = int(tok_torch[0].item())
                    completion.append(tok)
                    if tok in stop_ids:
                        break

            completions.append(completion)

        return completions

    def compute_nlog_probs(
        self, prompts: List[List[int]], completions: List[List[int]]
    ) -> tuple[ttml.autograd.Tensor, ttml.autograd.Tensor]:
        assert len(completions) == len(prompts)

        B = len(completions)
        pad_token = self._ctx._pad_token

        total_devices = self._num_devices
        assert B % total_devices == 0
        B_local = B // total_devices

        lengths = [len(p) + len(c) for p, c in zip(prompts, completions)]
        T = max(lengths) - 1
        assert T >= 1

        inputs_np = np.full((B, T), pad_token, dtype=np.uint32)
        targets_np = np.full((B, T), pad_token, dtype=np.uint32)
        loss_mask_np = np.zeros((B, T), dtype=np.float32)
        pad_lengths: List[int] = []

        for i, (p, c) in enumerate(zip(prompts, completions)):
            sequence = p + c
            L = len(sequence) - 1
            shift = T - L
            pad_lengths.append(shift)

            if len(p) < 2:
                raise ValueError("Prompt is too short")
            if len(sequence) < 2:
                raise ValueError("Sequence is too short")

            inputs_np[i, -L:] = np.asarray(sequence[:-1], dtype=np.uint32)
            targets_np[i, -L:] = np.asarray(sequence[1:], dtype=np.uint32)

            if c:
                start = -1 + shift + len(p)
                end = min(start + len(c), T)
                if start < end:
                    loss_mask_np[i, start:end] = 1.0

        logits = self._forward(inputs_np, pad_lengths, B)

        Tp = _round_up(T)
        targets_pad = np.full((B, Tp), pad_token, dtype=np.uint32)
        targets_pad[:, :T] = targets_np

        targets_tt = ttml.autograd.Tensor.from_numpy(
            targets_pad, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, self._dp_mapper
        )

        nlog = ttml.ops.loss.cross_entropy_loss(logits, targets_tt, ttml.ops.ReduceType.NONE)
        nlog = ttml.ops.reshape.reshape(nlog, [B_local, Tp])

        loss_mask_pad = np.zeros((B, Tp), dtype=np.float32)
        loss_mask_pad[:, :T] = loss_mask_np

        loss_mask_tt = ttml.autograd.Tensor.from_numpy(
            loss_mask_pad, ttnn.Layout.ROW_MAJOR, ttnn.DataType.BFLOAT16, self._dp_mapper
        )

        return nlog, loss_mask_tt

    def _forward(self, input_ids_np: np.ndarray, pad_lengths: List[int], B: int) -> ttml.autograd.Tensor:
        """Run a full forward pass (no KV cache) and return logits."""
        T = input_ids_np.shape[1]
        x_tt = self._tokens_to_tensor(input_ids_np, B)
        mask_tt = self._create_causal_mask(prompt_len=0, query_len=T, pad_lengths=pad_lengths, B=B)
        return self._model(x_tt, mask_tt)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tokens_to_tensor(self, tokens_np: np.ndarray, B: int) -> ttml.autograd.Tensor:
        padded_len = _round_up(tokens_np.shape[1])
        padded = np.full((B, padded_len), self._ctx._pad_token, dtype=np.uint32)
        padded[:, : tokens_np.shape[1]] = tokens_np
        return ttml.autograd.Tensor.from_numpy(
            padded.reshape(B, 1, 1, padded_len), ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, self._dp_mapper
        )

    def _create_causal_mask(
        self, prompt_len: int, query_len: int, pad_lengths: List[int], B: int
    ) -> ttml.autograd.Tensor:
        assert len(pad_lengths) == B

        whole_len = prompt_len + query_len
        padded_q = _round_up(query_len)
        padded_w = _round_up(whole_len)

        mask_one_token = np.zeros((padded_q, padded_w), dtype=np.float32)
        mask_one_token[:query_len, :padded_w] = np.tri(query_len, padded_w, k=prompt_len, dtype=np.float32)

        mask_3d = np.tile(mask_one_token, (B, 1, 1))
        for i in range(B):
            mask_3d[i, :, 0 : pad_lengths[i]] = 0

        mask_4d = mask_3d[:, np.newaxis, :, :]
        assert mask_4d.shape == (B, 1, padded_q, padded_w)

        return ttml.autograd.Tensor.from_numpy(mask_4d, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, self._dp_mapper)

    def _build_logits_mask(self, vocab_size: int, padded_vocab_size: int) -> ttml.autograd.Tensor:
        logits_mask = np.zeros((1, 1, 1, padded_vocab_size), dtype=np.float32)
        logits_mask[:, :, :, vocab_size:] = 1e4
        return ttml.autograd.Tensor.from_numpy(logits_mask, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16)

    def _get_stop_ids(self) -> set[int]:
        tokenizer = self._ctx._tokenizer
        stop_ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            stop_ids.add(int(tokenizer.eos_token_id))
        if tokenizer.pad_token_id is not None:
            stop_ids.add(int(tokenizer.pad_token_id))
        for tok in ["<|eot_id|>", "<|end_of_text|>", "<|eom_id|>"]:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if tid is not None and tid >= 0 and tid != tokenizer.unk_token_id:
                stop_ids.add(int(tid))
        return stop_ids

    def _get_kv_cache(self, B: int) -> ttml.models.KvCache:
        cfg = self.transformer_config
        head_dim = getattr(cfg, "head_dim", None) or (cfg.embedding_dim // cfg.num_heads)
        if self._kv_cache is None or self._kv_cache_B != B:
            self._kv_cache = ttml.models.KvCache(
                cfg.num_blocks,
                B,
                cfg.num_groups,
                cfg.max_sequence_length,
                head_dim,
            )
            self._kv_cache_B = B
        self._kv_cache.reset()
        return self._kv_cache

    def _completion_batched_impl(
        self,
        prompt_tokens_np: np.ndarray,
        pad_lengths: List[int],
        B: int,
        N: int,
    ) -> List[List[int]]:
        ctx = self._ctx
        assert prompt_tokens_np.shape == (B, N)
        assert len(pad_lengths) == B
        total_devices = self._num_devices
        assert B % total_devices == 0

        B_local = B // total_devices

        V = len(ctx._tokenizer)
        padded_V = _round_up(V)

        kv_cache = self._get_kv_cache(B_local)
        logits_mask_tensor = self._build_logits_mask(V, padded_V) if padded_V != V else None

        tokens_to_complete = min(
            ctx.max_tokens_to_complete,
            self.transformer_config.max_sequence_length - N,
        )

        mesh_device = self._mesh_device
        composer = self._dp_composer
        stop_ids = self._get_stop_ids()
        stop_arr = np.fromiter(stop_ids, dtype=np.int32) if stop_ids else np.empty(0, dtype=np.int32)

        generated_columns: List[Any] = []
        chunk_columns: List[Any] = []
        pending_hosts: List[Any] = []
        pending_event: Any = None
        done = np.zeros(B, dtype=bool)

        def to_np(column_list: List[Any]) -> np.ndarray:
            arr = np.empty((B, len(column_list)), dtype=np.int32)
            for j, column in enumerate(column_list):
                arr[:, j] = column.to_numpy(composer).reshape(B)
            return arr

        for i in range(tokens_to_complete):
            if kv_cache.get_cache_position() == 0:
                processed = 0
                new_tokens = prompt_tokens_np.shape[1]
                token_tensor = self._tokens_to_tensor(prompt_tokens_np, B)
            else:
                processed = N - 1
                new_tokens = 1
                token_tensor = ttnn.pad(
                    last_token_column,
                    [(0, 0), (0, 0), (0, 0), (0, TILE_SIZE - 1)],
                    ctx._pad_token,
                )
                token_tensor = ttml.autograd.Tensor(token_tensor, False)

            mask = self._create_causal_mask(processed, new_tokens, pad_lengths, B)
            logits = self._model(token_tensor, mask, kv_cache=kv_cache, new_tokens=new_tokens)

            next_token_tensor = ttml.ops.sample.sample_op(
                logits, ctx.temperature, np.random.randint(low=1e7), logits_mask_tensor
            )

            last_token_column = ttnn.slice(
                next_token_tensor.get_value(),
                [0, 0, new_tokens - 1, 0],
                [B_local, 1, new_tokens, 1],
            )

            generated_columns.append(last_token_column)
            chunk_columns.append(last_token_column)
            N += 1

            deallocate_tensors([token_tensor, mask, logits, next_token_tensor])

            if (i + 1) % CHUNK == 0:
                if pending_event is not None:
                    ttnn.event_synchronize(mesh_event=pending_event)
                    chunk_np = np.stack(
                        [h.to_numpy(mesh_composer=composer).reshape(B) for h in pending_hosts],
                        axis=1,
                    )
                    done |= np.isin(chunk_np, stop_arr).any(axis=1)
                    if done.all():
                        break

                pending_hosts, pending_event = _async_read_to_host(chunk_columns, mesh_device)
                chunk_columns = []

        completions_np = to_np(generated_columns)
        deallocate_tensors(generated_columns)
        deallocate_tensors([logits_mask_tensor])
        kv_cache.reset()

        completions: List[List[int]] = []
        for i in range(B):
            to = completions_np.shape[1]
            for j, token in enumerate(completions_np[i]):
                if token in stop_ids:
                    to = j
                    break
            completions.append(completions_np[i, :to].tolist())

        return completions
