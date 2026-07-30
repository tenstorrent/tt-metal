# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for the repo-local ``google/gemma-4-12B`` autoport."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import torch
import ttnn

from models.common.sampling import format_sampling_params
from models.common.warmup.warmup_utils import WarmupForwardMixin


def _load_sibling_module(name: str, filename: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(f"gemma4_12b_vllm_{name}", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    spec.loader.exec_module(module)
    return module


_generator_mod = _load_sibling_module("generator", "generator.py")
_model_mod = _load_sibling_module("model", "model.py")

SUPPORTED_HF_MODEL_ID = _model_mod.SUPPORTED_HF_MODEL_ID
DEFAULT_BLOCK_SIZE = _model_mod.DEFAULT_BLOCK_SIZE
DEFAULT_MAX_SEQ_LEN = _model_mod.DEFAULT_MAX_SEQ_LEN


def _as_text_config(hf_config):
    return getattr(hf_config, "text_config", hf_config)


def _first_device_torch(tensor) -> torch.Tensor:
    if isinstance(tensor, torch.Tensor):
        return tensor
    mesh_device = None
    if hasattr(tensor, "device"):
        try:
            mesh_device = tensor.device()
        except Exception:
            mesh_device = None
    return _ttnn_to_torch(tensor, mesh_device)


def _ttnn_to_torch(tensor, mesh_device=None) -> torch.Tensor:
    """Convert either a device TTNN tensor or an async-read host TTNN tensor."""
    if isinstance(tensor, torch.Tensor):
        return tensor
    if mesh_device is not None and hasattr(mesh_device, "shape"):
        try:
            shards = ttnn.get_device_tensors(tensor)
            if shards:
                return ttnn.to_torch(shards[0])
        except Exception:
            pass
    return ttnn.to_torch(tensor)


def _torch_dtype_to_ttnn(dtype):
    if dtype == torch.bfloat16:
        return ttnn.bfloat16
    if dtype == torch.float32:
        return ttnn.float32
    if dtype == torch.int32:
        return ttnn.int32
    return ttnn.bfloat16


class Gemma4ForCausalLM(WarmupForwardMixin):
    """Thin vLLM bridge around :class:`Gemma412BGenerator`.

    vLLM owns the serving KV cache. This class allocates that cache only from
    vLLM's cache specs and passes it through to the generator's low-level
    prefill/decode methods; the wrapped generator is constructed without its
    standalone readiness cache.
    """

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": True,
        "supports_sample_on_device": True,
    }

    supports_topk_logprobs = False

    def __init__(self, generator=None, *, vllm_config=None, prefix: str = ""):
        del vllm_config, prefix
        if generator is None:
            raise RuntimeError("Gemma4ForCausalLM must be constructed through initialize_vllm_model")
        self.generator = generator
        self.model = generator.model
        self.tokenizer = generator.tokenizer
        self.vocab_size = self.model.vocab_size
        self._last_decode_batch = 1
        self._last_prefill_offsets: list[int] = [0]
        self.already_warmed_up_prefill = False

    # vLLM protocol shims. Execution on TT uses prefill_forward/decode_forward.
    def embed_input_ids(self, input_ids):  # pragma: no cover - protocol shim
        raise NotImplementedError("Gemma4ForCausalLM executes embeddings inside the TT prefill/decode path.")

    def forward(self, input_ids, positions, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("Gemma4ForCausalLM is driven by the TT vLLM runner, not by forward().")

    def compute_logits(self, hidden_states, **kwargs):  # pragma: no cover - protocol shim
        raise NotImplementedError("Gemma4ForCausalLM produces logits inside prefill_forward/decode_forward.")

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> int:
        del model_name, num_devices, kwargs
        if tt_data_parallel != 1:
            raise ValueError(f"{SUPPORTED_HF_MODEL_ID} vLLM autoport supports tt_data_parallel=1, got {tt_data_parallel}")
        return int(max_model_len or DEFAULT_MAX_SEQ_LEN) * int(max_num_seqs or 1)

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        text_config = _as_text_config(model_config.hf_config)
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types is None:
            raise ValueError(f"{cls.__name__}.get_kv_cache_spec requires hf_config.text_config.layer_types")

        dtype = (
            model_config.dtype
            if cache_config.cache_dtype == "auto"
            else STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        )
        block_size = cache_config.block_size
        sliding_heads = int(text_config.num_key_value_heads)
        sliding_head_dim = int(text_config.head_dim)
        full_heads = int(getattr(text_config, "num_global_key_value_heads", None) or sliding_heads)
        full_head_dim = int(getattr(text_config, "global_head_dim", None) or sliding_head_dim)

        spec_per_layer = {}
        for layer_idx, layer_type in enumerate(layer_types):
            if layer_type == "sliding_attention":
                num_kv_heads = sliding_heads
                head_size = sliding_head_dim
            elif layer_type == "full_attention":
                num_kv_heads = full_heads
                head_size = full_head_dim
            else:
                raise ValueError(f"unsupported Gemma4 layer_types[{layer_idx}]={layer_type!r}")
            spec_per_layer[f"model.layers.{layer_idx}.self_attn"] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=dtype,
            )
        return spec_per_layer

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=DEFAULT_MAX_SEQ_LEN,
        n_layers=None,
        tt_data_parallel=1,
        optimizations: str | None = None,
    ):
        if optimizations not in (None, "performance"):
            raise ValueError(f"{SUPPORTED_HF_MODEL_ID} vLLM adapter only supports the optimized performance path")
        if tt_data_parallel != 1:
            raise ValueError(f"{SUPPORTED_HF_MODEL_ID} vLLM adapter supports tt_data_parallel=1, got {tt_data_parallel}")
        if int(max_batch_size) != 1:
            raise ValueError(
                f"{SUPPORTED_HF_MODEL_ID} autoport vLLM bridge is batch-1 today; "
                f"launch with --max-num-seqs 1 (got max_batch_size={max_batch_size})"
            )

        model_dir = Path(__file__).resolve().parents[1]
        hf_model_id = getattr(hf_config, "_name_or_path", None) or SUPPORTED_HF_MODEL_ID
        generator = _generator_mod.build_generator(
            model_dir=model_dir,
            mesh_device=mesh_device,
            hf_model_id=hf_model_id,
            max_seq_len=int(max_seq_len or DEFAULT_MAX_SEQ_LEN),
            max_batch_size=int(max_batch_size),
            block_size=DEFAULT_BLOCK_SIZE,
            max_num_blocks=math.ceil(int(max_seq_len or DEFAULT_MAX_SEQ_LEN) / DEFAULT_BLOCK_SIZE),
            num_layers=n_layers,
            use_on_device_sampling=True,
            allocate_standalone_cache=False,
        )
        return cls(generator)

    @property
    def cache_path(self) -> Path:
        path = self.model.tensor_cache_path / "vllm_kv_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def allocate_kv_cache_per_layer(self, per_layer_specs):
        if len(per_layer_specs) != self.model.num_layers:
            raise ValueError(f"expected {self.model.num_layers} KV specs, got {len(per_layer_specs)}")

        kv_cache = []
        for layer_idx, (shape, dtype, _tensor_idx) in enumerate(per_layer_specs):
            max_num_blocks, num_kv_heads, block_size, head_size = (int(x) for x in shape)
            layer = self.model.layers[layer_idx]
            expected_heads = int(layer.self_attn.local_kv_heads)
            expected_head_size = int(layer.attention_config.head_dim)
            if num_kv_heads != expected_heads or head_size != expected_head_size:
                raise ValueError(
                    f"vLLM KV spec mismatch for layer {layer_idx}: got heads={num_kv_heads}, "
                    f"head_size={head_size}; expected heads={expected_heads}, head_size={expected_head_size}"
                )
            kv_cache.append(
                layer.create_paged_kv_cache(
                    block_size=block_size,
                    max_num_blocks=max_num_blocks,
                    cache_dtype=_torch_dtype_to_ttnn(dtype),
                    tensor_cache_path=self.cache_path / f"layer_{layer_idx}",
                )
            )
        return kv_cache

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self.allocate_kv_cache_per_layer([(kv_cache_shape, dtype, i) for i in range(num_layers)])

    def _page_table_arg(self, page_table, page_tables_per_layer):
        return page_tables_per_layer if page_tables_per_layer is not None else page_table

    def _warmup_page_table(self, seq_len: int | None = None) -> torch.Tensor:
        if seq_len is None:
            num_blocks = math.ceil(int(self.generator.max_seq_len) / int(self.generator.block_size))
        else:
            num_blocks = max(1, math.ceil(int(seq_len) / int(self.generator.block_size)))
        return torch.zeros(1, num_blocks, dtype=torch.int32)

    def warmup_model_prefill(
        self,
        kv_cache,
        enable_trace,
        can_sample_on_device,
        greedy_only: bool = False,
        **kwargs: Any,
    ) -> None:
        del enable_trace, kwargs
        if self.already_warmed_up_prefill:
            return
        self.already_warmed_up_prefill = True

        seq_len = min(128, int(self.generator.max_seq_len))
        seq_len = max(ttnn.TILE_SIZE, math.ceil(seq_len / ttnn.TILE_SIZE) * ttnn.TILE_SIZE)
        tokens = torch.zeros(1, seq_len, dtype=torch.long)
        page_table = self._warmup_page_table(seq_len)
        prompt_lens = [seq_len]
        sampling_params = self._create_sampling_params(
            bool(can_sample_on_device),
            batch_size=1,
            greedy_only=greedy_only,
        )
        for param in sampling_params:
            self.prefill_forward(
                tokens=tokens,
                page_table=page_table,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                sampling_params=param,
                empty_slots=[0],
            )

    @staticmethod
    def _check_no_prefix_cache(start_pos) -> None:
        if start_pos is None:
            return
        values = torch.as_tensor(start_pos).reshape(-1)
        if values.numel() and bool((values != 0).any().item()):
            raise ValueError("prefix caching is not supported by the Gemma4 12B autoport vLLM adapter")

    def _sampling_module(self):
        sampling = getattr(self.model, "sampling", None)
        if sampling is None:
            raise RuntimeError("on-device sampling requested but the model sampling module is not initialized")
        return sampling

    def _sample_prefill_logits(self, logits, sampling_params, tokens, prompt_lens, empty_slots):
        sampling = self._sampling_module()
        formatted = format_sampling_params(sampling_params, sampling.tt_sampling.max_batch_size)
        if empty_slots is None:
            empty_slots = list(range(tokens.shape[0]))
        max_prompt_len = int(max(prompt_lens)) if len(prompt_lens) else tokens.shape[1]
        prompt_tokens = torch.full((tokens.shape[0], max_prompt_len), -1, dtype=torch.long)
        for row, prompt_len in enumerate(prompt_lens):
            prompt_tokens[row, : int(prompt_len)] = tokens[row, : int(prompt_len)].to(torch.long)
        sampling.apply_prefill_state(
            sampling_params=formatted,
            prompt_tokens=prompt_tokens,
            empty_slots=[int(slot) for slot in empty_slots],
        )
        sampled_tokens, log_probs = sampling.sample(logits, enable_trace=False)
        ttnn.synchronize_device(self.model.mesh_device)

        token_host = _model_mod._first_device_torch(sampled_tokens, self.model.mesh_device).reshape(-1)
        offsets = [int(prompt_len - 1) % ttnn.TILE_SIZE for prompt_len in prompt_lens]
        self._last_prefill_offsets = offsets
        out_tokens = torch.tensor([int(token_host[offset].item()) for offset in offsets], dtype=torch.long)
        if log_probs is None:
            return out_tokens
        log_probs_host = _model_mod._first_device_torch(log_probs, self.model.mesh_device).reshape(-1)
        out_log_probs = torch.tensor([float(log_probs_host[offset].item()) for offset in offsets], dtype=torch.float32)
        return out_tokens, out_log_probs

    def prefill_forward(
        self,
        tokens,
        page_table=None,
        kv_cache=None,
        prompt_lens=None,
        sampling_params=None,
        empty_slots=None,
        start_pos=None,
        page_tables_per_layer=None,
        return_all_logits=False,
        **kwargs: Any,
    ):
        self._check_no_prefix_cache(start_pos)
        if kv_cache is None:
            raise RuntimeError("vLLM must pass its serving KV cache into Gemma4ForCausalLM.prefill_forward")
        prompt_lens = [int(x) for x in (prompt_lens if prompt_lens is not None else [tokens.shape[1]] * tokens.shape[0])]
        if len(prompt_lens) != int(tokens.shape[0]):
            raise ValueError(f"prompt_lens length {len(prompt_lens)} does not match batch {tokens.shape[0]}")

        page_arg = self._page_table_arg(page_table, page_tables_per_layer)
        if sampling_params is None:
            return self.generator.prefill_forward(
                tokens,
                page_table=page_arg,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens,
                return_all_logits=return_all_logits,
            )

        logits = self.generator.prefill_forward(
            tokens,
            page_table=page_arg,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            return_all_logits=False,
            return_ttnn=True,
            gather_logits=False,
        )
        return self._sample_prefill_logits(logits, sampling_params, tokens, prompt_lens, empty_slots)

    def _apply_decode_sampling_state(
        self,
        sampling_params,
        *,
        reset_batch=False,
        prompt_tokens=None,
        output_tokens=None,
        slot_remap=None,
    ) -> None:
        if sampling_params is None:
            return
        sampling = self._sampling_module()
        sampling.apply_decode_state(
            [sampling_params],
            reset_batch=reset_batch,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        if slot_remap is not None:
            remap = torch.as_tensor(slot_remap, dtype=torch.int32).reshape(-1)
            sampling.seed_manager.apply_slot_remap(remap[: sampling.seed_manager.max_batch_size])
        sampling.seed_manager.get_new_values()

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table=None,
        kv_cache=None,
        enable_trace=True,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
        prompt_tokens=None,
        output_tokens=None,
        slot_remap=None,
        page_tables_per_layer=None,
        **kwargs: Any,
    ):
        if kv_cache is None:
            raise RuntimeError("vLLM must pass its serving KV cache into Gemma4ForCausalLM.decode_forward")
        page_arg = self._page_table_arg(page_table, page_tables_per_layer)
        self._last_decode_batch = int(tokens.shape[0])
        self._apply_decode_sampling_state(
            sampling_params,
            reset_batch=reset_batch,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            slot_remap=slot_remap,
        )
        out = self.generator.decode_forward(
            tokens,
            start_pos,
            page_table=page_arg,
            kv_cache=kv_cache,
            sample_on_device=sampling_params is not None,
            enable_trace=enable_trace,
            return_ttnn=not read_from_device,
            async_decode=not read_from_device,
        )
        if not read_from_device:
            return out
        if sampling_params is not None:
            return out.reshape(-1)[: self._last_decode_batch].to(torch.long)
        return out.unsqueeze(1) if isinstance(out, torch.Tensor) and out.dim() == 2 else out

    def _device_output_to_host(self, tt_out, *, is_tokens: bool):
        if isinstance(tt_out, torch.Tensor):
            return tt_out
        if isinstance(tt_out, tuple):
            tokens = self._device_output_to_host(tt_out[0], is_tokens=True)
            log_probs = None if tt_out[1] is None else self._device_log_probs_to_host(tt_out[1])
            return tokens, log_probs
        host = _ttnn_to_torch(tt_out, self.model.mesh_device)
        if is_tokens:
            return host.reshape(-1)[: self._last_decode_batch].to(torch.long)
        return host.float()[:, :, : self._last_decode_batch, : self.vocab_size].reshape(
            self._last_decode_batch, 1, self.vocab_size
        )

    def _device_log_probs_to_host(self, tt_out):
        if isinstance(tt_out, torch.Tensor):
            return tt_out.reshape(-1)[: self._last_decode_batch].to(torch.float32)
        host = _ttnn_to_torch(tt_out, self.model.mesh_device)
        return host.reshape(-1)[: self._last_decode_batch].to(torch.float32)

    def _async_read_tensor(self, tt_out):
        if tt_out is None or isinstance(tt_out, torch.Tensor):
            return tt_out
        if isinstance(tt_out, tuple):
            return tuple(self._async_read_tensor(item) for item in tt_out)
        if isinstance(tt_out, ttnn.Tensor):
            return tt_out.cpu(blocking=False)
        cpu = getattr(tt_out, "cpu", None)
        if callable(cpu):
            return cpu(blocking=False)
        return tt_out

    def read_decode_output(self, tt_out, async_read=False):
        if async_read:
            host = self._async_read_tensor(tt_out)
            return host, [ttnn.record_event(self.model.mesh_device, 0)]
        host = self._device_output_to_host(tt_out, is_tokens=isinstance(tt_out, tuple))
        return host

    def process_decode_output_host(self, tt_out, is_tokens=False):
        if isinstance(tt_out, torch.Tensor):
            if is_tokens:
                return tt_out.reshape(-1)[: self._last_decode_batch].to(torch.long)
            return tt_out.unsqueeze(1) if tt_out.dim() == 2 else tt_out
        if isinstance(tt_out, tuple):
            tokens = self.process_decode_output_host(tt_out[0], is_tokens=True)
            log_probs = None if tt_out[1] is None else self._device_log_probs_to_host(tt_out[1])
            return tokens, log_probs
        return self._device_output_to_host(tt_out, is_tokens=is_tokens)

    def teardown(self) -> None:
        self.generator.teardown()


__all__ = ["Gemma4ForCausalLM", "SUPPORTED_HF_MODEL_ID"]
