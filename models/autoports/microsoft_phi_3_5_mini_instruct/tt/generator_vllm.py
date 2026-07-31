# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for the Phi-3.5-mini TTNN autoport."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import ttnn

from models.autoports.microsoft_phi_3_5_mini_instruct.tt.generator import Phi35MiniGenerator
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.model import DEFAULT_REVISION, MODEL_ID
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.precision import dtype_name


class Phi3ForCausalLM:
    """Translate vLLM's TT runner API into the full-model Phi generator API."""

    model_capabilities = {
        "supports_async_decode": True,
        "tt_async_decode_allows_overlap": False,
        "supports_prefix_caching": False,
        "supports_sample_on_device": True,
        "supports_topk_logprobs": True,
    }

    def __init__(self, generator: Phi35MiniGenerator, *, max_batch_size: int, tt_data_parallel: int) -> None:
        self.generator = generator
        self.model = generator.model
        self.mesh_device = generator.mesh_device
        self.tokenizer = generator.tokenizer
        self.max_batch_size = int(max_batch_size)
        self.tt_data_parallel = int(tt_data_parallel)
        self.vllm_kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]] | None = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size: int,
        max_seq_len: int,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
    ) -> "Phi3ForCausalLM":
        if optimizations is not None:
            raise ValueError("Phi-3.5 vLLM adapter uses the datatype-sweep selected policy; optimizations is unused")
        if tt_data_parallel != 1:
            raise ValueError(f"Phi-3.5 vLLM adapter currently supports tt_data_parallel=1, got {tt_data_parallel}")
        if max_batch_size != 1:
            raise ValueError(f"Phi-3.5 optimized full model currently supports max_batch_size=1, got {max_batch_size}")

        model_dir = Path(__file__).resolve().parents[1]
        hf_model_id = getattr(hf_config, "_name_or_path", None) or MODEL_ID
        generator = Phi35MiniGenerator(
            model_dir=model_dir,
            mesh_device=mesh_device,
            hf_model_id=hf_model_id,
            revision=DEFAULT_REVISION,
            max_seq_len=max_seq_len,
            allocate_standalone_cache=False,
        )
        return cls(generator, max_batch_size=max_batch_size, tt_data_parallel=tt_data_parallel)

    @classmethod
    def get_max_tokens_all_users(
        cls,
        *,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **_: Any,
    ) -> int:
        if max_model_len is None or max_num_seqs is None:
            raise ValueError("Phi-3.5 vLLM cache sizing requires max_model_len and max_num_seqs")
        return int(max_model_len) * int(max_num_seqs)

    @property
    def cache_path(self) -> Path:
        return self.generator.model_dir

    def allocate_kv_cache(
        self,
        kv_cache_shape: tuple[int, int, int, int],
        dtype: torch.dtype,
        num_layers: int,
    ) -> list[tuple[ttnn.Tensor, ttnn.Tensor]]:
        if num_layers != self.model.full_config.num_layers:
            raise ValueError(f"vLLM requested {num_layers} KV layers, model has {self.model.full_config.num_layers}")

        shape = tuple(int(dim) for dim in kv_cache_shape)
        if shape[2] != self.model.full_config.block_size:
            raise ValueError(
                f"Phi-3.5 serving requires vLLM --block-size {self.model.full_config.block_size}, "
                f"got KV cache shape {shape}"
            )

        cache_dtype = self.model.precision_policy.kv_cache_dtype
        zero_cache = torch.zeros(shape, dtype=torch.bfloat16)
        kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]] = []
        for layer_idx in range(num_layers):
            layer_cache = []
            for name in ("k", "v"):
                layer_cache.append(
                    ttnn.as_tensor(
                        zero_cache,
                        device=self.mesh_device,
                        dtype=cache_dtype,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                        cache_file_name=self.cache_path / f"empty_{name}_cache_layer{layer_idx}_{shape}_{dtype_name(cache_dtype)}",
                    )
                )
            kv_cache.append((layer_cache[0], layer_cache[1]))
        self.vllm_kv_cache = kv_cache
        return kv_cache

    def prefill_forward(
        self,
        *,
        tokens: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]],
        prompt_lens: list[int] | torch.Tensor,
        sampling_params: Any | None = None,
        empty_slots: list[int] | None = None,
        enable_trace: bool = False,
        **kwargs: Any,
    ) -> Any:
        if sampling_params is None:
            raise ValueError("Phi-3.5 vLLM serving requires on-device sampling; host sampling is disabled")
        return self.generator.prefill_forward_token_out(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            sampling_params=sampling_params,
            empty_slots=empty_slots,
            enable_trace=False,
            **kwargs,
        )

    def decode_forward(
        self,
        *,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table: torch.Tensor,
        kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]],
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        if sampling_params is None:
            raise ValueError("Phi-3.5 vLLM serving requires on-device sampling; host sampling is disabled")
        return self.generator.decode_forward_token_out(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
            enable_trace=enable_trace,
            read_from_device=read_from_device,
            **kwargs,
        )

    def read_decode_output(self, tt_out: Any, async_read: bool = True) -> Any:
        return self.generator.read_decode_output(tt_out, async_read=async_read)

    def process_decode_output_host(self, tt_out: Any, is_tokens: bool = True) -> Any:
        return self.generator.process_decode_output_host(tt_out, is_tokens=is_tokens)

    def warmup_model_prefill(self, *args: Any, **kwargs: Any) -> None:
        self.generator.warmup_model_prefill(*args, **kwargs)

    def warmup_model_decode(self, *args: Any, **kwargs: Any) -> None:
        self.generator.warmup_model_decode(*args, **kwargs)

    def trace_counters(self) -> dict[str, int]:
        return self.generator.trace_counters()

    def __del__(self) -> None:
        if hasattr(self, "generator"):
            self.generator.teardown()
