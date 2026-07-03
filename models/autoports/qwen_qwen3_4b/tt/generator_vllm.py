# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""vLLM adapter for the Qwen/Qwen3-4B autoport."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

import torch

import ttnn
from models.autoports.qwen_qwen3_4b.tt.functional_decoder import HF_MODEL_ID
from models.autoports.qwen_qwen3_4b.tt.generator import Qwen3Generator, build_generator
from models.vllm_test_utils.generative_base import GenerativeTestModelBase

DEFAULT_MODEL_DIR = Path("models/autoports/qwen_qwen3_4b")
MODEL_DIR_ENV = "QWEN3_4B_AUTOPORT_DIR"


class Qwen3ForCausalLM(GenerativeTestModelBase):
    """Thin TT vLLM bridge over the datatype-selected Qwen3-4B generator."""

    model_capabilities = {
        "supports_async_decode": True,
        "supports_async_decode_overlap": True,
        "tt_async_decode_allows_overlap": True,
        "supports_prefix_caching": False,
        "supports_sample_on_device": True,
        "sample_on_device_policy": "greedy_only",
    }
    sample_on_device_policy = "greedy_only"

    def __init__(self, generator: Qwen3Generator, *, requested_max_batch_size: int) -> None:
        self.generator = generator
        self.model = generator.model
        self.mesh_device = generator.mesh_device
        self.max_batch_size = generator.max_batch_size
        self.requested_max_batch_size = int(requested_max_batch_size)
        self.vocab_size = generator.model.vocab_size
        self._vllm_page_table_tt = None
        self._vllm_page_table_host: torch.Tensor | None = None
        self._vllm_page_table_generation = 0
        self._decode_trace_batch_size: int | None = None
        self._num_kv_blocks: int | None = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size: int,
        max_seq_len: int = 40960,
        n_layers: int | None = None,
        tt_data_parallel: int = 1,
        optimizations: str | None = None,
        **_: Any,
    ) -> "Qwen3ForCausalLM":
        if optimizations is not None:
            raise ValueError("Qwen3-4B autoport uses doc/datatype_sweep/selected_precision_config.json")
        if int(tt_data_parallel) != 1:
            raise ValueError(f"Qwen3-4B autoport supports tt_data_parallel=1, got {tt_data_parallel}")
        if int(max_batch_size) > 32:
            raise ValueError(f"Qwen3-4B on-device sampler supports max_batch_size <= 32, got {max_batch_size}")

        model_mesh = _select_tp4_mesh(mesh_device)
        model_dir = Path(os.environ.get(MODEL_DIR_ENV, DEFAULT_MODEL_DIR)).resolve()
        hf_model_id = os.environ.get("HF_MODEL") or getattr(hf_config, "_name_or_path", None) or HF_MODEL_ID
        generator = build_generator(
            model_dir=model_dir,
            mesh_device=model_mesh,
            hf_model_id=hf_model_id,
            max_batch_size=max(32, int(max_batch_size)),
            num_layers=n_layers,
        )
        if int(max_seq_len) > int(generator.model.config.max_seq_len):
            raise ValueError(
                f"Qwen3-4B selected precision config supports max_model_len={generator.model.config.max_seq_len}, "
                f"got {max_seq_len}"
            )
        return cls(generator, requested_max_batch_size=max_batch_size)

    @classmethod
    def get_max_tokens_all_users(
        cls,
        *,
        model_name: str,
        num_devices: int,
        tt_data_parallel: int,
        max_model_len: int,
        max_num_seqs: int,
    ) -> int:
        del model_name, num_devices, tt_data_parallel, max_num_seqs
        return int(max_model_len)

    @property
    def cache_path(self) -> Path:
        path = self.generator.model_dir / "tt_cache" / "vllm_kv"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def allocate_kv_cache(self, kv_cache_shape: tuple[int, int, int, int], dtype: torch.dtype, num_layers: int):
        del dtype
        if int(num_layers) != int(self.model.num_layers):
            raise ValueError(f"vLLM requested {num_layers} KV-cache layers, model has {self.model.num_layers}")

        num_blocks, heads, block_size, head_dim = (int(value) for value in kv_cache_shape)
        self._num_kv_blocks = num_blocks
        cfg = self.model.decoder_cfg
        if int(block_size) != int(self.model.config.paged_kv_config.block_size):
            raise ValueError(
                f"Qwen3-4B requires vLLM --block-size {self.model.config.paged_kv_config.block_size}, "
                f"got kv_cache_shape={kv_cache_shape}"
            )
        if int(head_dim) != int(cfg.head_dim):
            raise ValueError(f"vLLM requested head_dim={head_dim}, model expects {cfg.head_dim}")

        local_heads = cfg.num_key_value_heads // self.mesh_device.get_num_devices()
        if heads not in (cfg.num_key_value_heads, local_heads):
            raise ValueError(
                f"vLLM requested {heads} KV heads; expected {cfg.num_key_value_heads} global or {local_heads} local"
            )
        local_shape = (num_blocks, local_heads, block_size, head_dim)
        kv_cache: list[list[ttnn.Tensor]] = []
        for layer_idx in range(int(num_layers)):
            zeros = torch.zeros(local_shape, dtype=torch.bfloat16)
            kv_cache.append(
                [
                    ttnn.as_tensor(
                        zeros,
                        device=self.mesh_device,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=self.model.config.paged_kv_config.cache_dtype,
                        cache_file_name=self.cache_path / f"layer_{layer_idx}_{name}_{local_shape}",
                    )
                    for name in ("k", "v")
                ]
            )
        self.generator.kv_cache = kv_cache
        return kv_cache

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        page_table=None,
        kv_cache=None,
        prompt_lens: Iterable[int] | torch.Tensor | None = None,
        sampling_params=None,
        start_pos=None,
        empty_slots: list[int] | None = None,
        **kwargs: Any,
    ):
        del start_pos, kwargs
        kv_cache = self._require_kv_cache(kv_cache)
        prompt_lens_list = _to_int_list(prompt_lens, default=int(tokens.shape[1]))
        page_table_tt = self._page_table_to_tt(page_table)
        if sampling_params is None:
            if page_table is not None:
                row_outputs = []
                for row, prompt_len in enumerate(prompt_lens_list):
                    row_outputs.append(
                        self.model.prefill_forward(
                            tokens[row : row + 1, : int(prompt_len)].to(torch.long),
                            page_table=self._page_table_row_to_tt(page_table, row),
                            kv_cache=kv_cache,
                            prompt_lens=[int(prompt_len)],
                            return_all_logits=False,
                            user_id=0,
                        )
                    )
                return torch.cat(row_outputs, dim=0)
            return self.generator.prefill_forward(
                tokens.to(torch.long),
                page_table=page_table_tt,
                kv_cache=kv_cache,
                prompt_lens=prompt_lens_list,
                return_all_logits=False,
            )

        output_tokens = []
        del empty_slots
        self.reset_warmup_state()
        for row, prompt_len in enumerate(prompt_lens_list):
            logits = self.model.prefill_forward_device_logits(
                tokens[row : row + 1, : int(prompt_len)].to(torch.long),
                page_table=self._page_table_row_to_tt(page_table, row),
                kv_cache=kv_cache,
                prompt_lens=[int(prompt_len)],
                user_id=0,
            )
            tt_out_tok = self.generator._new_replicated_token_buffer(1)
            sampled, _ = self.generator.sample_logits_on_device(
                logits,
                tt_out_tok=tt_out_tok,
                enable_trace=False,
                force_argmax=False,
            )
            sampled_host = _tokens_to_host(sampled, self.mesh_device, active_batch=1)
            output_tokens.append(sampled_host)
        return torch.cat(output_tokens, dim=0).reshape(len(output_tokens), 1).to(torch.int32).contiguous()

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: torch.Tensor,
        page_table=None,
        kv_cache=None,
        enable_trace: bool = True,
        read_from_device: bool = True,
        sampling_params=None,
        reset_batch: bool = False,
        perform_device_sampling: bool = False,
        prompt_tokens: torch.Tensor | None = None,
        output_tokens: torch.Tensor | None = None,
        slot_remap=None,
        **kwargs: Any,
    ):
        del prompt_tokens, output_tokens, slot_remap, kwargs
        kv_cache = self._require_kv_cache(kv_cache)
        if not perform_device_sampling:
            active = start_pos.reshape(-1).to(torch.int32) >= 0
            if not bool(active.all()):
                tokens = tokens.clone()
                start_pos = start_pos.reshape(-1).clone()
                tokens[~active] = 0
                start_pos[~active] = 0
                page_table = self._page_table_with_inactive_scratch(page_table, active)
            page_table_tt = self._page_table_to_tt(page_table)
            return self.generator.decode_forward(
                tokens.to(torch.long),
                start_pos.to(torch.int32),
                page_table=page_table_tt,
                kv_cache=kv_cache,
                enable_trace=enable_trace,
                return_device_logits=not read_from_device,
                use_persistent_ccl=False,
            )
        if not enable_trace:
            raise ValueError("Qwen3-4B vLLM on-device sampling requires traced decode")

        start_flat = start_pos.reshape(-1).to(torch.int32)
        active = start_flat >= 0
        if not bool(active.any()):
            return torch.empty((0, 1), dtype=torch.int32)
        if not bool(active.all()):
            tokens = tokens.reshape(tokens.shape[0], -1)[active, :1].contiguous()
            start_pos = start_flat[active].contiguous()
            page_table = _page_table_rows(page_table, active)
        page_table_tt = self._page_table_to_tt(page_table)
        batch_size = int(tokens.shape[0])
        # Sampling1D's force-argmax path writes through ttnn.argmax(..., output_tensor=...)
        # and is not trace-capture safe. Use the trace-safe TP4 greedy sampler for token feedback.
        force_argmax = False
        if reset_batch or self._decode_trace_batch_size != batch_size or self.model.trace_state.trace_id is None:
            self.generator.kv_cache = kv_cache
            device_out = self.generator.prepare_token_out_decode(
                first_input_tokens=tokens.reshape(batch_size, -1)[:, 0].to(torch.long),
                start_positions=start_pos.reshape(-1).to(torch.int32),
                prompt_lens=[int(value) for value in start_pos.reshape(-1).tolist()],
                page_table=page_table_tt,
                read_first_token=False,
                return_device_output=True,
                force_argmax=force_argmax,
            )
            self._decode_trace_batch_size = batch_size
        else:
            device_out = self.generator.decode_next_token_traced(
                page_table=page_table_tt,
                page_table_generation=self._vllm_page_table_generation,
                force_argmax=force_argmax,
            )

        if read_from_device:
            return self.process_decode_output_host(self.read_decode_output(device_out), is_tokens=True)
        return device_out

    def read_decode_output(self, tt_out: Any, async_read: bool = False):
        host_out, events = _read_tt_output(tt_out, self.mesh_device, async_read=async_read)
        return (host_out, events) if async_read else host_out

    def process_decode_output_host(self, tt_out: Any, *, is_tokens: bool = True):
        if is_tokens:
            return _tokens_to_host(tt_out, self.mesh_device, active_batch=self._decode_trace_batch_size)
        return _logits_to_host(tt_out, self.mesh_device, self.vocab_size)

    def warmup_model_prefill(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.reset_warmup_state()

    def warmup_model_decode(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self.reset_warmup_state()

    def reset_warmup_state(self) -> None:
        self.generator._release_traces()
        self._decode_trace_batch_size = None

    def _require_kv_cache(self, kv_cache):
        if kv_cache is None:
            raise ValueError("Qwen3-4B vLLM path requires a vLLM-owned kv_cache")
        return kv_cache

    def _page_table_to_tt(self, page_table):
        if page_table is None:
            return self.generator.page_table
        if isinstance(page_table, ttnn.Tensor):
            return page_table
        host = torch.as_tensor(page_table, dtype=torch.int32).contiguous()
        if self._vllm_page_table_host is not None and torch.equal(host, self._vllm_page_table_host):
            return self._vllm_page_table_tt
        self._vllm_page_table_host = host.clone()
        self._vllm_page_table_generation += 1
        if self._vllm_page_table_tt is None or list(self._vllm_page_table_tt.shape) != list(host.shape):
            self._vllm_page_table_tt = ttnn.from_torch(
                host,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            host_tensor = ttnn.from_torch(
                host,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            ttnn.copy_host_to_device_tensor(host_tensor, self._vllm_page_table_tt)
        return self._vllm_page_table_tt

    def _page_table_row_to_tt(self, page_table, row: int):
        row_table = _page_table_row(page_table, row)
        if row_table is None or isinstance(row_table, ttnn.Tensor):
            return self._page_table_to_tt(row_table)
        return ttnn.from_torch(
            torch.as_tensor(row_table, dtype=torch.int32).contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def _page_table_with_inactive_scratch(self, page_table, active: torch.Tensor):
        if page_table is None or isinstance(page_table, ttnn.Tensor):
            return page_table
        if self._num_kv_blocks is None:
            raise ValueError("vLLM KV cache must be allocated before padded decode")
        host = torch.as_tensor(page_table, dtype=torch.int32).clone().contiguous()
        if host.dim() < 2:
            return host
        inactive_rows = (~active.to(torch.bool).cpu()).nonzero(as_tuple=False).reshape(-1)
        if inactive_rows.numel() == 0:
            return host
        scratch_start = max(0, int(self._num_kv_blocks) - int(host.shape[0]))
        for row in inactive_rows.tolist():
            host[int(row), :] = scratch_start + int(row)
        return host


def _select_tp4_mesh(mesh_device):
    if not isinstance(mesh_device, ttnn.MeshDevice):
        return mesh_device
    if tuple(mesh_device.shape) == (1, 4) and mesh_device.get_num_devices() == 4:
        return mesh_device
    if mesh_device.get_num_devices() >= 4 and tuple(mesh_device.shape)[0] == 1:
        return mesh_device.create_submeshes(ttnn.MeshShape(1, 4))[0]
    raise ValueError(f"Qwen3-4B requires a 1x4 TP4 mesh or larger 1D mesh, got shape={tuple(mesh_device.shape)}")


def _to_int_list(values, *, default: int) -> list[int]:
    if values is None:
        return [int(default)]
    if isinstance(values, torch.Tensor):
        return [int(value) for value in values.reshape(-1).tolist()]
    return [int(value) for value in values]


def _page_table_row(page_table, row: int):
    if page_table is None or isinstance(page_table, ttnn.Tensor):
        return page_table
    host = torch.as_tensor(page_table, dtype=torch.int32)
    if host.dim() < 2:
        return host
    return host[int(row) : int(row) + 1].contiguous()


def _page_table_rows(page_table, rows: torch.Tensor):
    if page_table is None or isinstance(page_table, ttnn.Tensor):
        return page_table
    host = torch.as_tensor(page_table, dtype=torch.int32)
    if host.dim() < 2:
        return host
    return host[rows.to(torch.bool).cpu()].contiguous()


def _read_tt_output(obj: Any, mesh_device, *, async_read: bool) -> tuple[Any, list[Any]]:
    if obj is None or isinstance(obj, torch.Tensor):
        return obj, []
    if isinstance(obj, tuple):
        values = []
        events = []
        for item in obj:
            value, item_events = _read_tt_output(item, mesh_device, async_read=async_read)
            values.append(value)
            events.extend(item_events)
        return tuple(values), events
    if isinstance(obj, ttnn.Tensor):
        if async_read:
            host = ttnn.from_device(obj, blocking=False, cq_id=0)
            return host, [ttnn.record_event(mesh_device, 0)]
        return ttnn.from_device(obj, blocking=True, cq_id=0), []
    raise TypeError(f"Unsupported TT output type {type(obj).__name__}")


def _tokens_to_host(obj: Any, mesh_device, *, active_batch: int | None) -> torch.Tensor:
    if isinstance(obj, tuple):
        return _tokens_to_host(obj[0], mesh_device, active_batch=active_batch)
    if isinstance(obj, ttnn.Tensor):
        obj = ttnn.to_torch(ttnn.get_device_tensors(obj)[0])
    elif not isinstance(obj, torch.Tensor):
        obj = torch.as_tensor(obj)
    tokens = obj.reshape(-1).to(torch.int32)
    if active_batch is not None:
        tokens = tokens[: int(active_batch)]
    return tokens.contiguous()


def _force_argmax_sampling(sampling_params: Any) -> bool:
    if sampling_params is None or not hasattr(sampling_params, "temperature"):
        return False
    temperature = getattr(sampling_params, "temperature")
    if temperature is None:
        return False
    values = torch.as_tensor(temperature, dtype=torch.float32).reshape(-1)
    return bool(values.numel() > 0 and torch.all(values <= 0.0).item())


def _logits_to_host(obj: Any, mesh_device, vocab_size: int) -> torch.Tensor:
    if isinstance(obj, tuple):
        obj = obj[0]
    if isinstance(obj, ttnn.Tensor):
        obj = ttnn.to_torch(obj, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))
    elif not isinstance(obj, torch.Tensor):
        obj = torch.as_tensor(obj)
    if obj.dim() == 2:
        obj = obj.unsqueeze(1)
    elif obj.dim() == 4 and obj.shape[0] == 1 and obj.shape[1] == 1:
        obj = obj.reshape(obj.shape[2], 1, obj.shape[3])
    elif obj.dim() > 3:
        obj = obj.reshape(-1, obj.shape[-2], obj.shape[-1])
    return obj.to(torch.float32)[..., :vocab_size].contiguous()


def allocate_vllm_kv_cache(*args: Any, **kwargs: Any):
    del args, kwargs
    raise NotImplementedError("Use Qwen3ForCausalLM.allocate_kv_cache after initialize_vllm_model")


__all__ = ["Qwen3ForCausalLM", "allocate_vllm_kv_cache"]
