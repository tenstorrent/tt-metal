# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full TP=4 Llama 3.1 8B autoregressive model.

This wrapper deliberately retains the optimized multichip decoder as the only
block implementation.  Embedding and the LM head are tensor-parallel, the
inter-layer residual stays replicated across ranks and width-sharded in each
device's L1, and decode exposes device-resident positions for trace replay.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.multichip_decoder import (
    PAGED_BLOCK_SIZE,
    TARGET_MESH_SHAPE,
    TARGET_TP_DEGREE,
    MultiChipConfig,
    MultiChipDecoder,
    _mesh_tensor,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.optimized_decoder import (
    TILE_SIZE,
    _dram_matmul_program_config,
    _dram_weight_memory_config,
    _width_sharded_l1,
)
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.modules.tt_ccl import TT_CCL, get_tt_ccl

HF_CONTEXT_LENGTH = 131072
HF_VOCAB_SIZE = 128256
PADDED_VOCAB_SIZE = 131072
DEFAULT_NUM_BLOCKS = HF_CONTEXT_LENGTH // PAGED_BLOCK_SIZE
DEFAULT_PREFILL_CHUNK_SIZE = 2048
LM_HEAD_COLUMNS_PER_DEVICE = 8192


def _config_value(config, name: str, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _terminal_state_tensor(state_dict: Mapping[str, torch.Tensor], suffix: str) -> torch.Tensor:
    candidates = (suffix, f"model.{suffix}", f"module.{suffix}")
    for name in candidates:
        try:
            return state_dict[name]
        except (KeyError, TypeError):
            continue
    matches = [key for key in state_dict if key.endswith(suffix)]
    if len(matches) == 1:
        return state_dict[matches[0]]
    raise KeyError(f"unable to resolve terminal weight {suffix!r}; matches={matches[:8]}")


def _out_subblock_w(per_core_n: int) -> int:
    for width in (4, 3, 2, 1):
        if per_core_n % width == 0:
            return width
    return 1


@dataclass(frozen=True)
class FullModelConfig:
    """Full-stack policy carried forward from the accepted TP=4 decoder."""

    multichip: MultiChipConfig = field(default_factory=MultiChipConfig)
    max_batch_size: int = 1
    max_context_len: int = HF_CONTEXT_LENGTH
    num_blocks: int = DEFAULT_NUM_BLOCKS
    prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE
    kv_cache_dtype: object = ttnn.bfloat8_b
    lm_head_weight_dtype: object = ttnn.bfloat8_b
    lm_head_math_fidelity: object = ttnn.MathFidelity.HiFi2
    lm_head_cores: int = 64
    lm_head_in0_block_w: int = 2
    override_num_layers: int | None = None

    def validate(self, hf_config) -> None:
        if self.max_batch_size < 1 or self.max_batch_size > 32:
            raise ValueError("max_batch_size must be in [1, 32]")
        if self.max_context_len < 1 or self.max_context_len > int(
            _config_value(hf_config, "max_position_embeddings", HF_CONTEXT_LENGTH)
        ):
            raise ValueError("max_context_len exceeds the Hugging Face capability")
        if self.prefill_chunk_size < 128 or self.prefill_chunk_size % 128:
            raise ValueError("prefill_chunk_size must be a positive multiple of 128")
        if self.num_blocks * PAGED_BLOCK_SIZE < self.max_context_len:
            raise ValueError("num_blocks does not cover max_context_len")
        if self.kv_cache_dtype not in (ttnn.bfloat8_b, ttnn.bfloat16):
            raise ValueError("the accepted paged-cache split permits only BFP8 or BF16")
        if self.lm_head_weight_dtype not in (ttnn.bfloat8_b, ttnn.bfloat4_b):
            raise ValueError("lm_head_weight_dtype must be BFP8 or BFP4")


class Llama31FullModel:
    """Embedding, 32 optimized TP=4 blocks, final norm, LM head, and sampler."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        config: FullModelConfig,
        embedding_weight,
        final_norm_weight,
        lm_head_weights,
        logit_mask,
        layers,
        tt_ccl: TT_CCL,
        sampler: Sampling1D,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config
        self.embedding_weight = embedding_weight
        self.final_norm_weight = final_norm_weight
        self.lm_head_weights = lm_head_weights
        self.logit_mask = logit_mask
        self.layers = layers
        self.tt_ccl = tt_ccl
        self.sampler = sampler

        self.batch = config.max_batch_size
        self.hidden_size = int(_config_value(hf_config, "hidden_size"))
        self.num_heads = int(_config_value(hf_config, "num_attention_heads"))
        self.num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        self.head_dim = int(_config_value(hf_config, "head_dim", self.hidden_size // self.num_heads))
        self.vocab_size = int(_config_value(hf_config, "vocab_size"))
        self.padded_vocab_size = PADDED_VOCAB_SIZE
        self.local_vocab_size = self.padded_vocab_size // TARGET_TP_DEGREE
        self.num_layers = len(layers)

        padded_rows = TILE_SIZE * math.ceil(self.batch / TILE_SIZE)
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.lm_head_input_mem_config = _width_sharded_l1(
            rows=padded_rows,
            width=self.hidden_size,
            cores=config.lm_head_cores,
            device_grid=device_grid,
        )
        self.lm_head_decode_program_configs = [
            _dram_matmul_program_config(
                role=f"tp_lm_head_{split}",
                m=padded_rows,
                k=self.hidden_size,
                n=LM_HEAD_COLUMNS_PER_DEVICE,
                cores=config.lm_head_cores,
                in0_block_w=config.lm_head_in0_block_w,
            )
            for split in range(self.local_vocab_size // LM_HEAD_COLUMNS_PER_DEVICE)
        ]
        self.lm_head_compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=config.lm_head_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._prefill_lm_head_program_configs: dict[tuple[int, int], object] = {}

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        mesh_device,
        full_model_config: FullModelConfig | None = None,
    ) -> "Llama31FullModel":
        config = full_model_config or FullModelConfig()
        config.validate(hf_config)
        if mesh_device.get_num_devices() != TARGET_TP_DEGREE or tuple(mesh_device.shape) != TARGET_MESH_SHAPE:
            raise ValueError(f"full model requires mesh {TARGET_MESH_SHAPE}")

        hidden_size = int(_config_value(hf_config, "hidden_size"))
        num_heads = int(_config_value(hf_config, "num_attention_heads"))
        head_dim = int(_config_value(hf_config, "head_dim", hidden_size // num_heads))
        vocab_size = int(_config_value(hf_config, "vocab_size"))
        if (hidden_size, num_heads, int(_config_value(hf_config, "num_key_value_heads")), vocab_size) != (
            4096,
            32,
            8,
            HF_VOCAB_SIZE,
        ):
            raise ValueError("this optimized full-model policy is specific to Llama 3.1 8B")

        tt_ccl = get_tt_ccl(mesh_device)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        shard_hidden = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        shared_cos, shared_sin = MultiChipDecoder._make_rotary_tables(
            hf_config, config.max_context_len, head_dim, mesh_device
        )
        decode_cos, decode_sin = MultiChipDecoder._make_rotary_tables(
            hf_config,
            config.max_context_len,
            head_dim,
            mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        position_indices = _mesh_tensor(
            torch.arange(config.max_context_len, dtype=torch.int32).unsqueeze(1).expand(-1, config.max_batch_size),
            mesh_device,
            mesh_mapper=replicate,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
        )

        embedding = _terminal_state_tensor(state_dict, "embed_tokens.weight").to(torch.bfloat16)
        if tuple(embedding.shape) != (vocab_size, hidden_size):
            raise ValueError(f"embedding shape {tuple(embedding.shape)} is not {(vocab_size, hidden_size)}")
        embedding_weight = _mesh_tensor(
            embedding,
            mesh_device,
            mesh_mapper=shard_hidden,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        del embedding

        final_norm = _terminal_state_tensor(state_dict, "norm.weight").to(torch.bfloat16)
        final_norm_weight = _mesh_tensor(final_norm, mesh_device, mesh_mapper=replicate)
        del final_norm

        lm_head = _terminal_state_tensor(state_dict, "lm_head.weight").to(torch.bfloat16).transpose(0, 1)
        lm_head = torch.nn.functional.pad(lm_head, (0, PADDED_VOCAB_SIZE - vocab_size))
        local_vocab_size = PADDED_VOCAB_SIZE // TARGET_TP_DEGREE
        lm_head_weights = []
        for split_start in range(0, local_vocab_size, LM_HEAD_COLUMNS_PER_DEVICE):
            rank_splits = [
                lm_head[
                    :,
                    rank * local_vocab_size
                    + split_start : rank * local_vocab_size
                    + split_start
                    + LM_HEAD_COLUMNS_PER_DEVICE,
                ]
                for rank in range(TARGET_TP_DEGREE)
            ]
            combined = torch.cat(rank_splits, dim=1)
            lm_head_weights.append(
                _mesh_tensor(
                    combined,
                    mesh_device,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
                    dtype=config.lm_head_weight_dtype,
                    memory_config=_dram_weight_memory_config(
                        mesh_device,
                        k=hidden_size,
                        n=LM_HEAD_COLUMNS_PER_DEVICE,
                    ),
                )
            )
        del lm_head

        mask = torch.zeros((1, 1, 1, PADDED_VOCAB_SIZE), dtype=torch.bfloat16)
        mask[..., vocab_size:] = torch.finfo(torch.bfloat16).min
        logit_mask = _mesh_tensor(
            mask,
            mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
            dtype=ttnn.bfloat16,
        )

        total_layers = int(_config_value(hf_config, "num_hidden_layers"))
        num_layers = config.override_num_layers if config.override_num_layers is not None else total_layers
        if num_layers < 1 or num_layers > total_layers:
            raise ValueError(f"override_num_layers must be in [1, {total_layers}]")
        layers = []
        for layer_idx in range(num_layers):
            layer = MultiChipDecoder.from_state_dict(
                state_dict,
                hf_config=hf_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                batch=config.max_batch_size,
                max_cache_len=config.max_context_len,
                multichip_config=config.multichip,
                tt_ccl=tt_ccl,
                rotary_cos=shared_cos,
                rotary_sin=shared_sin,
                decode_rotary_cos=decode_cos,
                decode_rotary_sin=decode_sin,
                position_indices=position_indices,
            )
            layer.prepare_decode()
            layers.append(layer)

        sampler = Sampling1D.from_config(
            Sampling1DConfig(
                vocab_size=PADDED_VOCAB_SIZE,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                max_batch_size=32,
                max_top_k=32,
                num_gather_links=config.multichip.num_links,
                sampling_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                allow_force_argmax=True,
                num_argmax_gather_links=config.multichip.num_links,
                ag_topology=ttnn.Topology.Ring,
                # The TP4 LM-head shard is 32,768 entries wide (131,072 padded
                # globally). Stochastic top-k therefore already has a native
                # power-of-two local reduction width; exact greedy reduces the
                # same shard to one value/token candidate per rank.
                pad_to_power_of_2=True,
            )
        )
        return cls(
            mesh_device=mesh_device,
            hf_config=hf_config,
            config=config,
            embedding_weight=embedding_weight,
            final_norm_weight=final_norm_weight,
            lm_head_weights=lm_head_weights,
            logit_mask=logit_mask,
            layers=layers,
            tt_ccl=tt_ccl,
            sampler=sampler,
        )

    def allocate_kv_cache(self, *, num_blocks: int | None = None, dtype=None):
        """Allocate caller-owned paged cache with global heads sharded TP=4."""

        num_blocks = self.config.num_blocks if num_blocks is None else int(num_blocks)
        dtype = self.config.kv_cache_dtype if dtype is None else dtype
        if num_blocks < 1 or num_blocks > self.config.num_blocks:
            raise ValueError(f"num_blocks must be in [1, {self.config.num_blocks}]")
        if dtype not in (ttnn.bfloat8_b, ttnn.bfloat16):
            raise ValueError("KV cache dtype must be BFP8 or BF16")
        host_zero = torch.zeros(
            (num_blocks, self.num_kv_heads, PAGED_BLOCK_SIZE, self.head_dim),
            dtype=torch.bfloat16,
        )
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=1)
        caches = []
        for _ in range(self.num_layers):
            caches.append(
                [
                    ttnn.as_tensor(
                        host_zero,
                        device=self.mesh_device,
                        mesh_mapper=mapper,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=dtype,
                    )
                    for _ in range(2)
                ]
            )
        return caches

    @staticmethod
    def reset_kv_cache(kv_cache) -> None:
        """Zero cache buffers in place so trace-bound addresses remain stable."""

        for layer_cache in kv_cache:
            for cache in layer_cache:
                ttnn.fill(cache, 0.0, output_tensor=cache)

    def _all_gather_hidden(self, local_hidden):
        return ttnn.experimental.all_gather_async(
            local_hidden,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=self.config.multichip.num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    def embed_prefill(self, tokens):
        local = ttnn.embedding(
            tokens,
            self.embedding_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        local = ttnn.unsqueeze_to_4D(local)
        return self._all_gather_hidden(local)

    def embed_decode(self, tokens):
        local = ttnn.embedding(
            tokens,
            self.embedding_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        local = ttnn.unsqueeze_to_4D(local)
        hidden = self._all_gather_hidden(local)
        hidden = ttnn.slice(
            hidden,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.hidden_size],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(hidden, [1, self.batch, 1, self.hidden_size])

    def _prefill_lm_head_program_config(self, batch: int, seq_len: int):
        key = (batch, seq_len)
        if key in self._prefill_lm_head_program_configs:
            return self._prefill_lm_head_program_configs[key]
        grid_x, grid_y = (8, 8)
        per_core_n = LM_HEAD_COLUMNS_PER_DEVICE // TILE_SIZE // grid_x
        config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=8,
            out_subblock_h=1,
            out_subblock_w=_out_subblock_w(per_core_n),
            per_core_M=max(1, math.ceil(batch * math.ceil(seq_len / TILE_SIZE) / grid_y)),
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )
        self._prefill_lm_head_program_configs[key] = config
        return config

    def _final_norm_prefill(self, hidden):
        return ttnn.rms_norm(
            hidden,
            epsilon=float(_config_value(self.hf_config, "rms_norm_eps")),
            weight=self.final_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.layers[-1].norm_compute_kernel,
        )

    def _final_norm_decode(self, hidden):
        last = self.layers[-1]
        hidden = ttnn.reshape(hidden, [1, 1, self.batch, self.hidden_size])
        hidden = ttnn.to_memory_config(hidden, last.advisor_input_norm_mem_config)
        hidden = ttnn.rms_norm(
            hidden,
            epsilon=float(_config_value(self.hf_config, "rms_norm_eps")),
            weight=self.final_norm_weight,
            program_config=last.advisor_input_norm_program_config,
            memory_config=last.advisor_input_norm_mem_config,
            compute_kernel_config=last.norm_compute_kernel,
        )
        if self.batch < 32:
            hidden = ttnn.pad(
                hidden,
                [(0, 0), (0, 0), (0, 32 - self.batch), (0, 0)],
                value=0.0,
            )
        return ttnn.to_memory_config(hidden, self.lm_head_input_mem_config)

    def _lm_head_prefill(self, hidden, *, batch: int, seq_len: int):
        logits = ttnn.concat(
            [
                ttnn.linear(
                    hidden,
                    weight,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=self._prefill_lm_head_program_config(batch, seq_len),
                    compute_kernel_config=self.lm_head_compute_kernel,
                )
                for weight in self.lm_head_weights
            ],
            dim=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.add(
            logits,
            self.logit_mask,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _lm_head_decode(self, hidden):
        outputs = []
        for weight, program_config in zip(self.lm_head_weights, self.lm_head_decode_program_configs):
            split = ttnn.linear(
                hidden,
                weight,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=program_config,
                compute_kernel_config=self.lm_head_compute_kernel,
            )
            outputs.append(ttnn.sharded_to_interleaved(split, memory_config=ttnn.L1_MEMORY_CONFIG))
        logits = ttnn.concat(outputs, dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.add(
            logits,
            self.logit_mask,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

    def select_prefill_token_hidden(self, hidden, user_idx: int, token_idx: int):
        """Retain one user's final hidden row without retaining its whole chunk."""

        if user_idx < 0 or user_idx >= hidden.shape[1]:
            raise ValueError(f"user_idx={user_idx} is outside prefill hidden shape {tuple(hidden.shape)}")
        if token_idx < 0 or token_idx >= hidden.shape[2]:
            raise ValueError(f"token_idx={token_idx} is outside prefill hidden shape {tuple(hidden.shape)}")
        return ttnn.slice(
            hidden,
            [0, user_idx, token_idx, 0],
            [1, user_idx + 1, token_idx + 1, self.hidden_size],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def prefill_selected_hidden_logits(self, selected_rows, *, fixed_sampling_rows: bool = False):
        """Run the terminal path over ordered ``[1, 1, 1, hidden]`` rows."""

        if not selected_rows or len(selected_rows) > self.batch:
            raise ValueError(f"expected between 1 and {self.batch} selected prefill rows")
        selected = selected_rows[0] if len(selected_rows) == 1 else ttnn.concat(selected_rows, dim=2)
        seq_len = len(selected_rows)
        if fixed_sampling_rows and seq_len < 32:
            selected = ttnn.pad(selected, [(0, 0), (0, 0), (0, 32 - seq_len), (0, 0)], value=0.0)
            seq_len = 32
        return self._lm_head_prefill(
            self._final_norm_prefill(selected),
            batch=1,
            seq_len=seq_len,
        )

    def prefill_last_token_logits(self, hidden, token_idx: int, *, fixed_sampling_rows: bool = False):
        """Apply the terminal path only to batch row zero's selected prefill token."""

        return self.prefill_selected_hidden_logits(
            [self.select_prefill_token_hidden(hidden, 0, token_idx)],
            fixed_sampling_rows=fixed_sampling_rows,
        )

    def prefill_forward(
        self,
        tokens,
        *,
        page_table,
        kv_cache,
        prompt_lens,
        chunk_start_idx: int = 0,
        chunk_page_table=None,
        return_hidden: bool = False,
    ):
        batch, seq_len = int(tokens.shape[-2]), int(tokens.shape[-1])
        if batch != self.batch:
            raise ValueError(f"tokens batch={batch} does not match configured batch={self.batch}")
        hidden = self.embed_prefill(tokens)
        for layer, (key_cache, value_cache) in zip(self.layers, kv_cache):
            hidden = layer.prefill_forward(
                hidden,
                key_cache,
                value_cache,
                page_table=page_table,
                prompt_lens=prompt_lens,
                chunk_start_idx=chunk_start_idx,
                chunk_page_table=chunk_page_table,
            )
        if return_hidden:
            return hidden
        return self._lm_head_prefill(self._final_norm_prefill(hidden), batch=batch, seq_len=seq_len)

    def decode_forward(
        self,
        tokens,
        current_pos,
        rotary_position,
        *,
        page_table,
        kv_cache,
        advance_positions: bool = True,
    ):
        hidden = self.embed_decode(tokens)
        for layer, (key_cache, value_cache) in zip(self.layers, kv_cache):
            hidden = layer.decode_forward(
                hidden,
                key_cache,
                value_cache,
                current_pos=current_pos,
                rotary_position=rotary_position,
                page_table=page_table,
            )
        logits = self._lm_head_decode(self._final_norm_decode(hidden))
        if advance_positions:
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            ttnn.plus_one(rotary_position)
        return logits

    def sample_greedy_split(self, logits, *, k, p, temp, tt_out_tok=None):
        """Exact greedy split sampling via a packed per-rank max/token gather."""

        return self.sampler.greedy_decode_forward(logits, tt_out_tok=tt_out_tok)

    def sample_stochastic_split(self, logits, *, k, p, temp, tt_out_tok=None):
        """Traceable top-k/top-p sampling for requests with multiple candidates."""

        return self.sampler.decode_forward(
            logits,
            k=k,
            p=p,
            temp=temp,
            tt_out_tok=tt_out_tok,
            enable_log_probs=False,
        )[0]

    def sample_force_argmax(self, logits, *, tt_out_tok=None):
        """Comparison-only common path; gathers the vocabulary before argmax."""

        return self.sampler.decode_forward(logits, tt_out_tok=tt_out_tok, enable_log_probs=False)[0]


__all__ = [
    "DEFAULT_NUM_BLOCKS",
    "DEFAULT_PREFILL_CHUNK_SIZE",
    "FullModelConfig",
    "HF_CONTEXT_LENGTH",
    "HF_VOCAB_SIZE",
    "Llama31FullModel",
    "LM_HEAD_COLUMNS_PER_DEVICE",
    "PADDED_VOCAB_SIZE",
]
