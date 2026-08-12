# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full TP4 Falcon3-7B autoregressive model.

This module deliberately keeps the optimized multichip decoder contract intact:
the residual is replicated across the 1x4 mesh and width-sharded over 32 L1
cores inside each device, decoder weights use the selected BFP4/LoFi policy,
decode projection/CCL activations use BFP8, and paged K/V caches use BFP8.

Only true model boundaries change layout.  The embedding table is sharded over
the hidden dimension and gathered once into the decoder residual contract.  The
terminal LM head is sharded over vocabulary and emits sampler-ready local logits;
the normal token-out path never gathers the full vocabulary.
"""

from __future__ import annotations

import gc
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.tiiuae_falcon3_7b_base.tt.multichip_decoder import (
    DEFAULT_PAGE_BLOCK_SIZE,
    TARGET_MESH_SHAPE,
    TENSOR_PARALLEL_AXIS,
    TENSOR_PARALLEL_SIZE,
    MultichipDecoder,
    _mesh_weight,
)
from models.autoports.tiiuae_falcon3_7b_base.tt.optimized_decoder import (
    _compute_config,
    _dram_matmul_program_config,
    _dram_sharded_memory_config,
)
from models.common.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.modules.tt_ccl import get_tt_ccl
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict

HF_MODEL_ID = "tiiuae/Falcon3-7B-Base"
HF_REVISION = "bf3d7ed586cb22a921520e2d681a9d3d7642cde8"
VOCAB_SIZE = 131072
HIDDEN_SIZE = 3072
NUM_LAYERS = 28
MAX_CONTEXT = 32768
DEFAULT_MAX_BATCH_SIZE = 1
DEFAULT_TRACE_REGION_SIZE = 512_000_000
DEFAULT_NUM_BLOCKS = MAX_CONTEXT // DEFAULT_PAGE_BLOCK_SIZE
DEFAULT_PREFILL_CHUNK_SIZE = 2048
LM_HEAD_COLUMNS_PER_DEVICE = 8192


def _validate_mesh(mesh_device) -> None:
    shape = tuple(int(value) for value in mesh_device.shape)
    if shape != TARGET_MESH_SHAPE:
        raise ValueError(f"Falcon3Model requires mesh {TARGET_MESH_SHAPE}, got {shape}")
    if mesh_device.get_num_devices() != TENSOR_PARALLEL_SIZE:
        raise ValueError(f"Falcon3Model requires exactly {TENSOR_PARALLEL_SIZE} devices")


def _replicated_device_tensor(host: torch.Tensor, *, mesh_device, dtype, layout, memory_config):
    return ttnn.from_torch(
        host.contiguous(),
        device=mesh_device,
        dtype=dtype,
        layout=layout,
        memory_config=memory_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _checkpoint_state(checkpoint_path: str | Path) -> LazyStateDict:
    checkpoint_path = Path(checkpoint_path)
    index_path = checkpoint_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"Falcon3 checkpoint index is missing: {index_path}")
    return LazyStateDict(checkpoint_path)


class Falcon3Model:
    """Complete Falcon3 CausalLM over the selected TP4 decoder stack."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        state_dict: Mapping[str, torch.Tensor],
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_cache_len: int = MAX_CONTEXT,
        num_layers: int = NUM_LAYERS,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        weight_cache_path: str | Path | None = None,
    ) -> None:
        _validate_mesh(mesh_device)
        if not 1 <= int(max_batch_size) <= 32:
            raise ValueError(f"max_batch_size must be in [1,32], got {max_batch_size}")
        if not 1 <= int(num_layers) <= int(hf_config.num_hidden_layers):
            raise ValueError(f"num_layers must be in [1,{hf_config.num_hidden_layers}], got {num_layers}")
        if not 1 <= int(max_cache_len) <= int(hf_config.max_position_embeddings):
            raise ValueError(f"max_cache_len must be in [1,{hf_config.max_position_embeddings}], got {max_cache_len}")
        if int(hf_config.hidden_size) != HIDDEN_SIZE or int(hf_config.vocab_size) != VOCAB_SIZE:
            raise ValueError("HF config does not match the Falcon3-7B full-model contract")
        if bool(hf_config.tie_word_embeddings):
            raise ValueError("Falcon3-7B-Base uses an untied LM head; tied weights would be a different contract")

        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.max_batch_size = int(max_batch_size)
        self.max_cache_len = int(max_cache_len)
        self.num_layers = int(num_layers)
        self.page_block_size = int(page_block_size)
        self.vocab_size = int(hf_config.vocab_size)
        self.hidden_size = int(hf_config.hidden_size)
        self.padded_vocab_size = math.ceil(self.vocab_size / (32 * TENSOR_PARALLEL_SIZE)) * (32 * TENSOR_PARALLEL_SIZE)
        self.local_vocab_size = self.padded_vocab_size // TENSOR_PARALLEL_SIZE
        self.weight_cache_path = Path(weight_cache_path) if weight_cache_path is not None else None
        self.tt_ccl = get_tt_ccl(mesh_device)

        self.embedding = self._build_embedding(state_dict)
        self.layers = self._build_layers(state_dict)
        self.final_norm_weight = self._build_final_norm(state_dict)
        self.lm_head = self._build_lm_head(state_dict)
        self.sampler = Sampling1D.from_config(
            Sampling1DConfig(
                vocab_size=self.padded_vocab_size,
                mesh_device=self.mesh_device,
                tt_ccl=self.tt_ccl,
                max_batch_size=32,
                max_top_k=32,
                num_gather_links=2,
                sampling_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                allow_force_argmax=True,
                num_argmax_gather_links=2,
                ag_topology=ttnn.Topology.Ring,
                pad_to_power_of_2=True,
            )
        )

        # Force lazy terminal weights to materialize while their source mmap is
        # still alive.  Runtime paths contain no lazy host/device transfers.
        self.embedding.load_device_weights()
        self.lm_head.load_device_weights()

        self.kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]] | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        mesh_device,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        max_cache_len: int = MAX_CONTEXT,
        num_layers: int = NUM_LAYERS,
        page_block_size: int = DEFAULT_PAGE_BLOCK_SIZE,
        weight_cache_path: str | Path | None = None,
    ) -> "Falcon3Model":
        checkpoint_path = Path(checkpoint_path)
        hf_config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=True)
        state = _checkpoint_state(checkpoint_path)
        try:
            model = cls(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=state,
                max_batch_size=max_batch_size,
                max_cache_len=max_cache_len,
                num_layers=num_layers,
                page_block_size=page_block_size,
                weight_cache_path=weight_cache_path,
            )
        finally:
            state.clear_cache()
            state.close()
            gc.collect()
        return model

    def _cache_name(self, name: str) -> tuple[Path, str] | None:
        if self.weight_cache_path is None:
            return None
        directory = self.weight_cache_path / "full_model"
        directory.mkdir(parents=True, exist_ok=True)
        return directory, name

    def _build_embedding(self, state_dict: Mapping[str, torch.Tensor]) -> Embedding1D:
        host = state_dict["model.embed_tokens.weight"].unsqueeze(0).unsqueeze(0)
        weight = LazyWeight(
            source=host,
            dtype=ttnn.bfloat16,
            device=self.mesh_device,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(-1)],
                mesh_shape_override=ttnn.MeshShape([TENSOR_PARALLEL_SIZE]),
            ),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_dir_weight_name=self._cache_name("embedding_hidden_sharded"),
        )
        return Embedding1D.from_config(
            Embedding1DConfig(
                weights=weight,
                mesh_device=self.mesh_device,
                embed_scale=1.0,
                weights_dtype=ttnn.bfloat16,
                weights_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    def _build_layers(self, state_dict: Mapping[str, torch.Tensor]) -> list[MultichipDecoder]:
        layers: list[MultichipDecoder] = []
        rope_cache = None
        persistent_ccl_resources = None
        for layer_idx in range(self.num_layers):
            layer = MultichipDecoder.from_state_dict(
                state_dict,
                hf_config=self.hf_config,
                layer_idx=layer_idx,
                mesh_device=self.mesh_device,
                batch=self.max_batch_size,
                max_cache_len=self.max_cache_len,
                precision_policy="all_bfp4_lofi",
                decode_matmul_mode="dram_sharded",
                use_packed_mlp=False,
                return_sharded_decode_output=True,
                qkv_target_cores=4,
                o_target_cores=4,
                gate_up_target_cores=24,
                down_target_cores=8,
                ccl_dtype=ttnn.bfloat8_b,
                ccl_mode="persistent_async",
                num_links=2,
                attention_activation_dtype=ttnn.bfloat8_b,
                mlp_activation_dtype=ttnn.bfloat8_b,
                page_block_size=self.page_block_size,
                rope_cache=rope_cache,
                persistent_ccl_resources=persistent_ccl_resources,
            )
            if rope_cache is None:
                rope_cache = (layer.cos_cache, layer.sin_cache)
                persistent_ccl_resources = layer.persistent_ccl_resources
            layers.append(layer)
            clear_cache = getattr(state_dict, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()
        return layers

    def _build_final_norm(self, state_dict: Mapping[str, torch.Tensor]):
        return _mesh_weight(
            state_dict["model.norm.weight"],
            dtype=ttnn.bfloat16,
            mesh_device=self.mesh_device,
            shard_dim=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _build_lm_head(self, state_dict: Mapping[str, torch.Tensor]) -> LMHead1D:
        host = state_dict["lm_head.weight"].transpose(-2, -1).contiguous()
        if self.padded_vocab_size != self.vocab_size:
            host = torch.cat((host, host.new_zeros(self.hidden_size, self.padded_vocab_size - self.vocab_size)), dim=-1)
        grid = self.layers[-1].residual_grid
        weights = []
        memory_configs = []
        program_configs = []
        for split_start in range(0, self.local_vocab_size, LM_HEAD_COLUMNS_PER_DEVICE):
            rank_splits = [
                host[
                    :,
                    rank * self.local_vocab_size
                    + split_start : rank * self.local_vocab_size
                    + split_start
                    + LM_HEAD_COLUMNS_PER_DEVICE,
                ]
                for rank in range(TENSOR_PARALLEL_SIZE)
            ]
            combined = torch.cat(rank_splits, dim=-1)
            memory_config = _dram_sharded_memory_config(
                self.mesh_device,
                self.hidden_size,
                LM_HEAD_COLUMNS_PER_DEVICE,
            )
            weights.append(
                LazyWeight(
                    source=combined,
                    dtype=ttnn.bfloat4_b,
                    device=self.mesh_device,
                    mesh_mapper_config=ttnn.MeshMapperConfig(
                        placements=[ttnn.PlacementShard(-1)],
                        mesh_shape_override=ttnn.MeshShape([TENSOR_PARALLEL_SIZE]),
                    ),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=memory_config,
                    cache_dir_weight_name=self._cache_name(f"lm_head_vocab_split_{split_start}_bfp4"),
                )
            )
            memory_configs.append(memory_config)
            program_configs.append(
                _dram_matmul_program_config(
                    32,
                    self.hidden_size,
                    LM_HEAD_COLUMNS_PER_DEVICE,
                    grid,
                    in0_block_w=3,
                )
            )
        return LMHead1D.from_config(
            LMHead1DConfig(
                output_weights=weights,
                mesh_device=self.mesh_device,
                dim=self.hidden_size,
                max_batch_size=32,
                program_configs=program_configs,
                compute_kernel_config=_compute_config(self.mesh_device, ttnn.MathFidelity.LoFi),
                lm_head_dtype=ttnn.bfloat8_b,
                output_memcfg=ttnn.L1_MEMORY_CONFIG,
                input_memcfg=self.layers[-1].residual_memory_config,
                weights_memcfgs=memory_configs,
            )
        )

    def allocate_kv_cache(
        self,
        *,
        max_cache_len: int | None = None,
        paged: bool = True,
        num_blocks: int | None = None,
    ) -> list[tuple[ttnn.Tensor, ttnn.Tensor]]:
        cache_len = self.max_cache_len if max_cache_len is None else int(max_cache_len)
        caches = [
            layer.allocate_kv_cache(max_cache_len=cache_len, paged=paged, num_blocks=num_blocks)
            for layer in self.layers
        ]
        return caches

    def ensure_internal_kv_cache(self) -> list[tuple[ttnn.Tensor, ttnn.Tensor]]:
        if self.kv_cache is None:
            self.kv_cache = self.allocate_kv_cache(paged=True)
        return self.kv_cache

    def reset_kv_cache(self, kv_cache: Sequence[Sequence[ttnn.Tensor]] | None = None) -> None:
        selected = self.ensure_internal_kv_cache() if kv_cache is None else kv_cache
        for key_cache, value_cache in selected:
            ttnn.fill(key_cache, 0.0, memory_config=key_cache.memory_config(), output_tensor=key_cache)
            ttnn.fill(value_cache, 0.0, memory_config=value_cache.memory_config(), output_tensor=value_cache)

    def reset_decode_ccl_state(self) -> None:
        """Restore the two shared persistent async collectives to capture epoch zero."""
        resources = self.layers[0].persistent_ccl_resources
        if resources is None:
            return
        attention_buffer, mlp_buffer, attention_semaphore, mlp_semaphore = resources
        for buffer in (attention_buffer, mlp_buffer):
            ttnn.fill(buffer, 0.0, memory_config=buffer.memory_config(), output_tensor=buffer)
        ttnn.reset_global_semaphore_value(attention_semaphore, 0)
        ttnn.reset_global_semaphore_value(mlp_semaphore, 0)

    def _all_gather_hidden(self, hidden):
        gathered = ttnn.experimental.all_gather_async(
            hidden,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(TENSOR_PARALLEL_AXIS),
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=TENSOR_PARALLEL_AXIS,
            topology=ttnn.Topology.Ring,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        return gathered

    def embed_prefill(self, tokens: ttnn.Tensor):
        hidden = self.embedding.forward(tokens)
        hidden = ttnn.unsqueeze_to_4D(hidden)
        gathered = self._all_gather_hidden(hidden)
        ttnn.deallocate(hidden, True)
        return gathered

    def embed_decode(self, tokens: ttnn.Tensor):
        hidden = self.embedding.forward(tokens)
        hidden = ttnn.unsqueeze_to_4D(hidden)
        gathered = self._all_gather_hidden(hidden)
        ttnn.deallocate(hidden, True)
        if int(gathered.shape[-2]) != self.max_batch_size:
            logical = ttnn.slice(
                gathered,
                [0, 0, 0, 0],
                [1, 1, self.max_batch_size, self.hidden_size],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(gathered, True)
            gathered = logical
        return ttnn.reshape(gathered, (1, self.max_batch_size, 1, self.hidden_size))

    @staticmethod
    def _normalise_kv_cache(kv_cache, expected_layers: int):
        if kv_cache is None:
            return None
        if len(kv_cache) != expected_layers:
            raise ValueError(f"kv_cache has {len(kv_cache)} layers, expected {expected_layers}")
        return kv_cache

    def prefill_hidden(
        self,
        token_tensor: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor | None,
        kv_cache: Sequence[Sequence[ttnn.Tensor]] | None,
        prompt_lens: Sequence[int] | None = None,
        chunk_start_idx: int = 0,
        chunk_page_table: ttnn.Tensor | None = None,
    ):
        selected_cache = self.ensure_internal_kv_cache() if kv_cache is None else kv_cache
        self._normalise_kv_cache(selected_cache, self.num_layers)
        hidden = self.embed_prefill(token_tensor)
        for layer_idx, layer in enumerate(self.layers):
            key_cache, value_cache = selected_cache[layer_idx]
            hidden = layer.prefill_forward(
                hidden,
                key_cache=key_cache,
                value_cache=value_cache,
                page_table=page_table,
                prompt_lens=prompt_lens,
                chunk_start_idx=chunk_start_idx,
                chunk_page_table=chunk_page_table,
            )
        return hidden

    def decode_hidden(
        self,
        token_tensor: ttnn.Tensor,
        *,
        cache_position: ttnn.Tensor,
        rotary_position: ttnn.Tensor | None,
        page_table: ttnn.Tensor | None,
        kv_cache: Sequence[Sequence[ttnn.Tensor]] | None,
    ):
        selected_cache = self.ensure_internal_kv_cache() if kv_cache is None else kv_cache
        self._normalise_kv_cache(selected_cache, self.num_layers)
        hidden = self.embed_decode(token_tensor)
        for layer_idx, layer in enumerate(self.layers):
            key_cache, value_cache = selected_cache[layer_idx]
            hidden = layer.decode_forward(
                hidden,
                key_cache=key_cache,
                value_cache=value_cache,
                cache_position=cache_position,
                # position_index is retained only as a bounds assertion in the
                # inherited decoder; all RoPE/cache/SDPA positions are read from
                # the persistent device tensor above.
                position_index=0,
                page_table=page_table,
                rotary_position=rotary_position,
            )
        return hidden

    def _decode_terminal(self, hidden):
        residual = ttnn.reshape(hidden, (1, 1, self.max_batch_size, self.hidden_size))
        ttnn.deallocate(hidden, False)
        normed = self.layers[-1]._decode_norm(residual, self.final_norm_weight)
        if int(normed.shape[-2]) < 32:
            normed_dram = ttnn.to_memory_config(normed, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(normed, True)
            padded = ttnn.pad(
                normed_dram,
                [(0, 0), (0, 0), (0, 32 - int(normed_dram.shape[-2])), (0, 0)],
                value=0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(normed_dram, True)
            normed = ttnn.to_memory_config(padded, self.layers[-1].residual_memory_config)
            ttnn.deallocate(padded, True)
        return self.lm_head.forward(normed)

    def decode_forward_from_ttnn_inputs(
        self,
        token_tensor: ttnn.Tensor,
        cache_position: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor | None,
        kv_cache: Sequence[Sequence[ttnn.Tensor]] | None = None,
        rotary_position: ttnn.Tensor | None = None,
        advance_position: bool = True,
    ):
        """Device-only decode body used for warm-up, trace capture, and replay."""
        hidden = self.decode_hidden(
            token_tensor,
            cache_position=cache_position,
            rotary_position=rotary_position,
            page_table=page_table,
            kv_cache=kv_cache,
        )
        logits = self._decode_terminal(hidden)
        if advance_position:
            ttnn.plus_one(cache_position, skip_negative_entries=True)
            if rotary_position is not None:
                ttnn.plus_one(rotary_position)
        return logits

    def _prefill_norm(self, hidden):
        residual = ttnn.reshape(
            hidden,
            (1, 1, int(hidden.shape[1]) * int(hidden.shape[2]), self.hidden_size),
        )
        ttnn.deallocate(hidden, False)
        return ttnn.rms_norm(
            residual,
            epsilon=float(self.hf_config.rms_norm_eps),
            weight=self.final_norm_weight,
            compute_kernel_config=self.layers[-1].norm_compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def select_prefill_token_hidden(self, hidden, user_idx: int, token_idx: int):
        """Retain one logical prefill row without retaining the whole chunk."""
        batch, seq_len = int(hidden.shape[1]), int(hidden.shape[2])
        if user_idx < 0 or user_idx >= batch or token_idx < 0 or token_idx >= seq_len:
            raise ValueError(f"prefill selection ({user_idx},{token_idx}) is outside {tuple(hidden.shape)}")
        return ttnn.slice(
            hidden,
            [0, user_idx, token_idx, 0],
            [1, user_idx + 1, token_idx + 1, self.hidden_size],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def prefill_selected_hidden_logits(self, selected_rows, *, fixed_sampling_rows: bool = False):
        """Apply final norm/LM head to ordered selected prefill rows."""
        if not selected_rows or len(selected_rows) > self.max_batch_size:
            raise ValueError(f"expected between 1 and {self.max_batch_size} selected rows")
        selected = selected_rows[0] if len(selected_rows) == 1 else ttnn.concat(selected_rows, dim=2)
        normed = self._prefill_norm(selected)
        rows = int(normed.shape[-2])
        if fixed_sampling_rows and rows < 32:
            padded = ttnn.pad(
                normed,
                [(0, 0), (0, 0), (0, 32 - rows), (0, 0)],
                value=0.0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(normed, True)
            normed = padded
        sharded = ttnn.to_memory_config(normed, self.layers[-1].residual_memory_config)
        ttnn.deallocate(normed, True)
        return self.lm_head.forward(sharded)

    def sample_greedy_split(self, logits, *, tt_out_tok=None):
        """Semantically exact device greedy over TP vocab shards."""
        return self.sampler.greedy_decode_forward(logits, tt_out_tok=tt_out_tok)

    def sample_stochastic_split(self, logits, *, k, p, temp, seeds=None, tt_out_tok=None):
        return self.sampler.decode_forward(
            logits,
            k=k,
            p=p,
            temp=temp,
            seeds=seeds,
            tt_out_tok=tt_out_tok,
            enable_log_probs=False,
        )[0]

    def sample_force_argmax(self, logits, *, tt_out_tok=None):
        """Comparison-only common path which gathers full logits before argmax."""
        return self.sampler.decode_forward(logits, tt_out_tok=tt_out_tok, enable_log_probs=False)[0]

    def prefill_local_logits(
        self,
        token_tensor: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor | None,
        kv_cache: Sequence[Sequence[ttnn.Tensor]] | None = None,
        selected_flat_rows: Sequence[int] | None = None,
        prompt_lens: Sequence[int] | None = None,
        chunk_start_idx: int = 0,
        chunk_page_table: ttnn.Tensor | None = None,
    ) -> list[ttnn.Tensor]:
        """Return sampler-ready local-vocab logits in physical 32-row chunks."""
        hidden = self.prefill_hidden(
            token_tensor,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            chunk_start_idx=chunk_start_idx,
            chunk_page_table=chunk_page_table,
        )
        return self.prefill_hidden_local_logits(hidden, selected_flat_rows=selected_flat_rows)

    def prefill_hidden_local_logits(
        self,
        hidden,
        *,
        selected_flat_rows: Sequence[int] | None = None,
    ) -> list[ttnn.Tensor]:
        """Apply the terminal path to precomputed hidden rows in groups of 32."""
        normed = self._prefill_norm(hidden)
        total_rows = int(normed.shape[-2])
        row_groups: list[list[int]]
        if selected_flat_rows is None:
            row_groups = [list(range(start, min(start + 32, total_rows))) for start in range(0, total_rows, 32)]
        else:
            selected = [int(row) for row in selected_flat_rows]
            if any(row < 0 or row >= total_rows for row in selected):
                raise ValueError(f"selected prefill row is outside [0,{total_rows})")
            row_groups = [selected[start : start + 32] for start in range(0, len(selected), 32)]

        outputs: list[ttnn.Tensor] = []
        for rows in row_groups:
            pieces = [
                ttnn.slice(
                    normed,
                    [0, 0, row, 0],
                    [1, 1, row + 1, self.hidden_size],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for row in rows
            ]
            chunk = pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            if len(pieces) > 1:
                for piece in pieces:
                    ttnn.deallocate(piece, True)
            if len(rows) < 32:
                padded = ttnn.pad(
                    chunk,
                    [(0, 0), (0, 0), (0, 32 - len(rows)), (0, 0)],
                    value=0.0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                ttnn.deallocate(chunk, True)
                chunk = padded
            sharded = ttnn.to_memory_config(chunk, self.layers[-1].residual_memory_config)
            ttnn.deallocate(chunk, True)
            outputs.append(self.lm_head.forward(sharded))
        ttnn.deallocate(normed, True)
        return outputs

    def gather_logits_to_torch(self, local_logits: ttnn.Tensor, *, valid_rows: int | None = None) -> torch.Tensor:
        gathered = ttnn.all_gather(
            local_logits,
            dim=3,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=TENSOR_PARALLEL_AXIS,
            topology=ttnn.Topology.Ring,
        )
        first_rank = ttnn.get_device_tensors(gathered)[0]
        host = ttnn.to_torch(first_rank).float()
        ttnn.deallocate(gathered, True)
        host = host[..., : self.vocab_size]
        if valid_rows is not None:
            host = host[..., : int(valid_rows), :]
        return host

    def teardown(self) -> None:
        if self.kv_cache is not None:
            for pair in self.kv_cache:
                for tensor in pair:
                    tensor.deallocate(True)
            self.kv_cache = None


__all__ = [
    "DEFAULT_MAX_BATCH_SIZE",
    "DEFAULT_TRACE_REGION_SIZE",
    "Falcon3Model",
    "HF_MODEL_ID",
    "HF_REVISION",
    "MAX_CONTEXT",
]
