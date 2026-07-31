# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full TTNN autoregressive model for ``microsoft/Phi-3.5-mini-instruct``.

The decoder stack is intentionally built from the optimized 1x8 multichip
decoder. The full-model boundary keeps the decoder's layer-to-layer contract:
replicated BF16 residuals between layers, tensor-parallel work only inside a
layer, decode BF8 CCL payloads, LoFi decode math, and BF16/HiFi2 prefill.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import ttnn
from safetensors import safe_open
from transformers import AutoConfig

from models.autoports.microsoft_phi_3_5_mini_instruct.tt.multichip_decoder import (
    TARGET_MESH_SHAPE,
    TOPOLOGY,
    TP_AXIS,
    TP_FACTOR,
    MultichipDecoder,
    _core_grid_for_num_cores,
    _dram_matmul_config,
    _dram_shard_core_grid,
    _dram_sharded_weight_mem_config,
    _host_to_mesh,
    _math_fidelity_from_name,
    _validate_target_mesh,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.optimized_decoder import (
    DEFAULT_BLOCK_SIZE,
    HIDDEN_SIZE,
    MODEL_ID,
    TILE_SIZE,
    _prefill_matmul_config,
)
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.precision import (
    DEFAULT_SELECTED_CONFIG,
    Phi35MiniPrecisionPolicy,
    load_precision_policy,
)
from models.common.lightweightmodule import LightweightModule


DEFAULT_REVISION = "2fe192450127e6a83f7441aef6e3ca586c338b77"
DEFAULT_MAX_SEQ_LEN = int(os.getenv("PHI35_FULL_MODEL_MAX_SEQ_LEN", "512"))
DEFAULT_MAX_BATCH_SIZE = 1
SAMPLING_BATCH_SIZE = 32
LM_HEAD_DTYPE = ttnn.bfloat16
EMBEDDING_DTYPE = ttnn.bfloat16


@dataclass(frozen=True)
class Phi35MiniFullModelConfig:
    hf_model_id: str
    revision: str | None
    num_layers: int
    max_seq_len: int
    max_batch_size: int
    block_size: int
    vocab_size: int
    padded_vocab_size: int
    per_device_vocab_size: int

    @classmethod
    def from_hf_config(
        cls,
        hf_config,
        *,
        hf_model_id: str = MODEL_ID,
        revision: str | None = DEFAULT_REVISION,
        num_layers: int | None = None,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> "Phi35MiniFullModelConfig":
        if hf_config.hidden_size != HIDDEN_SIZE:
            raise ValueError(f"Phi-3.5 mini hidden_size must be {HIDDEN_SIZE}, got {hf_config.hidden_size}")
        if bool(getattr(hf_config, "tie_word_embeddings", False)):
            raise ValueError("Phi-3.5-mini-instruct is expected to use an untied lm_head")
        if max_batch_size != 1:
            raise ValueError("The optimized Phi-3.5 multichip decoder currently supports batch size 1")
        if max_seq_len % block_size != 0:
            max_seq_len = math.ceil(max_seq_len / block_size) * block_size

        vocab_size = int(hf_config.vocab_size)
        min_per_device_vocab = math.ceil(vocab_size / (TP_FACTOR * TILE_SIZE)) * TILE_SIZE
        # The split sampler's TopK op only selects its multicore implementation
        # when the per-device reduction width is a power of two and at least
        # 8192. Padding at the LM-head contract avoids a per-token pad op in the
        # captured decode path.
        per_device_vocab = max(8192, 1 << (min_per_device_vocab - 1).bit_length())
        return cls(
            hf_model_id=hf_model_id,
            revision=revision,
            num_layers=int(num_layers or hf_config.num_hidden_layers),
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            block_size=block_size,
            vocab_size=vocab_size,
            padded_vocab_size=per_device_vocab * TP_FACTOR,
            per_device_vocab_size=per_device_vocab,
        )


class Phi35MiniForCausalLMTT(LightweightModule):
    """Full autoregressive Phi-3.5-mini model on the optimized 1x8 decoder stack."""

    def __init__(
        self,
        *,
        hf_config,
        full_config: Phi35MiniFullModelConfig,
        mesh_device: ttnn.MeshDevice,
        layers: list[MultichipDecoder],
        embedding_weight: ttnn.Tensor,
        final_norm_weight: ttnn.Tensor,
        lm_head_weight_decode: ttnn.Tensor,
        lm_head_weight_prefill: ttnn.Tensor,
        logits_padding_mask: ttnn.Tensor,
        precision_policy: Phi35MiniPrecisionPolicy,
    ) -> None:
        super().__init__()
        _validate_target_mesh(mesh_device)
        self.hf_config = hf_config
        self.full_config = full_config
        self.precision_policy = precision_policy
        self.mesh_device = mesh_device
        self.layers = layers
        self.embedding_weight = embedding_weight
        self.final_norm_weight = final_norm_weight
        self.lm_head_weight_decode = lm_head_weight_decode
        self.lm_head_weight_prefill = lm_head_weight_prefill
        self.logits_padding_mask = logits_padding_mask
        self.embedding_dtype = precision_policy.embedding_dtype
        self.norm_dtype = precision_policy.norm_dtype
        self.lm_head_decode_dtype = precision_policy.weight_dtype("lm_head")
        self.lm_head_prefill_dtype = precision_policy.weight_dtype("lm_head", prefill=True)
        self.logits_dtype = precision_policy.logits_dtype
        self.decode_lm_head_num_cores = _dram_shard_core_grid(HIDDEN_SIZE).num_cores
        self.decode_lm_head_program_config = _dram_matmul_config(
            m=TILE_SIZE,
            k=HIDDEN_SIZE,
            n=full_config.per_device_vocab_size,
            num_cores=self.decode_lm_head_num_cores,
        )
        self.decode_lm_head_output_mem_config = _width_sharded_lm_head_mem_config(
            full_config.per_device_vocab_size, self.decode_lm_head_num_cores
        )
        self.prefill_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=_math_fidelity_from_name(precision_policy.compute_fidelity("lm_head_prefill", "hifi2")),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.decode_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=_math_fidelity_from_name(precision_policy.compute_fidelity("lm_head_decode", "lofi")),
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.final_norm_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=_math_fidelity_from_name(precision_policy.compute_fidelity("norm", "hifi4")),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def from_hf(
        cls,
        *,
        mesh_device: ttnn.MeshDevice,
        hf_model_id: str = MODEL_ID,
        revision: str | None = DEFAULT_REVISION,
        hf_snapshot: str | Path | None = None,
        num_layers: int | None = None,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        block_size: int = DEFAULT_BLOCK_SIZE,
        model_dir: str | Path | None = None,
        precision_config_path: str | Path | None = None,
    ) -> "Phi35MiniForCausalLMTT":
        _validate_target_mesh(mesh_device)
        precision = load_precision_policy(model_dir=model_dir, precision_config_path=precision_config_path)
        hf_config = AutoConfig.from_pretrained(hf_model_id, revision=revision, trust_remote_code=True)
        hf_config._attn_implementation = "eager"
        full_config = Phi35MiniFullModelConfig.from_hf_config(
            hf_config,
            hf_model_id=hf_model_id,
            revision=revision,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
            block_size=block_size,
        )
        snapshot_path = _resolve_hf_snapshot(hf_model_id, revision=revision, hf_snapshot=hf_snapshot)
        weights = _SafetensorIndex(snapshot_path)

        embedding_weight = _embedding_weight_to_device(
            weights.get("model.embed_tokens.weight"), mesh_device, dtype=precision.embedding_dtype
        )
        final_norm_weight = _norm_weight_to_device(weights.get("model.norm.weight"), mesh_device, dtype=precision.norm_dtype)
        lm_head_weight_decode = _lm_head_weight_to_device(
            weights.get("lm_head.weight"),
            mesh_device,
            padded_vocab_size=full_config.padded_vocab_size,
            memory_config=_dram_sharded_weight_mem_config(
                mesh_device, HIDDEN_SIZE, full_config.per_device_vocab_size
            ),
            dtype=precision.weight_dtype("lm_head"),
        )
        lm_head_weight_prefill = _lm_head_weight_to_device(
            weights.get("lm_head.weight"),
            mesh_device,
            padded_vocab_size=full_config.padded_vocab_size,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=precision.weight_dtype("lm_head", prefill=True),
        )
        logits_padding_mask = _logits_padding_mask_to_device(
            mesh_device,
            vocab_size=full_config.vocab_size,
            padded_vocab_size=full_config.padded_vocab_size,
            dtype=precision.logits_dtype,
        )

        layers = []
        for layer_idx in range(full_config.num_layers):
            layer_state = weights.layer_state_dict(layer_idx)
            layers.append(
                MultichipDecoder.from_state_dict(
                    layer_state,
                    hf_config=hf_config,
                    layer_idx=layer_idx,
                    mesh_device=mesh_device,
                    block_size=block_size,
                    max_position_embeddings=max(max_seq_len, block_size),
                    precision_policy=precision,
                )
            )

        return cls(
            hf_config=hf_config,
            full_config=full_config,
            mesh_device=mesh_device,
            layers=layers,
            embedding_weight=embedding_weight,
            final_norm_weight=final_norm_weight,
            lm_head_weight_decode=lm_head_weight_decode,
            lm_head_weight_prefill=lm_head_weight_prefill,
            logits_padding_mask=logits_padding_mask,
            precision_policy=precision,
        )

    def allocate_kv_cache(
        self,
        *,
        max_batch_size: int | None = None,
        max_seq_len: int | None = None,
    ) -> list[tuple[ttnn.Tensor, ttnn.Tensor]]:
        max_batch_size = int(max_batch_size or self.full_config.max_batch_size)
        max_seq_len = int(max_seq_len or self.full_config.max_seq_len)
        return [
            MultichipDecoder.allocate_paged_kv_cache(
                hf_config=self.hf_config,
                mesh_device=self.mesh_device,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                block_size=self.full_config.block_size,
                dtype=self.precision_policy.kv_cache_dtype,
            )
            for _ in self.layers
        ]

    def make_page_table(self, *, max_batch_size: int = 1, max_seq_len: int | None = None) -> ttnn.Tensor:
        max_seq_len = int(max_seq_len or self.full_config.max_seq_len)
        blocks_per_seq = math.ceil(max_seq_len / self.full_config.block_size)
        table = torch.arange(blocks_per_seq, dtype=torch.int32).reshape(1, blocks_per_seq)
        if max_batch_size != 1:
            table = table.repeat(max_batch_size, 1)
        return _host_to_mesh(
            table,
            self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def tokens_to_device(self, tokens: torch.Tensor) -> ttnn.Tensor:
        return _host_to_mesh(
            tokens.to(torch.uint32),
            self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    def embed_tokens(self, tokens: ttnn.Tensor, *, decode: bool = False) -> ttnn.Tensor:
        if decode:
            tokens = ttnn.slice(tokens, (0, 0, 0, 0), (1, 1, 1, 1))
        hidden_states = ttnn.embedding(
            tokens,
            self.embedding_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.embedding_dtype,
        )
        hidden_states = ttnn.unsqueeze_to_4D(hidden_states) if len(hidden_states.shape) == 3 else hidden_states
        hidden_states = ttnn.all_gather(
            hidden_states,
            dim=3,
            cluster_axis=TP_AXIS,
            num_links=1,
            topology=TOPOLOGY,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return hidden_states

    def prefill_forward_ttnn(
        self,
        tokens: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor,
        kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]],
        prompt_lens: list[int],
        return_all_logits: bool = False,
    ) -> ttnn.Tensor:
        if len(prompt_lens) != 1:
            raise ValueError("Phi-3.5 full model currently supports a single user")
        seq_len = int(tokens.shape[-1])
        if seq_len % self.full_config.block_size != 0:
            raise ValueError(f"prefill token length must be block-aligned, got {seq_len}")
        hidden_states = self.embed_tokens(tokens)
        for layer, layer_cache in zip(self.layers, kv_cache):
            hidden_states = layer.prefill_forward(
                hidden_states,
                page_table=page_table,
                kv_cache=layer_cache,
                start_pos=0,
                rope_sequence_length=max(prompt_lens[0], seq_len),
            )

        last_idx = int(prompt_lens[0]) - 1
        if not return_all_logits:
            hidden_states = ttnn.slice(hidden_states, (0, 0, last_idx, 0), (1, 1, last_idx + 1, HIDDEN_SIZE))

        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.hf_config.rms_norm_eps,
            weight=self.final_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.final_norm_compute_kernel_config,
        )
        return self._lm_head(hidden_states, decode=False)

    def decode_forward_from_ttnn_inputs(
        self,
        tokens: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        *,
        page_table: ttnn.Tensor,
        kv_cache: list[tuple[ttnn.Tensor, ttnn.Tensor]],
        rope_sequence_length: int,
        pad_for_sampling: bool = False,
        advance_position: bool = False,
    ) -> ttnn.Tensor:
        hidden_states = self.embed_tokens(tokens, decode=True)
        for layer, layer_cache in zip(self.layers, kv_cache):
            hidden_states = layer.decode_forward(
                hidden_states,
                current_pos=current_pos,
                page_table=page_table,
                kv_cache=layer_cache,
                rope_sequence_length=rope_sequence_length,
            )
        hidden_states = ttnn.rms_norm(
            hidden_states,
            epsilon=self.hf_config.rms_norm_eps,
            weight=self.final_norm_weight,
            memory_config=self.layers[-1].decode_hidden_mem_config,
            compute_kernel_config=self.final_norm_compute_kernel_config,
        )
        logits = self._lm_head(hidden_states, decode=True)
        if pad_for_sampling:
            logits = ttnn.pad(logits, padding=[(0, 0), (0, 0), (0, SAMPLING_BATCH_SIZE - 1), (0, 0)], value=0.0)
        if advance_position:
            ttnn.plus_one(current_pos, skip_negative_entries=True)
        return logits

    def logits_to_torch(self, logits: ttnn.Tensor) -> torch.Tensor:
        composed = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3))
        return composed[..., : self.full_config.vocab_size].contiguous()

    def _lm_head(self, hidden_states: ttnn.Tensor, *, decode: bool) -> ttnn.Tensor:
        seq_len = int(hidden_states.shape[-2])
        logits = ttnn.linear(
            hidden_states,
            self.lm_head_weight_decode if decode else self.lm_head_weight_prefill,
            dtype=self.logits_dtype,
            memory_config=self.decode_lm_head_output_mem_config if decode else ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.decode_lm_head_program_config
            if decode
            else _prefill_matmul_config(seq_len, HIDDEN_SIZE, self.full_config.per_device_vocab_size),
            compute_kernel_config=self.decode_compute_kernel_config if decode else self.prefill_compute_kernel_config,
        )
        logits = ttnn.add(logits, self.logits_padding_mask, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=self.logits_dtype)
        return logits


class _SafetensorIndex:
    def __init__(self, snapshot_path: Path) -> None:
        self.snapshot_path = Path(snapshot_path)
        index_path = self.snapshot_path / "model.safetensors.index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"missing HF safetensor index at {index_path}")
        index = json.loads(index_path.read_text())
        self.weight_map: dict[str, str] = dict(index["weight_map"])

    def get(self, key: str) -> torch.Tensor:
        shard = self.weight_map.get(key)
        if shard is None:
            raise KeyError(f"missing HF tensor {key!r}")
        with safe_open(self.snapshot_path / shard, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def layer_state_dict(self, layer_idx: int) -> Mapping[str, torch.Tensor]:
        prefix = f"model.layers.{layer_idx}."
        wanted = (
            "self_attn.qkv_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_up_proj.weight",
            "mlp.down_proj.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
        )
        return {name: self.get(prefix + name) for name in wanted}


def _resolve_hf_snapshot(
    hf_model_id: str,
    *,
    revision: str | None,
    hf_snapshot: str | Path | None,
) -> Path:
    if hf_snapshot is not None:
        path = Path(hf_snapshot).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"HF snapshot path does not exist: {path}")

    env_path = os.getenv("PHI35_HF_SNAPSHOT")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"PHI35_HF_SNAPSHOT does not exist: {path}")

    cache_root = Path.home() / ".cache/huggingface/hub" / "models--microsoft--Phi-3.5-mini-instruct" / "snapshots"
    if revision:
        candidate = cache_root / revision
        if candidate.exists():
            return candidate

    try:
        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                repo_id=hf_model_id,
                revision=revision,
                allow_patterns=["*.json", "*.py", "*.safetensors", "tokenizer*"],
            )
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not resolve local or downloaded HF snapshot for {hf_model_id}@{revision}. "
            "Set PHI35_HF_SNAPSHOT to a snapshot directory containing model.safetensors.index.json."
        ) from exc


def _embedding_weight_to_device(
    weight: torch.Tensor, mesh_device: ttnn.MeshDevice, *, dtype: ttnn.DataType = EMBEDDING_DTYPE
) -> ttnn.Tensor:
    return _host_to_mesh(
        weight.reshape(1, 1, weight.shape[0], weight.shape[1]).to(torch.bfloat16),
        mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )


def _norm_weight_to_device(
    weight: torch.Tensor, mesh_device: ttnn.MeshDevice, *, dtype: ttnn.DataType = ttnn.bfloat16
) -> ttnn.Tensor:
    return _host_to_mesh(
        weight.reshape(1, 1, 1, -1).to(torch.bfloat16),
        mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _lm_head_weight_to_device(
    weight: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    *,
    padded_vocab_size: int,
    memory_config: ttnn.MemoryConfig,
    dtype: ttnn.DataType = LM_HEAD_DTYPE,
) -> ttnn.Tensor:
    if weight.shape[0] > padded_vocab_size:
        raise ValueError(f"lm_head vocab {weight.shape[0]} exceeds padded vocab {padded_vocab_size}")
    if weight.shape[0] < padded_vocab_size:
        pad_rows = padded_vocab_size - weight.shape[0]
        weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_rows), value=0.0)
    weight_t = weight.T.contiguous()
    per_device_vocab = padded_vocab_size // TP_FACTOR
    return _host_to_mesh(
        weight_t.reshape(1, 1, HIDDEN_SIZE, padded_vocab_size).to(torch.bfloat16),
        mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )


def _logits_padding_mask_to_device(
    mesh_device: ttnn.MeshDevice,
    *,
    vocab_size: int,
    padded_vocab_size: int,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    mask = torch.zeros((1, 1, 1, padded_vocab_size), dtype=torch.bfloat16)
    if padded_vocab_size > vocab_size:
        mask[..., vocab_size:] = -10000.0
    return _host_to_mesh(
        mask,
        mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )


def _width_sharded_lm_head_mem_config(width: int, num_cores: int) -> ttnn.MemoryConfig:
    padded_width = math.ceil(width / (TILE_SIZE * num_cores)) * TILE_SIZE * num_cores
    return ttnn.create_sharded_memory_config(
        (TILE_SIZE, padded_width // num_cores),
        _core_grid_for_num_cores(num_cores),
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def full_model_strategy_summary() -> dict[str, object]:
    return {
        "model": MODEL_ID,
        "mesh_shape": TARGET_MESH_SHAPE,
        "decoder": "MultichipDecoder",
        "num_layers": 32,
        "residual_layout": "replicated BF16 between decoder layers",
        "embedding": "hidden-dim sharded lookup, all-gather once to replicated residual",
        "lm_head": "vocab-dim sharded logits, per-device power-of-two padded for multicore split sampling",
        "sampling_batch": SAMPLING_BATCH_SIZE,
        "force_argmax_default": False,
        "selected_precision_config": str(DEFAULT_SELECTED_CONFIG),
    }
