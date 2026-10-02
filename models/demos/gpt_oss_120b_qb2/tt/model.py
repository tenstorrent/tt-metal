# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Resident full-model assembly for the optimized GPT-OSS 120B autoport.

The terminal embedding, normalization, LM-head, RoPE, and TT-Transformer
interfaces intentionally reuse the maintained GPT-OSS implementation.  Every
decoder block, however, is the autoport's optimized :class:`MultichipDecoder`;
there is no single-chip, replicated-weight, or host-executed decoder fallback.
"""

from __future__ import annotations

import copy
import gc
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from loguru import logger
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

import ttnn
from models.demos.gpt_oss.config import MeshConfig, ModeConfig
from models.demos.gpt_oss.tt.attention.kv_cache import get_kv_memory_config
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.model import Model as _GPTOSSModel
from models.demos.gpt_oss.tt.model import compute_per_device_vocab, create_rope_setup
from models.demos.gpt_oss.utils.general_utils import get_cache_file_name, get_default_num_links
from models.demos.gpt_oss_120b_qb2.tt.decoder import DecodeRMSNorm
from models.demos.gpt_oss_120b_qb2.tt.multichip_decoder import DECODE_K_CHUNK_SIZE, MultichipDecoder, tensor_plan
from models.demos.gpt_oss_120b_qb2.tt.precision import (
    PrecisionConfig,
    dtype_name,
    load_precision_config,
    math_fidelity_name,
)

MODEL_ID = "openai/gpt-oss-120b"
MODEL_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_LAYERS = 36
HF_CONTEXT_LENGTH = 131072
PAGE_SIZE = 64
DEVICE_DRAM_BYTES = 32 * 1024**3
TRACE_ACTIVATION_RESERVE_BYTES = 2 * 1024**3
INTERLEAVED_LM_HEAD = "interleaved"
DRAM_SHARDED_LM_HEAD = "dram_sharded"
_BFP8_TILE_BYTES = 1088
_BFP4_TILE_BYTES = 576
_BF16_TILE_BYTES = 2048
_TILE_ELEMENTS = 32 * 32

# Measured physical TT tensor storage from the completed optimized multichip
# stage.  TP2 includes its duplicated DRAM-sharded O-projection tensor.
DECODER_WEIGHT_BYTES_PER_DEVICE_36_LAYERS = {
    1: 65_619_952_128,
    2: 32_964_795_648,
    4: 16_719_757_056,
}


class FullModelCapacityError(RuntimeError):
    """Raised instead of silently selecting a less capable runtime path."""


@dataclass(frozen=True)
class CapacityEvidence:
    tp: int
    num_layers: int
    max_batch_size: int
    max_context_length: int
    decoder_weight_bytes: int
    embedding_bytes: int
    final_norm_bytes: int
    lm_head_bytes: int
    rope_bytes: int
    page_table_bytes: int
    kv_cache_bytes: int
    kv_cache_dtype: str
    trace_activation_reserve_bytes: int
    total_bytes_per_device: int
    device_dram_bytes: int
    largest_context_for_batch: int
    fits: bool

    def to_dict(self):
        result = asdict(self)
        result["total_gib_per_device"] = self.total_bytes_per_device / 1024**3
        result["device_dram_gib"] = self.device_dram_bytes / 1024**3
        return result


# Decode trace buckets: vLLM pads a decode step to the smallest declared width
# that holds the active requests.  With batched expert decode the cost scales
# with the batch, so one intermediate bucket keeps small batches off the full
# serving-width graph while bounding trace-region use.
ENV_DECODE_BUCKETS = "GPT_OSS_120B_DECODE_BUCKETS"
INTERMEDIATE_DECODE_BUCKETS = tuple(
    sorted(int(v) for v in os.environ.get(ENV_DECODE_BUCKETS, "4,8").split(",") if v.strip())
)


def decode_trace_buckets(max_batch_size: int) -> tuple[int, ...]:
    """Return the sorted decode widths prepared for ``max_batch_size``."""

    max_batch_size = int(max_batch_size)
    widths = {1, max_batch_size}
    widths.update(bucket for bucket in INTERMEDIATE_DECODE_BUCKETS if bucket < max_batch_size)
    return tuple(sorted(widths))


def _bfp8_tensor_bytes(height: int, width: int) -> int:
    if height % 32 or width % 32:
        raise ValueError(f"BFP8 tensor dimensions must be tile aligned, got {(height, width)}")
    return height * width // _TILE_ELEMENTS * _BFP8_TILE_BYTES


def _dtype_tile_bytes(dtype) -> int:
    if dtype == ttnn.bfloat16:
        return _BF16_TILE_BYTES
    if dtype == ttnn.bfloat8_b:
        return _BFP8_TILE_BYTES
    if dtype == ttnn.bfloat4_b:
        return _BFP4_TILE_BYTES
    raise ValueError(f"unsupported KV-cache dtype for capacity accounting: {dtype!r}")


def _kv_cache_bytes(*, tp: int, num_layers: int, batch_size: int, context_length: int, dtype) -> int:
    physical_context = math.ceil(context_length / DECODE_K_CHUNK_SIZE) * DECODE_K_CHUNK_SIZE
    blocks = math.ceil(physical_context / PAGE_SIZE)
    local_kv_heads = 8 // tp
    elements = 2 * num_layers * batch_size * blocks * local_kv_heads * PAGE_SIZE * 64
    return math.ceil(elements / _TILE_ELEMENTS) * _dtype_tile_bytes(dtype)


def capacity_evidence(
    *,
    tp: int,
    max_batch_size: int = 1,
    max_context_length: int = HF_CONTEXT_LENGTH,
    num_layers: int = MODEL_LAYERS,
    reserve_bytes: int = TRACE_ACTIVATION_RESERVE_BYTES,
    kv_cache_dtype=ttnn.bfloat8_b,
) -> CapacityEvidence:
    """Return conservative physical TT storage for one resident TP rank."""

    if tp not in DECODER_WEIGHT_BYTES_PER_DEVICE_36_LAYERS:
        raise ValueError(f"capacity accounting supports TP=1/2/4, got TP={tp}")
    if not 1 <= num_layers <= MODEL_LAYERS:
        raise ValueError(f"num_layers must be within [1, {MODEL_LAYERS}], got {num_layers}")
    if not 1 <= max_batch_size <= 32:
        raise ValueError(f"max_batch_size must be within [1, 32], got {max_batch_size}")
    if not 1 <= max_context_length <= HF_CONTEXT_LENGTH:
        raise ValueError(f"max_context_length must be within [1, {HF_CONTEXT_LENGTH}], got {max_context_length}")

    decoder_weights = math.ceil(DECODER_WEIGHT_BYTES_PER_DEVICE_36_LAYERS[tp] * num_layers / MODEL_LAYERS)
    embedding = 201088 * 2880 * 2  # replicated BF16
    final_norm = 2880 * 2
    per_device_vocab = 262144 // tp
    lm_head = _bfp8_tensor_bytes(2880, per_device_vocab)
    rope = 2 * HF_CONTEXT_LENGTH * 64 * 2  # replicated BF16 cosine + sine
    fixed = decoder_weights + embedding + final_norm + lm_head + rope + reserve_bytes
    physical_context = math.ceil(max_context_length / DECODE_K_CHUNK_SIZE) * DECODE_K_CHUNK_SIZE
    page_table = max_batch_size * math.ceil(physical_context / PAGE_SIZE) * 4

    def resident_bytes(context_length: int) -> int:
        physical_context = math.ceil(context_length / DECODE_K_CHUNK_SIZE) * DECODE_K_CHUNK_SIZE
        return (
            fixed
            + max_batch_size * math.ceil(physical_context / PAGE_SIZE) * 4
            + _kv_cache_bytes(
                tp=tp,
                num_layers=num_layers,
                batch_size=max_batch_size,
                context_length=context_length,
                dtype=kv_cache_dtype,
            )
        )

    low, high = 0, HF_CONTEXT_LENGTH
    while low < high:
        midpoint = (low + high + 1) // 2
        if resident_bytes(midpoint) <= DEVICE_DRAM_BYTES:
            low = midpoint
        else:
            high = midpoint - 1
    largest_context = low
    kv_cache = _kv_cache_bytes(
        tp=tp,
        num_layers=num_layers,
        batch_size=max_batch_size,
        context_length=max_context_length,
        dtype=kv_cache_dtype,
    )
    total = fixed + page_table + kv_cache
    return CapacityEvidence(
        tp=tp,
        num_layers=num_layers,
        max_batch_size=max_batch_size,
        max_context_length=max_context_length,
        decoder_weight_bytes=decoder_weights,
        embedding_bytes=embedding,
        final_norm_bytes=final_norm,
        lm_head_bytes=lm_head,
        rope_bytes=rope,
        page_table_bytes=page_table,
        kv_cache_bytes=kv_cache,
        kv_cache_dtype=dtype_name(kv_cache_dtype),
        trace_activation_reserve_bytes=reserve_bytes,
        total_bytes_per_device=total,
        device_dram_bytes=DEVICE_DRAM_BYTES,
        largest_context_for_batch=largest_context,
        fits=total <= DEVICE_DRAM_BYTES,
    )


def require_resident_capacity(
    *,
    tp: int,
    max_batch_size: int,
    max_context_length: int,
    num_layers: int,
    allow_reduced_model: bool = False,
    kv_cache_dtype=ttnn.bfloat8_b,
) -> CapacityEvidence:
    evidence = capacity_evidence(
        tp=tp,
        max_batch_size=max_batch_size,
        max_context_length=max_context_length,
        num_layers=num_layers,
        kv_cache_dtype=kv_cache_dtype,
    )
    if num_layers != MODEL_LAYERS and not allow_reduced_model:
        raise FullModelCapacityError(
            f"Production full-model construction requires {MODEL_LAYERS} layers; got {num_layers}. "
            "Reduced stacks are accepted only by explicit hardware probes."
        )
    if not evidence.fits:
        raise FullModelCapacityError(
            "Resident optimized GPT-OSS 120B does not fit this target without a forbidden fallback: "
            f"TP={tp}, batch={max_batch_size}, context={max_context_length} requires "
            f"{evidence.total_bytes_per_device / 1024**3:.3f} GiB/device including the measured "
            f"decoder policy and {TRACE_ACTIVATION_RESERVE_BYTES / 1024**3:.1f} GiB trace/activation "
            f"reserve, but P150 provides {DEVICE_DRAM_BYTES / 1024**3:.0f} GiB/device. "
            f"Largest accounted context for this batch is {evidence.largest_context_for_batch}."
        )
    return evidence


class StreamingCheckpoint:
    """Load one dense decoder layer at a time from the public MXFP4 checkpoint."""

    _INDEX = "model.safetensors.index.json"

    def __init__(self, snapshot_path: str | Path):
        self.snapshot_path = Path(snapshot_path).expanduser().resolve()
        index_path = self.snapshot_path / self._INDEX
        if not index_path.is_file():
            raise FileNotFoundError(f"GPT-OSS checkpoint index is missing: {index_path}")
        with index_path.open(encoding="utf-8") as index_file:
            self.weight_map = json.load(index_file)["weight_map"]

    def _read(self, checkpoint_key: str) -> torch.Tensor:
        try:
            shard = self.weight_map[checkpoint_key]
        except KeyError as error:
            raise KeyError(f"GPT-OSS checkpoint tensor is missing: {checkpoint_key}") from error
        shard_path = self.snapshot_path / shard
        if not shard_path.is_file():
            raise FileNotFoundError(f"GPT-OSS checkpoint shard is missing: {shard_path}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(checkpoint_key)

    def terminal_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: self._read(key)
            for key in (
                "model.embed_tokens.weight",
                "model.norm.weight",
                "lm_head.weight",
            )
        }

    def layer_state_dict(self, layer_idx: int, *, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
        prefix = f"model.layers.{layer_idx}."
        raw = {key[len(prefix) :]: self._read(key) for key in self.weight_map if key.startswith(prefix)}
        for projection in ("gate_up_proj", "down_proj"):
            packed = f"mlp.experts.{projection}"
            blocks = raw.pop(f"{packed}_blocks")
            scales = raw.pop(f"{packed}_scales")
            raw[packed] = convert_moe_packed_tensors(blocks, scales, dtype=dtype)
        return {
            key: (tensor.to(dtype=dtype) if tensor.is_floating_point() and tensor.dtype != dtype else tensor)
            for key, tensor in raw.items()
        }


class FullModelArgs:
    """Small tt-transformers argument surface with no alternate model policy."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        generation_config,
        tokenizer,
        snapshot_path: Path,
        tensor_cache_path: Path,
        max_batch_size: int,
        max_context_length: int,
        num_layers: int,
        precision_config: PrecisionConfig,
        salt_duplicate_seeds: bool = True,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.generation_config = generation_config
        self.tokenizer = tokenizer
        self.processor = None
        self.model_path = str(snapshot_path)
        self.weights_path = str(snapshot_path)
        self.tensor_cache_path = Path(tensor_cache_path)
        self.model_name = "gpt-oss-120b"
        self.vocab_size = int(hf_config.vocab_size)
        sampling_shards = int(mesh_device.shape[1])
        self.padded_vocab_size = compute_per_device_vocab(self.vocab_size, sampling_shards) * sampling_shards
        self.n_layers = int(num_layers)
        self.dim = int(hf_config.hidden_size)
        self.head_dim = int(hf_config.head_dim)
        self.max_batch_size = int(max_batch_size)
        self.max_local_batch_size = int(max_batch_size)
        self.max_seq_len = int(max_context_length)
        self.max_context_len = int(max_context_length)
        self.decode_k_chunk_size = DECODE_K_CHUNK_SIZE
        self.physical_kv_context_len = (
            math.ceil(max_context_length / self.decode_k_chunk_size) * self.decode_k_chunk_size
        )
        self.max_prefill_chunk_size = self.physical_kv_context_len
        # Batched prefill: new requests of one scheduler step that share a padded
        # length run as one forward (tokens concatenated along the sequence axis;
        # attention fills the KV cache per user, the MoE sees one longer row
        # set).  Compact rows keep the device batch at the request count instead
        # of the highest physical slot.  GPT_OSS_120B_BATCHED_PREFILL=0 disables it.
        self.disable_batched_prefill = os.environ.get("GPT_OSS_120B_BATCHED_PREFILL", "1") != "1"
        self.batched_prefill_compact_rows = True
        # Batch only prompts up to this padded length: above it a single prefill
        # is already compute bound (4k: 0.98 s) and each new (batch, length)
        # pair would compile its own programs.
        self.batched_prefill_max_tokens_per_user = int(os.environ.get("GPT_OSS_120B_BATCHED_PREFILL_MAX_LEN", "2048"))
        self.capped_warmup_seq_len = min(128, max_context_length)
        self.trace_prefill_supported_seq_lens = [128] if max_context_length >= 128 else []
        self.cluster_shape = tuple(int(v) for v in mesh_device.shape)
        self.num_devices = mesh_device.get_num_devices()
        self.precision_config_id = precision_config.config_id
        self.precision_config_source = (
            str(precision_config.source_path) if precision_config.source_path is not None else "builtin"
        )
        # TTSampling currently has a single numerically supported accumulator
        # policy.  Carry the selected contract into its construction path so
        # the full model can validate the real device buffers below.
        self.sampling_accumulator_dtype = precision_config.terminal_dtypes()["sampling_accumulator"]
        full_logits_gather = precision_config.terminal_dtypes()["full_logits_gather"]
        self.sampling_all_gather_axis = 1
        self.sampling_dp = 1
        self.salt_duplicate_seeds = bool(salt_duplicate_seeds)
        self.use_topk_logprobs = True
        self.is_galaxy = False
        # The selected non-materialized full-logit-gather policy explicitly
        # disables TTSampling's full-vocabulary force-argmax path.  The regular
        # sampler consumes the sharded LM-head output and gathers only top-k
        # values/indices.
        self.model_config = {
            "SAMPLING_AG_CONFIG": {
                "allow_force_argmax": full_logits_gather["mode"] != "not_materialized",
                "num_links": 1,
                "chunks_per_sync": 10,
                "topology": ttnn.Topology.Linear,
            }
        }

    @property
    def base_model_name(self):
        return self.model_name

    def is_llama_vision(self):
        return False

    def can_enable_trace(self, prefill_seq_len, num_cached_tokens=0):
        return prefill_seq_len in self.trace_prefill_supported_seq_lens and num_cached_tokens == 0

    def get_warmup_prefill_supported_seq_lens(self):
        return list(self.trace_prefill_supported_seq_lens)

    def encode_prompt(self, prompt_text, instruct=False, system_prompt_text=None):
        if instruct:
            raise ValueError("GPT-OSS uses its tokenizer chat template; instruct=True is not a separate mode")
        if isinstance(prompt_text, str):
            messages = []
            if system_prompt_text:
                messages.append({"role": "system", "content": system_prompt_text})
            messages.append({"role": "user", "content": prompt_text})
        else:
            messages = prompt_text
        encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if isinstance(encoded, dict) or hasattr(encoded, "input_ids"):
            encoded = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.reshape(-1).tolist()
        while isinstance(encoded, list) and len(encoded) == 1 and isinstance(encoded[0], list):
            encoded = encoded[0]
        return [int(token) for token in encoded]


class _LayerAdapter:
    """Match the shared GPT-OSS stack call surface without changing the block."""

    def __init__(self, decoder: MultichipDecoder):
        self.decoder = decoder
        self.self_attn = decoder.self_attn

    @property
    def kv_cache(self):
        return self.decoder.kv_cache

    def __call__(
        self,
        hidden_states,
        *,
        position_embeddings,
        position_idx,
        page_table,
        kv_cache,
        is_decode,
        user_id,
        batch_size,
        fill_seq_lens=None,
        chunk_start_idx=None,
        ring_tail_block=None,
        fill_start_idx=None,
    ):
        if is_decode:
            decode_batch_size = int(hidden_states.shape[-2])
            if position_idx.shape[-1] < decode_batch_size:
                raise ValueError(
                    "decode current_position must cover every fixed slot, including inactive rows; "
                    f"got {position_idx.shape[-1]} positions for batch {decode_batch_size}"
                )
            return self.decoder.decode_forward(
                hidden_states,
                position_embeddings=position_embeddings,
                current_position=position_idx,
                page_table=page_table,
                kv_cache=kv_cache,
                batch_size=decode_batch_size,
            )
        return self.decoder.prefill_forward(
            hidden_states,
            position_embeddings=position_embeddings,
            page_table=page_table,
            kv_cache=kv_cache,
            user_id=user_id,
            batch_size=batch_size,
            fill_seq_lens=fill_seq_lens,
            chunk_start_idx=chunk_start_idx,
            ring_tail_block=ring_tail_block,
            fill_start_idx=fill_start_idx,
        )


class _DramShardedLMHead:
    """Opt-in BFP8/HiFi2 terminal candidate split over physical DRAM banks."""

    def __init__(
        self,
        *,
        mesh_device,
        mesh_config,
        torch_weight: torch.Tensor,
        vocab_size: int,
        hidden_size: int,
        input_memory_config,
        tensor_cache_path: Path,
        split_size: int = 8192,
    ):
        tp = int(mesh_device.shape[1])
        local_vocab_size = 1 << math.ceil(math.log2(math.ceil(vocab_size / tp)))
        if local_vocab_size % split_size:
            raise ValueError(f"local padded vocabulary {local_vocab_size} must divide split size {split_size}")
        dram_grid_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
                )
            }
        )
        dram_banks = dram_grid.num_cores()
        if split_size % (dram_banks * ttnn.TILE_SIZE):
            raise ValueError(
                "LM-head split must divide evenly over tile-aligned DRAM banks: "
                f"split={split_size}, banks={dram_banks}"
            )
        input_shard = input_memory_config.shard_spec.shape
        if hidden_size % input_shard[1]:
            raise ValueError(f"hidden size {hidden_size} is incompatible with input shard {input_shard}")

        self.input_memory_config = input_memory_config
        self.output_memory_config = ttnn.DRAM_MEMORY_CONFIG
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=input_shard[1] // ttnn.TILE_SIZE,
            per_core_M=1,
            per_core_N=split_size // dram_banks // ttnn.TILE_SIZE,
            fused_activation=None,
        )
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.weights = []
        cache_root = Path(tensor_cache_path)
        for split_index, offset in enumerate(range(0, local_vocab_size, split_size)):
            rank_splits = []
            for rank in range(tp):
                global_start = rank * local_vocab_size + offset
                global_end = global_start + split_size
                rank_weight = torch.zeros(hidden_size, split_size, dtype=torch_weight.dtype)
                valid_end = min(global_end, vocab_size)
                if global_start < valid_end:
                    valid_width = valid_end - global_start
                    rank_weight[:, :valid_width] = torch_weight[global_start:valid_end].transpose(0, 1)
                rank_splits.append(rank_weight)
            combined_weight = torch.cat(rank_splits, dim=-1)
            weight_memory_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(
                    dram_grid,
                    (hidden_size, split_size // dram_banks),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            self.weights.append(
                ttnn.as_tensor(
                    combined_weight,
                    device=mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat8_b,
                    cache_file_name=get_cache_file_name(
                        cache_root,
                        f"split_{split_index}_k{hidden_size}_n{split_size}_banks{dram_banks}",
                    ),
                    memory_config=weight_memory_config,
                    mesh_mapper=mesh_config.column_parallel(mesh_device),
                )
            )
            del combined_weight, rank_splits
            gc.collect()

        self.manifest = {
            "dtype": "BFP8_B",
            "compute_fidelity": "HiFi2",
            "local_padded_vocab": local_vocab_size,
            "split_size": split_size,
            "num_splits": len(self.weights),
            "dram_banks": dram_banks,
            "input_shard_shape": list(input_shard),
            "output_memory": "DRAM interleaved",
        }

    def __call__(self, hidden_states):
        owns_input = hidden_states.memory_config() != self.input_memory_config
        sharded_input = ttnn.to_memory_config(hidden_states, self.input_memory_config) if owns_input else hidden_states
        outputs = []
        for weight in self.weights:
            sharded_output = ttnn.linear(
                sharded_input,
                weight,
                compute_kernel_config=self.compute_kernel_config,
                program_config=self.program_config,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            outputs.append(ttnn.sharded_to_interleaved(sharded_output, memory_config=self.output_memory_config))
            sharded_output.deallocate(True)
        logits = ttnn.concat(outputs, dim=-1, memory_config=self.output_memory_config)
        for output in outputs:
            output.deallocate(True)
        if owns_input:
            sharded_input.deallocate(True)
        return logits


class Model(_GPTOSSModel):
    """Full autoregressive model with 36 optimized TP decoder layers."""

    optimization_manifest = MultichipDecoder.optimization_manifest + (
        "resident_stream_loaded_36_layer_stack",
        "replicated_bf16_embedding",
        "decode_sharded_final_rmsnorm",
        "tp_column_sharded_bfp8_pow2_lm_head",
        "canonical_sampling_generator_split_sampling",
    )

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        terminal_state_dict,
        layer_loader,
        args: FullModelArgs,
        tensor_cache_path: str | Path,
        max_batch_size: int,
        max_context_length: int,
        num_layers: int,
        precision_config: PrecisionConfig,
        lm_head_policy: str = INTERLEAVED_LM_HEAD,
        create_kv_cache: bool = True,
    ):
        tp = int(mesh_device.shape[1])
        tensor_plan(mesh_device.shape, hf_config)
        base_policy = precision_config.decoder_policy_for_layer(0)
        terminal_dtypes = precision_config.terminal_dtypes()
        mesh_config = MeshConfig(
            mesh_device.shape,
            decode=ModeConfig(tp=tp, ep=1, sp=1),
            prefill=ModeConfig(tp=tp, ep=1, sp=1),
        )
        ccl_manager = CCLManager(
            mesh_device,
            num_links=get_default_num_links(mesh_device),
            topology=base_policy.topology,
        )
        cache_root = Path(tensor_cache_path)
        cache_root.mkdir(parents=True, exist_ok=True)

        # Reuse only the maintained terminal/runtime portion of the demo model.
        # A zero-layer config prevents construction of its decoder family.
        # The base constructor also creates SamplingGenerator and consults
        # ``self.args`` for the target's seed policy, so install it first.
        self.args = args
        terminal_config = copy.deepcopy(hf_config)
        terminal_config.num_hidden_layers = 0
        terminal_config.layer_types = []
        super().__init__(
            mesh_device=mesh_device,
            hf_config=terminal_config,
            state_dict=terminal_state_dict,
            ccl_manager=ccl_manager,
            dtype=ttnn.bfloat8_b,
            tensor_cache_path=str(cache_root / "terminal"),
            paged_attention_config=None,
            mesh_config=mesh_config,
            create_kv_cache=False,
            max_local_batch_size=max_batch_size,
            users_row_sharded=False,
            use_throughput_experts=False,
        )
        self.embedding_dtype = self.embedding_weight.dtype
        self.lm_head_weight_dtype = self.lm_head_weight.dtype
        self.lm_head_output_dtype = terminal_dtypes["lm_head_output"]
        self.hf_config = hf_config
        self.vocab_size = int(hf_config.vocab_size)
        self.n_layers = int(num_layers)
        self.dtype = ttnn.bfloat8_b
        self.max_context_length = int(max_context_length)
        self.page_size = PAGE_SIZE
        self.precision_config = precision_config
        self.policy = base_policy
        self.precision_config_id = precision_config.config_id
        self.lm_head_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=terminal_dtypes["lm_head_math_fidelity"],
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.sampling_accumulator_dtype = terminal_dtypes["sampling_accumulator"]
        self.full_logits_gather_policy = terminal_dtypes["full_logits_gather"]
        self.topk_values_gather_dtype = terminal_dtypes["topk_values_gather_dtype"]
        self.capacity = capacity_evidence(
            tp=tp,
            # A vLLM build allocates one scheduler-owned shared cache after
            # model construction rather than a full-context cache per slot.
            # Keep the conservative batch-1 full-context resident check here;
            # the adapter records and validates the actual serving allocation.
            max_batch_size=max_batch_size if create_kv_cache else 1,
            max_context_length=max_context_length,
            num_layers=num_layers,
            kv_cache_dtype=base_policy.kv_cache_dtype,
        )
        self.kv_cache_owner = "model" if create_kv_cache else "vllm_pending"
        if lm_head_policy not in {INTERLEAVED_LM_HEAD, DRAM_SHARDED_LM_HEAD}:
            raise ValueError(f"unknown LM-head policy {lm_head_policy!r}")
        self.lm_head_policy = lm_head_policy
        self._terminal_uses_single_tile = False
        self.dram_sharded_lm_head = None

        # The final norm consumes the last decoder's L1 replicated residual.
        self.norm.tt_weight.deallocate(True)
        self.norm = DecodeRMSNorm(
            mesh_device,
            hf_config,
            {"weight": terminal_state_dict["model.norm.weight"]},
            tensor_cache_path=str(cache_root / "terminal" / "final_norm_decode_sharded"),
            mesh_config=mesh_config,
            weight_dtype=terminal_dtypes["normalization"],
            enable_decode_sharding=DecodeRMSNorm.sharding_enabled_for(max_batch_size),
        )
        if lm_head_policy == DRAM_SHARDED_LM_HEAD:
            self.dram_sharded_lm_head = _DramShardedLMHead(
                mesh_device=mesh_device,
                mesh_config=mesh_config,
                torch_weight=terminal_state_dict["lm_head.weight"],
                vocab_size=self.vocab_size,
                hidden_size=int(hf_config.hidden_size),
                input_memory_config=self.norm.decode_memory_config,
                tensor_cache_path=cache_root / "terminal" / "lm_head_dram_sharded",
            )

        self.layers = []
        for layer_idx in range(num_layers):
            logger.info(f"Loading optimized GPT-OSS 120B layer {layer_idx + 1}/{num_layers}")
            layer_state = layer_loader(layer_idx)
            layer_policy = precision_config.decoder_policy_for_layer(layer_idx)
            decoder = MultichipDecoder.from_state_dict(
                layer_state,
                hf_config=hf_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                max_batch_size=max_batch_size,
                max_context_length=max_context_length,
                page_size=PAGE_SIZE,
                tensor_cache_path=str(cache_root / f"layer_{layer_idx:02d}"),
                calibrated_checkpoint_revision=MODEL_REVISION,
                policy=layer_policy,
                create_kv_cache=create_kv_cache,
            )
            self.layers.append(_LayerAdapter(decoder))
            del layer_state
            gc.collect()
        self.kv_cache = [layer.kv_cache for layer in self.layers]
        self._decode_rope_setups = {max_batch_size: self.rope_setup}
        self._decode_layer_transformation_mats = {
            max_batch_size: [layer.self_attn.transformation_mats["decode"] for layer in self.layers]
        }
        self._decode_layer_kv_memory_configs = {max_batch_size: [layer.self_attn.kv_mem_cfg for layer in self.layers]}
        # RotarySetup fixes its decode sharding and transformation matrix to
        # the construction batch.  Every smaller decode trace bucket that vLLM
        # may pad to needs a matching setup; sharing one immutable transform
        # across every layer avoids constructing a second decoder stack.
        for bucket in decode_trace_buckets(max_batch_size):
            if bucket == max_batch_size:
                continue
            bucket_rope = create_rope_setup(
                mesh_device=mesh_device,
                hf_config=hf_config,
                max_local_batch_size=bucket,
                users_row_sharded=False,
                datatype=ttnn.bfloat16,
                shard_batch_to_mesh_dim=0,
            )
            self._decode_rope_setups[bucket] = bucket_rope
            bucket_transform = bucket_rope.get_both_trans_mats()["decode"]
            self._decode_layer_transformation_mats[bucket] = [bucket_transform] * len(self.layers)
            bucket_kv_memory_config = get_kv_memory_config(
                mesh_device,
                max_local_batch_size=bucket,
                num_local_kv_heads=int(hf_config.num_key_value_heads) // tp,
                head_dim=int(hf_config.head_dim),
            )
            self._decode_layer_kv_memory_configs[bucket] = [bucket_kv_memory_config] * len(self.layers)
        self._active_decode_batch_size = max_batch_size
        self._validate_precision_runtime()

    def activate_decode_batch_size(self, batch_size: int) -> None:
        """Select all batch-dependent state whose shape matches a decode trace."""

        batch_size = int(batch_size)
        setup = self._decode_rope_setups.get(batch_size)
        transforms = self._decode_layer_transformation_mats.get(batch_size)
        kv_memory_configs = self._decode_layer_kv_memory_configs.get(batch_size)
        if setup is None or transforms is None or kv_memory_configs is None:
            raise ValueError(f"decode RoPE batch {batch_size} was not initialized")
        self.rope_setup = setup
        self.cos_matrix = setup.cos_matrix
        self.sin_matrix = setup.sin_matrix
        self.transformation_mats = setup.get_both_trans_mats()
        enable_decode_sharding = DecodeRMSNorm.sharding_enabled_for(batch_size)
        self.norm.enable_decode_sharding = enable_decode_sharding
        for layer, transform, kv_memory_config in zip(self.layers, transforms, kv_memory_configs):
            layer.self_attn.transformation_mats["decode"] = transform
            layer.self_attn.kv_mem_cfg = kv_memory_config
            layer.decoder.input_layernorm.enable_decode_sharding = enable_decode_sharding
            layer.decoder.post_attention_layernorm.enable_decode_sharding = enable_decode_sharding
        self._active_decode_batch_size = batch_size

    def ttnn_decode_forward(self, tokens, current_pos, *args, **kwargs):
        """Select the batch-shaped RoPE state for every eager or traced call."""

        actual_batch = int(current_pos.shape[-1])
        if actual_batch != self._active_decode_batch_size:
            self.activate_decode_batch_size(actual_batch)
        return super().ttnn_decode_forward(tokens, current_pos, *args, **kwargs)

    def _forward_layers_and_head(self, *args, is_decode=True, **kwargs):
        self.norm.decode_mode = is_decode
        self._terminal_uses_single_tile = is_decode or int(kwargs.get("get_last_token", -1)) != -1
        batch_size = int(kwargs.get("batch_size", 1))
        if not is_decode and batch_size == 1:
            seq_len = int(kwargs["hidden_states"].shape[-2])
            last = int(kwargs.get("get_last_token", -1))
            kwargs["fill_seq_lens"] = [seq_len if last < 0 else min(seq_len, last + ttnn.TILE_SIZE)]
        if not is_decode and batch_size == 1 and getattr(self, "_prefill_row_resume", None):
            plan = self._prefill_row_resume
            cursor = int(getattr(self, "_prefill_row_cursor", 0))
            if cursor >= len(plan):
                raise RuntimeError(f"prefill call {cursor + 1} exceeds the {len(plan)} planned rows")
            start, fill_start, cold = plan[cursor]
            self._prefill_row_cursor = cursor + 1
            got = int(kwargs.get("chunk_start_idx") or 0)
            if got != int(start):
                raise RuntimeError(f"prefill row {cursor} resumes at {got}; the plan says {start}")
            if int(fill_start) > int(start):
                kwargs["fill_start_idx"] = int(fill_start)
            if cold and kwargs.get("ring_tail_blocks") is not None:
                kwargs["ring_tail_blocks"] = [None if block is None else -1 for block in kwargs["ring_tail_blocks"]]
        if not is_decode and batch_size > 1:
            # Batched prefill.  The shared generator concatenates the users along
            # the sequence axis ([1, 1, B*S, H] with RoPE for B*S positions); the
            # decoder stack takes [1, B, S, H] with RoPE for one user's S
            # positions and concatenates internally for attention and the MoE.
            hidden_states = kwargs["hidden_states"]
            total_tokens = int(hidden_states.shape[-2])
            per_user = total_tokens // batch_size
            if per_user * batch_size != total_tokens:
                raise ValueError(f"batched prefill rows {total_tokens} are not a multiple of batch_size {batch_size}")
            kwargs["hidden_states"] = ttnn.reshape(
                hidden_states, (1, batch_size, per_user, int(hidden_states.shape[-1]))
            )
            rope = self._prefill_rope_slices.get(per_user)
            if rope is None:
                rope = [
                    self.rope_setup.cos_matrix_prefill[:, :, :per_user, :],
                    self.rope_setup.sin_matrix_prefill[:, :, :per_user, :],
                ]
                self._prefill_rope_slices[per_user] = rope
            kwargs["rope_mats"] = rope
            # The generator gathers each user's last hidden row and applies the
            # terminal norm + LM head on those rows (``_apply_norm_and_lm_head``);
            # running them over every token here would be wasted work.
            kwargs["skip_lm_head"] = True
            rows = getattr(self, "_prefill_row_lengths", None)
            if rows:
                lengths = [int(length) for length in rows][:batch_size]
                kwargs["fill_seq_lens"] = lengths + [0] * (batch_size - len(lengths))
        if not is_decode:
            self._prefill_output_is_hidden = bool(kwargs.get("skip_lm_head", False))
        return self._run_decoder_stack(*args, is_decode=is_decode, **kwargs)

    def process_logits_after_prefill_trace(self, output, last_token_idx):
        """Finish the selected prefill tile before the shared generator reads logits.

        Single-user prefill applies the terminal head in the decoder stack.
        Batched prefill defers it until each user's last tile has been selected,
        for both device sampling and the shared host-logits warmup path.
        """
        tile = super().process_logits_after_prefill_trace(output, last_token_idx)
        return self._apply_norm_and_lm_head(tile) if self._prefill_output_is_hidden else tile

    def _run_decoder_stack(
        self,
        hidden_states,
        rope_mats,
        current_pos,
        page_table,
        kv_cache,
        get_last_token=-1,
        is_decode=True,
        user_id=0,
        batch_size=1,
        skip_lm_head=False,
        page_tables_per_layer=None,
        fill_seq_lens=None,
        chunk_start_idx=None,
        ring_tail_blocks=None,
        fill_start_idx=None,
    ):
        """
        Shared forward pass through decoder layers and final projection.

        Args:
            hidden_states: Input tensor
            rope_mats: RoPE rotation matrices [cos, sin]
            current_pos: Current position (for decode) or None (for prefill)
            page_table: Single page table; used for every layer when
                ``page_tables_per_layer`` is None (legacy / uniform attention).
            kv_cache: KV cache list per layer.
            page_tables_per_layer: Optional list of per-layer page tables, one
                entry per decoder layer. When set, each layer's attention
                receives ``page_tables_per_layer[i]`` instead of ``page_table``.
                vLLM's hybrid kv cache manager produces this list so
                sliding-window layers can index a smaller paged pool than
                full-attention layers (KV cache groups). When None, behavior is
                byte-equivalent to the pre-hybrid path.

        Returns:
            logits: Output logits
        """
        if page_tables_per_layer is not None and len(page_tables_per_layer) != len(self.layers):
            raise ValueError(
                f"page_tables_per_layer has {len(page_tables_per_layer)} entries "
                f"but model has {len(self.layers)} layers"
            )

        # Process through decoder layers
        for i, decoder_layer in enumerate(self.layers):
            layer_kv_cache = kv_cache[i] if kv_cache is not None else None
            layer_page_table = page_tables_per_layer[i] if page_tables_per_layer is not None else page_table
            extra = {}
            if fill_seq_lens is not None:
                extra["fill_seq_lens"] = fill_seq_lens
            if chunk_start_idx is not None:
                extra["chunk_start_idx"] = int(chunk_start_idx)
                if ring_tail_blocks is not None:
                    extra["ring_tail_block"] = ring_tail_blocks[i]
            if fill_start_idx is not None:
                extra["fill_start_idx"] = int(fill_start_idx)
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=rope_mats,
                position_idx=current_pos,
                page_table=layer_page_table,
                kv_cache=layer_kv_cache,
                is_decode=is_decode,
                user_id=user_id,
                batch_size=batch_size,
                **extra,
            )
        logits = hidden_states

        if get_last_token != -1:
            if len(logits.shape) == 3:
                logits = ttnn.unsqueeze(logits, dim=1)
            if batch_size > 1:
                # Batch>1: tokens are concatenated [1,1,B*S,H]. Extract each user's 32-token tile.
                per_user_seq = logits.shape[2] // batch_size
                tiles = []
                for b in range(batch_size):
                    start = b * per_user_seq + get_last_token
                    tile = ttnn.slice(logits, (0, 0, start, 0), (1, 1, start + 32, logits.shape[-1]))
                    tiles.append(tile)
                logits.deallocate(True)
                logits = ttnn.concat(tiles, dim=2)  # [1, 1, B*32, H]
                for t in tiles:
                    t.deallocate(True)
            else:
                logits_sliced = ttnn.slice(
                    logits,
                    (0, 0, get_last_token, 0),
                    (1, 1, get_last_token + 32, logits.shape[-1]),
                )
                logits.deallocate(True)
                logits = logits_sliced
            hidden_states = logits

        if skip_lm_head:
            return hidden_states

        # Final norm and lm_head
        hidden_states = self.norm(hidden_states)
        logits = self._apply_lm_head(hidden_states)
        hidden_states.deallocate(True)
        self._prefill_sampling_active = False
        # TP all-gather is deferred to process_output_prefill / process_output_decode
        # (outside trace capture) since all_gather_async writes to device,
        # which is forbidden during trace capture.

        return logits

    def _transient_prefill_page_tables_to_ttnn(self, page_tables_per_layer):
        """Materialize scheduler-row slices for an untraced sequential prefill."""

        converted_by_id = {}
        converted = []
        owned = []
        for table in page_tables_per_layer:
            if table is None or isinstance(table, ttnn.Tensor):
                converted.append(table)
                continue
            key = id(table)
            tt_table = converted_by_id.get(key)
            if tt_table is None:
                tt_table = ttnn.from_torch(
                    table,
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=self._page_table_mesh_mapper(table.shape[0]),
                )
                converted_by_id[key] = tt_table
                owned.append(tt_table)
            converted.append(tt_table)
        return converted, owned

    def ttnn_prefill_forward(
        self,
        x,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token=-1,
        kv_cache=None,
        batch_size=1,
        skip_lm_head=False,
        page_tables_per_layer=None,
    ):
        explicit_page_tables = page_tables_per_layer is not None
        if not explicit_page_tables:
            # See ttnn_decode_forward: the bridge stashes per-layer page tables
            # on the model when in vLLM hybrid mode, since Generator's prefill
            # path doesn't thread the kwarg.
            page_tables_per_layer = getattr(self, "_active_page_tables_per_layer", None)
        ring_tail_blocks = None
        if chunk_start_idx:
            ring_tail_blocks = self._ring_tail_blocks(page_tables_per_layer, int(chunk_start_idx))
            page_tables_per_layer = self._widen_chunk_page_tables(
                page_tables_per_layer, int(chunk_start_idx) + int(x.shape[-2]), kv_cache
            )
        transient_page_tables = []
        if explicit_page_tables:
            page_tables_per_layer, transient_page_tables = self._transient_prefill_page_tables_to_ttnn(
                page_tables_per_layer
            )
        else:
            page_tables_per_layer = self._page_tables_to_ttnn(page_tables_per_layer)
        """Prefill forward pass - processes full sequences"""
        # Use provided rotation matrices or slice from rope_setup (matches tt-transformers)
        seq_len = x.shape[-2]
        if rot_mats_global is not None:
            rope_mats = rot_mats_global
        else:
            # Slice cos/sin matrices for prefill sequence length (matches tt-transformers model.py lines 156-159)
            rope_mats = [
                self.rope_setup.cos_matrix_prefill[:, :, :seq_len, :],
                self.rope_setup.sin_matrix_prefill[:, :, :seq_len, :],
            ]

        # Forward through layers and head (shared with decode)
        try:
            logits = self._forward_layers_and_head(
                hidden_states=x,
                rope_mats=rope_mats,
                current_pos=None,  # No current_pos for prefill
                page_table=page_table,
                kv_cache=kv_cache,
                get_last_token=get_last_token,
                is_decode=False,
                user_id=user_id,
                batch_size=batch_size,
                skip_lm_head=skip_lm_head,
                page_tables_per_layer=page_tables_per_layer,
                chunk_start_idx=int(chunk_start_idx) if chunk_start_idx else None,
                ring_tail_blocks=ring_tail_blocks,
            )
        finally:
            for table in transient_page_tables:
                table.deallocate(True)

        return logits

    @staticmethod
    def _widen_chunk_page_tables(page_tables_per_layer, padded_end: int, kv_cache):
        """Give every host page table at least the columns the padded chunk end needs.

        The chunked attention requires the table to cover ``chunk_start + padded
        chunk length``; a trimmed table covers only the prompt's own blocks. The
        extra columns name block 0 (vLLM's null block): the keys read from them
        belong to padded query rows, whose outputs are dropped, and lie beyond
        every real query's causal bound.
        """
        if page_tables_per_layer is None:
            return None
        widened = []
        for layer_idx, table in enumerate(page_tables_per_layer):
            if not isinstance(table, torch.Tensor):
                widened.append(table)
                continue
            block_size = PAGE_SIZE
            if kv_cache is not None and layer_idx < len(kv_cache) and kv_cache[layer_idx] is not None:
                block_size = int(kv_cache[layer_idx][0].shape[2])
            needed = (padded_end + block_size - 1) // block_size
            if int(table.shape[-1]) >= needed:
                widened.append(table)
                continue
            pad = torch.zeros(table.shape[0], needed - int(table.shape[-1]), dtype=table.dtype)
            widened.append(torch.cat([table, pad], dim=-1))
        return widened

    def _ring_tail_blocks(self, page_tables_per_layer, chunk_start_idx: int):
        """First physical block of the previous window for every bounded-ring layer.

        A chunk that starts at ``chunk_start_idx`` needs the ``sliding_window``
        positions before it; a ring layer keeps them in the ring blocks that the
        host page table names at columns ``[start/block - window/block, start/block)``.
        The two blocks of a 128-token window are consecutive in the ring, so the
        first id is enough. Entries are None for full-attention layers.
        """
        if page_tables_per_layer is None:
            return None
        result = []
        for layer, table in zip(self.layers, page_tables_per_layer):
            attention = layer.self_attn
            modulo = attention.config.cache_position_modulo
            if modulo is None or table is None or not isinstance(table, torch.Tensor):
                result.append(None)
                continue
            window = int(attention.config.sliding_window)
            block_size = int(attention.kv_cache[0].shape[2]) if attention.kv_cache is not None else PAGE_SIZE
            if chunk_start_idx % block_size:
                raise ValueError(f"chunk_start_idx {chunk_start_idx} is not a multiple of the block size {block_size}")
            window_blocks = window // block_size
            first_col = chunk_start_idx // block_size - window_blocks
            if first_col < 0 or first_col >= int(table.shape[-1]):
                raise ValueError(f"page table with {table.shape[-1]} columns cannot address block {first_col}")
            ring_blocks = modulo // block_size
            if first_col % ring_blocks + window_blocks > ring_blocks:
                raise ValueError(f"the window before {chunk_start_idx} wraps the {modulo}-token ring")
            result.append(int(table[0, first_col]))
        return result

    def prepare_inputs_prefill(self, tokens, start_pos=0, **kwargs):
        prepared = super().prepare_inputs_prefill(tokens, start_pos=start_pos, **kwargs)
        if not start_pos:
            return prepared
        seq_len = self.args.max_seq_len if kwargs.get("trace_enabled", False) else prepared[0].shape[-2]
        rope_key = (int(start_pos), int(seq_len))
        rope = self._prefill_rope_slices.get(rope_key)
        if rope is None:
            rope = [
                self.rope_setup.cos_matrix_prefill[:, :, start_pos : start_pos + seq_len, :],
                self.rope_setup.sin_matrix_prefill[:, :, start_pos : start_pos + seq_len, :],
            ]
            self._prefill_rope_slices[rope_key] = rope
        return prepared[0], rope, *prepared[2:]

    def extract_last_tokens_batched_prefill(
        self, hidden_states, last_token_idx_list, padded_batch, prefill_seq_len, target_batch=None, slot_map=None
    ):
        """Gather each batched-prefill user's last hidden row into one [1, 1, target, H] tile.

        ``hidden_states`` is the un-normed decoder output viewed as
        ``[padded_batch, 1, prefill_seq_len, H]`` (replicated across the mesh);
        row ``i`` holds request ``i`` in compact-row mode.  Its last token goes to
        output row ``slot_map[i]`` (the request's physical sampler slot) or to
        row ``i`` when no map is given.  Device-side: one untilize and one
        embedding gather, both keyed only on ``(padded_batch * prefill_seq_len)``.
        """
        rows = int(padded_batch)
        seq = int(prefill_seq_len)
        target = int(target_batch) if target_batch is not None else rows
        hidden = int(hidden_states.shape[-1])
        flat = ttnn.reshape(hidden_states, (1, 1, rows * seq, hidden))
        table = ttnn.reshape(ttnn.to_layout(flat, ttnn.ROW_MAJOR_LAYOUT), (rows * seq, hidden))
        index = torch.zeros((1, target), dtype=torch.int32)
        mapped = list(slot_map) if slot_map is not None else list(range(min(rows, target)))
        for row, slot in enumerate(mapped):
            index[0, int(slot)] = row * seq + int(last_token_idx_list[row])
        index_device = ttnn.from_torch(
            index,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        gathered = ttnn.embedding(index_device, table, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        index_device.deallocate(True)
        table.deallocate(True)
        return ttnn.reshape(gathered, (1, 1, target, hidden))

    def _apply_norm_and_lm_head(self, x):
        """Terminal norm + LM head for the gathered batched-prefill rows ([1, 1, 32, H])."""
        self.norm.decode_mode = False
        normed = self.norm(x)
        self._terminal_uses_single_tile = True
        logits = self._apply_lm_head(normed)
        if normed is not x:
            normed.deallocate(True)
        return logits

    def _apply_lm_head(self, hidden_states):
        if self.dram_sharded_lm_head is not None and self._terminal_uses_single_tile:
            return self.dram_sharded_lm_head(hidden_states)
        return ttnn.matmul(
            hidden_states,
            self.lm_head_weight,
            dtype=self.lm_head_output_dtype,
            compute_kernel_config=self.lm_head_compute_kernel_config,
        )

    def _validate_precision_runtime(self) -> None:
        """Fail construction if any selected precision field misses the measured runtime path."""

        terminal = self.precision_config.terminal_dtypes()
        if self.embedding_weight.dtype != terminal["embedding"]:
            raise RuntimeError("precision config embedding dtype was not consumed")
        if self.norm.tt_weight.dtype != terminal["normalization"]:
            raise RuntimeError("precision config final normalization dtype was not consumed")
        if self.lm_head_weight.dtype != terminal["lm_head_weight"]:
            raise RuntimeError("precision config LM-head weight dtype was not consumed")
        if self.lm_head_output_dtype != terminal["lm_head_output"]:
            raise RuntimeError("precision config LM-head output dtype was not consumed")
        if self.sampling_accumulator_dtype != terminal["sampling_accumulator"]:
            raise RuntimeError("precision config sampling accumulator assumption was not consumed")
        if self.full_logits_gather_policy != terminal["full_logits_gather"]:
            raise RuntimeError("precision config full-logits-gather policy was not consumed")
        if self.topk_values_gather_dtype != terminal["topk_values_gather_dtype"]:
            raise RuntimeError("precision config top-k values gather dtype was not consumed")
        if self.sampling is None:
            raise RuntimeError("selected sampling precision cannot be validated without on-device sampling")
        sampling = self.sampling.tt_sampling
        if sampling._allow_force_argmax_sampling:
            raise RuntimeError("non-materialized logits-gather policy enabled the full-vocabulary argmax path")
        if sampling._force_argmax_sampling:
            raise RuntimeError("non-materialized logits-gather policy selected the full-vocabulary argmax path")
        if self.topk_values_gather_dtype != ttnn.bfloat16:
            raise RuntimeError("TTSampling's top-k values gather must use bfloat16")
        if any(
            tensor.dtype != self.sampling_accumulator_dtype
            for tensor in (
                sampling.p_tensor,
                sampling.temp_tensor,
                sampling._greedy_col,
            )
        ):
            raise RuntimeError("precision config sampling accumulator did not match TTSampling device buffers")
        for layer_idx, adapter in enumerate(self.layers):
            decoder = adapter.decoder
            policy = self.precision_config.decoder_policy_for_layer(layer_idx)
            attention = decoder.self_attn
            mlp = decoder.mlp
            checks = {
                "decoder_policy": decoder.policy == policy,
                "attention_qkv_weight": attention.weights.wqkv.dtype == policy.attention_weight_dtype,
                "attention_output_weight": attention.weights.o_proj.dtype == policy.attention_weight_dtype,
                "kv_cache": (
                    all(cache.dtype == policy.kv_cache_dtype for cache in decoder.kv_cache)
                    if decoder.kv_cache is not None
                    else attention.cache_dtype == policy.kv_cache_dtype
                ),
                "router_weight": mlp.router.weight.dtype == policy.router_weight_dtype,
                "input_normalization_weight": decoder.input_layernorm.tt_weight.dtype
                == policy.normalization_weight_dtype,
                "post_attention_normalization_weight": decoder.post_attention_layernorm.tt_weight.dtype
                == policy.normalization_weight_dtype,
                "expert_down_weight": mlp.indexed_down.dtype == policy.expert_weight_dtype,
                "attention_ccl": attention.activation_ccl_dtype
                == (policy.attention_activation_ccl_dtype or policy.activation_ccl_dtype),
                "expert_ccl": mlp.activation_ccl_dtype
                == (policy.expert_activation_ccl_dtype or policy.activation_ccl_dtype),
                "attention_projection_input": attention.prefill_projection_input_dtype
                == policy.attention_projection_input_dtype,
                "expert_intermediate": mlp.expert_intermediate_dtype == policy.expert_intermediate_dtype,
                "stack_residual": attention.residual_dtype == policy.residual_dtype,
                "decode_attention_fidelity": attention.decode_projection_compute_kernel_config.math_fidelity
                == policy.projection_math_fidelity,
                "prefill_attention_fidelity": attention.prefill_projection_compute_kernel_config.math_fidelity
                == policy.prefill_projection_math_fidelity,
                "expert_fidelity": mlp.expert_compute_kernel_config.math_fidelity == policy.expert_math_fidelity,
                "router_fidelity": mlp.router.compute_config.math_fidelity == policy.router_math_fidelity,
                "sdpa_fidelity": attention.program_config.math_fidelity == policy.attention_sdpa_math_fidelity.name,
            }
            if mlp.indexed_gate_up is not None:
                checks["expert_gate_up_weight"] = mlp.indexed_gate_up.dtype == policy.expert_weight_dtype
            failed = sorted(name for name, passed in checks.items() if not passed)
            if failed:
                raise RuntimeError(f"precision config fields not consumed by layer {layer_idx}: {failed}")

    def precision_runtime_evidence(self) -> dict:
        """Compact proof that the selected JSON controls the constructed tensors and kernels."""

        repo_root = Path(__file__).resolve().parents[4]

        def normalized_path(path: Path | None) -> str:
            if path is None:
                return "builtin"
            resolved = path.resolve()
            try:
                return str(resolved.relative_to(repo_root))
            except ValueError:
                return str(resolved)

        unique = {}
        for index, adapter in enumerate(self.layers):
            policy = self.precision_config.decoder_policy_for_layer(index)
            decoder = adapter.decoder
            attention = decoder.self_attn
            mlp = decoder.mlp
            signature = (
                dtype_name(attention.weights.wqkv.dtype),
                dtype_name(mlp.indexed_down.dtype),
                dtype_name(mlp.router.weight.dtype),
                dtype_name(decoder.input_layernorm.tt_weight.dtype),
                dtype_name(decoder.post_attention_layernorm.tt_weight.dtype),
                dtype_name(decoder.kv_cache[0].dtype if decoder.kv_cache is not None else attention.cache_dtype),
                dtype_name(attention.activation_ccl_dtype),
                dtype_name(mlp.activation_ccl_dtype),
                dtype_name(attention.residual_dtype),
                dtype_name(attention.prefill_projection_input_dtype),
                dtype_name(mlp.expert_intermediate_dtype),
                math_fidelity_name(attention.decode_projection_compute_kernel_config.math_fidelity),
                math_fidelity_name(attention.prefill_projection_compute_kernel_config.math_fidelity),
                attention.program_config.math_fidelity,
                math_fidelity_name(mlp.expert_compute_kernel_config.math_fidelity),
                math_fidelity_name(mlp.router.compute_config.math_fidelity),
            )
            unique.setdefault(signature, []).append(index)
        return {
            "schema_version": 2,
            "config_id": self.precision_config.config_id,
            "config_path": normalized_path(self.precision_config.source_path),
            "config_sha256": self.precision_config.source_sha256,
            "default_selected_config_path": normalized_path(Path(__file__).resolve().parents[1] / "precision.json"),
            "terminal": {
                "embedding_weight": dtype_name(self.embedding_weight.dtype),
                "normalization_weight": dtype_name(self.norm.tt_weight.dtype),
                "lm_head_weight": dtype_name(self.lm_head_weight.dtype),
                "lm_head_output": dtype_name(self.lm_head_output_dtype),
                "lm_head_math_fidelity": math_fidelity_name(self.lm_head_compute_kernel_config.math_fidelity),
                "sampling_accumulator": dtype_name(self.sampling_accumulator_dtype),
                "sampling_device_buffers": {
                    "p": dtype_name(self.sampling.tt_sampling.p_tensor.dtype),
                    "temperature": dtype_name(self.sampling.tt_sampling.temp_tensor.dtype),
                    "greedy_mask": dtype_name(self.sampling.tt_sampling._greedy_col.dtype),
                },
                "full_logits_gather": {
                    **self.full_logits_gather_policy,
                    "runtime_path": "sharded TTSampling consumes LM-head output directly",
                    "force_full_vocab_argmax_allowed": self.sampling.tt_sampling._allow_force_argmax_sampling,
                    "force_full_vocab_argmax_selected": self.sampling.tt_sampling._force_argmax_sampling,
                },
                "topk_values_gather_dtype": dtype_name(self.topk_values_gather_dtype),
            },
            "layer_runtime_groups": [
                {
                    "layers": layer_indices,
                    "attention_weight": signature[0],
                    "expert_weight": signature[1],
                    "router_weight": signature[2],
                    "input_normalization_weight": signature[3],
                    "post_attention_normalization_weight": signature[4],
                    "kv_cache": signature[5],
                    "attention_ccl": signature[6],
                    "expert_ccl": signature[7],
                    "residual": signature[8],
                    "attention_projection_input": signature[9],
                    "expert_intermediate": signature[10],
                    "decode_attention_fidelity": signature[11],
                    "prefill_attention_fidelity": signature[12],
                    "attention_sdpa_fidelity": signature[13],
                    "expert_fidelity": signature[14],
                    "router_fidelity": signature[15],
                }
                for signature, layer_indices in unique.items()
            ],
            "validation": "all selected fields matched constructed tensors/kernel configs",
        }

    @classmethod
    def from_checkpoint(
        cls,
        mesh_device,
        *,
        snapshot_path: str | Path | None = None,
        tensor_cache_path: str | Path | None = None,
        max_batch_size: int = 1,
        max_context_length: int = HF_CONTEXT_LENGTH,
        num_layers: int = MODEL_LAYERS,
        allow_reduced_model: bool = False,
        precision_config: PrecisionConfig | str | Path | None = None,
        lm_head_policy: str = INTERLEAVED_LM_HEAD,
        create_kv_cache: bool = True,
        salt_duplicate_seeds: bool = True,
    ):
        snapshot_path = Path(
            snapshot_path or os.environ.get("GPT_OSS_120B_SNAPSHOT", "") or os.environ.get("HF_MODEL", "")
        ).expanduser()
        if not str(snapshot_path) or str(snapshot_path) == ".":
            raise FileNotFoundError("Set GPT_OSS_120B_SNAPSHOT or HF_MODEL to the pinned openai/gpt-oss-120b snapshot")
        snapshot_path = snapshot_path.resolve()
        hf_config = AutoConfig.from_pretrained(snapshot_path, trust_remote_code=True, local_files_only=True)
        if int(hf_config.max_position_embeddings) != HF_CONTEXT_LENGTH:
            raise ValueError(
                f"Pinned GPT-OSS config must advertise {HF_CONTEXT_LENGTH}, got {hf_config.max_position_embeddings}"
            )
        tp = int(mesh_device.shape[1])
        precision_config = (
            precision_config
            if isinstance(precision_config, PrecisionConfig)
            else load_precision_config(precision_config)
        )
        base_policy = precision_config.decoder_policy_for_layer(0)
        evidence = require_resident_capacity(
            tp=tp,
            max_batch_size=max_batch_size if create_kv_cache else 1,
            max_context_length=max_context_length,
            num_layers=num_layers,
            allow_reduced_model=allow_reduced_model,
            kv_cache_dtype=base_policy.kv_cache_dtype,
        )
        logger.info(
            f"Resident full-model capacity: TP={tp}, {evidence.total_bytes_per_device / 1024**3:.3f} GiB/device"
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot_path, trust_remote_code=True, local_files_only=True)
        generation_config = GenerationConfig.from_pretrained(
            snapshot_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        cache_path = Path(
            tensor_cache_path
            or os.environ.get(
                "GPT_OSS_120B_FULL_MODEL_TENSOR_CACHE",
                str(Path(os.environ.get("TMPDIR", "/tmp")) / "gpt_oss_120b_full_model_tensor_cache"),
            )
        ).expanduser()
        checkpoint = StreamingCheckpoint(snapshot_path)
        args = FullModelArgs(
            mesh_device=mesh_device,
            hf_config=hf_config,
            generation_config=generation_config,
            tokenizer=tokenizer,
            snapshot_path=snapshot_path,
            tensor_cache_path=cache_path,
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            num_layers=num_layers,
            precision_config=precision_config,
            salt_duplicate_seeds=salt_duplicate_seeds,
        )
        terminal = checkpoint.terminal_state_dict()
        model = cls(
            mesh_device=mesh_device,
            hf_config=hf_config,
            terminal_state_dict=terminal,
            layer_loader=checkpoint.layer_state_dict,
            args=args,
            tensor_cache_path=cache_path,
            max_batch_size=max_batch_size,
            max_context_length=max_context_length,
            num_layers=num_layers,
            precision_config=precision_config,
            lm_head_policy=lm_head_policy,
            create_kv_cache=create_kv_cache,
        )
        del terminal
        gc.collect()
        return model, args


def build_model(mesh_device, **kwargs):
    """Standard autoport builder returning model, args, and owned paged cache."""

    model, args = Model.from_checkpoint(mesh_device, **kwargs)
    return model, args, model.kv_cache


__all__ = [
    "CapacityEvidence",
    "FullModelArgs",
    "FullModelCapacityError",
    "DRAM_SHARDED_LM_HEAD",
    "HF_CONTEXT_LENGTH",
    "INTERLEAVED_LM_HEAD",
    "MODEL_ID",
    "MODEL_REVISION",
    "Model",
    "StreamingCheckpoint",
    "build_model",
    "capacity_evidence",
    "require_resident_capacity",
]
