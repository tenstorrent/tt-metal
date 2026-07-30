# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full TTNN autoregressive model for ``google/gemma-4-12B``.

This wrapper assembles the repo-local optimized multichip decoder layer into a
complete dense Gemma 4 text model: token embeddings, all decoder layers, final
RMSNorm, tied LM head, softcap, paged KV cache helpers, and greedy on-device
decode sampling.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import torch
import ttnn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig

from models.common.modules.tt_ccl import get_tt_ccl
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.utils.general_utils import get_cache_file_name
from models.demos.gemma4.utils.substate import substate


SUPPORTED_HF_MODEL_ID = "google/gemma-4-12B"
TARGET_MESH_SHAPE = (1, 8)
TARGET_TP = 8
DEFAULT_BLOCK_SIZE = 64
DEFAULT_MAX_SEQ_LEN = 4096


def _load_multichip_module():
    path = Path(__file__).with_name("multichip_decoder.py")
    spec = importlib.util.spec_from_file_location("gemma4_12b_multichip_decoder_for_full_model", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"cannot load multichip decoder from {path}")
    spec.loader.exec_module(module)
    return module


_multichip = _load_multichip_module()
MultichipDecoder = _multichip.MultichipDecoder
OptimizedRMSNorm = _multichip.OptimizedRMSNorm
RingCCLManager = _multichip.RingCCLManager
_compute_kernel_config = _multichip._compute_kernel_config
_create_dram_sharded_mem_config = _multichip._create_dram_sharded_mem_config
_width_sharded_mem_config = _multichip._width_sharded_mem_config


def resolve_model_path(hf_model_id: str | Path = SUPPORTED_HF_MODEL_ID) -> str:
    """Resolve a local checkpoint path when possible, falling back to HF id."""

    model_id = str(hf_model_id)
    if Path(model_id).exists():
        return model_id
    try:
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return model_id


def _wanted_language_key(key: str, num_layers: int | None) -> bool:
    if key in {
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "embed_tokens.weight",
        "lm_head.weight",
    }:
        return True
    if key.startswith("model.language_model.layers.") or key.startswith("language_model.layers."):
        if num_layers is None:
            return True
        parts = key.split(".")
        try:
            layer_idx = int(parts[3] if parts[0] == "model" else parts[2])
        except (IndexError, ValueError):
            return False
        return layer_idx < num_layers
    return False


def _load_language_state_dict(model_path: str | Path, *, num_layers: int | None = None) -> dict[str, torch.Tensor]:
    """Load only the text-model tensors needed by this autoport."""

    model_path = Path(model_path)
    state_dict: dict[str, torch.Tensor] = {}
    safetensor_files = sorted(model_path.glob("*.safetensors")) if model_path.is_dir() else []
    if safetensor_files:
        for file_path in safetensor_files:
            shard = load_file(str(file_path))
            for key, value in shard.items():
                if _wanted_language_key(key, num_layers):
                    state_dict[key] = value.to(torch.bfloat16) if value.dtype == torch.float32 else value
        if not state_dict:
            raise RuntimeError(f"no Gemma4 language tensors found in {model_path}")
        return state_dict

    return Gemma4ModelArgs.load_state_dict(str(model_path), dummy_weights=False)


def _as_text_config(hf_config):
    return getattr(hf_config, "text_config", hf_config)


def _require_supported_config(hf_config) -> Gemma4ModelArgs:
    text_config = _as_text_config(hf_config)
    model_args = Gemma4ModelArgs.from_hf_config(text_config)
    if model_args.hidden_size != 3840:
        raise ValueError(f"{SUPPORTED_HF_MODEL_ID} hidden_size must be 3840, got {model_args.hidden_size}")
    if model_args.num_hidden_layers != 48:
        raise ValueError(
            f"{SUPPORTED_HF_MODEL_ID} num_hidden_layers must be 48, got {model_args.num_hidden_layers}"
        )
    if model_args.enable_moe_block:
        raise NotImplementedError(f"{SUPPORTED_HF_MODEL_ID} is expected to be dense; MoE path is out of scope")
    if model_args.hidden_size_per_layer_input:
        raise NotImplementedError(f"{SUPPORTED_HF_MODEL_ID} does not use per-layer input embeddings")
    if model_args.num_kv_shared_layers:
        raise NotImplementedError(f"{SUPPORTED_HF_MODEL_ID} does not use shared KV layers")
    if not model_args.tie_word_embeddings:
        raise NotImplementedError("only tied embedding/LM-head Gemma4 text checkpoints are supported")
    return model_args


def _dtype_name(dtype) -> str:
    if dtype == ttnn.bfloat16:
        return "bf16"
    if dtype == ttnn.bfloat8_b:
        return "bfp8"
    if hasattr(ttnn, "bfloat4_b") and dtype == ttnn.bfloat4_b:
        return "bfp4"
    return str(dtype).replace("ttnn.", "").replace("DataType.", "").lower()


def _dtype_from_env(name: str, default):
    value = os.getenv(name, "").lower()
    if value in ("", "default"):
        return default
    if value in ("bf16", "bfloat16"):
        return ttnn.bfloat16
    if value in ("bfp8", "bfloat8_b"):
        return ttnn.bfloat8_b
    if value in ("bfp4", "bfloat4_b") and hasattr(ttnn, "bfloat4_b"):
        return ttnn.bfloat4_b
    raise ValueError(f"unsupported {name}={value!r}")


def _math_fidelity_name(fidelity) -> str:
    return str(fidelity).replace("ttnn.", "").replace("MathFidelity.", "")


def _math_fidelity_from_env(name: str, default):
    value = os.getenv(name, "").lower()
    if value in ("", "default"):
        return default
    if value == "lofi":
        return ttnn.MathFidelity.LoFi
    if value == "hifi2":
        return ttnn.MathFidelity.HiFi2
    if value == "hifi3":
        return ttnn.MathFidelity.HiFi3
    if value == "hifi4":
        return ttnn.MathFidelity.HiFi4
    raise ValueError(f"unsupported {name}={value!r}")


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _width_sharded_padded_mem_config(*, rows: int, width: int, grid: ttnn.CoreGrid) -> ttnn.MemoryConfig:
    shard_width = _ceil_to_multiple(math.ceil(width / grid.num_cores), ttnn.TILE_SIZE)
    return ttnn.create_sharded_memory_config(
        (rows, shard_width),
        grid,
        ttnn.ShardStrategy.WIDTH,
        ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _mesh_shape_tuple(mesh_device) -> tuple[int, int]:
    shape = getattr(mesh_device, "shape", None)
    if shape is None:
        return (1, 1)
    return tuple(int(x) for x in shape)


def _replicate_mapper(mesh_device):
    return ttnn.ReplicateTensorToMesh(mesh_device) if hasattr(mesh_device, "shape") else None


def _first_device_torch(tensor, mesh_device) -> torch.Tensor:
    if hasattr(mesh_device, "shape"):
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    return ttnn.to_torch(tensor)


def _all_gather_tp(tensor, mesh_config: MeshConfig, ccl_manager: RingCCLManager, *, dim=3, memory_config=None):
    if mesh_config.tp <= 1:
        return tensor
    gathered = ttnn.all_gather(
        tensor,
        dim=dim,
        cluster_axis=mesh_config.tp_axis,
        num_links=ccl_manager.num_links,
        topology=ccl_manager.topology,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )
    tensor.deallocate(True)
    return gathered


def create_rope_caches(mesh_device, hf_text_config, max_seq_len: int):
    """Create per-layer-type Gemma4 HF-format RoPE cos/sin caches."""

    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    rope = Gemma4TextRotaryEmbedding(hf_text_config)
    x_dummy = torch.zeros(1, max_seq_len, hf_text_config.hidden_size)
    position_ids = torch.arange(max_seq_len).unsqueeze(0)
    mapper = _replicate_mapper(mesh_device)

    caches_4d = {}
    caches_2d = {}
    for layer_type in sorted(set(hf_text_config.layer_types)):
        cos, sin = rope(x_dummy, position_ids, layer_type=layer_type)
        caches_4d[layer_type] = (
            ttnn.from_torch(
                cos.unsqueeze(0).to(torch.bfloat16),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                sin.unsqueeze(0).to(torch.bfloat16),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=mapper,
            ),
        )
        caches_2d[layer_type] = (
            ttnn.from_torch(
                cos[0].to(torch.bfloat16),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=mapper,
            ),
            ttnn.from_torch(
                sin[0].to(torch.bfloat16),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=mapper,
            ),
        )
    return caches_4d, caches_2d


class Gemma412BModel:
    """Full Gemma 4 12B text model assembled around ``MultichipDecoder``."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        state_dict: dict[str, torch.Tensor],
        model_path: str | Path,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        max_batch_size: int = 1,
        num_layers: int | None = None,
        dtype=ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        enable_sampling: bool = True,
    ):
        self.mesh_device = mesh_device
        self.hf_text_config = _as_text_config(hf_config)
        self.hf_config = self.hf_text_config
        self.model_args = _require_supported_config(self.hf_text_config)
        self.hidden_size = self.model_args.hidden_size
        self.vocab_size = self.model_args.vocab_size
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.block_size = block_size
        self.dtype = dtype
        self.embed_scale = self.hidden_size**0.5
        self.final_logit_softcapping = self.model_args.final_logit_softcapping
        self.num_layers = self.model_args.num_hidden_layers if num_layers is None else int(num_layers)
        if self.num_layers < 1 or self.num_layers > self.model_args.num_hidden_layers:
            raise ValueError(f"num_layers must be in [1, {self.model_args.num_hidden_layers}], got {num_layers}")

        mesh_shape = _mesh_shape_tuple(mesh_device)
        tp = mesh_shape[1] if mesh_shape != (1, 1) else 1
        self.mesh_config = MeshConfig(mesh_shape, decode=ModeConfig(tp=tp), prefill=ModeConfig(tp=tp))
        self.ccl_manager = RingCCLManager(mesh_device=mesh_device) if tp > 1 else None

        if tensor_cache_path is None:
            cache_root = os.getenv("TT_CACHE_PATH")
            if cache_root is None:
                cache_root = Gemma4ModelArgs.resolve_model_cache_path(str(model_path))
            tensor_cache_path = Path(cache_root) / "gemma4_12b_full_model" / f"tensor_cache_{_dtype_name(dtype)}"
        self.tensor_cache_path = Path(tensor_cache_path)
        self.tensor_cache_path.mkdir(parents=True, exist_ok=True)

        self.rope_caches, self.rope_caches_2d = create_rope_caches(mesh_device, self.hf_text_config, max_seq_len)

        self._load_embedding_and_lm_head(state_dict)
        self.layers = [
            MultichipDecoder.from_state_dict(
                state_dict,
                hf_config=self.hf_text_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                mesh_config=self.mesh_config,
                ccl_manager=self.ccl_manager,
                tensor_cache_path=self.tensor_cache_path / "layers",
            )
            for layer_idx in range(self.num_layers)
        ]
        norm_state = substate(state_dict, "model.language_model.norm") or substate(state_dict, "model.norm")
        self.norm = OptimizedRMSNorm(
            mesh_device=mesh_device,
            hf_config=self.model_args,
            state_dict=norm_state,
            tensor_cache_path=self.tensor_cache_path / "final_norm",
            decode_sharded=True,
        )

        self.sampling = self._make_sampling() if enable_sampling and tp > 1 else None
        self.summary = {
            "hf_model_id": SUPPORTED_HF_MODEL_ID,
            "path": str(model_path),
            "layers": self.num_layers,
            "mesh_shape": mesh_shape,
            "tp": self.mesh_config.tp,
            "block_stack": "MultichipDecoder" if self.mesh_config.tp > 1 else "OptimizedDecoder fallback",
            "embedding": "TP column-parallel hidden shard, ring all-gather to replicated residual stream",
            "decoder": getattr(self.layers[0], "multichip_summary", getattr(self.layers[0], "optimization_summary", {})),
            "final_norm": "OptimizedRMSNorm decode-sharded; decode/last-token LM head consumes width-sharded L1",
            "lm_head": (
                "tied embedding weight, TP column-parallel vocab shard; "
                f"prefill={_dtype_name(self.lm_head_prefill_dtype)}, "
                f"decode={_dtype_name(self.lm_head_decode_dtype)} "
                f"{_math_fidelity_name(self.lm_head_decode_fidelity)} DRAM-sharded matmul"
            ),
            "sampling": "models.common.sampling SamplingGenerator" if self.sampling is not None else "host logits",
        }

    @classmethod
    def from_pretrained(
        cls,
        *,
        mesh_device,
        hf_model_id: str | Path = SUPPORTED_HF_MODEL_ID,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        max_batch_size: int = 1,
        num_layers: int | None = None,
        dtype=ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
        block_size: int = DEFAULT_BLOCK_SIZE,
        enable_sampling: bool = True,
    ) -> "Gemma412BModel":
        model_path = resolve_model_path(hf_model_id)
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        state_dict = _load_language_state_dict(model_path, num_layers=num_layers)
        return cls(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=state_dict,
            model_path=model_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            num_layers=num_layers,
            dtype=dtype,
            tensor_cache_path=tensor_cache_path,
            block_size=block_size,
            enable_sampling=enable_sampling,
        )

    def _load_embedding_and_lm_head(self, state_dict: dict[str, torch.Tensor]) -> None:
        embed_weight = None
        for key in ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight", "embed_tokens.weight"):
            if key in state_dict:
                embed_weight = state_dict[key]
                break
        if embed_weight is None:
            raise KeyError("missing tied embedding weight in state_dict")

        is_mesh = hasattr(self.mesh_device, "shape")
        replicate = _replicate_mapper(self.mesh_device)
        tp_suffix = f"_tp{self.mesh_config.tp}" if self.mesh_config.tp > 1 else ""
        embed_mapper = self.mesh_config.column_parallel(self.mesh_device) if self.mesh_config.tp > 1 else replicate
        lm_mapper = self.mesh_config.column_parallel(self.mesh_device) if self.mesh_config.tp > 1 else replicate
        cache_root = str(self.tensor_cache_path)
        has_bfp4 = hasattr(ttnn, "bfloat4_b")
        default_lm_head_decode_dtype = ttnn.bfloat4_b if has_bfp4 else ttnn.bfloat16
        default_lm_head_decode_fidelity = ttnn.MathFidelity.LoFi if has_bfp4 else ttnn.MathFidelity.HiFi2
        lm_head_prefill_dtype = _dtype_from_env("GEMMA4_12B_FULL_MODEL_LM_HEAD_PREFILL_DTYPE", ttnn.bfloat16)
        lm_head_decode_dtype = _dtype_from_env(
            "GEMMA4_12B_FULL_MODEL_LM_HEAD_DECODE_DTYPE",
            default_lm_head_decode_dtype,
        )
        self.lm_head_prefill_dtype = lm_head_prefill_dtype
        self.lm_head_decode_dtype = lm_head_decode_dtype

        self.embedding_weight = ttnn.as_tensor(
            embed_weight.unsqueeze(0).unsqueeze(0),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=embed_mapper if is_mesh else None,
            cache_file_name=get_cache_file_name(cache_root, f"embed_tokens.weight{tp_suffix}_bf16"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        lm_head_weight = embed_weight.transpose(0, 1).contiguous().unsqueeze(0).unsqueeze(0)
        self.lm_head_weight_prefill = ttnn.as_tensor(
            lm_head_weight,
            device=self.mesh_device,
            dtype=lm_head_prefill_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=lm_mapper if is_mesh else None,
            cache_file_name=get_cache_file_name(
                cache_root, f"lm_head.weight{tp_suffix}_{_dtype_name(lm_head_prefill_dtype)}"
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.lm_head_weight = self.lm_head_weight_prefill

        per_device_vocab = self.vocab_size // self.mesh_config.tp
        self.lm_head_decode_grid = ttnn.CoreGrid(x=8, y=5)
        self.lm_head_decode_input_memcfg = _width_sharded_mem_config(
            rows=ttnn.TILE_SIZE,
            width=self.hidden_size,
            grid=self.lm_head_decode_grid,
        )
        self.lm_head_decode_output_memcfg = _width_sharded_padded_mem_config(
            rows=ttnn.TILE_SIZE,
            width=per_device_vocab,
            grid=self.lm_head_decode_grid,
        )
        self.lm_head_decode_program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=1,
            per_core_M=1,
            per_core_N=math.ceil(per_device_vocab / (ttnn.TILE_SIZE * self.lm_head_decode_grid.num_cores)),
            fused_activation=None,
        )
        lm_head_decode_fidelity = _math_fidelity_from_env(
            "GEMMA4_12B_FULL_MODEL_LM_HEAD_DECODE_FIDELITY",
            default_lm_head_decode_fidelity
            if lm_head_decode_dtype == default_lm_head_decode_dtype
            else ttnn.MathFidelity.HiFi2,
        )
        self.lm_head_decode_fidelity = lm_head_decode_fidelity
        self.lm_head_decode_compute_config = _compute_kernel_config(
            lm_head_decode_fidelity,
            fp32_dest_acc_en=False,
        )
        self.lm_head_weight_decode = ttnn.as_tensor(
            lm_head_weight,
            device=self.mesh_device,
            dtype=lm_head_decode_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=lm_mapper if is_mesh else None,
            cache_file_name=get_cache_file_name(
                cache_root, f"lm_head.weight_dram_sharded{tp_suffix}_{_dtype_name(lm_head_decode_dtype)}"
            ),
            memory_config=_create_dram_sharded_mem_config(
                mesh_device=self.mesh_device,
                k=self.hidden_size,
                n=per_device_vocab,
            ),
        )

    def _make_sampling(self):
        per_device_vocab = _ceil_to_multiple(math.ceil(self.vocab_size / self.mesh_config.tp), ttnn.TILE_SIZE)
        per_device_vocab = 1 << (per_device_vocab - 1).bit_length()
        args = SimpleNamespace(
            vocab_size=self.vocab_size,
            padded_vocab_size=per_device_vocab * self.mesh_config.tp,
            cluster_shape=tuple(self.mesh_device.shape),
            sampling_all_gather_axis=self.mesh_config.tp_axis,
            sampling_dp=1,
            num_devices=self.mesh_device.get_num_devices(),
            max_batch_size=32,
            max_top_k=32,
            use_topk_logprobs=False,
            model_config={
                "SAMPLING_AG_CONFIG": {
                    "allow_force_argmax": True,
                    "num_links": 1,
                    "chunks_per_sync": 10,
                    "num_workers_per_link": 2,
                    "topology": ttnn.Topology.Ring,
                }
            },
        )
        sampling = SamplingGenerator(
            args=args,
            mesh_device=self.mesh_device,
            tt_ccl=get_tt_ccl(self.mesh_device),
            enable_internal_trace=False,
        )
        params = format_sampling_params(SamplingParams(temperature=1.0, top_k=1, top_p=1.0), 32)
        sampling.reset_sampling_params(params)
        return sampling

    def _to_tt_tokens(self, tokens: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            tokens.to(torch.uint32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=_replicate_mapper(self.mesh_device),
        )

    def _host_tt_tokens(self, tokens: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            tokens.to(torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=_replicate_mapper(self.mesh_device),
        )

    def _embed_tokens(self, tokens_tt: ttnn.Tensor) -> ttnn.Tensor:
        hidden_states = ttnn.embedding(tokens_tt, self.embedding_weight, dtype=ttnn.bfloat16)
        hidden_states = ttnn.mul(hidden_states, self.embed_scale)
        if len(hidden_states.shape) == 3:
            hidden_states = ttnn.unsqueeze_to_4D(hidden_states)
        if self.mesh_config.tp > 1:
            hidden_states = _all_gather_tp(
                hidden_states,
                self.mesh_config,
                self.ccl_manager,
                dim=3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)

    def _position_tensors(self, start_pos: int | torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if isinstance(start_pos, torch.Tensor):
            pos = int(start_pos.reshape(-1)[0].item())
        else:
            pos = int(start_pos)
        mapper = _replicate_mapper(self.mesh_device)
        position_idx = ttnn.from_torch(
            torch.tensor([[pos]], dtype=torch.uint32),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
            mesh_mapper=mapper,
        )
        position_idx_cache = ttnn.from_torch(
            torch.tensor([[pos]], dtype=torch.int32),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.int32,
            mesh_mapper=mapper,
        )
        return position_idx, position_idx_cache

    def _host_position_tensors(self, start_pos: int | torch.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        if isinstance(start_pos, torch.Tensor):
            pos = int(start_pos.reshape(-1)[0].item())
        else:
            pos = int(start_pos)
        mapper = _replicate_mapper(self.mesh_device)
        position_idx = ttnn.from_torch(
            torch.tensor([[pos]], dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        position_idx_cache = ttnn.from_torch(
            torch.tensor([[pos]], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=mapper,
        )
        return position_idx, position_idx_cache

    def prepare_decode_device_inputs(self, tokens: torch.Tensor, start_pos: int | torch.Tensor):
        host_inputs = (
            self._host_tt_tokens(tokens.reshape(tokens.shape[0], 1)),
            *self._host_position_tensors(start_pos),
        )
        return tuple(ttnn.to_device(tensor, device=self.mesh_device) for tensor in host_inputs)

    def update_decode_device_inputs(self, device_inputs, tokens: torch.Tensor, start_pos: int | torch.Tensor):
        host_inputs = (
            self._host_tt_tokens(tokens.reshape(tokens.shape[0], 1)),
            *self._host_position_tensors(start_pos),
        )
        for host_tensor, device_tensor in zip(host_inputs, device_inputs):
            ttnn.copy_host_to_device_tensor(host_tensor, device_tensor)
        return device_inputs

    def create_page_table(self, max_num_blocks: int | None = None) -> torch.Tensor:
        max_num_blocks = max_num_blocks or math.ceil(self.max_seq_len / self.block_size)
        return torch.arange(max_num_blocks, dtype=torch.int32).reshape(1, max_num_blocks)

    def page_table_to_device(self, page_table: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            page_table.to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=_replicate_mapper(self.mesh_device),
        )

    def create_paged_kv_cache(
        self,
        *,
        max_num_blocks: int | None = None,
        block_size: int | None = None,
        tensor_cache_path: str | Path | None = None,
    ):
        block_size = block_size or self.block_size
        max_num_blocks = max_num_blocks or math.ceil(self.max_seq_len / block_size)
        cache_root = Path(tensor_cache_path) if tensor_cache_path is not None else self.tensor_cache_path / "kv_cache"
        return [
            layer.create_paged_kv_cache(
                block_size=block_size,
                max_num_blocks=max_num_blocks,
                tensor_cache_path=cache_root / f"layer_{idx}",
            )
            for idx, layer in enumerate(self.layers)
        ]

    def reset_kv_cache(self, kv_cache) -> None:
        for layer_cache in kv_cache:
            for tensor in layer_cache:
                ttnn.mul(tensor, 0.0, output_tensor=tensor)

    def _layer_rope(self, layer_idx: int, *, is_decode: bool):
        layer_type = self.model_args.layer_types[layer_idx]
        return self.rope_caches_2d[layer_type] if is_decode else self.rope_caches[layer_type]

    def _decoder_stack(
        self,
        hidden_states,
        *,
        page_table,
        kv_cache,
        is_decode: bool,
        position_idx=None,
        position_idx_cache=None,
        token_index: int | None = None,
    ):
        for layer_idx, layer in enumerate(self.layers):
            layer_page_table = page_table[layer_idx] if isinstance(page_table, (list, tuple)) else page_table
            if is_decode:
                hidden_states = layer.decode_forward(
                    hidden_states,
                    rope_mats=self._layer_rope(layer_idx, is_decode=True),
                    page_table=layer_page_table,
                    kv_cache=kv_cache[layer_idx],
                    position_idx=position_idx,
                    position_idx_cache=position_idx_cache,
                    token_index=token_index,
                )
            else:
                hidden_states = layer.prefill_forward(
                    hidden_states,
                    rope_mats=self._layer_rope(layer_idx, is_decode=False),
                    page_table=layer_page_table,
                    kv_cache=kv_cache[layer_idx],
                )
        return hidden_states

    def _lm_head(self, hidden_states, *, is_decode: bool, gather_logits: bool) -> ttnn.Tensor:
        hidden_states = self.norm.forward(hidden_states, is_decode=is_decode)
        use_decode_matmul = is_decode or hidden_states.shape[-2] == ttnn.TILE_SIZE
        if use_decode_matmul:
            hidden_states = ttnn.to_memory_config(hidden_states, self.lm_head_decode_input_memcfg)
            logits = ttnn.linear(
                hidden_states,
                self.lm_head_weight_decode,
                dtype=ttnn.bfloat16,
                memory_config=self.lm_head_decode_output_memcfg,
                compute_kernel_config=self.lm_head_decode_compute_config,
                program_config=self.lm_head_decode_program_config,
            )
            logits = ttnn.to_memory_config(logits, ttnn.DRAM_MEMORY_CONFIG)
        else:
            logits = ttnn.linear(hidden_states, self.lm_head_weight_prefill)
        hidden_states.deallocate(True)
        if self.final_logit_softcapping and self.final_logit_softcapping > 0:
            cap = self.final_logit_softcapping
            logits = ttnn.mul(logits, 1.0 / cap)
            logits = ttnn.tanh(logits)
            logits = ttnn.mul(logits, cap)
        if gather_logits and self.mesh_config.tp > 1:
            logits = _all_gather_tp(
                logits,
                self.mesh_config,
                self.ccl_manager,
                dim=3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return logits

    def prefill_forward(
        self,
        tokens: torch.Tensor,
        *,
        page_table,
        kv_cache,
        prompt_lens: Iterable[int],
        return_all_logits: bool = False,
        return_ttnn: bool = False,
        gather_logits: bool = True,
    ):
        real_seq_len = int(list(prompt_lens)[0])
        padded_seq_len = _ceil_to_multiple(real_seq_len, ttnn.TILE_SIZE)
        if tokens.shape[1] < padded_seq_len:
            pad_token = 0
            pad = torch.full((tokens.shape[0], padded_seq_len - tokens.shape[1]), pad_token, dtype=tokens.dtype)
            tokens = torch.cat([tokens, pad], dim=1)
        elif tokens.shape[1] > padded_seq_len:
            tokens = tokens[:, :padded_seq_len]

        tokens_tt = self._to_tt_tokens(tokens)
        hidden_states = self._embed_tokens(tokens_tt)
        hidden_states = self._decoder_stack(
            hidden_states,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=False,
        )

        if not return_all_logits:
            tile_start = ((real_seq_len - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            hidden_states = ttnn.slice(
                hidden_states,
                (0, 0, tile_start, 0),
                (1, 1, tile_start + ttnn.TILE_SIZE, hidden_states.shape[-1]),
            )

        logits = self._lm_head(hidden_states, is_decode=False, gather_logits=gather_logits)
        if return_ttnn:
            return logits
        logits_torch = _first_device_torch(logits, self.mesh_device).float()
        if return_all_logits:
            return logits_torch[:, :, :real_seq_len, : self.vocab_size].reshape(1, real_seq_len, self.vocab_size)
        offset = (real_seq_len - 1) % ttnn.TILE_SIZE
        return logits_torch[:, :, offset : offset + 1, : self.vocab_size].reshape(1, 1, self.vocab_size)

    def decode_forward(
        self,
        tokens: torch.Tensor,
        start_pos: int | torch.Tensor,
        *,
        page_table,
        kv_cache,
        sample_on_device: bool = False,
        return_ttnn: bool = False,
    ):
        tokens_tt = self._to_tt_tokens(tokens.reshape(tokens.shape[0], 1))
        hidden_states = self._embed_tokens(tokens_tt)
        position_idx, position_idx_cache = self._position_tensors(start_pos)
        pos_value = int(start_pos.reshape(-1)[0].item()) if isinstance(start_pos, torch.Tensor) else int(start_pos)
        hidden_states = self._decoder_stack(
            hidden_states,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
            position_idx=position_idx,
            position_idx_cache=position_idx_cache,
            token_index=pos_value,
        )
        logits = self._lm_head(hidden_states, is_decode=True, gather_logits=not sample_on_device)
        if sample_on_device:
            if self.sampling is None:
                raise RuntimeError("on-device sampling requested but sampling is not initialized")
            batch_dim = logits.shape[2]
            if batch_dim < 32:
                logits = ttnn.pad(logits, padding=[(0, 0), (0, 0), (0, 32 - batch_dim), (0, 0)], value=0.0)
            sampled_tokens, log_probs = self.sampling.sample(logits, enable_trace=False)
            if return_ttnn:
                return sampled_tokens, log_probs
            return _first_device_torch(sampled_tokens, self.mesh_device).reshape(-1)[: tokens.shape[0]].to(torch.long)
        if return_ttnn:
            return logits
        logits_torch = _first_device_torch(logits, self.mesh_device).float()
        return logits_torch[:, :, : tokens.shape[0], : self.vocab_size].reshape(tokens.shape[0], self.vocab_size)

    def decode_forward_device_inputs(
        self,
        tokens_tt: ttnn.Tensor,
        position_idx: ttnn.Tensor,
        position_idx_cache: ttnn.Tensor,
        *,
        page_table,
        kv_cache,
        sample_on_device: bool = False,
        return_ttnn: bool = True,
        token_index: int | None = None,
    ):
        hidden_states = self._embed_tokens(tokens_tt)
        hidden_states = self._decoder_stack(
            hidden_states,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
            position_idx=position_idx,
            position_idx_cache=position_idx_cache,
            token_index=0 if token_index is None else token_index,
        )
        logits = self._lm_head(hidden_states, is_decode=True, gather_logits=not sample_on_device)
        if sample_on_device:
            if self.sampling is None:
                raise RuntimeError("on-device sampling requested but sampling is not initialized")
            batch_dim = logits.shape[2]
            if batch_dim < 32:
                logits = ttnn.pad(logits, padding=[(0, 0), (0, 0), (0, 32 - batch_dim), (0, 0)], value=0.0)
            sampled_tokens, log_probs = self.sampling.sample(logits, enable_trace=False)
            if return_ttnn:
                return sampled_tokens, log_probs
            batch = int(tokens_tt.shape[0])
            return _first_device_torch(sampled_tokens, self.mesh_device).reshape(-1)[:batch].to(torch.long)
        if return_ttnn:
            return logits
        batch = int(tokens_tt.shape[0])
        logits_torch = _first_device_torch(logits, self.mesh_device).float()
        return logits_torch[:, :, :batch, : self.vocab_size].reshape(batch, self.vocab_size)


def build_model(
    *,
    mesh_device,
    hf_model_id: str | Path = SUPPORTED_HF_MODEL_ID,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    max_batch_size: int = 1,
    num_layers: int | None = None,
    dtype=ttnn.bfloat16,
    tensor_cache_path: str | Path | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    enable_sampling: bool = True,
) -> Gemma412BModel:
    return Gemma412BModel.from_pretrained(
        mesh_device=mesh_device,
        hf_model_id=hf_model_id,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        num_layers=num_layers,
        dtype=dtype,
        tensor_cache_path=tensor_cache_path,
        block_size=block_size,
        enable_sampling=enable_sampling,
    )


__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_MAX_SEQ_LEN",
    "Gemma412BModel",
    "SUPPORTED_HF_MODEL_ID",
    "TARGET_MESH_SHAPE",
    "build_model",
    "resolve_model_path",
]
