# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace

import torch

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.functional_decoder import (
    BF16,
    DRAM,
    TILE,
    FunctionalDecoder,
    LlamaDecoderConfig,
    _lookup,
    _require_config,
    _to_tt_hidden,
    _to_tt_weight,
)
from models.common.models.llama3_8b.model import TransformerBlock1D
from models.common.modules.attention.attention_1d import Attention1D
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.mlp.mlp_1d import MLP1D, _create_dram_sharded_mem_config
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D
from models.common.tensor_utils import get_rot_transformation_mat, pad_dim_to_size
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.common import Mode, PagedAttentionConfig
from models.tt_transformers.tt.model_config import (
    DecodersPrecision,
    MathFidelitySetting,
    ModelArgs,
    ModelOptimizations,
    OpGroup,
    PrecisionSetting,
    TensorGroup,
)

DEFAULT_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


@dataclass(frozen=True)
class OptimizedDecoderEvidence:
    qkv_packed: bool
    decode_uses_common_optimized_block: bool
    paged_kv_block_size: int
    paged_kv_max_num_blocks: int
    kv_cache_dtype: str


@contextmanager
def _repo_local_model_args_env():
    old_hf_model = os.environ.get("HF_MODEL")
    old_ci = os.environ.get("CI")
    os.environ.setdefault("HF_MODEL", DEFAULT_HF_MODEL)
    os.environ.setdefault("CI", "true")
    try:
        yield
    finally:
        if old_hf_model is None:
            os.environ.pop("HF_MODEL", None)
        else:
            os.environ["HF_MODEL"] = old_hf_model
        if old_ci is None:
            os.environ.pop("CI", None)
        else:
            os.environ["CI"] = old_ci


class PackedGateUpMLP1D(MLP1D):
    """Env-gated candidate that packs gate/up into one decode matmul.

    This is deliberately local to the optimized autoport so the stage can
    measure whether a packed same-input projection beats the common MLP1D path
    without changing shared model code.
    """

    @classmethod
    def from_mlp(cls, mlp: MLP1D, *, mesh_device, model_args, state_dict, layer_idx: int, state_dict_prefix: str):
        num_devices = mesh_device.get_num_devices()
        dram_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1),
                )
            }
        )
        packed_source = torch.cat(
            [
                torch.transpose(state_dict[f"{state_dict_prefix}.w1.weight"], -2, -1),
                torch.transpose(state_dict[f"{state_dict_prefix}.w3.weight"], -2, -1),
            ],
            dim=-1,
        )
        packed_source = pad_dim_to_size(packed_source, dim=-1, size=2 * model_args.hidden_dim)
        packed_weight = LazyWeight(
            source=packed_source,
            dtype=mlp.config.w1_w3_dtype,
            device=mesh_device,
            mesh_mapper_config=ttnn.MeshMapperConfig(
                placements=[ttnn.PlacementShard(-1)],
                mesh_shape_override=ttnn.MeshShape([num_devices]),
            ),
            layout=ttnn.TILE_LAYOUT,
            memory_config=_create_dram_sharded_mem_config(
                k=model_args.dim,
                n=2 * model_args.hidden_dim // num_devices,
                dram_grid=dram_grid,
                tile_size=ttnn.TILE_SIZE,
                dram_cores=dram_size.x,
            ),
        )
        instance = object.__new__(cls)
        super(MLP1D, instance).__init__()
        instance.config = replace(
            mlp.config,
            w1=packed_weight,
            decode_w1_w3_prg_config=model_args.dram_matmul_config(
                m=model_args.tile_padded_batch_rows,
                k=model_args.dim,
                n=2 * model_args.hidden_dim // model_args.cluster_shape[1],
            ),
        )
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self):
        if self._device_weights_loaded:
            return
        assert self.config.is_resolved(), "config must be resolved before loading device weights!"
        self.w1 = self.config.w1.get_device_weight()
        self.w2 = self.config.w2.get_device_weight()
        self._device_weights_loaded = True

    def decode_forward(self, x: ttnn.Tensor | LazyWeight) -> ttnn.Tensor:
        self.load_device_weights()
        from models.common.modules.mlp.mlp_1d import _load_input_device_tensor

        x = _load_input_device_tensor(x, self.config, mode="decode")
        cfg = self.config
        hidden_dim = cfg.hidden_dim

        packed = ttnn.linear(
            x,
            self.w1,
            dtype=cfg.linear_dtype,
            core_grid=None,
            compute_kernel_config=cfg.decode_ff1_3_compute_kernel_cfg,
            program_config=cfg.decode_w1_w3_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(x)
        gate = ttnn.slice(
            packed,
            [0, 0, 0, 0],
            [packed.shape[0], packed.shape[1], packed.shape[2], hidden_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        up = ttnn.slice(
            packed,
            [0, 0, 0, hidden_dim],
            [packed.shape[0], packed.shape[1], packed.shape[2], 2 * hidden_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.deallocate(packed)
        w2_in = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[cfg.mlp_activation_type],
            dtype=cfg.mul_dtype,
            memory_config=gate.memory_config(),
        )
        w2_in = ttnn.to_memory_config(w2_in, cfg.decode_mlp2_input_memcfg)
        ttnn.deallocate(up)
        ttnn.deallocate(gate)

        w2_out = ttnn.linear(
            w2_in,
            self.w2,
            compute_kernel_config=cfg.decode_ff2_compute_kernel_cfg,
            dtype=cfg.linear_dtype,
            program_config=cfg.decode_w2_prg_config,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            core_grid=None,
        )
        ttnn.deallocate(w2_in)
        w2_out_reduced = self._all_reduce_decode(w2_out)
        original_shape = w2_out_reduced.shape
        w2_out_reduced = ttnn.reshape(
            w2_out_reduced, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )
        return ttnn.to_memory_config(w2_out_reduced, cfg.decode_residual_memcfg)


def _adapt_state_dict_for_common_modules(
    state_dict: dict[str, torch.Tensor], layer_idx: int
) -> dict[str, torch.Tensor]:
    return {
        f"layers.{layer_idx}.attention_norm.weight": _lookup(state_dict, layer_idx, "input_layernorm.weight"),
        f"layers.{layer_idx}.ffn_norm.weight": _lookup(state_dict, layer_idx, "post_attention_layernorm.weight"),
        f"layers.{layer_idx}.attention.wq.weight": _lookup(state_dict, layer_idx, "self_attn.q_proj.weight"),
        f"layers.{layer_idx}.attention.wk.weight": _lookup(state_dict, layer_idx, "self_attn.k_proj.weight"),
        f"layers.{layer_idx}.attention.wv.weight": _lookup(state_dict, layer_idx, "self_attn.v_proj.weight"),
        f"layers.{layer_idx}.attention.wo.weight": _lookup(state_dict, layer_idx, "self_attn.o_proj.weight"),
        f"layers.{layer_idx}.feed_forward.w1.weight": _lookup(state_dict, layer_idx, "mlp.gate_proj.weight"),
        f"layers.{layer_idx}.feed_forward.w2.weight": _lookup(state_dict, layer_idx, "mlp.down_proj.weight"),
        f"layers.{layer_idx}.feed_forward.w3.weight": _lookup(state_dict, layer_idx, "mlp.up_proj.weight"),
    }


def _decode_transform_mats(mesh_device, batch: int) -> dict[str, ttnn.Tensor]:
    compute_grid = mesh_device.compute_with_storage_grid_size()
    batch_grid = ttnn.num_cores_to_corerangeset(batch, compute_grid, row_wise=True)
    decode_mem_config = ttnn.create_sharded_memory_config(
        shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    decode_mat = get_rot_transformation_mat().repeat(1, 1, batch, 1)
    prefill_mat = torch.zeros(1, 1, 128, 128)
    prefill_mat[:, :, : ttnn.TILE_SIZE, : ttnn.TILE_SIZE] = get_rot_transformation_mat()
    return {
        "decode": ttnn.from_torch(
            decode_mat,
            device=mesh_device,
            layout=TILE,
            dtype=BF16,
            memory_config=decode_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
        "prefill": ttnn.from_torch(
            prefill_mat,
            device=mesh_device,
            layout=TILE,
            dtype=BF16,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        ),
    }


class OptimizedDecoder(FunctionalDecoder):
    """Optimized single Llama-3.1 decoder layer.

    Prefill keeps the functional layer math but uses explicit optimized TTNN
    operator configs. Decode uses the common optimized Llama 1D block so the
    runtime path exercises packed QKV, sharded decode matmuls, paged KV cache
    update, and SDPA decode instead of a functional fallback.
    """

    def __init__(
        self,
        *,
        config: LlamaDecoderConfig,
        layer_idx: int,
        mesh_device,
        weights: dict,
        batch: int,
        optimized_block: TransformerBlock1D,
        model_args: ModelArgs,
        paged_attention_config: PagedAttentionConfig,
    ):
        super().__init__(config=config, layer_idx=layer_idx, mesh_device=mesh_device, weights=weights, batch=batch)
        self.optimized_block = optimized_block
        self.model_args = model_args
        self.paged_attention_config = paged_attention_config
        self.decode_input_memcfg = model_args.get_model_config()["DECODE_RESIDUAL_MEMCFG"]
        self.evidence = OptimizedDecoderEvidence(
            qkv_packed=True,
            decode_uses_common_optimized_block=True,
            paged_kv_block_size=paged_attention_config.block_size,
            paged_kv_max_num_blocks=paged_attention_config.max_num_blocks,
            kv_cache_dtype=str(optimized_block.attention.config.kv_cache_dtype),
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx,
        mesh_device,
        batch=32,
        max_seq_len=128,
        page_block_size=32,
        decode_mlp_num_cores=None,
        **kwargs,
    ):
        del kwargs
        config = _require_config(hf_config)
        q_proj = _lookup(state_dict, layer_idx, "self_attn.q_proj.weight")
        k_proj = _lookup(state_dict, layer_idx, "self_attn.k_proj.weight")
        v_proj = _lookup(state_dict, layer_idx, "self_attn.v_proj.weight")
        weights = {
            "input_layernorm": _to_tt_weight(_lookup(state_dict, layer_idx, "input_layernorm.weight"), mesh_device),
            "post_attention_layernorm": _to_tt_weight(
                _lookup(state_dict, layer_idx, "post_attention_layernorm.weight"), mesh_device
            ),
            # Keep functional prefill's public batch/sequence contract, but use BFP8_B
            # projection weights to reduce DRAM traffic on the measured matmuls.
            "qkv": cls._to_optimized_projection_weight(torch.cat([q_proj.T, v_proj.T, k_proj.T], dim=1), mesh_device),
            "o_proj": cls._to_optimized_projection_weight(
                _lookup(state_dict, layer_idx, "self_attn.o_proj.weight").T, mesh_device
            ),
            "gate_proj": cls._to_optimized_projection_weight(
                _lookup(state_dict, layer_idx, "mlp.gate_proj.weight").T, mesh_device
            ),
            "up_proj": cls._to_optimized_projection_weight(
                _lookup(state_dict, layer_idx, "mlp.up_proj.weight").T, mesh_device
            ),
            "down_proj": cls._to_optimized_projection_weight(
                _lookup(state_dict, layer_idx, "mlp.down_proj.weight").T, mesh_device
            ),
        }
        max_num_blocks = int(batch) * math.ceil(int(max_seq_len) / int(page_block_size))
        paged_attention_config = PagedAttentionConfig(block_size=int(page_block_size), max_num_blocks=max_num_blocks)

        with _repo_local_model_args_env():
            model_args = ModelArgs(
                mesh_device,
                instruct=True,
                dummy_weights=True,
                max_batch_size=int(batch),
                max_seq_len=int(max_seq_len),
                cache_hf=False,
                optimizations=lambda args: cls._decoder_precision_policy(args.n_layers, args.model_name),
            )
        model_args.use_qk_fused = False

        adapted = _adapt_state_dict_for_common_modules(state_dict, layer_idx)
        weight_cache_path = None
        transform_mats = _decode_transform_mats(mesh_device, int(batch))
        model_config = model_args.get_model_config()
        model_config["DECODE_RESIDUAL_MEMCFG"] = model_args.get_residual_mem_config(Mode.DECODE)
        decode_mlp_num_cores = cls._decode_mlp_num_cores(decode_mlp_num_cores)
        attn_norm_cfg = model_args.get_norm_config("attn", Mode.DECODE)
        ff_norm_cfg = model_args.get_norm_config("ff", Mode.DECODE)

        attention_norm = RMSNorm1D.from_model_args(
            mesh_device=mesh_device,
            tt_ccl=None,
            args=model_args,
            state_dict=adapted,
            weight_cache_path=weight_cache_path,
            layer_num=layer_idx,
            weight_key="attention_norm",
            sharded_program_config=attn_norm_cfg.get("sharded_program_config"),
            sharded_output_config=attn_norm_cfg.get("sharded_output_config"),
        )
        attention = Attention1D.from_model_args(
            mesh_device=mesh_device,
            tt_ccl=None,
            args=model_args,
            state_dict=adapted,
            weight_cache_path=weight_cache_path,
            layer_num=layer_idx,
            transformation_mats=transform_mats,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=False,
        )
        ff_norm = RMSNorm1D.from_model_args(
            mesh_device=mesh_device,
            tt_ccl=None,
            args=model_args,
            state_dict=adapted,
            weight_cache_path=weight_cache_path,
            layer_num=layer_idx,
            weight_key="ffn_norm",
            sharded_program_config=ff_norm_cfg.get("sharded_program_config"),
            sharded_output_config=ff_norm_cfg.get("sharded_output_config"),
        )
        mlp = MLP1D.from_model_args(
            mesh_device=mesh_device,
            tt_ccl=None,
            args=model_args,
            state_dict=adapted,
            weight_cache_path=weight_cache_path,
            layer_num=layer_idx,
            dtype=None,
            model_config=model_config,
        )
        if cls._use_packed_gate_up_candidate():
            mlp = PackedGateUpMLP1D.from_mlp(
                mlp,
                mesh_device=mesh_device,
                model_args=model_args,
                state_dict=adapted,
                layer_idx=layer_idx,
                state_dict_prefix=model_args.get_state_dict_prefix("MLP", layer_idx),
            )
        if decode_mlp_num_cores is not None:
            mlp.config.decode_w1_w3_prg_config = model_args.dram_matmul_config(
                m=model_args.tile_padded_batch_rows,
                k=model_args.dim,
                n=(
                    2 * model_args.hidden_dim // model_args.cluster_shape[1]
                    if isinstance(mlp, PackedGateUpMLP1D)
                    else model_args.hidden_dim // model_args.cluster_shape[1]
                ),
                num_cores=int(decode_mlp_num_cores),
            )
        block = TransformerBlock1D(
            attention_norm=attention_norm,
            attention=attention,
            ff_norm=ff_norm,
            feed_forward=mlp,
            decode_residual_memcfg=model_config["DECODE_RESIDUAL_MEMCFG"],
            prefill_residual_memcfg=DRAM,
        )
        return cls(
            config=config,
            layer_idx=layer_idx,
            mesh_device=mesh_device,
            weights=weights,
            batch=int(batch),
            optimized_block=block,
            model_args=model_args,
            paged_attention_config=paged_attention_config,
        )

    @staticmethod
    def _decode_mlp_num_cores(value):
        if value is None:
            value = os.environ.get("TT_AUTOOPT_LLAMA_DECODE_MLP_NUM_CORES")
        if value in (None, ""):
            return None
        value = int(value)
        if value <= 0:
            raise ValueError("decode_mlp_num_cores must be positive")
        return value

    @staticmethod
    def _use_packed_gate_up_candidate() -> bool:
        return os.environ.get("TT_AUTOOPT_LLAMA_PACKED_GATE_UP", "1").lower() not in ("0", "false", "no")

    @staticmethod
    def _use_prefill_l1_activations() -> bool:
        return os.environ.get("TT_AUTOOPT_LLAMA_PREFILL_L1_ACTIVATIONS", "1").lower() not in ("0", "false", "no")

    @staticmethod
    def _decoder_precision_policy(num_decoders: int, model_name: str) -> DecodersPrecision:
        policy = DecodersPrecision.performance(num_decoders, model_name)
        candidate = os.environ.get("TT_AUTOOPT_LLAMA_PRECISION_CANDIDATE", "attention_ff2_bfp4")
        if candidate == "repo_performance":
            return policy

        tensor_precision = {TensorGroup.FF1_FF3: PrecisionSetting.BFP4}
        op_fidelity = {OpGroup.LI_FF1_FF3: MathFidelitySetting.LOFI}

        if candidate in ("ff2_bfp4", "attention_ff2_bfp4"):
            tensor_precision[TensorGroup.FF2] = PrecisionSetting.BFP4
            op_fidelity[OpGroup.LI_FF2] = MathFidelitySetting.LOFI
        if candidate in ("attention_bfp4", "attention_ff2_bfp4"):
            tensor_precision[TensorGroup.WQKV] = PrecisionSetting.BFP4
            tensor_precision[TensorGroup.WO] = PrecisionSetting.BFP4
            op_fidelity[OpGroup.LI_QKV_DECODE] = MathFidelitySetting.LOFI
            op_fidelity[OpGroup.LI_O_DECODE] = MathFidelitySetting.LOFI
            op_fidelity[OpGroup.LI_QKV_PREFILL] = MathFidelitySetting.LOFI
            op_fidelity[OpGroup.LI_O_PREFILL] = MathFidelitySetting.LOFI
        if candidate not in ("ff2_bfp4", "attention_bfp4", "attention_ff2_bfp4"):
            raise ValueError(f"unknown precision candidate {candidate!r}")

        policy.set_decoder_conf(
            0,
            ModelOptimizations(
                {
                    "TensorPrecision": tensor_precision,
                    "OpFidelity": op_fidelity,
                }
            ),
        )
        return policy

    @staticmethod
    def _to_optimized_projection_weight(tensor: torch.Tensor, mesh_device) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor.detach().to(torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=TILE,
            device=mesh_device,
            memory_config=DRAM,
        )

    @staticmethod
    def prepare_decode_inputs(hidden_states: torch.Tensor, mesh_device, memory_config=None) -> ttnn.Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError(f"decode hidden_states must be [batch, 1, hidden], got {tuple(hidden_states.shape)}")
        tt_hidden = ttnn.from_torch(
            hidden_states.transpose(0, 1).unsqueeze(0).to(torch.bfloat16),
            dtype=BF16,
            layout=TILE,
            device=mesh_device,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        if memory_config is not None:
            tt_hidden = ttnn.to_memory_config(tt_hidden, memory_config)
        return tt_hidden

    @staticmethod
    def prepare_decode_positions(position_ids: torch.Tensor, mesh_device) -> ttnn.Tensor:
        return ttnn.from_torch(
            position_ids.to(torch.int32),
            dtype=ttnn.int32,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    @staticmethod
    def prepare_page_table(page_table: torch.Tensor, mesh_device) -> ttnn.Tensor:
        return ttnn.from_torch(
            page_table.to(torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    @staticmethod
    def prepare_decode_rope(position_cos: torch.Tensor, position_sin: torch.Tensor, mesh_device):
        if position_cos.ndim != 3 or position_sin.ndim != 3:
            raise ValueError("decode RoPE tensors must be [batch, 1, head_dim]")
        batch = position_cos.shape[0]
        pad = nearest_32(batch) - batch
        if pad:
            position_cos = torch.nn.functional.pad(position_cos, (0, 0, 0, 0, 0, pad))
            position_sin = torch.nn.functional.pad(position_sin, (0, 0, 0, 0, 0, pad))
        position_cos = position_cos.unsqueeze(0)
        position_sin = position_sin.unsqueeze(0)
        cos = _to_tt_hidden(position_cos.to(torch.bfloat16), mesh_device)
        sin = _to_tt_hidden(position_sin.to(torch.bfloat16), mesh_device)
        batch_grid = ttnn.num_cores_to_corerangeset(batch, mesh_device.compute_with_storage_grid_size(), row_wise=True)
        mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, position_cos.shape[-1]),
            core_grid=batch_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return ttnn.interleaved_to_sharded(cos, mem_config), ttnn.interleaved_to_sharded(sin, mem_config)

    @staticmethod
    def build_contiguous_page_table(batch: int, max_seq_len: int, block_size: int) -> torch.Tensor:
        blocks_per_user = math.ceil(max_seq_len / block_size)
        page_table = torch.zeros(batch, blocks_per_user, dtype=torch.int32)
        for user in range(batch):
            start = user * blocks_per_user
            page_table[user] = torch.arange(start, start + blocks_per_user, dtype=torch.int32)
        return page_table

    def prefill_forward(self, hidden_states, *, position_cos, position_sin, attn_mask=None):
        cfg = self.config
        batch = hidden_states.shape[1]
        seq_len = hidden_states.shape[2]
        if hidden_states.shape[0] != 1 or hidden_states.shape[3] != cfg.hidden_size:
            raise ValueError("hidden_states must have shape [1, batch, seq, hidden_size]")
        if batch != self.batch:
            raise ValueError(f"runtime batch {batch} does not match configured batch {self.batch}")

        activation_memcfg = ttnn.L1_MEMORY_CONFIG if self._use_prefill_l1_activations() else DRAM

        residual = hidden_states
        normed = ttnn.rms_norm(
            hidden_states,
            epsilon=cfg.rms_norm_eps,
            weight=self.weights["input_layernorm"],
            memory_config=activation_memcfg,
        )
        qkv = ttnn.matmul(normed, self.weights["qkv"], dtype=BF16, memory_config=DRAM)

        kv_width = cfg.num_key_value_heads * cfg.head_dim
        q_width = cfg.num_attention_heads * cfg.head_dim
        query = ttnn.slice(qkv, [0, 0, 0, 0], [1, batch, seq_len, q_width], [1, 1, 1, 1], memory_config=DRAM)
        value = ttnn.slice(
            qkv,
            [0, 0, 0, q_width],
            [1, batch, seq_len, q_width + kv_width],
            [1, 1, 1, 1],
            memory_config=DRAM,
        )
        key = ttnn.slice(
            qkv,
            [0, 0, 0, q_width + kv_width],
            [1, batch, seq_len, q_width + kv_width + kv_width],
            [1, 1, 1, 1],
            memory_config=DRAM,
        )

        query = ttnn.reshape(query, [batch, seq_len, cfg.num_attention_heads, cfg.head_dim])
        key = ttnn.reshape(key, [batch, seq_len, cfg.num_key_value_heads, cfg.head_dim])
        value = ttnn.reshape(value, [batch, seq_len, cfg.num_key_value_heads, cfg.head_dim])
        query = ttnn.permute(query, [0, 2, 1, 3], memory_config=DRAM)
        key = ttnn.permute(key, [0, 2, 1, 3], memory_config=DRAM)
        value = ttnn.permute(value, [0, 2, 1, 3], memory_config=DRAM)

        query = self._apply_rotary(query, position_cos, position_sin, cfg.num_attention_heads, seq_len)
        key = self._apply_rotary(key, position_cos, position_sin, cfg.num_key_value_heads, seq_len)

        if attn_mask is None:
            attn = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=True,
                scale=self.scale,
                memory_config=DRAM,
            )
        else:
            attn = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_mask,
                is_causal=False,
                scale=self.scale,
                memory_config=DRAM,
            )

        attn = ttnn.permute(attn, [0, 2, 1, 3], memory_config=DRAM)
        attn = ttnn.reshape(attn, [1, batch, seq_len, cfg.hidden_size])
        attn = ttnn.matmul(attn, self.weights["o_proj"], dtype=BF16, memory_config=DRAM)
        hidden = ttnn.add(attn, residual, dtype=BF16, memory_config=DRAM)

        mlp_input = ttnn.rms_norm(
            hidden,
            epsilon=cfg.rms_norm_eps,
            weight=self.weights["post_attention_layernorm"],
            memory_config=activation_memcfg,
        )
        gate = ttnn.matmul(mlp_input, self.weights["gate_proj"], dtype=BF16, memory_config=DRAM)
        up = ttnn.matmul(mlp_input, self.weights["up_proj"], dtype=BF16, memory_config=DRAM)
        gated = ttnn.multiply(ttnn.silu(gate, memory_config=activation_memcfg), up, memory_config=activation_memcfg)
        mlp = ttnn.matmul(gated, self.weights["down_proj"], dtype=BF16, memory_config=DRAM)
        return ttnn.add(mlp, hidden, dtype=BF16, memory_config=DRAM)

    def decode_forward(self, hidden_states, *, current_pos, rot_mats, page_table):
        hidden_states = ttnn.to_memory_config(hidden_states, self.decode_input_memcfg)
        return self.optimized_block.decode_forward(hidden_states, current_pos, rot_mats, page_table)
