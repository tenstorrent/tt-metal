# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTTv2 Llama 3.1-8B Transformer model and executors.

Model:
    Llama3Transformer1D — pure forward methods, no input/output processing

Executors (thin wrappers around engines in models/common/models/executor.py):
    EagerLlamaExecutor  — direct execution
    TracedLlamaExecutor — traced execution with capture/replay

Architecture:
    Llama3Transformer1D (1D only — non-TG)
    ├── Embedding1D
    ├── RotarySetup1D
    ├── TransformerBlock1D × n_layers
    │   ├── RMSNorm1D  (attention_norm)
    │   ├── Attention1D
    │   ├── RMSNorm1D  (ff_norm)
    │   └── MLP1D
    ├── RMSNorm1D  (final norm)
    ├── LMHead1D
    └── Sampling1D (optional)

Loop policy functions (run_teacher_forcing, run_perf_benchmark) are in
models/common/models/executor.py.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.models.executor import EagerLLMExecutor, TracedLLMExecutor
from models.common.models.llama3_8b.rope import compute_gather_cos_sin, rope_scaling_model_factory
from models.common.modules.attention.attention_1d import Attention1D, Attention1DConfig
from models.common.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig, _compute_kernel_config_hifi2
from models.common.modules.mlp.mlp_1d import MLP1D, MLP1DConfig, _create_dram_sharded_mem_config
from models.common.modules.rmsnorm.rmsnorm_1d import SHARD_HEIGHT, RMSNorm1D, RMSNorm1DConfig
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.modules.tt_ccl import TT_CCL, default_topology, get_tt_ccl
from models.common.tensor_utils import TILE_SIZE, pad_dim_to_size

# =============================================================================
# TransformerBlock1D
# =============================================================================


@dataclass
class TransformerBlock1DConfig:
    attention_norm_config: RMSNorm1DConfig
    attention_config: Attention1DConfig
    ff_norm_config: RMSNorm1DConfig
    mlp_config: MLP1DConfig

    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None
    activation_dtype: ttnn.DataType | None = None


class TransformerBlock1D(LightweightModule):
    """Single transformer block for 1D topologies (N150, N300, T3K).

    Happy path (takes pre-built sub-modules):
        block = TransformerBlock1D(attn_norm, attention, ff_norm, mlp)

    Power-user path (builds from config):
        block = TransformerBlock1D.from_config(config)
    """

    def __init__(
        self,
        attention_norm: RMSNorm1D,
        attention: Attention1D,
        ff_norm: RMSNorm1D,
        feed_forward: MLP1D,
        decode_residual_memcfg: ttnn.MemoryConfig | None = None,
        prefill_residual_memcfg: ttnn.MemoryConfig | None = None,
        activation_dtype: ttnn.DataType | None = None,
    ):
        super().__init__()
        self.attention_norm = attention_norm
        self.attention = attention
        self.ff_norm = ff_norm
        self.feed_forward = feed_forward
        self.decode_residual_memcfg = decode_residual_memcfg
        self.prefill_residual_memcfg = prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtype = activation_dtype

    @classmethod
    def from_config(cls, config: TransformerBlock1DConfig):
        return cls(
            attention_norm=RMSNorm1D.from_config(config.attention_norm_config),
            attention=Attention1D.from_config(config.attention_config),
            ff_norm=RMSNorm1D.from_config(config.ff_norm_config),
            feed_forward=MLP1D.from_config(config.mlp_config),
            decode_residual_memcfg=config.decode_residual_memcfg,
            prefill_residual_memcfg=config.prefill_residual_memcfg,
            activation_dtype=config.activation_dtype,
        )

    def decode_forward(self, x: ttnn.Tensor, current_pos, rot_mats, page_table) -> ttnn.Tensor:
        residual = x

        x = _all_gather_rmsnorm_tensor(
            self.attention_norm, x, memory_config=self.attention_norm.config.decode_memory_config
        )
        attn_in = self.attention_norm.decode_forward(x)
        attn_out = self.attention.decode_forward(attn_in, current_pos, rot_mats, page_table=page_table)
        attn_out = ttnn.to_memory_config(attn_out, self.decode_residual_memcfg)

        hidden_states = ttnn.add(residual, attn_out, memory_config=self.decode_residual_memcfg)
        residual = hidden_states

        hidden_states = _all_gather_rmsnorm_tensor(
            self.ff_norm, hidden_states, memory_config=self.ff_norm.config.decode_memory_config
        )
        hidden_states = self.ff_norm.decode_forward(hidden_states)
        ttnn.deallocate(attn_out)
        hidden_states = self.feed_forward.decode_forward(hidden_states)

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=self.decode_residual_memcfg,
            dtype=self.activation_dtype or ttnn.bfloat16,
        )
        return out

    def prefill_forward(
        self, x: ttnn.Tensor, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx
    ) -> ttnn.Tensor:
        residual = x

        attn_in = self.attention_norm.prefill_forward(x)
        attn_in = _all_gather_rmsnorm_tensor(self.attention_norm, attn_in)
        attn_out = self.attention.prefill_forward(
            attn_in,
            rot_mats,
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        attn_out = ttnn.to_memory_config(attn_out, self.prefill_residual_memcfg)

        hidden_states = ttnn.add(residual, attn_out, memory_config=self.prefill_residual_memcfg)
        residual = hidden_states
        x.deallocate(True)

        hidden_states = self.ff_norm.prefill_forward(hidden_states)
        hidden_states = _all_gather_rmsnorm_tensor(self.ff_norm, hidden_states)
        ttnn.deallocate(attn_out)
        hidden_states = self.feed_forward.prefill_forward(hidden_states)

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=self.prefill_residual_memcfg,
            dtype=self.activation_dtype or ttnn.bfloat16,
        )
        return out

    def forward(
        self,
        x,
        current_pos=None,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
    ):
        if mode == "prefill":
            return self.prefill_forward(x, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx)
        return self.decode_forward(x, current_pos, rot_mats, page_table)


# =============================================================================
# Llama3Transformer1D
# =============================================================================


@dataclass
class Llama31_8BPagedAttentionConfig:
    block_size: int
    max_num_blocks: int


@dataclass
class Llama3Transformer1DConfig:
    """Full model config. Build via from_hf_config() or construct manually."""

    n_layers: int
    vocab_size: int
    max_batch_size: int
    max_seq_len: int
    num_devices: int
    mesh_device: ttnn.MeshDevice

    # Sub-module configs
    embedding_config: Embedding1DConfig
    rope_config: Rope1DConfig
    block_configs: list[TransformerBlock1DConfig]
    norm_config: RMSNorm1DConfig
    lm_head_config: LMHead1DConfig
    sampling_config: Sampling1DConfig | None = None

    # Model-level memory configs
    decode_residual_memcfg: ttnn.MemoryConfig | None = None
    prefill_residual_memcfg: ttnn.MemoryConfig | None = None

    # Per-layer activation dtypes (from decoders_optimizations)
    activation_dtypes: list[ttnn.DataType | None] = field(default_factory=list)

    # CCL
    tt_ccl: TT_CCL | None = None

    # Weight cache path (for from_hf_config)
    cache_path: "str | None" = None


def build_llama3_transformer_1d_config(
    *,
    mesh_device,
    args,
    state_dict,
    weight_cache_path,
    dtype=None,
    paged_attention_config=None,
    use_paged_kv_cache=None,
) -> Llama3Transformer1DConfig:
    """Translate Llama runtime args into explicit TTTv2 module configs."""
    if args.is_galaxy:
        raise ValueError("Llama3Transformer1D only supports 1D mesh topologies.")

    if use_paged_kv_cache is None:
        use_paged_kv_cache = paged_attention_config is not None

    num_devices = mesh_device.get_num_devices()
    tt_ccl_inst = get_tt_ccl(mesh_device) if num_devices > 1 else None
    model_config = args.get_model_config()
    model_config["DECODE_RESIDUAL_MEMCFG"] = args.get_decode_residual_mem_config()
    weight_cache_path = Path(weight_cache_path) if weight_cache_path else None
    embedding_cache_path = args.weight_cache_path(dtype or ttnn.bfloat8_b)

    def mesh_shard(dim: int) -> ttnn.MeshMapperConfig:
        return ttnn.MeshMapperConfig(
            placements=[ttnn.PlacementShard(dim)],
            mesh_shape_override=ttnn.MeshShape([num_devices]),
        )

    def cache_path_for(base, *parts):
        if base is None or getattr(args, "dummy_weights", False):
            return None
        return Path(base).joinpath(*parts)

    def make_embedding_config() -> Embedding1DConfig:
        base_name = args.get_state_dict_prefix("", None) + "tok_embeddings.weight"
        torch_weight = state_dict[base_name].unsqueeze(0).unsqueeze(0)
        cache_dir = cache_path_for(embedding_cache_path, "embedding")
        return Embedding1DConfig(
            weights=LazyWeight(
                source=torch_weight,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                mesh_mapper_config=mesh_shard(-1),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(cache_dir, "tok_embeddings") if cache_dir else None,
            ),
            mesh_device=mesh_device,
            weights_dtype=ttnn.bfloat16,
            weights_memcfg=ttnn.DRAM_MEMORY_CONFIG,
            output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        )

    def make_rope_config() -> Rope1DConfig:
        rope_scaling = None
        if getattr(args, "rope_scaling_params", None) is not None:
            rope_scaling = rope_scaling_model_factory(
                args.rope_scaling_params, getattr(args, "original_max_context_len", None)
            )
        cos_torch, sin_torch = compute_gather_cos_sin(
            dhead=args.head_dim,
            end=2 * args.max_seq_len,
            theta=args.rope_theta,
            rope_scaling=rope_scaling,
        )
        return Rope1DConfig(
            cos_matrix=LazyWeight(source=cos_torch, device=mesh_device),
            sin_matrix=LazyWeight(source=sin_torch, device=mesh_device),
            max_batch_size=args.max_batch_size,
            head_dim=args.head_dim,
            device=mesh_device,
            use_qk_fused=getattr(args, "use_qk_fused", False),
            datatype=ttnn.bfloat16,
        )

    def norm_weight_name(layer_num: int | None, weight_key: str, state_dict_prefix: str | None = None) -> str:
        if state_dict_prefix:
            return f"{state_dict_prefix}{weight_key}.weight"
        if layer_num is None:
            return f"{weight_key}.weight"
        return f"layers.{layer_num}.{weight_key}.weight"

    def make_norm_config(
        *,
        layer_num: int | None,
        weight_key: str,
        state_dict_prefix: str | None = None,
        sharded_program_config=None,
        sharded_output_config=None,
    ) -> RMSNorm1DConfig:
        weight_name = norm_weight_name(layer_num, weight_key, state_dict_prefix)
        torch_weight = (
            state_dict[weight_name]
            .unsqueeze(0)
            .view(1, 1, args.dim)
            .reshape([1, 1, args.dim // SHARD_HEIGHT, SHARD_HEIGHT])
        )
        if args.rms_norm_add_unit_offset:
            torch_weight = torch_weight + 1.0
        return RMSNorm1DConfig(
            weight=LazyWeight(
                source=torch_weight,
                dtype=ttnn.bfloat16,
                device=mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_dir_weight_name=(weight_cache_path, weight_name) if weight_cache_path else None,
                mesh_mapper_config=(
                    ttnn.MeshMapperConfig(
                        placements=[ttnn.PlacementReplicate()],
                        mesh_shape_override=ttnn.MeshShape([num_devices]),
                    )
                    if num_devices > 1
                    else None
                ),
            ),
            eps=args.norm_eps,
            add_unit_offset=False,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            max_batch_size=args.max_batch_size,
            prefill_distributed=num_devices > 1 and args.dim >= 4096,
            decode_program_config=sharded_program_config,
            decode_memory_config=sharded_output_config,
            compute_kernel_config=ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )

    def make_attention_config(layer_num: int, transformation_mats: dict[str, ttnn.Tensor]) -> Attention1DConfig:
        layer_name = args.get_state_dict_prefix("Attention", layer_num)
        wq_str = f"{layer_name}.wq"
        wk_str = f"{layer_name}.wk"
        wv_str = f"{layer_name}.wv"
        wo_str = f"{layer_name}.wo"
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        wqkv_dtype = args.get_tensor_dtype(layer_num, "wqkv")
        wo_dtype = args.get_tensor_dtype(layer_num, "wo")
        kv_cache_dtype = args.get_tensor_dtype(layer_num, "kv_cache")
        activation_dtype = args.get_tensor_dtype(layer_num, "activation")

        qkv_list = []
        for device_idx in range(num_devices):
            wq = torch.transpose(torch.chunk(state_dict[f"{wq_str}.weight"], num_devices, dim=0)[device_idx], -2, -1)
            wk = torch.transpose(torch.chunk(state_dict[f"{wk_str}.weight"], num_devices, dim=0)[device_idx], -2, -1)
            wv = torch.transpose(torch.chunk(state_dict[f"{wv_str}.weight"], num_devices, dim=0)[device_idx], -2, -1)
            qkv_list.append(torch.cat([wq, wk, wv], dim=-1))
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        use_fused_all_gather_matmul = getattr(args, "use_fused_all_gather_matmul", False)
        wqkv = LazyWeight(
            source=qkv_cat,
            dtype=wqkv_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=args.create_dram_sharded_mem_config(args.dim, args.qkv_size // num_devices),
            mesh_mapper_config=mesh_shard(-1),
            cache_dir_weight_name=(weight_cache_path / layer_name, "wqkv_sharded") if weight_cache_path else None,
        )
        wo = LazyWeight(
            source=state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0),
            dtype=wo_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=(
                ttnn.DRAM_MEMORY_CONFIG
                if use_fused_all_gather_matmul
                else args.create_dram_sharded_mem_config((args.n_heads * args.head_dim) // num_devices, args.dim)
            ),
            mesh_mapper_config=mesh_shard(-1 if use_fused_all_gather_matmul else -2),
            cache_dir_weight_name=(
                (weight_cache_path / layer_name, "wo_width_sharded" if use_fused_all_gather_matmul else "wo")
                if weight_cache_path
                else None
            ),
        )

        qk_norm_compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def make_qk_norm_config(name: str) -> RMSNorm1DConfig | None:
            weight_name = f"{name}.weight"
            if weight_name not in state_dict:
                return None
            return RMSNorm1DConfig(
                weight=LazyWeight(
                    source=state_dict[weight_name].reshape(1, 1, -1, TILE_SIZE),
                    dtype=ttnn.bfloat16,
                    device=mesh_device,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_dir_weight_name=(
                        (weight_cache_path / layer_name, name.rsplit(".", 1)[-1]) if weight_cache_path else None
                    ),
                ),
                mesh_device=mesh_device,
                eps=args.norm_eps,
                add_unit_offset=args.rms_norm_add_unit_offset,
                decode_in_sharded=False,
                decode_out_sharded=False,
                prefill_distributed=False,
                compute_kernel_config=qk_norm_compute_kernel,
            )

        wqkv_bias = None
        if f"{wq_str}.bias" in state_dict:
            wqkv_bias = LazyWeight(
                source=torch.concat(
                    [
                        torch.concat(
                            [
                                torch.chunk(state_dict[f"{wq_str}.bias"], num_devices)[device_idx],
                                torch.chunk(state_dict[f"{wk_str}.bias"], num_devices)[device_idx],
                                torch.chunk(state_dict[f"{wv_str}.bias"], num_devices)[device_idx],
                            ],
                            dim=-1,
                        )
                        for device_idx in range(num_devices)
                    ],
                    dim=-1,
                )
            )

        scale = args.query_pre_attn_scalar**-0.5 if args.query_pre_attn_scalar is not None else args.head_dim**-0.5
        attn_agmm_cfg = model_config.get("ATTN_AGMM_CONFIG", {})
        return Attention1DConfig(
            wqkv=wqkv,
            wo=wo,
            q_norm_config=make_qk_norm_config(q_norm_str),
            k_norm_config=make_qk_norm_config(k_norm_str),
            wqkv_bias=wqkv_bias,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            topology=args.ccl_topology(),
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            head_dim=args.head_dim,
            qkv_size=args.qkv_size,
            max_batch_size=args.max_batch_size,
            max_seq_len=args.max_seq_len,
            scale=scale,
            sliding_window=args.sliding_window if hasattr(args, "sliding_window") else None,
            use_qk_fused=getattr(args, "use_qk_fused", False),
            use_vllm_paged_kv_cache=use_paged_kv_cache,
            paged_attention_config=paged_attention_config,
            kv_cache_dtype=kv_cache_dtype,
            min_kv_prefill_shard_seqlen=args.min_kv_prefill_shard_seqlen,
            wqkv_dtype=wqkv_dtype,
            wo_dtype=wo_dtype,
            activation_dtype=activation_dtype,
            decode_xqkv_prg_config=model_config.get("XQKV_DECODE_PROGCFG"),
            decode_sdpa_prg_config=model_config.get("SDPA_DECODE_PROGCFG"),
            decode_attn_output_prg_config=model_config.get("ATTN_OUTPUT_PROGCFG"),
            decode_residual_memcfg=model_config.get("DECODE_RESIDUAL_MEMCFG"),
            decode_create_qkv_head_memcfg=model_config.get("CREATE_QKV_DECODE_SHARD"),
            decode_scores_memcfg=model_config.get("SCORES_BATCHED_MM_OUTPUT_MEMCFG"),
            prefill_xqkv_prg_config=model_config.get("XQKV_PREFILL_PROGCFG"),
            prefill_sdpa_prg_config=model_config.get("SDPA_PROGCFG"),
            prefill_wo_prg_config=model_config.get("WO_PREFILL_PROGCFG"),
            prefill_kv_memcfg=model_config.get("KV_PREFILL_MEM_CFG"),
            use_fused_all_gather_matmul=use_fused_all_gather_matmul,
            decode_all_gather_matmul_prg_config=model_config.get("ATTN_ALL_GATHER_MATMUL_PROGCFG"),
            decode_all_gather_matmul_memcfg=model_config.get("ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"),
            decode_agmm_num_links=attn_agmm_cfg.get("num_links", 1),
            decode_agmm_chunks_per_sync=attn_agmm_cfg.get("chunks_per_sync", 10),
            decode_agmm_num_workers_per_link=attn_agmm_cfg.get("num_workers_per_link", 2),
            li_qkv_decode_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_qkv_decode"),
            sdpa_decode_compute_kernel_cfg=args.get_math_fidelity(layer_num, "sdpa_decode"),
            li_o_decode_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_o_decode"),
            li_qkv_prefill_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_qkv_prefill"),
            sdpa_prefill_compute_kernel_cfg=args.get_math_fidelity(layer_num, "sdpa_prefill"),
            li_o_prefill_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_o_prefill"),
            transformation_mat_decode=transformation_mats.get("decode"),
            transformation_mat_prefill=transformation_mats.get("prefill"),
        )

    def make_mlp_config(layer_num: int) -> MLP1DConfig:
        state_dict_prefix = args.get_state_dict_prefix("MLP", layer_num)
        ff1_3_dtype = args.get_tensor_dtype(layer_num, "ff1_ff3")
        ff2_dtype = args.get_tensor_dtype(layer_num, "ff2")
        activation_dtype = args.get_tensor_dtype(layer_num, "activation")
        mlp_rs_cfg = model_config.get("MLP_RS_CONFIG", {})

        dram_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1))}
        )
        w1_w3_mem_config = _create_dram_sharded_mem_config(
            k=args.dim,
            n=args.hidden_dim // num_devices,
            dram_grid=dram_grid,
            tile_size=TILE_SIZE,
            dram_cores=dram_size.x,
        )
        w2_mem_config = _create_dram_sharded_mem_config(
            k=args.hidden_dim // num_devices,
            n=args.dim,
            dram_grid=dram_grid,
            tile_size=TILE_SIZE,
            dram_cores=dram_size.x,
        )
        cache_dir = cache_path_for(weight_cache_path, state_dict_prefix)

        def make_weight_source(name: str, shard_dim: int):
            tensor = torch.transpose(state_dict[f"{state_dict_prefix}.{name}.weight"], -2, -1)
            return pad_dim_to_size(tensor, dim=shard_dim, size=args.hidden_dim)

        return MLP1DConfig(
            w1=LazyWeight(
                source=make_weight_source("w1", -1),
                dtype=ff1_3_dtype,
                device=mesh_device,
                mesh_mapper_config=mesh_shard(-1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=w1_w3_mem_config,
                cache_dir_weight_name=(cache_dir, "w1_sharded") if cache_dir else None,
            ),
            w2=LazyWeight(
                source=make_weight_source("w2", -2),
                dtype=ff2_dtype,
                device=mesh_device,
                mesh_mapper_config=mesh_shard(-2),
                layout=ttnn.TILE_LAYOUT,
                memory_config=w2_mem_config,
                cache_dir_weight_name=(cache_dir, "w2_sharded") if cache_dir else None,
            ),
            w3=LazyWeight(
                source=make_weight_source("w3", -1),
                dtype=ff1_3_dtype,
                device=mesh_device,
                mesh_mapper_config=mesh_shard(-1),
                layout=ttnn.TILE_LAYOUT,
                memory_config=w1_w3_mem_config,
                cache_dir_weight_name=(cache_dir, "w3_sharded") if cache_dir else None,
            ),
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            max_batch_size=args.max_batch_size,
            mlp_activation_type=getattr(args, "mlp_activation_type", ttnn.UnaryOpType.SILU),
            topology=args.ccl_topology(),
            decode_rs_memory_config=mlp_rs_cfg.get("rs_memory_config", ttnn.L1_MEMORY_CONFIG),
            decode_rs_chunks_per_sync=mlp_rs_cfg.get("chunks_per_sync", 1),
            decode_rs_num_workers_per_link=mlp_rs_cfg.get("num_workers_per_link", 1),
            decode_w1_w3_prg_config=args.get_decode_mlp_ff1_3_prg_config(),
            decode_w2_prg_config=args.get_decode_mlp_ff2_prg_config(),
            decode_mlp2_input_memcfg=args.get_decode_mlp_binary_mult_mem_config(),
            decode_residual_memcfg=args.get_decode_mlp_output_mem_config(),
            w1_w3_dtype=ff1_3_dtype,
            w2_dtype=ff2_dtype,
            activation_dtype=activation_dtype,
            ff1_3_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_ff1_ff3"),
            ff2_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_ff2"),
            decode_ff1_3_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_ff1_ff3"),
            decode_ff2_compute_kernel_cfg=args.get_math_fidelity(layer_num, "li_ff2"),
            decode_spill_w1_to_dram_before_w3=False,
            prefill_len_cutoff=args.prefill_len_cutoff,
        )

    def make_lm_head_config() -> LMHead1DConfig:
        vocab_size = args.vocab_size
        padded_vocab_size = math.ceil(vocab_size / TILE_SIZE) * TILE_SIZE
        size_per_device = padded_vocab_size // num_devices
        num_splits = math.ceil(size_per_device / args.max_columns_per_device_lm_head)
        split_sizes = [min(size_per_device, args.max_columns_per_device_lm_head)] * (num_splits - 1)
        split_sizes.append(size_per_device - sum(split_sizes))

        state_dict_prefix = args.get_state_dict_prefix("", None)
        torch_output_weights = state_dict[f"{state_dict_prefix}output.weight"].permute(1, 0)
        if vocab_size < padded_vocab_size:
            torch_output_weights = torch.cat(
                [
                    torch_output_weights,
                    torch.zeros(
                        torch_output_weights.shape[0],
                        padded_vocab_size - vocab_size,
                        dtype=torch_output_weights.dtype,
                    ),
                ],
                dim=-1,
            )

        dram_size = mesh_device.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1))}
        )
        cache_dir = cache_path_for(weight_cache_path, "lm_head")
        output_weights = []
        weights_memcfgs = []
        for split_idx, split_size in enumerate(split_sizes):
            device_splits = []
            for device_idx in range(num_devices):
                start = device_idx * size_per_device + sum(split_sizes[:split_idx])
                end = start + split_size
                device_splits.append(torch_output_weights[:, start:end])
            combined_split = torch.cat(device_splits, dim=-1)
            mem_cfg = _create_dram_sharded_mem_config(
                k=args.dim,
                n=math.ceil(combined_split.shape[-1] / num_devices),
                dram_grid=dram_grid,
                tile_size=TILE_SIZE,
                dram_cores=dram_size.x,
            )
            weights_memcfgs.append(mem_cfg)
            output_weights.append(
                LazyWeight(
                    source=combined_split,
                    dtype=dtype if dtype is not None else ttnn.bfloat8_b,
                    device=mesh_device,
                    mesh_mapper_config=mesh_shard(-1),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mem_cfg,
                    cache_dir_weight_name=(cache_dir, f"output_split_{split_idx}_{combined_split.shape[-1]}")
                    if cache_dir
                    else None,
                )
            )

        tile_padded_batch_rows = TILE_SIZE * math.ceil(args.max_batch_size / TILE_SIZE)
        lm_head_core_grid = args.lm_head_core_grid
        input_memcfg = ttnn.create_sharded_memory_config(
            (tile_padded_batch_rows, math.ceil((args.dim // lm_head_core_grid.num_cores) / TILE_SIZE) * TILE_SIZE),
            lm_head_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        return LMHead1DConfig(
            output_weights=output_weights,
            mesh_device=mesh_device,
            dim=args.dim,
            max_batch_size=args.max_batch_size,
            program_configs=[
                args.dram_matmul_config(tile_padded_batch_rows, args.dim, split_size, lm_head_core_grid.num_cores)
                for split_size in split_sizes
            ],
            compute_kernel_config=_compute_kernel_config_hifi2(),
            lm_head_dtype=getattr(args, "lm_head_dtype", ttnn.bfloat8_b),
            output_memcfg=ttnn.L1_MEMORY_CONFIG,
            input_memcfg=input_memcfg,
            weights_memcfgs=weights_memcfgs,
        )

    def make_sampling_config() -> Sampling1DConfig | None:
        sampling_splits = num_devices if list(mesh_device.shape) != [1, 1] else 2
        if args.vocab_size // sampling_splits > 64 * 1024:
            return None

        num_gather_links = 1
        if "GALAXY_NUM_LINKS" in model_config:
            max_links = model_config["GALAXY_NUM_LINKS"]
            max_top_k = getattr(args, "max_top_k", 32)
            num_gather_links = min(max_top_k // 32, max_links) if max_top_k // 32 <= max_links else max_links

        ag_cfg = model_config.get("SAMPLING_AG_CONFIG", {})
        return Sampling1DConfig(
            vocab_size=getattr(args, "padded_vocab_size", args.vocab_size),
            valid_vocab_size=args.vocab_size,
            mesh_device=mesh_device,
            tt_ccl=tt_ccl_inst,
            max_batch_size=getattr(args, "max_batch_size", 32),
            max_top_k=getattr(args, "max_top_k", 32),
            sub_core_grids=getattr(args, "sub_core_grids", None),
            sub_core_grid_topk=getattr(args, "sub_core_grid_topk", None),
            start_core=getattr(args, "start_core", ttnn.CoreCoord(0, 0)),
            num_gather_links=num_gather_links,
            sampling_memory_config=model_config.get("DECODE_SAMPLING_INPUT_MEMCFG", ttnn.DRAM_MEMORY_CONFIG),
            allow_force_argmax=ag_cfg.get("allow_force_argmax", False),
            num_argmax_gather_links=ag_cfg.get("num_links", num_gather_links),
            ag_topology=ag_cfg.get("topology", ttnn.Topology.Linear),
            argmax_chunks_per_sync=ag_cfg.get("chunks_per_sync", 10),
            argmax_num_workers_per_link=1,
            pad_to_power_of_2=getattr(args, "pad_logits_to_power_of_2", False),
        )

    rope_config = make_rope_config()
    trans_mats_dict = RotarySetup1D.from_config(rope_config).get_both_trans_mats()
    attn_norm_cfg = args.get_decode_norm_config("attn")
    ff_norm_cfg = args.get_decode_norm_config("ff")
    lm_head_norm_cfg = args.get_decode_norm_config("lm_head")
    activation_dtypes = [args.get_tensor_dtype(i, "activation") for i in range(args.n_layers)]

    return Llama3Transformer1DConfig(
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        num_devices=num_devices,
        mesh_device=mesh_device,
        embedding_config=make_embedding_config(),
        rope_config=rope_config,
        block_configs=[
            TransformerBlock1DConfig(
                attention_norm_config=make_norm_config(
                    layer_num=i,
                    weight_key="attention_norm",
                    sharded_program_config=attn_norm_cfg.get("sharded_program_config"),
                    sharded_output_config=attn_norm_cfg.get("sharded_output_config"),
                ),
                attention_config=make_attention_config(i, trans_mats_dict),
                ff_norm_config=make_norm_config(
                    layer_num=i,
                    weight_key="ffn_norm",
                    sharded_program_config=ff_norm_cfg.get("sharded_program_config"),
                    sharded_output_config=ff_norm_cfg.get("sharded_output_config"),
                ),
                mlp_config=make_mlp_config(i),
                decode_residual_memcfg=model_config["DECODE_RESIDUAL_MEMCFG"],
                prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                activation_dtype=activation_dtypes[i],
            )
            for i in range(args.n_layers)
        ],
        norm_config=make_norm_config(
            layer_num=None,
            weight_key="norm",
            state_dict_prefix=args.get_state_dict_prefix("", None),
            sharded_program_config=lm_head_norm_cfg.get("sharded_program_config"),
            sharded_output_config=lm_head_norm_cfg.get("sharded_output_config"),
        ),
        lm_head_config=make_lm_head_config(),
        sampling_config=make_sampling_config(),
        decode_residual_memcfg=model_config["DECODE_RESIDUAL_MEMCFG"],
        prefill_residual_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        activation_dtypes=activation_dtypes,
        tt_ccl=tt_ccl_inst,
        cache_path=str(weight_cache_path) if weight_cache_path else None,
    )


class Llama3Transformer1D(LightweightModule):
    """TTTv2 Llama 3.1-8B Transformer.

    Constructor takes a config and builds everything internally:
        model = Llama3Transformer1D(config)

    Public sub-modules (accessible by executor for trace support):
        - embedding: Embedding1D
        - rope_setup: RotarySetup1D
        - layers: list[TransformerBlock1D]
        - norm: RMSNorm1D (final)
        - lm_head: LMHead1D
        - sampling: Sampling1D | None

    Forward methods take pre-embedded tensors. The executor handles
    embedding, input preparation, and output processing.
    """

    def __init__(self, config: Llama3Transformer1DConfig):
        from tqdm import tqdm

        super().__init__()
        self.config = config

        tt_ccl_inst = config.tt_ccl
        if tt_ccl_inst is None and config.num_devices > 1:
            tt_ccl_inst = get_tt_ccl(config.mesh_device)

        self.embedding = Embedding1D.from_config(config.embedding_config)
        self.rope_setup = RotarySetup1D.from_config(config.rope_config)

        self.layers = [
            TransformerBlock1D.from_config(config.block_configs[i])
            for i in tqdm(range(config.n_layers), desc="Building layers")
        ]

        self.norm = RMSNorm1D.from_config(config.norm_config)
        self.lm_head = LMHead1D.from_config(config.lm_head_config)

        self.sampling = None
        if config.sampling_config is not None:
            self.sampling = Sampling1D.from_config(config.sampling_config)
        self.supports_on_device_sampling = self.sampling is not None

        self.mesh_device = config.mesh_device
        self.tt_ccl = tt_ccl_inst
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.num_devices = config.num_devices
        self.decode_residual_memcfg = config.decode_residual_memcfg
        self.prefill_residual_memcfg = config.prefill_residual_memcfg or ttnn.DRAM_MEMORY_CONFIG
        self.activation_dtypes = config.activation_dtypes or [None] * config.n_layers

    # =========================================================================
    # KV Cache binding
    # =========================================================================

    def set_kv_cache(self, kv_cache: list):
        """Bind static kv_cache pool via each attention layer's config.

        Must be called before the first forward (before load_device_weights runs).
        The kv_cache is resolved from config during load_device_weights(), just
        like all other weights.
        """
        assert len(kv_cache) == len(
            self.layers
        ), f"kv_cache has {len(kv_cache)} entries but model has {len(self.layers)} layers"
        for i, layer in enumerate(self.layers):
            layer.attention.config.kv_cache = tuple(kv_cache[i])

    # =========================================================================
    # Forward methods — take pre-embedded tensors
    # =========================================================================

    def decode_forward(
        self,
        x_embed: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Decode forward. x_embed is already embedded, unsqueezed, and in decode_residual_memcfg."""
        x = x_embed

        for i, layer in enumerate(self.layers):
            x = ttnn.to_memory_config(x, self.decode_residual_memcfg, self.activation_dtypes[i])

            x = layer.decode_forward(x, current_pos, rot_mats, page_table)

        x = _all_gather_rmsnorm_tensor(self.norm, x, memory_config=self.norm.config.decode_memory_config)
        x = self.norm.decode_forward(x)
        x = self.lm_head.forward(x)
        return x

    def prefill_forward(
        self,
        x_embed: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
    ) -> ttnn.Tensor:
        """Prefill forward. x_embed is already embedded and unsqueezed to 4D."""
        x = x_embed

        for i, layer in enumerate(self.layers):
            activation_dtype = self.activation_dtypes[i]
            if activation_dtype is not None and x.dtype != activation_dtype:
                old = x
                x = ttnn.typecast(x, activation_dtype)
                ttnn.deallocate(old)

            x = layer.prefill_forward(x, rot_mats, user_id, page_table, chunk_page_table, chunk_start_idx)

        if get_last_token == -1:
            return x

        get_last_token_floor = (get_last_token // 32) * 32
        old = x
        x = ttnn.slice(x, (0, 0, get_last_token_floor, 0), (1, 1, get_last_token_floor + 32, x.shape[-1]))
        ttnn.deallocate(old)

        x = self.norm.prefill_forward(x)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return x

    def post_process_prefill_output(self, hidden_states: ttnn.Tensor, last_token_idx: int) -> ttnn.Tensor:
        """Convert traced prefill hidden states into logits for the last token block."""
        get_last_token_floor = (last_token_idx // 32) * 32
        x = ttnn.slice(
            hidden_states,
            (0, 0, get_last_token_floor, 0),
            (1, 1, get_last_token_floor + 32, hidden_states.shape[-1]),
        )

        x = self.norm.prefill_forward(x)
        x = _all_gather_rmsnorm_tensor(self.norm, x)
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        x = self.lm_head.forward(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        return x

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos=None,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id: int = 0,
        mode: str = "decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        get_last_token: int = -1,
    ) -> ttnn.Tensor:
        """Dispatcher for backward compatibility. Llama 3.1-8B has no local rope."""
        rot_mats = rot_mats_global
        if mode == "prefill":
            return self.prefill_forward(
                x,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                get_last_token=get_last_token,
            )
        return self.decode_forward(
            x,
            current_pos,
            rot_mats,
            page_table=page_table,
        )

    # =========================================================================
    # Embedding + output processing helpers (called by executor)
    # =========================================================================

    def embed_decode(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed tokens and prepare for decode. Returns tensor in decode_residual_memcfg."""
        x = self.embedding.forward(tokens)
        x = ttnn.unsqueeze_to_4D(x)
        x = ttnn.to_memory_config(x, self.decode_residual_memcfg)
        return x

    def embed_prefill(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """Embed tokens for prefill. Returns tensor in DRAM interleaved."""
        x = self.embedding.forward(tokens)
        x = ttnn.unsqueeze_to_4D(x)
        return x

    def gather_and_untilize_logits(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        """All-gather logits across devices and untilize for host argmax."""
        if self.num_devices > 1:
            logits = ttnn.experimental.all_gather_async(
                logits,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                memory_config=logits.memory_config(),
                topology=default_topology(self.mesh_device),
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        logits = ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return logits

    def increment_positions(self, current_pos: ttnn.Tensor, rot_mat_idxs: ttnn.Tensor):
        """Increment decode position counters on device."""
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs)


# =============================================================================
# RMSNorm gather helpers
# =============================================================================


def _all_gather_rmsnorm_tensor(
    norm: RMSNorm1D, x: ttnn.Tensor, *, memory_config: ttnn.MemoryConfig | None = None
) -> ttnn.Tensor:
    cfg = norm.config
    if cfg.mesh_device.get_num_devices() == 1 or x.shape[-1] == cfg.weight.source.numel():
        return x

    if memory_config is None:
        memory_config = x.memory_config()

    tt_ccl = cfg.tt_ccl or get_tt_ccl(cfg.mesh_device)
    return ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=tt_ccl.get_num_links(),
        topology=default_topology(cfg.mesh_device),
        memory_config=memory_config,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )


# =============================================================================
# EagerLlamaExecutor — thin wrapper
# =============================================================================


class EagerLlamaExecutor:
    """Thin wrapper: passes Llama model to EagerLLMExecutor.

    All actual logic lives in the engine. This class exists to:
    1. Provide a model-specific type for type hints
    2. Preserve the existing API for demos and tests
    """

    def __init__(self, model: Llama3Transformer1D, mesh_device: ttnn.MeshDevice, model_args=None):
        # Attach model_args to model so engine can access it via model.model_args
        if model_args is not None:
            model.model_args = model_args
        self._engine = EagerLLMExecutor(model, mesh_device, iter_named_modules=_iter_llama_executor_named_modules)

    @property
    def model(self):
        return self._engine.model

    @property
    def mesh_device(self):
        return self._engine.mesh_device

    @property
    def model_args(self):
        return self._engine.model_args

    @property
    def mode(self):
        return self._engine.mode

    @mode.setter
    def mode(self, value):
        self._engine.mode = value

    # =========================================================================
    # KV Cache — delegate to engine
    # =========================================================================

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self._engine.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    def _assert_kv_cache_identity(self, kv_cache):
        return self._engine._assert_kv_cache_identity(kv_cache)

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table):
        return self._engine.prepare_decode_inputs_host(tokens, current_pos, page_table)

    def prepare_decode_inputs_device(self, tokens, current_pos, page_table):
        return self._engine.prepare_decode_inputs_device(tokens, current_pos, page_table)

    # =========================================================================
    # Compile — delegate to engine
    # =========================================================================

    def compile_prefill(
        self,
        *,
        tokens,
        page_table,  # Required
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        start_pos=None,
        sampling_params=None,
    ):
        return self._engine.compile_prefill(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            sampling_params=sampling_params,
        )

    def compile_decode(
        self,
        *,
        tokens,
        start_pos,
        page_table,  # Required
        kv_cache=None,
        sampling_params=None,
    ):
        return self._engine.compile_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )

    # =========================================================================
    # Forward — delegate to engine
    # =========================================================================

    def prefill_forward(
        self,
        tokens,
        page_table,  # Required
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        sampling_params=None,
        start_pos=None,
    ):
        return self._engine.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
            start_pos=start_pos,
        )

    def _prefill_single_user(self, tokens, page_table, user_id, last_token_idx, num_cached_tokens=0):
        return self._engine._prefill_single_user(tokens, page_table, user_id, last_token_idx, num_cached_tokens)

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,  # Required
        kv_cache=None,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
    ):
        return self._engine.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=read_from_device,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )

    # =========================================================================
    # Cleanup — delegate to engine
    # =========================================================================

    def cleanup(self):
        return self._engine.cleanup()


# =============================================================================
# TracedLlamaExecutor — thin wrapper
# =============================================================================


class TracedLlamaExecutor:
    """Thin wrapper: passes Llama model to TracedLLMExecutor.

    All actual logic lives in the engine. This class exists to:
    1. Provide a model-specific type for type hints
    2. Preserve the existing API for demos and tests
    """

    def __init__(self, model: Llama3Transformer1D, mesh_device: ttnn.MeshDevice, model_args=None):
        # Attach model_args to model so engine can access it via model.model_args
        if model_args is not None:
            model.model_args = model_args
        self._engine = TracedLLMExecutor(model, mesh_device, iter_named_modules=_iter_llama_executor_named_modules)

    @property
    def model(self):
        return self._engine.model

    @property
    def mesh_device(self):
        return self._engine.mesh_device

    @property
    def model_args(self):
        return self._engine.model_args

    @property
    def mode(self):
        return self._engine.mode

    @mode.setter
    def mode(self, value):
        self._engine.mode = value

    # Expose internal state for tests/debugging
    @property
    def trace_id_prefill(self):
        return self._engine.trace_id_prefill

    @property
    def trace_ids_decode(self):
        return self._engine.trace_ids_decode

    @property
    def already_warmed_up_prefill(self):
        return self._engine.already_warmed_up_prefill

    # =========================================================================
    # KV Cache — delegate to engine
    # =========================================================================

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        return self._engine.allocate_kv_cache(kv_cache_shape, dtype, num_layers)

    # =========================================================================
    # Warmup — delegate to engine
    # =========================================================================

    def warmup_model_prefill(self, seq_lens, make_tokens, make_page_table):
        return self._engine.warmup_model_prefill(seq_lens, make_tokens, make_page_table)

    # =========================================================================
    # Compile — delegate to engine
    # =========================================================================

    def compile_prefill(
        self,
        *,
        tokens,
        page_table,  # Required
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        start_pos=None,
        sampling_params=None,
    ):
        return self._engine.compile_prefill(
            tokens=tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            start_pos=start_pos,
            sampling_params=sampling_params,
        )

    def compile_decode(
        self,
        *,
        tokens,
        start_pos,
        page_table,  # Required
        kv_cache=None,
        sampling_params=None,
    ):
        return self._engine.compile_decode(
            tokens=tokens,
            start_pos=start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            sampling_params=sampling_params,
        )

    # =========================================================================
    # Forward — delegate to engine
    # =========================================================================

    def prefill_forward(
        self,
        tokens,
        page_table,  # Required
        kv_cache=None,
        prompt_lens=None,
        empty_slots=None,
        sampling_params=None,
        start_pos=None,
    ):
        return self._engine.prefill_forward(
            tokens,
            page_table=page_table,
            kv_cache=kv_cache,
            prompt_lens=prompt_lens,
            empty_slots=empty_slots,
            sampling_params=sampling_params,
            start_pos=start_pos,
        )

    def decode_forward(
        self,
        tokens,
        start_pos,
        page_table,  # Required
        kv_cache=None,
        read_from_device=True,
        sampling_params=None,
        reset_batch=False,
    ):
        return self._engine.decode_forward(
            tokens,
            start_pos,
            page_table=page_table,
            kv_cache=kv_cache,
            read_from_device=read_from_device,
            sampling_params=sampling_params,
            reset_batch=reset_batch,
        )

    # =========================================================================
    # Cleanup — delegate to engine
    # =========================================================================

    def cleanup(self):
        return self._engine.cleanup()


# =============================================================================
# Executor Validation Traversal
# =============================================================================


def _iter_llama_executor_named_modules(model):
    """Yield named submodules that declare executor input contracts."""
    if not hasattr(model, "layers"):
        return

    for i, layer in enumerate(model.layers):
        for suffix, submodule in [
            ("attn_norm", getattr(layer, "attention_norm", None)),
            ("attention", getattr(layer, "attention", None)),
            ("ff_norm", getattr(layer, "ff_norm", None)),
            ("mlp", getattr(layer, "mlp", None)),
        ]:
            if submodule is not None:
                yield f"layer[{i}].{suffix}", submodule

    if hasattr(model, "norm"):
        yield "final_norm", model.norm
    if hasattr(model, "lm_head"):
        yield "lm_head", model.lm_head
