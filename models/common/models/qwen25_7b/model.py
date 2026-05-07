# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Qwen2.5-7B-Instruct — native TTTv2 stack (``Embedding1D``, ``RMSNorm1D``,
``Attention1D``, ``MLP1D``, ``RotarySetup1D``, ``LMHead1D``).

Tensor layout contracts:
  - **Prefill** hidden states: ``[1, 1, S, dim]`` TILE, ``S % 128 == 0``.
  - **Decode** hidden states: ``[1, 1, B, dim]`` TILE (``B`` padded to tile in modules).

Executor contract (``EagerLLMExecutor`` / ``TracedLLMExecutor``): pre-embedded forwards,
``set_kv_cache``, ``rope_setup``, ``page_table`` through attention, ``model_args`` holds a
:class:`Qwen25ExecutorRuntimeConfig` (not v1 ``ModelArgs``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.models.qwen25_7b import weight_utils
from models.common.modules.attention.attention_1d import (
    Attention1D,
    Attention1DConfig,
    _dram_matmul_config,
    _dram_shard_core_grid,
)
from models.common.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig, _nearest_32
from models.common.modules.mlp.mlp_1d import MLP1D, MLP1DConfig, _dram_shard_core_grid_k_n
from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig, _create_sharded_norm_program_config
from models.common.modules.rope.rope_1d import Rope1DConfig, RotarySetup1D, prepare_rot_idxs
from models.common.modules.tt_ccl import default_topology, get_tt_ccl
from models.common.tensor_utils import TILE_SIZE, get_padded_hidden_dim


def _lazy(
    tensor: torch.Tensor,
    *,
    dtype: ttnn.DataType,
    cache: tuple[Path, str] | None,
) -> LazyWeight:
    """Minimal LazyWeight; ``Attention1D`` / ``MLP1D`` / ``Embedding1D`` resolvers fill mesh + memory."""
    return LazyWeight(source=tensor, dtype=dtype, cache_dir_weight_name=cache)


@dataclass
class Qwen25PagedAttentionConfig:
    """Paged KV layout for ``Attention1D`` (``block_size`` / ``max_num_blocks`` only)."""

    block_size: int
    max_num_blocks: int


@dataclass
class Qwen25ExecutorRuntimeConfig:
    """Engine-facing runtime knobs. Exposed as ``model.model_args`` for shared ``EagerLLMExecutor``."""

    n_layers: int
    n_kv_heads: int
    head_dim: int
    max_batch_size: int
    max_seq_len: int
    cluster_shape: list[int]
    max_prefill_chunk_size: int = 2048
    model_cache_path: Path | None = None
    kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b
    optimizations: Any = None

    def can_enable_trace(self, prefill_seq_len: int, num_cached_tokens: int) -> bool:
        return False


@dataclass
class Qwen25_7BTTTConfig:
    """Resolved hyper-parameters for a loaded HF Qwen2.5-7B checkpoint."""

    hf_model_id: str
    dim: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    num_hidden_layers: int
    max_batch_size: int
    max_seq_len: int
    rope_table_len: int


def _qwen_wh_mlp_matmul_compute_kernel() -> ttnn.WormholeComputeKernelConfig:
    """HiFi4 lowers L1 circular-buffer footprint vs HiFi2 for wide FF matmuls on Wormhole."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


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
        num_links=1,
        topology=default_topology(cfg.mesh_device),
        memory_config=memory_config,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )


class Qwen25_7BDecoderLayer(LightweightModule):
    def __init__(
        self,
        *,
        input_layernorm: RMSNorm1D,
        self_attn: Attention1D,
        post_attention_layernorm: RMSNorm1D,
        mlp: MLP1D,
    ):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp

    def prefill_forward(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        *,
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
    ) -> ttnn.Tensor:
        # Match Llama ``TransformerBlock1D``: fractured embed / norm activations must be
        # all-gathered to full ``dim`` before Attention1D / MLP1D (QKV matmul expects width ``dim``).
        r = self.input_layernorm.prefill_forward(x)
        r = _all_gather_rmsnorm_tensor(self.input_layernorm, r)
        r = self.self_attn.forward(
            r,
            None,
            rot_mats,
            mode="prefill",
            user_id=user_id,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
        )
        h = ttnn.add(x, r, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        r2 = self.post_attention_layernorm.prefill_forward(h)
        r2 = _all_gather_rmsnorm_tensor(self.post_attention_layernorm, r2)
        r2 = self.mlp.prefill_forward(r2)
        return ttnn.add(h, r2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(
        self,
        x: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        xa = _all_gather_rmsnorm_tensor(
            self.input_layernorm, x, memory_config=self.input_layernorm.config.decode_memory_config
        )
        r = self.input_layernorm.forward(xa, "decode")
        r = self.self_attn.forward(r, current_pos, rot_mats, mode="decode", page_table=page_table)
        h = ttnn.add(x, r, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hf = _all_gather_rmsnorm_tensor(
            self.post_attention_layernorm, h, memory_config=self.post_attention_layernorm.config.decode_memory_config
        )
        r2 = self.post_attention_layernorm.forward(hf, "decode")
        r2 = self.mlp.forward(r2, "decode")
        return ttnn.add(h, r2, memory_config=ttnn.DRAM_MEMORY_CONFIG)


class Qwen25_7BTTT(LightweightModule):
    """
    Full decoder for Qwen2.5-7B-Instruct (TTTv2 modules only).

    Prefill/decode on **embedded** activations match ``EagerLLMExecutor``. Token embedding
    is ``embed_prefill`` / ``embed_decode``. Bind KV with ``set_kv_cache`` before first forward.
    """

    decode_residual_memcfg = ttnn.DRAM_MEMORY_CONFIG

    def __init__(
        self,
        cfg: Qwen25_7BTTTConfig,
        embed: Embedding1D,
        rope_setup: RotarySetup1D,
        layers: List[Qwen25_7BDecoderLayer],
        norm: RMSNorm1D,
        lm_head: LMHead1D,
        mesh_device: ttnn.MeshDevice,
    ):
        super().__init__()
        self.cfg = cfg
        self.embed = embed
        self.rope_setup = rope_setup
        self.layers = layers
        self.norm = norm
        self.lm_head = lm_head
        self.mesh_device = mesh_device
        self.sampling = None
        self.model_args: Qwen25ExecutorRuntimeConfig | None = None

        self.vocab_size = cfg.vocab_size
        self.n_layers = cfg.num_hidden_layers
        self.num_devices = mesh_device.get_num_devices()
        self.tt_ccl = get_tt_ccl(mesh_device) if self.num_devices > 1 else None

    @property
    def n_kv_heads(self) -> int:
        return self.cfg.n_kv_heads

    @classmethod
    def from_pretrained(
        cls,
        mesh_device: ttnn.MeshDevice,
        hf_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        *,
        revision: str | None = None,
        max_batch_size: int = 32,
        max_seq_len: int = 4096,
        num_layers: int | None = None,
        cache_dir: Path | str | None = None,
        wqkv_dtype: ttnn.DataType = ttnn.bfloat16,
        mlp_w_dtype: ttnn.DataType = ttnn.bfloat8_b,
        kv_cache_dtype: ttnn.DataType = ttnn.bfloat8_b,
        lm_head_dtype: ttnn.DataType = ttnn.bfloat8_b,
        block_size: int = 32,
        executor_mode: bool = False,
    ) -> Qwen25_7BTTT:
        """
        Load HF weights on host and build TTNN modules (weights materialize on first forward).

        Args:
            mesh_device: Open mesh device (N150 ``(1,1)``, N300 ``(1,2)``, …).
            hf_model_id: Hugging Face hub id.
            max_batch_size: Decode batch / KV allocation (tile-padded internally).
            max_seq_len: KV cache sequence budget (per layer).
            num_layers: If set, truncate stack for smoke tests.
            cache_dir: Optional directory for ``LazyWeight`` tensor caches.
            kv_cache_dtype: Device dtype for paged KV tensors (executor allocation).
            block_size: Paged attention block size (tokens per block).
            executor_mode: If True, use external paged KV (``set_kv_cache`` + shared executor).
                If False, internal KV tensors (smoke / ``prefill_from_token_ids`` without executor).
        """
        ttnn.SetDefaultDevice(mesh_device)
        cache_path = Path(cache_dir) if cache_dir else None
        num_dev = mesh_device.get_num_devices()
        tt_ccl = get_tt_ccl(mesh_device) if num_dev > 1 else None
        topology = default_topology(mesh_device)

        hf_cfg = AutoConfig.from_pretrained(hf_model_id, revision=revision)
        n_heads_hf = hf_cfg.num_attention_heads
        n_kv_hf = hf_cfg.num_key_value_heads
        if num_dev > 1 and (n_heads_hf % num_dev != 0 or n_kv_hf % num_dev != 0):
            raise ValueError(
                f"This checkpoint requires num_attention_heads ({n_heads_hf}) and "
                f"num_key_value_heads ({n_kv_hf}) to each be divisible by the mesh device "
                f"count ({num_dev}) for Attention1D sharding. "
                f"Use a smaller mesh (e.g. 2 or 4 devices) or a model whose head counts divide the mesh."
            )
        torch_dtype = torch.bfloat16
        logger.info(f"Loading HF weights: {hf_model_id} (revision={revision})")
        hf = AutoModelForCausalLM.from_pretrained(hf_model_id, revision=revision, torch_dtype=torch_dtype)
        hf.eval()
        base = hf.model
        n_layers = num_layers if num_layers is not None else hf_cfg.num_hidden_layers
        dim = hf_cfg.hidden_size
        n_heads = hf_cfg.num_attention_heads
        n_kv = hf_cfg.num_key_value_heads
        head_dim = dim // n_heads
        inter = hf_cfg.intermediate_size
        vocab = hf_cfg.vocab_size
        rope_len = max(max_seq_len * 2, 8192)
        rope_len = (rope_len + 127) // 128 * 128

        blocks_per_user = (max_seq_len + block_size - 1) // block_size
        max_num_blocks = blocks_per_user * max_batch_size
        paged_cfg = (
            Qwen25PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks) if executor_mode else None
        )

        qcfg = Qwen25_7BTTTConfig(
            hf_model_id=hf_model_id,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv,
            head_dim=head_dim,
            hidden_dim=inter,
            vocab_size=vocab,
            rms_norm_eps=hf_cfg.rms_norm_eps,
            rope_theta=getattr(hf_cfg, "rope_theta", 1_000_000.0),
            num_hidden_layers=n_layers,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            rope_table_len=rope_len,
        )

        emb_src = weight_utils.embed_tokens_torch(base.embed_tokens)
        emb = Embedding1D.from_config(
            Embedding1DConfig(
                weights=_lazy(
                    emb_src,
                    dtype=ttnn.bfloat16,
                    cache=(cache_path / "embedding", "tok_embeddings") if cache_path else None,
                ),
                mesh_device=mesh_device,
                embed_scale=1.0,
            )
        )

        cos_t, sin_t = weight_utils.build_rope_cos_sin_torch(base.rotary_emb, rope_len, head_dim, torch_dtype)
        cos_lw = _lazy(cos_t, dtype=ttnn.bfloat16, cache=(cache_path / "rope", "cos") if cache_path else None)
        sin_lw = _lazy(sin_t, dtype=ttnn.bfloat16, cache=(cache_path / "rope", "sin") if cache_path else None)
        rope_setup = RotarySetup1D.from_config(
            Rope1DConfig(
                cos_matrix=cos_lw,
                sin_matrix=sin_lw,
                max_batch_size=max_batch_size,
                head_dim=head_dim,
                device=mesh_device,
                use_qk_fused=False,
            )
        )

        # ``get_padded_prefill_len`` maps 129..1024 tokens to a 1024-wide tile. ``MLP1D`` then
        # reshapes/chunks using ``prefill_len_cutoff``. Cutoff 512 still trips
        # ``validate_circular_buffer_region`` on WH (prefill multicast + LM DRAM matmul vs L1).
        # 256 halves the per-kernel M tile; HiFi4 shrinks CB vs HiFi2 for FF and LM DRAM linears.
        # Decode W1→DRAM before W3 avoids L1 overlap between W1 activations and W3 matmul CBs (N300 batch 32).
        model_slug = hf_model_id.split("/")[-1]
        mlp_prefill_len_cutoff: int | None = None
        mlp_ff_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
        mlp_decode_spill_w1_to_dram: bool = False
        lm_head_compute_kernel_cfg: ttnn.WormholeComputeKernelConfig | None = None
        if (model_slug.startswith("Qwen2.5-7B") or model_slug.startswith("Qwen2.5-VL-7B")) and num_dev in (1, 2):
            mlp_prefill_len_cutoff = 256
            mlp_ff_compute_kernel_cfg = _qwen_wh_mlp_matmul_compute_kernel()
            mlp_decode_spill_w1_to_dram = True
            lm_head_compute_kernel_cfg = _qwen_wh_mlp_matmul_compute_kernel()
            logger.info(
                f"MLP/LM L1 tuning for {hf_model_id} on {num_dev} device(s): "
                "prefill_len_cutoff=256, FF HiFi4, decode W1→DRAM before W3, LM head HiFi4"
            )

        layers: list[Qwen25_7BDecoderLayer] = []
        for idx in range(n_layers):
            layer = base.layers[idx]
            prefix = f"layer{idx}"
            wqkv, wo, qn, kn, wqkv_b = weight_utils.attention_wqkv_wo_from_hf_layer(layer.self_attn, num_dev)
            lazy_wqkv = _lazy(
                wqkv,
                dtype=wqkv_dtype,
                cache=(cache_path / "attn", f"{prefix}_wqkv") if cache_path else None,
            )
            lazy_wo = _lazy(
                wo,
                dtype=wqkv_dtype,
                cache=(cache_path / "attn", f"{prefix}_wo") if cache_path else None,
            )
            q_norm_cfg = None
            k_norm_cfg = None
            if qn is not None:
                lazy_qn = _lazy(
                    qn.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch_dtype),
                    dtype=ttnn.bfloat16,
                    cache=(cache_path / "attn", f"{prefix}_qn") if cache_path else None,
                )
                q_norm_cfg = RMSNorm1DConfig(
                    weight=lazy_qn,
                    mesh_device=mesh_device,
                    eps=hf_cfg.rms_norm_eps,
                    decode_in_sharded=False,
                    decode_out_sharded=False,
                    prefill_distributed=False,
                    tt_ccl=tt_ccl,
                )
            if kn is not None:
                lazy_kn = _lazy(
                    kn.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(torch_dtype),
                    dtype=ttnn.bfloat16,
                    cache=(cache_path / "attn", f"{prefix}_kn") if cache_path else None,
                )
                k_norm_cfg = RMSNorm1DConfig(
                    weight=lazy_kn,
                    mesh_device=mesh_device,
                    eps=hf_cfg.rms_norm_eps,
                    decode_in_sharded=False,
                    decode_out_sharded=False,
                    prefill_distributed=False,
                    tt_ccl=tt_ccl,
                )
            bias_lw = (
                LazyWeight(
                    source=wqkv_b.to(torch_dtype),
                    dtype=ttnn.bfloat16,
                    cache_dir_weight_name=(cache_path / "attn", f"{prefix}_bias") if cache_path else None,
                )
                if wqkv_b is not None
                else None
            )

            attn_cfg = Attention1DConfig(
                wqkv=lazy_wqkv,
                wo=lazy_wo,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                topology=topology,
                n_heads=n_heads,
                n_kv_heads=n_kv,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                q_norm_config=q_norm_cfg,
                k_norm_config=k_norm_cfg,
                wqkv_bias=bias_lw,
                use_vllm_paged_kv_cache=executor_mode,
                paged_attention_config=paged_cfg,
                kv_cache=None,
                kv_cache_dtype=kv_cache_dtype,
            )
            attn = Attention1D.from_config(attn_cfg)

            w1, w2, w3 = weight_utils.mlp_weights_from_hf_layer(layer.mlp)
            lazy_w1 = _lazy(w1, dtype=mlp_w_dtype, cache=(cache_path / "mlp", f"{prefix}_w1") if cache_path else None)
            lazy_w2 = _lazy(w2, dtype=mlp_w_dtype, cache=(cache_path / "mlp", f"{prefix}_w2") if cache_path else None)
            lazy_w3 = _lazy(w3, dtype=mlp_w_dtype, cache=(cache_path / "mlp", f"{prefix}_w3") if cache_path else None)
            mlp = MLP1D.from_config(
                MLP1DConfig(
                    w1=lazy_w1,
                    w2=lazy_w2,
                    w3=lazy_w3,
                    mesh_device=mesh_device,
                    tt_ccl=tt_ccl,
                    topology=topology,
                    max_batch_size=max_batch_size,
                    prefill_len_cutoff=mlp_prefill_len_cutoff,
                    ff1_3_compute_kernel_cfg=mlp_ff_compute_kernel_cfg,
                    ff2_compute_kernel_cfg=mlp_ff_compute_kernel_cfg,
                    decode_ff1_3_compute_kernel_cfg=mlp_ff_compute_kernel_cfg,
                    decode_ff2_compute_kernel_cfg=mlp_ff_compute_kernel_cfg,
                    decode_spill_w1_to_dram_before_w3=mlp_decode_spill_w1_to_dram,
                )
            )

            # Post-attention RMSNorm decode output must match ``MLP1D`` decode activations: W1/W3
            # use ``_dram_shard_core_grid_k_n(dim, padded_hidden/N)``, while default RMSNorm uses
            # ``_compute_norm_core_grid(dim)`` only — different DRAM width shards (N300 Qwen2.5-7B).
            padded_hidden = get_padded_hidden_dim(inter, num_dev, TILE_SIZE)
            post_mlp_decode_grid = _dram_shard_core_grid_k_n(dim, padded_hidden // num_dev)
            tile_padded_batch_rows = TILE_SIZE * math.ceil(max_batch_size / TILE_SIZE)
            post_attn_decode_program_config = _create_sharded_norm_program_config(
                dim, post_mlp_decode_grid, tile_padded_batch_rows, TILE_SIZE
            )
            post_attn_decode_memory_config = mlp.config.decode_input_memcfg

            def make_norm(w_flat: torch.Tensor, name: str) -> RMSNorm1D:
                lw = _lazy(
                    w_flat.to(torch_dtype),
                    dtype=ttnn.bfloat16,
                    cache=(cache_path / "norm", f"{prefix}_{name}") if cache_path else None,
                )
                return RMSNorm1D.from_config(
                    RMSNorm1DConfig(
                        weight=lw,
                        mesh_device=mesh_device,
                        eps=hf_cfg.rms_norm_eps,
                        max_batch_size=max_batch_size,
                        tt_ccl=tt_ccl,
                    )
                )

            post_attn_lw = _lazy(
                weight_utils.rms_weight_torch(layer.post_attention_layernorm).to(torch_dtype),
                dtype=ttnn.bfloat16,
                cache=(cache_path / "norm", f"{prefix}_post_attn") if cache_path else None,
            )
            post_attention_layernorm = RMSNorm1D.from_config(
                RMSNorm1DConfig(
                    weight=post_attn_lw,
                    mesh_device=mesh_device,
                    eps=hf_cfg.rms_norm_eps,
                    max_batch_size=max_batch_size,
                    tt_ccl=tt_ccl,
                    decode_program_config=post_attn_decode_program_config,
                    decode_memory_config=post_attn_decode_memory_config,
                )
            )

            layers.append(
                Qwen25_7BDecoderLayer(
                    input_layernorm=make_norm(weight_utils.rms_weight_torch(layer.input_layernorm), "pre_attn"),
                    self_attn=attn,
                    post_attention_layernorm=post_attention_layernorm,
                    mlp=mlp,
                )
            )

        norm_lw = _lazy(
            weight_utils.rms_weight_torch(base.norm).to(torch_dtype),
            dtype=ttnn.bfloat16,
            cache=(cache_path / "norm", "final") if cache_path else None,
        )
        final_norm = RMSNorm1D.from_config(
            RMSNorm1DConfig(
                weight=norm_lw,
                mesh_device=mesh_device,
                eps=hf_cfg.rms_norm_eps,
                max_batch_size=max_batch_size,
                tt_ccl=tt_ccl,
            )
        )

        lm_w = hf.lm_head.weight.detach().to(torch_dtype).clone()
        lm_splits, lm_split_sizes, lm_weights_memcfgs = weight_utils.build_lm_head_lazy_weights(
            mesh_device,
            lm_w,
            dim=dim,
            vocab_size=vocab,
            dtype=lm_head_dtype,
            cache_dir=cache_path / "lm_head" if cache_path else None,
        )
        lm_head_core_grid = _dram_shard_core_grid(dim)
        tile = ttnn.TILE_SIZE
        # LM head DRAM matmul is sized for decode batch tiles (``max_batch_size``). Prefill logits
        # use a single 32-row tile via ``post_process_prefill_output`` / ``run_lm_head`` slice.
        tile_padded_batch_rows = tile * math.ceil(max_batch_size / tile)
        lm_prog_configs = [
            _dram_matmul_config(tile_padded_batch_rows, dim, ss, lm_head_core_grid.num_cores) for ss in lm_split_sizes
        ]
        lm_input_memcfg = ttnn.create_sharded_memory_config(
            (
                tile_padded_batch_rows,
                _nearest_32(dim // lm_head_core_grid.num_cores),
            ),
            lm_head_core_grid,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        lm = LMHead1D.from_config(
            LMHead1DConfig(
                output_weights=lm_splits,
                mesh_device=mesh_device,
                dim=dim,
                max_batch_size=max_batch_size,
                lm_head_dtype=lm_head_dtype,
                program_configs=lm_prog_configs,
                compute_kernel_config=lm_head_compute_kernel_cfg,
                input_memcfg=lm_input_memcfg,
                weights_memcfgs=lm_weights_memcfgs,
            )
        )

        del hf
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        model = cls(qcfg, emb, rope_setup, layers, final_norm, lm, mesh_device)
        if executor_mode:
            model.model_args = Qwen25ExecutorRuntimeConfig(
                n_layers=n_layers,
                n_kv_heads=n_kv,
                head_dim=head_dim,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                cluster_shape=list(mesh_device.shape),
                model_cache_path=cache_path,
                kv_cache_dtype=kv_cache_dtype,
            )
        return model

    def set_kv_cache(self, kv_cache: list) -> None:
        assert len(kv_cache) == len(
            self.layers
        ), f"kv_cache has {len(kv_cache)} entries but model has {len(self.layers)} layers"
        for i, layer in enumerate(self.layers):
            layer.self_attn.config.kv_cache = tuple(kv_cache[i])

    def embed_decode(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        x = self.embed.forward(tokens)
        x = ttnn.unsqueeze_to_4D(x)
        return ttnn.to_memory_config(x, self.decode_residual_memcfg)

    def embed_prefill(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        x = self.embed.forward(tokens)
        return ttnn.unsqueeze_to_4D(x)

    def prefill_forward(
        self,
        x_embed: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        *,
        user_id: int = 0,
        page_table: ttnn.Tensor | None = None,
        chunk_page_table: ttnn.Tensor | None = None,
        chunk_start_idx: int | None = None,
        get_last_token: int = -1,
    ) -> ttnn.Tensor:
        x = x_embed
        for layer in self.layers:
            x = layer.prefill_forward(
                x,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
            )

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
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

    def post_process_prefill_output(self, hidden_states: ttnn.Tensor, last_token_idx: int) -> ttnn.Tensor:
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
        return ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

    def decode_forward(
        self,
        x_embed: ttnn.Tensor,
        current_pos: ttnn.Tensor,
        rot_mats: tuple[ttnn.Tensor, ttnn.Tensor],
        page_table: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        x = x_embed
        for layer in self.layers:
            x = layer.decode_forward(x, current_pos, rot_mats, page_table=page_table)
        x = _all_gather_rmsnorm_tensor(self.norm, x, memory_config=self.norm.config.decode_memory_config)
        x = self.norm.decode_forward(x)
        return self.lm_head.forward(x)

    def gather_and_untilize_logits(self, logits: ttnn.Tensor) -> ttnn.Tensor:
        if self.num_devices > 1 and self.tt_ccl is not None:
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
        return ttnn.untilize(logits, use_multicore=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def increment_positions(self, current_pos: ttnn.Tensor, rot_mat_idxs: ttnn.Tensor) -> None:
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        ttnn.plus_one(rot_mat_idxs)

    def prefill_from_token_ids(self, token_ids_tt: ttnn.Tensor, *, start_pos: int = 0, user_id: int = 0) -> ttnn.Tensor:
        """Legacy path: embed + RoPE + blocks + final norm (no page table). For tests only."""
        x = self.embed_prefill(token_ids_tt)
        seq_len = x.shape[2]
        assert seq_len % 128 == 0, "prefill seq_len must be divisible by 128"
        rot = self.rope_setup.prefill_forward(start_pos, seq_len)
        h = x
        for layer in self.layers:
            h = layer.prefill_forward(h, rot, user_id=user_id, page_table=None)
        h = self.norm.prefill_forward(h)
        return _all_gather_rmsnorm_tensor(self.norm, h)

    def decode_from_token_ids(self, token_ids_tt: ttnn.Tensor, *, current_pos: int) -> ttnn.Tensor:
        """Legacy path: single-token decode without paged ``page_table``."""
        x = self.embed.forward(token_ids_tt)
        x = ttnn.unsqueeze_to_4D(x)
        pos = torch.tensor([current_pos], dtype=torch.long)
        rot_idxs = prepare_rot_idxs(self.rope_setup.config, pos, on_host=False)
        rot = self.rope_setup.decode_forward(rot_idxs)
        cur = ttnn.from_torch(
            pos,
            device=self.mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(self.mesh_device),
        )
        h = x
        for layer in self.layers:
            h = layer.decode_forward(h, cur, rot, page_table=None)
        h = _all_gather_rmsnorm_tensor(self.norm, h, memory_config=self.norm.config.decode_memory_config)
        return self.norm.forward(h, "decode")

    def lm_logits(self, hidden: ttnn.Tensor) -> ttnn.Tensor:
        """Project last hidden to logits (vocab-sharded on multi-device)."""
        x = hidden
        lm_head_memcfg = self.lm_head.config.input_memcfg
        if lm_head_memcfg is not None and lm_head_memcfg.is_sharded():
            x = ttnn.interleaved_to_sharded(x, lm_head_memcfg)
        return self.lm_head.forward(x)
