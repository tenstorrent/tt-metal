# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM generator wrapper for Qwen3.6-27B (text-only) on BH Galaxy.

Mirrors the construction in demo/text_demo_qwen36.py: local v2 TtTransformer +
TtQwen36ModelArgs. Weights are loaded from raw safetensors because the
checkpoint's `qwen3_5` architecture is not in any public transformers release
and cannot be loaded by the standard HF model loader.
"""
import os
from pathlib import Path

import torch
from tqdm import tqdm

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.generator import Generator
from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
from models.demos.qwen3_6_galaxy_v2.tt.load_checkpoints import load_hf_state_dict
from models.demos.qwen3_6_galaxy_v2.tt.qwen36_model_config import TtQwen36ModelArgs


def _resolve_ckpt_dir() -> Path:
    """Local checkpoint dir. The server sets HF_MODEL to a local symlink dir."""
    hf_model = os.getenv("HF_MODEL", "Qwen/Qwen3.6-27B")
    p = Path(hf_model)
    if p.is_dir():
        return p
    # Fall back to a resolved HF snapshot.
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(hf_model))


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, model: TtTransformer, tt_cache_path):
    """Paged KV cache allocation for qwen3.6.

    CRITICAL: the external (vLLM-allocated) KV cache MUST match the layout the
    model's paged ops were validated against — the model-internal cache from
    ``TtLlamaAttention.init_kv_cache`` (the demo / test_decode_eager_64L_pcc.py
    path). That layout is ROW-SHARDED across the 8 mesh rows:

        torch shape  = [num_blocks, n_kv_full=8, block_size, head_dim]
        mesh_mapper  = ShardTensor2dMesh(dims=(1, None), mesh_shape=cluster_shape)
        => per chip   [num_blocks, 1, block_size, head_dim], KV head h on row h,
           replicated across the 4 columns.

    vLLM passes ``kv_cache_shape = (num_blocks, num_kv_heads_per_dev=1,
    block_size, head_size)`` (already TP-divided). Allocating that REPLICATED
    (the old code) gives every device an identical 1-head buffer with REPLICATE
    metadata — which does NOT match the row-sharded layout the paged
    fill_cache / decode-SDPA + attention all-gather expect, so decode reads the
    wrong KV heads and degenerates into garbage (prefill's first token is
    computed pre-cache, so it stays correct — the observed symptom). Rebuild the
    cache here with the model's n_kv_full row-shard, keeping vLLM's num_blocks /
    block_size / head_size so the block_table indices stay valid.
    """
    kv_dtype = ttnn.bfloat8_b if os.getenv("QWEN36_KV_BF8", "0") == "1" else ttnn.bfloat16
    num_blocks, _vllm_num_kv_per_dev, block_size, head_size = kv_cache_shape
    n_kv_full = model.args.n_kv_heads  # 8 (padded); sharded 1/row across the 8 rows
    cluster_shape = model.args.cluster_shape
    submesh_devices = [model.mesh_device]
    kv_cache = []
    for mesh_idx, submesh in enumerate(submesh_devices):
        cache_kv = torch.zeros((num_blocks, n_kv_full, block_size, head_size), dtype=dtype)
        row_shard_kv = ttnn.ShardTensor2dMesh(submesh, dims=(1, None), mesh_shape=cluster_shape)
        kv_tt = []
        for _ in tqdm(range(num_layers), desc=f"Allocating TT kv caches (submesh {mesh_idx+1})"):
            kv_tt_i = [
                ttnn.as_tensor(
                    cache_kv,
                    device=submesh,
                    mesh_mapper=row_shard_kv,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=kv_dtype,
                    cache_file_name=tt_cache_path
                    / f"empty_{kv}cache_paged_rowshard_{num_blocks}_{n_kv_full}_{block_size}_{head_size}",
                )
                for kv in ["k", "v"]
            ]
            kv_tt.append(kv_tt_i)
        kv_cache.append(kv_tt)
    return kv_cache


def initialize_vllm_text_transformer_qwen36(
    hf_config,
    mesh_device,
    max_batch_size,
    max_seq_len,
    n_layers=None,
    dtype=ttnn.bfloat8_b,
):
    instruct = "instruct" in str(getattr(hf_config, "_name_or_path", "")).lower()
    args = TtQwen36ModelArgs(
        mesh_device,
        instruct=instruct,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
    )
    if n_layers is not None:
        args.n_layers = n_layers

    # Raw safetensors load — the standard HF model loader cannot parse `qwen3_5`.
    ckpt_dir = _resolve_ckpt_dir()
    state_dict = load_hf_state_dict(str(ckpt_dir))

    weight_cache_path = args.weight_cache_path(dtype)
    weight_cache_path.mkdir(parents=True, exist_ok=True)

    tt_model = TtTransformer(
        args=args,
        dtype=dtype,
        mesh_device=mesh_device,
        state_dict=state_dict,
        weight_cache_path=weight_cache_path,
        use_paged_kv_cache=True,
        mode="prefill",
    )
    return tt_model, args


class Qwen3_5ForConditionalGeneration(Generator):
    """Text-only vLLM serving class for Qwen3.6-27B. Name matches the HF arch
    so platform.py's `TT` prefix resolves to this class."""

    model_capabilities = {
        "supports_prefix_caching": False,
        "supports_async_decode": False,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len=262144,
        n_layers=None,
        tt_data_parallel=1,
        optimizations=None,
    ):
        assert optimizations is None, "Custom optimizations are not supported for this model"
        assert (
            tt_data_parallel == 1
        ), f"Qwen3.6 v2 galaxy is TP-only; data parallel > 1 unsupported, got tt_data_parallel={tt_data_parallel}"
        tt_model, model_args = initialize_vllm_text_transformer_qwen36(
            hf_config,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
        )
        inst = cls(tt_model, model_args, mesh_device)
        # Prefill trace-capture hits a GDN/DeltaNet L1 circular-buffer clash
        # (V2-9 trace-capture blocker); run prefill eager via the Generator's
        # _disable_prefill_tracing hook. DECODE is traced (worker
        # override_tt_config.trace_mode=true) — the demo's proven decode path;
        # eager decode was never multi-step-verified and produced garbage past
        # token 1. _disable_decode_tracing is intentionally left unset.
        inst._disable_prefill_tracing = True
        # Skip the built-in prefill warmup: it is hardcoded for batch-32 (loops
        # batch in (1,32) + forces on-device sampling which asserts
        # max_batch_size % 32 == 0). For batch-1 serving we sample on host
        # (sample_on_device_mode=None), so warmup is unnecessary; the first
        # request prefills directly (eager).
        inst.prefill_warmup_completed = True
        return inst

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, model=self.model, tt_cache_path=self.cache_path)
