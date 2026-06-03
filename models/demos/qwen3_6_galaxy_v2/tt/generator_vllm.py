# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""vLLM generator wrapper for Qwen3.6-27B (text-only) on BH Galaxy.

Mirrors the construction in demo/text_demo_qwen36.py: local v2 TtTransformer +
TtQwen36ModelArgs. Weights are loaded from raw safetensors because the
checkpoint's `qwen3_5` architecture is not in any public transformers release
and cannot be loaded by the standard HF model loader.
"""
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file as load_st
from tqdm import tqdm

import ttnn
from models.demos.qwen3_6_galaxy_v2.tt.generator import Generator
from models.demos.qwen3_6_galaxy_v2.tt.llama_model import TtTransformer
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


def _load_full_state_dict(ckpt_dir: Path) -> dict:
    """Load the raw HF state dict (model.language_model.* keys) from safetensors."""
    index_path = ckpt_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
        files = sorted(set(index["weight_map"].values()))
        sd = {}
        for fn in files:
            sd.update(load_st(str(ckpt_dir / fn)))
        return sd
    return load_st(str(ckpt_dir / "model.safetensors"))


def allocate_vllm_kv_cache(kv_cache_shape, dtype, num_layers, model: TtTransformer, tt_cache_path):
    """Paged KV cache allocation for qwen3.6 (n_kv_heads padded to 8)."""
    kv_dtype = ttnn.bfloat8_b if os.getenv("QWEN36_KV_BF8", "0") == "1" else ttnn.bfloat16
    submesh_devices = [model.mesh_device]
    kv_cache = []
    for mesh_idx, submesh in enumerate(submesh_devices):
        cache_kv = torch.zeros(kv_cache_shape, dtype=dtype)
        kv_tt = []
        for _ in tqdm(range(num_layers), desc=f"Allocating TT kv caches (submesh {mesh_idx+1})"):
            kv_tt_i = [
                ttnn.as_tensor(
                    cache_kv,
                    device=submesh,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=kv_dtype,
                    cache_file_name=tt_cache_path / f"empty_{kv}cache_paged_attention{kv_cache_shape}",
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
    state_dict = _load_full_state_dict(ckpt_dir)

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
        tt_model, model_args = initialize_vllm_text_transformer_qwen36(
            hf_config,
            mesh_device,
            max_batch_size,
            max_seq_len=max_seq_len,
            n_layers=n_layers,
            dtype=ttnn.bfloat8_b,
        )
        return cls(tt_model, model_args, mesh_device)

    @property
    def cache_path(self):
        return self.model_args.model_cache_path

    def prefill_forward(self, *args, **kwargs):
        return super().prefill_forward_text(*args, **kwargs)

    def decode_forward(self, *args, **kwargs):
        return super().decode_forward(*args, **kwargs)

    def allocate_kv_cache(self, *args, **kwargs):
        return allocate_vllm_kv_cache(*args, **kwargs, model=self.model, tt_cache_path=self.cache_path)
