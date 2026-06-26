# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Shared CPU reference for the DSA (Deepseek Sparse Attention) MLA path — DeepSeek V3.2 & GLM-5.1.

The dense MLA tests compare against `create_mla_reference` (config-driven HF dense attention). That
truth is WRONG for a sparse variant once the causal context exceeds `index_topk`: the device attends
to only the top-k selected positions while the dense reference attends to everything. So `has_indexer`
variants compare against the `reference_cpu` MLACPU/IndexerCPU instead — it runs the lightning indexer
and the {0, -inf} index mask, matching the device sparse path.

This module is the variant-driven, importable home for the DSA CPU-reference helpers (it subsumed the
former DeepSeek-only copy in the v3.2 test suite). The device side needs no changes: the unified ttMLA auto-runs the
sparse path whenever the weights dict carries the indexer keys (see `INDEXER_WEIGHT_NAMES`). The crux
is that the device weights and the CPU truth come from the SAME MLACPU instance (remapped through
`WEIGHT_NAME_MAP`), so they are bit-identical and PCC is meaningful.
"""

import os
from pathlib import Path

import torch
from loguru import logger

from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.model import MLACPU, ModelArgs
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.utils import precompute_freqs_cis
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32.weights import DEFAULT_REPO, initialize_weights

# MLACPU param name -> v3 ttMLA weights-dict name (same [out, in] layout). The indexer keys make the
# unified ttMLA take the sparse path; note the LayerNorm bias is remapped onto a separate
# `indexer.k_norm_bias` weight (the device keeps γ and β as distinct tensors).
WEIGHT_NAME_MAP = {
    "wq_a.weight": "q_a_proj.weight",
    "q_norm.weight": "q_a_layernorm.weight",
    "wq_b.weight": "q_b_proj.weight",
    "wkv_a.weight": "kv_a_proj_with_mqa.weight",
    "kv_norm.weight": "kv_a_layernorm.weight",
    "wkv_b.weight": "kv_b_proj.weight",
    "wo.weight": "o_proj.weight",
    "indexer.wq_b.weight": "indexer.wq_b.weight",
    "indexer.wk.weight": "indexer.wk.weight",
    "indexer.k_norm.weight": "indexer.k_norm.weight",
    "indexer.k_norm.bias": "indexer.k_norm_bias.weight",
    "indexer.weights_proj.weight": "indexer.weights_proj.weight",
}


def cpu_ref_cache_dir(variant) -> Path:
    """Disk cache dir for this variant's CPU reference outputs (env override, else /tmp)."""
    env = variant.mla_ref_cache_env or "DEEPSEEK_V3_MLA_REF_CACHE"
    return Path(os.environ.get(env, f"/tmp/{variant.name}_mla_ref_cache"))


def build_cpu_reference(variant, seq_len, seed=42, layer=None, checkpoint_path=None, repo=None):
    """MLACPU + a v3-format weights dict (MLA + indexer). Returns (args, model, weights, src_tag).

    `args` come from `variant.cpu_model_args` (GLM's ModelArgs) or the default ModelArgs (DeepSeek
    dims). Weights are random (default) or pretrained layer `layer` (HF `repo`, or local
    `checkpoint_path` shards). `src_tag` identifies the source so random/pretrained ref caches never
    collide. The device weights are the SAME tensors as the CPU truth (remapped via WEIGHT_NAME_MAP).
    """
    args = variant.cpu_model_args() if variant.cpu_model_args is not None else ModelArgs(max_batch_size=1)
    # seq_len must fit the rope/cache window. (DeepSeek runs YaRN since max_seq_len > original_seq_len;
    # GLM intentionally sets max == original to disable YaRN — so we do NOT assert the YaRN condition.)
    assert seq_len <= args.max_seq_len, f"seq_len {seq_len} > ModelArgs.max_seq_len {args.max_seq_len}"
    torch.manual_seed(seed)
    # simulate_fp8=False: device KVPE cache stores bf16, keep the truth identical. Functional-parity
    # indexer (use_fp8_path=False): Hadamard + fp8 dropped on both sides.
    mla_cpu = MLACPU(args, simulate_fp8=False).eval()
    mla_cpu.indexer.use_fp8_path = False
    if checkpoint_path is not None:
        initialize_weights(mla_cpu, layer=layer or 0, checkpoint_path=checkpoint_path)
        src_tag = f"ckptL{layer or 0}"
    elif layer is not None:
        initialize_weights(mla_cpu, layer=layer, repo=repo or DEFAULT_REPO)
        src_tag = f"layer{layer}"
    else:
        initialize_weights(mla_cpu)  # random
        src_tag = f"random_seed{seed}"
    sd = mla_cpu.state_dict()
    weights = {v3_name: sd[cpu_name].clone() for cpu_name, v3_name in WEIGHT_NAME_MAP.items()}
    return args, mla_cpu, weights, src_tag


def make_hidden(seq_len, hidden_size, seed=42, input_path=None):
    """MLA/indexer input [1, seq, hidden] bf16: from `input_path` (.pt, sliced/checked) or randn(seed)."""
    if input_path:
        t = torch.load(input_path, weights_only=True)
        t = t["hidden_states"] if isinstance(t, dict) else t
        t = t.reshape(-1, t.shape[-1])  # [.., hidden] -> [tokens, hidden]
        assert (
            t.shape[0] >= seq_len and t.shape[-1] == hidden_size
        ), f"input {tuple(t.shape)} can't supply [{seq_len}, {hidden_size}]"
        return t[:seq_len].reshape(1, seq_len, hidden_size).to(torch.bfloat16)
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)


def run_cpu_reference(args, mla_cpu, hidden_states, seq_len, cache_dir, cache_tag):
    """Single-shot MLACPU forward (indexer active). Disk-cached (output + KVPE) under `cache_dir`."""
    cache_path = Path(cache_dir) / f"{cache_tag}_seq{seq_len}.pt"
    if cache_path.exists():
        logger.info(f"Loading cached CPU reference from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        return cached["ref_output"], cached["ref_kvpe"]

    freqs_cis = precompute_freqs_cis(args)[:seq_len]
    mask = torch.full((seq_len, seq_len), float("-inf")).triu_(1)
    with torch.no_grad():
        ref_output = mla_cpu.forward(hidden_states.to(torch.bfloat16), 0, freqs_cis, mask)
    # KVPE truth in device layout: [1, 1, seq, kv_lora_rank + rope] = latent kv ++ k_pe
    ref_kvpe = torch.cat([mla_cpu.kv_cache[:1, :seq_len], mla_cpu.pe_cache[:1, :seq_len]], dim=-1).unsqueeze(1)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"ref_output": ref_output, "ref_kvpe": ref_kvpe}, cache_path)
    logger.info(f"Saved CPU reference to {cache_path}")
    return ref_output, ref_kvpe


def run_cpu_reference_chunked(args, mla_cpu, hidden_states, seq_len, chunk, cache_dir, cache_tag):
    """Chunk-loop MLACPU truth via the decode branch; mask [chunk, end_pos] keeps causality."""
    cache_path = Path(cache_dir) / f"chunked_{cache_tag}_seq{seq_len}_c{chunk}.pt"
    if cache_path.exists():
        logger.info(f"Loading cached chunked CPU reference from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        return cached["ref_output"], cached["ref_kvpe"]

    freqs_all = precompute_freqs_cis(args)
    outs = []
    with torch.no_grad():
        for s in range(0, seq_len, chunk):
            mask = torch.full((chunk, s + chunk), float("-inf")).triu_(s + 1)
            outs.append(
                mla_cpu.forward(hidden_states[:, s : s + chunk].to(torch.bfloat16), s, freqs_all[s : s + chunk], mask)
            )
    ref_output = torch.cat(outs, dim=1)
    ref_kvpe = torch.cat([mla_cpu.kv_cache[:1, :seq_len], mla_cpu.pe_cache[:1, :seq_len]], dim=-1)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"ref_output": ref_output, "ref_kvpe": ref_kvpe}, cache_path)
    logger.info(f"Saved chunked CPU reference to {cache_path}")
    return ref_output, ref_kvpe


def assert_config_matches(config, args):
    """The device runs on the HF config; it must agree with the CPU ModelArgs shapes."""
    pairs = [
        ("hidden_size", "dim"),
        ("num_attention_heads", "n_heads"),
        ("q_lora_rank", "q_lora_rank"),
        ("kv_lora_rank", "kv_lora_rank"),
        ("qk_nope_head_dim", "qk_nope_head_dim"),
        ("qk_rope_head_dim", "qk_rope_head_dim"),
        ("v_head_dim", "v_head_dim"),
    ]
    for hf_name, args_name in pairs:
        assert getattr(config, hf_name) == getattr(
            args, args_name
        ), f"HF config.{hf_name}={getattr(config, hf_name)} != ModelArgs.{args_name}={getattr(args, args_name)}"
