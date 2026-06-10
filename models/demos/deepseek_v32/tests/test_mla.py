# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
E2E MLA layer test for DeepSeek V3.2 — PCC truth is reference_cpu (MLACPU).

Device side reuses the v3 harness (run_mla_inference, monkeypatched to the v32
ttMLA); only the host reference differs from v3: the same random weights are
loaded into MLACPU and the output/KVPE caches are compared against it.

Bring-up scale: seq_len <= args.index_topk (2048) — there the DSA index mask is
zero over the causal region, so the CPU reference is dense-equivalent and the
passthrough v32 MLA must match. Above 2048 outputs diverge by design until the
indexer + sparse SDPA land in v32.

CPU reference results are cached under $DEEPSEEK_V32_MLA_REF_CACHE (default
/tmp/deepseek_v32_mla_ref_cache).
"""

import os
from pathlib import Path

import pytest
import torch
from loguru import logger
from ttnn.device import is_blackhole

import models.demos.deepseek_v3_d_p.tests.test_mla as v3_test_mla
import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import WH_WORKER_L1_SIZE
from models.demos.deepseek_v32.reference_cpu.model import MLACPU, ModelArgs
from models.demos.deepseek_v32.reference_cpu.utils import precompute_freqs_cis
from models.demos.deepseek_v32.reference_cpu.weights import initialize_weights
from models.demos.deepseek_v32.tt.mla import ttMLA as ttMLAv32
from tests.ttnn.utils_for_testing import assert_with_pcc

OUTPUT_PCC = 0.98
KVPE_PCC = 0.99

# MLACPU layer name -> v3 ttMLA weights-dict name (same [out, in] layout).
WEIGHT_NAME_MAP = {
    "wq_a.weight": "q_a_proj.weight",
    "q_norm.weight": "q_a_layernorm.weight",
    "wq_b.weight": "q_b_proj.weight",
    "wkv_a.weight": "kv_a_proj_with_mqa.weight",
    "kv_norm.weight": "kv_a_layernorm.weight",
    "wkv_b.weight": "kv_b_proj.weight",
    "wo.weight": "o_proj.weight",
    # v32-only: indexer weights, consumed by the v32 ttMLA (popped before v3 sees them)
    "indexer.wq_b.weight": "indexer.wq_b.weight",
    "indexer.wk.weight": "indexer.wk.weight",
    "indexer.k_norm.weight": "indexer.k_norm.weight",
    "indexer.k_norm.bias": "indexer.k_norm_bias.weight",
    "indexer.weights_proj.weight": "indexer.weights_proj.weight",
}


@pytest.fixture(autouse=True)
def use_v32_mla(monkeypatch):
    monkeypatch.setattr(v3_test_mla, "ttMLA", ttMLAv32)


def build_cpu_reference(seq_len: int, seed: int = 42):
    """MLACPU with random weights; returns (args, model, v3-format weights dict)."""
    # Keep max_seq_len at the production 16384, NOT seq_len: YaRN inv_freq and the
    # mscale'd softmax scale only activate when max_seq_len > original_seq_len (4096),
    # and the device tables (HF DeepseekV3YarnRotaryEmbedding) always apply YaRN.
    args = ModelArgs(max_batch_size=1)
    assert seq_len <= args.max_seq_len and args.max_seq_len > args.original_seq_len
    torch.manual_seed(seed)
    # simulate_fp8=False: device KVPE cache stores bf16 (spec.md), keep truth identical.
    mla_cpu = MLACPU(args, simulate_fp8=False).eval()
    # Functional-parity indexer (spec.md §104): Hadamard+fp8 dropped on both sides.
    mla_cpu.indexer.use_fp8_path = False
    initialize_weights(mla_cpu)
    sd = mla_cpu.state_dict()
    weights = {v3_name: sd[cpu_name].clone() for cpu_name, v3_name in WEIGHT_NAME_MAP.items()}
    return args, mla_cpu, weights


def run_cpu_reference(args, mla_cpu, hidden_states, seq_len, seed, cache_tag):
    """Dense-equivalent CPU forward (seq<=topk); disk-cached output + KVPE."""
    cache_dir = Path(os.environ.get("DEEPSEEK_V32_MLA_REF_CACHE", "/tmp/deepseek_v32_mla_ref_cache"))
    cache_path = cache_dir / f"{cache_tag}_seq{seq_len}_seed{seed}.pt"
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

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"ref_output": ref_output, "ref_kvpe": ref_kvpe}, cache_path)
    logger.info(f"Saved CPU reference to {cache_path}")
    return ref_output, ref_kvpe


def assert_config_matches(config, args):
    """Device side runs on the HF config; it must agree with ModelArgs shapes."""
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


# QuietBox bring-up: 1x4 pure TP (agreement 13); 2x2 SP x TP later.
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        }
    ],
    ids=["line"],
    indirect=True,
)
# seq256/seq2k: dense-equivalent (seq <= index_topk, DSA inactive, v32 passthrough).
# seq4k: sparse path active on both sides (device indexer+sparse_mla vs CPU index mask).
@pytest.mark.parametrize("seq_len", [256, 2048, 4096], ids=["seq256", "seq2k", "seq4k"])
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.timeout(0)
def test_v32_mla_vs_cpu_reference(mesh_device, seq_len, device_params, variant, config_only):
    config = config_only
    seed = 42
    args, mla_cpu, weights = build_cpu_reference(seq_len, seed)
    assert_config_matches(config, args)
    config.max_seq_len = seq_len  # rope table length (same hack as v3 run_model)

    mesh_shape = list(mesh_device.shape)
    sp_axis, tp_axis = 0, 1

    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    tt_output, hidden_states, _, shard_dims = v3_test_mla.run_mla_inference(
        config=config,
        weights=weights,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        topology=ttnn.Topology.Linear,
        tt_kvpe_cache=tt_kvpe_cache,
    )

    ref_output, ref_kvpe = run_cpu_reference(
        args, mla_cpu, hidden_states, seq_len, seed, cache_tag=f"random_funcidx_max{args.max_seq_len}"
    )

    tt_output_cpu = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)
    if seq_len > 2048:  # DSA diagnostics: dense rows (<topk) vs sparse rows
        from models.common.utility_functions import comp_pcc

        for name, sl in [("rows<2048", slice(0, 2048)), ("rows>=2048", slice(2048, seq_len))]:
            _, m = comp_pcc(ref_output[:, sl], tt_output_cpu[0, :, sl], 0)
            logger.info(f"band {name}: {m}")
    _, pcc_message = assert_with_pcc(ref_output.unsqueeze(0), tt_output_cpu, OUTPUT_PCC)
    logger.info(f"Output PCC: {pcc_message}")

    # KVPE cache: replicated across TP — concat TP replicas on unused dim 1, keep first.
    tt_kvpe = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)[:1, :1]
    kv = config.kv_lora_rank
    _, kv_pcc = assert_with_pcc(ref_kvpe[..., :kv], tt_kvpe[..., :kv], KVPE_PCC)
    _, pe_pcc = assert_with_pcc(ref_kvpe[..., kv:], tt_kvpe[..., kv:], KVPE_PCC)
    logger.info(f"KVPE cache PCC: kv={kv_pcc} pe={pe_pcc}")

    ttnn.synchronize_device(mesh_device)


def run_cpu_reference_chunked(args, mla_cpu, hidden_states, seq_len, chunk, seed):
    """Chunk-loop truth on MLACPU decode branch; mask [chunk, end_pos] keeps causality."""
    cache_dir = Path(os.environ.get("DEEPSEEK_V32_MLA_REF_CACHE", "/tmp/deepseek_v32_mla_ref_cache"))
    cache_path = cache_dir / f"chunked_funcidx_seq{seq_len}_c{chunk}_seed{seed}.pt"
    if cache_path.exists():
        logger.info(f"Loading cached chunked CPU reference from {cache_path}")
        return torch.load(cache_path, weights_only=True)["ref_output"]

    freqs_all = precompute_freqs_cis(args)
    outs = []
    with torch.no_grad():
        for s in range(0, seq_len, chunk):
            mask = torch.full((chunk, s + chunk), float("-inf")).triu_(s + 1)
            outs.append(
                mla_cpu.forward(hidden_states[:, s : s + chunk].to(torch.bfloat16), s, freqs_all[s : s + chunk], mask)
            )
    ref_output = torch.cat(outs, dim=1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"ref_output": ref_output}, cache_path)
    logger.info(f"Saved chunked CPU reference to {cache_path}")
    return ref_output


# Chunked prefill e2e (step 4): chunk size is a parameter — 1k dev default (agreement 15).
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["1x4"], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "worker_l1_size": ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE if is_blackhole() else WH_WORKER_L1_SIZE,
        }
    ],
    ids=["line"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len,chunk", [(4096, 1024)], ids=["4k_c1k"])
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.timeout(0)
def test_v32_mla_chunked_vs_cpu_reference(mesh_device, seq_len, chunk, device_params, variant, config_only):
    from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup

    config = config_only
    seed = 42
    args, mla_cpu, weights = build_cpu_reference(seq_len, seed)
    assert_config_matches(config, args)
    config.max_seq_len = seq_len

    mesh_shape = list(mesh_device.shape)
    sp_axis, tp_axis = 0, 1
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_len,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
    )

    mla_tt = ttMLAv32(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=chunk,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        layer_num=1,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope.get_rope_tensors_indexed(seq_len, chunk)

    torch.manual_seed(seed)
    hidden = torch.randn(1, seq_len, config.hidden_size).to(torch.bfloat16)
    shard_dims = [None, None]
    shard_dims[tp_axis], shard_dims[sp_axis] = -1, -2

    outs = []
    for s in range(0, seq_len, chunk):
        tt_x = ttnn.from_torch(
            hidden[:, s : s + chunk].unsqueeze(0),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=shard_dims),
        )
        out = mla_tt.forward(tt_x, rope_tensors, tt_kvpe_cache, kv_actual_isl=s)
        outs.append(
            ttnn.to_torch(
                out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape)
            ).to(torch.bfloat16)
        )
    tt_output = torch.cat(outs, dim=2)

    ref_output = run_cpu_reference_chunked(args, mla_cpu, hidden, seq_len, chunk, seed)
    _, pcc_message = assert_with_pcc(ref_output.unsqueeze(0), tt_output, OUTPUT_PCC)
    logger.info(f"Chunked output PCC: {pcc_message}")
    ttnn.synchronize_device(mesh_device)
