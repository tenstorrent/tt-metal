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
# seq <= index_topk keeps the CPU reference dense-equivalent (see module docstring).
# seq256: smoke run (cheap CPU ref, fast iteration); seq2k: largest dense-equivalent.
@pytest.mark.parametrize("seq_len", [256, 2048], ids=["seq256", "seq2k"])
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
        args, mla_cpu, hidden_states, seq_len, seed, cache_tag=f"random_max{args.max_seq_len}"
    )

    tt_output_cpu = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape),
    ).to(torch.bfloat16)
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
