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
from models.demos.deepseek_v32.reference_cpu.weights import DEFAULT_REPO, initialize_weights
from models.demos.deepseek_v32.tests.mesh_utils import (
    parametrize_mesh_device,
    skip_if_seq_too_small_for_sp,
    skip_if_tp1_dense_mla,
)
from models.demos.deepseek_v32.tt.mla import ttMLA as ttMLAv32
from tests.ttnn.utils_for_testing import assert_with_pcc

# e2e/correctness tests — pre-commit/CI (CPU truths must be cached). seq256 also
# carries `dev` (see its param mark) so `-m dev` gets a fast e2e smoke.
pytestmark = pytest.mark.gate

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


def build_cpu_reference(seq_len: int, seed: int = 42, layer=None, checkpoint_path=None, repo=None):
    """MLACPU + v3-format weights dict. Returns (args, model, weights, src_tag).

    Weights: random (default) or pretrained layer `layer` (HF repo, or local
    `checkpoint_path` shards). src_tag identifies the source for ref-cache keys
    so random and pretrained truths never collide.
    """
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
    """MLA/indexer input [1, seq, hidden] bf16: from --ds-input file (sliced/checked)
    or deterministic randn(seed)."""
    if input_path:
        t = torch.load(input_path, weights_only=True)
        t = t["hidden_states"] if isinstance(t, dict) else t
        t = t.reshape(-1, t.shape[-1])  # [.., hidden] -> [tokens, hidden]
        assert (
            t.shape[0] >= seq_len and t.shape[-1] == hidden_size
        ), f"--ds-input {tuple(t.shape)} can't supply [{seq_len}, {hidden_size}]"
        return t[:seq_len].reshape(1, seq_len, hidden_size).to(torch.bfloat16)
    torch.manual_seed(seed)
    return torch.randn(1, seq_len, hidden_size, dtype=torch.bfloat16)


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


# Mesh shapes adapt to the box (mesh_utils): single chip / TP=4 / SP2×TP2 on a
# QuietBox; SP=8 / SP4×TP2 / SP2×TP4 on a LoudBox; SP8×TP4 on Galaxy (prod).
@parametrize_mesh_device()
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
@pytest.mark.parametrize(
    "seq_len", [pytest.param(256, marks=pytest.mark.dev), 2048, 4096], ids=["seq256", "seq2k", "seq4k"]
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.timeout(0)
def test_v32_mla_vs_cpu_reference(
    mesh_device, seq_len, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo
):
    # NOTE: input is generated inside v3 run_mla_inference (seeded); --ds-input does
    # not apply here. Use the chunked test for file-driven input.
    skip_if_seq_too_small_for_sp(seq_len, mesh_device)
    skip_if_tp1_dense_mla(seq_len, mesh_device)  # dense path overflows L1 at TP=1
    config = config_only
    seed = 42
    args, mla_cpu, weights, src_tag = build_cpu_reference(seq_len, seed, ds_layer, ds_checkpoint, ds_repo)
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
        args, mla_cpu, hidden_states, seq_len, seed, cache_tag=f"{src_tag}_funcidx_max{args.max_seq_len}"
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


# Determinism (backlog 18): same weights + same input run N times must give the
# SAME device output. Guards run-to-run non-determinism in the DSA path (CCL
# reductions, host fallback, topk ties). seq4k exercises the active DSA path.
# No CPU truth needed → fast (no cold reference).
@parametrize_mesh_device()
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
@pytest.mark.parametrize("seq_len", [4096], ids=["seq4k"])
@pytest.mark.parametrize("n_runs", [3], ids=["x3"])
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.timeout(0)
def test_v32_mla_determinism(mesh_device, seq_len, n_runs, device_params, variant, config_only):
    skip_if_seq_too_small_for_sp(seq_len, mesh_device)
    config = config_only
    args, _, weights, _ = build_cpu_reference(seq_len)
    config.max_seq_len = seq_len
    mesh_shape = list(mesh_device.shape)
    sp_axis, tp_axis = 0, 1

    outs = []
    for run in range(n_runs):
        tt_kvpe_cache = init_kvpe_cache(
            kvpe_cache_head_dim=config.kv_lora_rank + config.qk_rope_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_len,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=1,
        )
        # dict(weights): the v32 ttMLA pops indexer keys, so each run needs a fresh dict.
        tt_output, _, _, shard_dims = v3_test_mla.run_mla_inference(
            config=config,
            weights=dict(weights),
            mesh_device=mesh_device,
            seq_len=seq_len,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            is_balanced=False,
            topology=ttnn.Topology.Linear,
            tt_kvpe_cache=tt_kvpe_cache,
        )
        outs.append(
            ttnn.to_torch(
                tt_output,
                mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=shard_dims, mesh_shape=mesh_device.shape),
            ).to(torch.bfloat16)
        )
        ttnn.synchronize_device(mesh_device)

    for i in range(1, n_runs):
        exact = torch.equal(outs[0], outs[i])
        _, msg = assert_with_pcc(outs[0].float(), outs[i].float(), 0.9999)
        logger.info(f"determinism run0 vs run{i}: exact={exact} pcc={msg}")


def run_cpu_reference_chunked(args, mla_cpu, hidden_states, seq_len, chunk, src_tag):
    """Chunk-loop truth on MLACPU decode branch; mask [chunk, end_pos] keeps causality."""
    cache_dir = Path(os.environ.get("DEEPSEEK_V32_MLA_REF_CACHE", "/tmp/deepseek_v32_mla_ref_cache"))
    cache_path = cache_dir / f"chunked_{src_tag}_funcidx_seq{seq_len}_c{chunk}_v2.pt"
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
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"ref_output": ref_output, "ref_kvpe": ref_kvpe}, cache_path)
    logger.info(f"Saved chunked CPU reference to {cache_path}")
    return ref_output, ref_kvpe


# Chunked prefill e2e (step 4 + 5.5): chunk size is a parameter — 1k dev default (agreement 15).
@parametrize_mesh_device()
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
def test_v32_mla_chunked_vs_cpu_reference(
    mesh_device, seq_len, chunk, device_params, variant, config_only, ds_layer, ds_checkpoint, ds_repo, ds_input
):
    from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup

    skip_if_seq_too_small_for_sp(seq_len, mesh_device)
    # The chunked epilogue (nlp_concat_heads over all 128 heads) needs TP head-sharding to
    # fit L1: at TP=1 the unsharded 128-head concat overflows (~2.2 MB > 1.57 MB). The
    # single-shot path fits the same config, so only the chunked path is gated here.
    if mesh_device.shape[1] == 1:
        pytest.skip("chunked MLA epilogue (nlp_concat_heads, 128 heads) exceeds L1 without TP head-sharding (TP=1)")
    config = config_only
    seed = 42
    args, mla_cpu, weights, src_tag = build_cpu_reference(seq_len, seed, ds_layer, ds_checkpoint, ds_repo)
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

    # seq_len sizes the persistent ring buffers — full cache, not chunk.
    mla_tt = ttMLAv32(
        config,
        weights,
        mesh_device,
        layer_idx=0,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_chunked=True,
        layer_num=1,
    )
    rope = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    rope_tensors = rope.get_rope_tensors_indexed(seq_len, chunk)

    hidden = make_hidden(seq_len, config.hidden_size, seed, ds_input)
    # File-driven input changes the truth → fold a stable tag into the ref-cache key.
    if ds_input:
        src_tag = f"{src_tag}_in{abs(hash(ds_input)) % 10**8}"
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

    ref_output, ref_kvpe = run_cpu_reference_chunked(args, mla_cpu, hidden, seq_len, chunk, src_tag)
    from models.common.utility_functions import comp_pcc

    for s in range(0, seq_len, chunk):
        _, m = comp_pcc(ref_output[:, s : s + chunk], tt_output[0, :, s : s + chunk], 0)
        logger.info(f"chunk@{s}: {m}")
    # KVPE prefix diagnostic (SP-aware): read TP replica 0, un-rotate the block-cyclic
    # chunked cache to natural order with blockcyclic_positions, compare to reference.
    from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions

    sp = mesh_device.shape[0]
    cache_sr = ttnn.to_torch(
        tt_kvpe_cache, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)[
        :, :1
    ]  # [slots, 1, seq_cache, kvpe]
    p = blockcyclic_positions(sp, chunk, cache_sr.shape[2])
    nat = torch.empty(cache_sr.shape[2], cache_sr.shape[-1], dtype=torch.bfloat16)
    nat[p] = cache_sr[0, 0]
    _, m = comp_pcc(ref_kvpe, nat[:seq_len].unsqueeze(0), 0)
    logger.info(f"kvpe prefix: {m}")
    _, pcc_message = assert_with_pcc(ref_output.unsqueeze(0), tt_output, OUTPUT_PCC)
    logger.info(f"Chunked output PCC: {pcc_message}")
    ttnn.synchronize_device(mesh_device)
