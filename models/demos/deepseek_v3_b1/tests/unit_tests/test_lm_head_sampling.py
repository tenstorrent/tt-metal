# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN LM Head Sampling CCL Broadcast + Mcast + Matmul Op Test

In multi-device mode: CCL broadcasts input_a [1, 7168] from sender device to all
devices, then on each device the sender core multicasts to 101 matmul cores.
Each matmul core holds a weight shard [7168, N_per_core] and computes
[1, 7168] x [7168, N_per_core] -> [1, N_per_core].
Output stays width-sharded across matmul cores.

In single-device mode (skip_ccl=True): CCL is skipped and the input is used directly.
"""

import os

import pytest
import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.common.utility_functions import is_slow_dispatch
from models.demos.deepseek_v3_b1.demo.pipeline import (
    PipelineConfiguration,
    create_single_galaxy_pipeline_configuration,
    create_single_galaxy_spec_decode_pipeline_configuration,
)
from models.demos.deepseek_v3_b1.demo.stage import (
    TOKEN_META_PAGE_SIZE_BYTES,
    TOKEN_PAGE_SIZE_BYTES,
    BaseLMHeadStage,
    EmbeddingStage,
    PassthroughPayload,
    PassthroughStage,
)
from models.demos.deepseek_v3_b1.demo.weight_provider import StateDictWeightProvider
from models.demos.deepseek_v3_b1.fused_ops.lm_head_sampling.op import LMHeadSampling
from models.demos.deepseek_v3_b1.micro_ops.d2d_exchange.op import MeshWrapper, SocketInterface
from models.demos.deepseek_v3_b1.micro_ops.host_io.op import HostInterface
<<<<<<< HEAD
from models.demos.deepseek_v3_b1.prepare_weights import prepare_embedding_weights, prepare_lm_head_weights
from models.demos.deepseek_v3_b1.tests.unit_tests.ccl_test_utils import (
    build_broadcast_test_inputs,
    create_fabric_router_config,
)
=======
from models.demos.deepseek_v3_b1.prepare_weights import (
    DeepSeekV3MTPWeights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles
>>>>>>> 6ffd11b80d (mtp 2 wip)
from models.perf.benchmarking_utils import BenchmarkProfiler
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc

<<<<<<< HEAD
=======

def create_fabric_router_config(max_payload_size):
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


def _create_mcast_working_bufs(
    device, mcast_core, matmul_core_grid, M, K, a_tile, embedding_dim=None, num_devices=1, mesh_mapper=None
):
    """Allocate HEIGHT_SHARDED working buffer tensors on the mcast receiver grid (bounding box minus sender).

    Returns (mcast_dst_buf, mcast_eh_dst_buf).
    mcast_eh_dst_buf is None when embedding_dim is None (MTP disabled).
    """
    matmul_bbox = matmul_core_grid.bounding_box()
    mcast_end_x = max(matmul_bbox.end.x, mcast_core.x)
    mcast_end_y = max(matmul_bbox.end.y, mcast_core.y)

    receiver_ranges = []
    if mcast_core.y > 0:
        receiver_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(mcast_end_x, mcast_core.y - 1)))
    if mcast_core.x > 0:
        receiver_ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(0, mcast_core.y), ttnn.CoreCoord(mcast_core.x - 1, mcast_core.y))
        )
    if mcast_core.x < mcast_end_x:
        receiver_ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(mcast_core.x + 1, mcast_core.y), ttnn.CoreCoord(mcast_end_x, mcast_core.y))
        )
    if mcast_core.y < mcast_end_y:
        receiver_ranges.append(
            ttnn.CoreRange(ttnn.CoreCoord(0, mcast_core.y + 1), ttnn.CoreCoord(mcast_end_x, mcast_end_y))
        )
    receiver_grid = ttnn.CoreRangeSet(receiver_ranges)
    num_receiver_cores = (mcast_end_x + 1) * (mcast_end_y + 1) - 1

    dst_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(receiver_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    from_torch_kwargs = dict(
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=a_tile, device=device, memory_config=dst_mem_config
    )
    if mesh_mapper is not None:
        from_torch_kwargs["mesh_mapper"] = mesh_mapper
    mcast_dst_buf = ttnn.from_torch(
        torch.zeros((num_devices * num_receiver_cores, K), dtype=torch.bfloat16),
        **from_torch_kwargs,
    )

    mcast_eh_dst_buf = None
    if embedding_dim is not None:
        eh_k = K + embedding_dim
        eh_dst_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(receiver_grid, (M, eh_k), ttnn.ShardOrientation.ROW_MAJOR),
        )
        eh_from_torch_kwargs = dict(
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, tile=a_tile, device=device, memory_config=eh_dst_mem_config
        )
        if mesh_mapper is not None:
            eh_from_torch_kwargs["mesh_mapper"] = mesh_mapper
        mcast_eh_dst_buf = ttnn.from_torch(
            torch.zeros((num_devices * num_receiver_cores, eh_k), dtype=torch.bfloat16),
            **eh_from_torch_kwargs,
        )

    return mcast_dst_buf, mcast_eh_dst_buf


>>>>>>> 6ffd11b80d (mtp 2 wip)
# Synthetic weight provider: same layout as prepare_* (state dict + move_to_device); used for pipeline tests.
_VOCAB_SIZE = 129280
_EMBED_HIDDEN = 7168
_LM_HEAD_N_SYNTHETIC = 101 * 160  # 16160
_REAL_WEIGHTS_PERSISTENT_INPUT_TOKEN_SEED = 42


class _SyntheticWeightProvider:
    """Provider that creates deterministic synthetic embedding and LM head weights (one-hot / winner_per_row)."""

    def load_embedding(self, device):
        w = torch.zeros((_VOCAB_SIZE, _EMBED_HIDDEN), dtype=torch.bfloat16)
        w[torch.arange(_VOCAB_SIZE), torch.arange(_VOCAB_SIZE, dtype=torch.int64) % _EMBED_HIDDEN] = 1
        return prepare_embedding_weights({"model.embed_tokens.weight": w}, device, move_to_device=True)

    def load_lm_head(self, device):
        lm_w = torch.full((_VOCAB_SIZE, _EMBED_HIDDEN), -1.0, dtype=torch.bfloat16)
        lm_w[torch.arange(_EMBED_HIDDEN, dtype=torch.int64) % _LM_HEAD_N_SYNTHETIC, torch.arange(_EMBED_HIDDEN)] = 1
        return prepare_lm_head_weights(
            {"lm_head.weight": lm_w, "model.norm.weight": torch.ones(_EMBED_HIDDEN, dtype=torch.bfloat16)},
            device,
            move_to_device=True,
        )

    def load_mtp_weights(self, device):
        M = 1
        K = _EMBED_HIDDEN
        embedding_dim = _EMBED_HIDDEN
        mtp_output_dim = _EMBED_HIDDEN
        tile_width = 32
        num_dram_banks = 8
        mtp_n_per_core = mtp_output_dim // num_dram_banks
        mtp_padded_dim = num_dram_banks * mtp_n_per_core
        num_matmul_cores = 101
        n_per_core = 160
        n_total = num_matmul_cores * n_per_core
        num_devices = device.shape[0] * device.shape[1]

        a_tile = ttnn.Tile([1, 32])
        b_tile = ttnn.Tile([32, 32])

        mcast_core = ttnn.CoreCoord(10, 9)
        mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
        input_a_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
        )

        torch.manual_seed(42)
        torch_embedding = torch.randn((num_devices * n_total, embedding_dim), dtype=torch.bfloat16)
        torch_h_gamma = torch.randn((M, K), dtype=torch.bfloat16)
        torch_e_gamma = torch.randn((M, embedding_dim), dtype=torch.bfloat16)
        torch_eh_proj = torch.randn((K + embedding_dim, mtp_output_dim), dtype=torch.bfloat16)
        torch_eh_proj_padded = torch.zeros((K + embedding_dim, mtp_padded_dim), dtype=torch.bfloat16)
        torch_eh_proj_padded[:, :mtp_output_dim] = torch_eh_proj

        ttnn_embedding = ttnn.from_torch(
            torch_embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        ttnn_h_gamma = ttnn.from_torch(
            torch_h_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_a_mem_config,
            tile=a_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        ttnn_e_gamma = ttnn.from_torch(
            torch_e_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=input_a_mem_config,
            tile=a_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        eh_shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                )
            }
        )
        eh_proj_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(eh_shard_grid, (K + embedding_dim, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
        )
        torch_eh_proj_shuffled = shuffle_tensor_tiles(torch_eh_proj_padded, tile_width, num_dram_banks)
        ttnn_eh_proj = ttnn.from_torch(
            torch_eh_proj_shuffled,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=eh_proj_mem_config,
            tile=b_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )

        return DeepSeekV3MTPWeights(
            embedding=ttnn_embedding,
            h_gamma=ttnn_h_gamma,
            e_gamma=ttnn_e_gamma,
            eh_projection=ttnn_eh_proj,
        )


def create_single_pod_passthrough_pipeline_configuration(
    weight_provider,
    *,
    lm_head_fp32_dest_acc_en: bool = True,
    lm_head_persistent_mode: bool = True,
) -> PipelineConfiguration:
    """16-stage pod topology with passthrough middle stages for LM-head-focused synthetic testing."""

    def stage_0(device):
        return EmbeddingStage(weight_provider.load_embedding(device))

    def stage_14(device):
        return BaseLMHeadStage(
            weights=weight_provider.load_lm_head(device),
            fp32_dest_acc_en=lm_head_fp32_dest_acc_en,
            persistent_mode=lm_head_persistent_mode,
        )

    return PipelineConfiguration(
        {
            0: stage_0,
            **{i: (lambda d: PassthroughStage(PassthroughPayload.ACTIVATION)) for i in range(1, 14)},
            14: stage_14,
            15: (lambda d: PassthroughStage(PassthroughPayload.TOKEN)),
        }
    )


# Golden helper: same deterministic formula as _SyntheticWeightProvider (one-hot embedding, winner_per_row).
def _compute_expected_lm_head_indices_synthetic(iterations: int) -> torch.Tensor:
    """Compute expected output indices for synthetic weights. Same math as _SyntheticWeightProvider."""
    K = 7168
    n_total = 101 * 160
    torch_gamma = torch.ones((1, K), dtype=torch.bfloat16)
    row_indices = torch.arange(iterations, dtype=torch.int64) % K
    torch_embedding_table = torch.zeros((iterations, K), dtype=torch.bfloat16)
    torch_embedding_table[torch.arange(iterations), row_indices] = 1
    winner_per_row = torch.arange(K, dtype=torch.int64) % n_total
    torch_b = torch.full((K, n_total), fill_value=-1.0, dtype=torch.bfloat16)
    torch_b[torch.arange(K), winner_per_row] = 1
    torch_indices_flat = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
    torch_expected_indices = torch.stack(
        [
            LMHeadSampling.golden(
                torch_embedding_table[iteration : iteration + 1].float(),
                torch_gamma.float(),
                torch_b.float().unsqueeze(0),
                indices=torch_indices_flat,
                k=1,
                p=1.0,
            )[0].to(torch.uint32)
            for iteration in range(iterations)
        ],
        dim=0,
    )
    return torch_expected_indices


def _hf_functional_lm_logits_flat(
    embed_1h: torch.Tensor,
    norm_w: torch.Tensor,
    lm_w: torch.Tensor,
    *,
    epsilon: float = 1e-6,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """HuggingFace-style last-step logits: RMSNorm(hidden) then ``logits = hidden @ lm_head.weight.T``.

    Same as ``nn.Linear(hidden, vocab)`` with weight ``lm_w`` of shape ``(vocab, hidden)`` and no bias:
    ``logits = x @ lm_w.mT`` with ``x`` shape ``(1, hidden)``. RMSNorm matches the usual HF pattern
    ``x * rsqrt(mean(x**2) + eps) * weight`` with ``model.norm.weight`` shape ``(hidden,)``.
    """
    x = embed_1h.to(dtype)
    nw = norm_w.to(dtype)
    lw = lm_w.to(dtype)
    var = x.pow(2).mean(-1, keepdim=True)
    eps_t = torch.tensor(epsilon, device=x.device, dtype=dtype)
    x = x * torch.rsqrt(var + eps_t)
    x = x * nw.unsqueeze(0)
    logits = x @ lw.mT
    return logits.reshape(-1)


def _topk_vocab_ids_from_scores(scores_flat: torch.Tensor, k: int = 10) -> torch.Tensor:
    k_eff = min(k, int(scores_flat.numel()))
    _, idx = torch.topk(scores_flat, k_eff, largest=True, sorted=True)
    return idx.to(torch.uint32)


def _compute_reference_topk_token_ids_real(state_dict, input_token_ids: torch.Tensor, topk: int = 10) -> torch.Tensor:
    """Per row of ``input_token_ids`` (vocab ids into ``embed_tokens``), top-`topk` output vocab ids from HF logits.

    ``input_token_ids`` shape ``(n,)`` with values in ``[0, vocab_size)``. Returns shape ``(n, topk)``.
    """
    embed_w = state_dict["model.embed_tokens.weight"]
    norm_w = state_dict["model.norm.weight"]
    lm_w = state_dict["lm_head.weight"]
    K = 7168
    assert embed_w.shape == (_VOCAB_SIZE, K), f"Unexpected embed shape {embed_w.shape}"
    assert norm_w.shape == (K,), f"Unexpected norm shape {norm_w.shape}"
    assert lm_w.shape == (_VOCAB_SIZE, K), f"Unexpected lm_head shape {lm_w.shape}"
    rows = []
    for in_tok in input_token_ids.tolist():
        tid = int(in_tok)
        assert 0 <= tid < _VOCAB_SIZE, f"input token id {tid} out of range"
        scores_flat = _hf_functional_lm_logits_flat(
            embed_w[tid : tid + 1],
            norm_w,
            lm_w,
        )
        rows.append(_topk_vocab_ids_from_scores(scores_flat, k=topk))
    return torch.stack(rows, dim=0)


def _real_weights_topk_table_row(
    embed_w: torch.Tensor,
    norm_w: torch.Tensor,
    lm_w: torch.Tensor,
    in_tok: int,
    got_id: int,
    top_ids: list[int],
) -> tuple[int, float, str]:
    """One table row: ref_rank (1-based), Δ@got vs ref top-1, space-separated top-k ids string."""
    scores_flat = _hf_functional_lm_logits_flat(
        embed_w[in_tok : in_tok + 1],
        norm_w,
        lm_w,
    )
    order = torch.argsort(scores_flat, descending=True)
    rank = int((order == got_id).nonzero(as_tuple=True)[0][0].item()) + 1
    top1 = int(top_ids[0])
    logit_got = float(scores_flat[got_id].float().item())
    logit_top1 = float(scores_flat[top1].float().item())
    delta = logit_got - logit_top1
    topk_str = " ".join(str(t) for t in top_ids)
    return rank, delta, topk_str


def _format_real_weights_topk_results_table(
    state_dict,
    input_token_ids: torch.Tensor,
    got_flat: torch.Tensor,
    ref_topk: torch.Tensor,
    *,
    topk: int,
) -> str:
    """Full per-iteration table: same columns as mismatch report, plus in_topk (Y/N). Always logged after the test run."""
    embed_w = state_dict["model.embed_tokens.weight"]
    norm_w = state_dict["model.norm.weight"]
    lm_w = state_dict["lm_head.weight"]
    iterations = int(got_flat.numel())
    assert int(input_token_ids.numel()) == iterations
    lines = [
        "",
        "=" * 88,
        f"REAL WEIGHTS top-{topk} RESULTS (all {iterations} iterations)",
        "=" * 88,
        "",
        "  iter = pipeline loop index; in_tok = random input vocab id written to H2D (embed lookup row).",
        "  got = vocab id from device; ref_rank = 1-based rank of `got` in HF functional bf16 logits (descending).",
        "  Δ@got = logit(got) − logit(ref_top1), float32 view of bf16 scores (negative ⇒ below best).",
        "  in_topk = Y if `got` is in the reference top-k list, else N.",
        "  Reference = RMSNorm(embed) then x @ lm_head.weight.T (HuggingFace-style).",
        "",
        f"{'iter':>5}  {'in_tok':>7}  {'got':>8}  {'ref_rank':>9}  {'Δ@got':>10}  {'in_topk':>7}  reference top-{topk} (best → …)",
        "-" * 88,
    ]
    for i in range(iterations):
        in_tok = int(input_token_ids[i].item())
        got_id = int(got_flat[i].item())
        top_ids = [int(x) for x in ref_topk[i].tolist()]
        rank, delta, topk_str = _real_weights_topk_table_row(embed_w, norm_w, lm_w, in_tok, got_id, top_ids)
        in_top = "Y" if got_id in top_ids else "N"
        lines.append(f"{i:5d}  {in_tok:7d}  {got_id:8d}  {rank:9d}  {delta:10.4f}  {in_top:>7}  {topk_str}")
    lines.extend(["-" * 88, ""])
    return "\n".join(lines)


def _format_real_weights_topk_mismatch_report(
    state_dict,
    mismatches: list[tuple[int, int, int, list[int]]],
    *,
    topk: int,
    total_iters: int,
) -> str:
    """Human-readable report for top-k failures: ranks, logits, one row per bad iteration.

    Each mismatch is ``(iter_idx, in_tok, got_id, top_ids)``.
    """
    embed_w = state_dict["model.embed_tokens.weight"]
    norm_w = state_dict["model.norm.weight"]
    lm_w = state_dict["lm_head.weight"]
    lines = [
        "",
        "=" * 80,
        f"REAL WEIGHTS top-{topk} CHECK: {len(mismatches)} failing iteration(s) out of {total_iters}",
        "=" * 80,
        "",
        "  iter = pipeline loop index; in_tok = input vocab id written to H2D (embed lookup row).",
        "  got = vocab id from device; ref_rank = 1-based rank of `got` in HF functional bf16 logits (descending).",
        "  Δ@got = logit(got) − logit(ref_top1), float32 view of bf16 scores (negative ⇒ below best).",
        "  Reference = RMSNorm(embed) then x @ lm_head.weight.T (HuggingFace-style).",
        "",
        f"{'iter':>5}  {'in_tok':>7}  {'got':>8}  {'ref_rank':>9}  {'Δ@got':>10}  reference top-{topk} (best → …)",
        "-" * 80,
    ]
    for iter_idx, in_tok, got_id, top_ids in mismatches:
        rank, delta, topk_str = _real_weights_topk_table_row(embed_w, norm_w, lm_w, in_tok, got_id, top_ids)
        lines.append(f"{iter_idx:5d}  {in_tok:7d}  {got_id:8d}  {rank:9d}  {delta:10.4f}  {topk_str}")
    lines.extend(
        [
            "-" * 80,
            "",
        ]
    )
    return "\n".join(lines)


def _compute_expected_spec_decode_tokens_synthetic(iterations: int):
    """Compute expected (base_token, spec_token) pairs for the 4-stage MTP pipeline.

    Full golden chain:
      1. Embedding lookup (one-hot table, token → activation)
      2. LM head sampling (base stage) → base_token + MTP logits
      3. LM head sampling (verify stage on MTP logits) → spec_token
    Returns list of (base_token, spec_token) tuples.
    """
    K = _EMBED_HIDDEN
    n_total = _LM_HEAD_N_SYNTHETIC
    num_devices = 8

    # Base stage weights (same as _SyntheticWeightProvider)
    torch_gamma = torch.ones((1, K), dtype=torch.bfloat16)
    winner_per_row = torch.arange(K, dtype=torch.int64) % n_total
    torch_b = torch.full((K, n_total), fill_value=-1.0, dtype=torch.bfloat16)
    torch_b[torch.arange(K), winner_per_row] = 1
    torch_indices_flat = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)

    # MTP weights (seed=42, same as _SyntheticWeightProvider.load_mtp_weights)
    embedding_dim = K
    mtp_output_dim = K
    torch.manual_seed(42)
    torch_embedding = torch.randn((num_devices * n_total, embedding_dim), dtype=torch.bfloat16)
    torch_h_gamma = torch.randn((1, K), dtype=torch.bfloat16)
    torch_e_gamma = torch.randn((1, embedding_dim), dtype=torch.bfloat16)
    torch_eh_proj = torch.randn((K + embedding_dim, mtp_output_dim), dtype=torch.bfloat16)

    results = []
    for iteration in range(iterations):
        # Step 1: Embedding lookup (one-hot table: token_id → row with 1.0 at position token_id % K)
        row_idx = iteration % K
        torch_input = torch.zeros((1, K), dtype=torch.bfloat16)
        torch_input[0, row_idx] = 1

        # Step 2: Base stage LM head sampling + MTP fusion → base_token + mtp_logits
        base_token_tensor, mtp_logits = LMHeadSampling.golden(
            torch_input.float(),
            torch_gamma.float(),
            torch_b.float().unsqueeze(0),
            indices=torch_indices_flat,
            k=1,
            p=1.0,
            fuse_mtp=True,
            embedding_tensor=torch_embedding.float(),
            h_gamma_tensor=torch_h_gamma.float(),
            e_gamma_tensor=torch_e_gamma.float(),
            eh_projection_tensor=torch_eh_proj.float(),
        )
        base_token = base_token_tensor.to(torch.uint32).item()

        # Step 3: Verify stage LM head sampling on MTP logits → spec_token
        spec_token_tensor, _ = LMHeadSampling.golden(
            mtp_logits,
            torch_gamma.float(),
            torch_b.float().unsqueeze(0),
            indices=torch_indices_flat,
            k=1,
            p=1.0,
        )
        spec_token = spec_token_tensor.to(torch.uint32).item()

        results.append((base_token, spec_token))
    return results


def _is_lm_head_sampling_perf_enabled():
    return os.getenv("RUN_LM_HEAD_SAMPLING_PERF", "0") == "1"


def _is_persistent_mode_enabled():
    return os.getenv("TT_RUN_PERSISTENT_MODE", "0") == "1"


@pytest.mark.skipif(not _is_lm_head_sampling_perf_enabled(), reason="Set RUN_LM_HEAD_SAMPLING_PERF=1 to run perf test")
@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1), (1, 0), (2, 1), (2, 0)])
@pytest.mark.parametrize("num_iters,num_warmup_iters", [(20, 6)])
@pytest.mark.parametrize("enable_mtp", [False, True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 1600000,
        }
    ],
    indirect=True,
)
<<<<<<< HEAD
def test_perf(bh_2d_mesh_device, use_fp32, final_mesh_coord, num_iters, num_warmup_iters, device_params):
=======
def test_perf(bh_2d_mesh_device, use_fp32, final_mesh_coord, num_iters, num_warmup_iters, enable_mtp):
    """Performance test for LM-head sampling with optional MTP fusion.

    When enable_mtp=True, also runs:
    - Embedding lookup from argmax output token
    - h_rmsnorm and e_rmsnorm
    - Concat [h_norm|e_norm]
    - EH projection DRAM streaming matmul
    """
>>>>>>> 6ffd11b80d (mtp 2 wip)
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    seed = 7

    # MTP dimensions
    embedding_dim = 7168
    mtp_output_dim = 7168
    tile_width = 32
    num_dram_banks = 8
    mtp_n_per_core = mtp_output_dim // num_dram_banks
    mtp_padded_dim = num_dram_banks * mtp_n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)

    # MTP tensors (only used when enable_mtp=True)
    # Embedding table must cover all possible token IDs (num_devices * n_total for global indices)
    torch_embedding = torch.randn((num_devices * n_total, embedding_dim), dtype=torch.bfloat16) if enable_mtp else None
    torch_h_gamma = torch.randn((M, K), dtype=torch.bfloat16) if enable_mtp else None
    torch_e_gamma = torch.randn((M, embedding_dim), dtype=torch.bfloat16) if enable_mtp else None
    torch_eh_proj = torch.randn((K + embedding_dim, mtp_output_dim), dtype=torch.bfloat16) if enable_mtp else None
    torch_eh_proj_padded = None
    if enable_mtp:
        torch_eh_proj_padded = torch.zeros((K + embedding_dim, mtp_padded_dim), dtype=torch.bfloat16)
        torch_eh_proj_padded[:, :mtp_output_dim] = torch_eh_proj

    torch_expected_idx, torch_mtp_output = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
        fuse_mtp=enable_mtp,
        embedding_tensor=torch_embedding.float() if enable_mtp else None,
        h_gamma_tensor=torch_h_gamma.float() if enable_mtp else None,
        e_gamma_tensor=torch_e_gamma.float() if enable_mtp else None,
        eh_projection_tensor=torch_eh_proj.float() if enable_mtp else None,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes + (256 + 8 if enable_mtp else 0)) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    # MTP-specific memory configs
    compute_cores = submesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0) if enable_mtp else None
    compute_core_grid = (
        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores])
        if enable_mtp
        else None
    )
    eh_shard_grid = (
        ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(submesh.dram_grid_size().x - 1, submesh.dram_grid_size().y - 1),
                )
            }
        )
        if enable_mtp
        else None
    )
    mtp_output_mem_config = (
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(compute_core_grid, (M, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
        )
        if enable_mtp
        else None
    )
    eh_proj_mem_config = (
        ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(eh_shard_grid, (K + embedding_dim, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
        )
        if enable_mtp
        else None
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=mcast_core,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
        input_tensor_torch=torch_a,
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    # MTP tensors
    ttnn_embedding = None
    ttnn_h_gamma = None
    ttnn_e_gamma = None
    ttnn_eh_proj = None
    ttnn_mtp_output = None
    if enable_mtp:
        ttnn_embedding = ttnn.from_torch(
            torch_embedding,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=submesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        ttnn_h_gamma = ttnn.from_torch(
            torch_h_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=input_a_mem_config,
            tile=a_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        ttnn_e_gamma = ttnn.from_torch(
            torch_e_gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=input_a_mem_config,
            tile=a_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        torch_eh_proj_shuffled = shuffle_tensor_tiles(torch_eh_proj_padded, tile_width, num_dram_banks)
        ttnn_eh_proj = ttnn.from_torch(
            torch_eh_proj_shuffled,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=eh_proj_mem_config,
            tile=b_tile,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )
        ttnn_mtp_output = ttnn.from_torch(
            torch.zeros((num_devices, M, mtp_padded_dim), dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=submesh,
            memory_config=mtp_output_mem_config,
            tile=out_tile,
            mesh_mapper=mesh_mapper,
        )

    mcast_dst_buf, mcast_eh_dst_buf = _create_mcast_working_bufs(
        submesh,
        mcast_core,
        matmul_core_grid,
        M,
        K,
        a_tile,
        embedding_dim=embedding_dim if enable_mtp else None,
        num_devices=num_devices,
        mesh_mapper=mesh_mapper,
    )

    stage1_semaphores = [ttnn.create_global_semaphore(submesh, final_core_grid, 0) for _ in range(2)]
    stage2_semaphores = [ttnn.create_global_semaphore(submesh, final_core_grid, 0) for _ in range(2)]
    ttnn.synchronize_device(submesh)

    submesh.enable_program_cache()
    profiler = BenchmarkProfiler()

    # Initial run to compile
    _ = LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        output_mtp_tensor=ttnn_mtp_output,
        embedding_tensor=ttnn_embedding,
        h_gamma_tensor=ttnn_h_gamma,
        e_gamma_tensor=ttnn_e_gamma,
        eh_projection_tensor=ttnn_eh_proj,
        mcast_dst_working_buf_tensor=mcast_dst_buf,
        mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        bcast_semaphores=bcast_inputs.semaphores,
        global_semaphore=stage1_semaphores[0],
        global_stage2_semaphore=stage2_semaphores[0],
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
<<<<<<< HEAD
        fabric_config=device_params["fabric_config"],
=======
        enable_mtp=enable_mtp,
>>>>>>> 6ffd11b80d (mtp 2 wip)
    )
    ttnn.synchronize_device(submesh)

    trace_id_warmup = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_warmup_iters):
        _ = LMHeadSampling.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            ttnn_gamma,
            ttnn_b,
            ttnn_scores,
            output_mtp_tensor=ttnn_mtp_output,
            embedding_tensor=ttnn_embedding,
            h_gamma_tensor=ttnn_h_gamma,
            e_gamma_tensor=ttnn_e_gamma,
            eh_projection_tensor=ttnn_eh_proj,
            mcast_dst_working_buf_tensor=mcast_dst_buf,
            mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
            sender_coord=sender_coord,
            indices_tensor=ttnn_indices,
            output_index_tensor=ttnn_output_index,
            argmax_final_core_coord=final_core,
            argmax_final_mesh_coord=final_mesh_coord,
            bcast_semaphores=bcast_inputs.semaphores,
            global_semaphore=stage1_semaphores[i % 2],
            global_stage2_semaphore=stage2_semaphores[i % 2],
            fabric_scratch_tensor=ttnn_fabric_scratch,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
<<<<<<< HEAD
            fabric_config=device_params["fabric_config"],
=======
            enable_mtp=enable_mtp,
>>>>>>> 6ffd11b80d (mtp 2 wip)
        )
    ttnn.end_trace_capture(submesh, trace_id_warmup, cq_id=0)
    ttnn.synchronize_device(submesh)

    trace_id = ttnn.begin_trace_capture(submesh, cq_id=0)
    for i in range(num_iters):
        _ = LMHeadSampling.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            ttnn_gamma,
            ttnn_b,
            ttnn_scores,
            output_mtp_tensor=ttnn_mtp_output,
            embedding_tensor=ttnn_embedding,
            h_gamma_tensor=ttnn_h_gamma,
            e_gamma_tensor=ttnn_e_gamma,
            eh_projection_tensor=ttnn_eh_proj,
            mcast_dst_working_buf_tensor=mcast_dst_buf,
            mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
            sender_coord=sender_coord,
            indices_tensor=ttnn_indices,
            output_index_tensor=ttnn_output_index,
            argmax_final_core_coord=final_core,
            argmax_final_mesh_coord=final_mesh_coord,
            bcast_semaphores=bcast_inputs.semaphores,
            global_semaphore=stage1_semaphores[i % 2],
            global_stage2_semaphore=stage2_semaphores[i % 2],
            fabric_scratch_tensor=ttnn_fabric_scratch,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
<<<<<<< HEAD
            fabric_config=device_params["fabric_config"],
=======
            enable_mtp=enable_mtp,
>>>>>>> 6ffd11b80d (mtp 2 wip)
        )
    ttnn.end_trace_capture(submesh, trace_id, cq_id=0)
    ttnn.synchronize_device(submesh)

    mtp_suffix = "+MTP" if enable_mtp else ""
    profiler.start(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace-warmup")
    ttnn.execute_trace(submesh, trace_id_warmup, blocking=False)
    ttnn.release_trace(submesh, trace_id_warmup)
    ttnn.synchronize_device(submesh)
    profiler.end(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace-warmup")

    signpost("start")
    profiler.start(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace")
    ttnn.execute_trace(submesh, trace_id, blocking=False)
    ttnn.release_trace(submesh, trace_id)
    ttnn.synchronize_device(submesh)
    profiler.end(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace")
    signpost("stop")

    trace_duration_ns = profiler.get_duration(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace")
    warmup_duration_ns = profiler.get_duration(f"lm-head-sampling{mtp_suffix}-mesh-4x2-trace-warmup")
    effective_duration_ns = max(0.0, trace_duration_ns - warmup_duration_ns)
    avg_iter_ns = effective_duration_ns / float(max(1, num_iters))
    logger.info(
        f"LMHead+Argmax{mtp_suffix} mesh(4x2) trace perf: final_mesh_coord={final_mesh_coord}, "
        f"iters={num_iters}, total_ns={effective_duration_ns:.2f}, avg_iter_ns={avg_iter_ns:.2f}"
    )

    final_output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_torch = ttnn.to_torch(final_output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    logger.info(f"Final output: {final_output_torch}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        final_output_torch, torch_expected_idx
    ), f"Perf run fused mesh argmax mismatch. expected={torch_expected_idx.item()}, got={int(final_output_torch.item())}"

    # MTP PCC check
    if enable_mtp:
        assert torch_mtp_output is not None, "MTP output cannot be None"
        final_mtp_shards = ttnn.get_device_tensors(ttnn_mtp_output)
        final_mtp_torch = (
            ttnn.to_torch(final_mtp_shards[final_device_idx])
            .to(torch.float32)
            .reshape(1, mtp_padded_dim)[:, :mtp_output_dim]
        )
        mtp_passing_pcc, _ = comp_pcc(final_mtp_torch, torch_mtp_output.float(), 0.99)
        if not mtp_passing_pcc:
            max_diff = (final_mtp_torch - torch_mtp_output.float()).abs().max()
            logger.warning(f"MTP output PCC check failed. Max diff: {max_diff}")
        assert mtp_passing_pcc, "Perf run MTP output PCC check failed"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("seed", [123, 1337, 52098])
def test_single_device(
    bh_2d_mesh_device,
    use_fp32,
    seed,
    device_params,
):
    """Single-device fused LM-head + argmax sampling with pre-cached width-sharded indices."""
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(), torch_gamma.float(), torch_b.float(), indices=torch_indices, k=1, p=1.0
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)]), (1, 1), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )

    input_tensor_mesh = ttnn.from_torch(
        torch_a,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
    )

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
    )
    ttnn.synchronize_device(submesh)

    output_index_torch = ttnn.to_torch(ttnn_output_index).to(torch.uint32).reshape(1, 1)
    logger.info(f"Output index: {output_index_torch}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        output_index_torch, torch_expected_idx
    ), f"Fused argmax index mismatch. expected={torch_expected_idx.item()}, got={output_index_torch.item()}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("seed", [123, 1337])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_single_device_mtp(
    bh_2d_mesh_device,
    use_fp32,
    seed,
):
    """Single-device fused LM-head + argmax + MTP fusion test."""
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    device_grid_size = submesh.compute_with_storage_grid_size()

    M = 1
    K = 7168
    embedding_dim = 7168
    mtp_output_dim = 7168
    num_matmul_cores = 101
    n_per_core = 160
    num_dram_banks = 8
    n_total = num_matmul_cores * n_per_core
    tile_width = 32
    mtp_n_per_core = mtp_output_dim // num_dram_banks
    mtp_padded_dim = num_dram_banks * mtp_n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)

    # MTP tensors
    torch_embedding = torch.randn((n_total, embedding_dim), dtype=torch.bfloat16)
    torch_h_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_e_gamma = torch.randn((M, embedding_dim), dtype=torch.bfloat16)
    torch_eh_proj = torch.randn((K + embedding_dim, mtp_output_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded = torch.zeros((K + embedding_dim, mtp_padded_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded[:, :mtp_output_dim] = torch_eh_proj

    torch_expected_idx, torch_expected_mtp = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float(),
        indices=torch_indices,
        k=1,
        p=1.0,
        fuse_mtp=True,
        embedding_tensor=torch_embedding.float(),
        h_gamma_tensor=torch_h_gamma.float(),
        e_gamma_tensor=torch_e_gamma.float(),
        eh_projection_tensor=torch_eh_proj.float(),
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)]), (1, 1), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )
    # --- MTP specific memory configs ---
    compute_cores = submesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )
    eh_shard_grid = ttnn.CoreCoord(submesh.dram_grid_size().x - 1, submesh.dram_grid_size().y - 1)
    eh_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), eh_shard_grid)})
    mtp_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(compute_core_grid, (M, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    eh_proj_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(eh_shard_grid, (K + embedding_dim, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )

    input_tensor_mesh = ttnn.from_torch(
        torch_a,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
    )
    # --- MTP embedding tensor ---
    ttnn_embedding = ttnn.from_torch(
        torch_embedding,
        dtype=ttnn.bfloat16,
        # layout=ttnn.TILE_LAYOUT,
        device=submesh,
        # tile=b_tile,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # --- MTP rmsnorm gamma tensors ---
    ttnn_h_gamma = ttnn.from_torch(
        torch_h_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    ttnn_e_gamma = ttnn.from_torch(
        torch_e_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    # --- MTP EH matmul tensors ---
    # DRAM streaming matmul requires column-major tile order within each bank shard
    torch_eh_proj_shuffled = shuffle_tensor_tiles(torch_eh_proj_padded, tile_width, num_dram_banks)
    ttnn_eh_proj = ttnn.from_torch(
        torch_eh_proj_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=eh_proj_mem_config,
        tile=b_tile,
    )
    ttnn_mtp_output = ttnn.from_torch(
        torch.zeros((M, mtp_padded_dim), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=mtp_output_mem_config,
        tile=out_tile,
    )
    mcast_dst_buf, mcast_eh_dst_buf = _create_mcast_working_bufs(
        submesh,
        mcast_core,
        matmul_core_grid,
        M,
        K,
        a_tile,
        embedding_dim=embedding_dim,
    )
    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        output_mtp_tensor=ttnn_mtp_output,
        embedding_tensor=ttnn_embedding,
        h_gamma_tensor=ttnn_h_gamma,
        e_gamma_tensor=ttnn_e_gamma,
        eh_projection_tensor=ttnn_eh_proj,
        mcast_dst_working_buf_tensor=mcast_dst_buf,
        mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        semaphores=None,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
        enable_mtp=True,
    )
    ttnn.synchronize_device(submesh)

    output_index_torch = ttnn.to_torch(ttnn_output_index).to(torch.uint32).reshape(1, 1)
    logger.info(f"Output index: {output_index_torch}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        output_index_torch, torch_expected_idx
    ), f"Fused argmax index mismatch. expected={torch_expected_idx.item()}, got={output_index_torch.item()}"
    mtp_output_torch = ttnn.to_torch(ttnn_mtp_output).to(torch.float32).reshape(1, mtp_padded_dim)[:, :mtp_output_dim]
    logger.info(f"MTP output shape: {mtp_output_torch.shape}")
    logger.info(f"Expected MTP shape: {torch_expected_mtp.shape}")
    mtp_passing_pcc, output = comp_pcc(mtp_output_torch, torch_expected_mtp.float(), 0.99)
    if not mtp_passing_pcc:
        logger.warning(f"MTP output PCC check failed: {mtp_passing_pcc}")
    assert mtp_passing_pcc, "MTP output PCC check failed"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("seed", [1337])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
            "worker_l1_size": 1480000,
        }
    ],
    indirect=True,
)
def test_single_device_mtp_verification(
    bh_2d_mesh_device,
    use_fp32,
    seed,
):
    """Single-device MTP verification test.

    Runs the base LM head + MTP to produce T_base, then runs a second LM head
    with enable_mtp_verification=True to produce T_spec and verify T_spec == T_base.
    Uses same input/weights for both stages so they should produce the same token -> match=1.
    """
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    device_grid_size = submesh.compute_with_storage_grid_size()

    M = 1
    K = 7168
    embedding_dim = 7168
    mtp_output_dim = 7168
    num_matmul_cores = 101
    n_per_core = 160
    num_dram_banks = 8
    n_total = num_matmul_cores * n_per_core
    tile_width = 32
    mtp_n_per_core = mtp_output_dim // num_dram_banks
    mtp_padded_dim = num_dram_banks * mtp_n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)

    # MTP tensors (for base LM head stage)
    torch_embedding = torch.randn((n_total, embedding_dim), dtype=torch.bfloat16)
    torch_h_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_e_gamma = torch.randn((M, embedding_dim), dtype=torch.bfloat16)
    torch_eh_proj = torch.randn((K + embedding_dim, mtp_output_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded = torch.zeros((K + embedding_dim, mtp_padded_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded[:, :mtp_output_dim] = torch_eh_proj

    # Golden: compute expected base token
    torch_expected_idx, torch_expected_mtp = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float(),
        indices=torch_indices,
        k=1,
        p=1.0,
        fuse_mtp=True,
        embedding_tensor=torch_embedding.float(),
        h_gamma_tensor=torch_h_gamma.float(),
        e_gamma_tensor=torch_e_gamma.float(),
        eh_projection_tensor=torch_eh_proj.float(),
    )

    # Golden: verification should match since we use same input/weights
    torch_verify_idx, torch_verify_result = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float(),
        indices=torch_indices,
        k=1,
        p=1.0,
        fuse_mtp_verification=True,
        reference_token=torch_expected_idx,
    )
    assert torch_verify_result.item() == 1, "Golden verification should match when same inputs are used"
    assert torch.equal(torch_verify_idx, torch_expected_idx), "Golden spec token should match base token"

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )

    # --- Stage 1: Base LM Head + MTP ---
    compute_cores = submesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )
    eh_shard_grid = ttnn.CoreCoord(submesh.dram_grid_size().x - 1, submesh.dram_grid_size().y - 1)
    eh_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), eh_shard_grid)})
    mtp_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(compute_core_grid, (M, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    eh_proj_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(eh_shard_grid, (K + embedding_dim, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )

    def to_device(t, mem, **kw):
        return ttnn.from_torch(t, device=submesh, memory_config=mem, **kw)

    input_tensor = to_device(torch_a, input_a_mem_config, layout=ttnn.TILE_LAYOUT, tile=a_tile, dtype=ttnn.bfloat16)
    intermediate_tensor = to_device(
        torch.zeros_like(torch_a), input_a_mem_config, layout=ttnn.TILE_LAYOUT, tile=a_tile, dtype=ttnn.bfloat16
    )
    ttnn_gamma = to_device(torch_gamma, input_a_mem_config, layout=ttnn.TILE_LAYOUT, tile=a_tile, dtype=ttnn.bfloat16)
    ttnn_b = to_device(torch_b, width_shard_mem_config, layout=ttnn.TILE_LAYOUT, tile=b_tile, dtype=ttnn.bfloat8_b)
    ttnn_scores = to_device(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        output_mem_config,
        layout=ttnn.TILE_LAYOUT,
        tile=out_tile,
        dtype=ttnn.bfloat16,
    )
    ttnn_indices = to_device(torch_indices, indices_mem_config, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32)
    ttnn_output_index = to_device(
        torch.zeros((1, 1), dtype=torch.uint32),
        output_index_mem_config,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )

    ttnn_embedding = to_device(
        torch_embedding, ttnn.DRAM_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    ttnn_h_gamma = to_device(
        torch_h_gamma, input_a_mem_config, layout=ttnn.TILE_LAYOUT, tile=a_tile, dtype=ttnn.bfloat16
    )
    ttnn_e_gamma = to_device(
        torch_e_gamma, input_a_mem_config, layout=ttnn.TILE_LAYOUT, tile=a_tile, dtype=ttnn.bfloat16
    )
    torch_eh_proj_shuffled = shuffle_tensor_tiles(torch_eh_proj_padded, tile_width, num_dram_banks)
    ttnn_eh_proj = to_device(
        torch_eh_proj_shuffled, eh_proj_mem_config, layout=ttnn.TILE_LAYOUT, tile=b_tile, dtype=ttnn.bfloat8_b
    )
    ttnn_mtp_output = to_device(
        torch.zeros((M, mtp_padded_dim), dtype=torch.bfloat16),
        mtp_output_mem_config,
        layout=ttnn.TILE_LAYOUT,
        tile=out_tile,
        dtype=ttnn.bfloat16,
    )
    mcast_dst_buf, mcast_eh_dst_buf = _create_mcast_working_bufs(
        submesh,
        mcast_core,
        matmul_core_grid,
        M,
        K,
        a_tile,
        embedding_dim=embedding_dim,
    )

    # Run base LM head + MTP
    LMHeadSampling.op(
        input_tensor,
        intermediate_tensor,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        output_mtp_tensor=ttnn_mtp_output,
        embedding_tensor=ttnn_embedding,
        h_gamma_tensor=ttnn_h_gamma,
        e_gamma_tensor=ttnn_e_gamma,
        eh_projection_tensor=ttnn_eh_proj,
        mcast_dst_working_buf_tensor=mcast_dst_buf,
        mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        semaphores=None,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
        enable_mtp=True,
    )
    ttnn.synchronize_device(submesh)

    base_token = ttnn.to_torch(ttnn_output_index).to(torch.uint32).reshape(1, 1)
    logger.info(f"Base token: {base_token.item()}, expected: {torch_expected_idx.item()}")
    assert torch.equal(
        base_token, torch_expected_idx
    ), f"Base token mismatch: {base_token.item()} != {torch_expected_idx.item()}"

    # --- Stage 2: MTP Verification LM Head ---
    # Pre-load the reference token (T_base from stage 1)
    reference_token_tensor = to_device(
        base_token.reshape(1, 1),
        output_index_mem_config,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    verification_result_tensor = to_device(
        torch.zeros((1, 1), dtype=torch.uint32),
        output_index_mem_config,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )
    speculative_tokens_tensor = to_device(
        torch.zeros((1, 1), dtype=torch.uint32),
        output_index_mem_config,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )

    # Reset scores and output for the verification op (reuse same weights/input)
    ttnn_scores_v = to_device(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        output_mem_config,
        layout=ttnn.TILE_LAYOUT,
        tile=out_tile,
        dtype=ttnn.bfloat16,
    )
    ttnn_output_index_v = to_device(
        torch.zeros((1, 1), dtype=torch.uint32),
        output_index_mem_config,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
    )

    LMHeadSampling.op(
        input_tensor,
        intermediate_tensor,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores_v,
        mcast_dst_working_buf_tensor=mcast_dst_buf,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index_v,
        argmax_final_core_coord=final_core,
        semaphores=None,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
        enable_mtp=False,
        enable_mtp_verification=True,
        reference_token_tensor=reference_token_tensor,
        verification_result_tensor=verification_result_tensor,
        speculative_tokens_tensor=speculative_tokens_tensor,
    )
    ttnn.synchronize_device(submesh)

    spec_token = ttnn.to_torch(ttnn_output_index_v).to(torch.uint32).reshape(1, 1)
    verify_result = ttnn.to_torch(verification_result_tensor).to(torch.uint32).reshape(1, 1)
    stored_spec = ttnn.to_torch(speculative_tokens_tensor).to(torch.uint32).reshape(1, 1)

    logger.info(f"Spec token: {spec_token.item()}, base token: {base_token.item()}")
    logger.info(f"Verification result: {verify_result.item()} (1=match, 0=no_match)")
    logger.info(f"Stored speculative token: {stored_spec.item()}")

    assert torch.equal(
        spec_token, base_token
    ), f"Spec token should match base token (same inputs). spec={spec_token.item()}, base={base_token.item()}"
    assert verify_result.item() == 1, f"Verification should match (same inputs). Got {verify_result.item()}"
    assert (
        stored_spec.item() == spec_token.item()
    ), f"Stored spec token should equal the spec token. stored={stored_spec.item()}, spec={spec_token.item()}"
    logger.info("MTP verification test PASSED: speculative token matches base token")


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("seed", [1337])
def test_single_device_d2h(
    bh_2d_mesh_device,
    use_fp32,
    seed,
    device_params,
):
    """Single-device fused LM-head + argmax with optional D2H token emission enabled."""
    if not is_slow_dispatch():
        pytest.skip("Skipping D2H socket test in fast dispatch mode")
    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((1, 1)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    d2h_page_size_bytes = 64

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices = torch.arange(n_total, dtype=torch.int32).reshape(1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(), torch_gamma.float(), torch_b.float(), indices=torch_indices, k=1, p=1.0
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)]), (1, 1), ttnn.ShardOrientation.ROW_MAJOR
        ),
    )

    input_tensor_mesh = ttnn.from_torch(
        torch_a,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
    )

    d2h_socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), final_core)
    d2h_socket = ttnn.D2HSocket(submesh, d2h_socket_core, d2h_page_size_bytes * 4)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=ttnn.MeshCoordinate(0, 0),
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=True,
        socket_output=d2h_socket,
    )

    output_index_torch = ttnn.to_torch(ttnn_output_index).to(torch.uint32).reshape(1, 1)
    assert torch.equal(
        output_index_torch, torch_expected_idx
    ), f"Fused argmax index mismatch. expected={torch_expected_idx.item()}, got={output_index_torch.item()}"

    d2h_page_words = d2h_page_size_bytes // 4
    d2h_read_tensor = ttnn.from_torch(
        torch.zeros((1, d2h_page_words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    d2h_socket.read_tensor(d2h_read_tensor)
    d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
    d2h_socket.barrier()
    ttnn.synchronize_device(submesh)
    logger.info(f"D2H token: {d2h_token}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        d2h_token, torch_expected_idx
    ), f"D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "sender_coord, final_mesh_coord, seed",
    [
        ((1, 1), (0, 0), 7),
        ((0, 0), (1, 1), 1337),
        ((3, 0), (2, 0), 4242),
    ],
)
def test_multidevice(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    sender_coord,
    seed,
    device_params,
):
    """4x2 mesh fused LM-head + k=1 sampling (argmax) with CCL enabled."""
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    # Global indices are unique across mesh devices.
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_mesh_coord = ttnn.MeshCoordinate(sender_coord[0], sender_coord[1])
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_mesh_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=mcast_core,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
        input_tensor_torch=torch_a,
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_mesh_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        bcast_semaphores=bcast_inputs.semaphores,
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        fabric_config=device_params["fabric_config"],
    )
    ttnn.synchronize_device(submesh)

    output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_index = ttnn.to_torch(output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    logger.info(f"Final output index: {final_output_index}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        final_output_index, torch_expected_idx
    ), f"Fused mesh argmax index mismatch. expected={torch_expected_idx.item()}, got={final_output_index.item()}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1)])
@pytest.mark.parametrize("seed", [7, 1337])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 1600000,
        }
    ],
    indirect=True,
)
def test_multidevice_mtp(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
):
    """4x2 mesh fused LM-head + argmax + MTP fusion with CCL enabled."""
    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core

    embedding_dim = 7168
    mtp_output_dim = 7168
    tile_width = 32
    num_dram_banks = 8
    mtp_n_per_core = mtp_output_dim // num_dram_banks
    mtp_padded_dim = num_dram_banks * mtp_n_per_core

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)

    torch_embedding = torch.randn((num_devices * n_total, embedding_dim), dtype=torch.bfloat16)
    torch_h_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_e_gamma = torch.randn((M, embedding_dim), dtype=torch.bfloat16)
    torch_eh_proj = torch.randn((K + embedding_dim, mtp_output_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded = torch.zeros((K + embedding_dim, mtp_padded_dim), dtype=torch.bfloat16)
    torch_eh_proj_padded[:, :mtp_output_dim] = torch_eh_proj

    torch_expected_idx, torch_expected_mtp = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
        fuse_mtp=True,
        embedding_tensor=torch_embedding.float(),
        h_gamma_tensor=torch_h_gamma.float(),
        e_gamma_tensor=torch_e_gamma.float(),
        eh_projection_tensor=torch_eh_proj.float(),
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes + 256 + 8) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    compute_cores = submesh.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores]
    )
    eh_shard_grid = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(submesh.dram_grid_size().x - 1, submesh.dram_grid_size().y - 1),
            )
        }
    )
    mtp_output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(compute_core_grid, (M, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    eh_proj_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(eh_shard_grid, (K + embedding_dim, mtp_n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    device_inputs = []
    device_intermediate = []
    for r in range(mesh_rows):
        for c in range(mesh_cols):
            if r == sender_coord[0] and c == sender_coord[1]:
                device_inputs.append(torch_a)
            else:
                device_inputs.append(torch.zeros_like(torch_a))
            device_intermediate.append(torch.zeros_like(torch_a))
    mesh_input = torch.cat(device_inputs, dim=0)
    mesh_intermediate = torch.cat(device_intermediate, dim=0)

    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    input_tensor_mesh = ttnn.from_torch(
        mesh_input,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    ttnn_embedding = ttnn.from_torch(
        torch_embedding,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_h_gamma = ttnn.from_torch(
        torch_h_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_e_gamma = ttnn.from_torch(
        torch_e_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    torch_eh_proj_shuffled = shuffle_tensor_tiles(torch_eh_proj_padded, tile_width, num_dram_banks)
    ttnn_eh_proj = ttnn.from_torch(
        torch_eh_proj_shuffled,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=eh_proj_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_mtp_output = ttnn.from_torch(
        torch.zeros((num_devices, M, mtp_padded_dim), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=mtp_output_mem_config,
        tile=out_tile,
        mesh_mapper=mesh_mapper,
    )

    mcast_dst_buf, mcast_eh_dst_buf = _create_mcast_working_bufs(
        submesh,
        mcast_core,
        matmul_core_grid,
        M,
        K,
        a_tile,
        embedding_dim=embedding_dim,
        num_devices=num_devices,
        mesh_mapper=mesh_mapper,
    )

    out_ready_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    barrier_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    secondary_sync_semaphore = ttnn.create_global_semaphore(submesh, worker_crs, 0)
    global_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        output_mtp_tensor=ttnn_mtp_output,
        embedding_tensor=ttnn_embedding,
        h_gamma_tensor=ttnn_h_gamma,
        e_gamma_tensor=ttnn_e_gamma,
        eh_projection_tensor=ttnn_eh_proj,
        mcast_dst_working_buf_tensor=mcast_dst_buf,
        mcast_eh_dst_working_buf_tensor=mcast_eh_dst_buf,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        semaphores=[out_ready_semaphore, barrier_semaphore, secondary_sync_semaphore],
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        enable_mtp=True,
    )
    ttnn.synchronize_device(submesh)

    final_output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_torch = ttnn.to_torch(final_output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    assert torch.equal(
        final_output_torch, torch_expected_idx
    ), f"Fused mesh argmax index mismatch. expected={torch_expected_idx.item()}, got={int(final_output_torch.item())}"

    final_mtp_shards = ttnn.get_device_tensors(ttnn_mtp_output)
    final_mtp_torch = (
        ttnn.to_torch(final_mtp_shards[final_device_idx])
        .to(torch.float32)
        .reshape(1, mtp_padded_dim)[:, :mtp_output_dim]
    )
    mtp_passing_pcc, _ = comp_pcc(final_mtp_torch, torch_expected_mtp.float(), 0.99)
    if not mtp_passing_pcc:
        max_diff = (final_mtp_torch - torch_expected_mtp.float()).abs().max()
        logger.warning(f"MTP output PCC check failed. Max diff: {max_diff}")
    assert mtp_passing_pcc, "MTP output PCC check failed"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1)])
@pytest.mark.parametrize("seed", [5449])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_X,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
def test_d2h(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
    device_params,
):
    """4x2 mesh fused LM-head + argmax with optional D2H token emission on final mesh device."""
    if not is_slow_dispatch():
        pytest.skip("Skipping D2H socket test in fast dispatch mode")

    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    d2h_page_size_bytes = 64

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    mcast_core_x = device_grid_size.x - 1
    mcast_core_y = 9
    mcast_core = ttnn.CoreCoord(mcast_core_x, mcast_core_y)
    mcast_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_core, mcast_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    final_core = ttnn.CoreCoord(0, 0)
    final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(final_core, final_core)])

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(mcast_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=mcast_core,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
        input_tensor_torch=torch_a,
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, final_core_grid, 0)

    d2h_socket_core = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(final_mesh_coord[0], final_mesh_coord[1]), final_core)
    d2h_socket = ttnn.D2HSocket(submesh, d2h_socket_core, d2h_page_size_bytes * 4)

    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        bcast_semaphores=bcast_inputs.semaphores,
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        socket_output=d2h_socket,
        fabric_config=device_params["fabric_config"],
    )
    ttnn.synchronize_device(submesh)

    output_shards = ttnn.get_device_tensors(ttnn_output_index)
    final_device_idx = int(final_mesh_coord[0]) * mesh_cols + int(final_mesh_coord[1])
    final_output_index = ttnn.to_torch(output_shards[final_device_idx]).to(torch.uint32).reshape(1, 1)
    assert torch.equal(
        final_output_index, torch_expected_idx
    ), f"Fused mesh argmax index mismatch. expected={torch_expected_idx.item()}, got={final_output_index.item()}"

    d2h_page_words = d2h_page_size_bytes // 4
    d2h_read_tensor = ttnn.from_torch(
        torch.zeros((1, d2h_page_words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    d2h_socket.read_tensor(d2h_read_tensor)
    d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
    logger.info(f"D2H token: {d2h_token}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        d2h_token, torch_expected_idx
    ), f"Mesh D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1), (0, 1), (1, 0), (0, 0), (3, 1)])
@pytest.mark.parametrize("seed", [5449])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
def test_d2d_to_d2h_pipeline(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
    device_params,
):
    """4x2 mesh fused LM-head + argmax with D2D output routed through D2D forwarding to D2H."""
    if ttnn.get_num_devices() < 32:
        pytest.skip("Test requires a full galaxy")
    if not is_slow_dispatch():
        pytest.skip("Skipping D2D/D2H pipeline test in fast dispatch mode")

    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    ttnn.enable_asynchronous_slow_dispatch(submesh)

    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    socket_page_size_bytes = 64
    socket_fifo_size = 256

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    lmhead_input_core_x = 10
    lmhead_input_core_y = 9
    lmhead_input_core = ttnn.CoreCoord(lmhead_input_core_x, lmhead_input_core_y)
    lmhead_input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(lmhead_input_core, lmhead_input_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    argmax_final_core = ttnn.CoreCoord(0, 0)
    argmax_final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)])

    mcast_bbox = matmul_core_grid.bounding_box()
    reserved_cores = {(argmax_final_core.x, argmax_final_core.y), (lmhead_input_core.x, lmhead_input_core.y)}
    extra_cores = []
    for y in range(device_grid_size.y):
        for x in range(device_grid_size.x):
            if (x, y) in reserved_cores:
                continue
            if mcast_bbox.contains(ttnn.CoreCoord(x, y)):
                continue
            extra_cores.append(ttnn.CoreCoord(x, y))
    logger.info(f"Extra cores: {extra_cores}")
    if len(extra_cores) < 4:
        pytest.skip("Test requires at least 4 spare cores for D2D/D2H pipeline wiring")
    d2d1_core = ttnn.CoreCoord(11, 0)
    d2d2_core = ttnn.CoreCoord(11, 1)
    d2h_core = ttnn.CoreCoord(11, 2)
    dummy_h2d_core = ttnn.CoreCoord(11, 3)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(lmhead_input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=lmhead_input_core,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
        input_tensor_torch=torch_a,
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)

    final_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        argmax_final_core,
    )

    d2d1_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2d1_core,
    )
    d2d2_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2d2_core,
    )
    d2h_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2h_core,
    )
    dummy_h2d_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        dummy_h2d_core,
    )

    logger.info(f"final_mesh_core: {final_mesh_core}")
    logger.info(f"d2d1_mesh_core: {d2d1_mesh_core}")
    logger.info(f"d2d2_mesh_core: {d2d2_mesh_core}")
    logger.info(f"d2h_mesh_core: {d2h_mesh_core}")
    logger.info(f"dummy_h2d_mesh_core: {dummy_h2d_mesh_core}")

    h2d_socket = ttnn.H2DSocket(
        submesh, dummy_h2d_mesh_core, ttnn.BufferType.L1, socket_fifo_size, ttnn.H2DMode.HOST_PUSH
    )
    d2h_socket = ttnn.D2HSocket(submesh, d2h_mesh_core, socket_fifo_size)
    logger.info("Creating HostInterface")
    host_io = HostInterface(
        h2d_socket,
        d2h_socket,
        socket_page_size_bytes,
        socket_page_size_bytes,
        core_to_core_socket_buffer_size=socket_fifo_size,
        h2d_downstream_core=dummy_h2d_mesh_core,
        d2h_upstream_core=d2d2_mesh_core,
    )
    logger.info("Creating SocketInterface")
    socket_interface = SocketInterface(
        socket_page_size_bytes,
        socket_fifo_size,
        socket_page_size_bytes,
        d2d1_mesh_core,
        d2d2_mesh_core,
        upstream_core_coord=final_mesh_core,
        downstream_socket=host_io.get_upstream_socket(),
        sender_mesh=MeshWrapper(mesh_device=submesh),
        receiver_mesh=MeshWrapper(mesh_device=submesh),
    )

    logger.info("Running HostInterface")
    host_io.run()
    logger.info("Running SocketInterface")
    socket_interface.run()
    logger.info("Running LMHeadSampling")
    LMHeadSampling.op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        ttnn_gamma,
        ttnn_b,
        ttnn_scores,
        sender_coord=sender_coord,
        indices_tensor=ttnn_indices,
        output_index_tensor=ttnn_output_index,
        argmax_final_core_coord=argmax_final_core,
        argmax_final_mesh_coord=final_mesh_coord,
        bcast_semaphores=bcast_inputs.semaphores,
        global_semaphore=global_semaphore,
        global_stage2_semaphore=global_stage2_semaphore,
        fabric_scratch_tensor=ttnn_fabric_scratch,
        fp32_dest_acc_en=use_fp32,
        skip_ccl=False,
        socket_output=socket_interface.get_upstream_socket(),
        fabric_config=device_params["fabric_config"],
    )
    d2h_page_words = socket_page_size_bytes // 4
    d2h_read_tensor = ttnn.from_torch(
        torch.zeros((1, d2h_page_words), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    d2h_socket.read_tensor(d2h_read_tensor)
    d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
    logger.info(f"D2D->D2H token: {d2h_token}")
    logger.info(f"Expected index: {torch_expected_idx}")
    assert torch.equal(
        d2h_token, torch_expected_idx
    ), f"Mesh D2D->D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"

    host_io.terminate(False)
    socket_interface.terminate(True)

    ttnn.synchronize_device(submesh)


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize("final_mesh_coord", [(1, 1), (0, 1), (2, 0), (2, 1)])
@pytest.mark.parametrize("seed", [5449])
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
def test_4stage_galaxy_1_iteration(
    bh_2d_mesh_device,
    use_fp32,
    final_mesh_coord,
    seed,
    device_params,
):
    """4x2 mesh lm_head pipeline with H2D ingress + D2D ingress before compute, then D2D->D2H egress."""
    if ttnn.get_num_devices() < 32:
        pytest.skip("Test requires a full galaxy")
    if not is_slow_dispatch():
        pytest.skip("Skipping D2D/D2H pipeline test in fast dispatch mode")

    mesh_rows, mesh_cols = 4, 2
    num_devices = mesh_rows * mesh_cols
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))
    ttnn.enable_asynchronous_slow_dispatch(submesh)

    device_grid_size = submesh.compute_with_storage_grid_size()
    worker_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))}
    )

    M = 1
    K = 7168
    num_matmul_cores = 101
    n_per_core = 160
    n_total = num_matmul_cores * n_per_core
    activation_page_size_bytes = K * 2  # bf16 [1, 7168]
    activation_fifo_size = activation_page_size_bytes * 2
    socket_page_size_bytes = 64
    socket_fifo_size = 512
    assert activation_page_size_bytes == 14336
    assert socket_fifo_size == 8 * socket_page_size_bytes

    a_tile = ttnn.Tile([1, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([1, 32])

    lmhead_input_core_x = 10
    lmhead_input_core_y = 9
    lmhead_input_core = ttnn.CoreCoord(lmhead_input_core_x, lmhead_input_core_y)
    lmhead_input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(lmhead_input_core, lmhead_input_core)])
    matmul_core_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(9, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(10, 0), ttnn.CoreCoord(10, 0)),
        ]
    )
    argmax_final_core = ttnn.CoreCoord(0, 0)
    argmax_final_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)])

    reserved_cores = {(argmax_final_core.x, argmax_final_core.y), (lmhead_input_core.x, lmhead_input_core.y)}
    extra_cores = []
    for y in range(device_grid_size.y):
        for x in range(device_grid_size.x):
            if (x, y) in reserved_cores:
                continue
            if matmul_core_grid.bounding_box().contains(ttnn.CoreCoord(x, y)):
                continue
            extra_cores.append(ttnn.CoreCoord(x, y))
    if len(extra_cores) < 4:
        pytest.skip("Test requires at least 4 spare cores for H2D/D2D and D2D/D2H pipeline wiring")

    ingress_forward_core = ttnn.CoreCoord(11, 0)
    egress_sink_core = ttnn.CoreCoord(11, 1)
    d2h_endpoint_core = ttnn.CoreCoord(11, 2)
    h2d_endpoint_core = ttnn.CoreCoord(11, 3)
    ingress_relay_core = ttnn.CoreCoord(11, 4)

    torch.manual_seed(seed)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    torch_gamma = torch.randn((M, K), dtype=torch.bfloat16)
    torch_b = torch.randn((K, n_total), dtype=torch.bfloat16)
    torch_indices_all = torch.arange(num_devices * n_total, dtype=torch.int32).reshape(num_devices, 1, n_total)
    torch_expected_idx, _ = LMHeadSampling.golden(
        torch_a.float(),
        torch_gamma.float(),
        torch_b.float().unsqueeze(0).repeat(num_devices, 1, 1),
        indices=torch_indices_all,
        k=1,
        p=1.0,
    )

    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(lmhead_input_core_grid, (M, K), ttnn.ShardOrientation.ROW_MAJOR),
    )
    width_shard_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (K, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    indices_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(matmul_core_grid, (M, n_per_core), ttnn.ShardOrientation.ROW_MAJOR),
    )
    output_index_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, (1, 1), ttnn.ShardOrientation.ROW_MAJOR),
    )
    winner_page_bytes = 16
    scratch_shape_per_device = (1, ((mesh_rows + mesh_cols) * winner_page_bytes) // 4)
    scratch_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(argmax_final_core_grid, scratch_shape_per_device, ttnn.ShardOrientation.ROW_MAJOR),
    )

    sender_coord = ttnn.MeshCoordinate(1, 0)
    bcast_inputs = build_broadcast_test_inputs(
        mesh_device=submesh,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        sender_coord=sender_coord,
        output_shape=torch_a.shape,
        input_shard_shape=(M, K),
        tensor_mem_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        layout=ttnn.TILE_LAYOUT,
        input_dtype=ttnn.bfloat16,
        bcast_core=lmhead_input_core,
        input_tensor_torch=torch_a,
        create_output_tensor_mesh=True,
        create_semaphores=True,
        num_links=1,
        tile=a_tile,
        output_mesh_mapper="shard_dim0",
    )
    input_tensor_mesh = bcast_inputs.input_tensor_mesh
    intermediate_tensor_mesh = bcast_inputs.output_tensor_mesh
    mesh_mapper = ttnn.ShardTensorToMesh(submesh, dim=0)
<<<<<<< HEAD
=======
    input_tensor_mesh = ttnn.from_torch(
        mesh_input,
        device=submesh,
        layout=ttnn.TILE_LAYOUT,
        tile=a_tile,
        dtype=ttnn.bfloat16,
        memory_config=input_a_mem_config,
        mesh_mapper=mesh_mapper,
    )
>>>>>>> 6ffd11b80d (mtp 2 wip)
    ttnn_gamma = ttnn.from_torch(
        torch_gamma,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=input_a_mem_config,
        tile=a_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=width_shard_mem_config,
        tile=b_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_scores = ttnn.from_torch(
        torch.zeros((M, n_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=submesh,
        memory_config=output_mem_config,
        tile=out_tile,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )
    ttnn_indices = ttnn.from_torch(
        torch_indices_all,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=indices_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_output_index = ttnn.from_torch(
        torch.zeros((num_devices, 1, 1), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=output_index_mem_config,
        mesh_mapper=mesh_mapper,
    )
    ttnn_fabric_scratch = ttnn.from_torch(
        torch.zeros((num_devices, *scratch_shape_per_device), dtype=torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=submesh,
        memory_config=scratch_mem_config,
        mesh_mapper=mesh_mapper,
    )

    global_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)
    global_stage2_semaphore = ttnn.create_global_semaphore(submesh, argmax_final_core_grid, 0)
    sender_mesh_coord = ttnn.MeshCoordinate(int(sender_coord[0]), int(sender_coord[1]))

    lmhead_input_mesh_core = ttnn.MeshCoreCoord(sender_mesh_coord, lmhead_input_core)
    ingress_relay_mesh_core = ttnn.MeshCoreCoord(sender_mesh_coord, ingress_relay_core)
    ingress_forward_mesh_core = ttnn.MeshCoreCoord(sender_mesh_coord, ingress_forward_core)
    h2d_endpoint_mesh_core = ttnn.MeshCoreCoord(sender_mesh_coord, h2d_endpoint_core)

    argmax_final_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        argmax_final_core,
    )

    egress_forward_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        ingress_forward_core,
    )
    egress_sink_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        egress_sink_core,
    )
    d2h_endpoint_mesh_core = ttnn.MeshCoreCoord(
        ttnn.MeshCoordinate(int(final_mesh_coord[0]), int(final_mesh_coord[1])),
        d2h_endpoint_core,
    )
    h2d_host_socket = ttnn.H2DSocket(
        submesh,
        h2d_endpoint_mesh_core,
        ttnn.BufferType.L1,
        activation_fifo_size,
        ttnn.H2DMode.HOST_PUSH,
    )
    d2h_host_socket = ttnn.D2HSocket(submesh, d2h_endpoint_mesh_core, socket_fifo_size)
    host_io_bridge = HostInterface(
        h2d_host_socket,
        d2h_host_socket,
        activation_page_size_bytes,
        socket_page_size_bytes,
        core_to_core_socket_buffer_size=activation_fifo_size,
        h2d_downstream_core=ingress_relay_mesh_core,
        d2h_upstream_core=egress_sink_mesh_core,
    )
    ingress_d2d_link = SocketInterface(
        activation_page_size_bytes,
        activation_fifo_size,
        activation_page_size_bytes,
        ingress_relay_mesh_core,
        ingress_forward_mesh_core,
        upstream_socket=host_io_bridge.get_downstream_socket(),
        downstream_core_coord=lmhead_input_mesh_core,  # LMHead sender/socket-receiver core
        sender_mesh=MeshWrapper(submesh),
        receiver_mesh=MeshWrapper(submesh),
    )
    egress_d2d_link = SocketInterface(
        socket_page_size_bytes,
        socket_fifo_size,
        socket_page_size_bytes,
        egress_forward_mesh_core,
        egress_sink_mesh_core,
        upstream_core_coord=argmax_final_mesh_core,  # sampling winner core / socket sender core
        downstream_socket=host_io_bridge.get_upstream_socket(),
        sender_mesh=MeshWrapper(submesh),
        receiver_mesh=MeshWrapper(submesh),
    )

    logger.info("Running HostInterface")
    host_io_bridge.run()
    logger.info("Running Input SocketInterface")
    ingress_d2d_link.run()
    logger.info("Running Output SocketInterface")
    egress_d2d_link.run()

    try:
        h2d_activation_tensor = ttnn.from_torch(
            torch_a.contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        logger.info("Running H2D socket write")
        h2d_host_socket.write_tensor(h2d_activation_tensor)

        logger.info("Running LMHeadSampling")
        LMHeadSampling.op(
            input_tensor_mesh,
            intermediate_tensor_mesh,
            ttnn_gamma,
            ttnn_b,
            ttnn_scores,
            sender_coord=sender_coord,
            indices_tensor=ttnn_indices,
            output_index_tensor=ttnn_output_index,
            argmax_final_core_coord=argmax_final_core,
            argmax_final_mesh_coord=final_mesh_coord,
            bcast_semaphores=bcast_inputs.semaphores,
            global_semaphore=global_semaphore,
            global_stage2_semaphore=global_stage2_semaphore,
            fabric_scratch_tensor=ttnn_fabric_scratch,
            fp32_dest_acc_en=use_fp32,
            skip_ccl=False,
            socket_input=ingress_d2d_link.get_downstream_socket(),
            socket_output=egress_d2d_link.get_upstream_socket(),
            fabric_config=device_params["fabric_config"],
        )
        logger.info("Running D2H socket read")
        d2h_page_words = socket_page_size_bytes // 4
        d2h_read_tensor = ttnn.from_torch(
            torch.zeros((1, d2h_page_words), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        d2h_host_socket.read_tensor(d2h_read_tensor)
        d2h_token = ttnn.to_torch(d2h_read_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
        assert torch.equal(
            d2h_token, torch_expected_idx
        ), f"Mesh H2D->D2D->LMHead->D2D->D2H token mismatch. expected={torch_expected_idx.item()}, got={int(d2h_token.item())}"
    finally:
        host_io_bridge.terminate(False)
        ingress_d2d_link.terminate(False)
        egress_d2d_link.terminate(True)
        ttnn.synchronize_device(submesh)


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
def test_pipline_block_4stage_galaxy_1_iteration(mesh_device, use_fp32, device_params):
    """
    4-stage 4x2 single-galaxy pipeline:
    P1(H2D) -> P2(LMHead+Sampling) -> P3(forward) -> P4(forward) -> P1(D2H).
    One-shot LMHead (no persistent mode); single token; terminate in finally.
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 4:
        pytest.skip("This test requires exactly 4 distributed pipeline processes (P1..P4)")

    torch_expected_indices = _compute_expected_lm_head_indices_synthetic(1)
    torch_expected_idx = torch_expected_indices[0]

    config = create_single_galaxy_pipeline_configuration(
        _SyntheticWeightProvider(),
        fp32_dest_acc_en=use_fp32,
        persistent_mode=False,
    )
    pipeline = config.build_pipeline(mesh_device)
    try:
        pipeline.setup_and_run()

        if pipeline.my_mesh_id == 0:
            torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
            torch_token[0, 0] = 0
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.from_torch(
                torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            pipeline.write_token(token_tensor)
            pipeline.read_output(output_tensor)
            got = ttnn.to_torch(output_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
            assert torch.equal(
                got, torch_expected_idx
            ), f"PipelineBlock 4-stage token mismatch. expected={int(torch_expected_idx.item())}, got={int(got.item())}"

        pipeline.barrier()
    finally:
        pipeline.terminate()


@pytest.mark.skipif(
    not _is_persistent_mode_enabled(), reason="Set TT_RUN_PERSISTENT_MODE=1 to run persistent mode test"
)
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_persistent_mode(mesh_device, use_fp32, device_params):
    """
    4-stage 4x2 single-galaxy pipeline:
    P1(H2D) -> P2(LMHead+Sampling) -> P3(forward) -> P4(forward) -> P1(D2H).
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 4:
        pytest.skip("This test requires exactly 4 distributed pipeline processes (P1..P4)")

    iterations = 100
    torch_expected_indices = _compute_expected_lm_head_indices_synthetic(iterations)
    config = create_single_galaxy_pipeline_configuration(
        _SyntheticWeightProvider(),
        fp32_dest_acc_en=use_fp32,
    )
    pipeline = config.build_pipeline(mesh_device)
    pipeline.setup_and_run()

    if pipeline.my_mesh_id == 0:
        for iteration in range(iterations):
            logger.info(f"Writing token for iteration {iteration}")
            torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
            torch_token[0, 0] = iteration
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.from_torch(
                torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            pipeline.write_token(token_tensor)
            pipeline.read_output(output_tensor)
            got = ttnn.to_torch(output_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
            expected_idx = torch_expected_indices[iteration]
            logger.info(f"Iteration {iteration} output token: {got}, expected: {expected_idx}")
            assert torch.equal(
                got, expected_idx
            ), f"PipelineBlock 4-stage token mismatch. expected={int(expected_idx.item())}, got={int(got.item())}"

    logger.info(f"Barrier for P{pipeline.my_mesh_id}")
    pipeline.barrier()
    logger.info(f"Barrier completed for P{pipeline.my_mesh_id}")


@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 1600000,
            "worker_l1_size": 1499000,
        }
    ],
    indirect=True,
)
def test_persistent_mode_mtp(mesh_device, use_fp32):
    """
    4-stage 4x2 single-galaxy pipeline with MTP + verification:
    P1(Embed) -> P2(LMHead+MTP) -> P3(Passthrough ACTIVATION_W_TOKEN_META) -> P4(Verify) -> P1(D2H TOKEN_META).

    The verification stage (P4) receives gathered logits + token metadata, runs its
    own LM head + argmax, then outputs a TOKEN_META page (64 bytes) back to P1.

    TOKEN_META page layout (uint32 words):
      [0] num_tokens  (0=stale, 1=accept, 2=reject)
      [1] tok0_id     [2] tok0_type (0=BASE,1=SPEC)  [3] tok0_pos
      [4] tok1_id     [5] tok1_type                   [6] tok1_pos
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 4:
        pytest.skip("This test requires exactly 4 distributed pipeline processes (P1..P4)")

    iterations = 100

    # print(f"[TEST] computing golden", flush=True)
    # golden = _compute_expected_spec_decode_tokens_synthetic(iterations)
    # print(f"[TEST] golden computed, creating config", flush=True)

    config = create_single_galaxy_spec_decode_pipeline_configuration(
        _SyntheticWeightProvider(),
        fp32_dest_acc_en=use_fp32,
    )
    print(f"[TEST] config created, building pipeline", flush=True)
    pipeline = config.build_pipeline(mesh_device)
    pid = pipeline.my_mesh_id
    print(f"[TEST P{pid}] pipeline built, calling setup_and_run", flush=True)
    try:
        pipeline.setup_and_run()
        print(f"[TEST P{pid}] setup_and_run complete", flush=True)

        token_meta_words = TOKEN_META_PAGE_SIZE_BYTES // 4

        if pipeline.my_mesh_id == 0:
            for iteration in range(iterations):
                print(f"[TEST P{pid}] iter {iteration} write_token", flush=True)
                torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
                torch_token[0, 0] = iteration
                token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
                output_tensor = ttnn.from_torch(
                    torch.zeros(1, token_meta_words, dtype=torch.uint32),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
                pipeline.write_token(token_tensor)
                print(f"[TEST P{pid}] iter {iteration} read_output", flush=True)
                pipeline.read_output(output_tensor)
                print(f"[TEST P{pid}] iter {iteration} to_torch", flush=True)
                raw = ttnn.to_torch(output_tensor).to(torch.uint32).flatten()

                num_tokens = raw[0].item()
                tok0_id = raw[1].item()
                tok0_type = raw[2].item()
                tok0_pos = raw[3].item()
                tok1_id = raw[4].item()
                tok1_type = raw[5].item()
                tok1_pos = raw[6].item()

                # expected_base, expected_spec = golden[iteration]
                type_name = {0: "BASE", 1: "SPEC"}
                print(
                    f"[TEST P{pid}] iter {iteration} "
                    f"ntok={num_tokens} t0={tok0_id}/{type_name.get(tok0_type,'?')} "
                    f"t1={tok1_id}/{type_name.get(tok1_type,'?')} ",
                    flush=True,
                )
                # assert num_tokens == 2, f"Iteration {iteration}: expected num_tokens=2, got {num_tokens}"
                # assert (
                #     tok0_id == expected_base
                # ), f"Iteration {iteration}: base token mismatch: got {tok0_id}, expected {expected_base}"
                # assert tok0_type == 0, f"Iteration {iteration}: tok0_type should be BASE(0), got {tok0_type}"
                # assert (
                #     tok1_id == expected_spec
                # ), f"Iteration {iteration}: spec token mismatch: got {tok1_id}, expected {expected_spec}"
                # assert tok1_type == 1, f"Iteration {iteration}: tok1_type should be SPEC(1), got {tok1_type}"

        print(f"[TEST P{pid}] all iterations done, barrier", flush=True)
        pipeline.barrier()
        print(f"[TEST P{pid}] barrier done, terminate", flush=True)
        pipeline.terminate()
        print(f"[TEST P{pid}] terminate done, final barrier", flush=True)
        pipeline.barrier()
        print(f"[TEST P{pid}] final barrier done", flush=True)
    finally:
        pass


# @pytest.mark.skipif(not _is_persistent_mode_enabled(), reason="Set RUN_PERSISTENT_MODE=1 to run persistent mode test")
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_persistent_mode_real_weights(mesh_device, use_fp32, hf_model_path, hf_state_dict):
    """
    Same as test_persistent_mode but uses real HF weights (DEEPSEEK_V3_HF_MODEL) via StateDictWeightProvider.
    Each pipeline step writes a **random** input vocab id (fixed seed) for the embedding lookup; the device
    output must lie in the reference top-k from HuggingFace-style functional logits (RMSNorm then
    hidden @ lm_head.weight.T in bfloat16), since device numerics (e.g. bfloat8_b weights) can disagree
    with the reference argmax.
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 4:
        pytest.skip("This test requires exactly 4 distributed pipeline processes (P1..P4)")

    iterations = 300
    topk = 5
    rng = torch.Generator().manual_seed(_REAL_WEIGHTS_PERSISTENT_INPUT_TOKEN_SEED)
    input_token_ids = torch.randint(0, _VOCAB_SIZE, (iterations,), generator=rng, dtype=torch.int64)
    ref_topk = _compute_reference_topk_token_ids_real(hf_state_dict, input_token_ids, topk=topk)
    config = create_single_galaxy_pipeline_configuration(
        StateDictWeightProvider(hf_model_path),
        lm_head_fp32_dest_acc_en=use_fp32,
    )
    pipeline = config.build_pipeline(mesh_device)
    pipeline.setup_and_run()

    if pipeline.my_mesh_id == 0:
        got_tokens = []
        for iteration in range(iterations):
            in_tok = int(input_token_ids[iteration].item())
            logger.info(f"Writing token for iteration {iteration} (in_tok={in_tok})")
            torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
            torch_token[0, 0] = in_tok
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.from_torch(
                torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            pipeline.write_token(token_tensor)
            pipeline.read_output(output_tensor)
            got = ttnn.to_torch(output_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
            got_tokens.append(got)
        got_all = torch.stack(got_tokens, dim=0)
        got_flat = got_all.squeeze(-1).squeeze(-1)
        logger.info(f"Random input token ids (in_tok per iter): {input_token_ids.tolist()}")
        logger.info(f"All output tokens (real weights): {got_flat.tolist()}")
        logger.info(f"Reference top-{topk} per iteration (first row): {ref_topk[0].tolist()}")
        mismatches = []
        for i in range(iterations):
            in_tok = int(input_token_ids[i].item())
            g = int(got_flat[i].item())
            top_ids = [int(x) for x in ref_topk[i].tolist()]
            if g not in top_ids:
                mismatches.append((i, in_tok, g, top_ids))
        results_table = _format_real_weights_topk_results_table(
            hf_state_dict,
            input_token_ids,
            got_flat,
            ref_topk,
            topk=topk,
        )
        logger.info(results_table)
        if mismatches:
            report = _format_real_weights_topk_mismatch_report(
                hf_state_dict,
                mismatches,
                topk=topk,
                total_iters=iterations,
            )
            logger.error(report)
            pytest.fail(
                f"PipelineBlock (real weights): {len(mismatches)} output(s) not in HF functional top-{topk}.\n{report}"
            )

    logger.info(f"Barrier for P{pipeline.my_mesh_id} (real weights)")
    pipeline.barrier()
    logger.info(f"Barrier completed for P{pipeline.my_mesh_id} (real weights)")


@pytest.mark.skipif(
    not _is_persistent_mode_enabled(), reason="Set TT_RUN_PERSISTENT_MODE=1 to run persistent mode test"
)
@pytest.mark.parametrize("use_fp32", [True])
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 2)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_Y,
            "fabric_router_config": create_fabric_router_config(15232),
            "trace_region_size": 573440,
        }
    ],
    indirect=True,
)
def test_persistent_mode_pod(mesh_device, use_fp32, device_params):
    """
    16-stage 4x2 pod pipeline (4 galaxies):
    Stage1(H2D+Embed) -> Stage2..14(activation fwd) -> Stage15(LMHead+Sampling) -> Stage16(token fwd) -> Stage1(D2H).
    """
    if not is_slow_dispatch():
        pytest.skip("Skipping test in fast dispatch mode")

    ttnn.enable_asynchronous_slow_dispatch(mesh_device)
    num_procs = int(ttnn.distributed_context_get_size())
    if num_procs != 16:
        pytest.skip("This test requires exactly 16 distributed pipeline processes (pod: 4 galaxies)")

    iterations = 100
    torch_expected_indices = _compute_expected_lm_head_indices_synthetic(iterations)
    config = create_single_pod_passthrough_pipeline_configuration(
        _SyntheticWeightProvider(),
        lm_head_fp32_dest_acc_en=use_fp32,
    )
    pipeline = config.build_pipeline(mesh_device)
    pipeline.setup_and_run()

    if pipeline.my_mesh_id == 0:
        for iteration in range(iterations):
            torch_token = torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32)
            torch_token[0, 0] = iteration
            token_tensor = ttnn.from_torch(torch_token, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            output_tensor = ttnn.from_torch(
                torch.zeros(1, TOKEN_PAGE_SIZE_BYTES // 4, dtype=torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            logger.info(f"Writing token for iteration {iteration}")
            pipeline.write_token(token_tensor)
            logger.info(f"Reading output for iteration {iteration}")
            pipeline.read_output(output_tensor)
            got = ttnn.to_torch(output_tensor).to(torch.uint32)[0, 0].reshape(1, 1)
            expected_idx = torch_expected_indices[iteration]
            logger.info(f"Iteration {iteration} output token: {got}, expected: {expected_idx}")
            assert torch.equal(
                got, expected_idx
            ), f"Pod 16-stage token mismatch at iter {iteration}. expected={int(expected_idx.item())}, got={int(got.item())}"

    logger.info(f"Barrier for stage {pipeline.my_mesh_id + 1}")
    pipeline.barrier()
    logger.info(f"Barrier completed for stage {pipeline.my_mesh_id + 1}")
