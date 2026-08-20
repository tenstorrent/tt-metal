# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Random-weight Mistral model over the single-galaxy PP4 pipeline.

probe_pp4.py proved the TRANSPORT (four 8-chip meshes, D2D sockets, a payload reaching stage 3).
This proves the MODEL on top of it: each rank builds its own slice of the transformer, allocates its
own KV cache, compiles, and runs one real chunk, forwarding the hidden state down the pipeline.

Why random weights are synthesized here rather than reused from the existing helpers:

  * The prefill RUNNER cannot do random weights at all. It always passes ``state_dict={}``
    (adapters/mla.py), and TtPrefillTransformer hard-fails without either a non-empty state_dict or a
    populated .tensorbin cache. We have no (8,1) cache, and building one for a 119B checkpoint is not
    a quick detour -- hence going around the runner and constructing the runtime directly.
  * The usual random-weight path (``create_hf_model`` -> ``extract_tt_state_dict``) instantiates
    ``variant.reference_model_cls``, which Mistral deliberately leaves unwired: transformers'
    Mistral4Attention has an incompatible forward signature and its router has no
    ``e_score_correction_bias``. So the state dict is built straight from the config dimensions,
    which needs no reference model and costs one ``torch.randn`` per weight.

The model is scaled DOWN by default (4 layers, 32 experts) because the point is to exercise the
pipeline mechanics, not the full parameter count: a random state dict is materialized in HOST memory
as bf16, so the real 36x128 model would need ~230 GB across the four co-located ranks. 32 experts on
8 chips gives 4 experts/chip, the same per-chip expert count the shipped 128-expert/32-chip
configuration has. Override with PREFILL_NUM_LAYERS / PROBE_N_EXPERTS.

Run:  ./mistral4_bringup/pp4/run_probe_pp4.sh model
"""

import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.runners.runner_utils import compute_layer_split, open_mesh_device
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_119b_config import (
    Mistral4Small119BConfig,
    mistral4_hf_config,
)
from probe_pp4 import _build_endpoints, _recv, _send

SP = int(os.environ.get("PREFILL_SP", 8))
TP = int(os.environ.get("PREFILL_TP", 1))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5120))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 5120))
# PROBE_* wins over PREFILL_* so the shipped 36-layer rank binding stays the default while a debug
# loop can shrink the model (build time is dominated by the per-layer weight conversion).
NUM_LAYERS = int(os.environ.get("PROBE_NUM_LAYERS") or os.environ.get("PREFILL_NUM_LAYERS", 4))
N_EXPERTS = int(os.environ.get("PROBE_N_EXPERTS", 32))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 1))

# Rank 0's token chunk is sharded across the sp rows and replicated on tp, matching what the H2D
# service would have produced had we gone through the runner.
H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])


def _rand(*shape) -> torch.Tensor:
    # Scaled down from unit variance: stacking random projections without trained norms otherwise
    # grows the residual until the bf16/bfp8 activations saturate, which looks like a transport bug.
    return (torch.randn(*shape) * 0.02).to(torch.bfloat16)


def _random_layer_state(cfg) -> dict:
    """One layer in TtPrefillBlock's expected format (mirrors extract_layer_state_dict's keys).

    Shapes follow the HF (out_features, in_features) convention the TT modules transpose themselves.
    Every Mistral layer is MoE (first_k_dense_replace = 0), so the dense ``ffn_weights`` branch is
    never populated.
    """
    h = cfg.hidden_size
    n_heads = cfg.num_attention_heads
    qk_head = cfg.qk_nope_head_dim + cfg.qk_rope_head_dim
    moe_i = cfg.moe_intermediate_size
    return {
        "attn_norm_weight": _rand(h),
        "mla_weights": {
            "q_a_proj.weight": _rand(cfg.q_lora_rank, h),
            "q_a_layernorm.weight": _rand(cfg.q_lora_rank),
            "q_b_proj.weight": _rand(n_heads * qk_head, cfg.q_lora_rank),
            "kv_a_proj_with_mqa.weight": _rand(cfg.kv_lora_rank + cfg.qk_rope_head_dim, h),
            "kv_a_layernorm.weight": _rand(cfg.kv_lora_rank),
            "kv_b_proj.weight": _rand(n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim), cfg.kv_lora_rank),
            "o_proj.weight": _rand(h, n_heads * cfg.v_head_dim),
        },
        "ffn_norm_weight": _rand(h),
        "gate_weights": {
            "weight": _rand(cfg.n_routed_experts, h),
            # Mistral's router is a plain softmax with no bias term, but the DeepSeek-shaped TT gate
            # always reads one. Zeros make it a no-op rather than a fictitious bias.
            "e_score_correction_bias": torch.zeros(cfg.n_routed_experts, dtype=torch.bfloat16),
        },
        "routed_expert_weights": [
            {"gate_proj": _rand(moe_i, h), "up_proj": _rand(moe_i, h), "down_proj": _rand(h, moe_i)}
            for _ in range(cfg.n_routed_experts)
        ],
        "shared_expert_weights": {
            "gate_proj": _rand(moe_i * cfg.n_shared_experts, h),
            "up_proj": _rand(moe_i * cfg.n_shared_experts, h),
            "down_proj": _rand(h, moe_i * cfg.n_shared_experts),
        },
    }


def _random_state_dict(cfg, num_my_layers: int, is_first: bool, is_last: bool) -> dict:
    """Only this rank's slice. The embedding exists on the first rank and the final norm on the last
    (TtPrefillTransformer builds them conditionally), so building them everywhere would burn host
    memory on tensors the other stages discard -- the embedding alone is vocab x hidden."""
    sd = {"layers": [_random_layer_state(cfg) for _ in range(num_my_layers)]}
    if is_first:
        sd["embed_weight"] = (torch.randn(cfg.vocab_size, cfg.hidden_size) * 0.02).float()
    if is_last:
        sd["norm_weight"] = _rand(cfg.hidden_size)
        sd["lm_head_weight"] = _rand(cfg.vocab_size, cfg.hidden_size)
    return sd


def main() -> None:
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    num_ranks = int(ttnn.distributed_context_get_size())

    from models.demos.common.prefill.adapter import PrefillRunParams
    from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
    from models.demos.deepseek_v3_d_p.tt.runners.adapters.mistral_small_4_119b import MistralSmall4119BAdapter
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

    adapter = MistralSmall4119BAdapter()
    cfg = mistral4_hf_config(max_seq=MAX_SEQ_LEN)
    cfg.max_seq_len = MAX_SEQ_LEN
    # Shrink both the layer count and the expert count. n_routed_experts has to move on the hf_config
    # AND on the static model_cfg the TT layer code reads its expert counts from, or the two disagree
    # and the dispatch table is sized for a different model than the weights.
    cfg.num_hidden_layers = NUM_LAYERS
    cfg.n_routed_experts = N_EXPERTS

    class _Cfg(Mistral4Small119BConfig):
        NUM_ROUTED_EXPERTS = N_EXPERTS
        NUM_LAYERS = NUM_LAYERS

    layer_split = compute_layer_split(NUM_LAYERS, num_ranks, adapter.layer_split_boundaries(NUM_LAYERS))
    first_layer_idx, num_my_layers = layer_split[rank]
    is_first, is_last = rank == 0, rank == num_ranks - 1
    logger.info(
        f"[pp4 rank {rank}/{num_ranks}] layers=[{first_layer_idx},{first_layer_idx + num_my_layers}) "
        f"experts={N_EXPERTS} chunk={CHUNK_SIZE} is_first={is_first} is_last={is_last}"
    )

    mesh_device = open_mesh_device((SP, TP), _Cfg, l1_small_size=adapter.l1_small_size)
    logger.info(f"[pp4 rank {rank}] mesh {mesh_device.shape} / {mesh_device.get_num_devices()} devices")

    t0 = time.perf_counter()
    state_dict = _random_state_dict(cfg, num_my_layers, is_first, is_last)
    logger.info(f"[pp4 rank {rank}] random state_dict built ({time.perf_counter() - t0:.1f}s)")

    params = PrefillRunParams(
        mesh_shape=(SP, TP),
        num_layers=num_my_layers,
        first_layer_idx=first_layer_idx,
        is_first_rank=is_first,
        is_last_rank=is_last,
        max_seq_len=MAX_SEQ_LEN,
        chunk_size=CHUNK_SIZE,
        num_users=NUM_USERS,
        capacity_factor=2,
        num_links=2,
        gate_mode_name=adapter.default_gate_mode,
        kv_only_last_layer=is_last,
        weight_cache_path=None,
        sparse_kv_cache_format=adapter.default_sparse_kv_cache_format,
        use_trace=False,
        overlap_shared_expert_with_dispatch=False,
    )

    runtime = TtPrefillRuntime(
        mesh_device=mesh_device,
        hf_config=cfg,
        state_dict=state_dict,
        config=TtPrefillRuntimeConfig(
            num_layers=num_my_layers,
            max_seq_len=MAX_SEQ_LEN,
            mesh_shape=(SP, TP),
            chunk_size=CHUNK_SIZE,
            num_users=NUM_USERS,
            sp_axis=0,
            tp_axis=1,
            num_links=2,
            topology=per_axis_topology(),
            capacity_factor=2,
            gate_fallback_mode=GateComputeMode[adapter.default_gate_mode],
            weight_cache_path=None,
            model_cfg=_Cfg,
            first_layer_idx=first_layer_idx,
            is_first_rank=is_first,
            is_last_rank=is_last,
            kv_only_last_layer=is_last,
            routing_use_l1_small_for_semaphores=adapter.routing_use_l1_small_for_semaphores,
            sparse_kv_cache_format=adapter.resolve_sparse_kv_cache_format(adapter.default_sparse_kv_cache_format),
            use_trace=False,
            overlap_shared_expert_with_dispatch=False,
        ),
    )
    del state_dict
    logger.info(f"[pp4 rank {rank}] MODEL BUILT ({time.perf_counter() - t0:.1f}s)")

    kv_caches = adapter.allocate_kv_cache(mesh_device=mesh_device, hf_config=cfg, params=params)
    runtime.compile(kv_caches)
    ttnn.distributed_context_barrier()
    logger.info(f"[pp4 rank {rank}] COMPILED + KV ALLOCATED")

    inbound, outbound = _build_endpoints(mesh_device, rank, num_ranks)
    ttnn.distributed_context_barrier()
    # Grant the inbound its fabric links before receiving (see probe_pp4 / the runner's _lease_reclaim).
    if inbound is not None:
        inbound.wait_for_fabric_links()
    if outbound is not None:
        outbound.wait_for_fabric_links()
    if inbound is not None:
        inbound.release_fabric_links()
    logger.info(f"[pp4 rank {rank}] D2D ENDPOINTS OK")

    meta = {"slot_id": 0, "actual_start": 0, "actual_end": CHUNK_SIZE}
    if is_first:
        tokens = torch.randint(0, cfg.vocab_size, (SP, 1, CHUNK_SIZE // SP), dtype=torch.int32)
        inp = ttnn.from_torch(
            tokens,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.create_mesh_mapper(mesh_device, H2D_MAPPER_CONFIG),
        )
    else:
        inp, meta = _recv(inbound)
        logger.info(f"[pp4 rank {rank}] received activation from rank {rank - 1} meta={meta}")

    t1 = time.perf_counter()
    out = runtime.prefill_chunk(inp, kv_caches, slot_id=0, actual_start=0, actual_end=CHUNK_SIZE, request_id=0)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[pp4 rank {rank}] CHUNK DONE in {(time.perf_counter() - t1) * 1000:.0f}ms")

    if not is_last:
        _send(outbound, out, meta)
        outbound.release_fabric_links()
        logger.info(f"[pp4 rank {rank}] forwarded hidden state to rank {rank + 1}")
    else:
        # The last stage is headless: prefill_chunk returns None because the populated KV cache IS the
        # output of a chunked prefill, so there is no tensor to check here.
        logger.info(f"[pp4 rank {rank}] PIPELINE COMPLETE — KV populated on all {num_ranks} stages")

    ttnn.distributed_context_barrier()
    # Drop the sockets before the mesh: their shutdown submits to a command queue, so letting them be
    # collected after close_mesh_device turns a clean exit into a wall of "cq_id 0 is out of range".
    del inbound, outbound
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[pp4 rank {rank}] done")


if __name__ == "__main__":
    main()
