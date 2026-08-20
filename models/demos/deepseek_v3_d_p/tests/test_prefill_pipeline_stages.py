# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PP=4 layer slicing, validated inside ONE process.

Builds four `TtPrefillTransformer` instances over the same mesh with `first_layer_idx` 0/9/18/27 and
the rank-boundary flags set, runs them in sequence handing each stage's output hidden state straight
into the next, and asserts the sampled token is identical to a single-rank 36-layer run over the same
weights.

Why this test exists, and what it deliberately does NOT cover: pipeline parallelism has two halves,
and only one of them is unbuilt. The *slicing* half — which rank owns which layers, who runs the
embedding, who runs the norm/LM-head tail, and whether a stage's output activation is a valid input to
the next — is already implemented in `TtPrefillTransformer` and is the half most likely to be subtly
wrong. The *transport* half (a D2D socket publishing rank N's activation to rank N+1) is explicitly a
placeholder in `TtPrefillRuntime.make_placeholder_activation`. Inside one process, transport is a
Python variable, so this isolates the half that can be tested today from the half that has to be
written.

Runs at mesh-8x4 on purpose: TP=1 is covered separately by the `mesh-8x1` rows of
`test_mistral4_prefill_transformer`, and using (8,4) here means the warm pretrained TTNN cache applies,
so both configurations read byte-identical weights and the token comparison is exact rather than
approximate. Each stage gets its OWN 9-slot KV cache, because `forward` writes at the LOCAL
`cache_layer_idx` while blocks are built at the GLOBAL `layer_idx` — a real PP rank owns exactly its
own slice's cache, and sharing one cache across stages would have them overwrite each other.
"""

import gc

import pytest
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.mistral_small4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import PROMPT_1K_PATH, tokenize_prompt_to_isl

TOTAL_LAYERS = 36
ISL = 1024


def _build(mesh_device, config, cache_path, num_layers, first_layer_idx, is_first, is_last, num_links, topology):
    return TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=MistralSmall4Config,
        state_dict={},  # weights come from the TTNN cache
        num_layers=num_layers,
        seq_len=ISL,
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=0,
        tp_axis=1,
        is_balanced=False,
        gate_fallback_mode=GateComputeMode.GPT_DEVICE,
        weight_cache_path=cache_path,
        # Must match test_prefill_transformer: the constructor DEFAULTS to False, and a row-parallel
        # LM head expects the full 4096 hidden, so the last rank's tail matmul fails with
        # "width=1024 height=4096" against a TP-sharded activation. Not a PP issue -- ranks 0-2 have no
        # tail and ran fine without it -- but it is the kind of default a PP driver must not inherit.
        lm_head_is_column_parallel=True,
        routing_use_l1_small_for_semaphores=True,
        first_layer_idx=first_layer_idx,
        is_first_rank=is_first,
        is_last_rank=is_last,
    )


def _kv(mesh_device, config, mesh_shape, num_layers):
    return init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=ISL,
        mesh_shape=mesh_shape,
        sp_axis=0,
        num_kvpe_cache_layers=num_layers,
    )


def _tokens_to_device(token_ids, mesh_device, sp_factor):
    return ttnn.from_torch(
        token_ids.reshape(sp_factor, 1, ISL // sp_factor),
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=tuple(mesh_device.shape)),
    )


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE
                ),
                "l1_small_size": 768,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["mistral_small4"], indirect=True, ids=["mistral4"])
# Layer slicing is independent of stage geometry -- this runs every stage on the full mesh -- so one
# row per pipeline depth covers both candidate configurations: PP=4 for (8,1), PP=2 for (8,2).
@pytest.mark.parametrize("PP", [4, 2], ids=["pp4", "pp2"])
@pytest.mark.timeout(0)
def test_mistral4_pp4_stages_single_process(
    variant, config_only, mesh_device, device_params, weight_cache_path, tokenizer, num_links, topology, PP
):
    if weight_cache_path is None:
        pytest.skip(f"pretrained TTNN cache unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    # config_only, not hf_config: Mistral's config is hand-built (multimodal wrapper + the transformers 5.x
    # rope rename), so the AutoConfig path behind hf_config does not express it. Matches what
    # test_mistral4_prefill_transformer uses.
    assert TOTAL_LAYERS % PP == 0, f"{TOTAL_LAYERS} layers do not divide into {PP} stages"
    LAYERS_PER_STAGE = TOTAL_LAYERS // PP
    config = config_only
    config.max_seq_len = ISL
    mesh_shape = tuple(mesh_device.shape)
    sp_factor = mesh_shape[0]
    cache_path = weight_cache_path / f"{mesh_shape[0]}x{mesh_shape[1]}"

    from models.demos.deepseek_v3.demo.demo import load_prompts_from_json

    prompt = load_prompts_from_json(str(PROMPT_1K_PATH), max_prompts=1)[0]
    token_ids, attention_mask, _ = tokenize_prompt_to_isl(tokenizer, max_isl=ISL, prompt_text=prompt)
    actual_isl = int(attention_mask.sum())
    logger.info(f"PP={PP} x {LAYERS_PER_STAGE} layers, isl={ISL}, real tokens={actual_isl}, mesh={mesh_shape}")

    # --- PP stages, run in sequence, hidden state handed straight over ---
    stages, kvs = [], []
    pp_token = None
    try:
        for rank in range(PP):
            stages.append(
                _build(
                    mesh_device,
                    config,
                    cache_path,
                    LAYERS_PER_STAGE,
                    rank * LAYERS_PER_STAGE,
                    rank == 0,
                    rank == PP - 1,
                    num_links,
                    topology,
                )
            )
            kvs.append(_kv(mesh_device, config, mesh_shape, LAYERS_PER_STAGE))
        logger.info("all 4 stages built")

        handoff = _tokens_to_device(token_ids, mesh_device, sp_factor)
        for rank, (stage, kv) in enumerate(zip(stages, kvs)):
            out = stage.forward(handoff, kv, actual_isl=actual_isl)
            if rank < PP - 1:
                # Non-last rank returns its output hidden state — this is the inter-stage handoff.
                assert isinstance(out, ttnn.Tensor), f"rank {rank} should hand back an activation, got {type(out)}"
                logger.info(
                    f"  rank {rank}: layers {rank*LAYERS_PER_STAGE}-{(rank+1)*LAYERS_PER_STAGE-1} "
                    f"-> activation {list(out.shape)}"
                )
                handoff = out
            else:
                pp_token, pp_prob, _ = out
                logger.info(f"  rank {rank}: tail ran -> token {pp_token} (p={pp_prob:.4f})")
    finally:
        for stage in stages:
            stage.release_sub_device_managers()
    del stages, kvs
    gc.collect()

    # --- single-rank 36 layers over the same cache, for an exact comparison ---
    ref = _build(mesh_device, config, cache_path, TOTAL_LAYERS, 0, True, True, num_links, topology)
    try:
        ref_kv = _kv(mesh_device, config, mesh_shape, TOTAL_LAYERS)
        ref_token, ref_prob, _ = ref.forward(
            _tokens_to_device(token_ids, mesh_device, sp_factor), ref_kv, actual_isl=actual_isl
        )
        logger.info(f"single-rank 36L -> token {ref_token} (p={ref_prob:.4f})")
    finally:
        ref.release_sub_device_managers()

    assert pp_token == ref_token, (
        f"PP={PP} sliced into {LAYERS_PER_STAGE}-layer stages sampled token {pp_token}, "
        f"single-rank 36-layer sampled {ref_token} — the layer slicing or a rank boundary is wrong"
    )
    logger.success(f"PP={PP} slicing matches single-rank: both sampled token {pp_token}")
