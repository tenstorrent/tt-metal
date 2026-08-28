# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Three 5120-token chunks of GLM-5.2 MTP4 through the real ``TtPrefillTransformer`` (#53533).

The MTP token contract: per chunk the transformer is handed ``C + K`` ids (5120 + 4), level ``k``
consumes the slice ``[k+1 : k+1+C]``, and that slice -- embedded, not a hidden state -- is level
``k``'s input. Three things in it can be wrong, none of them a shape error:

1. **Stream formation.** An interior chunk borrows the next chunk's first ``K`` prompt ids; the LAST
   chunk has no next chunk and must fill ``K`` slots from its own LM head, one per level, in order.
2. **Slicing / upload.** The window is sharded row -> position by exactly the permutation the trunk
   input took, so ``t_{p+k+1}`` lands on the row whose hidden sits at ``p``. Shift the ids before
   the shard and it is free; shift after and it is wrong by a whole chip.
3. **Numerics, chunked.** ``test_mtp.py::test_mtp_predictor_pcc`` gates ``TtMTPPredictor``
   single-shot; nothing gated it *chunked*, where each level's KV cache is written three times and
   every chunk after the first attends over keys the earlier ones left behind. The reference is
   **teacher-forced** -- level ``k`` is fed the DEVICE's own ``H^{k-1}`` -- so each level's PCC is a
   statement about that level alone. Its embedding is a HOST gather of the ids the level *should*
   have asked for: bit-identical to the device's when the window is right, unrelated to it when it
   is not, so a wrong window and wrong math fail the same assertion.

**Placement is not this test's subject.** ``_mtp_embed_window`` takes a host ``list[int]``, so the
packing under it is the trunk's own and byte-identical for every level and shift;
``test_mtp_device_windows.py`` measures that directly, in both balance modes. What this test owns is
everything upstream of that list: which ids each level asks for, across three chunks, through the
real predictor and the real LM head.

The prompt id at absolute position ``p`` is ``p + 1`` (0 stays reserved for pad), so a failure reads
as "row j is carrying position q".

What it deliberately does NOT claim
-----------------------------------
* **A legible diagnosis of a wrong window.** Claim 2 detects one decisively, but reports it as a
  collapsed PCC rather than as "level 2 asked for id X, expected Y":
  ``glm_mtp_predictor_reference`` takes *embeddings* and never sees a token id. The alternative --
  a test-only recorder wrapped round ``_mtp_embed_window`` -- has no counterpart in production and
  made the numerics blind to the mapping, since the reference was then fed the device's own
  embeddings. Assertion order recovers most of the localisation: fused projection first, at 0.999.
* **The KVPE cache contents.** Level outputs are compared; the slots they wrote are not
  (``test_mtp.py::test_mtp_predictor_pcc`` does that single-shot). A slot *collision* still shows up
  from chunk 1 on -- two levels sharing a slot read each other's keys -- but a consistent off-by-N
  into slots nothing else uses does not.
* **Mid-slab chunk starts.** All three chunks start at a multiple of the global chunk size, so
  ``rotated_chip_positions``' KV-pad-aware rotation is degenerate. A non-aligned boundary is a
  placement question, so it belongs to ``test_mtp_device_windows.py``.
* **Multi-rank.** ``TtPrefillTransformer`` asserts MTP needs an embedding table on the rank that
  runs the tail, which today means single-galaxy (``is_first_rank == is_last_rank``).
* **Weight loading, past the four MTP tensors.** MLA, the 256 experts, the embedding table and the
  LM head stay seeded-random on both legs; the decoder layer's real weights cannot be covered at
  this scale anyway -- a trained GLM gate on a synthetic input collapses isolated-block PCC to ~0.1
  (measured by ``test_glm_prefill_block``). See the ``use_pretrained`` parametrize.
"""

from __future__ import annotations

import copy
import gc
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import SparseMLAReference
from models.demos.deepseek_v3_d_p.reference.glm_5_2.mtp import glm_mtp_predictor_reference
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.sparse_mla.sparse_mla_reference import build_weights
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank, num_full_indexer_layers
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.token_windows import GEN_SLOT, mtp_chunk_stream
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.tt_mtp import CHAIN_FROM_NORM, TtMTPPredictor
from models.demos.deepseek_v3_d_p.tt.mtp_prefill.utils import enable_mtp_indexer_slot
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from tests.ttnn.utils_for_testing import assert_with_pcc

SP_AXIS, TP_AXIS = 0, 1

# Restated from tests/mtp_prefill/test_mtp.py, not imported: importing a test module from a test
# module is how one file ends up collected twice under two names. Same claims -- two RMSNorms and a
# matmul earn the rmsnorm/ffn threshold, and the MTP layer is an ordinary GLM MoE block, so it earns
# test_prefill_block.py's block threshold. No per-level drift allowance: the reference is
# teacher-forced, so nothing is inherited.
FUSED_MTP_PCC = 0.999
MTP_MODULE_OUTPUT_PCC = 0.98

# The pretrained leg's own allowance; the random leg keeps 0.98 rather than both dropping to the
# looser number, which would cost it half its sensitivity to buy nothing it needs.
#
# MEASURED L0 block output by chunk -- random 0.99319 / 0.99286 / 0.99268, flat; pretrained
# 0.98259 / 0.97435 / 0.97085, sliding by -0.0082 then -0.0035. Two drivers, only the first
# established:
#   * Teacher forcing pins each level's hidden INPUT but NOT the KV caches -- each side accumulates
#     K/V from its own outputs -- so chunk 0's divergence seeds chunk 1's and compounds. The deltas
#     halve per chunk, so it saturates rather than diverging.
#   * The starting gap is HYPOTHESIS: a realistically-distributed ``x`` through a SYNTHETIC gate
#     plausibly puts more top-8 selections near ties. Unmeasured -- it needs a device-vs-CPU top-8
#     mismatch count.
#
# 0.96, not the measured 0.97085 floor: run-to-run noise is 1.6e-4 to 2.9e-4, and extrapolating the
# halving deltas puts a 5-chunk run near 0.9686. It does not weaken the claim this leg exists for --
# the fused projection is flat at 0.99995, gated at 0.999.
MTP_MODULE_OUTPUT_PCC_PRETRAINED = 0.96

CHUNK = 5 * 1024  # 5120 -- the "5k chunk" of #53533, and TtPrefillTransformer.seq_len
NUM_CHUNKS = 3
NUM_LEVELS = 4  # MTP4
TOTAL = CHUNK * NUM_CHUNKS  # 15360 prompt tokens

# Only needs to exceed TOTAL so "id = absolute position + 1" stays injective; GLM-5.2's real 154880
# would cost ~1.9 GiB per table instead of ~400 MiB. Nothing under test depends on it: the embedding
# is a gather, and the LM head need only produce SOME in-vocab id for the generation slots.
VOCAB = 32768


def _shard_dims():
    dims = [None, None]
    dims[TP_AXIS] = -1
    dims[SP_AXIS] = -2
    return dims


def _from_device(t: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """``[1, 1, C/sp, H/tp]`` per chip -> ``[1, 1, C, H]`` in POSITION order.

    Position order because this test runs ``is_balanced=False``: ``prepare_prefill_input_tensor``
    then shards by a plain ``reshape(sp, 1, C // sp)``, so concatenating the chips back along ``-2``
    is the exact inverse. Under ``is_balanced=True`` it is not, and the CPU reference -- which reads
    row ``j`` as absolute position ``actual_start + j`` -- would need
    ``reverse_reorder_tensor_chunks`` first.
    """
    return ttnn.to_torch(
        t, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=_shard_dims(), mesh_shape=mesh_device.shape)
    ).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# The token-level reference
# ---------------------------------------------------------------------------


def _expected_window(full_seq: list[int], chunk_idx: int, level: int) -> list[int]:
    """The ONE statement this whole test is checking, written independently of production code.

    Level ``k`` of chunk ``c`` sees ``full_seq[c*C + k + 1 : c*C + k + 1 + C]``. ``full_seq`` is the
    prompt followed by the ``K`` tokens the last chunk generated, so one expression covers interior
    chunks (lookahead borrowed from the next chunk) and the last one (tail = generated) with no
    special case. Derived from the definition of MTP, not from ``mtp_chunk_stream`` -- derived from
    that it would agree with it however wrong it was.
    """
    start = chunk_idx * CHUNK + level + 1
    return list(full_seq[start : start + CHUNK])


def _host_window_embedding(embed_table: torch.Tensor, window_ids: list[int]) -> torch.Tensor:
    """The embedding the device MUST have produced for ``window_ids`` [C, H], derived here.

    Bit-identical to the device's, not merely close, and that is what lets it drive the CPU
    reference: ``_mtp_embed_window`` is a token upload, ``ttnn.embedding``'s row gather and an
    optional multiply by a 0/1 mask -- no arithmetic anywhere -- and ``TtParallelEmbedding`` stores
    the table as ``ttnn.bfloat16``. Driving the reference from the EXPECTED ids rather than the ones
    the device actually asked for is what makes the numerics sensitive to the token->window mapping.

    The position-0 mask is deliberately NOT applied here: ``fused_mtp_reference`` zeroes row 0 itself
    from the ``positions`` this test passes, and leaving it out is what lets the row-0 check below
    tell a device that masked from one that did not.
    """
    return embed_table[torch.tensor(window_ids, dtype=torch.long)]


_MESH_PARAMS = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(
            fabric_payload_size=GLM52Config.FABRIC_PAYLOAD_SIZE,
            worker_l1_size=ttnn._ttnn.device.DEFAULT_WORKER_L1_SIZE,
        ),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
]


@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
# One trunk layer. GLM-5.2's NUM_DENSE_LAYERS is 3, so layer 0 is dense: a real h^0 through the real
# embedding, norm and LM head, without 256 random experts whose numerics this test never checks.
@pytest.mark.parametrize("num_layers", [1], ids=["layers1"])
@pytest.mark.parametrize("variant", ["glm_5_2"], indirect=True, ids=["glm52"])
# Weight axis. ``use_pretrained`` reaches ONLY the four MTP tensors -- enorm, hnorm, eh_proj and
# shared_head.norm, via conftest's ``mtp_cfg`` + ``mtp_state_dict``. MLA, the 256-expert MoE, the
# embedding table and the LM head are seeded-random on BOTH legs, built below.
#
# That narrowness is what makes this leg a measurement rather than noise: test_mtp.py's module and
# predictor tests are random-only because a TRAINED GLM gate on a synthetic input picks different
# top-8 experts on device than on CPU, collapsing isolated-block PCC to ~0.1 while the same layer
# scores ~0.995 in context (measured by ``test_glm_prefill_block``) -- an artifact, not an op bug.
# This axis never touches the gate, so that mode is ATTENUATED, not absent, and that was learned the
# hard way by this leg failing at 0.98: making the INPUT realistic while the gate stays synthetic
# moved L0's block output 0.9932 -> 0.9827 -> 0.9746. Hence MTP_MODULE_OUTPUT_PCC_PRETRAINED above.
#
# What the axis changes, measured off the checkpoint at layer 78:
#
#     eh_proj embed half   std 0.0149   max|w| 0.227   max/mean|w|    19.2
#     eh_proj hidden half  std 0.0238   max|w| 2.359   max/mean|w|   211.6
#     enorm gain           std 0.0033   max|w| 0.053   max/mean|w|     1.3
#     hnorm gain           std 0.0110   max|w| 0.459   max/mean|w|     6.2
#
# ``random_mtp_state_dict`` draws eh_proj from ``randn * (2H)**-0.5`` (both halves alike, no tail)
# and both gains from ``randn*0.1 + 1``, so this leg is the only place the real half-to-half
# asymmetry and 211:1 tail go through the concat, the bf16 matmul and the reduce_scatter at four
# levels and across chunk seams. It holds at 0.99995.
# ``test_mtp.py::test_fused_mtp_pcc[pretrained]`` covers the same four tensors standalone and
# single-shot; this is the only place they run inside the transformer. Four tensors out of 141
# shards, and it skips cleanly on a box with no checkpoint.
@pytest.mark.parametrize("use_pretrained", [False, True], ids=["random", "pretrained"], indirect=True)
@pytest.mark.skipif(not is_blackhole(), reason="DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_mtp_transformer_chunks(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    num_layers,
    use_pretrained,
    mtp_cfg,
    mtp_state_dict,
    monkeypatch,
):
    """3 x 5120 tokens of GLM-5.2 MTP4 end to end: every window, every level, exact ids.

    Assertions, most-local first so a failure localises itself:

    ===  ======================================================================================
     #   claim
    ===  ======================================================================================
     1   ``mtp_chunk_stream`` hands the transformer ``C + K`` ids: for an interior chunk exactly
         ``prompt[cC : cC + C + K]``, for the last chunk the prompt tail plus ``K`` sentinels.
     2   Every level's fused projection, block output and post-``shared_head.norm`` match
         ``glm_mtp_predictor_reference`` on every chunk, teacher-forced from the device's own
         ``H^{k-1}`` and driven by the embedding of :func:`_expected_window`'s ids -- so this is
         also the next-token-formation claim: adjacent token embeddings are unrelated, so a level
         that embedded a shifted window decorrelates against it outright.
     3   Absolute position 0 -- and only it, and only on chunk 0 -- is masked, vLLM's
         ``torch.where(positions == 0, 0, inputs_embeds)``. Asserted on row 0 alone: one row in
         ``C`` moves the whole-tensor PCC by less than its run-to-run noise.
     4   The last chunk's ``K`` sentinels became the ``K`` LM-head tokens, level by level: level
         ``k``'s window ends with ``generated[:k+1]``. Asserted on those ``k+1`` rows, same reason.
    ===  ======================================================================================

    Claim 2 is where the runtime goes: 12 CPU evaluations of a 256-expert GLM MoE decoder layer over
    a KV cache growing to 15360 positions -- 45.7 / 48.9 / 77.8s per chunk, rising with the chunk's
    start because that is how much cache each level attends over. It is not in any CI yaml.
    """
    torch.manual_seed(42)

    block_output_pcc = MTP_MODULE_OUTPUT_PCC_PRETRAINED if use_pretrained else MTP_MODULE_OUTPUT_PCC

    topology = per_axis_topology(device_params["fabric_config"])
    mesh_shape = list(mesh_device.shape)
    sp_factor = mesh_shape[SP_AXIS]

    # copy.copy, not the shared object: config_only is lru_cached, so the mutations below would
    # otherwise follow every other GLM-5.2 test. They rebind attributes, so shallow is enough.
    config = copy.copy(config_only)
    config.max_seq_len = TOTAL
    config.vocab_size = VOCAB

    # The indexer map is rebuilt for the truncated model this test builds, and the MTP layer sits at
    # NUM_DENSE_LAYERS rather than its real 78. Both follow from ONE constraint: trunk and MTP share
    # a single index-K cache whose per-user slot stride TtIndexer derives two ways -- the trunk from
    # ``full_indexer_rank(first_layer_idx + layer_num)``, the MTP block from
    # ``num_full_indexer_layers`` (TtMTPModule passes no first_layer_idx, so it reads the whole-model
    # count) -- and update_padded_kv_cache asserts the cache's batch dim is a multiple of it. Against
    # the model's own 78-entry map the two disagree (3 vs 22 at num_layers=1), because layer 78's
    # compacted rank of 21 is a fact about the full model, not a truncated one. Truncating the map,
    # then appending the MTP slot, makes both sides agree by construction. NUM_DENSE_LAYERS is the
    # smallest index that still sends TtPrefillBlock down its MoE branch, so the MTP block stays the
    # full 256-expert MoE that the real layer 78 is.
    config.indexer_types = list(config.indexer_types[:num_layers])
    layer_idx = enable_mtp_indexer_slot(config, GLM52Config.NUM_DENSE_LAYERS)
    trunk_stride = full_indexer_rank(config, num_layers + NUM_LEVELS)
    mtp_stride = num_full_indexer_layers(config)
    assert trunk_stride == mtp_stride, (
        f"index-K cache stride disagrees: trunk {trunk_stride} vs MTP {mtp_stride}. They share one "
        "cache, so update_padded_kv_cache raises TT_FATAL 'cache batch dim must be a multiple of "
        "num_layers' on the first indexer write -- before any assertion below is reached."
    )
    hidden = config.hidden_size
    assert hidden == mtp_cfg.hidden_size

    logger.info(
        f"[mtp chunks] mesh={mesh_shape} chunks={NUM_CHUNKS}x{CHUNK} total={TOTAL} K={NUM_LEVELS} "
        f"trunk_layers={num_layers} mtp_layer={layer_idx} vocab={VOCAB} "
        f"mtp_weights={'pretrained' if use_pretrained else 'random'}"
    )

    # --- Prompt: id at absolute position p is p + 1, so a decoded id names its own position -------
    prompt = list(range(1, TOTAL + 1))
    assert max(prompt) < VOCAB

    # --- Weights ---------------------------------------------------------------------------------
    # Assembled directly in TtPrefillTransformer's state_dict format, not via create_hf_model /
    # extract_tt_state_dict: those need ``variant.reference_model_cls`` and neither GLM adapter wires
    # one, GLM shipping no HF modeling code this repo can instantiate. So the weights are generated
    # here, seeded, at the shapes the device modules ask for.
    #
    # build_weights (not extract_layer_state_dict) supplies MLA because it emits GLM's indexer
    # tensors -- wk / wq_b / weights_proj / k_norm -- which the extractor does not, and TtIndexer's
    # fallback for a missing weight is torch.empty, which would make the last chunk's generated
    # tokens irreproducible. One set shared by every block: nothing here checks their numerics.
    mla_weights, _ = build_weights(variant, config, seed=42)

    def _randn(*shape, seed):
        return torch.randn(*shape, generator=torch.Generator().manual_seed(seed)) * hidden**-0.5

    state_dict = {
        # float32 to match what extract_tt_state_dict hands the embedding.
        "embed_weight": _randn(VOCAB, hidden, seed=10).float(),
        "norm_weight": (torch.randn(hidden, generator=torch.Generator().manual_seed(12)) * 0.1 + 1.0).to(
            torch.bfloat16
        ),
        "layers": [_random_glm_layer_state_dict(config, mla_weights, layer_idx=i) for i in range(num_layers)],
        "lm_head_weight": _randn(VOCAB, hidden, seed=11).to(torch.bfloat16),
    }

    # --- The MTP predictor -------------------------------------------------------------------------
    # Built outside the constructor call and kept: claim (2)'s CPU reference needs the SAME layer
    # weights the device got, so the dict must outlive construction. ~19 GiB resident (256 x 3 x
    # [2048, 6144] bf16 plus the shared expert), which is why the trunk's state_dict is dropped
    # below.
    mtp_layer_sd = _random_glm_layer_state_dict(config, mla_weights, layer_idx=layer_idx)
    ref_moe_weights = {k: mtp_layer_sd[k] for k in ("gate_weights", "routed_expert_weights", "shared_expert_weights")}

    # first_cache_slot / layer_num are the two the transformer asserts on: levels write KV slots
    # [num_layers, num_layers + K) and every block must stride users by the cache's true depth.
    # is_chunked/max_seq_len/slot_num reach TtPrefillBlock through TtMTPModule's **block_kwargs, so
    # the MTP layer is chunked exactly like the trunk's.
    predictor = TtMTPPredictor(
        mesh_device,
        config,
        variant.model_config,
        {"mtp": mtp_state_dict, "layer": mtp_layer_sd},
        mtp_cfg,
        seq_len=CHUNK,
        num_levels=NUM_LEVELS,
        layer_idx=layer_idx,
        first_cache_slot=num_layers,
        tp_axis=TP_AXIS,
        sp_axis=SP_AXIS,
        num_links=num_links,
        topology=topology,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        is_chunked=True,
        max_seq_len=TOTAL,
        slot_num=1,
        layer_num=num_layers + NUM_LEVELS,
    )

    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict=state_dict,
        num_layers=num_layers,
        seq_len=CHUNK,
        max_seq_len=TOTAL,
        num_links=num_links,
        topology=topology,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        is_balanced=False,
        padding_side="right",
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        lm_head_is_column_parallel=True,
        is_chunked=True,
        slot_num=1,
        mtp_predictor=predictor,
    )
    # Taken BEFORE the free: every MTP window below is embedded from this, host side. bf16 is the
    # dtype TtParallelEmbedding stores the table in, so the gather matches the device row for row.
    embed_table = state_dict["embed_weight"].to(torch.bfloat16)
    del state_dict
    gc.collect()
    ttnn.synchronize_device(mesh_device)

    assert transformer.num_kvpe_cache_layers == num_layers + NUM_LEVELS
    assert transformer.num_mtp_levels == NUM_LEVELS

    # Teacher forcing hands level k the device's ``out_head_normed[k-1]``, which is H^{k-1} only
    # under the "norm" chaining convention. Asserted, not assumed: chain_from is a constructor
    # argument, and flipping it makes every level>0 comparison below meaningless, with no shape
    # error to say so.
    assert predictor.chain_from == CHAIN_FROM_NORM, (
        f"the reference below is teacher-forced with out_head_normed, so the device must chain from "
        f"it too; predictor.chain_from is {predictor.chain_from!r}"
    )

    # One persistent SparseMLAReference PER LEVEL, carried across all three chunks. Per level because
    # each level owns its own KV cache -- one shared instance would let level k attend to level k-1's
    # keys. Persistent because the caches and the fill watermark live on the instance, so a fresh one
    # per chunk would make every chunk attend over itself alone: exactly the bug this test exists to
    # catch, baked into its reference instead.
    ref_mla = [SparseMLAReference(config, mla_weights, seq_len=TOTAL) for _ in range(NUM_LEVELS)]

    # --- Caches ------------------------------------------------------------------------------------
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=TOTAL,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=transformer.num_kvpe_cache_layers,
        num_users=1,
    )
    # Sized over the compacted full-indexer space of the map built above, which both sides agree on
    # (asserted there). TtIndexer derives each block's slot from its static layer_idx: the trunk
    # takes slots [0, num_layers), MTP the appended one.
    index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=TOTAL,
        mesh_shape=mesh_shape,
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=num_full_indexer_layers(config),
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )

    mesh_device.enable_program_cache()

    # --- Drive the three chunks --------------------------------------------------------------------
    # h^0 -- the post-``model.norm`` trunk output the predictor is seeded with -- is the one thing
    # this test needs that no production path hands back: not in forward()'s return value, not on
    # ``MTPPredictorOutput``. So it is taken where it is passed; the call site is
    # ``self.run_mtp(...)``, so an instance attribute wins at call time.
    h0_host: dict = {}
    real_run_mtp = transformer.run_mtp

    def _capture_h0(h_normed, *args, **kwargs):
        h0_host["h"] = _from_device(h_normed, mesh_device)
        return real_run_mtp(h_normed, *args, **kwargs)

    monkeypatch.setattr(transformer, "run_mtp", _capture_h0)

    captured: dict = {}

    def _on_mtp_complete(mtp_out, mtp_generated):
        captured["out"] = mtp_out
        captured["generated"] = list(mtp_generated)

    windows: list[list[list[int]]] = []  # [chunk][level] -> ids in POSITION order
    generated: list[int] = []

    for chunk_idx in range(NUM_CHUNKS):
        start = chunk_idx * CHUNK
        is_last = chunk_idx == NUM_CHUNKS - 1

        stream, real_len = mtp_chunk_stream(prompt, chunk_idx, CHUNK, NUM_LEVELS)

        # (1) what the transformer is handed: C + K ids, K of them lookahead.
        assert len(stream) == CHUNK + NUM_LEVELS, f"chunk {chunk_idx} stream is {len(stream)}, expected 5124"
        assert real_len == CHUNK, f"chunk {chunk_idx} real_len {real_len}; all three chunks are full"
        assert stream[:CHUNK] == prompt[start : start + CHUNK]
        if is_last:
            assert stream[CHUNK:] == [GEN_SLOT] * NUM_LEVELS, (
                "the last chunk has no next chunk, so its K lookahead ids must be generation "
                f"sentinels; got {stream[CHUNK:]}"
            )
        else:
            assert stream == prompt[start : start + CHUNK + NUM_LEVELS], (
                f"chunk {chunk_idx} is interior, so its extended stream is exactly the plain prompt "
                "slice -- the K lookahead ids are the NEXT chunk's first K tokens"
            )

        tt_tokens = prepare_prefill_input_tensor(
            stream[:CHUNK], mesh_device, sp_factor, False, tuple(mesh_shape), SP_AXIS
        )
        h0_host.clear()
        captured.clear()

        logger.info(f"[mtp chunks] chunk {chunk_idx}: start={start} real_len={real_len} last={is_last}")
        transformer.forward(
            tt_tokens,
            kvpe_cache,
            actual_isl=real_len,
            actual_start=start,
            actual_end=start + real_len,
            cache_user_id=0,
            index_kv_cache=index_kv_cache,
            mtp_tokens=stream,
            on_mtp_complete=_on_mtp_complete,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(tt_tokens)

        assert "generated" in captured, "on_mtp_complete never fired -- the MTP branch did not run"
        assert len(captured["out"].x) == NUM_LEVELS
        chunk_generated = captured["generated"]
        if is_last:
            assert (
                len(chunk_generated) == NUM_LEVELS
            ), f"the last chunk must generate one token per level to fill its K slots; got {len(chunk_generated)}"
            generated = chunk_generated
        else:
            assert chunk_generated == [], (
                f"chunk {chunk_idx} is interior: every window is a pure prompt slice, so no LM-head "
                f"round trip should have happened, but it generated {chunk_generated}"
            )

        # Complete only once the last chunk has generated; interior chunks never index past TOTAL.
        full_seq = prompt + generated
        windows.append([])

        # The window each level MUST embed, derived from the definition of MTP rather than observed.
        # The reference below is driven by these ids -- claim (2) -- and adjacent token embeddings
        # are unrelated, so a level that embedded a shifted window fails its fused PCC outright.
        windows[chunk_idx].extend(_expected_window(full_seq, chunk_idx, level) for level in range(NUM_LEVELS))
        assert all(len(w) == CHUNK for w in windows[chunk_idx]), (
            f"chunk {chunk_idx}: a window came out short -- full_seq is {len(full_seq)} ids, which "
            f"does not cover this chunk's shift-{NUM_LEVELS} lookahead"
        )
        host_embeds = [_host_window_embedding(embed_table, w) for w in windows[chunk_idx]]

        # --- (2) numerics, teacher-forced ---------------------------------------------------------
        res = captured["out"]
        assert "h" in h0_host, "run_mtp never ran, so h^0 was never captured"

        # H^{k-1} for the reference is the DEVICE's, never the reference's own previous output:
        # level 0 gets h^0 off the trunk, level k>0 gets level k-1's device output.
        #
        # EVERY device readback happens here, before the reference runs. The reference is 45-140s of
        # pure host torch, and pulling a device tensor across that gap killed the first run: SIGBUS
        # inside ttnn.to_torch at chunk 4, on the first device touch after the longest gap (137.6s).
        # Stale mapping vs. the host failing to back a page at peak RSS was never established (no
        # dmesg here), but neither can happen if the device is untouched until the next chunk. It
        # costs ~60 MiB per extra host copy.
        dev_x = [_from_device(res.x[k], mesh_device) for k in range(NUM_LEVELS)]
        dev_out = [_from_device(res.out[k], mesh_device) for k in range(NUM_LEVELS)]
        dev_normed = [_from_device(res.out_head_normed[k], mesh_device) for k in range(NUM_LEVELS)]
        del res
        ref_hiddens = [h0_host["h"]] + dev_normed[:-1]

        t0 = time.monotonic()
        ref_xs, ref_outs, ref_normeds, _ = glm_mtp_predictor_reference(
            config,
            mla_weights,
            mtp_state_dict,
            mtp_layer_sd["attn_norm_weight"],
            mtp_layer_sd["ffn_norm_weight"],
            [e.unsqueeze(0) for e in host_embeds],
            ref_hiddens[0].squeeze(0),
            TOTAL,
            moe_weights=ref_moe_weights,
            num_levels=NUM_LEVELS,
            index_share=predictor.index_share,
            hiddens=[h.squeeze(0) for h in ref_hiddens],
            mla_refs=ref_mla,
            # Row j of this chunk is absolute position start + j. Without this, fused_mtp_reference
            # defaults to arange(C) and every chunk zeroes its own row 0, which the device does only
            # on chunk 0.
            positions=torch.arange(start, start + CHUNK),
            actual_start=start,
            actual_end=start + real_len,
        )
        logger.info(f"[mtp chunks] chunk {chunk_idx}: CPU reference took {time.monotonic() - t0:.1f}s")

        for level in range(NUM_LEVELS):
            # Most-local first: the fused projection is two norms and a matmul, the block output
            # adds a whole DSA-MLA + 256-expert MoE, the third is that plus one more norm.
            _, msg = assert_with_pcc(ref_xs[level].unsqueeze(0), dev_x[level], FUSED_MTP_PCC)
            logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: fused projection PCC {msg}")
            _, msg = assert_with_pcc(ref_outs[level].unsqueeze(0), dev_out[level], block_output_pcc)
            logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: block output PCC {msg}")
            _, msg = assert_with_pcc(ref_normeds[level].unsqueeze(0), dev_normed[level], block_output_pcc)
            logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: shared_head.norm PCC {msg}")

            # Two claims the whole-tensor PCC cannot resolve: each turns on at most K rows out of C,
            # ~1e-4 of the elements, under the gate AND under the 1.6e-4 to 2.9e-4 run-to-run noise.
            # On their own rows a wrong token is a total mismatch.
            if chunk_idx == 0:
                # (3) the position-0 mask: the reference zeroes row 0 from the positions passed
                # above and _host_window_embedding does not, so this row agrees only if the DEVICE
                # masked it too.
                _, msg = assert_with_pcc(ref_xs[level].unsqueeze(0)[:, :, :1], dev_x[level][:, :, :1], FUSED_MTP_PCC)
                logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: position-0 mask PCC {msg}")
            if is_last:
                # (4) the generation seam: the last level+1 rows carry ids only the LM head can
                # produce, and the reference got them from _expected_window, not from the device.
                seam = CHUNK - level - 1
                _, msg = assert_with_pcc(
                    ref_xs[level].unsqueeze(0)[:, :, seam:], dev_x[level][:, :, seam:], FUSED_MTP_PCC
                )
                logger.info(f"[mtp chunks] chunk {chunk_idx} L{level}: generation seam PCC {msg}")

        del dev_x, dev_out, dev_normed, ref_hiddens, ref_xs, ref_outs, ref_normeds

    logger.info(f"[mtp chunks] generated tokens for the final K slots: {generated}")


def _random_glm_layer_state_dict(config, mla_weights, layer_idx):
    """Random weights for one GLM-5.2 DSA decoder block, in ``TtPrefillBlock``'s state_dict format.

    ``layer_idx`` selects dense vs MoE by ``TtPrefillBlock``'s own rule (``layer_idx >=
    first_k_dense_replace``), so the choice is made the same way the device makes it. Defined here
    rather than reusing ``test_mtp.py``'s ``_glm_layer_weights`` -- that module is itself collected
    by pytest, and only the state_dict half of what it returns is needed. ``mla_weights`` is shared
    across blocks: nothing here checks their numerics, and one set means one allocation.
    """
    hidden = config.hidden_size
    g = torch.Generator().manual_seed(3 + layer_idx)
    hs = hidden**-0.5

    def _norm(seed):
        return (torch.randn(hidden, generator=torch.Generator().manual_seed(seed)) * 0.1 + 1.0).to(torch.bfloat16)

    def _mlp(inter):
        ds = inter**-0.5
        return {
            "gate_proj": (torch.randn(inter, hidden, generator=g) * hs).to(torch.bfloat16),
            "up_proj": (torch.randn(inter, hidden, generator=g) * hs).to(torch.bfloat16),
            "down_proj": (torch.randn(hidden, inter, generator=g) * ds).to(torch.bfloat16),
        }

    layer_sd = {
        "attn_norm_weight": _norm(1 + layer_idx),
        "mla_weights": mla_weights,
        "ffn_norm_weight": _norm(1001 + layer_idx),
    }
    if layer_idx < config.first_k_dense_replace:
        layer_sd["ffn_weights"] = _mlp(config.intermediate_size)
        return layer_sd
    moe_inter = GLM52Config.MOE_INTERMEDIATE_SIZE
    layer_sd["gate_weights"] = {
        "weight": (torch.randn(GLM52Config.NUM_ROUTED_EXPERTS, hidden, generator=g) * hs).to(torch.bfloat16),
        "e_score_correction_bias": (torch.randn(GLM52Config.NUM_ROUTED_EXPERTS, generator=g) * 0.01).to(torch.float32),
    }
    layer_sd["routed_expert_weights"] = [_mlp(moe_inter) for _ in range(GLM52Config.NUM_ROUTED_EXPERTS)]
    layer_sd["shared_expert_weights"] = _mlp(moe_inter)
    return layer_sd
