# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Kimi-K3 prefilled in 5120-token chunks, carried across all 102400 tokens of the 100k trace.

The depth ladder proves one chunk. This proves the thing a chunk cannot: that the KDA recurrent
carry survives the boundary between chunks and still equals the model's own, twenty times in a row.
That is the part of Kimi-K3 with genuine cross-chunk state — AttnRes has none, since every one of its
reductions is over the hidden dimension and the token axis is a free batch axis, so each chunk opens
a fresh walk while the KDA carries advance.

5120 is not a chosen number. `ttnn.TILE_SIZE(32) * KDA_SUMMARY_GROUP_CHUNKS(20) * SP(8)` is both the
chunk size and the shortest sequence the KDA recurrence accepts on this mesh, so it is the only
chunk size available here.

Two oracles, and the second is the one that matters:

  * each chunk's residual stream against `decoder_output_layer_0` over that window;
  * the KDA recurrent carry at every boundary against `kda_recurrent_state_layer_0`, which the trace
    snapshots every 640 tokens — so a 5120-token boundary is snapshot row `8k - 1`. A run that
    silently restarted its recurrence each chunk would still pass the first oracle on chunk 0 and
    drift afterwards; this one fails immediately at the first boundary.

The golden stores the carry as `[heads, v_dim, k_dim]` and the layer produces `[heads, k_dim,
v_dim]`. Both are `[96, 128, 128]`, so omitting the transpose reports PCC ~0.01 and looks exactly
like a broken recurrence. `test_kda_golden.py` pins that convention; this file just obeys it.

Twenty chunks also covers the standing memory gate: device DRAM is sampled after every chunk and
must not grow, which is what would catch the KDA carries or the walk's per-block batches leaking.
"""

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_hf_config
from models.demos.deepseek_v3_d_p.tests.attn_res.checkpoint_utils import load_attn_res_state_dict
from models.demos.deepseek_v3_d_p.tests.kda.checkpoint_utils import resolve_model_root
from models.demos.deepseek_v3_d_p.tests.kimi_k3.golden import TRACE_1M, TRACE_100K, resolve_checkpoint, resolve_trace
from models.demos.deepseek_v3_d_p.tests.kimi_k3.test_transformer_depth import (
    PLACEMENTS,
    SP_AXIS,
    TP_AXIS,
    _compose,
    _model_state_dict,
)
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res import TtAttnRes
from models.demos.deepseek_v3_d_p.tt.attn_res.attn_res_stream import TtAttnResWalk
from models.demos.deepseek_v3_d_p.tt.attn_res.weights import load_attn_res_weights
from models.demos.deepseek_v3_d_p.tt.kimi_k3.residual import TtAttnResResidual
from models.demos.deepseek_v3_d_p.tt.kimi_k3.transformer import TtKimiK3Transformer
from models.demos.deepseek_v3_d_p.tt.kimi_k3.weights import cache_root
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
from models.demos.deepseek_v3_d_p.tt.runners.input_prep import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import cache_half_pccs, gather_cache_tp0, unrotate_cache_layer

CHUNK = 5120
# 11 chunks is 56320 tokens — the "55k" leg, and past the 10-chunk mark the memory gate wants.
NUM_CHUNKS = 11
# Two depths, because the two traces cover different halves and neither covers both.
#
#   5  on the 100k trace: it snapshots the KDA recurrent state every 640 tokens, so the carry can be
#      checked at every chunk boundary — the one piece of genuine cross-chunk state Kimi-K3 has. It
#      records decoder_output only for layers 0..4, which is what caps this depth. Layer 3 is MLA, so
#      a KV slab written in chunk N is still read back in chunk N+1.
#   24 on the 1M trace: decoder_output for layers 0..24 and all 24 KV slabs, so every layer and the
#      cumulative KV can be scored at full depth — including past the second AttnRes seal at layer
#      12, which is where the sealed set first has two blocks. It carries no KDA snapshots, so the
#      carry oracle is the depth-5 case's job.
DEPTHS = [5, 24]
TOTAL_LEN = CHUNK * NUM_CHUNKS  # the KV cache must span every chunk, not just one
DEEP_TRACE_FROM = 12
SNAPSHOT_STRIDE = 640

# Chunking costs nothing at the first chunk — chunk 0's worst layer is 0.997075 against the 0.997013
# the same layer scores in the one-shot L5 rung — and then decays gently with context as the carry
# and the KV both accumulate: 0.9971 at chunk 0 to 0.9932 at chunk 10 over 56320 tokens. 0.99 sits
# below the whole measured curve while staying far tighter than the package's 0.88 depth floor, so a
# real regression in the carry or the KV handoff still shows.
# Same depth-aware split the ladder uses, for the same reason: past the second AttnRes seal the
# sealed candidate carries a block's worth of accumulated error into every read. Measured here, the
# worst layer is 0.9932 at depth 5 and 0.9814 at depth 24, and in both cases it is the same layer
# and the same value as the corresponding one-shot rung — chunking itself costs nothing.
SHALLOW_OUTPUT_PCC = 0.99
DEEP_OUTPUT_PCC = 0.98
# The carry is a 5120-step bf16 recurrence per chunk compounded across chunks, and it is compared
# against a snapshot the model wrote in fp32; the ladder's own KDA output sits at 0.9999.
CARRY_PCC = 0.99
# Same shallow-layer bar the depth ladder uses for the KV cache.
KV_CACHE_PCC = 0.96


def _compose_carry(mesh_device, state):
    """The global `[heads, k_dim, v_dim]` carry from its TP shards.

    The carry is TP-sharded on heads and SP-replicated, so one SP row holds the whole thing and the
    other seven are duplicates. Take row 0 and concatenate the TP shards on the head axis.
    """
    shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(state)]
    rows, columns = tuple(mesh_device.shape)
    tp_size = (rows, columns)[TP_AXIS]
    head_shards = []
    for tp_rank in range(tp_size):
        row, column = (0, tp_rank) if SP_AXIS == 0 else (tp_rank, 0)
        head_shards.append(
            shards[row * columns + column].reshape(-1, KimiK3Config.KDA_HEAD_DIM, KimiK3Config.KDA_HEAD_DIM)
        )
    return torch.cat(head_shards, dim=0).float()


def _dram_bytes(mesh_device):
    view = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM)
    return view.total_bytes_allocated_per_bank * view.num_banks


@pytest.mark.parametrize("mesh_device, device_params", PLACEMENTS, indirect=True)
@pytest.mark.parametrize("num_layers", DEPTHS, ids=[f"L{n}" for n in DEPTHS])
def test_chunked_prefill_carries_kda_state(mesh_device, device_params, num_layers):
    checkpoint = resolve_checkpoint()
    trace = resolve_trace(TRACE_1M if num_layers >= DEEP_TRACE_FROM else TRACE_100K)
    if checkpoint is None or trace is None:
        pytest.skip("needs KIMI_K3_HF_MODEL and the 100k golden trace")

    checkpoint = Path(checkpoint)
    root = resolve_model_root(checkpoint)
    config = kimi_k3_hf_config(max_seq=TOTAL_LEN)
    cache = cache_root(checkpoint, tuple(mesh_device.shape), TP_AXIS)

    attn_res = TtAttnRes(
        mesh_device,
        hidden_size=KimiK3Config.EMB_SIZE,
        eps=KimiK3Config.RMS_NORM_EPS,
        tp_axis=TP_AXIS,
        weights=load_attn_res_weights(
            mesh_device,
            load_attn_res_state_dict(checkpoint, num_layers, root),
            None,
            num_layers=num_layers,
            tensor_parallel_axis=TP_AXIS,
            prefix=root,
        ),
    )

    def residual_factory(hidden, block_residual=None):
        # Single-rank test: nothing is inherited, so the second argument is always None.
        # A fresh walk per chunk. AttnRes state is per token — every reduction is over the hidden
        # dimension — so there is nothing to carry, and `finish()` frees the stream each time.
        return TtAttnResResidual(
            TtAttnResWalk(
                attn_res,
                hidden,
                list(attn_res.weights.pre),
                list(attn_res.weights.post),
                attn_res.weights.output,
                num_layers,
            )
        )

    model = TtKimiK3Transformer(
        mesh_device,
        config,
        KimiK3Config,
        _model_state_dict(checkpoint, num_layers, root, cache),
        num_layers=num_layers,
        seq_len=CHUNK,
        residual_factory=residual_factory,
        sp_axis=SP_AXIS,
        tp_axis=TP_AXIS,
        max_seq_len=TOTAL_LEN,
        is_chunked=True,
        weight_cache_path=cache,
    )

    # One slot per MLA layer, spanning the whole sequence: chunk N+1's attention reads the KV that
    # chunk N wrote, so a cache sized to a single chunk would silently drop everything before it.
    kvpe = None
    if model.schedule.num_mla_layers:
        kvpe = allocate_mla_kvpe_cache(
            mesh_device=mesh_device,
            hf_config=config,
            max_seq_len=TOTAL_LEN,
            mesh_shape=tuple(mesh_device.shape),
            sp_axis=SP_AXIS,
            num_layers=model.schedule.num_mla_layers,
            num_users=1,
        )

    # Only the 100k trace snapshots the carry; at depth 24 the oracle is the per-layer output and the
    # cumulative KV instead, and the carry is covered by the depth-5 case.
    golden_carry = (
        trace.rows("kda", "kda_recurrent_state_layer_0") if trace.has("kda", "kda_recurrent_state_layer_0") else None
    )
    # `KdaState` has two halves and the recurrent one is only the larger. The convolution carry is
    # the causal depthwise conv's history — kernel 4, so 3 tokens by 3*q_dim channels — and it has to
    # cross the boundary too, or the FIRST THREE TOKENS of every chunk convolve against zeros
    # instead of the tail of the previous chunk. That is a small, localized error that per-chunk PCC
    # over 5120 tokens would comfortably hide, which is exactly why it needs its own oracle.
    golden_conv = trace.rows("kda", "kda_conv_state_layer_0") if trace.has("kda", "kda_conv_state_layer_0") else None
    output_pcc_bar = DEEP_OUTPUT_PCC if num_layers > KimiK3Config.ATTN_RES_BLOCK_SIZE else SHALLOW_OUTPUT_PCC
    footprints = []
    failures = []
    per_chunk_layer_pccs = {}

    for chunk in range(NUM_CHUNKS):
        start = chunk * CHUNK
        if chunk == 0:
            # Only at the head of a request: a carry summarizes the prefix behind it, so zeroing it
            # between chunks is precisely the bug this test exists to catch.
            model.reset_streams()

        tokens_tt = prepare_prefill_input_tensor(
            trace.token_ids(CHUNK, start)[0].tolist(),
            mesh_device,
            tuple(mesh_device.shape)[SP_AXIS],
            False,
            tuple(mesh_device.shape),
            SP_AXIS,
        )
        # The comparison is against the LIVE running sum, which is what `decoder_output_layer_i`
        # records — not `forward`'s return, which has passed through the final norm. `layer_tap` is
        # the same seam the depth ladder uses.
        captured = {}
        out = model.forward(
            tokens_tt,
            kvpe_cache=kvpe,
            actual_start=start,
            layer_tap=lambda idx, h: captured.__setitem__(idx, _compose(mesh_device, h)),
        )
        if out is not None:
            ttnn.deallocate(out)

        layer_pccs = {
            idx: float(
                str(
                    comp_pcc(trace.decoder_output(idx, start, start + CHUNK), captured[idx], output_pcc_bar)[1]
                ).split()[-1]
            )
            for idx in range(num_layers)
        }
        output_pcc = min(layer_pccs.values())
        per_chunk_layer_pccs[chunk] = layer_pccs

        # Snapshots land every 640 tokens, so the boundary after chunk k is row 8(k+1) - 1. The
        # golden's [heads, v_dim, k_dim] needs transposing into the layer's [heads, k_dim, v_dim].
        conv_pcc = float("nan")
        if golden_conv is not None:
            # The conv carry is TP-sharded and SP-replicated like the recurrent half, but its
            # channel axis is GROUPED, not contiguous: `group_output_shards` fuses the q, k and v
            # taps so each chip holds [q_slice | k_slice | v_slice] of width 3 * (q_dim / tp). The
            # golden is laid out [all q | all k | all v]. Concatenating the shards naively therefore
            # produces q0k0v0q1k1v1... against q0q1q2q3k0k1..., which is a permutation of the right
            # values and reads as PCC ~0.05-0.10 — bouncing, not zero, which is how a layout error
            # tells itself apart from lost state.
            shards = [ttnn.to_torch(sh) for sh in ttnn.get_device_tensors(model.kda_states.read(0, 0).convolution)]
            columns = tuple(mesh_device.shape)[1]
            tp_size = tuple(mesh_device.shape)[TP_AXIS]
            history = KimiK3Config.KDA_SHORT_CONV_KERNEL_SIZE - 1
            per_chip = KimiK3Config.KDA_NUM_HEADS * KimiK3Config.KDA_HEAD_DIM // tp_size
            tp_shards = []
            for t in range(tp_size):
                row, column = (0, t) if SP_AXIS == 0 else (t, 0)
                tp_shards.append(shards[row * columns + column].reshape(history, -1))
            got_conv = torch.cat(
                [
                    torch.cat([sh[:, stream * per_chip : (stream + 1) * per_chip] for sh in tp_shards], dim=-1)
                    for stream in range(3)  # q, k, v
                ],
                dim=-1,
            ).float()
            want_conv = golden_conv[(start + CHUNK) // SNAPSHOT_STRIDE - 1].reshape(history, -1)
            conv_pcc = float(str(comp_pcc(want_conv, got_conv, CARRY_PCC)[1]).split()[-1])

        carry_pcc = float("nan")
        if golden_carry is not None:
            row = (start + CHUNK) // SNAPSHOT_STRIDE - 1
            want_carry = golden_carry[row].transpose(-1, -2)
            got_carry = _compose_carry(mesh_device, model.kda_states.read(0, 0).recurrent)
            carry_pcc = float(str(comp_pcc(want_carry, got_carry, CARRY_PCC)[1]).split()[-1])

        footprints.append(_dram_bytes(mesh_device))
        logger.info(
            f"  chunk {chunk:2d} [{start:6d}:{start + CHUNK:6d}]  worst-layer {output_pcc:.6f} "
            f"(L{min(layer_pccs, key=layer_pccs.get)})  "
            f"carry {carry_pcc:.6f}  conv {conv_pcc:.6f}  dram {footprints[-1] / 2**20:8.1f} MiB"
        )
        if output_pcc < output_pcc_bar:
            failures.append(f"chunk {chunk} worst layer {min(layer_pccs, key=layer_pccs.get)} {output_pcc}")
        if golden_carry is not None and carry_pcc < CARRY_PCC:
            failures.append(f"chunk {chunk} carry {carry_pcc}")
        if golden_conv is not None and conv_pcc < CARRY_PCC:
            failures.append(f"chunk {chunk} conv carry {conv_pcc}")

    # The per-layer curve at the first, middle and last chunk. A worst-layer summary hides where the
    # error sits, and quoting only the final chunk hides whether degradation is progressive or
    # saturates after the first few chunks — which is the difference between something that will
    # keep getting worse at 1M tokens and something that will not.
    sampled = (0, NUM_CHUNKS // 2, NUM_CHUNKS - 1)
    logger.info(
        "  per-layer PCC by context length: " + "  ".join(f"chunk {c} = {(c + 1) * CHUNK} tok" for c in sampled)
    )
    for idx in sorted(per_chunk_layer_pccs[0]):
        seal = "  <- seal" if idx % KimiK3Config.ATTN_RES_BLOCK_SIZE == 0 else ""
        cells = "  ".join(f"{per_chunk_layer_pccs[c][idx]:.6f}" for c in sampled)
        logger.info(f"    layer {idx:2d}: {cells}{seal}")

    # The KV slabs, after every chunk has been written. This is where a stale or misindexed slab
    # shows: within one chunk MLA reads back what it just wrote and can be self-consistent while
    # wrong, but across 11 chunks the cache is the only record of the prefix and chunk 10's
    # attention depends on all of it. The device cache is indexed by rank-local MLA slot and the
    # golden by model layer — the schedule owns that mapping.
    if kvpe is not None:
        cache = gather_cache_tp0(kvpe.storage, mesh_device)
        positions = blockcyclic_positions(tuple(mesh_device.shape)[SP_AXIS], CHUNK, TOTAL_LEN)
        for slot, model_layer in enumerate(model.schedule.mla_layer_ids[: model.schedule.num_mla_layers]):
            if not trace.has_kv_cache(model_layer):
                continue
            device_rows = unrotate_cache_layer(cache[slot], positions, TOTAL_LEN)
            golden_rows = trace.kv_cache(model_layer, 0, TOTAL_LEN)
            # NoPE: the second half carries no rotation to re-base.
            pcc_nope, pcc_pe = cache_half_pccs(golden_rows, device_rows, KimiK3Config.KV_LORA_RANK, pe_interleave=False)
            logger.info(
                f"  KV slot {slot} (model layer {model_layer}) over {TOTAL_LEN} tokens: "
                f"lora={pcc_nope:.6f} rope={pcc_pe:.6f}"
            )
            if min(pcc_nope, pcc_pe) < KV_CACHE_PCC:
                failures.append(f"KV slot {slot} (layer {model_layer}) {min(pcc_nope, pcc_pe):.6f}")

    # The carries and the walk's per-block batches are the two new allocation surfaces; both are
    # supposed to be steady-state after the first chunk warms the pools.
    steady = footprints[1:]
    growth = max(steady) - min(steady)
    logger.info(
        f"  DRAM after chunk 1: {min(steady) / 2**20:.1f} MiB, "
        f"drift over {NUM_CHUNKS - 1} chunks: {growth / 2**20:.1f} MiB"
    )
    assert growth == 0, f"device DRAM grew {growth} bytes across chunks 1..{NUM_CHUNKS - 1}: {footprints}"
    assert not failures, "chunked prefill diverged from the model: " + "; ".join(failures)
