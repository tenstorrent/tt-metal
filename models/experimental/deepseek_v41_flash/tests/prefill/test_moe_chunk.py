# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``DeepSeekV41PrefillMoE`` over a production prefill chunk, on the model's own TP4 configuration.

The routed experts are the cheapest per token when one op call covers the whole chunk. Every call re-reads
all ``n_routed_experts`` experts' weights (each expert's tokens go to whichever call owns them, and the two
halves of a chunk select the same experts), so at 1024 tokens an op capped at 512 does the work -- and the
1.7 GB of weight traffic -- twice. 1024 token rows over 384 experts is 16 tokens per expert, which is still
a single 32-row tile row, so the second call buys nothing.

**The configuration here is the model's**, not a small stand-in: a ``1 x 4`` submesh of the Galaxy, bf4 routed
experts, ``I = 576`` per rank, and ``balance_dram_banks=True`` (the K-split gate_up shards that cover every DRAM
bank). The flag is spelled out rather than left to the constructor's default so this file keeps testing the
model's layout even if that default ever moves.

* ``test_moe_prefill_chunk_single_op_call`` -- a 1024-token chunk issues exactly one routed op call, and the
  whole block still matches the checkpoint's reference.
* ``test_moe_prefill_op_call_scales_with_chunk`` -- the number of calls is ``ceil(T / MAX_OP_TOKENS)`` for
  chunk sizes on either side of the cap.
* ``test_moe_prefill_chunk_512_and_1024_cost_the_same`` -- the performance claim, timed on the routed op: two
  512-token calls (what the old cap produced) cost clearly more than one 1024-token call, because each call
  re-reads every expert's weights. Measured 11.71 ms vs 7.88 ms at TP4, a 33% saving. The op is not perfectly
  flat across token count -- its routing prologue and its 6x-redundant x gathers grow with T -- so the test
  asserts the call-count win rather than claiming 1024 tokens are free.

The converted ttnn weights are cached under ``CACHE_DIR`` (``$DEEPSEEK_V4_CACHE_DIR``, default ``../cache``, per checkpoint).

    pytest models/experimental/deepseek_v41_flash/tests/prefill/test_moe_chunk.py
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.experimental.deepseek_v41_flash.tests.prefill.test_moe import _hidden, _snapshot
from models.experimental.deepseek_v41_flash.tests.reference import load_reference, reference_moe
from models.experimental.deepseek_v41_flash.tt.config import (
    CACHE_DIR,
    expert_provider,
    load_config,
    moe_weights,
)
from models.experimental.deepseek_v41_flash.tt.decode.moe import DeepSeekV41PreloadedExperts
from models.experimental.deepseek_v41_flash.tt.prefill.moe import DeepSeekV41PrefillMoE
from models.experimental.deepseek_v4_flash.tt.prefill.moe import MAX_OP_TOKENS
from models.experimental.deepseek_v4_flash.tt.weight_cache import WeightCache
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader

PCC = 0.95
# The production configuration (``test_full_model_prefill_demo``): TP4 on a 1x4 submesh of the Galaxy, bf4
# routed experts, gate_up shards split across every DRAM bank.
TP_SIZE = 4
PARENT_MESH = (8, 4)
EXPERT_DTYPE = ttnn.bfloat4_b
# The production prefill chunk (``DEEPSEEK_V4_PREFILL_CHUNK`` defaults to 1024).
PRODUCTION_CHUNK = 1024

pytestmark = [
    pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True),
    pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["galaxy_8x4"]),
    pytest.mark.skipif(_snapshot() is None, reason="V4.1-Flash checkpoint not found"),
    pytest.mark.timeout(7200),
]


@pytest.fixture
def tp4(mesh_device):
    """The model's ``1 x 4`` submesh, at the origin of the Galaxy (as the demo places stage 0)."""
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"need an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")
    submesh = mesh_device.create_submesh(ttnn.MeshShape(1, TP_SIZE), ttnn.MeshCoordinate(0, 0))
    assert submesh.get_num_devices() == TP_SIZE
    return submesh


def _to_device(x: torch.Tensor, device) -> ttnn.Tensor:
    """``[T, D]`` host rows -> the replicated ``[1, 1, T, D]`` bf16 TILE tensor the block takes."""
    return ttnn.from_torch(
        x.reshape(1, 1, *x.shape),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def _build(tp4, layer: int):
    """The layer's MoE block, built exactly as the prefill model builds it."""
    snapshot = _snapshot()
    config = load_config(snapshot)
    loader = DeepseekV4WeightLoader(snapshot)
    cache = WeightCache(CACHE_DIR).sub(f"layers.{layer}.ffn")
    experts = DeepSeekV41PreloadedExperts(
        config,
        expert_provider(loader, layer),
        tp4,
        dtype=EXPERT_DTYPE,
        cache=cache,
        tp_size=TP_SIZE,
        balance_dram_banks=True,
    )
    moe = DeepSeekV41PrefillMoE(config, moe_weights(loader, layer), tp4, experts=experts, cache=cache, tp_size=TP_SIZE)
    return config, loader, moe


@torch.no_grad()
@pytest.mark.parametrize("layer", (2, 39))
def test_moe_prefill_chunk_single_op_call(tp4, layer, monkeypatch):
    """A 1024-token chunk must reach the routed op as ONE call, and still be correct."""
    snapshot = _snapshot()
    config, loader, moe = _build(tp4, layer)
    reference = reference_moe(load_reference(snapshot), snapshot, loader, layer)

    calls = []
    real = ttnn.experimental.deepseek_prefill.fused_experts_prefill

    def counted(*args, **kwargs):
        calls.append(args[0].shape[2])  # the tokens this call was given
        return real(*args, **kwargs)

    hidden = _hidden(config, PRODUCTION_CHUNK)
    expected = reference(hidden.reshape(PRODUCTION_CHUNK, 1, -1)).reshape(PRODUCTION_CHUNK, -1)

    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "fused_experts_prefill", counted)
    out_tt = moe(_to_device(hidden, tp4))
    # The all-reduce replicates the residual; read one chip's copy.
    out = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(tp4, dim=0))[0]
    out = out.reshape(PRODUCTION_CHUNK, -1).float()

    logger.info(f"layer {layer}: routed op called {len(calls)} time(s) over {calls} tokens")
    assert calls == [PRODUCTION_CHUNK], (
        f"layer {layer}: a {PRODUCTION_CHUNK}-token chunk took {len(calls)} routed op call(s) over {calls} "
        f"tokens; each call re-reads every expert's weights, so the chunk must go in one "
        f"(MAX_OP_TOKENS={MAX_OP_TOKENS})"
    )

    passing, pcc = comp_pcc(expected, out, PCC)
    logger.info(f"layer {layer}: PCC {pcc}")
    assert passing, f"layer {layer}: PCC {pcc} < {PCC}"


@torch.no_grad()
def test_moe_prefill_op_call_scales_with_chunk(tp4, monkeypatch):
    """Calls per chunk are ``ceil(T / MAX_OP_TOKENS)``: one at the production chunk, more beyond the cap."""
    config, _, moe = _build(tp4, 2)

    calls = []
    real = ttnn.experimental.deepseek_prefill.fused_experts_prefill

    def counted(*args, **kwargs):
        calls.append(args[0].shape[2])
        return real(*args, **kwargs)

    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "fused_experts_prefill", counted)
    # Below the cap, at it (the production chunk: one call), and past it (the split must still work).
    for seq_len in (256, PRODUCTION_CHUNK, 2 * MAX_OP_TOKENS):
        calls.clear()
        out = moe(_to_device(_hidden(config, seq_len), tp4))
        ttnn.synchronize_device(tp4)
        ttnn.deallocate(out)
        expected = -(-seq_len // MAX_OP_TOKENS)
        logger.info(f"T={seq_len}: {len(calls)} routed op call(s) over {calls} tokens (expected {expected})")
        assert len(calls) == expected, (
            f"T={seq_len}: {len(calls)} routed op call(s) over {calls} tokens, expected {expected} "
            f"with MAX_OP_TOKENS={MAX_OP_TOKENS}"
        )


@torch.no_grad()
def test_moe_prefill_chunk_512_and_1024_cost_the_same(tp4, monkeypatch):
    """One 1024-token routed call must cost less than the two 512-token calls the old cap forced.

    The routed op is mostly, not purely, weight-bound. It reads the same ~1.7 GB of expert weights either
    way (both token counts put one tile row on every expert), but three parts of it do grow with the tokens:

      * the routing prologue, O(T) on every core (each core derives all tokens' top-k to pick its own),
      * the x rows, gathered once per (token, expert) pair -- 6x redundant, so 63 MB at T=1024 vs 31 MB,
      * the partials, one [T, H] slice per slot, likewise 63 MB vs 31 MB.

    Measured at TP4: 5.85 ms for 512 tokens, 7.88 ms for 1024 -- a fixed ~3.8 ms of weight streaming plus
    ~4.0 us/token of the above. So the honest claim is not "1024 is free" but "one call beats two": two
    512-token calls pay the weight streaming twice, and measure 11.71 ms against 7.88 ms (1.49x, a 33%
    saving). That is what the cap buys, so that is what this asserts.

    The bar is 1.25x: comfortably under the measured 1.49x, and above the 1.0x a purely token-bound op would
    give -- i.e. it fails if the weights ever stop dominating, which is the regression worth catching. The
    absolute numbers are machine-dependent, so only the ratio is asserted.
    """
    config, _, moe = _build(tp4, 2)

    routed_times = []
    real = ttnn.experimental.deepseek_prefill.fused_experts_prefill

    def timed(*args, **kwargs):
        # Drain what moe() queued before this (the router's ops) so the clock measures the routed op alone.
        ttnn.synchronize_device(tp4)
        start = time.perf_counter()
        out = real(*args, **kwargs)
        ttnn.synchronize_device(tp4)  # the op is async; drain it so this is its own time
        routed_times.append(time.perf_counter() - start)
        return out

    def routed_seconds(seq_len: int, repeats: int = 5) -> float:
        x = _to_device(_hidden(config, seq_len), tp4)
        ttnn.deallocate(moe(x))  # warm
        routed_times.clear()
        for _ in range(repeats):
            ttnn.deallocate(moe(x))
        ttnn.deallocate(x)
        return min(routed_times)

    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "fused_experts_prefill", timed)
    t512 = routed_seconds(512)
    t1024 = routed_seconds(PRODUCTION_CHUNK)

    two_call_ratio = 2 * t512 / t1024
    logger.info(f"routed op: {1e3 * t512:.2f} ms at 512 tokens, {1e3 * t1024:.2f} ms at 1024")
    logger.info(f"one 1024-token call costs {two_call_ratio:.2f}x less than two 512-token calls")

    assert two_call_ratio > 1.25, (
        f"two 512-token routed calls ({2e3 * t512:.2f} ms) should cost clearly more than one 1024-token call "
        f"({1e3 * t1024:.2f} ms), since each call re-reads every expert's weights; got {two_call_ratio:.2f}x, "
        f"which means the weights no longer dominate and the cap is not buying what it should"
    )
