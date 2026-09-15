# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Config sweep for the Qwen3.5/3.6 VISION TOWER's SDPA.

The tower's SDPA is its single largest op: 18.1 ms/block on the 9B (N300, 8 heads/device) and
6.1 ms/block on the 27B (T3K, 2 heads/device), ~44% and ~15% of a block. The profile says it runs
**HiFi4 on BFP8 inputs** with **K left in BF16** while Q and V are BFP8 -- so QK^T pays bf16 rates
and double the K bandwidth. This sweeps both, plus the flash chunk sizes and exp_approx_mode.

No checkpoint and no model needed: SDPA's cost depends only on shape, dtype and config, so this
builds the per-device tensors directly. Shapes come from the demo image (11008 patches -> 12288
padded, head_dim 72 -> 96 tile-padded, 16 heads / TP).

Run::

    MESH_DEVICE=N300 pytest models/demos/blackhole/qwen36/tests/perf/test_sweep_vision_sdpa.py -v -s

`QWEN36_SDPA_HEADS` overrides the per-device head count (default: from the mesh, 16 // TP), and
`QWEN36_SDPA_ITERS` the timed iteration count.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc, run_for_wormhole_b0_or_blackhole

SEQ_LEN = 11008  # the demo grid's real patch count; the tower pads rows only to a multiple of 128,
# and 11008 = 86 x 128 already is one. It used to run at 12288 (a 2048 multiple) -- see
# `DropInVisionTransformer.forward`. The winning chunk pair depends on this: the kernel parallelises
# over heads x q_chunks across 64 cores, so what divides evenly changed when the row count did.
HEAD_DIM = 96  # 1152/16 = 72, tile-padded to 96 (padded_head_dim)
REAL_HEAD_DIM = 72  # columns that carry signal; the rest are exactly zero
N_HEADS = 16  # vision_config.num_heads, split across TP
SCALE = REAL_HEAD_DIM**-0.5

ITERS = int(os.environ.get("QWEN36_SDPA_ITERS", "5"))

# A candidate must hold this per-op PCC to be eligible. The baseline measures 0.99991 and the tower
# gate is 0.9985 end-to-end over 27 blocks, so a per-op regression compounds: LoFi is only ~3% faster
# than HiFi2 here but lands at 0.9656 per op, which no 27-block tower survives. Fastest-wins without
# this floor picks LoFi every time.
PCC_FLOOR = float(os.environ.get("QWEN36_SDPA_PCC_FLOOR", "0.999"))

# Baseline == what the tower runs today (vision_attention.forward_prefill + the model_config
# defaults): K in bf16 from kv_cache_dtype, HiFi4 from decoders_optimizations' SDPA_PREFILL, and
# get_attn_sdpa_prefill_program_config's 256/256 chunks with exact exp.
BASELINE = dict(k_bf16=True, fidelity="hifi4", q_chunk=256, k_chunk=256, exp_approx=False)

FIDELITIES = {
    # fp32_dest_acc_en stays True everywhere: the flash softmax accumulates over
    # SEQ_LEN/k_chunk chunks and loses the sum in fp16 (a known ~0.94 PCC cliff).
    "hifi4": ttnn.MathFidelity.HiFi4,
    "hifi2": ttnn.MathFidelity.HiFi2,
    "lofi": ttnn.MathFidelity.LoFi,
}


@dataclass(frozen=True)
class Cand:
    k_bf16: bool
    fidelity: str
    q_chunk: int
    k_chunk: int
    exp_approx: bool

    def label(self):
        return (
            f"K={'bf16' if self.k_bf16 else 'bf8b'} {self.fidelity:5s} "
            f"q/k={self.q_chunk}/{self.k_chunk} exp_approx={int(self.exp_approx)}"
        )


class SdpaBench:
    """Holds the device tensors and the torch reference for one (heads, seq) shape."""

    def __init__(self, mesh_device, heads):
        self.mesh_device = mesh_device
        self.heads = heads
        self.grid = mesh_device.compute_with_storage_grid_size()

        torch.manual_seed(0)

        # Zero the tile-padding columns, as nlp_create_qkv_heads does (the qkv weights are padded).
        def make():
            t = torch.randn(1, heads, SEQ_LEN, HEAD_DIM) * 0.3
            t[..., REAL_HEAD_DIM:] = 0.0
            return t

        self.q_t, self.k_t, self.v_t = make(), make(), make()

        mapper = ttnn.ReplicateTensorToMesh(mesh_device)
        to_dev = lambda t, dt: ttnn.from_torch(
            t,
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        # Every device runs the identical per-device SDPA on its own heads, so replicated tensors of
        # the local shape reproduce the in-model device op exactly.
        self.q = to_dev(self.q_t, ttnn.bfloat8_b)
        self.v = to_dev(self.v_t, ttnn.bfloat8_b)
        self.k_by_dtype = {True: to_dev(self.k_t, ttnn.bfloat16), False: to_dev(self.k_t, ttnn.bfloat8_b)}

        # Reference from the QUANTISED inputs, so the only error measured is the kernel's own.
        # Round-tripping through ttnn is the cheapest exact bf8b quantiser available.
        deq = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].float()
        self.refs = {}
        for k_bf16, k_dev in self.k_by_dtype.items():
            self.refs[k_bf16] = torch.nn.functional.scaled_dot_product_attention(
                deq(self.q), deq(k_dev), deq(self.v), attn_mask=None, is_causal=False, scale=SCALE
            )

    def run(self, cand: Cand):
        """Median device-synced latency over ITERS, plus PCC against the torch reference.

        Returns ``(inf, 0.0)`` for a config the device rejects -- big chunks raise the flash CBs
        past L1 (512/512 asks for 1,949,888 B against 1,499,136 B at this shape), which is a legal
        answer to "is this config usable", not a reason to abandon the sweep.
        """
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=FIDELITIES[cand.fidelity],
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        pcfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(self.grid.x, self.grid.y),
            exp_approx_mode=cand.exp_approx,
            q_chunk_size=cand.q_chunk,
            k_chunk_size=cand.k_chunk,
        )
        call = lambda: ttnn.transformer.scaled_dot_product_attention(
            self.q,
            self.k_by_dtype[cand.k_bf16],
            self.v,
            is_causal=False,
            scale=SCALE,
            compute_kernel_config=ckc,
            program_config=pcfg,
        )

        try:
            out = call()  # warmup: compile + cache the program
        except RuntimeError as e:
            reason = "L1 overflow" if "beyond max L1 size" in str(e) else str(e).split("\n")[0][:80]
            logger.info(f"  {cand.label()}  ->  rejected ({reason})")
            return float("inf"), 0.0
        ttnn.synchronize_device(self.mesh_device)
        got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0))[:1].float()
        ttnn.deallocate(out)
        pcc = float(comp_pcc(self.refs[cand.k_bf16], got, 0.0)[1])

        times = []
        for _ in range(ITERS):
            t0 = time.time()
            out = call()
            ttnn.synchronize_device(self.mesh_device)
            times.append((time.time() - t0) * 1e3)
            ttnn.deallocate(out)
        times.sort()
        return times[len(times) // 2], pcc


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    explicit = {"P150": (1, 1), "N150": (1, 1), "P150X4": (1, 4), "N150X4": (1, 4), "N300": (1, 2), "T3K": (1, 8)}
    return explicit.get(name, (1, max(1, min(ttnn.get_num_devices(), 2))))


MESH_SHAPE = _mesh_shape()
DEVICE_PARAMS = [
    {"l1_small_size": 24576, **({"fabric_config": ttnn.FabricConfig.FABRIC_1D} if MESH_SHAPE != (1, 1) else {})}
]


@pytest.mark.timeout(1800)
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_sweep_vision_sdpa(mesh_device, device_params):
    """Staged sweep: (K dtype x fidelity) -> chunk sizes -> exp_approx_mode."""
    del device_params
    mesh_device.enable_program_cache()

    tp = MESH_SHAPE[1]
    heads = int(os.environ.get("QWEN36_SDPA_HEADS") or N_HEADS // tp)
    logger.info(f"vision SDPA sweep: {heads} heads/device, seq {SEQ_LEN}, head_dim {HEAD_DIM}, TP={tp}")

    bench = SdpaBench(mesh_device, heads)
    base = Cand(**BASELINE)
    results = {}

    def measure(cand):
        if cand in results:
            return results[cand]
        us, pcc = bench.run(cand)
        results[cand] = (us, pcc)
        logger.info(f"  {cand.label()}  ->  {us:8.2f} ms   pcc {pcc:.6f}")
        return results[cand]

    base_ms, base_pcc = measure(base)
    logger.info(f"BASELINE {base.label()}: {base_ms:.2f} ms, pcc {base_pcc:.6f}")

    def pick(cands):
        """Fastest candidate that clears PCC_FLOOR; falls back to the baseline if none do."""
        ok = [c for c in cands if results[c][1] >= PCC_FLOOR]
        return min(ok or [base], key=lambda c: results[c][0])

    # Stage 1 -- the two suspects from the profile, crossed.
    logger.info("stage 1: K dtype x fidelity")
    stage1 = [replace(base, k_bf16=kb, fidelity=f) for kb in (True, False) for f in FIDELITIES]
    for c in stage1:
        measure(c)
    best = pick(stage1)

    # Stage 2 -- flash chunk sizes at the winning dtype/fidelity.
    logger.info(f"stage 2: chunk sizes (on {best.label()})")
    # At 11008 rows the arithmetic that matters: q_chunk sets heads x (SEQ/q) work units over 64
    # cores, and k_chunk should divide SEQ or the last flash chunk runs half empty.
    #   q=256 -> 43 chunks -> 8x43=344 units in 6 rounds of 64 (40 idle slots, 10.4% waste)
    #   q=128 -> 86 chunks -> 8x86=688 units in 11 rounds (16 idle, 2.3%)
    #   k=256 divides 11008 exactly; k=512 leaves 11008 = 21x512 + 256
    stage2 = [
        replace(best, q_chunk=q, k_chunk=k)
        for q, k in ((128, 128), (128, 256), (128, 512), (256, 256), (256, 512), (512, 512))
    ]
    for c in stage2:
        measure(c)
    best = pick([best] + stage2)

    # Stage 3 -- approximate exp. Cheap, but error accumulates over SEQ_LEN/k_chunk flash chunks.
    logger.info("stage 3: exp_approx_mode")
    alt = replace(best, exp_approx=True)
    measure(alt)
    best = pick([best, alt])

    ranked = sorted((kv for kv in results.items() if kv[1][0] != float("inf")), key=lambda kv: kv[1][0])
    rejected = [c.label() for c, (ms, _) in results.items() if ms == float("inf")]
    logger.info("=" * 96)
    logger.info(f"RANKED ({heads} heads/device, TP={tp}) -- baseline {base_ms:.2f} ms / pcc {base_pcc:.6f}")
    for cand, (ms, pcc) in ranked:
        logger.info(
            f"  {ms:8.2f} ms  {base_ms / ms:5.2f}x  pcc {pcc:.6f} {'    ' if pcc >= PCC_FLOOR else 'FAIL'}"
            f"  {cand.label()}{'   <-- baseline' if cand == base else ''}"
        )
    ms, pcc = results[best]
    logger.info("=" * 96)
    if rejected:
        logger.info(f"rejected by the device: {', '.join(rejected)}")
    logger.info(f"WINNER: {best.label()}  {base_ms:.2f} -> {ms:.2f} ms ({base_ms / ms:.2f}x), pcc {pcc:.6f}")
    logger.info(f"  per tower (27 blocks): {base_ms * 27:.0f} -> {ms * 27:.0f} ms")
    logger.info("=" * 96)

    assert ms <= base_ms, "no PCC-clearing candidate beat the baseline"
    assert pcc >= PCC_FLOOR, f"winner {best.label()} is below the PCC floor {PCC_FLOOR}"
