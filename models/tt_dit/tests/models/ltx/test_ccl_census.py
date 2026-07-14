# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Price each collective the LTX AV DiT block issues, in the TRACED regime.

Every op is priced by its SLOPE: capture a trace holding K copies of it, replay, and take
(T(K_hi) − T(K_lo)) / (K_hi − K_lo). That is the marginal cost of adding one such op to a traced
step — the number that ranks a cut — and it cancels the replay's fixed overhead. It is valid
because ops on one subdevice are serialized, so their costs add. Slope-pricing also sidesteps the
device profiler, whose program-count cap and marker buffer both overflow on a step this large.

Running each shape at both S1 and S2 (4x the video tokens, identical audio) separates a
collective's FIXED cost from its bandwidth cost — the two want opposite optimizations.

Shapes are production 1080p-high on BH Galaxy 4x8, SP=8 (axis 1) × TP=4 (axis 0): video_N=9728 (S1)
/ 38912 (S2) → 1216 / 4864 rows per device; audio_N=256 → 32 rows (ONE tile) per device; prompt
L=32. Audio is identical in both stages — both use F=19 latent frames — so every audio op is
work-independent.

⚠ THE SLOPE FLATTERS A MULTI-OP BODY AGAINST A SINGLE-OP ONE. K back-to-back copies of a body let
consecutive iterations pipeline: copy i's matmul overlaps copy i+1's all-gather, because a gather
(fabric) and a matmul (compute) occupy different engines. A body that is ONE fused op cannot
overlap with itself that way. So a 2-op body's slope understates what it costs on a real dependency
chain, where its gather has nothing to hide behind. The bias is visible in the rows themselves —
every (AG + matmul) body below prices ~2-24 us CHEAPER than the same AG and matmul priced
separately and added. It is only a lower bound on the real bias: de-fusing the video ff1 priced
33 us/site CHEAPER here and came back +143 us/site WORSE in the traced pipeline (see CUT 1c).
Compare a fused op to a de-fused one ONLY end-to-end; use these slopes to RANK, not to sign a cut.
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.tt_dit.layers.linear import ColParallelLinear, Linear, RowParallelLinear
from models.tt_dit.parallel.config import DiTParallelConfig, ParallelFactor
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.matmul import get_matmul_config
from models.tt_dit.utils.test import ring_params

VIDEO_DIM = 4096
AUDIO_DIM = 2048
NUM_HEADS = 32

# Per-device row counts (SP=8): video S1/S2 and the padded audio sequence.
V_ROWS_S1 = 9728 // 8  # 1216
V_ROWS_S2 = 38912 // 8  # 4864
A_ROWS = 256 // 8  # 32 — one tile
PROMPT_ROWS = 32

# Trace-replay slope: two op counts, one trace each. K_LO cancels the replay's fixed overhead.
K_LO = 16
K_HI = 64
REPLAYS = 20


def _bf16(t: torch.Tensor, device: ttnn.MeshDevice) -> ttnn.Tensor:
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)


def _fp32(t: torch.Tensor, device: ttnn.MeshDevice) -> ttnn.Tensor:
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=device)


def _free(out, dealloc: bool) -> None:
    """Free a freshly allocated op output, so K copies of an op reuse one address.

    A CCL with a persistent output buffer returns the CCLManager's CACHED buffer, not a new
    allocation: deallocating it destroys the ping-pong entry and every later call on that shape
    fails ``Input Tensor is not allocated``. Such ops pass ``dealloc=False``; matmuls, whose
    output really is new each call, pass True."""
    if not dealloc:
        return
    for t in out if isinstance(out, (list, tuple)) else [out]:
        if isinstance(t, ttnn.Tensor):
            ttnn.deallocate(t)


def _time_trace(mesh_device: ttnn.MeshDevice, body, k: int, dealloc: bool) -> float:
    """Capture a trace of ``k`` calls to ``body`` and return mean replay wall time (ms)."""
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    try:
        for _ in range(k):
            # Free each output inside the capture so K copies reuse one address instead of
            # holding K live buffers (a chunked matmul returns a list).
            _free(body(), dealloc)
    finally:
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

    ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)  # warm the replay path
    ttnn.synchronize_device(mesh_device)

    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    elapsed_ms = (time.perf_counter() - t0) * 1000 / REPLAYS

    ttnn.release_trace(mesh_device, trace_id)
    return elapsed_ms


def _price(mesh_device: ttnn.MeshDevice, name: str, body, dealloc: bool, results: dict) -> None:
    """Eager warmup (compile), then slope-price the op and log it immediately."""
    t_start = time.perf_counter()
    _free(body(), dealloc)
    ttnn.synchronize_device(mesh_device)

    t_lo = _time_trace(mesh_device, body, K_LO, dealloc)
    t_hi = _time_trace(mesh_device, body, K_HI, dealloc)
    per_op_us = (t_hi - t_lo) * 1000 / (K_HI - K_LO)
    results[name] = per_op_us
    logger.info(
        f"CCLCENSUS {name:28s} per_op={per_op_us:8.2f} us "
        f"(t{K_LO}={t_lo:.3f}ms t{K_HI}={t_hi:.3f}ms compile+warm={time.perf_counter() - t_start:.1f}s)"
    )


@pytest.mark.parametrize(
    ("mesh_device", "device_params"),
    [pytest.param((4, 8), {**ring_params, "trace_region_size": 90000000}, id="ring_bh_4x8")],
    indirect=True,
)
def test_ccl_census(mesh_device: ttnn.MeshDevice) -> None:
    # LTX_CCL_NUM_LINKS sweeps the fabric links the CCLs may use (production default: 2 on BH 4x8).
    # A TP all-gather is bandwidth-bound (only ~6.6 us of its 105 us is fixed), so links are the
    # first-order knob on it.
    sp_axis, tp_axis = 1, 0
    num_links = int(os.environ.get("LTX_CCL_NUM_LINKS", "2"))
    tp = tuple(mesh_device.shape)[tp_axis]
    sp = tuple(mesh_device.shape)[sp_axis]
    assert (tp, sp) == (4, 8), f"expected 4x8 with tp on axis0, got tp={tp} sp={sp}"

    ccl = CCLManager(mesh_device, num_links=num_links, topology=ttnn.Topology.Ring)
    pc = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        sequence_parallel=ParallelFactor(factor=sp, mesh_axis=sp_axis),
        tensor_parallel=ParallelFactor(factor=tp, mesh_axis=tp_axis),
    )

    v_loc, a_loc = VIDEO_DIM // tp, AUDIO_DIM // tp  # 1024, 512
    results: dict[str, float] = {}

    def col_linear(in_f: int, out_f: int, chunks: int | None = None) -> ColParallelLinear:
        m = ColParallelLinear(
            in_f, out_f, bias=True, mesh_device=mesh_device, mesh_axis=tp_axis, ccl_manager=ccl, chunks=chunks
        )
        m.load_torch_state_dict({"weight": torch.randn(out_f, in_f) * 0.02, "bias": torch.zeros(out_f)})
        return m

    # ---- inputs (per-device shards at production shapes) ----
    stat_v1 = _fp32(torch.randn(1, 1, V_ROWS_S1, 32), mesh_device)  # RMSNorm stats, video S1
    stat_v2 = _fp32(torch.randn(1, 1, V_ROWS_S2, 32), mesh_device)  # RMSNorm stats, video S2
    stat_a = _fp32(torch.randn(1, 1, A_ROWS, 32), mesh_device)  # RMSNorm stats, audio (1 tile)
    stat_v1_x2 = _fp32(torch.randn(1, 1, V_ROWS_S1, 64), mesh_device)  # 2 norms' stats merged
    stat_v1_x4 = _fp32(torch.randn(1, 1, V_ROWS_S1, 128), mesh_device)
    stat_a_x2 = _fp32(torch.randn(1, 1, A_ROWS, 64), mesh_device)
    stat_p = _fp32(torch.randn(1, 1, PROMPT_ROWS, 32), mesh_device)  # cross-attn K norm (prompt)

    x_v1 = _bf16(torch.randn(1, 1, V_ROWS_S1, v_loc), mesh_device)  # TP-sharded video activation
    x_v1_full = _bf16(torch.randn(1, 1, V_ROWS_S1, VIDEO_DIM), mesh_device)  # already TP-gathered
    x_v1_bf8 = ttnn.from_torch(
        torch.randn(1, 1, V_ROWS_S1, v_loc),
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        device=mesh_device,
    )  # half the gather payload
    x_v2 = _bf16(torch.randn(1, 1, V_ROWS_S2, v_loc), mesh_device)  # S2: 4x the tokens
    x_v2_full = _bf16(torch.randn(1, 1, V_ROWS_S2, VIDEO_DIM), mesh_device)
    ff_v1 = _bf16(torch.randn(1, 1, V_ROWS_S1, 4 * VIDEO_DIM // tp), mesh_device)  # ff2 input (TP-sharded)
    gate_col = _bf16(torch.randn(1, 1, 1, v_loc), mesh_device)  # AdaLN gate slice
    # Row-parallel gate candidate: partials (N,32) all-gathered to (N,128), then one tiny matmul
    # both sums the 4 TP partials and selects this device's 8 heads.
    gate_partials = _bf16(torch.randn(1, 1, V_ROWS_S1, 32), mesh_device)
    gate_partials_ag = _bf16(torch.randn(1, 1, V_ROWS_S1, 32 * tp), mesh_device)
    x_a = _bf16(torch.randn(1, 1, A_ROWS, a_loc), mesh_device)  # TP-sharded audio activation
    x_a_full = _bf16(torch.randn(1, 1, A_ROWS, AUDIO_DIM), mesh_device)
    kv_a_sp = _bf16(torch.randn(1, 1, A_ROWS, a_loc), mesh_device)  # a2v audio K/V, SP-sharded
    k_bhne_a = _bf16(torch.randn(1, NUM_HEADS // tp, A_ROWS, AUDIO_DIM // NUM_HEADS), mesh_device)
    tiny_a = _bf16(torch.randn(1, 1, 32, 32), mesh_device)
    tiny_b = _bf16(torch.randn(1, 1, 32, 32), mesh_device)

    # ---- modules ----
    gate_v = col_linear(VIDEO_DIM, NUM_HEADS)  # attn.to_gate_logits (video)
    qkv_v = col_linear(VIDEO_DIM, 3 * VIDEO_DIM, chunks=3)  # attn1.to_qkv (video)
    gate_a = col_linear(AUDIO_DIM, NUM_HEADS)  # attn.to_gate_logits (audio)
    qkv_a = col_linear(AUDIO_DIM, 3 * AUDIO_DIM, chunks=3)  # audio_attn1.to_qkv
    out_v = col_linear(VIDEO_DIM, VIDEO_DIM)  # attn.to_out (video)
    ff1_v = col_linear(VIDEO_DIM, 4 * VIDEO_DIM)  # ffn.ff1 (video)
    ff2_v = RowParallelLinear(
        4 * VIDEO_DIM, VIDEO_DIM, bias=True, mesh_device=mesh_device, mesh_axis=tp_axis, ccl_manager=ccl
    )
    ff2_v.load_torch_state_dict(
        {"weight": torch.randn(VIDEO_DIM, 4 * VIDEO_DIM) * 0.01, "bias": torch.zeros(VIDEO_DIM)}
    )
    # Row-parallel gate candidate: local partial matmul (no gather), then the reduce+select matmul.
    gate_partial_mm = Linear(v_loc, NUM_HEADS, bias=True, mesh_device=mesh_device)  # K = D/tp: eats x as-is
    gate_partial_mm.load_torch_state_dict(
        {"weight": torch.randn(NUM_HEADS, v_loc) * 0.02, "bias": torch.zeros(NUM_HEADS)}
    )
    gate_reduce_select = col_linear(32 * tp, NUM_HEADS)  # (N, 32*tp) -> this device's 8 head gates

    # CUT 1b: the block's six attentions each gather their input activation TWICE (to_gate_logits
    # and to_q/to_qkv both fuse a gather of it). Below, each attention's Q-side projection pair is
    # priced BOTH ways as one traced body — old = 2 fused agmms, new = 1 gather + 2 plain matmuls —
    # so the cut's per-attention saving is measured, not summed from separately priced parts.
    q_v = col_linear(VIDEO_DIM, VIDEO_DIM)  # attn2.to_q (video cross)
    q_a2v = col_linear(VIDEO_DIM, AUDIO_DIM)  # audio_to_video_attn.to_q (video Q, audio dim)
    q_a = col_linear(AUDIO_DIM, AUDIO_DIM)  # audio_attn2.to_q and video_to_audio_attn.to_q

    def _dealloc(*outs) -> None:
        for out in outs:
            for t in out if isinstance(out, (list, tuple)) else [out]:
                ttnn.deallocate(t)

    def old_pair(gate_mm, q_mm, x):
        """Today: gate and Q each fuse their own all-gather of the same activation."""

        def body():
            _dealloc(gate_mm(x, parallel_config=pc), q_mm(x, parallel_config=pc))

        return body

    def new_pair(gate_mm, q_mm, x):
        """CUT 1b: gather the activation once, then two plain matmuls on the gathered tensor."""

        def body():
            # The gather returns CCLManager's persistent buffer — never deallocate it (see _free).
            xg = ccl.all_gather_persistent_buffer(x, dim=3, mesh_axis=tp_axis)
            _dealloc(gate_mm(xg), q_mm(xg))

        return body

    # CUT 1c: every AG-matmul W1 did NOT touch, priced BOTH ways — fused, versus a standalone
    # all-gather feeding a plain matmul.
    #
    # ⚠ THE CUT THESE ROWS ARGUE FOR IS REFUTED END-TO-END. DO NOT SIGN IT OFF THESE NUMBERS.
    # They say de-fusing audio to_out / audio ff1 / video ff1 (stage 1) wins 4-33 us a site. The
    # traced pipeline says otherwise: de-fusing them is +6.63 ms/step at S1 (332.73 -> 339.36,
    # n=15, 66 sigma), and the audio sites alone are a null (-0.22 +/- 0.35 ms/step at S2, where
    # the video ff1 shape does not fire). The rows are kept because the shapes they price are real
    # and the LOSSES they show are trustworthy — to_out and v2a's to_kv must stay fused, and that
    # is worth knowing — but the WINS are an artifact of the harness (see the module docstring).
    #
    # Each site is priced as ONE traced body per arm, so the de-fused arm at least carries its own
    # extra program launch and intermediate buffer. That was not enough to make it honest.
    out_a2v = col_linear(AUDIO_DIM, VIDEO_DIM)  # audio_to_video_attn.to_out (audio heads -> video dim)
    out_a = col_linear(AUDIO_DIM, AUDIO_DIM)  # audio_attn1/audio_attn2/v2a .to_out
    kv_a2v = col_linear(AUDIO_DIM, 2 * AUDIO_DIM, chunks=2)  # audio_to_video_attn.to_kv (audio context)
    kv_v2a = col_linear(VIDEO_DIM, 2 * AUDIO_DIM, chunks=2)  # video_to_audio_attn.to_kv (video context)
    ff1_a = col_linear(AUDIO_DIM, 4 * AUDIO_DIM)  # audio_ff.ff1

    # to_out inputs are the concatenated-heads SDPA output: TP-sharded on the head dim, so the
    # gather payload matches the attention's own dim, not the residual stream's.
    o_a2v_s1 = _bf16(torch.randn(1, 1, V_ROWS_S1, a_loc), mesh_device)  # a2v: video rows, audio dim
    o_a2v_s2 = _bf16(torch.randn(1, 1, V_ROWS_S2, a_loc), mesh_device)
    kv_a2v_in = _bf16(torch.randn(1, 1, A_ROWS * sp, a_loc), mesh_device)  # SP-gathered audio, TP-sharded
    res_v1 = _bf16(torch.randn(1, 1, V_ROWS_S1, v_loc), mesh_device)  # gated-residual operands of the
    res_v2 = _bf16(torch.randn(1, 1, V_ROWS_S2, v_loc), mesh_device)  # fused to_out epilogue
    res_a = _bf16(torch.randn(1, 1, A_ROWS, a_loc), mesh_device)
    gate_a_col = _bf16(torch.randn(1, 1, 1, a_loc), mesh_device)

    full_grid = mesh_device.compute_with_storage_grid_size()
    ag_grid = ttnn.CoreCoord(full_grid.x, full_grid.y - 1)  # the AG-mm reserves one row for CCL workers

    def fused_addcmul(lin, x, residual, gate):
        """to_out today: gather + matmul + gated residual in one op (attention_ltx.py:341)."""

        def body():
            w = lin.weight.data
            out = ttnn.experimental.all_gather_minimal_matmul_async(
                input_tensor=x,
                weight_tensor=w,
                bias_tensor=lin.bias.data,
                config=get_matmul_config(x.padded_shape[-2], w.padded_shape[-2], w.padded_shape[-1], ag_grid),
                compute_kernel_config=lin.compute_config,
                persistent_output_buffer=ccl.get_ag_ping_pong_buffer(x.shape, 3, tp_axis, dtype=x.get_dtype()),
                multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(tp_axis),
                num_links=ccl.num_links,
                topology=ccl.topology,
                cluster_axis=tp_axis,
                barrier_semaphore=None,
                force_transpose=True,
                num_workers_per_link=full_grid.x // ccl.num_links,
                num_buffers_per_channel=48 if not is_blackhole() else 24,
                scalar=1.0,
                addcmul_input_tensor1=residual,
                addcmul_input_tensor2=gate,
            )[0]
            _dealloc(out)

        return body

    def defused_addcmul(lin, x, residual, gate):
        """CUT 1c: standalone gather, then the matmul+addcmul kernel the Linear path already uses."""

        def body():
            xg = ccl.all_gather_persistent_buffer(x, dim=3, mesh_axis=tp_axis)
            w = lin.weight.data
            out = ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                xg,
                w,
                1.0,
                residual,
                gate,
                bias_tensor=lin.bias.data,
                config=get_matmul_config(xg.padded_shape[-2], xg.padded_shape[-1], w.padded_shape[-1], full_grid),
                compute_kernel_config=lin.compute_config,
            )
            _dealloc(out)

        return body

    def fused_plain(lin, x):
        def body():
            _dealloc(lin(x, parallel_config=pc))

        return body

    def defused_plain(lin, x):
        def body():
            xg = ccl.all_gather_persistent_buffer(x, dim=3, mesh_axis=tp_axis)
            _dealloc(lin(xg))

        return body

    cut1c = [
        # to_out x3 shapes (fused epilogue carries the gated residual, so the de-fused arm must too)
        ("c1c_fused_out_v_s1", fused_addcmul(out_v, x_v1, res_v1, gate_col), False),
        ("c1c_defused_out_v_s1", defused_addcmul(out_v, x_v1, res_v1, gate_col), False),
        ("c1c_fused_out_a2v_s1", fused_addcmul(out_a2v, o_a2v_s1, res_v1, gate_col), False),
        ("c1c_defused_out_a2v_s1", defused_addcmul(out_a2v, o_a2v_s1, res_v1, gate_col), False),
        ("c1c_fused_out_audio", fused_addcmul(out_a, x_a, res_a, gate_a_col), False),
        ("c1c_defused_out_audio", defused_addcmul(out_a, x_a, res_a, gate_a_col), False),
        # cross-attn K/V projections whose context is TP-sharded (a2v: audio, v2a: video)
        ("c1c_fused_kv_a2v", fused_plain(kv_a2v, kv_a2v_in), False),
        ("c1c_defused_kv_a2v", defused_plain(kv_a2v, kv_a2v_in), False),
        ("c1c_fused_kv_v2a_s1", fused_plain(kv_v2a, x_v1), False),
        ("c1c_defused_kv_v2a_s1", defused_plain(kv_v2a, x_v1), False),
        # FFN ff1 (ff2's matmul+reduce-scatter is a separate op and is left fused)
        ("c1c_fused_ff1_v_s1", fused_plain(ff1_v, x_v1), False),
        ("c1c_defused_ff1_v_s1", defused_plain(ff1_v, x_v1), False),
        ("c1c_fused_ff1_audio", fused_plain(ff1_a, x_a), False),
        ("c1c_defused_ff1_audio", defused_plain(ff1_a, x_a), False),
        # pure matmuls: the term that decides the sign (fusion pays only if this is big)
        ("c1c_mm_out_v_s1_gathered", lambda: out_v(x_v1_full), True),
        ("c1c_mm_ff1_v_s1_gathered", lambda: ff1_v(x_v1_full), True),
        ("c1c_mm_out_audio_gathered", lambda: out_a(x_a_full), True),
        ("c1c_mm_ff1_audio_gathered", lambda: ff1_a(x_a_full), True),
        # S2 (4x video tokens): the video sites' sign can flip with the matmul's size
        ("c1c_fused_out_v_s2", fused_addcmul(out_v, x_v2, res_v2, gate_col), False),
        ("c1c_defused_out_v_s2", defused_addcmul(out_v, x_v2, res_v2, gate_col), False),
        ("c1c_fused_out_a2v_s2", fused_addcmul(out_a2v, o_a2v_s2, res_v2, gate_col), False),
        ("c1c_defused_out_a2v_s2", defused_addcmul(out_a2v, o_a2v_s2, res_v2, gate_col), False),
        ("c1c_fused_kv_v2a_s2", fused_plain(kv_v2a, x_v2), False),
        ("c1c_defused_kv_v2a_s2", defused_plain(kv_v2a, x_v2), False),
        ("c1c_fused_ff1_v_s2", fused_plain(ff1_v, x_v2), False),
        ("c1c_defused_ff1_v_s2", defused_plain(ff1_v, x_v2), False),
    ]

    # ---- variants, priority-ordered (each logs as soon as it is measured) ----
    # (name, body, dealloc-output?) — see _free() for why CCL outputs must not be deallocated.
    variants = [
        # program-launch reference: the cheapest possible traced program
        ("prog_launch_ref", lambda: ttnn.multiply(tiny_a, tiny_b), True),
        # RMSNorm stat all-gathers (TP axis, dim=-1) — 20 per block
        ("stat_ag_video_s1", lambda: ccl.all_gather_persistent_buffer(stat_v1, dim=3, mesh_axis=tp_axis), False),
        ("stat_ag_audio", lambda: ccl.all_gather_persistent_buffer(stat_a, dim=3, mesh_axis=tp_axis), False),
        ("stat_ag_prompt", lambda: ccl.all_gather_persistent_buffer(stat_p, dim=3, mesh_axis=tp_axis), False),
        ("stat_ag_video_s2", lambda: ccl.all_gather_persistent_buffer(stat_v2, dim=3, mesh_axis=tp_axis), False),
        # merged stats: does one wider AG beat two narrow ones?
        ("stat_ag_video_s1_x2", lambda: ccl.all_gather_persistent_buffer(stat_v1_x2, dim=3, mesh_axis=tp_axis), False),
        ("stat_ag_video_s1_x4", lambda: ccl.all_gather_persistent_buffer(stat_v1_x4, dim=3, mesh_axis=tp_axis), False),
        ("stat_ag_audio_x2", lambda: ccl.all_gather_persistent_buffer(stat_a_x2, dim=3, mesh_axis=tp_axis), False),
        # the gate/QKV redundancy: both fuse an all-gather of the SAME activation
        ("agmm_gate_video_s1", lambda: gate_v(x_v1, parallel_config=pc), True),
        ("agmm_qkv_video_s1", lambda: qkv_v(x_v1, parallel_config=pc), True),
        ("ag_activation_video_s1", lambda: ccl.all_gather_persistent_buffer(x_v1, dim=3, mesh_axis=tp_axis), False),
        ("mm_gate_video_gathered", lambda: gate_v(x_v1_full), True),
        ("mm_qkv_video_gathered", lambda: qkv_v(x_v1_full), True),
        ("agmm_gate_audio", lambda: gate_a(x_a, parallel_config=pc), True),
        ("agmm_qkv_audio", lambda: qkv_a(x_a, parallel_config=pc), True),
        ("ag_activation_audio", lambda: ccl.all_gather_persistent_buffer(x_a, dim=3, mesh_axis=tp_axis), False),
        ("mm_gate_audio_gathered", lambda: gate_a(x_a_full), True),
        ("mm_qkv_audio_gathered", lambda: qkv_a(x_a_full), True),
        # SP-axis collectives on the audio path
        ("sp_ag_audio_kv", lambda: ccl.all_gather_persistent_buffer(kv_a_sp, dim=2, mesh_axis=sp_axis), False),
        ("sp_ag_audio_k_bhne", lambda: ccl.all_gather_persistent_buffer(k_bhne_a, dim=2, mesh_axis=sp_axis), False),
        # rest of the block's TP collectives, so the census table has a price for every row
        ("agmm_out_video_s1", lambda: out_v(x_v1, parallel_config=pc), True),
        ("agmm_ff1_video_s1", lambda: ff1_v(x_v1, parallel_config=pc), True),
        ("rs_ff2_video_s1", lambda: ff2_v.forward_fused_addcmul(ff_v1, x_v1, gate_col), True),
        # CUT #1 candidate: row-parallel gate — no activation gather, only an (N,32) partial gather
        ("cut_gate_partial_mm", lambda: gate_partial_mm(x_v1), True),
        (
            "cut_gate_ag_partials",
            lambda: ccl.all_gather_persistent_buffer(gate_partials, dim=3, mesh_axis=tp_axis),
            False,
        ),
        ("cut_gate_reduce_select", lambda: gate_reduce_select(gate_partials_ag), True),
        (
            "ag_activation_video_s1_bf8",
            lambda: ccl.all_gather_persistent_buffer(x_v1_bf8, dim=3, mesh_axis=tp_axis),
            False,
        ),
        # CUT 1b, priced per attention: the block runs attn1_v + attn2_v + a2v (video Q) and
        # audio_attn1 + audio_attn2 + v2a (audio Q); v2a shares audio_attn2's shape.
        ("cut1b_old_attn1_v_s1", old_pair(gate_v, qkv_v, x_v1), False),
        ("cut1b_new_attn1_v_s1", new_pair(gate_v, qkv_v, x_v1), False),
        ("cut1b_old_attn2_v_s1", old_pair(gate_v, q_v, x_v1), False),
        ("cut1b_new_attn2_v_s1", new_pair(gate_v, q_v, x_v1), False),
        ("cut1b_old_a2v_v_s1", old_pair(gate_v, q_a2v, x_v1), False),
        ("cut1b_new_a2v_v_s1", new_pair(gate_v, q_a2v, x_v1), False),
        ("cut1b_old_attn1_a", old_pair(gate_a, qkv_a, x_a), False),
        ("cut1b_new_attn1_a", new_pair(gate_a, qkv_a, x_a), False),
        ("cut1b_old_attn2_a", old_pair(gate_a, q_a, x_a), False),
        ("cut1b_new_attn2_a", new_pair(gate_a, q_a, x_a), False),
        # S2 (4x tokens): separates a collective's fixed cost from its bandwidth cost
        ("ag_activation_video_s2", lambda: ccl.all_gather_persistent_buffer(x_v2, dim=3, mesh_axis=tp_axis), False),
        ("agmm_gate_video_s2", lambda: gate_v(x_v2, parallel_config=pc), True),
        ("agmm_qkv_video_s2", lambda: qkv_v(x_v2, parallel_config=pc), True),
        ("mm_gate_video_s2_gathered", lambda: gate_v(x_v2_full), True),
        ("mm_qkv_video_s2_gathered", lambda: qkv_v(x_v2_full), True),
    ]

    variants += cut1c

    only = {v.strip() for v in os.environ.get("LTX_CCL_VARIANTS", "").split(",") if v.strip()}
    for name, body, dealloc in variants:
        if only and name not in only:
            continue
        _price(mesh_device, name, body, dealloc, results)

    logger.info("CCLCENSUS SUMMARY (us/op, traced steady state):")
    for name, us in results.items():
        logger.info(f"CCLCENSUS_ROW {name},{us:.2f}")
