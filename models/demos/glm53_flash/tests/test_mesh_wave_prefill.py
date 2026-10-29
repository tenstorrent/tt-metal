# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-row prefill ownership (waves) on a multi-row mesh (``doc/mesh/row_ownership.md`` mechanism (a), phase 1:
one slot index per wave): ``GLM53MeshModel.prefill_wave`` against the sequence-parallel ``prefill`` on the KDA +
dense layers (0-2 by default) or, with GLM53_MODEL_LAYERS=0-3 and the partitioned heads2users cache (GLM53_PARTITION_USERS=1 GLM53_PREFILL_PARTITION=heads2users GLM53_WAVE_B=4), the DSA + MoE layer too, real weights, two short prompts:

(i)   the same prompt through SP and through a wave on row 0 (row 1 idle) and on row 1: the first token equal, the
      last-token logits within bf16 noise (the local and the sequence-parallel scans group the recurrence
      differently);
(ii)  two different prompts in one wave vs each alone in a wave: bit-identical per row (the wave must not mix);
(iii) an idle row's slot is untouched by a wave (its KDA carry before == after, bit for bit);
(iv)  per-row starts: both rows continue their prompts at 128 (wave chunks start at multiples of 128), then a wave with row 0 at 128 and row 1 restarting at 0
      with row 0 idle; against the unchunked wave and the SP chunked prefill; a non-contiguous start raises.

    GLM53_MESH_SHAPE=2,4 GLM53_MODEL_LAYERS=0-2 pytest models/demos/glm53_flash/tests/test_mesh_wave_prefill.py -s
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mesh import mesh_device_params
from models.demos.glm53_flash.tt.mesh_model import GLM53MeshModel
from models.demos.glm53_flash.tt.vllm_config import WaveCall, WaveMember

os.environ.setdefault("GLM53_KDA_FUSED", "1")  # the block KDA decode: tiled conv history (the wave carry blend)
MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "2,4").split(","))
_cache_env = os.environ.get("GLM53_MESH_CACHE", "/home/ttuser/ashai/glm53/_cache/mesh")
CACHE_ROOT = None if _cache_env.lower() == "none" else Path(_cache_env)
PCC_BAR = 0.999  # the same path, chunked vs unchunked: bf16 noise only
PCC_BAR_SP = 0.99  # vs the sequence-parallel path: its KDA scan groups the recurrence differently (ulp-level residual
# differences) and the MoE router flips near-tie expert choices on a few tokens; the accuracy gate of a wave is the
# 45-layer first token vs HF (the e2e harness), not bit agreement with SP
NEAR_TIE = 0.05  # a differing greedy token is accepted when the reference's top-2 logit gap is below this
B = int(
    os.environ.get("GLM53_WAVE_B", "2")
)  # user slots per row (a multiple of the columns with the partitioned cache)


def _layers() -> list[int]:
    raw = os.environ.get("GLM53_MODEL_LAYERS", "0-2")
    out: list[int] = []
    for part in raw.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def _wave(model, members: dict[int, tuple[torch.Tensor, int]], user: int) -> WaveCall:
    """``members``: row -> (chunk ids, absolute start); the wave's ``t_pad`` is the longest chunk rounded to 32."""
    t_pad = max(-(-int(ids.numel()) // 32) * 32 for ids, _ in members.values())
    ms = tuple(
        WaveMember(logical=row, row=row, user=user, start=start, end=start + int(ids.numel()))
        for row, (ids, start) in sorted(members.items())
    )
    call = WaveCall(rows=model.mesh.rows, t_pad=t_pad, members=ms, user=user)
    for row, (ids, _) in members.items():
        call.ids[row] = ids.clone()
    return call


def _sp(model, ids: torch.Tensor, user: int, start: int = 0) -> tuple[int, torch.Tensor]:
    streams, logits, owner = model.prefill(ids, user=user, start=start)
    first = model.first_token(logits, owner)
    host = model.mesh.to_host(logits, rows=0).float().reshape(model.mesh.rows, -1)[owner]
    ttnn.deallocate(streams)
    ttnn.deallocate(logits)
    return first, host


def _kda_slot_state(model, row: int, user: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Every KDA layer's (recurrent [H,K,V], conv tiles [taps,W]) of slot ``user`` on row ``row``'s first chip."""
    out = []
    for dec in model.layers:
        if not dec.is_kda:
            continue
        chip = row * model.mesh.cols
        rec = ttnn.to_torch(model.mesh.device_tensors(dec.kda_batched.recurrent)[chip])[user].clone()
        conv = ttnn.to_torch(model.mesh.device_tensors(dec.kda_batched.convolution)[chip])[:, user].clone()
        out.append((rec, conv))
    return out


def _same_state(a, b) -> bool:
    return all(torch.equal(ra, rb) and torch.equal(ca, cb) for (ra, ca), (rb, cb) in zip(a, b))


def _same_first(got: int, ref: int, ref_logits: torch.Tensor) -> bool:
    """The greedy tokens agree, or the reference had no clear winner (a near-tie: 1-layer models, MoE flips)."""
    if got == ref:
        return True
    top2 = torch.topk(ref_logits, 2).values
    return float(top2[0] - top2[1]) < NEAR_TIE


def _timed(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    logger.info(f"  {label}: {time.perf_counter() - t0:.2f} s")
    return out


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_wave_prefill_vs_sequence_parallel(mesh_device, expect_error):
    from transformers import AutoTokenizer

    snapshot = default_snapshot_dir()
    tok = AutoTokenizer.from_pretrained(str(snapshot))
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    layers = _layers()
    ids_a = torch.tensor(
        tok.encode(
            "The lighthouse keeper collected stamps from every ship that passed, and on winter nights he sorted them by the colour of their seas. "
            * 8
        ),
        dtype=torch.long,
    )
    ids_b = torch.tensor(tok.encode("量子计算机使用量子比特，它可以同时处于多个状态；这使得某些问题的求解速度远远超过经典计算机。" * 10), dtype=torch.long)
    n_a, n_b = int(ids_a.numel()), int(ids_b.numel())
    assert n_a > 96 and n_b > 128 and n_a != n_b, (n_a, n_b)
    model = GLM53MeshModel(mesh_device, cfg, layers, max_batch=B, max_seq_len=2048, weight_cache_path=CACHE_ROOT)
    rows = model.mesh.rows
    assert rows > 1, "waves are the multi-row case"
    logger.info(f"wave test: {len(layers)} layers {layers}, prompts {n_a} / {n_b} tokens, mesh {MESH}")

    # (i) SP vs a wave on row 0, then on row 1; (iii) the idle row's slot untouched
    first_sp, logits_sp = _timed("SP prefill a (slot 0)", lambda: _sp(model, ids_a, user=0))
    row1_before = _kda_slot_state(model, 1, 0)
    res0 = _timed("wave a on row 0 (row 1 idle)", lambda: model.prefill_wave(_wave(model, {0: (ids_a, 0)}, user=0)))
    assert set(res0) == {0}
    p0 = pcc(logits_sp, res0[0][1])
    logger.info(f"(i) row 0: wave first token {res0[0][0]} vs SP {first_sp}; logits pcc {p0:.6f}")
    assert _same_first(res0[0][0], first_sp, logits_sp) and p0 > PCC_BAR_SP
    assert _same_state(row1_before, _kda_slot_state(model, 1, 0)), "(iii) the idle row's slot changed"
    res1 = _timed("wave a on row 1 (row 0 idle)", lambda: model.prefill_wave(_wave(model, {1: (ids_a, 0)}, user=0)))
    p1 = pcc(logits_sp, res1[1][1])
    p01 = pcc(res0[0][1], res1[1][1])
    logger.info(
        f"(i) row 1: first token {res1[1][0]} vs SP {first_sp}; pcc vs SP {p1:.6f}, vs row 0's wave {p01:.6f} (equal: {torch.equal(res0[0][1], res1[1][1])})"
    )
    assert _same_first(res1[1][0], first_sp, logits_sp) and p1 > PCC_BAR_SP and p01 > 0.9999

    # (ii) two prompts in one wave vs each alone (slot 1)
    res_ab = _timed(
        "wave a | b (slot 1)", lambda: model.prefill_wave(_wave(model, {0: (ids_a, 0), 1: (ids_b, 0)}, user=1))
    )
    res_a = model.prefill_wave(_wave(model, {0: (ids_a, 0)}, user=1))
    res_b = model.prefill_wave(_wave(model, {1: (ids_b, 0)}, user=1))
    first_sp_b, logits_sp_b = _sp(model, ids_b, user=1)
    pb = pcc(logits_sp_b, res_ab[1][1])
    logger.info(
        f"(ii) a|b vs a alone: equal {torch.equal(res_ab[0][1], res_a[0][1])}; b alone: equal "
        f"{torch.equal(res_ab[1][1], res_b[1][1])}; b first {res_ab[1][0]} vs SP {first_sp_b}, pcc {pb:.6f}"
    )
    assert (
        res_ab[0][0] == res_a[0][0]
        and _same_first(res_ab[0][0], first_sp, logits_sp)
        and torch.equal(res_ab[0][1], res_a[0][1])
    )
    assert (
        res_ab[1][0] == res_b[1][0]
        and _same_first(res_ab[1][0], first_sp_b, logits_sp_b)
        and torch.equal(res_ab[1][1], res_b[1][1])
        and pb > PCC_BAR_SP
    )

    # (iv) per-row starts: chunked continuation on each row, against the unchunked wave and the SP chunked prefill
    model.prefill_wave(_wave(model, {0: (ids_a[:128], 0), 1: (ids_b[:128], 0)}, user=0))
    res2 = _timed(
        "wave a[128:] on row 0 (start 128), row 1 idle",
        lambda: model.prefill_wave(_wave(model, {0: (ids_a[128:], 128)}, user=0)),
    )
    res3 = model.prefill_wave(_wave(model, {1: (ids_b[128:], 128)}, user=0))
    p2 = pcc(res0[0][1], res2[0][1])
    p3 = pcc(res_ab[1][1], res3[1][1])
    logger.info(
        f"(iv) row 0 continued at 128: first {res2[0][0]} (unchunked wave {res0[0][0]}), pcc vs unchunked wave {p2:.6f}; row 1 continued at 128: first {res3[1][0]} vs {res_ab[1][0]}, pcc {p3:.6f}"
    )
    assert res2[0][0] == res0[0][0] and p2 > PCC_BAR
    assert res3[1][0] == res_ab[1][0] and p3 > PCC_BAR
    model.prefill_wave(_wave(model, {0: (ids_a[:128], 0), 1: (ids_b[:128], 0)}, user=0))
    res4 = model.prefill_wave(_wave(model, {0: (ids_a[128:], 128), 1: (ids_b, 0)}, user=0))  # mixed starts in one wave
    p4a, p4b = pcc(res0[0][1], res4[0][1]), pcc(res_ab[1][1], res4[1][1])
    logger.info(
        f"(iv) mixed starts: row 0 at 128 pcc {p4a:.6f} (first {res4[0][0]}), row 1 at 0 pcc {p4b:.6f} (first {res4[1][0]})"
    )
    assert res4[0][0] == res0[0][0] and p4a > PCC_BAR and res4[1][0] == res_ab[1][0] and p4b > PCC_BAR
    with expect_error(ValueError, "not contiguous"):
        model.prefill_wave(_wave(model, {0: (ids_a[128:], 256)}, user=0))
    logger.info("waves: (i)-(iv) PASS")

    # (v) the traced waves (every member at start 0): the captured per-row lengths that hold the prompts; a|b in slot 0
    # through the trace vs the eager wave (the trace's dense attention runs over the row's own latent, the eager wave's
    # over the slot's first P cached rows: the same keys), then a alone with row 1 idle (its slot 0 untouched)
    lengths = sorted({-(-max(n_a, n_b) // 256) * 256, 2048})
    _timed(f"capture waves {lengths}", lambda: model.capture_waves(lengths))
    eager_ab = model.prefill_wave(_wave(model, {0: (ids_a, 0), 1: (ids_b, 0)}, user=0))
    traced_ab = _timed(
        "traced wave a | b (slot 0)",
        lambda: model.prefill_traced_wave(_wave(model, {0: (ids_a, 0), 1: (ids_b, 0)}, user=0)),
    )
    pa_t, pb_t = pcc(eager_ab[0][1], traced_ab[0][1]), pcc(eager_ab[1][1], traced_ab[1][1])
    logger.info(
        f"(v) traced vs eager wave: a first {traced_ab[0][0]} vs {eager_ab[0][0]} pcc {pa_t:.6f} (equal "
        f"{torch.equal(eager_ab[0][1], traced_ab[0][1])}); b first {traced_ab[1][0]} vs {eager_ab[1][0]} pcc {pb_t:.6f}"
    )
    assert traced_ab[0][0] == eager_ab[0][0] and traced_ab[1][0] == eager_ab[1][0] and pa_t > PCC_BAR and pb_t > PCC_BAR
    row1_before = _kda_slot_state(model, 1, 0)
    traced_a = model.prefill_traced_wave(_wave(model, {0: (ids_a, 0)}, user=0))
    assert traced_a[0][0] == traced_ab[0][0] and torch.equal(
        traced_a[0][1], traced_ab[0][1]
    ), "a alone through the trace"
    assert _same_state(row1_before, _kda_slot_state(model, 1, 0)), "(v) the idle row's slot changed under the trace"
    model.release_wave_traces()
    logger.info("waves: (v) traced PASS")


DUMP = Path("bringup/references/glm53_flash/L45_S128")


def _host_first_token(cfg, layers) -> int:
    """The dump's HF first token after the whole 128-token prompt: the host head on HF's last-layer row 127."""
    from models.demos.glm53_flash.reference.checkpoint import FINAL_NORM_KEY, LM_HEAD_KEY, load_tensors, shard_headers

    snapshot = default_snapshot_dir()
    ref = torch.load(DUMP / f"layer{layers[-1]}_prefill.pt")["x_out"][0]  # [128, 4, 4096]
    w = load_tensors(snapshot, [FINAL_NORM_KEY, LM_HEAD_KEY], shard_headers(snapshot))
    h = ref.float().mean(1)[-1:]
    hn = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps) * w[FINAL_NORM_KEY].float()
    return int((hn @ w[LM_HEAD_KEY].float().t()).argmax(-1)[0])


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_wave_first_token_vs_hf(mesh_device):
    """The accuracy gate of waves (doc/mesh/row_ownership.md test 3): the dump's 128-token prompt through SP, through
    one wave with the prompt on EVERY row (identical inputs: the rows' outputs bit-identical), on row 0 with the other
    rows idle, and through the captured wave: every first token equals the host head on HF's rows (102647 on 45
    layers). Then the throughput point: one wave of ``rows`` x 2048-token prompts against the SP prefill of one
    ``rows`` x 2048-token chunk (the same rows per chip; the wave does ``rows`` prompts in that time)."""
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    layers = _layers()
    assert layers[0] == 0, "the dump's prompt embeddings start at layer 0"
    prompt = torch.load(DUMP / "prefill_input.pt")["prompt_ids"].reshape(-1)[:128].long()
    hf_first = _host_first_token(cfg, layers)
    rows = MESH[0]
    T_big = 2048
    model = GLM53MeshModel(
        mesh_device, cfg, layers, max_batch=B, max_seq_len=rows * T_big + 256, weight_cache_path=CACHE_ROOT
    )
    assert model.mesh.rows == rows > 1
    logger.info(f"wave gate: {len(layers)} layers on {MESH}, HF first token {hf_first}")
    first_sp, logits_sp = _timed("SP prefill (128 tokens)", lambda: _sp(model, prompt, user=0))
    all_rows = _timed(
        f"wave: the prompt on all {rows} rows",
        lambda: model.prefill_wave(_wave(model, {r: (prompt, 0) for r in range(rows)}, user=0)),
    )
    row0 = model.prefill_wave(_wave(model, {0: (prompt, 0)}, user=0))
    firsts = {r: all_rows[r][0] for r in range(rows)}
    same_rows = all(torch.equal(all_rows[0][1], all_rows[r][1]) for r in range(1, rows))
    p_rows = min(pcc(all_rows[0][1], all_rows[r][1]) for r in range(1, rows))  # rows differ at the ulp on the 8x4
    p_sp = pcc(logits_sp, all_rows[0][1])
    # the dump's prompt is random tokens: its 45-layer logits are depth-chaotic (the SP chain itself correlates with
    # HF's rows at ~0.947), so beyond a few layers the gate is the first token and a loose correlation
    bar = PCC_BAR_SP if len(layers) <= 8 else 0.9
    logger.info(
        f"first tokens: HF {hf_first}, SP {first_sp}, wave rows {firsts}, row 0 alone {row0[0][0]}; rows bit-identical "
        f"{same_rows} (rows' logits pcc >= {p_rows:.6f}); wave vs SP logits pcc {p_sp:.6f} (bar {bar})"
    )
    assert first_sp == hf_first, "the SP first token must match HF"
    assert all(v == hf_first for v in firsts.values()) and row0[0][0] == hf_first and p_rows > bar and p_sp > bar
    model.capture_waves([256])
    traced = _timed(
        "traced wave (256 per row)",
        lambda: model.prefill_traced_wave(_wave(model, {r: (prompt, 0) for r in range(rows)}, user=0)),
    )
    p_tr = pcc(traced[0][1], all_rows[0][1])
    logger.info(
        f"traced wave first tokens {[traced[r][0] for r in range(rows)]}, row 0 vs the eager wave pcc {p_tr:.6f} (equal {torch.equal(traced[0][1], all_rows[0][1])})"
    )
    assert all(traced[r][0] == hf_first for r in range(rows)) and p_tr > 0.9999
    model.release_wave_traces()
    # throughput: rows x T_big-token prompts in one wave vs the SP prefill of one rows x T_big chunk
    torch.manual_seed(2)
    big = [torch.randint(1000, 20000, (T_big,), dtype=torch.long) for _ in range(rows)]
    sp_chunk = torch.cat(big)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    streams, logits, owner = model.prefill(sp_chunk, user=1)
    ttnn.synchronize_device(mesh_device)
    t_sp = time.perf_counter() - t0
    ttnn.deallocate(streams)
    ttnn.deallocate(logits)
    for _ in range(2):  # the first wave at this length compiles its programs
        t0 = time.perf_counter()
        model.prefill_wave(_wave(model, {r: (big[r], 0) for r in range(rows)}, user=1))
        ttnn.synchronize_device(mesh_device)
        t_wave = time.perf_counter() - t0
    logger.info(
        f"PERF waves on {MESH}: SP prefill of one {rows * T_big}-token chunk {t_sp * 1e3:.0f} ms vs one wave of {rows} x "
        f"{T_big}-token prompts {t_wave * 1e3:.0f} ms (warm): {rows * T_big / t_sp:.0f} vs {rows * T_big / t_wave:.0f} tok/s"
    )


# ------------------------------------------------------------------------------------------ the served adapter
SERVED_W = 2048  # the wave width (a prefill bucket, >= the dense range): the continuation waves start there
SERVED_CHUNK = 1024  # the SP device chunk: 512 local rows per rank on 2 rows (ttKDA's SP scan clashes its L1 group
# summaries with the exclusive scan's CBs at 1024 local rows x 2 ranks: the 2026-10-29 block runs; the 8x4's 16384
# chunk = 2048 local rows is unaffected)
SERVED_MAX_SEQ_LEN = 8192  # the prefix-bucket starts for the ladders
SERVED_MAX_SEQ_LEN_LONG = (
    32768  # the long profile: the served needle stage's spans (prompts of 18-21 K in 2048-row waves)
)
TRACE_MB = int(os.environ.get("GLM53_MODEL_TRACE_MB", "512"))
N_DECODE = 4  # decode steps compared after every prefill


def _served_ids(n: int, seed: int, vocab: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(5, vocab - 5, (n,), generator=g).to(torch.int32)


@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "device_params", [mesh_device_params(*MESH, trace_region_size=TRACE_MB * 1_000_000)], indirect=True
)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
@pytest.mark.parametrize("profile", ["short", "long"])
def test_served_waves_vs_sequence_parallel(mesh_device, monkeypatch, profile):
    """The served adapter under per-row prefill ownership (``ServingGeometry.rows_independent``, the
    GLM53_VLLM_ROWS_INDEPENDENT flip) on the partitioned heads2users cache: the warm-up ladders (SP + waves), the wave
    traces prepared before and captured after the prefill traces, the program cache sealed (no request-time compile);
    then, each against the same prompt alone through the sequence-parallel path from scratch (a released slot index):
    (a) two prompts of one step on different rows = one traced wave, decoded jointly; (b) a prompt beyond the wave
    width beside a short one: a traced wave, then its eager continuation wave at the width (the sparse rows);
    (c) vLLM's chunked prefill of one request over two steps (a wave member starting at 1024). The first-token
    logits and every decode step >= PCC_BAR_SP with the near-tie rule (the SP scan groups the recurrence differently).
    """
    from models.demos.glm53_flash.tests.test_serving_prefix_cache_mesh import _assert_agree, _Served
    from models.demos.glm53_flash.tt.generator_vllm import GLM53FlashForCausalLM
    from models.demos.glm53_flash.tt.vllm_config import ServingGeometry

    if B % MESH[1]:
        pytest.skip(f"the user partition needs users per row {B} % cols {MESH[1]} == 0")
    long = profile == "long"
    if long and MESH[0] < 4:
        pytest.skip("the long profile (case g: four prompts advancing in pairs over 16 K spans) needs 4 mesh rows")
    max_seq_len = SERVED_MAX_SEQ_LEN_LONG if long else SERVED_MAX_SEQ_LEN
    monkeypatch.setenv("GLM53_PARTITION_USERS", "1")
    monkeypatch.setenv("GLM53_PREFILL_PARTITION", "heads2users")
    # the adapter's launcher sets the prefix bucket (max(chunk, max_seq_len / 8 to the chunk)); a direct build must
    monkeypatch.setenv(
        "GLM53_PREFIX_BUCKET", str(max(SERVED_CHUNK, -(-(max_seq_len // 8) // SERVED_CHUNK) * SERVED_CHUNK))
    )
    if long:  # the ladders' prefix-start walk is the boot cost at 32 K; the sealed cache names any program it misses
        monkeypatch.setenv("GLM53_VLLM_WARMUP_PREFIX_STARTS", "0")
    cfg = GLM53FlashConfig.from_hf_config_json(default_snapshot_dir() / "config.json")
    layers = _layers()
    model = GLM53MeshModel(mesh_device, cfg, layers, max_batch=B, max_seq_len=max_seq_len, weight_cache_path=CACHE_ROOT)
    rows = model.mesh.rows
    assert rows > 1, "waves are the multi-row case"
    g = ServingGeometry(
        rows=rows,
        cols=MESH[1],
        users_per_row=B,
        max_seq_len=max_seq_len,
        partitioned=True,
        rows_independent=True,
        device_prefill_chunk=SERVED_CHUNK,
        pad_token_id=cfg.pad_token_id,
        traced_prefill_lengths=(),  # the SP reference stays eager; the waves are traced (the model's per-row lengths)
        wave_rows=SERVED_W,
    )
    a = GLM53FlashForCausalLM(model, g, pad_token_id=cfg.pad_token_id)
    assert (
        a.geometry.traced_wave_lengths == tuple(model.wave_trace_lengths()) and a.geometry.request_capacity == rows * B
    )
    n = a.geometry.request_capacity
    t0 = time.perf_counter()
    a.warmup_model_prefill(enable_trace=False, can_sample_on_device=False)
    a.warmup_model_decode(enable_trace=False, max_batch_size=n, num_blocks=1)
    a.warmup_model_prefill(enable_trace=True, can_sample_on_device=False)
    a.warmup_model_decode(enable_trace=True, max_batch_size=n, num_blocks=1)
    logger.info(
        f"served waves: {len(layers)} layers, {rows} rows x {B} users, warm-up + captures in {time.perf_counter() - t0:.0f} s; "
        f"wave traces {sorted(model.wave_traces)}"
    )
    srv = _Served(a)
    vocab = cfg.vocab_size

    def step_prefill(reqs: list[tuple[int, torch.Tensor, int]]) -> dict[int, torch.Tensor]:
        """One vLLM prefill step: ``(logical, ids of the whole prompt so far, start)`` per request -> last logits."""
        L = max(int(ids.numel()) for _, ids, _ in reqs)
        toks = torch.zeros(len(reqs), L, dtype=torch.int32)
        for i, (_, ids, _) in enumerate(reqs):
            toks[i, : ids.numel()] = ids
        out = a.prefill_forward(
            toks,
            prompt_lens=[int(ids.numel()) for _, ids, _ in reqs],
            start_pos=torch.tensor([s for _, _, s in reqs]),
            empty_slots=[l for l, _, _ in reqs],
            enable_trace=True,
        )
        return {l: out[i, 0].clone() for i, (l, _, _) in enumerate(reqs)}

    def decode_jointly(first: dict[int, torch.Tensor], lengths: dict[int, int]) -> dict[int, list[torch.Tensor]]:
        """``N_DECODE`` greedy steps of every request at once -> per request the logits per step."""
        toks = {l: int(first[l].argmax()) for l in first}
        outs: dict[int, list[torch.Tensor]] = {l: [] for l in first}
        for k in range(N_DECODE):
            res = srv.decode({l: (toks[l], lengths[l] + k) for l in first})
            for l in first:
                outs[l].append(res[l])
                toks[l] = int(res[l].argmax())
        return outs

    def reference(ids: torch.Tensor) -> list[torch.Tensor]:
        """The prompt alone in logical 0 (SP-eligible: alone on its slot index) through the eager SP chunks."""
        logits = srv.prefill(0, ids, 0)
        _, outs = srv.generate(0, logits, int(ids.numel()), N_DECODE)
        a.release_request(0)
        return [logits] + outs

    if long:
        # (g) the served needle stage's shape at 4 layers: four prompts of 18-21 K on rows 0-3 of slot index 0 advancing
        # in PAIRS of 16 K step-chunks (vLLM's 32768 budget), so every wave pairs a member deep in its prompt with one
        # near its start (spans of 16 K), 8-10 waves per prompt, the served prefix bucket (4096 here: rows past a
        # member's end inside the bucket); each against its own SP path from scratch
        pg = {20 + i: _served_ids(n, 40 + i, vocab) for i, n in enumerate((18000, 19000, 20000, 21000))}
        K16 = 16384
        plan = [
            [(20, pg[20][:K16], 0), (21, pg[21][:K16], 0)],
            [(20, pg[20], K16), (22, pg[22][:K16], 0)],
            [(21, pg[21], K16), (23, pg[23][:K16], 0)],
            [(22, pg[22], K16), (23, pg[23], K16)],
        ]
        firsts: dict[int, torch.Tensor] = {}
        for reqs in plan:
            out = step_prefill(reqs)
            for l, ids, _ in reqs:
                if int(ids.numel()) == int(pg[l].numel()):
                    firsts[l] = out[l]
        assert set(firsts) == set(pg)
        placed = {l: (a.slots.physical(l).row, a.slots.physical(l).user) for l in pg}
        assert sorted(placed.values()) == [(r, 0) for r in range(4)], placed
        outs = decode_jointly(firsts, {l: int(ids.numel()) for l, ids in pg.items()})
        gots = {l: [firsts[l]] + outs[l] for l in pg}
        for l in pg:
            a.release_request(l)
        for l, ids in pg.items():
            ref = reference(ids)
            _assert_agree(
                f"(g) row {placed[l][0]} prompt {int(ids.numel())} (paired 16 K spans) vs SP",
                gots[l],
                ref,
                pcc_min=PCC_BAR_SP,
            )
            logger.info(
                f"(g) row {placed[l][0]}: first token {int(firsts[l].argmax())} (SP {int(ref[0].argmax())}), pcc {pcc(firsts[l], ref[0]):.5f}"
            )
        logger.info("served waves: PASS (g) four 18-21 K prompts in paired waves")
        return

    # (a) two fresh prompts in one step on slot index 0 of rows 0 and 1: one traced wave (both within 1024)
    p0, p1 = _served_ids(300, 1, vocab), _served_ids(700, 2, vocab)
    first = step_prefill([(0, p0, 0), (1, p1, 0)])
    s0, s1 = a.slots.physical(0), a.slots.physical(1)
    assert (s0.user, s1.user) == (0, 0) and s0.row != s1.row, [(s0.row, s0.user), (s1.row, s1.user)]
    outs = decode_jointly(first, {0: 300, 1: 700})
    got0, got1 = [first[0]] + outs[0], [first[1]] + outs[1]
    a.release_request(0)
    a.release_request(1)
    ref0, ref1 = reference(p0), reference(p1)
    _assert_agree("(a) wave member 300 (row 0) vs SP", got0, ref0, pcc_min=PCC_BAR_SP)
    _assert_agree("(a) wave member 700 (row 1) vs SP", got1, ref1, pcc_min=PCC_BAR_SP)
    logger.info(
        f"(a) traced wave of 300 + 700 tokens: first tokens {int(first[0].argmax())} / {int(first[1].argmax())} "
        f"(SP {int(ref0[0].argmax())} / {int(ref1[0].argmax())}), pcc {pcc(first[0], ref0[0]):.5f} / {pcc(first[1], ref1[0]):.5f}"
    )

    # (b) a prompt beyond the wave width beside a short one: wave 1 traced at 2048 (both at 0), wave 2 = the long
    # prompt's [2048, 2348) alone (eager: the sparse rows at the width, row 1 idle)
    p2, p3 = _served_ids(SERVED_W + 300, 3, vocab), _served_ids(500, 4, vocab)
    first = step_prefill([(2, p2, 0), (3, p3, 0)])
    outs = decode_jointly(first, {2: int(p2.numel()), 3: 500})
    got2, got3 = [first[2]] + outs[2], [first[3]] + outs[3]
    a.release_request(2)
    a.release_request(3)
    ref2, ref3 = reference(p2), reference(p3)
    _assert_agree("(b) long member 2348 (two waves) vs SP", got2, ref2, pcc_min=PCC_BAR_SP)
    _assert_agree("(b) short member 500 vs SP", got3, ref3, pcc_min=PCC_BAR_SP)
    logger.info(
        f"(b) 2348 + 500 tokens: first tokens {int(first[2].argmax())} / {int(first[3].argmax())}, pcc {pcc(first[2], ref2[0]):.5f} / {pcc(first[3], ref3[0]):.5f}"
    )

    # (c) vLLM's chunked prefill: one request over two steps ([0, 1024) then [1024, 1500)), the second chunk a wave
    # member starting at 1024 (eager), with another fresh prompt in the second step (a wave of two)
    p4, p5 = _served_ids(1500, 5, vocab), _served_ids(200, 6, vocab)
    step_prefill([(4, p4[:1024], 0)])
    first = step_prefill([(4, p4, 1024), (5, p5, 0)])
    outs = decode_jointly(first, {4: 1500, 5: 200})
    got4, got5 = [first[4]] + outs[4], [first[5]] + outs[5]
    a.release_request(4)
    a.release_request(5)
    ref4, ref5 = reference(p4), reference(p5)
    _assert_agree("(c) chunked request 1024 + 476 vs SP", got4, ref4, pcc_min=PCC_BAR_SP)
    _assert_agree("(c) the fresh prompt 200 beside it vs SP", got5, ref5, pcc_min=PCC_BAR_SP)
    logger.info(
        f"(c) chunked 1500 + fresh 200: first tokens {int(first[4].argmax())} / {int(first[5].argmax())}, pcc {pcc(first[4], ref4[0]):.5f} / {pcc(first[5], ref5[0]):.5f}"
    )
    logger.info("served waves: (a) traced wave, (b) traced + eager continuation wave, (c) chunked member at 1024: PASS")

    # (d) two long prompts whose waves carry DIFFERENT sparse starts per row (the 2026-10-29 served needle stage: rows
    # at 54016 / 37760 in one wave, 3 of 4 answers degenerate): A on row 0 walked in 2048-row waves from step 1, B on
    # row 1 arriving a step later, so every shared wave pairs A's start with B's start - 2048; both beyond the dense
    # range from step 3 on; then both decode jointly, each against its own SP path from scratch
    p6, p7 = _served_ids(6000, 7, vocab), _served_ids(4500, 8, vocab)
    # stale slot content first (the served slots hold earlier, longer prompts): a 6000-token prompt alone takes the
    # eager SP chunks into slot index 0 of EVERY row, released; B's waves then read rows past its own 4500 that hold
    # real latent data (user_rows_rm covers the wave's longest prefix on every row) and must never select them
    stale = _served_ids(6000, 9, vocab)
    step_prefill([(7, stale, 0)])
    assert a.slots.physical(7).user == 0 and a.slots.physical(7).sp_only  # the SP replica in slot index 0 everywhere
    a.release_request(7)
    step_prefill([(6, p6[:2048], 0)])  # A alone: a traced wave of one
    step_prefill([(6, p6[:4096], 2048), (7, p7[:2048], 0)])  # A's sparse continuation beside B's fresh chunk
    first_a = step_prefill([(6, p6, 4096), (7, p7[:4096], 2048)])  # A's tail at 4096 beside B at 2048: both sparse
    first_b = step_prefill([(7, p7, 4096)])  # B's tail alone
    outs = decode_jointly({6: first_a[6], 7: first_b[7]}, {6: 6000, 7: 4500})
    got6, got7 = [first_a[6]] + outs[6], [first_b[7]] + outs[7]
    a.release_request(6)
    a.release_request(7)
    ref6, ref7 = reference(p6), reference(p7)
    _assert_agree("(d) A (row 0, 2048 + 2048 + 1904 in waves) vs SP", got6, ref6, pcc_min=PCC_BAR_SP)
    _assert_agree("(d) B (row 1, its sparse starts paired with A's) vs SP", got7, ref7, pcc_min=PCC_BAR_SP)
    logger.info(
        f"(d) mixed sparse starts: first tokens {int(first_a[6].argmax())} / {int(first_b[7].argmax())} "
        f"(SP {int(ref6[0].argmax())} / {int(ref7[0].argmax())}), pcc {pcc(first_a[6], ref6[0]):.5f} / {pcc(first_b[7], ref7[0]):.5f}"
    )
    logger.info("served waves: PASS (a)-(d)")

    # (e) rows beyond the block's two: four long prompts on rows 0-3 of slot index 0 arriving one step apart (the
    # served needle stage: the requests on rows 2 and 3 degenerated while rows 0 and 1 were right), every wave pairing
    # four different sparse starts; each against its own SP path from scratch
    if rows >= 4:
        lens = (4000, 4500, 5000, 5500)
        ps = {10 + i: _served_ids(n, 20 + i, vocab) for i, n in enumerate(lens)}
        firsts: dict[int, torch.Tensor] = {}
        done: dict[int, int] = {}
        for step in range(8):
            reqs = []
            for i, (l, ids) in enumerate(ps.items()):
                if step < i or l in done:
                    continue
                start = max(0, (step - i)) * 2048
                end = min(int(ids.numel()), start + 2048)
                reqs.append((l, ids[:end], start))
                if end == int(ids.numel()):
                    done[l] = step
            if not reqs:
                break
            out = step_prefill(reqs)
            for l, ids, _ in reqs:
                if done.get(l) == step:
                    firsts[l] = out[l]
        assert set(firsts) == set(ps), (sorted(firsts), sorted(ps))
        placed = {l: (a.slots.physical(l).row, a.slots.physical(l).user) for l in ps}
        assert sorted(placed.values()) == [(r, 0) for r in range(4)], placed
        outs = decode_jointly(firsts, {l: int(ids.numel()) for l, ids in ps.items()})
        gots = {l: [firsts[l]] + outs[l] for l in ps}
        for l in ps:
            a.release_request(l)
        for l, ids in ps.items():
            ref = reference(ids)
            _assert_agree(
                f"(e) row {placed[l][0]} prompt {int(ids.numel())} (staggered waves) vs SP",
                gots[l],
                ref,
                pcc_min=PCC_BAR_SP,
            )
            logger.info(
                f"(e) row {placed[l][0]}: first token {int(firsts[l].argmax())} (SP {int(ref[0].argmax())}), pcc {pcc(firsts[l], ref[0]):.5f}"
            )
        logger.info("served waves: PASS (e) four staggered prompts on rows 0-3")

    # (f) a member whose first chunk is vLLM's budget remainder (224 tokens: 32-aligned, not 128-aligned), continued
    # at 224 mod 2048 beside another prompt (the served needle stage's late requests: '(2, 0, 224)'): the pool start
    # 56 sits inside a 32-pool tile
    pa, pb = _served_ids(4000, 30, vocab), _served_ids(4500, 31, vocab)
    step_prefill([(14, pa[:2048], 0), (15, pb[:224], 0)])
    first_a = step_prefill([(14, pa, 2048), (15, pb[:2272], 224)])
    step_prefill([(15, pb[:4320], 2272)])
    first_b = step_prefill([(15, pb, 4320)])
    outs = decode_jointly({14: first_a[14], 15: first_b[15]}, {14: 4000, 15: 4500})
    gots = {14: [first_a[14]] + outs[14], 15: [first_b[15]] + outs[15]}
    a.release_request(14)
    a.release_request(15)
    ref_a, ref_b = reference(pa), reference(pb)
    _assert_agree("(f) A (2048 + 1952) vs SP", gots[14], ref_a, pcc_min=PCC_BAR_SP)
    _assert_agree(
        "(f) B (224 + 2048 + 2048 + 180: continuations at 224 mod 2048) vs SP", gots[15], ref_b, pcc_min=PCC_BAR_SP
    )
    logger.info(
        f"(f) budget-remainder first chunk: first tokens {int(first_a[14].argmax())} / {int(first_b[15].argmax())} "
        f"(SP {int(ref_a[0].argmax())} / {int(ref_b[0].argmax())}), pcc {pcc(first_a[14], ref_a[0]):.5f} / {pcc(first_b[15], ref_b[0]):.5f}"
    )
    logger.info("served waves: PASS (f) continuations at 224 mod 2048")
