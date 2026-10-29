# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end prefill of MiMo-V2.6-Flash-RL on a Blackhole Galaxy 8x4 mesh: all 48 layers, the final norm and the LM
head, against the H200 HF golden (tests/e2e/goldens.py). No device profiler, no bring-up framework. Modelled on the
Qwen3.8-27B e2e test (models/demos/qwen38_27b/tests/e2e), with its thresholds.

    scripts/run_safe_pytest.sh --run-all models/demos/mimo_v26_flash_rl/tests/e2e/test_prefill_e2e.py -k <case set>

Case sets (one test item each, one model load per item; select one with -k):
    all       isl5120, isl56320, in that order. Without -k / -m (and no node ids) it is the only item that runs: the
              other items' cases are part of it, so tests/e2e/conftest.py deselects them
    isl56320  the full 56320-token golden alone
    isl5120   quick smoke: the first 5120 tokens of the isl56320 golden
Cases: isl5120 is the first 5120 tokens of the isl56320 golden (which ran every layer as 5120-row chunks, so its first
chunk is a 5120-token prefill); isl56320 is the whole golden (wikitext, no prefix, bf16 compute on an H200). Every case
runs 5120-token chunks from position 0 on slot 0 of one state; later chunks overwrite the KV rows below the case's
length.

The model is built as the prefill runner builds it (tt/runners/runtime.py compile and prefill_chunk, kv_caches.py):
MimoModel directly over every layer, chunk 5120, capacity 56320, on the engine-owned served caches
(allocate_kv_caches, SLOTS slots), each chunk embed (at its own start) -> every layer as prefill_chunk runs it (no
acks). Unlike the runtime it loads the LM head and runs the final norm on the last chunk. Not through the bring-up
hooks. MimoModel.layer turns ttnn's fast runtime mode off around each block itself (the report records it). The mesh
opens with the bring-up's box params (spec bringup.box.device_params: FABRIC_1D_RING, default fabric payload), not
the runner's own.

Per case, four timed passes (a device sync before each start, nothing read back inside a timer), then an accuracy
pass:
    cold      embed -> 48 layers for every chunk, the final norm of the last chunk, one sync: cold_ms (includes the
              compile of every program this case is the first to run)
    warm      the same: prefill_ms, tok_s, host_enqueue_ms (the host loop returned, before the sync). After the timer:
              the last-chunk final hidden read back (determinism) and the device LM head on its last row, twice
              (compile, then lm_head_ms: head + read-back from an idle device)
    chunks    a sync after every chunk: chunk_ms[c]
    ttft      warm prefill + final norm + the device LM head on the last position + the next token on the host:
              ttft_ms. The head is model.logits' ttnn.linear (model.lm_head, model.head_cfg, fp32) on the 32-row tile
              of the TP-gathered rows that holds the last position, read back from the 4 chips of its SP row only
              (4 x [32, 38144] fp32 = 20 MB); model.logits reads back [5120, 152576] fp32 (3 GB), which no timed pass
              may do
    accuracy  every chunk: final norm -> host vs the golden final hidden, and a host LM head (the checkpoint's untied
              lm_head.weight, fp32) on sampled rows (every 16th + the last 32 of the chunk, as the ladder samples) of
              the device's and of the golden's final hidden. Last chunk: every layer's output vs the golden;
              model.logits (the device LM head) on the sampled rows vs the host head on the golden final hidden there,
              and its last row vs the golden's stored logits (isl56320), cross-checked against the TTFT head path.
              Then the state of slot 0: K / V of all 48 attention layers (9 full, 39 sliding) over [0, isl)

Asserted, at tools/prefill/bringup_fw/defaults.yaml's gate thresholds with testing/ladder.py's formulas (THRESHOLDS,
the Qwen e2e test's):
    final hidden   every chunk: PCC >= 0.97
    layers         every layer's output on the last chunk: PCC >= 0.97, rel L2 <= 0.15, as the ladder gates them; its
                   norm bias is reported only (below)
    state          key and value of every attention layer over [0, isl): PCC >= 0.97
    LM head        every case: model.logits on the last chunk's sampled rows (350) vs the host head on the golden final
                   hidden there, the ladder's top-k rule with k = 5 (ties at the golden's 5th logit count), mean over
                   the rows >= 0.9: the ladder's grade with a CPU golden; no non-finite logit. isl56320, whose golden
                   stores the logits of its last position: the device's next token there is the stored logits' top 1
                   (STORED_TOP1); top-5 on that one row is reported only
    host head      the host head (the device head's golden above) on the golden's own last row vs its stored logits:
                   PCC >= 0.999 (HOST_HEAD_SANITY)
    determinism    the warm and the accuracy pass's last-chunk final hidden are bit-identical
    completeness   each check above covered every chunk / layer / row it names (an empty comparison fails), the device
                   LM head ran (the TTFT pass needs it), and the item ran: tests/e2e/conftest.py fails a skip (the
                   mesh_device fixture skips on a system mesh smaller than 8x4, and a skipped run exits 0)
Reported, not asserted: the layer drift curve (every layer's PCC, rel, norm bias and worst row), per-chunk final hidden
rel, per-layer state rel, KV PCC by chunk and by rotary / pass-through key dims, the LM head's top-1 and row PCC on the
sampled rows, its grade against the host head on the device's own final hidden (the head alone, without the layers'
drift), its last row against the golden (top-5 included), the host head's top-1, top-5, top-5 hit, teacher-forced
accuracy, NLL and perplexity of device and golden on the sampled rows of every chunk, and every timing.

Report: $DEVICE_TEST_OUT/mimo_v26_flash_rl_e2e.json when DEVICE_TEST_OUT is set, else
generated/mimo_v26_flash_rl_e2e/report.json (relative to the working directory: the workspace root under
run_safe_pytest.sh), rewritten after every pass and chunk, so a case that raises keeps what it measured before (status
"error"); a skip or an error in setup writes it too (tests/e2e/conftest.py). One file per session: with several items
selected, it holds the last one's.
Golden: $MIMO_E2E_GOLDENS (goldens.py); model switches: tt/settings.py (MIMO_V26_FLASH_RL_*).
"""

from __future__ import annotations

import functools
import json
import math
import os
import socket
import subprocess
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.mimo_v26_flash_rl.reference.weights import Checkpoint
from models.demos.mimo_v26_flash_rl.tests.e2e import goldens
from models.demos.mimo_v26_flash_rl.tt import ccl
from models.demos.mimo_v26_flash_rl.tt.model import SLOT, MimoModel
from models.demos.mimo_v26_flash_rl.tt.runners.kv_caches import allocate_kv_caches, is_sliding
from models.demos.mimo_v26_flash_rl.tt.runners.runtime import _Spec
from models.demos.mimo_v26_flash_rl.tt.settings import settings

TEST_PATH = "models/demos/mimo_v26_flash_rl/tests/e2e/test_prefill_e2e.py"
NUM_LAYERS = 48  # tt/runners/adapter.py MimoServiceConfig.NUM_LAYERS
CHUNK = 5120  # the served chunk (adapter.py CHUNK_SIZE)
CAPACITY = 56320  # the served context (adapter.py MAX_SEQ_LEN)
SLOTS = 2  # the served slots (bringup/hooks.py SLOTS, serving_answers.yaml); the test runs slot SLOT (0)
TILE = 32
MESH = (8, 4)  # SP 8 on axis 0, TP 4 on axis 1 (adapter.py MESH_SHAPE)
DEVICE_PARAMS = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}  # the spec's bringup.box.device_params
TIMEOUT_S = 9000  # 48 layers x 11 chunks x 5 passes plus a ~10 min load; the stage's 3 h covers it and the reset

# tools/prefill/bringup_fw/defaults.yaml thresholds (the package may not import the framework); the Qwen e2e test's
THRESHOLDS = {
    "layer": 0.97,  # every layer's output PCC (lower limit)
    "layer_rel": 0.15,  # every layer's output rel L2 (upper limit)
    "layer_bias": 0.005,  # every layer's output |median row norm ratio - 1|, reported against this, not asserted
    "final_hidden": 0.97,  # every chunk's final norm output PCC
    "state": 0.97,  # every key / value tensor PCC
    "top5": 0.9,  # ladder topk_match, k = 5, mean over the last chunk's sampled rows
}
STORED_TOP1 = 1.0  # the device's next token at a stored-logits row is the golden's top 1 (ties count)
HOST_HEAD_SANITY = 0.999  # the host LM head (the device head's golden) on the golden's own last row vs its logits
TTFT_HEAD_SANITY = 0.9999  # the TTFT head path vs model.logits on the same hidden state (warned below it)
SPEC_LOWER_BOUND = 0.85  # the spec's acceptance.pcc_lower_bound: no asserted PCC limit goes below it
assert all(THRESHOLDS[k] >= SPEC_LOWER_BOUND for k in ("layer", "final_hidden", "state"))


@dataclass(frozen=True)
class Case:
    golden: str  # goldens.load name
    isl: int  # the golden's first isl tokens, prefilled in CHUNK-token chunks
    layers: bool  # the golden stores every layer's output there: graded on the last chunk (asserted)
    logits: bool  # the golden stores the logits of position isl - 1 (the device's top 1 there asserted against them)


CASES = {
    "isl5120": Case("isl56320", 5120, layers=True, logits=False),
    "isl56320": Case("isl56320", 56320, layers=True, logits=True),
}
CASE_SETS = {"all": ("isl5120", "isl56320"), "isl56320": ("isl56320",), "isl5120": ("isl5120",)}
HEAD_BLOCK = 128  # host LM head rows per matmul ([128, 152576] fp32 logits = 78 MB)
STORED = "stored golden logits"
HOST_GOLDEN = "host head on the golden final hidden"
TIMING_KEYS = ("cold_ms", "prefill_ms", "tok_s", "host_enqueue_ms", "ttft_ms", "lm_head_ms")


# ---- metrics (host, torch only)
def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation in float64, as the framework's core/metrics.py pcc, except that two constant tensors agree
    only when they are equal (a NaN anywhere gives 0 or NaN, both below every limit)."""
    a = a.double().flatten()
    b = b.double().flatten()
    da, db = a - a.mean(), b - b.mean()
    den = da.norm() * db.norm()
    return float((da @ db) / den) if den > 0 else float(torch.equal(a, b))


def rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    g, w = got.double(), want.double()
    return float((g - w).norm() / w.norm().clamp_min(1e-30))


def worst(values, high: bool = False) -> float:
    """The minimum (with high, the maximum) of a non-empty sequence, NaN when any value is NaN (min() and max() would
    pass over it)."""
    values = list(values)
    if any(math.isnan(v) for v in values):
        return float("nan")
    return max(values) if high else min(values)


def perplexity(nll: float) -> float:
    """exp(nll); inf past exp's range (math.exp raises OverflowError above ~709.78, which a blown-up but finite final
    hidden state reaches), NaN for NaN."""
    try:
        return math.exp(nll)
    except OverflowError:
        return math.inf


def layer_errors(got: torch.Tensor, want: torch.Tensor) -> dict:
    """A layer output vs the golden beyond PCC (testing/ladder.py layer_errors): rel L2, |median row norm ratio - 1|
    (bias) and the worst row's rel L2 (rows = tokens)."""
    g, w = got.float().reshape(-1, want.shape[-1]), want.float().reshape(-1, want.shape[-1])
    wn = w.norm(dim=1).clamp_min(1e-6 * w.norm(dim=1).mean().item() + 1e-30)
    return {
        "rel": ((g - w).norm() / w.norm().clamp_min(1e-30)).item(),
        "bias": abs((g.norm(dim=1) / wn).median().item() - 1.0),
        "row": ((g - w).norm(dim=1) / wn).max().item(),
    }


def topk_match(dev_logits: torch.Tensor, gold_logits: torch.Tensor, k: int) -> torch.Tensor:
    """Per row, the share of the device's top-k ids that the golden scores at least as high as its own k-th logit
    (testing/ladder.py topk_match: two ids tied at the golden's k-th logit are both its top k). [R, V] -> [R]."""
    dev_ids = dev_logits.topk(k, dim=-1).indices
    kth = gold_logits.topk(k, dim=-1).values[:, -1:]
    hit = (gold_logits.gather(-1, dev_ids) >= kth).sum(-1).clamp(max=k)
    return hit.float() / k


def row_pcc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """PCC of every row of two [R, V] fp32 tensors -> [R] (reported only)."""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-30)


def sampled_rows(chunk: int) -> torch.Tensor:
    """Every 16th row of a chunk and its last 32 (testing/ladder.py sampled_rows)."""
    return torch.unique(torch.cat([torch.arange(0, chunk, 16), torch.arange(chunk - 32, chunk)]))


class HostHead:
    """The checkpoint's lm_head (untied, [V, H]) on the host in fp32: logits = h @ W.T."""

    def __init__(self, hf_dir):
        t0 = time.perf_counter()
        self.w = Checkpoint(hf_dir, torch.float32).get("lm_head.weight")
        self.load_s = time.perf_counter() - t0

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return h.float() @ self.w.T


@dataclass
class HeadStats:
    """The host LM head on device vs golden final hidden rows, summed over a case's rows (and per chunk)."""

    sums: dict = field(default_factory=dict)

    def add(self, head: HostHead, dev: torch.Tensor, gold: torch.Tensor, targets: torch.Tensor) -> dict:
        """dev, gold [R, H]; targets [R] the next token of each row, -1 where it is not graded. Returns this call's own
        sums (one chunk)."""
        part = {}
        for b0 in range(0, dev.shape[0], HEAD_BLOCK):
            ld = head.logits(dev[b0 : b0 + HEAD_BLOCK])
            lg = head.logits(gold[b0 : b0 + HEAD_BLOCK])
            t = targets[b0 : b0 + HEAD_BLOCK]
            arg_d, arg_g = ld.argmax(-1), lg.argmax(-1)
            p = row_pcc(ld, lg)
            acc = {
                "rows": float(ld.shape[0]),
                "top1": float(topk_match(ld, lg, 1).sum()),
                "top5": float(topk_match(ld, lg, 5).sum()),
                "top1_exact": float((arg_d == arg_g).sum()),
                "top5_hit": float((lg.topk(5, dim=-1).indices == arg_d[:, None]).any(-1).sum()),
                "logits_pcc": float(p.sum()),
            }
            part["logits_pcc_min"] = worst([part.get("logits_pcc_min", 1.0), *p.tolist()])
            ok = t >= 0
            if ok.any():
                tt = t[ok][:, None]
                acc["graded"] = float(ok.sum())
                acc["acc_dev"] = float((arg_d[ok] == t[ok]).sum())
                acc["acc_gold"] = float((arg_g[ok] == t[ok]).sum())
                acc["nll_dev"] = float((ld[ok].logsumexp(-1) - ld[ok].gather(-1, tt)[:, 0]).sum())
                acc["nll_gold"] = float((lg[ok].logsumexp(-1) - lg[ok].gather(-1, tt)[:, 0]).sum())
            for k, v in acc.items():
                part[k] = part.get(k, 0.0) + v
            del ld, lg
        for k, v in part.items():
            if k == "logits_pcc_min":
                self.sums[k] = worst([self.sums.get(k, 1.0), v])
            else:
                self.sums[k] = self.sums.get(k, 0.0) + v
        return part

    @staticmethod
    def summary(s: dict) -> dict:
        n, m = s.get("rows", 0.0), s.get("graded", 0.0)
        out = {"rows": int(n), "graded_rows": int(m)}
        if n:
            for k in ("top1", "top5", "top1_exact", "top5_hit", "logits_pcc"):
                out[k] = s[k] / n
            out["logits_pcc_min"] = s["logits_pcc_min"]
        if m:
            for k in ("acc_dev", "acc_gold", "nll_dev", "nll_gold"):
                out[k] = s[k] / m
            out["ppl_dev"], out["ppl_gold"] = perplexity(out["nll_dev"]), perplexity(out["nll_gold"])
            out["nll_ratio"] = out["nll_dev"] / out["nll_gold"] if out["nll_gold"] > 0 else None
        return out


# ---- device boundary: the model's own API, plus read-backs
@dataclass
class Pass:
    ms: float  # start -> the device is done (or -> the next token is on the host)
    enqueue_ms: float  # start -> the host loop returned, before the final sync
    chunk_ms: list
    hidden: object  # the last chunk's final hidden state on the device; the caller frees it
    token: int | None = None
    error: str | None = None


def lm_head_last(model: MimoModel, hidden) -> torch.Tensor:
    """The logits of the staged chunk's last position, [V] fp32: model.logits' ttnn.linear (model.lm_head, bfloat8_b
    [V / tp, H] on TP column c with transpose_b, model.head_cfg, an fp32 output; tt/model.py logits) on the one 32-row
    tile of the TP-gathered rows (ccl.gather_rows: [1, 1, 640, H], the SP row's rows in TP column order) that holds it,
    instead of all 640. The position's device row comes from the chunk's block-cyclic order (model._perm, as
    model.logits and to_host place rows), so only its SP row's chips are read back: tp x [32, V / tp] fp32 (20 MB)."""
    sp, tp = model.sp, model.tp
    per_row = model.chunk // sp
    d = int((model._perm() == model.chunk - 1).nonzero()[0])  # device row of the chunk's last position
    r, local = d // per_row, d % per_row
    t0 = local - local % TILE
    dram = ttnn.DRAM_MEMORY_CONFIG
    xg = ccl.gather_rows(hidden)
    tile = ttnn.slice(xg, (0, 0, t0, 0), (1, 1, t0 + TILE, xg.shape[-1]), memory_config=dram)
    ttnn.deallocate(xg)
    lg = ttnn.linear(
        tile,
        model.lm_head,
        transpose_b=True,
        dtype=ttnn.float32,
        compute_kernel_config=model.head_cfg,
        memory_config=dram,
    )
    ttnn.deallocate(tile)
    shards = ttnn.get_device_tensors(lg)
    out = torch.cat([ttnn.to_torch(shards[r * tp + c]).reshape(TILE, -1)[local - t0] for c in range(tp)])
    ttnn.deallocate(lg)
    return out[: model.vocab].float()


def prefill(model: MimoModel, state, ids: torch.Tensor, *, sync_each=False, next_token=False) -> Pass:
    """Every chunk as the runtime's prefill_chunk runs it (its metadata staged at its own start, embed -> every layer),
    plus the final norm of the last chunk. sync_each: a sync after every chunk (chunk_ms). next_token: then the LM head
    on the last position and its argmax on the host; the timer stops there."""
    chunk = model.chunk
    n = ids.shape[0] // chunk
    marks, hidden = [], None
    model.sync()  # start from an idle device
    t0 = time.perf_counter()
    for c in range(n):
        s0 = c * chunk
        h = model.embed(ids[s0 : s0 + chunk], start=s0)  # never the cursor: each pass starts again at position 0
        for i in model.layers:
            h2 = model.layer(i, h, s0, state)
            model.free(h)
            h = h2
        if c == n - 1:
            hidden = model.final_norm(h)
        model.free(h)
        if sync_each:
            model.sync()
            marks.append(time.perf_counter())
    t_loop = time.perf_counter()
    token, error = None, None
    if next_token:
        try:
            token = int(lm_head_last(model, hidden).argmax())
        except RuntimeError as e:
            error = f"{type(e).__name__}: {e}"
    if token is None:
        model.sync()
    t_end = time.perf_counter()
    bounds = [t0, *marks]
    return Pass(
        ms=(t_end - t0) * 1e3,
        enqueue_ms=(t_loop - t0) * 1e3,
        chunk_ms=[round((bounds[c + 1] - bounds[c]) * 1e3, 1) for c in range(len(marks))],
        hidden=hidden,
        token=token,
        error=error,
    )


# ---- one case
@dataclass
class CaseRun:
    """One case on the loaded model: the model, its state, the host head, the golden, the case, its token ids, its
    result (filled in by the passes; the report holds it) and save (rewrites the report)."""

    model: MimoModel
    state: object
    head: HostHead
    g: goldens.Golden
    case: Case
    ids: torch.Tensor
    res: dict
    save: Callable[[], object]
    cfg: dict  # the checkpoint's config.json
    rope_dim: int  # the rotary dims at the start of every key head (head_dim * partial_rotary_factor)


def timed_passes(run: CaseRun) -> torch.Tensor:
    """cold, warm, chunks, ttft (module docstring), each pass's timing saved as it ends. Returns the warm pass's
    last-chunk final hidden on the host."""
    model, state, ids, res = run.model, run.state, run.ids, run.res
    isl, errors, tokens = ids.shape[0], res["errors"], res.setdefault("tokens", {})
    cols = tuple(model.mesh.shape)[1]
    t = res["timing"] = {"ttft_readback_mb": round(cols * TILE * -(-model.vocab // cols) * 4 / 1e6, 1)}
    cold = prefill(model, state, ids)
    model.free(cold.hidden)
    t["cold_ms"] = round(cold.ms, 1)
    run.save()
    warm = prefill(model, state, ids)
    t["prefill_ms"] = round(warm.ms, 1)
    t["tok_s"] = round(isl / max(warm.ms / 1e3, 1e-9), 1)
    t["host_enqueue_ms"] = round(warm.enqueue_ms, 1)
    det_warm = model.to_host(warm.hidden)  # after the timer: the determinism reference
    lm_ms = None
    try:
        lm_head_last(model, warm.hidden)  # compiles the head's programs
        model.sync()
        t0 = time.perf_counter()
        tokens["warm"] = int(lm_head_last(model, warm.hidden).argmax())
        lm_ms = (time.perf_counter() - t0) * 1e3
    except RuntimeError as e:
        errors.append(f"device LM head: {type(e).__name__}: {e}")
        model.sync()
    model.free(warm.hidden)
    t["lm_head_ms"] = round(lm_ms, 1) if lm_ms is not None else None
    run.save()
    per_chunk = prefill(model, state, ids, sync_each=True)
    model.free(per_chunk.hidden)
    t["chunk_ms"] = per_chunk.chunk_ms
    run.save()
    ttft = None
    if "warm" in tokens:
        ttft = prefill(model, state, ids, next_token=True)
        model.free(ttft.hidden)
        if ttft.error:
            errors.append(f"device LM head in the TTFT pass: {ttft.error}")
        else:
            tokens["ttft"] = ttft.token
    t["ttft_ms"] = round(ttft.ms, 1) if ttft is not None and ttft.token is not None else None
    run.save()
    print(
        f"[{res['case']}] cold {t['cold_ms']:.0f} ms, warm {t['prefill_ms']:.0f} ms ({t['tok_s']:.0f} tok/s, host loop "
        f"{t['host_enqueue_ms']:.0f} ms), ttft {t['ttft_ms']} ms, LM head {t['lm_head_ms']} ms; "
        f"per chunk {t['chunk_ms']}",
        flush=True,
    )
    return det_warm


def compare_head(dev_row: torch.Tensor, gold_row: torch.Tensor) -> dict:
    d, g = dev_row[None], gold_row[None]
    top5 = gold_row.topk(5)
    return {
        "pcc": pcc(dev_row, gold_row),
        "top1": float(topk_match(d, g, 1)[0]),
        "top5": float(topk_match(d, g, 5)[0]),
        "token_device": int(dev_row.argmax()),
        "token_golden": int(gold_row.argmax()),
        "device_top5": dev_row.topk(5).indices.tolist(),
        "golden_top5": top5.indices.tolist(),
        "golden_top5_logits": [round(v, 4) for v in top5.values.tolist()],
    }


def grade_head_rows(
    head: HostHead, dev: torch.Tensor, got: torch.Tensor, want: torch.Tensor
) -> tuple[dict, torch.Tensor]:
    """dev [R, V]: the device LM head's logits of R rows; got / want [R, H]: those rows' final hidden, device and
    golden. The ladder's top-k rule (mean over the rows) and the row PCC against the host head on the golden rows (the
    golden, as a CPU golden's logits are its own head on its own final hidden), and, reported, against the host head on
    the device's own rows (the device head alone, without the layers' drift). Returns them and the golden logits of the
    last row."""
    t1, t5, rp, own_t1, own_t5, own_rp = [], [], [], [], [], []
    gold = None
    for b0 in range(0, dev.shape[0], HEAD_BLOCK):
        d = dev[b0 : b0 + HEAD_BLOCK]
        gold = head.logits(want[b0 : b0 + HEAD_BLOCK])
        t1.append(topk_match(d, gold, 1))
        t5.append(topk_match(d, gold, 5))
        rp.append(row_pcc(d, gold))
        own = head.logits(got[b0 : b0 + HEAD_BLOCK])
        own_t1.append(topk_match(d, own, 1))
        own_t5.append(topk_match(d, own, 5))
        own_rp.append(row_pcc(d, own))
        del own
    top1, top5, p = torch.cat(t1), torch.cat(t5), torch.cat(rp)
    out = {
        "rows": int(dev.shape[0]),
        "top1": float(top1.mean()),
        "top5": float(top5.mean()),
        "top5_min": float(top5.min()),
        "rows_top5_below_1": int((top5 < 1).sum()),
        "pcc_mean": float(p.mean()),
        "pcc_min": worst(p.tolist()),
        "nonfinite_rows": int((~torch.isfinite(dev)).any(-1).sum()),
        "head_only": {
            "vs": "host head on the device final hidden",
            "top1": float(torch.cat(own_t1).mean()),
            "top5": float(torch.cat(own_t5).mean()),
            "pcc_min": worst(torch.cat(own_rp).tolist()),
        },
    }
    return out, gold[-1]


def device_head(run: CaseRun, hid, got: torch.Tensor, want: torch.Tensor) -> dict:
    """The device LM head on the case's last chunk (hid: its final hidden on the device; got / want: on the host,
    device and golden), graded on model.logits (tt/model.py:338-354, the model's own head: [chunk, V] fp32 read back,
    outside every timer) as the ladder grades a CPU golden (testing/ladder.py:219-234): the chunk's sampled rows vs the
    host head on the golden final hidden there (asserted: mean top-5), and the last row, position isl - 1, vs the
    stored golden logits where the golden has them (isl56320: top-1 asserted). The TTFT head path (lm_head_last) is
    compared with model.logits' last row."""
    model, head, res = run.model, run.head, run.res
    chunk, pos = model.chunk, run.case.isl - 1
    rows = sampled_rows(chunk)
    assert int(rows[-1]) == chunk - 1, "the sampled rows end at the chunk's last row"
    rec = {"position": pos, "golden": HOST_GOLDEN}
    dev = tail = None
    try:
        dev = model.logits(hid, rows)  # [rows, V]; the whole [chunk, V] is read back either way
        rec["device"] = "model.logits"
    except RuntimeError as e:
        res["errors"].append(f"model.logits: {type(e).__name__}: {e}")
    try:
        tail = lm_head_last(model, hid)
    except RuntimeError as e:
        res["errors"].append(f"device LM head (TTFT path) in the accuracy pass: {type(e).__name__}: {e}")
    if dev is not None and tail is not None:
        path = {"pcc": pcc(tail, dev[-1]), "max_abs_diff": float((tail - dev[-1]).abs().max())}
        path["argmax_equal"] = int(tail.argmax()) == int(dev[-1].argmax())
        rec["ttft_path_vs_model_logits"] = path
        if not (path["pcc"] >= TTFT_HEAD_SANITY and path["argmax_equal"]):
            res["warnings"].append(f"the TTFT head path differs from model.logits: {path}")
    elif dev is None and tail is not None:  # one row: reported, and evaluate fails the grade (and the error)
        dev, rows, rec["device"] = tail[None], rows[-1:], "TTFT head path, last row only (model.logits failed)"
    if dev is None:
        return rec
    graded, gold_last = grade_head_rows(head, dev, got[rows], want[rows])
    rec.update(graded)
    rec["last_row"] = compare_head(dev[-1], gold_last)
    stored = run.g.logits_at(pos)
    if stored is not None:
        rec["stored"] = {"golden": STORED, **compare_head(dev[-1], stored)}
    return rec


def accuracy_pass(run: CaseRun, det_warm: torch.Tensor) -> None:
    """Every chunk with read-backs (module docstring); fills final_hidden, host_head, layers, determinism and lm_head
    as they come (the report saved after every chunk)."""
    model, state, head, g, case, ids, res = run.model, run.state, run.head, run.g, run.case, run.ids, run.res
    chunk, isl = model.chunk, case.isl
    n = isl // chunk
    last = n - 1
    rows = sampled_rows(chunk)
    stored_layers = set(g.output_layers()) if case.layers else set()
    n_prefix = res["golden"]["n_prefix_tokens"]
    stats, final, by_chunk, by_layer = HeadStats(), [], [], {}
    res["final_hidden"] = {"by_chunk": final}
    res["host_head"] = {"by_chunk": by_chunk}
    t0 = time.perf_counter()
    for c in range(n):
        s0, s1 = c * chunk, (c + 1) * chunk
        h = model.embed(ids[s0:s1], start=s0)
        for i in model.layers:
            h2 = model.layer(i, h, s0, state)
            model.free(h)
            h = h2
            if c == last and i in stored_layers:
                got, want = model.to_host(h), g.layer_out(i, s0, s1)
                by_layer[str(i)] = {"pcc": pcc(got, want), **layer_errors(got, want)}
                del got, want
        if c == last and by_layer:
            res["layers"] = layer_summary(by_layer, model.layers, s0=s0)
        hid = model.final_norm(h)
        model.free(h)
        got, want = model.to_host(hid), g.final(s0, s1).float()
        final.append({"start": s0, "pcc": pcc(got, want), "rel": rel_l2(got, want)})
        nxt = s0 + rows + 1  # the next token of each sampled row; a chat-prefix token is not graded
        graded = (nxt < isl) & (nxt >= n_prefix)
        targets = torch.where(graded, ids[nxt.clamp(max=isl - 1)], torch.full_like(nxt, -1))
        part = HeadStats.summary(stats.add(head, got[rows], want[rows], targets))
        by_chunk.append({"start": s0, **part})
        if c == last:
            same = torch.equal(got, det_warm)
            res["determinism"] = {
                "final_hidden_equal": same,
                "max_abs_diff": 0.0 if same else float((got - det_warm).abs().max()),
            }
            run.save()  # before the LM head's 5 GB read-back
            res["lm_head"] = device_head(run, hid, got, want)
        model.free(hid)
        del got, want
        run.save()
        print(
            f"[{res['case']}] chunk {c} [{s0},{s1}): final hidden pcc {final[-1]['pcc']:.6f}, host-head top1 "
            f"{part.get('top1', float('nan')):.4f} top5 {part.get('top5', float('nan')):.4f}",
            flush=True,
        )
    res["final_hidden"]["min_pcc"] = worst(r["pcc"] for r in final)
    res["final_hidden"]["max_rel"] = worst((r["rel"] for r in final), high=True)
    res["host_head"] = {**HeadStats.summary(stats.sums), "by_chunk": by_chunk}
    res["accuracy_pass_s"] = round(time.perf_counter() - t0, 1)
    run.save()


def worst_layer(by_layer: dict, graded: list[int], k: str, high: bool = False) -> list:
    """[layer, value] of the worst graded layer by metric k: the lowest, or with high the highest; NaN is the worst."""

    def badness(i):
        v = by_layer[str(i)][k]
        return math.inf if math.isnan(v) else (v if high else -v)

    i = max(graded, key=badness)
    return [i, by_layer[str(i)][k]]


def layer_summary(by_layer: dict, layers: list[int], s0: int) -> dict:
    graded = [i for i in layers if str(i) in by_layer]
    return {
        "chunk_start": s0,
        "graded": graded,
        **{k: [by_layer[str(i)][k] for i in graded] for k in ("pcc", "rel", "bias", "row")},
        "by_layer": by_layer,
        "worst_pcc": worst_layer(by_layer, graded, "pcc"),
        "worst_rel": worst_layer(by_layer, graded, "rel", high=True),
        "worst_bias": worst_layer(by_layer, graded, "bias", high=True),
        "bias_over_limit": [i for i in graded if not by_layer[str(i)]["bias"] <= THRESHOLDS["layer_bias"]],
    }


def check_state(run: CaseRun) -> dict:
    """The served KV of slot SLOT after the accuracy pass (positions [0, isl)) vs the golden, into res["state"] as it
    goes: every attention layer through the model's own read-back (State.to_torch: [KV, isl, width] fp32 in global
    positions, one layer at a time, so the host holds one layer's every slot and chip at once, < 1 GB)."""
    model, state, g = run.model, run.state, run.g
    chunk, isl, rot = model.chunk, run.case.isl, run.rope_dim
    n = isl // chunk
    out = run.res["state"] = {"kv": {}}
    by_chunk = [[] for _ in range(n)]
    t0 = time.perf_counter()
    for i in model.layers:
        got_kv, want = state.to_torch(i, isl), g.kv(i, isl)
        rec = {"kind": "sliding" if is_sliding(run.cfg, i) else "full"}
        for name in ("key", "value"):
            got, w = got_kv[name].float(), want[name].float()
            assert got.shape == w.shape, f"layer {i} {name}: device {tuple(got.shape)}, golden {tuple(w.shape)}"
            rec[name] = pcc(got, w)
            rec[f"{name}_rel"] = rel_l2(got, w)
            for c in range(n):
                by_chunk[c].append(pcc(got[:, c * chunk : (c + 1) * chunk], w[:, c * chunk : (c + 1) * chunk]))
            if name == "key":  # a golden stored before RoPE would show as low PCC on the rotary dims only
                rec["key_rot_min"] = worst(pcc(got[h, :, :rot], w[h, :, :rot]) for h in range(got.shape[0]))
                rec["key_pass_min"] = worst(pcc(got[h, :, rot:], w[h, :, rot:]) for h in range(got.shape[0]))
            del got, w
        out["kv"][str(i)] = rec
        del got_kv, want
    out["kv_pcc_by_chunk"] = [worst(v) if v else None for v in by_chunk]
    out["state_s"] = round(time.perf_counter() - t0, 1)
    return out


def evaluate(res: dict, case: Case, layers: list[int]) -> list[str]:
    """The asserted checks (module docstring) on one case's result. Each check must have covered everything it names:
    a missing chunk, layer or row fails, and so does a NaN."""
    T, fails = THRESHOLDS, []
    fh = (res.get("final_hidden") or {}).get("by_chunk") or []
    if len(fh) != res["chunks"] or not fh:
        fails.append(f"final hidden graded on {len(fh)} of {res['chunks']} chunks")
    for r in fh:
        if not r["pcc"] >= T["final_hidden"]:
            fails.append(f"final hidden, chunk at {r['start']}: pcc {r['pcc']:.5f} < {T['final_hidden']}")
    if case.layers:
        by = (res.get("layers") or {}).get("by_layer") or {}
        miss = [i for i in layers if str(i) not in by]
        if miss or not layers:
            fails.append(f"layer outputs graded on {len(layers) - len(miss)} of {len(layers)} layers: missing {miss}")
        for i in layers:
            m = by.get(str(i))
            if m is None:
                continue
            if not m["pcc"] >= T["layer"]:
                fails.append(f"layer {i}: pcc {m['pcc']:.5f} < {T['layer']}")
            if not m["rel"] <= T["layer_rel"]:
                fails.append(f"layer {i}: rel {m['rel']:.5f} > {T['layer_rel']}")
    got = (res.get("state") or {}).get("kv") or {}
    miss = [i for i in layers if str(i) not in got]
    if miss or not layers:
        fails.append(f"kv state graded on {len(layers) - len(miss)} of {len(layers)} layers: missing {miss}")
    for i, rec in got.items():
        for k in ("key", "value"):
            v = rec.get(k, float("nan"))
            if not v >= T["state"]:
                fails.append(f"state {k}, layer {i}: pcc {v:.5f} < {T['state']}")
    lm = res.get("lm_head") or {}
    n_rows = len(sampled_rows(res["chunk"]))
    if lm.get("device") != "model.logits" or lm.get("rows") != n_rows or "top5" not in lm:
        fails.append(
            f"the device LM head was not graded through model.logits on the {n_rows} sampled rows (graded: "
            f"{lm.get('rows', 0)} rows through {lm.get('device', 'nothing')})"
        )
    elif not lm["top5"] >= T["top5"]:
        fails.append(
            f"LM head on {n_rows} sampled rows: top5 {lm['top5']:.4f} < {T['top5']} (ladder topk rule, vs the "
            f"{lm['golden']}; worst row {lm['top5_min']:.2f}, {lm['rows_top5_below_1']} rows below 1)"
        )
    if lm.get("nonfinite_rows"):
        fails.append(f"LM head: {lm['nonfinite_rows']} sampled rows hold a non-finite logit")
    if case.logits:
        sto = lm.get("stored") or {}
        if "top1" not in sto:
            fails.append("the device LM head's last row was not graded against the stored golden logits")
        elif not sto["top1"] >= STORED_TOP1:
            fails.append(
                f"LM head at position {lm['position']}: next token {sto['token_device']}, the stored golden logits' "
                f"{sto['token_golden']} (device top5 {sto['device_top5']}, golden {sto['golden_top5']})"
            )
    det = res.get("determinism")
    if det is None:
        fails.append("determinism was not checked")
    elif not det["final_hidden_equal"]:
        diff = det["max_abs_diff"]
        fails.append(f"the warm and accuracy passes' last-chunk final hidden differ (max abs {diff:.3g})")
    if (res.get("timing") or {}).get("ttft_ms") is None:
        fails.append("no TTFT: the device LM head did not run")
    return fails


def run_case(
    model: MimoModel, state, cfg: dict, head: HostHead, name: str, res: dict, save: Callable[[], object]
) -> None:
    """One case into res, which the report already holds: save() rewrites the report as the passes go, so an
    exception keeps what the case measured before it."""
    case = CASES[name]
    chunk, isl = model.chunk, case.isl
    res.update(case=name, isl=isl, chunk=chunk, chunks=isl // chunk, expects=asdict(case), errors=[], warnings=[])
    g = goldens.load(case.golden)
    res["golden"] = g.provenance()
    assert isl % chunk == 0 and isl <= min(model.max_seq, g.n_tokens), (isl, chunk, model.max_seq, g.n_tokens)
    kinds = g.layer_kinds()
    mine = ["sliding_attention" if is_sliding(cfg, i) else "full_attention" for i in range(NUM_LAYERS)]
    if kinds is not None and kinds[:NUM_LAYERS] != mine:
        res["errors"].append(f"golden {g.name}: its layer kinds are not the checkpoint's hybrid_layer_pattern")
    print(f"\n==== {name}: {isl} tokens in {res['chunks']} chunks of {chunk}, golden {g.dir}", flush=True)
    rope_dim = int(cfg["head_dim"] * cfg["partial_rotary_factor"])  # the model's own formula (tt/model.py)
    run = CaseRun(model, state, head, g, case, g.ids(isl), res, save, cfg, rope_dim)
    det_warm = timed_passes(run)
    accuracy_pass(run, det_warm)
    check_state(run)
    res["failures"] = res["errors"] + evaluate(res, case, model.layers)
    res["status"] = "failed" if res["failures"] else "passed"
    print_case(res)


def host_head_sanity(head: HostHead) -> dict:
    """The host head on the isl56320 golden's own final hidden at its last position vs the stored logits there: the
    head that is the device LM head's golden (expected PCC ~1: the golden ran the head in bf16 on the H200)."""
    g = goldens.load("isl56320")
    pos = g.n_tokens - 1
    want = g.logits_at(pos)
    got = head.logits(g.final(pos, pos + 1).float())[0]
    out = {"golden": g.name, "position": pos, "pcc": pcc(got, want), "limit": HOST_HEAD_SANITY}
    out["top1"] = float(topk_match(got[None], want[None], 1)[0])
    out["top5"] = float(topk_match(got[None], want[None], 5)[0])
    return out


# ---- report
def report_path() -> Path:
    out = os.environ.get("DEVICE_TEST_OUT")
    return Path(out) / "mimo_v26_flash_rl_e2e.json" if out else Path("generated/mimo_v26_flash_rl_e2e/report.json")


def _json_safe(x):
    """NaN / inf as null (strict JSON readers reject them); the failures already name every NaN check."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def write_report(report: dict) -> Path:
    p = report_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_safe(report), indent=1, default=str) + "\n")
    os.replace(tmp, p)
    return p


def record_setup_problem(case_set: str | None, what: str) -> Path:
    """The report of an item that never reached its body (tests/e2e/conftest.py: a skip or an error in setup, such as
    the mesh_device fixture's), status error, so that a report an earlier run left is not taken for this run's."""
    return write_report(
        {
            "test": TEST_PATH,
            "case_set": case_set,
            "cases_planned": list(CASE_SETS.get(case_set, ())),
            "status": "error",
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "host": socket.gethostname(),
            "error": {"status": "error", "error": what},
            "metrics": {},
            "cases": {},
            "failures": [what],
            "warnings": [],
        }
    )


def repo_info() -> dict:
    repo = Path(__file__).resolve().parents[5]  # models/demos/mimo_v26_flash_rl/tests/e2e/<this file>

    def git(*args):
        try:
            cmd = ["git", "--no-optional-locks", "-C", str(repo), *args]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    changes = git("status", "--porcelain", "--", "models/demos/mimo_v26_flash_rl")
    return {
        "path": str(repo),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "package_changes": changes.splitlines() if changes else [],
    }


def flat_metrics(res: dict) -> dict:
    """The case's headline numbers under flat names (prefill_ms_<isl>, tok_s_<isl>, ttft_ms_<isl>, ...): whatever it
    measured (a case that raised keeps its earlier passes' numbers)."""
    isl, out = res["isl"], {}
    t = res.get("timing") or {}
    out.update({f"{k}_{isl}": t[k] for k in TIMING_KEYS if k in t})
    out.update({f"chunk_ms_{isl}_c{c:02d}": ms for c, ms in enumerate(t.get("chunk_ms") or [])})
    fh = res.get("final_hidden") or {}
    if "min_pcc" in fh:
        out[f"pcc_final_hidden_min_{isl}"] = fh["min_pcc"]
    L = res.get("layers")
    if L:
        out[f"pcc_layer_min_{isl}"] = L["worst_pcc"][1]
        out[f"rel_layer_max_{isl}"] = L["worst_rel"][1]
        out[f"bias_layer_max_{isl}"] = L["worst_bias"][1]
    st = res.get("state") or {}
    for k in ("key", "value"):
        vals = [rec[k] for rec in (st.get("kv") or {}).values() if k in rec]
        if vals:
            out[f"pcc_state_{k}_min_{isl}"] = worst(vals)
    hh = res.get("host_head") or {}
    for k in ("top1", "top5", "top5_hit", "acc_dev", "acc_gold", "nll_dev", "nll_gold"):
        if k in hh:
            out[f"host_head_{k}_{isl}"] = hh[k]
    lm = res.get("lm_head") or {}
    for k in ("rows", "top1", "top5", "top5_min", "pcc_min"):
        if k in lm:
            out[f"lm_head_{k}_{isl}"] = lm[k]
    for k in ("top1", "top5", "pcc"):
        if k in (lm.get("stored") or {}):
            out[f"lm_head_stored_{k}_{isl}"] = lm["stored"][k]
    if "failures" in res:
        out[f"failures_{isl}"] = len(res["failures"])
    return out


def _f(v, spec: str) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def print_case(res: dict) -> None:
    fh = res["final_hidden"]
    print(f"[{res['case']}] final hidden pcc by chunk: " + " ".join(f"{r['pcc']:.5f}" for r in fh["by_chunk"]))
    if "layers" in res:
        L = res["layers"]
        print(f"[{res['case']}] layer drift at chunk start {L['chunk_start']} (pcc / rel / norm bias):")
        cells = [
            f"L{i:02d} {p:.5f}/{r:.4f}/{b:.4f}"
            for i, p, r, b in zip(L["graded"], L["pcc"], L["rel"], L["bias"], strict=True)
        ]
        for i0 in range(0, len(cells), 8):
            print("    " + "  ".join(cells[i0 : i0 + 8]))
        print(f"    layers with norm bias over {THRESHOLDS['layer_bias']} (reported): {L['bias_over_limit']}")
    st = res["state"]
    cells = [f"L{i} {rec['key']:.4f}/{rec['value']:.4f}" for i, rec in st["kv"].items()]
    for i0 in range(0, len(cells), 8):
        print(f"[{res['case']}] state kv pcc (key/value): " + ", ".join(cells[i0 : i0 + 8]))
    print(f"[{res['case']}] KV pcc by chunk: " + " ".join(_f(p, ".5f") for p in st["kv_pcc_by_chunk"]))
    lm = res.get("lm_head") or {}
    if "top5" in lm:
        ho = lm["head_only"]
        print(
            f"[{res['case']}] LM head ({lm['device']}) on {lm['rows']} sampled rows vs the {lm['golden']}: top1 "
            f"{lm['top1']:.4f} top5 {lm['top5']:.4f} (worst row {lm['top5_min']:.2f}), row pcc mean "
            f"{lm['pcc_mean']:.5f} min {_f(lm['pcc_min'], '.5f')}; the head alone (vs the {ho['vs']}): top1 "
            f"{ho['top1']:.4f} top5 {ho['top5']:.4f} row pcc min {_f(ho['pcc_min'], '.5f')}"
        )
    for key in ("last_row", "stored"):
        r = lm.get(key)
        if r:
            print(
                f"[{res['case']}] LM head at {lm['position']} vs the {lm['golden'] if key == 'last_row' else STORED}: "
                f"pcc {r['pcc']:.5f} top1 {r['top1']:.0f} top5 {r['top5']:.2f}; device {r['device_top5']}, golden "
                f"{r['golden_top5']}"
            )
    hh = res["host_head"]
    print(
        f"[{res['case']}] host head on {hh['rows']} rows: top1 {_f(hh.get('top1'), '.4f')} top5 "
        f"{_f(hh.get('top5'), '.4f')} top5 hit {_f(hh.get('top5_hit'), '.4f')}; teacher-forced acc device "
        f"{_f(hh.get('acc_dev'), '.4f')} golden {_f(hh.get('acc_gold'), '.4f')}, NLL device "
        f"{_f(hh.get('nll_dev'), '.4f')} golden {_f(hh.get('nll_gold'), '.4f')} ({hh['graded_rows']} rows)"
    )
    for w in res["warnings"]:
        print(f"[{res['case']}] WARN {w}")


def _state_min(st: dict, kind: str, k: str) -> str:
    vals = [rec[k] for rec in st[kind].values() if k in rec]
    return _f(worst(vals), ".4f") if vals else "-"


def print_table(report: dict) -> None:
    print(
        f"\n{'case':9} {'isl':>6} {'prefill_ms':>10} {'tok/s':>7} {'ttft_ms':>8} {'enq_ms':>7} {'cold_ms':>8} "
        f"{'final_min':>9} {'layer_min(L)':>14} {'rel_max(L)':>12} {'k_min':>7} {'v_min':>7} "
        f"{'lm_top5':>7} {'h_top1':>6} {'h_top5':>6} {'nll dev/gold':>13} {'det':>4} {'fails':>5}"
    )
    for name, r in report["cases"].items():
        if "timing" not in r or "state" not in r or "failures" not in r:
            print(f"{name:9} (no result: {str(r.get('error', r.get('status', '')))[:100]})")
            continue
        t, hh, st, lay = r["timing"], r["host_head"], r["state"], r.get("layers")
        lm = r.get("lm_head") or {}
        lay_min = f"{_f(lay['worst_pcc'][1], '.5f')}(L{lay['worst_pcc'][0]:02d})" if lay else "-"
        rel_max = f"{_f(lay['worst_rel'][1], '.4f')}(L{lay['worst_rel'][0]:02d})" if lay else "-"
        print(
            f"{name:9} {r['isl']:>6} {_f(t['prefill_ms'], '.0f'):>10} {_f(t['tok_s'], '.0f'):>7} "
            f"{_f(t['ttft_ms'], '.0f'):>8} {_f(t['host_enqueue_ms'], '.0f'):>7} {_f(t['cold_ms'], '.0f'):>8} "
            f"{_f(r['final_hidden']['min_pcc'], '.5f'):>9} {lay_min:>14} {rel_max:>12} "
            f"{_state_min(st, 'kv', 'key'):>7} {_state_min(st, 'kv', 'value'):>7} {_f(lm.get('top5'), '.4f'):>7} {_f(hh.get('top1'), '.4f'):>6} "
            f"{_f(hh.get('top5'), '.4f'):>6} {_f(hh.get('nll_dev'), '.3f'):>6}/{_f(hh.get('nll_gold'), '.3f'):<6} "
            f"{'ok' if r.get('determinism', {}).get('final_hidden_equal') else 'DIFF':>4} {len(r['failures']):>5}"
        )


# ---- the test
@pytest.mark.timeout(TIMEOUT_S)
@pytest.mark.parametrize("case_set", list(CASE_SETS), ids=list(CASE_SETS))
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True, ids=["ring"])
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True, ids=["8x4"])
def test_prefill_e2e(mesh_device, case_set):
    assert tuple(mesh_device.shape) == MESH, f"MiMo-V2.6-Flash prefill runs an 8x4 mesh, got {tuple(mesh_device.shape)}"
    hf = settings.get("SERVE_HF_DIR")
    cfg = json.loads((Path(hf) / "config.json").read_text())
    assert cfg["num_hidden_layers"] == NUM_LAYERS, f"{hf}: {cfg['num_hidden_layers']} layers, the served model has 48"
    report = {
        "test": TEST_PATH,
        "case_set": case_set,
        "cases_planned": list(CASE_SETS[case_set]),
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": socket.gethostname(),
        "torch_threads": torch.get_num_threads(),
        "thresholds": THRESHOLDS,
        "limits": {"stored_top1": STORED_TOP1, "host_head_sanity": HOST_HEAD_SANITY},
        "reported_limits": {"ttft_head_vs_model_logits": TTFT_HEAD_SANITY},
        "spec_lower_bound": SPEC_LOWER_BOUND,
        "tt_metal": repo_info(),
        "settings": settings.all(),
        "mesh": {"shape": list(MESH), "device_params": {k: str(v) for k, v in DEVICE_PARAMS.items()}},
        "metrics": {},
        "cases": {},
        "failures": [],
        "warnings": [],
    }
    save = functools.partial(write_report, report)
    print(f"report: {save().resolve()}", flush=True)

    name = None
    try:
        layers = list(range(NUM_LAYERS))
        # as MimoPrefillRuntime.compile: the engine-owned served caches, the model over them at their capacity
        kv = allocate_kv_caches(
            mesh_device, cfg, layers=layers, num_users=SLOTS, max_seq_len=CAPACITY, chunk_size=CHUNK
        )
        model = MimoModel(
            mesh_device,
            _Spec(hf, CHUNK, kv.max_seq_len),
            layers,
            lm_head=True,
            slots=kv.num_users,
            kv_caches=kv.layers,
        )
        state = model.new_state(CAPACITY)
        head = HostHead(hf)
        report["model"] = {
            "hf_dir": str(hf),
            "layers": len(model.layers),
            "sliding_layers": [i for i in layers if is_sliding(cfg, i)],
            "chunk": model.chunk,
            "capacity": model.max_seq,
            "slots": kv.num_users,
            "slot": SLOT,
            "lm_head": True,
            "load_s": round(model.load_seconds, 1),
            "fast_runtime_mode": bool(ttnn.CONFIG.enable_fast_runtime_mode),
            "fast_runtime_mode_in_layers": False,  # MimoModel.layer turns it off around each block
            "host_head_load_s": round(head.load_s, 1),
        }
        print(f"model loaded in {model.load_seconds:.0f} s, host LM head in {head.load_s:.0f} s", flush=True)
        sanity = report["host_head_sanity"] = host_head_sanity(head)
        if not sanity["pcc"] >= HOST_HEAD_SANITY:  # the host head is the device LM head's golden
            report["failures"].append(
                f"the host LM head on the {sanity['golden']} golden's last row vs its stored logits: pcc "
                f"{sanity['pcc']:.6f} < {HOST_HEAD_SANITY} (top1 {sanity['top1']:.0f}, top5 {sanity['top5']:.2f})"
            )
        save()

        for name in CASE_SETS[case_set]:
            res = report["cases"][name] = {"case": name, "status": "running"}
            run_case(model, state, cfg, head, name, res, save)
            report["metrics"].update(flat_metrics(res))
            report["failures"] += [f"{name}: {f}" for f in res["failures"]]
            report["warnings"] += [f"{name}: {w}" for w in res["warnings"]]
            save()
    except BaseException as e:  # record, then re-raise (pytest-timeout's and pytest's own exceptions included)
        err = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
        res = report["cases"].get(name) if name is not None else None
        if res is not None and res.get("status") == "running":
            res.update(status="error", **err)  # keeps what the case measured before the error
            if "isl" in res:
                try:
                    report["metrics"].update(flat_metrics(res))
                except (KeyError, IndexError, TypeError, ValueError) as m:  # never mask the test's own exception
                    res["metrics_error"] = f"{type(m).__name__}: {m}"
        else:
            report["error"] = {"status": "error", **err}
        report["status"] = "error"
        save()
        raise

    report["status"] = "failed" if report["failures"] else "passed"
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path = save()
    print_table(report)
    for w in report["warnings"]:
        print(f"WARN {w}")
    for f in report["failures"]:
        print(f"FAIL {f}")
    print(f"report: {path.resolve()}")
    assert not report["failures"], f"{len(report['failures'])} checks failed: " + "; ".join(report["failures"][:6])
