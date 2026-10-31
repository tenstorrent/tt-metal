# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end prefill of Qwen3.8-27B on a Blackhole Galaxy 8x4 mesh: all 64 layers, the final norm and the LM head,
against the HF goldens (transformers on the CPU; tests/e2e/goldens.py). No device profiler, no bring-up framework.

    scripts/run_safe_pytest.sh --run-all models/demos/qwen38_27b/tests/e2e/test_prefill_e2e.py -k <case set>

Case sets (one test item each, one model load per item; select one with -k):
    all       isl5120, isl10240, isl56320, in that order. Without -k / -m (and no node ids) it is the only item that
              runs: the other items' cases are part of it, so tests/e2e/conftest.py deselects them
    isl56320  the full 56320-token golden alone
    isl5120   quick smoke: the first 5120 tokens of the isl56320 golden
Cases: isl5120 is the first 5120 tokens of the isl56320 golden (which ran every layer as 5120-row chunks, so its first
chunk is a 5120-token prefill); isl10240 is the isl10240 golden (its own wikitext tokens, no chat prefix, fp32
compute); isl56320 is the whole isl56320 golden (55-token chat prefix + wikitext, bf16 compute). Every case runs
5120-token chunks from position 0 on slot 0 of one state: a new request (start 0 resets the Gated DeltaNet state,
later chunks overwrite the KV rows below the case's length).

The model is built as the prefill runner builds it (tt/runners/runtime.py Qwen38PrefillRuntime and compile,
adapter.py allocate_kv_cache): TtQwen38Model (tt/model.py) directly over every layer of the checkpoint config's
layer_types, target seq 56320 and chunk 5120, its state bound to an engine-owned TtKVCache (tt/kv_cache.py: settings
NUM_SLOTS slots x the attention layers) by new_state(kv=...), slot 0; each chunk embed(ids, start) -> every layer as
prefill_chunk runs it (no acks). Unlike the runtime it loads the LM head and runs the final norm on the last chunk.
Not through the bring-up hooks; the model's own layer() and final_norm() turn ttnn's fast runtime mode off around
their ops (tt/model.py hooked_dispatch), as they do under the runner and the ladder, so every timing includes the
standard runtime's host dispatch. The mesh opens with the bring-up's box params (FABRIC_1D_RING, default fabric
payload), not the runner's own (PREFILL_FABRIC_MODE, RELAXED_INIT, payload 4352). Checkpoint: $PREFILL_HF_MODEL (the
runner's own override), else adapter.py HF_SNAPSHOT.

Per case, four timed passes (a device sync before each start, nothing read back inside a timer), then an accuracy
pass:
    cold      embed -> 64 layers for every chunk, the final norm of the last chunk, one sync: cold_ms (includes the
              compile of every program this case is the first to run)
    warm      the same: prefill_ms, tok_s, host_enqueue_ms (the host loop returned, before the sync). After the timer:
              the last-chunk final hidden read back (determinism) and the device LM head on its last row, twice
              (compile, then lm_head_ms: head + read-back from an idle device)
    chunks    a sync after every chunk: chunk_ms[c]
    ttft      warm prefill + final norm + the device LM head on the last position + the next token on the host:
              ttft_ms. The head is model.logits' ttnn.linear (model.lm_head, transpose_b, tt/mlp.py compute_config,
              fp32) on the 32 rows of each chip around the last position, read back from the 4 chips holding it only
              (4 x [32, 62080] fp32 = 32 MB);
              model.logits reads back [5120, 248320] fp32 (5 GB), which no timed pass may do
    accuracy  every chunk: final norm -> host vs the golden final hidden, and a host LM head (the checkpoint's untied
              lm_head.weight, fp32) on sampled rows (every 16th + the last 32 of the chunk, as the ladder samples) of
              the device's and of the golden's final hidden. Last chunk: every layer's output vs the golden (where it
              stores them: isl56320, isl5120); model.logits (the device LM head) on the sampled rows vs the host head
              on the golden final hidden there, and its last row vs the golden's stored logits (isl56320), cross-
              checked against the TTFT head path. Then the state of slot 0: K / V of the 16 attention layers over
              [0, isl); conv / recurrent of the 48 Gated DeltaNet layers where the golden holds them (at its own
              length: isl10240, isl56320)

Asserted, at tools/prefill/bringup_fw/defaults.yaml's gate thresholds with testing/ladder.py's formulas (THRESHOLDS):
    final hidden   every chunk: PCC >= 0.97
    layers         every layer's output on the last chunk (goldens with layer outputs): PCC >= 0.97, as the ladder
                   gates them (ladder.py:186-202); its rel L2 and norm bias are reported only (below): the ladder's
                   rel L2 <= 0.15 is a limit for a few layers, and the drift grows with depth (isl56320 on the
                   device: about 0.004 at layer 0, 0.15 to 0.17 at layers 54-63)
    state          key and value of every attention layer; conv and recurrent of every Gated DeltaNet layer where the
                   golden holds them: PCC >= 0.97 (conv on the newest inputs the golden stores: all 4 for isl56320, the
                   last 3 of the device's 4 for isl10240)
    LM head        every case: model.logits on the last chunk's sampled rows (350) vs the host head on the golden final
                   hidden there, the ladder's top-k rule with k = 5 (ties at the golden's 5th logit count), mean over
                   the rows >= 0.9: the ladder's grade with a CPU golden (ladder.py:219-234); no non-finite logit.
                   isl56320, whose golden stores the logits of its last position: the device's next token there is
                   the stored logits' top 1 (STORED_TOP1; their 1st / 2nd gap is 1.25, their 5th / 6th 0.44, so top-5
                   on that one row is reported only)
    host head      the host head (the device head's golden above) on the isl56320 golden's own last row vs its stored
                   logits: PCC >= 0.999 (HOST_HEAD_SANITY)
    determinism    the warm and the accuracy pass's last-chunk final hidden are bit-identical
    completeness   each check above covered every chunk / layer / row it names (an empty comparison fails), the device
                   LM head ran (the TTFT pass needs it), and the item ran: tests/e2e/conftest.py fails a skip (the
                   mesh_device fixture skips on a system mesh smaller than 8x4, and a skipped run exits 0)
Reported, not asserted: the layer drift curve (every layer's PCC, rel, norm bias and worst row), per-chunk final hidden
rel, per-layer state rel, KV PCC by chunk and by rotary / pass-through dims, the LM head's top-1 and row PCC on the
sampled rows, its grade against the host head on the device's own final hidden (the head alone, without the layers'
drift), its last row against the golden (top-5 included), the host head's top-1, top-5, top-5 hit, teacher-forced
accuracy, NLL and perplexity of device and golden on the sampled rows of every chunk, and every timing.

Report: $DEVICE_TEST_OUT/qwen38_27b_e2e.json when DEVICE_TEST_OUT is set, else generated/qwen38_27b_e2e/report.json
(relative to the working directory: the workspace root under run_safe_pytest.sh), rewritten after every pass and
chunk, so a case that raises keeps what it measured before (status "error"); a skip or an error in setup writes it
too (tests/e2e/conftest.py). One file per session: with several items selected, it holds the last one's.
Goldens: $QWEN38_E2E_GOLDENS (goldens.py); model switches: tt/settings.py (QWEN38_27B_*).
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
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.qwen38_27b.reference.qwen38_ref import Checkpoint, Config
from models.demos.qwen38_27b.tests.e2e import goldens
from models.demos.qwen38_27b.tt.gdn import chip_order
from models.demos.qwen38_27b.tt.kv_cache import TtKVCache, blockcyclic_positions
from models.demos.qwen38_27b.tt.mlp import compute_config as mlp_compute_config
from models.demos.qwen38_27b.tt.model import TtQwen38Model
from models.demos.qwen38_27b.tt.runners.adapter import HF_SNAPSHOT, Qwen38ServiceConfig, resolve_hf_dir
from models.demos.qwen38_27b.tt.settings import settings

TEST_PATH = "models/demos/qwen38_27b/tests/e2e/test_prefill_e2e.py"
NUM_LAYERS = Qwen38ServiceConfig.NUM_LAYERS  # 64
CHUNK = Qwen38ServiceConfig.CHUNK_SIZE  # the served chunk (spec target.chunk), 5120
CAPACITY = Qwen38ServiceConfig.MAX_SEQ_LEN  # the served context (spec target.seq), 56320
TILE = 32
MESH = Qwen38ServiceConfig.MESH_SHAPE  # SP 8 on axis 0, TP 4 on axis 1
SLOT = 0  # the state slot every case runs on
DEVICE_PARAMS = {"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}  # the spec's box.device_params

# tools/prefill/bringup_fw/defaults.yaml thresholds (the package may not import the framework)
THRESHOLDS = {
    "layer": 0.97,  # every layer's output PCC (lower limit)
    "layer_rel": 0.15,  # every layer's output rel L2, reported against this, not asserted (it grows with depth)
    "layer_bias": 0.005,  # every layer's output |median row norm ratio - 1|, reported against this, not asserted
    "final_hidden": 0.97,  # every chunk's final norm output PCC
    "state": 0.97,  # every key / value / conv / recurrent tensor PCC
    "top5": 0.9,  # ladder topk_match, k = 5, mean over the last chunk's sampled rows
}
STORED_TOP1 = 1.0  # the device's next token at a stored-logits row is the golden's top 1 (ties count)
HOST_HEAD_SANITY = 0.999  # the host LM head (the device head's golden) on a golden's own last row vs its logits
TTFT_HEAD_SANITY = 0.9999  # the TTFT head path vs model.logits on the same hidden state (warned below it)
SPEC_LOWER_BOUND = 0.85  # the spec's acceptance.pcc_lower_bound: no asserted PCC limit goes below it
assert all(THRESHOLDS[k] >= SPEC_LOWER_BOUND for k in ("layer", "final_hidden", "state"))


@dataclass(frozen=True)
class Case:
    golden: str  # goldens.load name
    isl: int  # the golden's first isl tokens, prefilled in CHUNK-token chunks
    layers: bool  # the golden stores every layer's output there: graded on the last chunk (asserted)
    gdn: bool  # the golden holds the Gated DeltaNet state after position isl - 1 (asserted)
    logits: bool  # the golden stores the logits of position isl - 1 (the device's top 1 there asserted against them)


CASES = {
    "isl5120": Case("isl56320", 5120, layers=True, gdn=False, logits=False),
    "isl10240": Case("isl10240", 10240, layers=False, gdn=True, logits=False),
    "isl56320": Case("isl56320", 56320, layers=True, gdn=True, logits=True),
}
CASE_SETS = {"all": ("isl5120", "isl10240", "isl56320"), "isl56320": ("isl56320",), "isl5120": ("isl5120",)}
HEAD_BLOCK = 128  # host LM head rows per matmul ([128, 248320] fp32 logits = 127 MB)
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
        self.w = Checkpoint(hf_dir).get("lm_head.weight", torch.float32)
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
def attn_layers(model: TtQwen38Model) -> list[int]:
    """The full-attention layers among the model's, in order (the KV cache's layer order; checkpoint layer_types)."""
    return [i for i in model.layers if model.is_attn(i)]


def linear_layers(model: TtQwen38Model) -> list[int]:
    """The Gated DeltaNet layers among the model's, in order."""
    return [i for i in model.layers if not model.is_attn(i)]


@dataclass
class Pass:
    ms: float  # start -> the device is done (or -> the next token is on the host)
    enqueue_ms: float  # start -> the host loop returned, before the final sync
    chunk_ms: list
    hidden: object  # the last chunk's final hidden state on the device; the caller frees it
    token: int | None = None
    error: str | None = None


def lm_head_last(model: TtQwen38Model, hidden) -> torch.Tensor:
    """The logits of the chunk's last position, [V] fp32: model.logits' ttnn.linear (model.lm_head, bfloat8_b
    [V / cols, H] on TP column c, transpose_b, tt/mlp.py compute_config and an fp32 output; TtQwen38Model.logits) on the
    32 rows of every chip around that position instead of all 640. The chunk's rows are in the engine's chip order of
    its start (gdn.chip_order, model.last_start): the last position's chip-order index gives its mesh row and local row
    (at a chunk-aligned start, local row 639 of mesh row 7), so only that row's chips are read back: cols x [32, V / cols]
    fp32 (32 MB)."""
    rows, cols = tuple(model.mesh.shape)
    T, H = model.rows, model.cfg.hidden
    r, lr = divmod(chip_order(model.last_start, model.chunk, rows)[model.chunk - 1], T)
    lo = lr - lr % TILE
    dram = ttnn.DRAM_MEMORY_CONFIG
    tail = ttnn.slice(hidden, (0, 0, lo, 0), (1, 1, lo + TILE, H), memory_config=dram)
    lg = ttnn.linear(
        tail,
        model.lm_head,
        transpose_b=True,
        compute_kernel_config=mlp_compute_config(),
        dtype=ttnn.float32,
        memory_config=dram,
    )
    ttnn.deallocate(tail)
    shards = ttnn.get_device_tensors(lg)
    out = torch.cat([ttnn.to_torch(shards[r * cols + c]).reshape(TILE, -1)[lr - lo] for c in range(cols)])
    ttnn.deallocate(lg)
    return out[: model.cfg.vocab].float()


def prefill(model: TtQwen38Model, state, ids: torch.Tensor, *, sync_each=False, next_token=False) -> Pass:
    """Every chunk as the runtime's prefill_chunk runs it (embed -> every layer), plus the final norm of the last chunk.
    sync_each: a sync after every chunk (chunk_ms). next_token: then the LM head on the last position and its argmax
    on the host; the timer stops there."""
    chunk = model.chunk
    n = ids.shape[0] // chunk
    marks, hidden = [], None
    model.sync()  # start from an idle device
    t0 = time.perf_counter()
    for c in range(n):
        s0 = c * chunk
        h = model.embed(ids[s0 : s0 + chunk], s0)
        for i in model.layers:
            h2 = model.layer(i, h, s0, state, slot=SLOT)
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


def kv_slot_host(model: TtQwen38Model, state) -> tuple[torch.Tensor, torch.Tensor]:
    """Slot SLOT's K and V of every attention layer in natural order, bf16 [n_attn, n_kv, max_seq, head_dim] each
    (bfloat8_b has at most 8 significant bits: exact in bf16). One chip at a time, so the host holds ~3.7 GB and not
    TtKVCache._read's every slot in fp32 at once. Chip (r, c) holds kv head c and, at local row lr, position
    blockcyclic_positions(...)[r * local + lr] of batch slot * n_attn + attention ordinal (tt/kv_cache.py), the order
    TtKVCache._read restores. The ND-sharded cache is copied to interleaved DRAM first, as _read does."""
    kv = state.kv
    rows, cols = tuple(model.mesh.shape)
    n_attn, cap = kv.num_layers, kv.max_seq
    assert n_attn == len(attn_layers(model)), (n_attn, attn_layers(model))
    local = cap // rows
    pos = blockcyclic_positions(rows, kv.chunk, cap).reshape(rows, local)
    b0 = kv.batch(SLOT, 0)
    out = []
    for t in (kv.k, kv.v):
        nat = torch.empty(n_attn, cols, cap, kv.head_dim, dtype=torch.bfloat16)
        flat = ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
        try:
            shards = ttnn.get_device_tensors(flat)
            for r in range(rows):
                for c in range(cols):
                    d = ttnn.to_torch(shards[r * cols + c])  # [slots * n_attn, 1, local, head_dim]
                    nat[:, c, pos[r]] = d[b0 : b0 + n_attn, 0].to(torch.bfloat16)
                    del d
        finally:
            ttnn.deallocate(flat)
        out.append(nat)
    return out[0], out[1]


def kv_host(model: TtQwen38Model, state) -> tuple[torch.Tensor, torch.Tensor, str]:
    """kv_slot_host, or the cache's own whole read-back (TtKVCache._read, every slot in fp32) if it fails."""
    try:
        k, v = kv_slot_host(model, state)
        return k, v, "per-chip shards"
    except (RuntimeError, TypeError, ValueError, IndexError) as e:
        kv = state.kv
        b0, n_attn = kv.batch(SLOT, 0), kv.num_layers
        out = []
        for t in (kv.k, kv.v):
            whole = kv._read(t)
            out.append(whole[b0 : b0 + n_attn].to(torch.bfloat16))
            del whole
        return out[0], out[1], f"TtKVCache._read (the per-chip read-back failed: {type(e).__name__}: {e})"


# ---- one case
@dataclass
class CaseRun:
    """One case on the loaded model: the model, its state, the host head, the golden, the case, its token ids, its
    result (filled in by the passes; the report holds it) and save (rewrites the report)."""

    model: TtQwen38Model
    state: object
    head: HostHead
    g: goldens.Golden
    case: Case
    ids: torch.Tensor
    res: dict
    save: Callable[[], object]


def timed_passes(run: CaseRun) -> torch.Tensor:
    """cold, warm, chunks, ttft (module docstring), each pass's timing saved as it ends. Returns the warm pass's
    last-chunk final hidden on the host."""
    model, state, ids, res = run.model, run.state, run.ids, run.res
    isl, errors, tokens = ids.shape[0], res["errors"], res.setdefault("tokens", {})
    cols = tuple(model.mesh.shape)[1]
    t = res["timing"] = {"ttft_readback_mb": round(cols * TILE * -(-model.cfg.vocab // cols) * 4 / 1e6, 1)}
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
    device and golden), graded on model.logits (TtQwen38Model.logits, the model's own head: [chunk, V] fp32 read back,
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
        h = model.embed(ids[s0:s1], s0)
        for i in model.layers:
            h2 = model.layer(i, h, s0, state, slot=SLOT)
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
        "rel_over_limit": [i for i in graded if not by_layer[str(i)]["rel"] <= THRESHOLDS["layer_rel"]],
    }


def check_state(run: CaseRun) -> dict:
    """The state slot after the accuracy pass (positions [0, isl)) vs the golden, into res["state"] as it goes."""
    model, state, g = run.model, run.state, run.g
    chunk, isl, rot = model.chunk, run.case.isl, model.cfg.rope_dim
    n = isl // chunk
    out = run.res["state"] = {"kv": {}, "gdn": {}}
    by_chunk = [[] for _ in range(n)]
    t0 = time.perf_counter()
    k_nat, v_nat, out["kv_readback"] = kv_host(model, state)
    out["kv_readback_s"] = round(time.perf_counter() - t0, 1)
    for a, i in enumerate(attn_layers(model)):
        want = g.kv(i, isl)
        rec = {}
        for name, nat in (("key", k_nat), ("value", v_nat)):
            got, w = nat[a, :, :isl].float(), want[name].float()
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
        del want
    del k_nat, v_nat
    out["kv_pcc_by_chunk"] = [worst(v) if v else None for v in by_chunk]
    run.save()
    for i in linear_layers(model):
        want = g.gdn(i, isl)
        if want is None:
            continue  # the golden has no state at isl (evaluate requires it where the case expects it)
        # TtModelState.to_torch: conv [conv_dim, conv_k] (the last conv_k inputs, oldest first), recurrent
        # [v_heads, k_dim, v_dim]
        got = state.to_torch(i, isl, slot=SLOT)
        rec = {}
        for name in ("conv", "recurrent"):
            gt, w = got[name].float(), want[name].float()
            if name == "conv" and gt.dim() == w.dim() == 2 and 0 < w.shape[1] < gt.shape[1]:
                gt = gt[:, gt.shape[1] - w.shape[1] :]  # a golden that stores fewer inputs: the newest ones
            assert gt.shape == w.shape, f"layer {i} {name}: device {tuple(gt.shape)}, golden {tuple(w.shape)}"
            rec[name] = pcc(gt, w)
            rec[f"{name}_rel"] = rel_l2(gt, w)
        out["gdn"][str(i)] = rec
    out["state_s"] = round(time.perf_counter() - t0, 1)
    return out


def evaluate(res: dict, case: Case, layers: list[int], attn: list[int], linear: list[int]) -> list[str]:
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
    st = res.get("state") or {}
    for kind, names, need, required in (
        ("kv", ("key", "value"), attn, True),
        ("gdn", ("conv", "recurrent"), linear, case.gdn),
    ):
        got = st.get(kind) or {}
        if required:
            miss = [i for i in need if str(i) not in got]
            if miss or not need:
                fails.append(f"{kind} state graded on {len(need) - len(miss)} of {len(need)} layers: missing {miss}")
        for i, rec in got.items():
            for k in names:
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


def run_case(model: TtQwen38Model, state, head: HostHead, name: str, res: dict, save: Callable[[], object]) -> None:
    """One case into res, which the report already holds: save() rewrites the report as the passes go, so an
    exception keeps what the case measured before it."""
    case = CASES[name]
    chunk, isl = model.chunk, case.isl
    res.update(case=name, isl=isl, chunk=chunk, chunks=isl // chunk, expects=asdict(case), errors=[], warnings=[])
    g = goldens.load(case.golden)
    res["golden"] = g.provenance()
    assert isl % chunk == 0 and isl <= min(state.max_seq, g.n_tokens), (isl, chunk, state.max_seq, g.n_tokens)
    kinds = g.layer_kinds()
    if kinds is not None and kinds != list(model.cfg.layer_types):
        res["errors"].append(f"golden {g.name}: its layer kinds are not the checkpoint config's layer_types")
    print(f"\n==== {name}: {isl} tokens in {res['chunks']} chunks of {chunk}, golden {g.dir}", flush=True)
    run = CaseRun(model, state, head, g, case, g.ids(isl), res, save)
    det_warm = timed_passes(run)
    accuracy_pass(run, det_warm)
    check_state(run)
    res["failures"] = res["errors"] + evaluate(res, case, model.layers, attn_layers(model), linear_layers(model))
    res["status"] = "failed" if res["failures"] else "passed"
    print_case(res)


def host_head_sanity(head: HostHead) -> dict:
    """The host head on the isl56320 golden's own final hidden at its last position vs the stored logits there: the
    head that is the device LM head's golden (expected PCC ~1: the golden ran the head in bf16)."""
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
    return Path(out) / "qwen38_27b_e2e.json" if out else Path("generated/qwen38_27b_e2e/report.json")


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
    repo = Path(__file__).resolve().parents[5]  # models/demos/qwen38_27b/tests/e2e/<this file>

    def git(*args):
        try:
            cmd = ["git", "--no-optional-locks", "-C", str(repo), *args]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    changes = git("status", "--porcelain", "--", "models/demos/qwen38_27b")
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
    for kind, names in (("kv", ("key", "value")), ("gdn", ("conv", "recurrent"))):
        for k in names:
            vals = [rec[k] for rec in (st.get(kind) or {}).values() if k in rec]
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
        print(f"    layers with rel L2 over {THRESHOLDS['layer_rel']} (reported): {L['rel_over_limit']}")
        print(f"    layers with norm bias over {THRESHOLDS['layer_bias']} (reported): {L['bias_over_limit']}")
    st = res["state"]
    for kind in ("kv", "gdn"):
        cells = [f"L{i} " + "/".join(f"{v:.4f}" for k, v in rec.items() if "_" not in k) for i, rec in st[kind].items()]
        for i0 in range(0, len(cells), 8):
            print(f"[{res['case']}] state {kind} pcc: " + ", ".join(cells[i0 : i0 + 8]))
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
        f"{'final_min':>9} {'layer_min(L)':>14} {'rel_max(L)':>12} {'k_min':>7} {'v_min':>7} {'conv_min':>8} "
        f"{'rec_min':>7} {'lm_top5':>7} {'h_top1':>6} {'h_top5':>6} {'nll dev/gold':>13} {'det':>4} {'fails':>5}"
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
            f"{_state_min(st, 'kv', 'key'):>7} {_state_min(st, 'kv', 'value'):>7} {_state_min(st, 'gdn', 'conv'):>8} "
            f"{_state_min(st, 'gdn', 'recurrent'):>7} {_f(lm.get('top5'), '.4f'):>7} {_f(hh.get('top1'), '.4f'):>6} "
            f"{_f(hh.get('top5'), '.4f'):>6} {_f(hh.get('nll_dev'), '.3f'):>6}/{_f(hh.get('nll_gold'), '.3f'):<6} "
            f"{'ok' if r.get('determinism', {}).get('final_hidden_equal') else 'DIFF':>4} {len(r['failures']):>5}"
        )


# ---- the test
@pytest.mark.timeout(5400)
@pytest.mark.parametrize("case_set", list(CASE_SETS), ids=list(CASE_SETS))
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True, ids=["ring"])
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True, ids=["8x4"])
def test_prefill_e2e(mesh_device, case_set):
    assert tuple(mesh_device.shape) == MESH, f"Qwen3.8-27B prefill runs an 8x4 mesh, got {tuple(mesh_device.shape)}"
    hf = resolve_hf_dir(os.environ.get("PREFILL_HF_MODEL") or HF_SNAPSHOT)
    cfg = Config.load(hf)
    assert cfg.num_layers == NUM_LAYERS, f"{hf}: {cfg.num_layers} layers, the served model has {NUM_LAYERS}"
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

    state, cache, name = None, None, None
    try:
        # as Qwen38PrefillRuntime builds it: the device model reads hf_dir and target {seq, chunk} off the spec
        spec = SimpleNamespace(
            hf_dir=Path(hf), get=lambda key: {"seq": CAPACITY, "chunk": CHUNK} if key == "target" else None
        )
        model = TtQwen38Model(mesh_device, spec, list(range(cfg.num_layers)), lm_head=True)
        slots = settings.get("NUM_SLOTS")
        # the engine-owned cache, as adapter.py allocate_kv_cache allocates it, bound as runtime.compile binds it
        cache = TtKVCache(
            mesh_device, cfg.n_kv, cfg.head_dim, CAPACITY, CHUNK, num_slots=slots, num_layers=len(attn_layers(model))
        )
        state = model.new_state(CAPACITY, num_slots=slots, kv=cache)
        head = HostHead(hf)
        report["model"] = {
            "hf_dir": str(hf),
            "layers": len(model.layers),
            "attention_layers": attn_layers(model),
            "chunk": model.chunk,
            "capacity": state.max_seq,
            "slots": slots,
            "slot": SLOT,
            "lm_head": True,
            "load_s": round(model.load_seconds, 1),
            "fast_runtime_mode": bool(ttnn.CONFIG.enable_fast_runtime_mode),
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
            run_case(model, state, head, name, res, save)
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
    finally:
        try:
            if state is not None:
                state.free()  # its KV cache included
            elif cache is not None:
                cache.free()
        except RuntimeError as e:  # do not mask the test's own exception
            print(f"freeing the model state failed: {type(e).__name__}: {e}")

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
