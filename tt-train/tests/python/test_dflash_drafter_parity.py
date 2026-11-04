# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device parity of the ttml DFlash drafter against a speculators golden (TT_PORT_PLAN D2).

Replays the golden's training steps (same weights, packed rows and fixed anchors) and
compares, in pipeline order:

    weights -> embed -> fc -> hidden_norm -> layer{i}.{input_norm,attn,mlp,out} -> norm
    -> logits -> teacher_logits -> loss -> grads (pre-clip) -> grad_norm
    -> AdamW update (fp32 master-weight deltas) -> later-step losses -> final weights

Every stage is measured even after a failure; the test then fails naming the *first*
failing stage, which is where to look. A table is printed (run with ``-s``) and written
as JSON to ``$DFLASH_REPORT_DIR`` (default: ``tt-train/dflash_parity_reports``).
``DFLASH_DUMP_ACTIVATIONS=1`` also saves every captured device tensor (including
per-layer q/k/v/sdpa) as ``.npz`` next to the report for offline diffing.

Inputs: ``DFLASH_GOLDEN`` = golden ``.npz`` or a directory of them, produced by the
speculators repo's ``scripts/tt/dump_dflash_golden.py``.

Tolerances compare bf16 device math to an fp32 CPU reference. They are deliberately
centralized in ``TOL`` so they can be tuned from the first hardware run.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pytest
import ttnn

import ttml
from ttml.models.dflash import DFlashDeviceBatch, DFlashDrafter, DFlashTrainStep
from ttml.models.dflash.weights import (
    load_speculators_weights,
    read_gradients,
    read_master_weights,
    read_parameters,
    speculators_to_ttml,
    ttml_to_speculators,
)

from test_dflash_drafter_host import GOLDENS, Golden

# (max relative L2 error, min cosine similarity) per stage family.
TOL = {
    "weights": (5e-3, 0.99999),  # bf16 rounding of the loaded fp32 weights
    "activation": (3e-2, 0.9995),
    "logits": (3e-2, 0.9995),
    "loss": (3e-2, None),
    "grad": (8e-2, 0.997),
    "grad_norm": (5e-2, None),
    # One AdamW step is ~ -lr * sign(g): sign flips of tiny bf16 grads are expected, so
    # the update is checked by direction plus a magnitude ratio.
    "update": (0.35, 0.93),
}
UPDATE_NORM_RATIO = (0.8, 1.25)
CONTEXT_STAGES = {"fc", "hidden_norm"}
ALL_ROW_STAGES = {"embed", "teacher_logits"}  # identical for padded anchor slots too


@dataclass
class StageResult:
    stage: str
    family: str
    rel_err: float
    cos: float | None
    max_abs: float
    ok: bool
    note: str = ""


def _metrics(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float, float]:
    a = np.asarray(actual, np.float64).reshape(-1)
    e = np.asarray(expected, np.float64).reshape(-1)
    if a.shape != e.shape:
        raise ValueError(f"shape mismatch {actual.shape} vs {expected.shape}")
    rel = float(np.linalg.norm(a - e) / (np.linalg.norm(e) + 1e-30))
    cos = float(a @ e / (np.linalg.norm(a) * np.linalg.norm(e) + 1e-30))
    return rel, cos, float(np.abs(a - e).max(initial=0.0))


class Report:
    def __init__(self, name: str):
        self.name = name
        self.results: list[StageResult] = []

    def check(self, stage: str, family: str, actual, expected, note: str = "") -> StageResult:
        rel_tol, cos_tol = TOL[family]
        try:
            rel, cos, max_abs = _metrics(actual, expected)
            ok = bool(rel <= rel_tol and (cos_tol is None or cos >= cos_tol) and np.isfinite(rel))
        except Exception as exc:  # shape bugs are findings too; keep going
            rel, cos, max_abs, ok, note = float("nan"), float("nan"), float("nan"), False, f"{note} {exc!r}"
        r = StageResult(stage, family, rel, cos if cos_tol is not None else None, max_abs, ok, note.strip())
        self.results.append(r)
        return r

    def fail(self, stage: str, family: str, note: str) -> None:
        self.results.append(StageResult(stage, family, float("nan"), None, float("nan"), False, note))

    def first_failure(self) -> StageResult | None:
        return next((r for r in self.results if not r.ok), None)

    def table(self) -> str:
        lines = [f"{'stage':44s} {'family':10s} {'rel_err':>10s} {'cos':>10s} {'max_abs':>10s}  ok"]
        for r in self.results:
            cos = f"{r.cos:10.6f}" if r.cos is not None else f"{'-':>10s}"
            lines.append(
                f"{r.stage:44s} {r.family:10s} {r.rel_err:10.3e} {cos} {r.max_abs:10.3e}  {'ok' if r.ok else 'FAIL'}"
                + (f"  ({r.note})" if r.note else "")
            )
        return "\n".join(lines)

    def write(self, extra: dict) -> Path:
        out_dir = Path(
            os.environ.get("DFLASH_REPORT_DIR", Path(__file__).resolve().parents[2] / "dflash_parity_reports")
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.name}.json"
        path.write_text(json.dumps({"results": [asdict(r) for r in self.results], **extra}, indent=2))
        return path


def _to_numpy(t) -> np.ndarray:
    return np.asarray(t.to_numpy(ttnn.DataType.FLOAT32), np.float32)


def _rows(a: np.ndarray) -> np.ndarray:
    """[1, 1, rows, feat] -> [rows, feat]."""
    return a.reshape(-1, a.shape[-1])


@pytest.fixture
def device_ctx():
    ctx = ttml.autograd.AutoContext.get_instance()
    ctx.get_device()
    ctx.set_gradient_mode(ttml.autograd.GradMode.ENABLED)
    yield ctx
    ctx.reset_graph()
    ctx.close_device()


@pytest.mark.requires_device
@pytest.mark.skipif(not GOLDENS, reason="set DFLASH_GOLDEN to a speculators golden .npz")
@pytest.mark.parametrize("golden_path", GOLDENS, ids=lambda p: p.stem)
def test_dflash_drafter_matches_golden(golden_path, device_ctx):
    g = Golden(golden_path)
    cfg, meta = g.cfg, g.meta
    B = cfg.block_size
    report = Report(f"parity_{golden_path.stem}")
    dump = os.environ.get("DFLASH_DUMP_ACTIVATIONS") == "1"
    dumped: dict[str, np.ndarray] = {}

    w0 = g.weights(0)
    spec_shapes = {k: v.shape for k, v in w0.items()}

    model = DFlashDrafter(cfg)
    load_speculators_weights(model, w0)
    expected = speculators_to_ttml(w0)
    for name, value in read_parameters(model).items():
        report.check(f"weights/{name}", "weights", value, expected[name])

    opt = meta["optimizer"]
    stepper = DFlashTrainStep(
        model,
        lr=opt["lr"],
        weight_decay=opt["weight_decay"],
        betas=tuple(opt["betas"]),
        eps=opt["eps"],
        max_grad_norm=opt["max_grad_norm"],
    )
    masters0 = ttml_to_speculators(read_master_weights(stepper.optimizer), spec_shapes)

    for step in range(meta["steps"]):
        hb = g.host_batch(step)
        batch = DFlashDeviceBatch.from_host(hb)
        valid_rows = np.repeat(hb.anchor_valid, B)
        acts: dict[str, np.ndarray] = {}

        def capture(name, tensor, acts=acts):
            acts[name] = _to_numpy(tensor)

        loss = stepper.forward_backward(batch, capture=capture if step == 0 or dump else lambda *_: None)
        report.check(f"step{step}/loss", "loss", loss, g[f"loss{step}"])

        if step == 0 or dump:
            if dump:
                dumped.update({f"step{step}/{k}": v for k, v in acts.items()})
        if step == 0:
            for name in (
                ["embed", "fc", "hidden_norm"]
                + [f"layer{i}.{part}" for i in range(cfg.num_layers) for part in ("input_norm", "attn", "mlp", "out")]
                + ["norm", "logits", "teacher_logits"]
            ):
                ref_key = f"act0/{name}"
                if name not in acts:
                    report.fail(f"step0/{name}", "activation", "not captured")
                    continue
                got, ref = _rows(acts[name]), g[ref_key]
                if name not in CONTEXT_STAGES | ALL_ROW_STAGES:
                    got, ref = got[valid_rows], ref[valid_rows]
                family = "logits" if "logits" in name else "activation"
                report.check(f"step0/{name}", family, got, ref)

        if step in meta["grad_steps"]:
            grads = read_gradients(model, strict=False)
            ref_grads = {k[len(f"g{step}/") :]: g[k] for k in g.z.files if k.startswith(f"g{step}/")}
            # A fused kv_proj without a gradient drops both k_proj and v_proj.
            grads = ttml_to_speculators(grads, spec_shapes)
            for name in sorted(ref_grads):
                if name in grads:
                    report.check(f"step{step}/grad/{name}", "grad", grads[name], ref_grads[name])
                else:
                    report.fail(f"step{step}/grad/{name}", "grad", "no gradient on device (disconnected from loss?)")
            for name in sorted(set(grads) - set(ref_grads)):
                report.fail(f"step{step}/grad/{name}", "grad", "device trains a parameter the reference keeps frozen")

        grad_norm = stepper.clip_and_step()
        report.check(f"step{step}/grad_norm", "grad_norm", grad_norm, g[f"grad_norm{step}"])

        if step + 1 in meta["weight_steps"]:
            masters = ttml_to_speculators(read_master_weights(stepper.optimizer), spec_shapes)
            ref_w = g.weights(step + 1)
            for name in sorted(ref_w):
                if name not in masters:
                    report.fail(f"w{step + 1}/{name}", "update", "no master weight (frozen on device?)")
                    continue
                d_dev = masters[name] - masters0[name]
                d_ref = ref_w[name] - w0[name]
                r = report.check(f"w{step + 1}/update/{name}", "update", d_dev, d_ref)
                ratio = float(np.linalg.norm(d_dev) / (np.linalg.norm(d_ref) + 1e-30))
                if not UPDATE_NORM_RATIO[0] <= ratio <= UPDATE_NORM_RATIO[1]:
                    r.ok = False
                    r.note = f"|update| ratio {ratio:.3f} outside {UPDATE_NORM_RATIO}"

    extra = {"golden": str(golden_path), "meta": meta, "tolerances": TOL}
    if dump:
        dump_path = Path(report.write(extra)).with_suffix(".activations.npz")
        np.savez(dump_path, **dumped)
        extra["activations"] = str(dump_path)
    path = report.write(extra)
    print(f"\n{report.table()}\nreport: {path}")
    first = report.first_failure()
    n_fail = sum(not r.ok for r in report.results)
    assert first is None, (
        f"{n_fail} stage(s) failed; first failing stage: {first.stage} "
        f"(rel_err={first.rel_err:.3e}, cos={first.cos}) {first.note}. Full table in {path}"
    )
