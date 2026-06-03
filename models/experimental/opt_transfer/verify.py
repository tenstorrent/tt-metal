import numpy as np
import torch


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def golden_outputs(reference_module, example_inputs: tuple) -> torch.Tensor:
    reference_module.eval()
    with torch.no_grad():
        return reference_module(*example_inputs)


def drift_metrics(golden_logits, fused_logits) -> dict:
    g = np.asarray(golden_logits)
    f = np.asarray(fused_logits)
    T = min(len(g), len(f))
    g, f = g[:T], f[:T]
    matches = np.argmax(g, axis=-1) == np.argmax(f, axis=-1)
    token_match_rate = float(matches.mean())
    mism = np.where(~matches)[0]
    first_divergence_step = int(mism[0]) if len(mism) else T
    err = np.linalg.norm(g - f, axis=-1)
    slope = float(np.polyfit(np.arange(T), err, 1)[0]) if T > 1 else 0.0
    return {
        "token_match_rate": token_match_rate,
        "first_divergence_step": first_divergence_step,
        "drift_slope": slope,
        "horizon": T,
    }


def perf_gain_pct(naive_ms: float, fused_ms: float) -> float:
    return (naive_ms - fused_ms) / naive_ms * 100.0


def perf_gate_pass(naive_ms: float, fused_ms: float, min_gain_pct: float) -> bool:
    return perf_gain_pct(naive_ms, fused_ms) >= min_gain_pct


from pathlib import Path

_SHAPE_TEST = """# AUTO-GENERATED model-shape validation for {op} @ dims={dims}
import pytest, torch
from models.experimental.opt_transfer.verify import pcc

@pytest.mark.device
def test_{slug}_model_shape():
    import ttnn
    from models.experimental.opt_transfer.codegen import build_fused
    from models.experimental.opt_transfer.schema import FusionProposal, KBEntry
    d = torch.load("{fixture}")
    prop, entry = FusionProposal(**d["proposal"]), KBEntry.from_dict(d["entry"])
    device = ttnn.open_device(device_id=0)
    try:
        run = build_fused(prop, entry, d["weights"], device, d["dims"])
        out = run(d["input"]); outs = out if isinstance(out, tuple) else (out,)
        golds = d["golden"] if isinstance(d["golden"], (list, tuple)) else (d["golden"],)
        assert min(pcc(g, t) for g, t in zip(golds, outs)) > {thr}
    finally:
        ttnn.close_device(device)
"""


def emit_shape_test(entry, proposal, weights, dims, golden, sample_input, out_dir, threshold=0.99) -> Path:
    """Write a durable, re-runnable model-shape PCC test + its fixture. Returns the test path."""
    out_dir = Path(out_dir)
    (out_dir / "fixtures").mkdir(parents=True, exist_ok=True)
    slug = entry.id.replace(".", "_")
    fixture = out_dir / "fixtures" / f"{slug}.pt"
    torch.save(
        {
            "proposal": proposal.__dict__,
            "entry": entry.to_dict(),
            "weights": weights,
            "dims": dims,
            "golden": golden,
            "input": sample_input,
        },
        fixture,
    )
    path = out_dir / f"test_{slug}_model_shape.py"
    path.write_text(_SHAPE_TEST.format(op=entry.fused_op, dims=dims, slug=slug, fixture=str(fixture), thr=threshold))
    return path


def l1_feasible(total_bytes: int, l1_budget) -> bool:
    """Pre-run guard: does this L1 choice fit the budget? (False -> caller falls back to DRAM)."""
    return l1_budget.fits(total_bytes)


def run_shape_test(path) -> tuple[bool, str]:
    """Run a generated model-shape test via pytest; return (passed, output)."""
    import subprocess

    r = subprocess.run(["pytest", str(path), "-q", "-m", "device"], capture_output=True, text=True)
    return r.returncode == 0, r.stdout + r.stderr
