# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bench goldens: one background `ttq submit --kind cpu` job in the reference venv.

Stores inputs and outputs only (D0: no per-block intermediates; those belong to the test goldens):
the context, per-forward head output [R, 2, 576], loc/scale, routing (weights, indices, the selection scores passed to topk, N, n_valid),
the bias after every forward, the min top-3 margin, and prediction_outputs.

    ttq submit --model kairos --kind cpu -- /home/andy/bringup/venvs/kairos/bin/python -I \
        <worktree>/models/experimental/kairos_forecast/tests/golden/make_bench_goldens.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "golden"))
sys.path.insert(0, str(HERE.parents[2] / "reference"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from inputs import make_batch  # noqa: E402
from load_reference import load, restore_bias  # noqa: E402

OUT = Path("/home/andy/bringup/goldens/kairos/bench")
CASES = {  # name: (seed, n_series, horizon, lengths, gift)
    "paper": (1001, 256, 64, None, True),
    "long_horizon": (1002, 64, 720, None, True),
    "ragged": (1003, 128, 64, "spread", True),
    "single": (1004, 1, 64, None, True),
    "single_plain": (1004, 1, 64, None, False),
    "single_short": (1005, 1, 64, [300], True),
    "single_long_horizon": (1006, 1, 720, None, True),
}
LEVEL_TOKENS = torch.tensor([1, 2, 4])


class Capture:
    """Records every firing of the head and the gate (the flip fires each twice, the rollout once per step)."""

    def __init__(self, m):
        self.m, self.head, self.gate, self.bias, self.context, self.loc_scale = m, [], [], [], [], []
        self.scores = []
        enc = m.encode

        def encode(context, *a, **k):  # the step-k context, for the teacher-forced long_horizon scoring
            self.context.append(context.detach().clone())
            return enc(context, *a, **k)

        m.encode = encode
        m.instance_norm.register_forward_hook(lambda _m, i, o: self.loc_scale.append(_det(o[1])) if len(i) == 1 else None)
        m.patch.moe.gate.register_forward_pre_hook(self._scores)
        m.output_patch_embedding.register_forward_hook(lambda _m, _i, o: self.head.append(o.detach().clone()))
        m.patch.moe.gate.register_forward_hook(self._gate)

    def _gate(self, _mod, _inp, out):
        w, idx = out[0], out[1]
        self.gate.append((w.detach().clone(), idx.detach().clone()))
        self.bias.append(self.m.patch.moe.gate.bias.detach().clone())  # confirm: hook fires after the update

    def _scores(self, mod, args):
        # the selection scores passed to topk, recomputed with moe.py's exact ops on the pre-update bias
        x = torch.nan_to_num(args[0], nan=0.0)
        self.scores.append((torch.nn.functional.linear(x, mod.weight) + mod.bias).softmax(-1, dtype=torch.float32))

    def reset(self):
        self.head, self.gate, self.bias, self.context, self.loc_scale, self.scores = [], [], [], [], [], []


def _det(x):
    return tuple(v.detach().clone() for v in x) if isinstance(x, tuple) else x.detach().clone()


def token_counts(idx, context):
    """Per-row N (routed tokens, padding patches included) and n_valid (tokens of patches with any real value)."""
    rows = context.shape[0]
    lvl = torch.where(idx < 3, idx, -1).amax(-1).clamp(min=0).view(rows, -1)
    tok = LEVEL_TOKENS[lvl]
    valid = (~torch.isnan(context.view(rows, -1, 128))).any(-1)
    return tok.sum(1), (tok * valid).sum(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m, bias0 = load()
    cap = Capture(m)
    manifest = {}
    for name, (seed, n, h, lengths, gift) in CASES.items():
        rng = np.random.default_rng(seed + 7)
        lens = lengths if lengths is None or isinstance(lengths, list) else rng.integers(64, 2049, n)
        x, kinds = make_batch(seed, n, lens)
        restore_bias(m, bias0)
        cap.reset()
        with torch.no_grad():
            y = m(
                past_target=torch.from_numpy(x),
                prediction_length=h,
                generation=True,
                force_flip_invariance=gift,
                infer_is_positive=gift,
            )["prediction_outputs"]
        torch.save(
            {
                "context": torch.from_numpy(x),
                "kinds": kinds,
                "lengths": lens,
                "horizon": h,
                "bias0": bias0,
                "head": cap.head,
                "gate": cap.gate,
                "bias_after": cap.bias,
                "scores": cap.scores,
                "step_context": cap.context,
                "loc_scale": cap.loc_scale,
                "gift": gift,
                "prediction_outputs": y,
            },
            OUT / f"{name}.pt",
        )
        N, nv = token_counts(cap.gate[0][1], cap.context[0])
        margin = min(float((t.topk(4, -1).values[:, 2] - t.topk(4, -1).values[:, 3]).min()) for t in cap.scores)
        manifest[name] = {"seed": seed, "n_series": n, "horizon": h, "gift": gift, "forwards": len(cap.head),
                          "N_hist": torch.bincount(N, minlength=65).tolist(),
                          "n_valid_hist": torch.bincount(nv, minlength=65).tolist(), "min_top3_margin": margin}
        print(name, "done", manifest[name], flush=True)
    import load_reference as R

    (OUT / "cases.json").write_text(json.dumps({"cases": manifest, **R.manifest_fields(m)}, indent=1))


if __name__ == "__main__":
    main()
