# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Small test goldens with per-block intermediates (CONTRACT §6 A). Reference venv, `ttq run --kind cpu`.

Every case saves its inputs, bias0, every capture firing (in call order) and the outputs, as one .pt file.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "reference"))

OUT = Path("/home/andy/bringup/goldens/kairos/test")

# name: what it exercises. Bias restored before every call except "bias_5call".
CASES = {
    "single": "1 x 2048 -> 64, GIFT settings, every intermediate",
    "single_plain": "1 x 2048 -> 64, no flip (the doc's workload)",
    "ragged4": "lengths 2048, 700, 130, 2048 (interior NaN gap): per-row N, padding, REG position",
    "forced_n16": "all level 0",
    "forced_n32": "all level 1",
    "forced_n64": "all level 2",
    "forced_mixed": "1, 2 and 3 real experts in the top-3",
    "rollout2": "1 x 2048 -> 256, two forwards, context per step",
    "bias_5call": "5 consecutive unrestored calls: bias trajectory",
    "front_end_long": "5000-step input truncated to its last 2048",
    "modules": "small random-input module goldens",
}

# Capture points (CONTRACT §5), every firing recorded.
CAPTURE = [
    "fft_norm",
    "patch.moe.gate (weights, indices, selection scores passed to topk, bias before/after)",
    "patch (patched_x, patched_mask, size, weights, indices, x_final, parent/position maps, granularity_mask)",
    "input_patch_embedding",
    "rotary_emb.instance_wise_param_net x18 (gamma, beta)",
    "rotated q/k, block 0 of each stack (and decoder self-attn block 1: D0 amendment 3)",
    "encoder.block[i], encoder.final_layer_norm",
    "decoder.block[i], decoder.final_layer_norm",
    "output_patch_embedding",
    "loc, scale per call",
    "prediction_outputs",
]


TOP = ("fft_norm", "patch.moe.gate", "patch", "instance_norm", "input_patch_embedding", "encoder.final_layer_norm",
       "decoder.final_layer_norm", "output_patch_embedding")


def _wanted(name):
    if name in TOP or name.endswith("instance_wise_param_net"):
        return True
    parts = name.split(".")
    return len(parts) == 3 and parts[0] in ("encoder", "decoder") and parts[1] == "block"


def _det(x):
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, (tuple, list)):
        return type(x)(_det(v) for v in x)
    if isinstance(x, dict):
        return {k: _det(v) for k, v in x.items()}
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return f"<{type(x).__name__}>"  # tensors and plain values only, so python_env can load the goldens


class Capture:
    """Forward hooks on the capture points; `firings` is a list of (name, inputs, kwargs, output) in call order."""

    def __init__(self, m, block_inputs=True):
        import torch.nn.functional as F

        self.firings, self.gate = [], []
        self.handles = []
        for name, mod in m.named_modules():
            if not _wanted(name):
                continue
            keep_in = block_inputs and (".block." in name or name in ("encoder.final_layer_norm",))

            def hook(mod, args, kwargs, out, name=name, keep_in=keep_in):
                self.firings.append((name, _det(args) if keep_in else None, _det(kwargs) if keep_in else None, _det(out)))

            self.handles.append(mod.register_forward_hook(hook, with_kwargs=True))
        gate = m.patch.moe.gate

        def gate_pre(mod, args):
            import torch

            x = args[0]
            scores = (F.linear(torch.nan_to_num(x, nan=0.0), mod.weight) + mod.bias).softmax(-1, dtype=torch.float32)
            self.gate.append({"x": _det(x), "bias_before": mod.bias.detach().clone(), "scores": scores.detach()})

        def gate_post(mod, args, out):
            self.gate[-1].update(weights=_det(out[0]), indices=_det(out[1]), bias_after=mod.bias.detach().clone())

        self.handles += [gate.register_forward_pre_hook(gate_pre), gate.register_forward_hook(gate_post)]

    def close(self):
        for h in self.handles:
            h.remove()


GIFT = dict(force_flip_invariance=True, infer_is_positive=True)


def build_cases(np):
    from inputs import CONTEXT, make_batch, make_series, pad_or_truncate

    rng = np.random.default_rng(1234)
    single, _ = make_batch(1, 1)
    r = []
    for n, kind in ((2048, "rw_seasonal"), (700, "intermittent"), (130, "level_shift"), (2048, "multi_seasonal")):
        r.append(pad_or_truncate(make_series(rng, n, kind)))
    r[3][900:960] = np.nan  # interior NaN gap
    long = make_series(rng, 5000, "multi_seasonal")
    # name: (x [B, T], prediction_length, generate kwargs, forced bias or None, calls without restore)
    return {
        "single": (single, 64, {}, None, 1),
        "single_gift": (single, 64, GIFT, None, 1),
        "ragged4": (np.stack(r), 64, GIFT, None, 1),
        "forced_n16": (single, 64, {}, (0, 3, 4), 1),
        "forced_n32": (single, 64, {}, (1, 3, 4), 1),
        "forced_n64": (single, 64, {}, (2, 3, 4), 1),
        "forced_mixed": (make_batch(7, 16)[0], 64, {}, {2: -1.0}, 1),  # levels 19/139/98 of 256 patches
        "rollout2": (single, 256, {}, None, 1),
        "bias_5call": (make_batch(5, 4)[0], 64, GIFT, None, 5),
        "front_end_long": (long[None, :], 64, {}, None, 1),
    }


def main():
    import numpy as np
    import torch
    import load_reference as R

    OUT.mkdir(parents=True, exist_ok=True)
    m, bias0 = R.load()
    manifest = {"cases": {}, **R.manifest_fields(m), "bias0": bias0.tolist()}
    for name, (x, h, kw, forced, calls) in build_cases(np).items():
        b0 = bias0.clone()
        if isinstance(forced, dict):  # bias offsets that give a mix of levels (mixed_search.py)
            for e, d in forced.items():
                b0[e] += d
        elif forced is not None:  # +50 on the chosen experts: every patch picks exactly that top-3
            b0[list(forced)] += 50.0
        R.restore_bias(m, b0)
        cap = Capture(m)
        xt = torch.tensor(x, dtype=torch.float32)
        outs = []
        with torch.no_grad():
            for _ in range(calls):
                outs.append(m(past_target=xt, prediction_length=h, generation=True, **kw)["prediction_outputs"])
        cap.close()
        rec = {"x": xt, "prediction_length": h, "kwargs": kw, "bias0": b0, "calls": calls, "outputs": outs,
               "gate": cap.gate, "firings": cap.firings, "bias_final": m.patch.moe.gate.bias.detach().clone()}
        torch.save(rec, OUT / f"{name}.pt")
        manifest["cases"][name] = {"shape": list(x.shape), "h": h, "kwargs": kw, "forced": str(forced), "calls": calls,
                                   "gate_calls": len(cap.gate), "firings": len(cap.firings),
                                   "out_shape": list(outs[0].shape)}
        print(name, list(outs[0].shape), "gate calls", len(cap.gate), "firings", len(cap.firings), flush=True)
    R.restore_bias(m, bias0)
    (OUT / "MANIFEST.json").write_text(__import__("json").dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
