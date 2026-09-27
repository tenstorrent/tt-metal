"""Does the streaming CFM change leave the NON-streaming path's timing unchanged (not just its numbers)?

Loads `tt/flow/decoder.py` as of BASE_COMMIT (pre-change) straight from git, next to the current one, and times
both on the same device with the same real `flow.pt` estimator weights, same inputs, alternating old/new blocks
so drift affects both equally. Non-streaming, all-ones mask (the only non-streaming case fused SDPA accepts):
  - eager 10-step solve (the pipeline default: COSYVOICE2_FLOW_CFM_TRACE=0)
  - traced 10-step solve, steady state (trace reused; capture excluded)
Also checks old vs new outputs are identical (same program, same inputs -> expected bit-exact).

Run (device) from its in-repo path (REPO is found as `parents[6]` of this file); python_env's .pth files put the
repo, ttnn/ and tools/ on sys.path, so no PYTHONPATH:
  HF_HOME=/home/user/models /home/user/tt-metal/python_env/bin/python /home/user/tt-metal/models/demos/audio/cosyvoice2/scripts/perf_2026_09_25/nonstreaming_timing_old_vs_new.py
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

import ttnn
from models.demos.audio.cosyvoice2.tt.checkpoint import load_checkpoint_file
from models.demos.audio.cosyvoice2.tt.flow import decoder as new_decoder
from models.demos.audio.cosyvoice2.tt.flow.flow import CausalMaskedDiffWithXvecRef

BASE_COMMIT = "5c65be445c"  # last commit before the streaming CFM change
REPO = Path(__file__).resolve().parents[6]
LENGTHS = [384, 768, 1536]
N_STEPS = 10
REPS = 4  # solves per block
BLOCKS = 2  # old/new alternations per mode


def load_old_decoder():
    src = subprocess.check_output(
        ["git", "-C", str(REPO), "show", f"{BASE_COMMIT}:models/demos/audio/cosyvoice2/tt/flow/decoder.py"]
    )
    path = Path(tempfile.mkdtemp()) / "decoder_old.py"
    path.write_bytes(src)
    name = "models.demos.audio.cosyvoice2.tt.flow.decoder_old"  # same package -> relative imports resolve
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def solve_ms(cfm, device, mu, mask, spks, cond, use_trace):
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    out = cfm.forward(mu, mask, N_STEPS, spks, cond, use_trace=use_trace)
    ttnn.synchronize_device(device)
    return out, (time.perf_counter() - t0) * 1e3


def main():
    old_decoder = load_old_decoder()
    ref = CausalMaskedDiffWithXvecRef.from_checkpoint(load_checkpoint_file("flow.pt"))
    ref.eval()
    est = ref.decoder.estimator

    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=200_000_000)
    rows = []
    try:
        impls = {
            "old": old_decoder.TtCausalConditionalCFM(
                device, old_decoder.TtCausalConditionalDecoder(device, est), ref.decoder.rand_noise, ref.decoder
            ),
            "new": new_decoder.TtCausalConditionalCFM(
                device, new_decoder.TtCausalConditionalDecoder(device, est), ref.decoder.rand_noise, ref.decoder
            ),
        }
        for t_len in LENGTHS:
            torch.manual_seed(t_len)
            mu, cond = torch.randn(1, t_len, 80), torch.randn(1, t_len, 80)
            spks, mask = torch.randn(1, 80), torch.ones(1, t_len, 1)
            for mode, use_trace in (("eager", False), ("traced", True)):
                times = {"old": [], "new": []}
                outs = {}
                for name in ("old", "new"):  # warm-up: kernel compile, conv caches, first capture
                    impls["new" if name == "old" else "old"].release_cfm_trace()
                    outs[name], _ = solve_ms(impls[name], device, mu, mask, spks, cond, use_trace)
                for _ in range(BLOCKS):
                    for name in ("old", "new"):
                        if use_trace:
                            # Only one implementation's trace resident at a time: release the other's first.
                            impls["new" if name == "old" else "old"].release_cfm_trace()
                            solve_ms(impls[name], device, mu, mask, spks, cond, True)  # (re)capture, not timed
                        for _ in range(REPS):
                            times[name].append(solve_ms(impls[name], device, mu, mask, spks, cond, use_trace)[1])
                for impl in impls.values():
                    impl.release_cfm_trace()
                same = torch.equal(outs["old"], outs["new"])
                rows.append((t_len, mode, times["old"], times["new"], same))
                print(f"done T={t_len} {mode}", flush=True)
    finally:
        ttnn.close_device(device)

    print("\n| T | mode | old mean (min-max) ms | new mean (min-max) ms | new/old | outputs bit-identical |")
    print("|---|---|---|---|---|---|")
    for t_len, mode, o, n, same in rows:
        mo, mn = sum(o) / len(o), sum(n) / len(n)
        print(
            f"| {t_len} | {mode} | {mo:.1f} ({min(o):.1f}-{max(o):.1f}) | {mn:.1f} ({min(n):.1f}-{max(n):.1f}) "
            f"| {mn / mo:.3f} | {same} |"
        )


if __name__ == "__main__":
    main()
