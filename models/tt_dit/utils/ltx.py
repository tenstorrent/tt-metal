# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

DEFAULT_LTX_PROMPT = (
    "A young woman with shoulder-length wavy brown hair sits on a wooden stool, "
    "cradling an acoustic guitar. The camera holds a steady medium close-up, "
    "framing her face and guitar neck. Warm key light illuminates her left side "
    "while soft fill light prevents harsh shadows. She strums gently, looking "
    "directly at camera with genuine warmth. Her mouth opens clearly as she sings "
    '"Doo-be-doo, doo-be-day, oh what a sunny day" with precise lip sync and '
    "natural facial expressions. Her head moves subtly with the rhythm. Simple "
    "chord progression underlies her melodic voice. Shot with 50mm lens at f/2.0, "
    "shallow depth of field, warm color grade emphasizing skin tones."
)


def default_ltx_checkpoint(filename: str) -> str:
    explicit = os.environ.get("LTX_CHECKPOINT")
    if explicit:
        return explicit
    local = os.path.expanduser(f"~/.cache/ltx-checkpoints/{filename}")
    if os.path.exists(local):
        return local
    return f"Lightricks/LTX-2.3:{filename}"


def default_ltx_gemma() -> str:
    return os.environ.get("GEMMA_PATH") or "google/gemma-3-12b-it-qat-q4_0-unquantized"


# Fast-mode env bundle: bf8 DiT-linear quant + step-cut denoise schedule. The single authority for
# both the pytest harness (via conftest) and the ltx_server worker, so the schedule can't drift
# between them. Validated at ~11.7s warm gen (bh_2x4sp1tp0): S1 8->6 steps (the 1.0->0.975 head is
# redundant within the structure basin), S2 3->1 (Stage-2 only refines an already-coherent latent,
# so one step holds composition byte-identical to the 3-step baseline). Cutting S2 is the dominant
# lever: Stage-2 costs ~3.15s/step vs Stage-1's ~0.72s/step.
FAST_QUANT = "all_bf8_lofi"
FAST_S1_SIGMAS = "1.0,0.9875,0.975,0.909375,0.725,0.421875,0.0"
FAST_S2_SIGMAS = "0.909375,0.0"


def apply_fast_env(env=None):
    """Expand LTX_FAST=1 into the concrete fast-mode vars; no-op unless LTX_FAST is truthy.

    setdefault, so an explicit var still wins. TT_DIT_HOST_WEIGHT_CACHE is read at layer import, so a
    caller wanting it must apply this before importing the pipeline (or set it in the process env).
    """
    env = os.environ if env is None else env
    if env.get("LTX_FAST", "0") not in ("1", "true", "True"):
        return
    env.setdefault("LTX_QUANT", FAST_QUANT)
    env.setdefault("LTX_S1_SIGMAS", FAST_S1_SIGMAS)
    env.setdefault("LTX_S2_SIGMAS", FAST_S2_SIGMAS)
    env.setdefault("LTX_TRACED", "1")
    env.setdefault("TT_DIT_HOST_WEIGHT_CACHE", "1")


def print_ltx_timing_table(
    pipeline, *, label, num_frames, height, width, mesh_shape, sp_axis, tp_axis, topology, output_path, prompt
):
    timings = getattr(pipeline, "last_timings", None)
    if not timings:
        return

    mesh = tuple(mesh_shape)
    topo = str(topology).split(".")[-1]
    prompt_short = prompt if len(prompt) <= 60 else prompt[:57] + "..."
    meta = [
        f"Resolution   {height}x{width} · {num_frames} frames",
        f"Mesh         {mesh} · sp={mesh[sp_axis]} tp={mesh[tp_axis]} · {topo}",
        f"Output       {output_path}",
        f"Prompt       {prompt_short}",
    ]
    rows = [(name, f"{secs:.2f} s") for name, secs in timings]
    rows.append(("Total", f"{sum(s for _, s in timings):.2f} s"))

    lw = max([len(n) for n, _ in rows] + [len("Stage")])
    rw = max([len(t) for _, t in rows] + [len("Time")])
    full = max(lw + rw + 5, max(len(m) for m in meta) + 1)
    lw = full - rw - 5

    out = ["", "┌" + "─" * full + "┐", "│" + f"{label} — PERFORMANCE".center(full) + "│"]
    for m in meta:
        out.append("│ " + m.ljust(full - 1) + "│")
    out.append("├" + "─" * (lw + 2) + "┬" + "─" * (rw + 2) + "┤")
    out.append("│ " + "Stage".ljust(lw) + " │ " + "Time".rjust(rw) + " │")
    out.append("├" + "─" * (lw + 2) + "┼" + "─" * (rw + 2) + "┤")
    for name, t in rows[:-1]:
        out.append("│ " + name.ljust(lw) + " │ " + t.rjust(rw) + " │")
    out.append("├" + "─" * (lw + 2) + "┼" + "─" * (rw + 2) + "┤")
    out.append("│ " + rows[-1][0].ljust(lw) + " │ " + rows[-1][1].rjust(rw) + " │")
    out.append("└" + "─" * (lw + 2) + "┴" + "─" * (rw + 2) + "┘")
    print("\n".join(out))
