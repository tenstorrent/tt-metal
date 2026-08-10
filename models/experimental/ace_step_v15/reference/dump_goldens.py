# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Golden-tensor generator for the ACE-Step 1.5 TTNN bringup — Block 0.

Writes `golden/<block>/s<S>/<name>.pt` (`torch.save`, fp32) from the fp32 CPU
`diffusers` reference at seed 1234, for the four reference durations of
ACE_STEP_1_5_BRINGUP.md §5b:

    2.56 s -> S=32 (unit) · 10.24 s -> S=128 (block)
    20.48 s -> S=256 (mask; also exactly one 512-frame VAE decode chunk)
    61.44 s -> S=768 (end-to-end)

Both the *inputs* and the *outputs* of every hooked stage are dumped, so a downstream
block can drive a submodule in isolation without re-running the reference.

Blocks:
    dit       Block 1 — DiT: whole-model in/out, proj_in/proj_out, time embeds,
              condition_embedder, norm_out, all 24 layer boundaries, plus full
              intra-layer detail for layer 0 (sliding_attention) and layer 1
              (full_attention) at 2.56/10.24 s, layer 0 only at 20.48/61.44 s
              (those intermediates are 6-19 MB apiece).
    cond      Block 2 — condition encoder: text_projector / lyric_encoder /
              timbre_encoder in+out, plus the packed encoder_hidden_states.
    vae       Block 3 — Oobleck VAE decoder: conv1 / block.0..4 (+ per-block
              snake1 / conv_t1 / res_unit1..3) / snake1 / conv2.
    solver    Block 4 — every denoising step's (x_t, t) -> velocity.
    pipeline  Block 4 — final waveform, per-step latents, timestep schedule.

Usage:
    python models/experimental/ace_step_v15/reference/dump_goldens.py
    python .../dump_goldens.py --durations 2.56 --blocks dit cond
    python .../dump_goldens.py --verify          # re-dump to a temp dir, assert bit-exact
    python .../dump_goldens.py --inventory       # rewrite golden/INVENTORY.txt, dump nothing

Host-only: never imports ttnn, never opens a device.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from models.experimental.ace_step_v15.reference import ace_step_ref as ref  # noqa: E402

GOLDEN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "golden"))

ALL_BLOCKS = ("dit", "cond", "vae", "solver", "pipeline")

# VAE decoder intermediates are enormous — block.4's output is 128 ch x 1920*T samples,
# i.e. 503 MB fp32 at T=512. So the VAE goldens are duration-gated (see
# `ace_step_ref.vae_specs` for what each mode covers):
#   detail    2.56 s only (T=64)
#   boundary  (none by default — the 2.56 s detail dump already contains it)
#   io        10.24 s and 20.48 s: whole-decoder in/out. 20.48 s is exactly T=512,
#             one default VAE decode chunk, which is the shape chunked decode must match.
# At 61.44 s (T=1536) even a boundary dump is >3 GB, so the VAE is skipped entirely there;
# the waveform for that duration lives in the `pipeline` block.
VAE_DETAIL_DURATIONS = (2.56,)
VAE_BOUNDARY_DURATIONS = ()
VAE_IO_DURATIONS = (10.24, 20.48)


# --------------------------------------------------------------------------------------
# spec assembly
# --------------------------------------------------------------------------------------


def _block_specs(pipe, block: str, duration: float):
    """Return the HookSpec list for `block` at `duration`, or [] if not applicable."""
    if block == "dit":
        return ref.dit_specs(pipe, duration=duration)
    if block == "cond":
        return ref.cond_specs(pipe)
    if block == "solver":
        return ref.solver_specs(pipe)
    if block == "vae":
        if duration in VAE_DETAIL_DURATIONS:
            return ref.vae_specs(pipe, mode="detail")
        if duration in VAE_BOUNDARY_DURATIONS:
            return ref.vae_specs(pipe, mode="boundary")
        if duration in VAE_IO_DURATIONS:
            return ref.vae_specs(pipe, mode="io")
        return []
    if block == "pipeline":
        return []  # assembled from the run result, not from hooks
    raise ValueError(f"unknown block {block!r}")


_BLOCK_PREFIXES = {
    "dit": ("transformer.",),
    "cond": ("condition_encoder.", "qwen3_text_encoder."),
    "vae": ("vae.",),
    "solver": ("solver.",),
}


def _route(block: str, name: str) -> bool:
    """True if capture key `name` belongs to `block`. Prefix routing keeps the
    `transformer.*` keys out of the solver block and vice versa."""
    return any(name.startswith(p) for p in _BLOCK_PREFIXES[block])


# --------------------------------------------------------------------------------------
# dumping
# --------------------------------------------------------------------------------------


def _pipeline_tensors(result: dict) -> dict:
    """Non-hook goldens for the `pipeline` block."""
    out = {
        "audio": result["audio"].to(torch.float32),
        "timesteps": result["timesteps"],
        "final_latents": result["step_latents"][-1].to(torch.float32),
    }
    for i, lat in enumerate(result["step_latents"]):
        out[f"step_latents.call{i}"] = lat.to(torch.float32)
    return out


def _meta(result: dict, pipe) -> dict:
    tc = pipe.transformer.config
    vc = pipe.vae.config
    return {
        "seed": result["seed"],
        "duration": result["duration"],
        "latent_frames_T": result["latent_frames"],
        "dit_tokens_S": result["dit_tokens"],
        "prompt": ref.GOLDEN_PROMPT,
        "lyrics": ref.GOLDEN_LYRICS,
        "call_kwargs": result["call_kwargs"],
        "timesteps": result["timesteps"].tolist(),
        "transformer_config": dict(tc),
        "vae_config": dict(vc),
        "latents_per_second": ref.LATENTS_PER_SECOND,
        "diffusers_version": __import__("diffusers").__version__,
        "torch_version": torch.__version__,
    }


def dump(
    out_root: str,
    durations,
    blocks,
    seed: int = ref.GOLDEN_SEED,
    pipeline_path: str | None = None,
    pipe=None,
    quiet: bool = False,
) -> dict:
    """Run the reference once per duration and write every requested block's goldens.

    Returns `{(block, S): {name: (shape, dtype)}}` — the inventory.
    """
    pipe = pipe if pipe is not None else ref.load_pipeline(pipeline_path)
    inventory: dict = {}

    for duration in durations:
        S = ref.dit_tokens_for_duration(duration)
        specs = []
        for block in blocks:
            specs += _block_specs(pipe, block, duration)
        if not quiet:
            print(
                f"[goldens] duration={duration}s  T={ref.latent_frames_for_duration(duration)}  S={S}  "
                f"hooks={len(specs)}"
            )

        result, cap = ref.run_with_capture(pipe, duration, specs, seed=seed)

        for block in blocks:
            if block == "pipeline":
                tensors = _pipeline_tensors(result)
            else:
                tensors = {name: t for name, t in cap.tensors.items() if _route(block, name)}
            if not tensors:
                continue
            bdir = os.path.join(out_root, block, f"s{S}")
            os.makedirs(bdir, exist_ok=True)
            for name, t in tensors.items():
                torch.save(t, os.path.join(bdir, f"{name}.pt"))
            torch.save(_meta(result, pipe), os.path.join(bdir, "meta.pt"))
            inventory[(block, S)] = {n: (tuple(t.shape), str(t.dtype)) for n, t in tensors.items()}
            nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
            if not quiet:
                print(f"[goldens]   {block}/s{S}: {len(tensors)} tensors, {nbytes / 2**20:.1f} MiB")

        del cap, result

    return inventory


# --------------------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------------------


def _bitwise_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-exact tensor comparison. Uses a raw byte view rather than `torch.equal` so
    that NaN == NaN and -0.0 != 0.0 (both matter for a reproducibility gate)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    av = a.detach().contiguous().view(torch.uint8) if a.numel() else a
    bv = b.detach().contiguous().view(torch.uint8) if b.numel() else b
    return bool(torch.equal(av, bv))


def verify(out_root: str, durations, blocks, seed: int = ref.GOLDEN_SEED, pipeline_path: str | None = None) -> int:
    """Re-dump to a temp dir and assert bit-exact equality with what is on disk.

    Returns the number of mismatches (0 == pass). Missing/extra files count as
    mismatches, so this also catches a stale golden dir.
    """
    tmp = tempfile.mkdtemp(prefix="ace_step_goldens_verify_")
    failures = 0
    try:
        print(f"[verify] re-dumping to {tmp}")
        dump(tmp, durations, blocks, seed=seed, pipeline_path=pipeline_path, quiet=False)
        for block in blocks:
            for duration in durations:
                S = ref.dit_tokens_for_duration(duration)
                rel = os.path.join(block, f"s{S}")
                new_dir, old_dir = os.path.join(tmp, rel), os.path.join(out_root, rel)
                if not os.path.isdir(new_dir):
                    continue
                if not os.path.isdir(old_dir):
                    print(f"[verify] MISSING on disk: {rel}")
                    failures += 1
                    continue
                new_files = {f for f in os.listdir(new_dir) if f.endswith(".pt")}
                old_files = {f for f in os.listdir(old_dir) if f.endswith(".pt")}
                for f in sorted(new_files - old_files):
                    print(f"[verify] MISSING on disk: {rel}/{f}")
                    failures += 1
                for f in sorted(old_files - new_files):
                    print(f"[verify] EXTRA on disk (stale?): {rel}/{f}")
                    failures += 1
                for f in sorted(new_files & old_files):
                    if f == "meta.pt":
                        continue  # dict of scalars/strings, compared below
                    a = torch.load(os.path.join(old_dir, f), map_location="cpu", weights_only=True)
                    b = torch.load(os.path.join(new_dir, f), map_location="cpu", weights_only=True)
                    if not _bitwise_equal(a, b):
                        d = (a.float() - b.float()).abs().max().item() if a.shape == b.shape else float("nan")
                        print(f"[verify] MISMATCH {rel}/{f}  max|diff|={d}")
                        failures += 1
                if "meta.pt" in new_files & old_files:
                    a = torch.load(os.path.join(old_dir, "meta.pt"), map_location="cpu", weights_only=False)
                    b = torch.load(os.path.join(new_dir, "meta.pt"), map_location="cpu", weights_only=False)
                    if a != b:
                        print(f"[verify] MISMATCH {rel}/meta.pt")
                        failures += 1
                print(f"[verify] {rel}: {len(new_files & old_files)} files compared")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[verify] BIT-EXACT" if failures == 0 else f"[verify] {failures} MISMATCH(ES)")
    return failures


# --------------------------------------------------------------------------------------
# inventory
# --------------------------------------------------------------------------------------


def inventory_on_disk(out_root: str) -> str:
    lines = []
    total = 0
    for block in ALL_BLOCKS:
        bdir = os.path.join(out_root, block)
        if not os.path.isdir(bdir):
            continue
        for sdir in sorted(os.listdir(bdir), key=lambda s: int(s[1:]) if s[1:].isdigit() else 0):
            d = os.path.join(bdir, sdir)
            if not os.path.isdir(d):
                continue
            files = sorted(f for f in os.listdir(d) if f.endswith(".pt") and f != "meta.pt")
            nbytes = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d))
            total += nbytes
            lines.append(f"\n### {block}/{sdir}  ({len(files)} tensors, {nbytes / 2**20:.1f} MiB)")
            for f in files:
                t = torch.load(os.path.join(d, f), map_location="cpu", weights_only=True)
                name = f[: -len(".pt")]
                lines.append(f"  {name:80s} {str(tuple(t.shape)):26s} {str(t.dtype).replace('torch.', '')}")
    lines.append(f"\nTOTAL {total / 2**20:.1f} MiB")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=GOLDEN_ROOT, help="golden root (default: models/.../ace_step_v15/golden)")
    ap.add_argument("--pipeline", default=None, help="diffusers-format ACE-Step 1.5 dir ($ACE_STEP_PIPELINE)")
    ap.add_argument("--durations", type=float, nargs="+", default=list(ref.REFERENCE_DURATIONS))
    ap.add_argument("--blocks", nargs="+", default=list(ALL_BLOCKS), choices=list(ALL_BLOCKS))
    ap.add_argument("--seed", type=int, default=ref.GOLDEN_SEED)
    ap.add_argument("--verify", action="store_true", help="re-dump to a temp dir, assert bit-exact vs --out")
    ap.add_argument("--inventory", action="store_true", help="print the on-disk inventory and exit")
    args = ap.parse_args()

    for d in args.durations:
        S = ref.dit_tokens_for_duration(d)
        if S % 32 != 0:
            print(f"[goldens] WARNING duration {d}s gives S={S}, not tile-aligned (expected 32*k)")

    def write_inventory(note: str) -> None:
        inv_path = os.path.join(args.out, "INVENTORY.txt")
        body = inventory_on_disk(args.out)
        with open(inv_path, "w") as f:
            f.write("# ACE-Step 1.5 golden inventory — reference/dump_goldens.py\n")
            f.write("# Scan of everything currently on disk, not just the last run's --durations.\n")
            f.write(f"# {note}\n")
            f.write(body + "\n")
        print(body)
        print(f"[goldens] inventory -> {inv_path}")

    if args.inventory:
        write_inventory("regenerated by --inventory (no dump)")
        return 0

    if args.verify:
        return 1 if verify(args.out, args.durations, args.blocks, args.seed, args.pipeline) else 0

    inv = dump(args.out, args.durations, args.blocks, seed=args.seed, pipeline_path=args.pipeline)
    n = sum(len(v) for v in inv.values())
    print(f"[goldens] wrote {n} tensors under {args.out}")
    write_inventory(f"last dump: seed={args.seed} durations={args.durations} blocks={args.blocks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
