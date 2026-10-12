"""t421: per-seed on-vs-off comparison of the FastH3 precomputed-AdaLN A/B (runs on g15blx02, no device).

Pulls out_{off,on}-s<N>/run.log and the mp4 from blx01 /var/tmp/fasth3/t421 into mp4/, parses every
"Denoising (per step)" row (one per timed replay), computes ffmpeg PSNR/SSIM and RGB PCC of on vs off,
writes side-by-side stills (off left, on right) at 2.5 s, 4.5 s and the last frame, and prints a table
to summary.md.
"""

import re
import subprocess
import sys
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent
M = D / "mp4"
REMOTE = "g15blx01:/var/tmp/fasth3/t421"
MP4 = "fl2va_turbo_1344x768_5s_4fwd.mp4"
W, H = 1344, 768


def sh(*cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def fetch(tag):
    for name, local in ((f"out_{tag}/run.log", f"{tag}.log"), (f"out_{tag}/{MP4}", f"{tag}.mp4")):
        sh("scp", "-q", "-o", "BatchMode=yes", f"{REMOTE}/{name}", str(M / local))
    return (M / f"{tag}.log").exists(), (M / f"{tag}.mp4").exists()


def per_step(tag):
    log = M / f"{tag}.log"
    if not log.exists():
        return []
    return [float(x) for x in re.findall(r"Denoising \(per step\)\s*\|\s*([\d.]+)s", log.read_text())]


def frames(path):
    raw = sh_bytes("ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-")
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)


def sh_bytes(*cmd):
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def psnr_ssim(a, b):
    out = sh(
        "ffmpeg", "-v", "info", "-i", str(a), "-i", str(b), "-lavfi", "[0:v][1:v]psnr;[0:v][1:v]ssim", "-f", "null", "-"
    ).stderr
    psnr = re.search(r"PSNR .*average:([\d.inf]+) min:([\d.inf]+)", out)
    ssim = re.search(r"SSIM .*All:([\d.]+)", out)
    return (psnr.group(1), psnr.group(2)) if psnr else ("?", "?"), ssim.group(1) if ssim else "?"


def pcc(x, y):
    x = x.astype(np.float64).ravel()
    y = y.astype(np.float64).ravel()
    x -= x.mean()
    y -= y.mean()
    return float((x @ y) / np.sqrt((x @ x) * (y @ y)))


def still(off, on, t, out):
    sel = ["-ss", str(t)]
    sh(
        "ffmpeg",
        "-v",
        "error",
        "-y",
        *sel,
        "-i",
        str(off),
        *sel,
        "-i",
        str(on),
        "-lavfi",
        "[0:v][1:v]hstack",
        "-frames:v",
        "1",
        str(out),
    )


def main(seeds):
    M.mkdir(exist_ok=True)
    rows = []
    for s in seeds:
        off, on = f"off-s{s}", f"on-s{s}"
        fetch(off)
        fetch(on)
        t_off, t_on = per_step(off), per_step(on)
        row = {"seed": s, "off": t_off, "on": t_on}
        a, b = M / f"{off}.mp4", M / f"{on}.mp4"
        if a.exists() and b.exists():
            (row["psnr"], row["psnr_min"]), row["ssim"] = psnr_ssim(a, b)
            fa, fb = frames(a), frames(b)
            n = min(len(fa), len(fb))
            row["pcc"] = pcc(fa[:n], fb[:n])
            row["pcc_frames"] = [round(pcc(fa[i], fb[i]), 3) for i in (0, n // 2, n - 1)]
            for t, name in ((2.5, "2.5s"), (4.5, "4.5s"), (5.1, "last")):
                still(a, b, t, M / f"s{s}_off_vs_on_{name}.png")
        rows.append(row)

    lines = [
        "| seed | off s/step (replays) | on s/step (replays) | on vs off | PSNR avg/min dB | SSIM | RGB PCC | PCC f0/mid/last |",
        "|---|---|---|---|---|---|---|---|",
    ]
    all_off, all_on = [], []
    for r in rows:
        mo = np.mean(r["off"]) if r["off"] else float("nan")
        mn = np.mean(r["on"]) if r["on"] else float("nan")
        all_off += r["off"]
        all_on += r["on"]
        d = f"{(mn / mo - 1) * 100:+.1f}%" if r["off"] and r["on"] else "n/a"
        lines.append(
            f"| {r['seed']} | {', '.join(f'{x:.4f}' for x in r['off']) or 'n/a'} | {', '.join(f'{x:.4f}' for x in r['on']) or 'n/a'} "
            f"| {d} | {r.get('psnr', 'n/a')}/{r.get('psnr_min', 'n/a')} | {r.get('ssim', 'n/a')} "
            f"| {r['pcc']:.3f} | {r['pcc_frames']} |"
            if "pcc" in r
            else f"| {r['seed']} | {r['off']} | {r['on']} | {d} | n/a | n/a | n/a | n/a |"
        )
    if all_off and all_on:
        lines.append(
            f"\nAll replays: off mean {np.mean(all_off):.4f} s/step (n={len(all_off)}, sd {np.std(all_off):.4f}), "
            f"on mean {np.mean(all_on):.4f} s/step (n={len(all_on)}, sd {np.std(all_on):.4f}), "
            f"on vs off {(np.mean(all_on) / np.mean(all_off) - 1) * 100:+.1f}% (900 MHz clamp: relative only)"
        )
    text = "\n".join(lines) + "\n"
    (D / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main([int(x) for x in sys.argv[1:]] or [0, 1, 2, 3, 4])
