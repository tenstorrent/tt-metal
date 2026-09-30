"""t19: CPU conv crop decode vs the same region of the device conv mp4 (checks latent layout and crop mapping).
Usage: sanity.py <crop_dir> <mp4> t0 h0 w0. Crop frames 1..16 are pixel frames 8*t0+1..8*t0+16 (frame 0 is causal)."""
import subprocess, sys, numpy as np, torch

d, mp4 = sys.argv[1], sys.argv[2]
t0, h0, w0 = map(int, sys.argv[3:6])
conv = torch.load(f"{d}/conv.pt")  # T,H,W,3 uint8
T, Hh, Ww, _ = conv.shape
f0 = 8 * t0 + 1 if t0 else 0
raw = subprocess.run(
    ["ffmpeg", "-loglevel", "error", "-i", mp4, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
    capture_output=True,
    check=True,
).stdout
vid = np.frombuffer(raw, np.uint8).reshape(-1, 1088, 1920, 3)
cpu = conv[1:] if t0 else conv
dev = torch.from_numpy(vid[f0 : f0 + cpu.shape[0], h0 * 32 : h0 * 32 + Hh, w0 * 32 : w0 * 32 + Ww].copy())
a, b = cpu[:, 64:-64, 64:-64].float(), dev[:, 64:-64, 64:-64].float()
print(
    f"{d}: CPU conv vs device conv mp4 [interior]: PSNR {10 * np.log10(255**2 / ((a - b) ** 2).mean().item()):.2f} dB"
)
