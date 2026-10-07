# t148: CPU-only fast-motion quality check. ref = ref_dv145 (DiffVAE, veryfast/crf23 export),
# base = t170 baseline5 (conv VAE, ultrafast/crf20, same motion path as ref),
# fast = ref_t48_f6b8 (#171, current t48 defaults: conv VAE + gate/adaln fusion, ultrafast/crf20).
import json
import sys

import cv2
import numpy as np
from skimage.metrics import structural_similarity

P = "/home/smarton/fasth3/tt-metal/tt-project"
CLIPS = {
    "ref": P + "/baselines/ltx25_1080p_6s/ref_dv145/seed{}.mp4",
    "base": P + "/data/g15/t170/baseline5/seeds/seed{}.mp4",
    "fast": P + "/data/g15/ref_t48_f6b8/seed{}.mp4",
}
OUT = P + "/data/g15/t148"
TOPK = 5
FW = 480  # flow width


def load(path):
    cap = cv2.VideoCapture(path)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    return np.stack(fr)


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0**2 / mse)


def ssim(a, b):
    return structural_similarity(
        cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), data_range=255
    )


def small_gray(v):
    h = int(v.shape[1] * FW / v.shape[2])
    return [cv2.cvtColor(cv2.resize(f, (FW, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for f in v]


def flow_stats(v):
    """Per-frame mean flow magnitude (px at 1920 wide) and warp error (t-1 warped onto t, mean abs, 0-255)."""
    g = small_gray(v)
    scale = v.shape[2] / FW
    h, w = g[0].shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mag, werr = [0.0], [0.0]
    for t in range(1, len(g)):
        fl = cv2.calcOpticalFlowFarneback(g[t], g[t - 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag.append(float(np.mean(np.linalg.norm(fl, axis=2)) * scale))
        warped = cv2.remap(g[t - 1], gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR)
        werr.append(float(np.mean(np.abs(warped.astype(np.float32) - g[t].astype(np.float32)))))
    return np.array(mag), np.array(werr)


casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def face_box(f):
    s = 0.25
    sm = cv2.cvtColor(cv2.resize(f, None, fx=s, fy=s), cv2.COLOR_BGR2GRAY)
    d = casc.detectMultiScale(sm, 1.1, 5, minSize=(24, 24))
    if len(d) == 0:
        return None
    x, y, w, h = max(d, key=lambda r: r[2] * r[3]) / s
    cx, cy, side = x + w / 2, y + h / 2, max(w, h) * 1.8
    x0 = int(np.clip(cx - side / 2, 0, f.shape[1] - side))
    y0 = int(np.clip(cy - side / 2, 0, f.shape[0] - side))
    return x0, y0, int(side)


def label(img, txt):
    img = img.copy()
    cv2.putText(img, txt, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, txt, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    return img


res = {}
face_rows = []
for s in range(5):
    v = {k: load(p.format(s)) for k, p in CLIPS.items()}
    n = min(len(x) for x in v.values())
    fs = {k: flow_stats(x[:n]) for k, x in v.items()}
    mag = fs["ref"][0]
    order = np.argsort(mag[1:])[::-1] + 1
    top, low = sorted(order[:TOPK].tolist()), sorted(order[-TOPK:].tolist())
    r = {
        "frames": n,
        "top_motion_frames": top,
        "top_motion_px": [round(mag[t], 1) for t in top],
        "low_motion_frames": low,
        "low_motion_px": [round(mag[t], 2) for t in low],
    }
    for k in ("base", "fast"):
        pf = np.array([psnr(v[k][t], v["ref"][t]) for t in range(n)])
        r[k] = {
            "psnr_all_mean": round(float(pf.mean()), 2),
            "psnr_top": [round(float(pf[t]), 2) for t in top],
            "ssim_top": [round(float(ssim(v[k][t], v["ref"][t])), 4) for t in top],
            "psnr_low": [round(float(pf[t]), 2) for t in low],
            "ssim_low": [round(float(ssim(v[k][t], v["ref"][t])), 4) for t in low],
            "corr_motion_psnr": round(float(np.corrcoef(mag[1:], pf[1:])[0, 1]), 3),
        }
    for k in CLIPS:
        r[k + "_flow_mean_px"] = round(float(fs[k][0][1:].mean()), 2)
        r[k + "_warp_err_mean"] = round(float(fs[k][1][1:].mean()), 3)
        r[k + "_warp_err_top"] = round(float(np.mean([fs[k][1][t] for t in top])), 3)
    # face crop at the fastest-motion frame that has a detectable face in ref
    tf, box = None, None
    for t in order[:30]:
        box = face_box(v["ref"][t])
        if box:
            tf = int(t)
            break
    r["face_frame"], r["face_box"] = tf, box
    if box:
        x0, y0, side = box
        crops = [
            label(
                cv2.resize(v[k][tf][y0 : y0 + side, x0 : x0 + side], (400, 400), interpolation=cv2.INTER_AREA),
                f"s{s} f{tf} {k}",
            )
            for k in CLIPS
        ]
        face_rows.append(np.hstack(crops))
        # full-res crops for close inspection
        cv2.imwrite(
            f"{OUT}/face_s{s}_f{tf}_full.png", np.hstack([v[k][tf][y0 : y0 + side, x0 : x0 + side] for k in CLIPS])
        )
    # half-size full frame at the fastest-motion frame: ref | fast
    t0 = top[int(np.argmax([mag[t] for t in top]))]
    half = lambda f: cv2.resize(f, (960, 544), interpolation=cv2.INTER_AREA)
    cv2.imwrite(
        f"{OUT}/still_s{s}_f{t0}_ref_base_fast.jpg",
        np.hstack([label(half(v[k][t0]), f"{k} f{t0}") for k in CLIPS]),
        [cv2.IMWRITE_JPEG_QUALITY, 90],
    )
    res[s] = r
    print(s, json.dumps(r), flush=True)
    del v

if face_rows:
    cv2.imwrite(f"{OUT}/faces_ref_base_fast.png", np.vstack(face_rows))
json.dump(res, open(f"{OUT}/motion_report.json", "w"), indent=1)
