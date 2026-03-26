#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Generate UI walkthrough video for the single-chip interactive demo.

Simulates the full user experience: model selection, task input,
camera angle switching, live metrics, and model swap mid-session.

Output: single_chip_ui_walkthrough.mp4 (~55s, 1280x720, 20 FPS)
"""

import os, sys, math, time
import numpy as np
import cv2
import imageio
import pybullet as p
import pybullet_data

sys.path.insert(0, os.environ.get("TT_METAL_HOME",
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from models.experimental.robotics_demo_n150.sim_env import FrankaCubeEnv

FPS = 20
W, H = 1280, 720
OUT = os.path.join(os.path.dirname(__file__), "single_chip_ui_walkthrough.mp4")
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Colours
BG       = (26, 26, 46)
SIDEBAR  = (16, 33, 62)
PANEL    = (13, 17, 23)
TEAL     = (0, 206, 201)
PURPLE   = (108, 92, 231)
GOLD     = (253, 203, 110)
GREEN    = (126, 231, 135)
WHITE    = (223, 230, 233)
GRAY     = (150, 155, 160)
DIM      = (80, 85, 90)
BORDER   = (40, 55, 80)
RED_D    = (60, 20, 20)
TEAL_D   = (0, 120, 115)

SB_W = 300
HDR_H = 55
VID_X = SB_W + 10
VID_Y = HDR_H + 8
MET_W = 180

CAMERAS = {
    "front-right": {"eye": [1.3, 0.6, 0.9], "target": [0.3, 0, 0.3]},
    "side":        {"eye": [0.2, 1.2, 0.8], "target": [0.3, 0, 0.3]},
    "overhead":    {"eye": [0.4, 0.0, 1.4], "target": [0.4, 0, 0.1]},
    "close-up":    {"eye": [0.7, 0.3, 0.4], "target": [0.5, 0, 0.1]},
}

# ── Helpers ───────────────────────────────────────────────────────────

def rrect(img, x, y, w, h, col, r=6):
    cv2.rectangle(img, (x+r,y), (x+w-r,y+h), col, -1)
    cv2.rectangle(img, (x,y+r), (x+w,y+h-r), col, -1)
    for cx,cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
        cv2.circle(img, (cx,cy), r, col, -1)

def txt(img, t, x, y, s=0.4, c=WHITE, th=1):
    cv2.putText(img, t, (x,y), FONT, s, c, th, cv2.LINE_AA)

def cursor(img, x, y, t):
    sz = 12 + int(2*math.sin(t*3))
    pts = np.array([[x,y],[x,y+sz],[x+sz//2,y+sz*2//3]], np.int32)
    cv2.fillPoly(img, [pts], WHITE)

def lerp(a, b, steps):
    out = []
    for i in range(steps):
        t = 0.5*(1-math.cos(i/(steps-1)*math.pi)) if steps > 1 else 1.0
        out.append((int(a[0]+(b[0]-a[0])*t), int(a[1]+(b[1]-a[1])*t)))
    return out


def draw_sidebar(img, model="pi0", task="pick up the cube", cam="front-right",
                 running=False):
    cv2.rectangle(img, (0,0), (SB_W,H), SIDEBAR, -1)
    cv2.line(img, (SB_W,0), (SB_W,H), BORDER, 1)

    txt(img, "Choose Your Model", 15, 30, 0.45, GRAY)

    # Model buttons
    for i, (name, label, clr) in enumerate([
        ("pi0", "PI0  |  2.3B  |  ~330ms", TEAL),
        ("smolvla", "SmolVLA  |  450M  |  ~229ms", PURPLE),
    ]):
        y = 48 + i * 42
        bg = clr if model == name else PANEL
        border = clr if model == name else BORDER
        rrect(img, 12, y, SB_W-24, 35, bg)
        cv2.rectangle(img, (12,y), (SB_W-12,y+35), border, 1)
        txt(img, label, 22, y+23, 0.38, WHITE if model == name else GRAY)

    txt(img, "Task Instruction", 15, 155, 0.45, GRAY)
    rrect(img, 12, 165, SB_W-24, 28, PANEL)
    cv2.rectangle(img, (12,165), (SB_W-12,193), BORDER, 1)
    txt(img, f'"{task}"', 20, 185, 0.35, TEAL)

    txt(img, "Camera View", 15, 220, 0.45, GRAY)
    cams = list(CAMERAS.keys())
    for i, c in enumerate(cams):
        bx = 12 + i * 68
        sel = (c == cam)
        rrect(img, bx, 235, 64, 22, TEAL_D if sel else PANEL)
        txt(img, c[:6], bx+4, 252, 0.28, WHITE if sel else DIM)

    txt(img, "Settings", 15, 285, 0.45, GRAY)
    for i, (label, val) in enumerate([("Steps:", "300"), ("Replan:", "5"), ("Velocity:", "0.5")]):
        y = 302 + i * 26
        rrect(img, 12, y, SB_W-24, 22, PANEL)
        txt(img, f"{label} {val}", 20, y+16, 0.33, WHITE)

    y = 395
    rrect(img, 12, y, SB_W//2-18, 30, (20,80,50))
    txt(img, "N150 Wormhole", 22, y+20, 0.35, GREEN)

    if running:
        rrect(img, 12, H-50, SB_W-24, 35, RED_D)
        txt(img, "RUNNING", 22, H-28, 0.4, GREEN, 2)
    else:
        rrect(img, 12, H-50, SB_W-24, 35, TEAL_D)
        txt(img, "Ready", 22, H-28, 0.4, WHITE)


def draw_header(img, model="pi0"):
    cv2.rectangle(img, (SB_W,0), (W,HDR_H), BG, -1)
    cv2.line(img, (SB_W,HDR_H), (W,HDR_H), BORDER, 1)
    txt(img, "Tenstorrent Robotics Demo", SB_W+15, 25, 0.6, TEAL, 2)
    txt(img, f"Single-chip  |  {model.upper()}", SB_W+15, 45, 0.35, GRAY)


def draw_metrics(img, m, x=None):
    if x is None:
        x = W - MET_W - 10
    y = VID_Y
    rrect(img, x, y, MET_W, H-HDR_H-20, PANEL, 8)
    cv2.rectangle(img, (x,y), (x+MET_W, H-12), BORDER, 1)

    txt(img, "Live Metrics", x+15, y+22, 0.4, TEAL, 1)
    metrics = [
        (f"{m.get('freq_hz',0):.1f} Hz", "Frequency", GREEN),
        (f"{m.get('inf_ms',0):.0f} ms", "Inference", TEAL),
        (f"{m.get('dist',0):.3f} m", "Distance", GOLD),
        (f"{m.get('step',0)}", "Step", WHITE),
    ]
    for i, (val, label, col) in enumerate(metrics):
        my = y + 50 + i * 75
        rrect(img, x+8, my, MET_W-16, 60, (20,30,50), 5)
        txt(img, val, x+20, my+28, 0.55, col, 2)
        txt(img, label, x+20, my+48, 0.3, DIM)

    # Mini chart placeholder
    cy = y + 360
    rrect(img, x+8, cy, MET_W-16, 80, (20,30,50), 5)
    txt(img, "Dist. History", x+15, cy+15, 0.3, GRAY)
    history = m.get("history", [])
    if len(history) > 2:
        pts = []
        for j, d in enumerate(history[-40:]):
            px = x + 15 + int(j * (MET_W-30) / 40)
            py = cy + 70 - int(min(d, 1.0) * 50)
            pts.append((px, py))
        for j in range(len(pts)-1):
            cv2.line(img, pts[j], pts[j+1], TEAL, 1, cv2.LINE_AA)


def draw_buttons(img, running):
    bx = VID_X
    by = H - 50
    w = 100
    if running:
        rrect(img, bx, by, w, 35, RED_D)
        txt(img, "Stop", bx+32, by+23, 0.45, (231,76,60))
    else:
        rrect(img, bx, by, w, 35, TEAL_D)
        txt(img, "Start", bx+26, by+23, 0.45, WHITE)
    rrect(img, bx+w+10, by, w, 35, PANEL)
    txt(img, "Reset", bx+w+35, by+23, 0.4, GRAY)


def render_sim_frame(env, cam_name, width, height):
    cam = CAMERAS[cam_name]
    view = p.computeViewMatrix(cameraEyePosition=cam["eye"],
                               cameraTargetPosition=cam["target"],
                               cameraUpVector=[0, 0, 1])
    proj = p.computeProjectionMatrixFOV(fov=60, aspect=width/height,
                                        nearVal=0.1, farVal=5.0)
    _, _, rgba, _, _ = p.getCameraImage(width=width, height=height,
                                        viewMatrix=view, projectionMatrix=proj,
                                        renderer=p.ER_TINY_RENDERER,
                                        physicsClientId=env.physics_client)
    return np.array(rgba[:,:,:3], dtype=np.uint8)


def compose_frame(model, task, cam, running, sim_img, metrics, cursor_pos=None, ct=0):
    img = np.full((H,W,3), BG, dtype=np.uint8)
    draw_sidebar(img, model=model, task=task, cam=cam, running=running)
    draw_header(img, model=model)

    met_x = W - MET_W - 10
    vid_w = met_x - VID_X - 5
    vid_h = H - HDR_H - 70

    if sim_img is not None:
        resized = cv2.resize(sim_img, (vid_w, vid_h))
        img[VID_Y:VID_Y+vid_h, VID_X:VID_X+vid_w] = resized
        cv2.rectangle(img, (VID_X,VID_Y), (VID_X+vid_w,VID_Y+vid_h), BORDER, 1)
        if running:
            prog = metrics.get("step",0) / max(metrics.get("total",300), 1)
            pw = int(vid_w * prog)
            cv2.rectangle(img, (VID_X,VID_Y+vid_h), (VID_X+pw,VID_Y+vid_h+3), TEAL, -1)
    else:
        rrect(img, VID_X, VID_Y, vid_w, vid_h, PANEL, 8)
        txt(img, "Select a model and click Start", VID_X+vid_w//2-130, VID_Y+vid_h//2, 0.5, DIM)

    draw_metrics(img, metrics, met_x)
    draw_buttons(img, running)

    if cursor_pos:
        cursor(img, cursor_pos[0], cursor_pos[1], ct)
    return img


# ── IK motion ─────────────────────────────────────────────────────────

def ik_motion(env, step, task="pick"):
    cube = env.get_cube_position()
    cx, cy, cz = cube
    phase = (step % 200) / 200.0
    if phase < 0.3:
        tgt = [cx, cy, cz + 0.18 - phase * 0.47]
    elif phase < 0.5:
        tgt = [cx, cy, cz + 0.04]
    elif phase < 0.8:
        tgt = [cx, cy, cz + 0.04 + (phase - 0.5) * 1.5]
    else:
        tgt = [0.3, 0.0, 0.55]
    ik = p.calculateInverseKinematics(env.robot_id, 11, tgt,
                                       physicsClientId=env.physics_client)
    return np.array(ik[:7])


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Generating Single-Chip UI Walkthrough Video")
    print("=" * 60)

    writer = imageio.get_writer(OUT, fps=FPS, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"])
    fc = [0]
    def w(f): writer.append_data(f); fc[0] += 1

    env = FrankaCubeEnv(image_size=224)
    env.reset()
    history = []

    # ── ACT 1: Dashboard appears (2s) ────────────────────────────
    print("[1/7] Dashboard loads...")
    for i in range(FPS*2):
        alpha = min(1.0, i/(FPS*1.0))
        f = compose_frame("pi0", "pick up the cube", "front-right", False, None, {})
        f = (f.astype(np.float32)*alpha).astype(np.uint8)
        w(f)

    # ── ACT 2: User selects PI0, types task, clicks Start (4s) ──
    print("[2/7] User picks PI0, enters task, clicks Start...")
    cpos = (640, 400)
    pi0_btn = (140, 65)
    task_box = (100, 185)
    start_btn = (VID_X+40, H-35)

    for pos in lerp(cpos, pi0_btn, FPS):
        w(compose_frame("pi0", "pick up the cube", "front-right", False, None, {}, pos, fc[0]*0.05))
    for _ in range(FPS//3):
        w(compose_frame("pi0", "pick up the cube", "front-right", False, None, {}, pi0_btn, fc[0]*0.05))

    for pos in lerp(pi0_btn, task_box, FPS//2):
        w(compose_frame("pi0", "pick up the cube", "front-right", False, None, {}, pos, fc[0]*0.05))
    tasks_typing = ["pick", "pick up", "pick up the", "pick up the cube"]
    for t_text in tasks_typing:
        for _ in range(FPS//4):
            w(compose_frame("pi0", t_text, "front-right", False, None, {}, task_box, fc[0]*0.05))

    for pos in lerp(task_box, start_btn, FPS//2):
        w(compose_frame("pi0", "pick up the cube", "front-right", False, None, {}, pos, fc[0]*0.05))
    for _ in range(FPS//4):
        w(compose_frame("pi0", "pick up the cube", "front-right", True, None, {}, start_btn, fc[0]*0.05))

    # ── ACT 3: PI0 running (10s) ─────────────────────────────────
    print("[3/7] PI0 running (front-right view)...")
    for step in range(FPS*10):
        joints = ik_motion(env, step)
        env.apply_actions(joints, use_delta=False, max_velocity=8.0)
        env.step()
        sim_img = render_sim_frame(env, "front-right", 780, 540)
        dist = env.get_distance_to_target()
        history.append(dist)
        base_ms = 330 + np.random.normal(0,3)
        m = {"freq_hz": round(1000/(base_ms/5),1), "inf_ms": round(base_ms), "dist": round(dist,3),
             "step": step+1, "total": FPS*10, "history": history}
        w(compose_frame("pi0", "pick up the cube", "front-right", True, sim_img, m))
        if step % (FPS*3) == 0:
            print(f"    step {step}")

    # ── ACT 4: Switch camera to overhead (3s) ────────────────────
    print("[4/7] User switches camera to overhead...")
    cam_btn_overhead = (148+2*68, 248)
    for pos in lerp((640,400), cam_btn_overhead, FPS//2):
        w(compose_frame("pi0", "pick up the cube", "front-right", True,
                         render_sim_frame(env, "front-right", 780, 540),
                         m, pos, fc[0]*0.05))
    for _ in range(FPS//4):
        w(compose_frame("pi0", "pick up the cube", "overhead", True,
                         render_sim_frame(env, "overhead", 780, 540),
                         m, cam_btn_overhead, fc[0]*0.05))

    for step in range(FPS*2):
        joints = ik_motion(env, FPS*10+step)
        env.apply_actions(joints, use_delta=False, max_velocity=8.0)
        env.step()
        sim_img = render_sim_frame(env, "overhead", 780, 540)
        dist = env.get_distance_to_target()
        history.append(dist)
        m["dist"] = round(dist,3); m["step"] = FPS*10+step+1; m["history"] = history
        w(compose_frame("pi0", "pick up the cube", "overhead", True, sim_img, m))

    # ── ACT 5: Switch to SmolVLA (3s) ────────────────────────────
    print("[5/7] User switches to SmolVLA...")
    env.reset()
    history = []

    smol_btn = (140, 107)
    for pos in lerp(cam_btn_overhead, smol_btn, FPS//2):
        w(compose_frame("pi0", "pick up the cube", "overhead", False, None, {}, pos, fc[0]*0.05))
    for _ in range(FPS//3):
        w(compose_frame("smolvla", "pick up the cube", "overhead", False, None, {}, smol_btn, fc[0]*0.05))

    cam_btn_close = (148+3*68, 248)
    for pos in lerp(smol_btn, cam_btn_close, FPS//3):
        w(compose_frame("smolvla", "pick up the cube", "overhead", False, None, {}, pos, fc[0]*0.05))
    for _ in range(FPS//4):
        w(compose_frame("smolvla", "pick up the cube", "close-up", False, None, {}, cam_btn_close, fc[0]*0.05))

    for pos in lerp(cam_btn_close, start_btn, FPS//3):
        w(compose_frame("smolvla", "pick up the cube", "close-up", False, None, {}, pos, fc[0]*0.05))
    for _ in range(FPS//4):
        w(compose_frame("smolvla", "pick up the cube", "close-up", True, None, {}, start_btn, fc[0]*0.05))

    # ── ACT 6: SmolVLA running close-up (10s) ────────────────────
    print("[6/7] SmolVLA running (close-up view)...")
    for step in range(FPS*10):
        joints = ik_motion(env, step)
        env.apply_actions(joints, use_delta=False, max_velocity=8.0)
        env.step()
        sim_img = render_sim_frame(env, "close-up", 780, 540)
        dist = env.get_distance_to_target()
        history.append(dist)
        base_ms = 229 + np.random.normal(0,3)
        m = {"freq_hz": round(1000/(base_ms/5),1), "inf_ms": round(base_ms), "dist": round(dist,3),
             "step": step+1, "total": FPS*10, "history": history}
        w(compose_frame("smolvla", "pick up the cube", "close-up", True, sim_img, m))
        if step % (FPS*3) == 0:
            print(f"    step {step}")

    # ── ACT 7: Complete (3s) ──────────────────────────────────────
    print("[7/7] Demo complete...")
    for _ in range(FPS*3):
        f = compose_frame("smolvla", "pick up the cube", "close-up", False, sim_img, m)
        txt(f, "Demo Complete!", VID_X + 250, H//2 - 20, 0.9, GREEN, 2)
        txt(f, "Switch models, tasks, and cameras at any time", VID_X + 180, H//2 + 20, 0.45, WHITE)
        w(f)

    env.close()
    writer.close()
    dur = fc[0]/FPS; sz = os.path.getsize(OUT)/(1024*1024)
    print(f"\n  Saved: {OUT}")
    print(f"  {fc[0]} frames, {dur:.1f}s, {sz:.1f} MB\n")


if __name__ == "__main__":
    main()
