#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Generate a UI walkthrough video simulating the live Streamlit dashboard.

Shows the full user experience: dashboard loads, user selects scenarios,
simulation runs live inside the UI frame with metrics updating in real time.

Output: robotics_demo_ui_walkthrough.mp4 (~90s, 1280x720, 20 FPS)
"""

import os, sys, math, time
import numpy as np

sys.path.insert(0, os.environ.get("TT_METAL_HOME",
    os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import cv2
import imageio
import pybullet as p
import pybullet_data

from models.experimental.robotics_demo.multi_env import MultiEnvironment
from models.experimental.robotics_demo.video_composer import compose_quad_view, compose_side_by_side

FPS = 20
W, H = 1280, 720
OUT_PATH = os.path.join(os.path.dirname(__file__), "robotics_demo_ui_walkthrough.mp4")

# ── Colour palette ────────────────────────────────────────────────────
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
RED_SOFT = (231, 76, 60)
BORDER   = (40, 55, 80)

FONT = cv2.FONT_HERSHEY_SIMPLEX
SIDEBAR_W = 280
HEADER_H = 50
EE_LINK = 11

# ── IK helpers (reuse from generate_demo_video) ──────────────────────

def solve_ik(phys, robot, link, pos):
    ik = p.calculateInverseKinematics(robot, link, pos, physicsClientId=phys)
    return np.array(ik[:7])

def build_pick_waypoints(phys, robot, cube_pos):
    cx, cy, cz = cube_pos
    pts = [
        ([0.3, 0.0, 0.55], 15),
        ([cx, cy, cz+0.18], 25),
        ([cx, cy, cz+0.04], 25),
        ([cx, cy, cz+0.04], 15),
        ([cx, cy, cz+0.40], 30),
        ([cx, cy, cz+0.50], 20),
        ([0.3, 0.0, 0.55], 20),
    ]
    return [(solve_ik(phys, robot, EE_LINK, xyz), h) for xyz, h in pts]

def build_push_waypoints(phys, robot, cube_pos):
    cx, cy, cz = cube_pos
    pts = [
        ([0.3, 0.0, 0.55], 15),
        ([cx-0.15, cy, cz+0.06], 30),
        ([cx, cy, cz+0.04], 20),
        ([cx+0.18, cy, cz+0.06], 35),
        ([cx+0.18, cy, cz+0.25], 20),
        ([0.3, 0.0, 0.55], 20),
    ]
    return [(solve_ik(phys, robot, EE_LINK, xyz), h) for xyz, h in pts]

def build_lift_waypoints(phys, robot, cube_pos):
    cx, cy, cz = cube_pos
    pts = [
        ([0.3, 0.0, 0.55], 15),
        ([cx, cy, cz+0.18], 25),
        ([cx, cy, cz+0.04], 25),
        ([cx, cy, cz+0.04], 10),
        ([cx, cy, 0.50], 35),
        ([cx, cy, 0.55], 20),
        ([0.3, 0.0, 0.55], 20),
    ]
    return [(solve_ik(phys, robot, EE_LINK, xyz), h) for xyz, h in pts]

def build_reach_waypoints(phys, robot, cube_pos):
    cx, cy, cz = cube_pos
    pts = [
        ([0.3, 0.0, 0.55], 15),
        ([cx+0.1, cy+0.1, cz+0.15], 30),
        ([cx, cy, cz+0.18], 20),
        ([cx-0.1, cy-0.1, cz+0.15], 30),
        ([cx, cy, cz+0.04], 20),
        ([cx, cy, cz+0.18], 15),
        ([0.3, 0.0, 0.55], 20),
    ]
    return [(solve_ik(phys, robot, EE_LINK, xyz), h) for xyz, h in pts]

WP_BUILDERS = {
    "pick cube": build_pick_waypoints,
    "push block": build_push_waypoints,
    "lift object": build_lift_waypoints,
    "reach target": build_reach_waypoints,
}

def waypoints_to_traj(wps):
    traj = []
    for i, (jt, hold) in enumerate(wps):
        if i == 0:
            for _ in range(hold):
                traj.append(jt.copy())
        else:
            prev = wps[i-1][0]
            for f in range(hold):
                a = 0.5 * (1 - math.cos(f / hold * math.pi))
                traj.append((1 - a) * prev + a * jt)
    return traj


# ── UI drawing primitives ─────────────────────────────────────────────

def draw_rounded_rect(img, x, y, w, h, color, radius=8):
    cv2.rectangle(img, (x+radius, y), (x+w-radius, y+h), color, -1)
    cv2.rectangle(img, (x, y+radius), (x+w, y+h-radius), color, -1)
    for cx_, cy_ in [(x+radius, y+radius), (x+w-radius, y+radius),
                      (x+radius, y+h-radius), (x+w-radius, y+h-radius)]:
        cv2.circle(img, (cx_, cy_), radius, color, -1)

def text(img, txt, x, y, scale=0.45, color=WHITE, thick=1):
    cv2.putText(img, txt, (x, y), FONT, scale, color, thick, cv2.LINE_AA)

def draw_sidebar(img, selected_scenario=0, hw_chips=4, steps=400,
                 replan=5, task="pick up the cube", fusion="weighted_average"):
    cv2.rectangle(img, (0, 0), (SIDEBAR_W, H), SIDEBAR, -1)
    cv2.line(img, (SIDEBAR_W, 0), (SIDEBAR_W, H), BORDER, 1)

    # Hardware status
    draw_rounded_rect(img, 12, 12, SIDEBAR_W-24, 38, (20, 80, 50), 6)
    text(img, f"{hw_chips}x Blackhole Detected", 22, 38, 0.45, GREEN, 1)

    # Scenario selector
    text(img, "Scenario Selection", 15, 80, 0.42, GRAY)
    scenarios = [
        "S1: Data-Parallel PI0",
        "S2: PI0 vs SmolVLA",
        "S3: Ensemble Pipeline",
        "S4: Scaling Benchmark",
    ]
    for i, s in enumerate(scenarios):
        y_ = 98 + i * 28
        if i == selected_scenario:
            cv2.circle(img, (25, y_ + 2), 6, TEAL, -1)
            text(img, s, 38, y_ + 6, 0.38, TEAL, 1)
        else:
            cv2.circle(img, (25, y_ + 2), 6, DIM, 1)
            text(img, s, 38, y_ + 6, 0.38, GRAY, 1)

    # Configuration
    text(img, "Configuration", 15, 230, 0.42, GRAY)
    fields = [
        (f"Steps: {steps}", 248),
        (f"Replan interval: {replan}", 272),
        (f"Task: {task}", 296),
        (f"Chips: {hw_chips}", 320),
    ]
    for label, y_ in fields:
        draw_rounded_rect(img, 15, y_ - 14, SIDEBAR_W - 30, 22, PANEL, 4)
        text(img, label, 22, y_ + 1, 0.35, WHITE)

    if selected_scenario == 2:
        text(img, "Fusion Settings", 15, 360, 0.42, GRAY)
        draw_rounded_rect(img, 15, 374, SIDEBAR_W - 30, 22, PANEL, 4)
        text(img, f"Strategy: {fusion}", 22, 389, 0.35, WHITE)
        # Alpha slider
        draw_rounded_rect(img, 15, 402, SIDEBAR_W - 30, 18, PANEL, 4)
        slider_x = 15 + int((SIDEBAR_W - 30) * 0.6)
        cv2.circle(img, (slider_x, 411), 6, TEAL, -1)
        text(img, "Alpha: 0.6", 22, 415, 0.33, WHITE)


def draw_header(img):
    cv2.rectangle(img, (SIDEBAR_W, 0), (W, HEADER_H), BG, -1)
    text(img, "Tenstorrent Robotics Intelligence Suite", SIDEBAR_W + 15, 33, 0.65, TEAL, 2)
    cv2.line(img, (SIDEBAR_W, HEADER_H), (W, HEADER_H), BORDER, 1)


def draw_metric_cards(img, cards, y_base=None):
    """Draw 4 metric cards at the bottom of the main area."""
    if y_base is None:
        y_base = H - 75
    n = len(cards)
    card_w = (W - SIDEBAR_W - 40) // max(n, 1)
    for i, card in enumerate(cards):
        x = SIDEBAR_W + 15 + i * (card_w + 5)
        draw_rounded_rect(img, x, y_base, card_w - 5, 60, PANEL, 6)
        cv2.rectangle(img, (x, y_base), (x + card_w - 5, y_base + 60), BORDER, 1)
        freq = card.get("freq_hz", 0)
        color = GREEN if freq > 8 else TEAL if freq > 3 else GOLD
        text(img, f"{freq:.1f} Hz", x + 10, y_base + 25, 0.55, color, 2)
        text(img, card.get("label", ""), x + 10, y_base + 42, 0.3, GRAY)
        text(img, f"Dist: {card.get('distance', 0):.3f}m", x + 10, y_base + 55, 0.3, DIM)


def draw_buttons(img, running=False, y_base=None):
    if y_base is None:
        y_base = H - 110
    bx = SIDEBAR_W + 15
    if running:
        draw_rounded_rect(img, bx, y_base, 110, 30, (50, 20, 20), 6)
        text(img, "Stop", bx + 38, y_base + 21, 0.45, RED_SOFT, 1)
        # pulsing dot
        text(img, "RUNNING", bx + 130, y_base + 21, 0.4, GREEN, 1)
    else:
        draw_rounded_rect(img, bx, y_base, 110, 30, (0, 120, 115), 6)
        text(img, "Start Demo", bx + 12, y_base + 21, 0.45, WHITE, 1)


def draw_cursor(img, x, y, t):
    """Animated cursor with subtle pulse."""
    sz = 12 + int(2 * math.sin(t * 3))
    pts = np.array([[x, y], [x, y + sz], [x + sz//2, y + sz*2//3]], np.int32)
    cv2.fillPoly(img, [pts], WHITE)
    cv2.polylines(img, [pts], True, (50, 50, 50), 1)


def build_dashboard_frame(scenario, sim_frame=None, cards=None, running=False,
                          step_num=0, total_steps=0, cursor_pos=None, cursor_t=0,
                          task="pick up the cube", fusion="weighted_average"):
    """Compose a full dashboard frame with sidebar + main area."""
    img = np.full((H, W, 3), BG, dtype=np.uint8)
    draw_sidebar(img, selected_scenario=scenario, task=task, fusion=fusion)
    draw_header(img)

    main_x = SIDEBAR_W + 10
    main_y = HEADER_H + 8

    if sim_frame is not None:
        vid_w = W - SIDEBAR_W - 20
        vid_h = H - HEADER_H - 130
        resized = cv2.resize(sim_frame, (vid_w, vid_h))
        img[main_y:main_y + vid_h, main_x:main_x + vid_w] = resized
        cv2.rectangle(img, (main_x, main_y), (main_x + vid_w, main_y + vid_h), BORDER, 1)

        if running and total_steps > 0:
            pbar_y = main_y + vid_h + 2
            pbar_w = int(vid_w * step_num / total_steps)
            cv2.rectangle(img, (main_x, pbar_y), (main_x + pbar_w, pbar_y + 4), TEAL, -1)
    else:
        vid_w = W - SIDEBAR_W - 20
        vid_h = H - HEADER_H - 130
        draw_rounded_rect(img, main_x, main_y, vid_w, vid_h, PANEL, 8)
        text(img, "Select a scenario and click Start Demo", main_x + vid_w//2 - 170, main_y + vid_h//2, 0.5, DIM)

    if cards:
        draw_metric_cards(img, cards)
    draw_buttons(img, running=running)

    if cursor_pos:
        draw_cursor(img, cursor_pos[0], cursor_pos[1], cursor_t)

    return img


# ── Cursor animation helper ──────────────────────────────────────────

def lerp_cursor(start, end, steps):
    """Generate smooth cursor positions from start to end."""
    positions = []
    for i in range(steps):
        t = i / max(steps - 1, 1)
        t = 0.5 * (1 - math.cos(t * math.pi))  # ease in-out
        x = int(start[0] + (end[0] - start[0]) * t)
        y = int(start[1] + (end[1] - start[1]) * t)
        positions.append((x, y))
    return positions


# ── Simulation metric helpers ─────────────────────────────────────────

def sim_metrics(step, total, env_id, model="PI0"):
    base = 330.0 if model == "PI0" else 229.0
    inf_ms = base + np.random.normal(0, 3)
    frac = step / max(total, 1)
    dist = max(0.03, 0.88 - frac * 0.80 + env_id * 0.02 + np.random.normal(0, 0.003))
    return {"inference_ms": round(inf_ms), "freq_hz": round(1000/(inf_ms/5), 1), "distance": round(dist, 3)}


# ======================================================================
# Main
# ======================================================================

def main():
    print("=" * 60)
    print("  Generating UI Walkthrough Video")
    print("=" * 60)

    writer = imageio.get_writer(OUT_PATH, fps=FPS, codec="libx264",
                                output_params=["-pix_fmt", "yuv420p"])
    fc = [0]
    def write(f):
        writer.append_data(f); fc[0] += 1

    # ── ACT 1: Dashboard loads (3s) ──────────────────────────────
    print("[1/8] Dashboard loading...")
    for i in range(FPS * 3):
        alpha = min(1.0, i / (FPS * 1.5))
        f = build_dashboard_frame(scenario=0, cursor_pos=(640, 400), cursor_t=i * 0.05)
        if alpha < 1.0:
            f = (f.astype(np.float32) * alpha).astype(np.uint8)
        write(f)

    # ── ACT 2: Cursor moves to S1, clicks Start (3s) ─────────────
    print("[2/8] User selects Scenario 1...")
    s1_radio = (25, 100)
    start_btn = (SIDEBAR_W + 60, H - 95)
    cursor_start = (640, 400)

    # Move to S1 radio
    for pos in lerp_cursor(cursor_start, s1_radio, FPS):
        write(build_dashboard_frame(0, cursor_pos=pos, cursor_t=fc[0]*0.05))

    # Hover on S1 for a beat
    for _ in range(FPS // 2):
        write(build_dashboard_frame(0, cursor_pos=s1_radio, cursor_t=fc[0]*0.05))

    # Move to Start button
    for pos in lerp_cursor(s1_radio, start_btn, FPS):
        write(build_dashboard_frame(0, cursor_pos=pos, cursor_t=fc[0]*0.05))

    # Click flash
    for _ in range(FPS // 3):
        f = build_dashboard_frame(0, running=True, cursor_pos=start_btn, cursor_t=fc[0]*0.05)
        write(f)

    # ── ACT 3: Scenario 1 runs (12s) ─────────────────────────────
    print("[3/8] Scenario 1 running...")
    tasks = ["pick cube", "push block", "lift object", "reach target"]
    envs = MultiEnvironment(num_envs=4, image_size=64, seed=42)
    trajs = []
    for i in range(4):
        e = envs.envs[i]
        cube = e.get_cube_position()
        wp = WP_BUILDERS[tasks[i]](e.physics_client, e.robot_id, cube)
        trajs.append(waypoints_to_traj(wp))
    sim_len = max(len(t) for t in trajs)
    for i in range(4):
        while len(trajs[i]) < sim_len:
            trajs[i].append(trajs[i][-1].copy())

    for step in range(sim_len):
        for i in range(4):
            envs.envs[i].apply_actions(trajs[i][step], use_delta=False, max_velocity=8.0)
            envs.envs[i].step()

        frames = envs.capture_all_display_frames(320, 240)
        labels = [f"Chip {i}: {tasks[i]}" for i in range(4)]
        from models.experimental.robotics_demo.video_composer import compose_quad_view as cq
        mets = [sim_metrics(step, sim_len, i) for i in range(4)]
        sim_frame = cq(frames, labels=labels, metrics=mets,
                        output_width=W - SIDEBAR_W - 20, output_height=H - HEADER_H - 130)

        cards = [{"freq_hz": mets[i]["freq_hz"], "label": f"Chip {i} - PI0",
                  "distance": mets[i]["distance"]} for i in range(4)]
        f = build_dashboard_frame(0, sim_frame=sim_frame, cards=cards,
                                   running=True, step_num=step, total_steps=sim_len)
        write(f)
        if step % 50 == 0:
            print(f"    step {step}/{sim_len}")

    envs.close()

    # Hold final frame briefly
    for _ in range(FPS):
        f = build_dashboard_frame(0, sim_frame=sim_frame, cards=cards, running=False,
                                   step_num=sim_len, total_steps=sim_len)
        text(f, "Scenario 1 Complete", SIDEBAR_W + 100, H//2, 0.7, GREEN, 2)
        write(f)

    # ── ACT 4: User switches to S2 (2s) ──────────────────────────
    print("[4/8] User selects Scenario 2...")
    s2_radio = (25, 128)
    cursor_pos = (SIDEBAR_W + 60, H - 95)
    for pos in lerp_cursor(cursor_pos, s2_radio, FPS):
        write(build_dashboard_frame(0, cursor_pos=pos, cursor_t=fc[0]*0.05))
    for _ in range(FPS // 3):
        write(build_dashboard_frame(1, cursor_pos=s2_radio, cursor_t=fc[0]*0.05))
    for pos in lerp_cursor(s2_radio, start_btn, FPS // 2):
        write(build_dashboard_frame(1, cursor_pos=pos, cursor_t=fc[0]*0.05))
    for _ in range(FPS // 3):
        write(build_dashboard_frame(1, running=True, cursor_pos=start_btn, cursor_t=fc[0]*0.05))

    # ── ACT 5: Scenario 2 runs (12s) ─────────────────────────────
    print("[5/8] Scenario 2 running...")
    pi0_env = MultiEnvironment(num_envs=1, image_size=64, seed=42)
    smol_env = MultiEnvironment(num_envs=1, image_size=64, seed=42)
    pi0_cube = pi0_env.envs[0].get_cube_position()
    smol_cube = smol_env.envs[0].get_cube_position()
    pi0_traj = waypoints_to_traj(build_pick_waypoints(pi0_env.envs[0].physics_client, pi0_env.envs[0].robot_id, pi0_cube))
    sx, sy, sz = smol_cube
    smol_wp = [
        (solve_ik(smol_env.envs[0].physics_client, smol_env.envs[0].robot_id, EE_LINK, [0.3, 0.0, 0.55]), 15),
        (solve_ik(smol_env.envs[0].physics_client, smol_env.envs[0].robot_id, EE_LINK, [sx, sy, sz+0.12]), 30),
        (solve_ik(smol_env.envs[0].physics_client, smol_env.envs[0].robot_id, EE_LINK, [sx, sy, sz+0.04]), 20),
        (solve_ik(smol_env.envs[0].physics_client, smol_env.envs[0].robot_id, EE_LINK, [sx, sy, sz+0.04]), 10),
        (solve_ik(smol_env.envs[0].physics_client, smol_env.envs[0].robot_id, EE_LINK, [sx, sy, sz+0.35]), 25),
        (solve_ik(smol_env.envs[0].physics_client, smol_env.envs[0].robot_id, EE_LINK, [sx, sy, sz+0.45]), 15),
        (solve_ik(smol_env.envs[0].physics_client, smol_env.envs[0].robot_id, EE_LINK, [0.3, 0.0, 0.55]), 20),
    ]
    smol_traj = waypoints_to_traj(smol_wp)
    sim_len2 = max(len(pi0_traj), len(smol_traj))
    while len(pi0_traj) < sim_len2: pi0_traj.append(pi0_traj[-1].copy())
    while len(smol_traj) < sim_len2: smol_traj.append(smol_traj[-1].copy())

    for step in range(sim_len2):
        pi0_env.envs[0].apply_actions(pi0_traj[step], use_delta=False, max_velocity=8.0)
        pi0_env.envs[0].step()
        smol_env.envs[0].apply_actions(smol_traj[step], use_delta=False, max_velocity=8.0)
        smol_env.envs[0].step()

        vid_w = W - SIDEBAR_W - 20
        vid_h = H - HEADER_H - 130
        lf = pi0_env.capture_all_display_frames(vid_w // 2, vid_h)
        rf = smol_env.capture_all_display_frames(vid_w // 2, vid_h)
        pi0_m = sim_metrics(step, sim_len2, 0, "PI0")
        smol_m = sim_metrics(step, sim_len2, 0, "SmolVLA")
        sim_frame = compose_side_by_side(lf, rf, "PI0 (2.3B)", "SmolVLA (450M)",
                                          pi0_m, smol_m, output_width=vid_w, output_height=vid_h)

        cards = [
            {"freq_hz": pi0_m["freq_hz"], "label": "PI0 - Chip 0", "distance": pi0_m["distance"]},
            {"freq_hz": smol_m["freq_hz"], "label": "SmolVLA - Chip 1", "distance": smol_m["distance"]},
        ]
        f = build_dashboard_frame(1, sim_frame=sim_frame, cards=cards,
                                   running=True, step_num=step, total_steps=sim_len2,
                                   task="pick up the cube")
        write(f)
        if step % 50 == 0:
            print(f"    step {step}/{sim_len2}")

    pi0_env.close(); smol_env.close()

    for _ in range(FPS):
        f = build_dashboard_frame(1, sim_frame=sim_frame, cards=cards, running=False,
                                   step_num=sim_len2, total_steps=sim_len2)
        text(f, "Scenario 2 Complete", SIDEBAR_W + 100, H//2, 0.7, GREEN, 2)
        write(f)

    # ── ACT 6: User switches to S3 (2s) ──────────────────────────
    print("[6/8] User selects Scenario 3...")
    s3_radio = (25, 156)
    for pos in lerp_cursor(start_btn, s3_radio, FPS):
        write(build_dashboard_frame(1, cursor_pos=pos, cursor_t=fc[0]*0.05))
    for _ in range(FPS // 3):
        write(build_dashboard_frame(2, cursor_pos=s3_radio, cursor_t=fc[0]*0.05,
                                     fusion="weighted_average"))
    for pos in lerp_cursor(s3_radio, start_btn, FPS // 2):
        write(build_dashboard_frame(2, cursor_pos=pos, cursor_t=fc[0]*0.05,
                                     fusion="weighted_average"))
    for _ in range(FPS // 3):
        write(build_dashboard_frame(2, running=True, cursor_pos=start_btn, cursor_t=fc[0]*0.05,
                                     fusion="weighted_average"))

    # ── ACT 7: Scenario 3 runs (10s) ─────────────────────────────
    print("[7/8] Scenario 3 running...")
    ens_env = MultiEnvironment(num_envs=1, image_size=64, seed=42)
    ens_cube = ens_env.envs[0].get_cube_position()
    ex, ey, ez = ens_cube
    ens_wp_xyz = [
        ([0.3, 0.0, 0.55], 20),
        ([ex, ey, ez+0.15], 35),
        ([ex, ey, ez+0.04], 30),
        ([ex, ey, ez+0.04], 15),
        ([ex, ey, ez+0.45], 35),
        ([ex+0.1, ey, ez+0.45], 25),
        ([ex+0.1, ey, ez+0.15], 25),
        ([0.3, 0.0, 0.55], 25),
    ]
    ens_wp = [(solve_ik(ens_env.envs[0].physics_client, ens_env.envs[0].robot_id, EE_LINK, xyz), h) for xyz, h in ens_wp_xyz]
    ens_traj = waypoints_to_traj(ens_wp)
    strategies = ["Weighted Average", "Temporal Blend", "Confidence Gate"]

    for step in range(len(ens_traj)):
        ens_env.envs[0].apply_actions(ens_traj[step], use_delta=False, max_velocity=8.0)
        ens_env.envs[0].step()

        vid_w = W - SIDEBAR_W - 20
        vid_h = H - HEADER_H - 130
        disp = ens_env.capture_all_display_frames(vid_w, vid_h)
        sim_frame = disp[0]

        si = min(step * 3 // len(ens_traj), 2)
        strat = strategies[si]
        pi0_ms = 330 + np.random.normal(0, 3)
        smol_ms = 229 + np.random.normal(0, 3)
        wall_ms = max(pi0_ms, smol_ms)
        frac = step / len(ens_traj)
        dist = max(0.03, 0.85 - frac * 0.75)

        # Overlay on sim_frame
        cv2.putText(sim_frame, f"Fusion: {strat}", (10, 25), FONT, 0.5, GOLD, 1, cv2.LINE_AA)
        cv2.putText(sim_frame, f"PI0: {pi0_ms:.0f}ms  SmolVLA: {smol_ms:.0f}ms  Wall: {wall_ms:.0f}ms  1.7x speedup",
                    (10, vid_h - 15), FONT, 0.4, GOLD, 1, cv2.LINE_AA)

        cards = [{"freq_hz": round(1000/wall_ms*5, 1), "label": f"Ensemble - {strat}", "distance": round(dist, 3)}]
        f = build_dashboard_frame(2, sim_frame=sim_frame, cards=cards,
                                   running=True, step_num=step, total_steps=len(ens_traj),
                                   fusion=strat.lower().replace(" ", "_"))
        write(f)
        if step % 50 == 0:
            print(f"    step {step}/{len(ens_traj)}")

    ens_env.close()

    for _ in range(FPS):
        f = build_dashboard_frame(2, sim_frame=sim_frame, cards=cards, running=False,
                                   step_num=len(ens_traj), total_steps=len(ens_traj),
                                   fusion="confidence_gate")
        text(f, "Scenario 3 Complete", SIDEBAR_W + 100, H//2, 0.7, GREEN, 2)
        write(f)

    # ── ACT 8: Outro (3s) ────────────────────────────────────────
    print("[8/8] Outro...")
    for i in range(FPS * 3):
        f = np.full((H, W, 3), BG, dtype=np.uint8)
        draw_sidebar(f, 2)
        draw_header(f)
        text(f, "Demo Complete", SIDEBAR_W + (W-SIDEBAR_W)//2 - 120, H//2 - 20, 0.9, TEAL, 2)
        text(f, "Tenstorrent Robotics Intelligence Suite", SIDEBAR_W + (W-SIDEBAR_W)//2 - 190, H//2 + 20, 0.5, WHITE)
        text(f, "Ready for live presentation on 4x Blackhole chips", SIDEBAR_W + (W-SIDEBAR_W)//2 - 200, H//2 + 50, 0.45, GRAY)
        write(f)

    writer.close()
    dur = fc[0] / FPS
    sz = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"\n  Saved: {OUT_PATH}")
    print(f"  {fc[0]} frames, {dur:.1f}s, {sz:.1f} MB, {W}x{H} @ {FPS} FPS\n")


if __name__ == "__main__":
    main()
