#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Interactive Streamlit UI for single-chip robotics demo (N150 / P150).

Provides a live, user-interactive experience:
  - Pick your model (PI0 or SmolVLA) with one click
  - Type a custom task instruction
  - Watch the robot move in real time with live metrics
  - Switch models or tasks mid-session
  - Camera view toggle (front / side / overhead)

Launch:
    streamlit run models/experimental/robotics_demo_n150/interactive_ui.py
"""

import os
import sys
import time
import threading
import math
from pathlib import Path
from collections import deque

import numpy as np

import streamlit as st

TT_METAL_HOME = os.environ.get("TT_METAL_HOME",
    str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, TT_METAL_HOME)
os.environ.setdefault("TT_METAL_HOME", TT_METAL_HOME)

import pybullet as p
import pybullet_data

# ── Page setup ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tenstorrent Robotics Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] {background-color: #16213e;}
    .main-title {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 0.8rem;
    }
    .main-title h2 {color: #00cec9; margin: 0;}
    .main-title p {color: #dfe6e9; margin: 0.2rem 0 0 0; font-size: 0.9rem;}
    .metric-box {
        background: #16213e; border: 1px solid #0a3d62; border-radius: 8px;
        padding: 0.8rem; text-align: center; margin-bottom: 0.5rem;
    }
    .metric-box h3 {color: #00cec9; margin: 0; font-size: 1.6rem;}
    .metric-box p {color: #b2bec3; margin: 0; font-size: 0.75rem;}
    .model-card {
        border: 2px solid #0a3d62; border-radius: 10px; padding: 1rem;
        text-align: center; cursor: pointer; transition: border-color 0.2s;
    }
    .model-card:hover {border-color: #00cec9;}
    .model-card.selected {border-color: #00cec9; background: #0a3d62;}
    .model-card h4 {margin: 0; font-size: 1.1rem;}
    .model-card p {margin: 0.2rem 0 0 0; font-size: 0.8rem; color: #b2bec3;}
    .status-running {color: #7ee787; font-weight: bold;}
    .status-idle {color: #b2bec3;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────
for key, default in [
    ("running", False), ("frame", None), ("metrics", {}),
    ("model_name", "pi0"), ("task", "pick up the cube"),
    ("steps_done", 0), ("total_steps", 300),
    ("camera_view", "front-right"), ("history", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Header ────────────────────────────────────────────────────────────
arch = os.environ.get("ARCH_NAME", "auto-detect")
chip_label = "N150 (Wormhole)" if "wormhole" in arch.lower() else "P150 (Blackhole)" if "blackhole" in arch.lower() else "Single Chip"

st.markdown(f"""
<div class="main-title">
    <h2>Tenstorrent Robotics Demo</h2>
    <p>Single-chip VLA inference &mdash; {chip_label} &mdash; PI0 or SmolVLA</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Choose Your Model")

    col_pi0, col_smol = st.columns(2)
    with col_pi0:
        if st.button("PI0\n2.3B params\n~330ms", use_container_width=True,
                      type="primary" if st.session_state.model_name == "pi0" else "secondary"):
            st.session_state.model_name = "pi0"
    with col_smol:
        if st.button("SmolVLA\n450M params\n~229ms", use_container_width=True,
                      type="primary" if st.session_state.model_name == "smolvla" else "secondary"):
            st.session_state.model_name = "smolvla"

    st.markdown(f"**Selected:** `{st.session_state.model_name.upper()}`")

    st.markdown("---")
    st.markdown("### Task Instruction")
    task_presets = [
        "pick up the cube",
        "push the block right",
        "lift the object",
        "reach the target",
        "grasp the red cube",
    ]
    preset = st.selectbox("Quick presets:", task_presets, index=0)
    custom = st.text_input("Or type your own:", value="")
    st.session_state.task = custom if custom.strip() else preset

    st.markdown("---")
    st.markdown("### Simulation Settings")
    st.session_state.total_steps = st.slider("Steps", 100, 1000, 300, 50)
    replan = st.slider("Replan interval", 1, 20, 5)
    velocity = st.slider("Max velocity (rad/s)", 0.1, 2.0, 0.5, 0.1)

    st.markdown("---")
    st.markdown("### Camera View")
    st.session_state.camera_view = st.radio(
        "Angle:", ["front-right", "side", "overhead", "close-up"],
        index=0, horizontal=True,
    )

    st.markdown("---")
    hw_status = "Detected" if arch != "auto-detect" else "Demo mode available"
    st.success(f"Hardware: {hw_status}")

# ── Main area ─────────────────────────────────────────────────────────

# Top row: model info + live status
info_col, status_col = st.columns([3, 1])
with info_col:
    model_desc = {
        "pi0": "PI0 -- SigLIP 27L + Gemma 2B VLM + 300M Action Expert | Flow matching 10 steps",
        "smolvla": "SmolVLA -- SigLIP 12L + VLM 16L + Expert 16L | Flow matching 10 steps",
    }
    st.markdown(f"**Model:** {model_desc.get(st.session_state.model_name, '')}")
    st.markdown(f'**Task:** "{st.session_state.task}"')
with status_col:
    if st.session_state.running:
        st.markdown('<p class="status-running">RUNNING</p>', unsafe_allow_html=True)
        progress = st.session_state.steps_done / max(st.session_state.total_steps, 1)
        st.progress(progress)
    else:
        st.markdown('<p class="status-idle">IDLE</p>', unsafe_allow_html=True)

# Video + metrics
vid_col, met_col = st.columns([3, 1])

with vid_col:
    video_placeholder = st.empty()
    frame = st.session_state.frame
    if frame is not None:
        video_placeholder.image(frame, channels="RGB", use_container_width=True)
    else:
        video_placeholder.info("Click **Start** to begin the simulation")

with met_col:
    st.markdown("### Live Metrics")
    m = st.session_state.metrics
    st.markdown(f"""
    <div class="metric-box"><h3>{m.get('freq_hz', 0):.1f} Hz</h3><p>Control Frequency</p></div>
    <div class="metric-box"><h3>{m.get('inference_ms', 0):.0f} ms</h3><p>Inference Latency</p></div>
    <div class="metric-box"><h3>{m.get('distance', 0):.3f} m</h3><p>Distance to Cube</p></div>
    <div class="metric-box"><h3>{m.get('step', 0)}</h3><p>Steps Completed</p></div>
    """, unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("### Distance Over Time")
        st.line_chart({"distance (m)": st.session_state.history[-100:]})

# Controls
ctrl_col1, ctrl_col2, ctrl_col3, _ = st.columns([1, 1, 1, 3])

CAMERA_CONFIGS = {
    "front-right": {"eye": [1.3, 0.6, 0.9], "target": [0.3, 0, 0.3]},
    "side":        {"eye": [0.2, 1.2, 0.8], "target": [0.3, 0, 0.3]},
    "overhead":    {"eye": [0.4, 0.0, 1.4], "target": [0.4, 0, 0.1]},
    "close-up":    {"eye": [0.7, 0.3, 0.4], "target": [0.5, 0, 0.1]},
}


def _run_simulation():
    """Background thread that runs the simulation loop."""
    from models.experimental.robotics_demo_n150.sim_env import FrankaCubeEnv

    env = FrankaCubeEnv(image_size=224)
    env.reset()

    model_name = st.session_state.model_name
    task = st.session_state.task
    total = st.session_state.total_steps
    use_demo = True  # Always demo mode in UI for now (hardware loaded separately)

    device = None
    model = None

    try:
        import ttnn
        _test = ttnn.bfloat16
        device = ttnn.open_device(device_id=0, l1_small_size=24576)
        if model_name == "pi0":
            from models.experimental.robotics_demo_n150.run_demo import load_pi0_model
            tt_home = os.environ.get("TT_METAL_HOME", "")
            cp = os.path.join(tt_home, "models/experimental/pi0/weights/pi0_base")
            if Path(cp).exists():
                model, _ = load_pi0_model(device, cp)
                use_demo = False
        elif model_name == "smolvla":
            from models.experimental.robotics_demo_n150.run_demo import load_smolvla_model
            model = load_smolvla_model(device)
            use_demo = False
    except Exception:
        pass

    from models.experimental.robotics_demo_n150.tokenizer import DemoTokenizer
    tokenizer = DemoTokenizer()

    inference_times = deque(maxlen=100)
    loop_times = deque(maxlen=100)
    action_buffer = None
    buf_idx = 0

    # Warmup
    if model is not None:
        for _ in range(2):
            if model_name == "pi0":
                imgs, st_ = env.capture_observations()
                import ttnn as _t
                imgs_tt = [_t.from_torch(im, dtype=_t.bfloat16, layout=_t.TILE_LAYOUT,
                           device=device, memory_config=_t.DRAM_MEMORY_CONFIG) for im in imgs]
                toks, masks = tokenizer.encode(task)
                model.sample_actions(
                    images=imgs_tt,
                    img_masks=[__import__('torch').ones(1, dtype=__import__('torch').bool)] * 2,
                    lang_tokens=_t.from_torch(toks, dtype=_t.uint32, layout=_t.ROW_MAJOR_LAYOUT, device=device),
                    lang_masks=_t.from_torch(masks.float(), dtype=_t.bfloat16, layout=_t.TILE_LAYOUT, device=device),
                    state=_t.from_torch(st_, dtype=_t.bfloat16, layout=_t.TILE_LAYOUT, device=device),
                )
            else:
                model.sample_actions(images=[env.capture_pil_image()], instruction=task,
                                     num_inference_steps=10, action_dim=7)

    for step in range(total):
        if not st.session_state.running:
            break

        loop_start = time.time()
        need_replan = (step % replan == 0) or (action_buffer is None)

        if need_replan:
            inf_start = time.time()
            if use_demo:
                cube = env.get_cube_position()
                cx, cy, cz = cube
                phase = (step % 200) / 200.0
                if phase < 0.3:
                    tgt = [cx, cy, cz + 0.18 - phase * 0.4]
                elif phase < 0.5:
                    tgt = [cx, cy, cz + 0.04]
                elif phase < 0.8:
                    tgt = [cx, cy, cz + 0.04 + (phase - 0.5) * 1.5]
                else:
                    tgt = [0.3, 0.0, 0.55]
                ik = p.calculateInverseKinematics(env.robot_id, 11, tgt,
                                                   physicsClientId=env.physics_client)
                action_buffer = np.array(ik[:7]).reshape(1, -1)
            elif model_name == "pi0":
                import ttnn as _t
                imgs, st_ = env.capture_observations()
                imgs_tt = [_t.from_torch(im, dtype=_t.bfloat16, layout=_t.TILE_LAYOUT,
                           device=device, memory_config=_t.DRAM_MEMORY_CONFIG) for im in imgs]
                toks, masks = tokenizer.encode(task)
                raw = model.sample_actions(
                    images=imgs_tt,
                    img_masks=[__import__('torch').ones(1, dtype=__import__('torch').bool)] * 2,
                    lang_tokens=_t.from_torch(toks, dtype=_t.uint32, layout=_t.ROW_MAJOR_LAYOUT, device=device),
                    lang_masks=_t.from_torch(masks.float(), dtype=_t.bfloat16, layout=_t.TILE_LAYOUT, device=device),
                    state=_t.from_torch(st_, dtype=_t.bfloat16, layout=_t.TILE_LAYOUT, device=device),
                )
                if isinstance(raw, _t.Tensor):
                    raw = _t.to_torch(raw)
                action_buffer = raw.float().cpu().numpy()
            else:
                raw = model.sample_actions(images=[env.capture_pil_image()],
                                           instruction=task, num_inference_steps=10, action_dim=7)
                action_buffer = np.asarray(raw, dtype=np.float32)
            buf_idx = 0
            inference_times.append((time.time() - inf_start) * 1000)
        else:
            buf_idx += 1

        if action_buffer is not None:
            if action_buffer.ndim == 3:
                idx = min(buf_idx, action_buffer.shape[1] - 1)
                act = action_buffer[0, idx, :7]
            elif action_buffer.ndim == 2:
                idx = min(buf_idx, action_buffer.shape[0] - 1)
                act = action_buffer[idx, :7]
                if len(act) < 7:
                    act = np.pad(act, (0, 7 - len(act)))
            else:
                act = action_buffer[:7]

            if use_demo:
                env.apply_actions(act, use_delta=False, max_velocity=8.0)
            else:
                env.apply_actions(act, use_delta=True, delta_scale=1.0, max_velocity=velocity)

        env.step()
        loop_ms = (time.time() - loop_start) * 1000
        loop_times.append(loop_ms)

        # Capture frame with selected camera view
        cam = CAMERA_CONFIGS.get(st.session_state.camera_view, CAMERA_CONFIGS["front-right"])
        view = p.computeViewMatrix(cameraEyePosition=cam["eye"],
                                   cameraTargetPosition=cam["target"],
                                   cameraUpVector=[0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=16/9, nearVal=0.1, farVal=5.0)
        _, _, rgba, _, _ = p.getCameraImage(width=960, height=540, viewMatrix=view,
                                            projectionMatrix=proj, renderer=p.ER_TINY_RENDERER,
                                            physicsClientId=env.physics_client)
        frame_rgb = np.array(rgba[:, :, :3], dtype=np.uint8)

        try:
            import cv2
            dist = env.get_distance_to_target()
            ee = env.get_ee_position()
            avg_inf = np.mean(inference_times) if inference_times else 0
            hz = 1000 / np.mean(loop_times) if loop_times and np.mean(loop_times) > 0 else 0
            cv2.putText(frame_rgb, f"{model_name.upper()} on {chip_label} | \"{task}\"",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 206, 201), 2)
            cv2.putText(frame_rgb, f"Inf: {avg_inf:.0f}ms | {hz:.1f} Hz | Step {step}/{total}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame_rgb, f"EE: [{ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f}] | Dist: {dist:.3f}m",
                        (10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        except ImportError:
            dist = env.get_distance_to_target()
            avg_inf = 0
            hz = 0

        st.session_state.frame = frame_rgb
        st.session_state.steps_done = step + 1
        st.session_state.metrics = {
            "freq_hz": hz,
            "inference_ms": avg_inf,
            "distance": dist,
            "step": step + 1,
        }
        st.session_state.history.append(dist)

    env.close()
    if device is not None:
        try:
            import ttnn
            ttnn.close_device(device)
        except Exception:
            pass
    st.session_state.running = False


with ctrl_col1:
    if st.button("Start", type="primary", disabled=st.session_state.running,
                  use_container_width=True):
        st.session_state.running = True
        st.session_state.frame = None
        st.session_state.metrics = {}
        st.session_state.history = []
        st.session_state.steps_done = 0
        threading.Thread(target=_run_simulation, daemon=True).start()

with ctrl_col2:
    if st.button("Stop", disabled=not st.session_state.running,
                  use_container_width=True):
        st.session_state.running = False

with ctrl_col3:
    if st.button("Reset View", use_container_width=True):
        st.session_state.frame = None
        st.session_state.metrics = {}
        st.session_state.history = []

# ── Auto-refresh loop ─────────────────────────────────────────────────
if st.session_state.running:
    for _ in range(st.session_state.total_steps * 2):
        if not st.session_state.running:
            break
        f = st.session_state.frame
        if f is not None:
            video_placeholder.image(f, channels="RGB", use_container_width=True)
        m = st.session_state.metrics
        if m:
            with met_col:
                st.markdown(f"""
                <div class="metric-box"><h3>{m.get('freq_hz', 0):.1f} Hz</h3><p>Control Frequency</p></div>
                <div class="metric-box"><h3>{m.get('inference_ms', 0):.0f} ms</h3><p>Inference Latency</p></div>
                <div class="metric-box"><h3>{m.get('distance', 0):.3f} m</h3><p>Distance to Cube</p></div>
                <div class="metric-box"><h3>{m.get('step', 0)}</h3><p>Steps Completed</p></div>
                """, unsafe_allow_html=True)
        time.sleep(0.12)

    if not st.session_state.running:
        st.success(f"Simulation complete! Final distance: {st.session_state.metrics.get('distance', '?')}m")

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<p style="text-align:center;color:#636e72;font-size:0.75rem;">'
    f'Tenstorrent Robotics Demo v0.1 &mdash; {chip_label} &mdash; PI0 + SmolVLA'
    f'</p>', unsafe_allow_html=True,
)
