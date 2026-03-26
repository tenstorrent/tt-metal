#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent Robotics Intelligence Suite -- Live Streamlit Dashboard.

Run with:
    streamlit run models/experimental/robotics_demo/streamlit_app.py

This is the customer-facing live demo interface. It controls all four
scenarios, displays real-time video feeds, and shows live metrics.
"""

import os
import sys
import time
import threading
from pathlib import Path

import numpy as np

# Streamlit must be imported before other heavy imports
import streamlit as st

TT_METAL_HOME = os.environ.get("TT_METAL_HOME", str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, TT_METAL_HOME)
os.environ.setdefault("TT_METAL_HOME", TT_METAL_HOME)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Tenstorrent Robotics Intelligence Suite",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for dark professional look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .main-header h1 {
        color: #00cec9;
        margin: 0;
        font-size: 2rem;
    }
    .main-header p {
        color: #dfe6e9;
        margin: 0.3rem 0 0 0;
    }
    .metric-card {
        background: #16213e;
        border: 1px solid #0a3d62;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card h3 {
        color: #00cec9;
        margin: 0;
        font-size: 1.8rem;
    }
    .metric-card p {
        color: #b2bec3;
        margin: 0.2rem 0 0 0;
        font-size: 0.85rem;
    }
    .scenario-desc {
        background: #0a3d62;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        color: #dfe6e9;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Tenstorrent Robotics Intelligence Suite</h1>
    <p>Live multi-model robotic simulation on 4x Blackhole chips</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "running" not in st.session_state:
    st.session_state.running = False
if "latest_frame" not in st.session_state:
    st.session_state.latest_frame = None
if "latest_metrics" not in st.session_state:
    st.session_state.latest_metrics = None

# ---------------------------------------------------------------------------
# Sidebar -- Hardware status + configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Hardware Status")
    try:
        import ttnn
        device_ids = ttnn.get_device_ids()
        num_chips = len(device_ids)
        st.success(f"Detected {num_chips} Tenstorrent chip(s)")
        for did in device_ids:
            st.markdown(f"  - Chip {did}: Blackhole")
    except Exception:
        num_chips = 0
        st.warning("TTNN not available -- demo will run in simulation-only mode")

    st.markdown("---")
    st.markdown("### Scenario Selection")

    scenario = st.radio(
        "Choose demo scenario:",
        options=[
            "S1: Data-Parallel PI0 (4 Robots)",
            "S2: PI0 vs SmolVLA Comparison",
            "S3: Ensemble Pipeline",
            "S4: Throughput Scaling Benchmark",
        ],
        index=0,
    )
    scenario_num = int(scenario[1])

    st.markdown("---")
    st.markdown("### Configuration")

    num_steps = st.slider("Simulation steps", 100, 2000, 400, 50)
    replan_interval = st.slider("Re-plan interval", 1, 30, 5)
    task = st.text_input("Task instruction", "pick up the cube")
    record_video = st.checkbox("Record video", value=True)

    chips_to_use = st.slider("Chips to use", 1, max(num_chips, 4), min(num_chips, 4))

    if scenario_num == 3:
        st.markdown("#### Fusion Settings")
        fusion_strategy = st.selectbox(
            "Fusion strategy",
            ["weighted_average", "temporal_blend", "confidence_gate"],
        )
        alpha = st.slider("PI0 weight (alpha)", 0.0, 1.0, 0.6, 0.05)
    else:
        fusion_strategy = "weighted_average"
        alpha = 0.6

    checkpoint_path = os.path.join(
        TT_METAL_HOME, "models/experimental/pi0/weights/pi0_base"
    )

# ---------------------------------------------------------------------------
# Scenario descriptions
# ---------------------------------------------------------------------------
DESCRIPTIONS = {
    1: "Run 4 independent PI0 model replicas, each controlling its own Franka Panda "
       "robot in a separate PyBullet environment with a different task. Demonstrates "
       "near-linear throughput scaling across Blackhole chips.",
    2: "Run PI0 and SmolVLA side-by-side on the same task in mirrored environments. "
       "Compare inference latency, control frequency, and task completion in real-time.",
    3: "An innovative multi-model pipeline where SmolVLA provides rapid coarse action "
       "proposals and PI0 provides refined control. Three fusion strategies blend their "
       "outputs for superior robotic control.",
    4: "Clean performance benchmark measuring throughput scaling from 1 to 4 chips. "
       "Generates scaling charts and latency breakdown waterfalls.",
}

# ---------------------------------------------------------------------------
# Main content area
# ---------------------------------------------------------------------------
st.markdown(f'<div class="scenario-desc">{DESCRIPTIONS[scenario_num]}</div>',
            unsafe_allow_html=True)

col_video, col_metrics = st.columns([3, 1])

with col_video:
    st.markdown("### Live Simulation Feed")
    video_placeholder = st.empty()

with col_metrics:
    st.markdown("### Live Metrics")
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()

# ---------------------------------------------------------------------------
# Metric cards row
# ---------------------------------------------------------------------------
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
metric_slots = [mcol1.empty(), mcol2.empty(), mcol3.empty(), mcol4.empty()]

# ---------------------------------------------------------------------------
# Start / stop controls
# ---------------------------------------------------------------------------
col_start, col_stop, _ = st.columns([1, 1, 4])

def _update_frame(frame: np.ndarray):
    st.session_state.latest_frame = frame

def _update_metrics(summary: dict):
    st.session_state.latest_metrics = summary


def _run_scenario():
    from models.experimental.robotics_demo.demo_orchestrator import (
        run_scenario_1, run_scenario_2, run_scenario_3, run_scenario_4,
    )

    try:
        if scenario_num == 1:
            run_scenario_1(
                num_steps=num_steps, num_devices=chips_to_use,
                checkpoint_path=checkpoint_path,
                replan_interval=replan_interval,
                record_video=record_video,
                on_frame=_update_frame,
                on_metrics=_update_metrics,
            )
        elif scenario_num == 2:
            run_scenario_2(
                num_steps=num_steps, task=task,
                checkpoint_path=checkpoint_path,
                replan_interval=replan_interval,
                record_video=record_video,
                on_frame=_update_frame,
                on_metrics=_update_metrics,
            )
        elif scenario_num == 3:
            run_scenario_3(
                num_steps=num_steps, task=task,
                checkpoint_path=checkpoint_path,
                fusion_strategy=fusion_strategy,
                alpha=alpha,
                replan_interval=replan_interval,
                record_video=record_video,
                on_frame=_update_frame,
                on_metrics=_update_metrics,
            )
        elif scenario_num == 4:
            run_scenario_4(
                max_chips=chips_to_use,
                checkpoint_path=checkpoint_path,
                on_metrics=_update_metrics,
            )
    except Exception as e:
        st.error(f"Error during scenario execution: {e}")
    finally:
        st.session_state.running = False


with col_start:
    if st.button("Start Demo", type="primary", disabled=st.session_state.running):
        st.session_state.running = True
        st.session_state.latest_frame = None
        st.session_state.latest_metrics = None
        thread = threading.Thread(target=_run_scenario, daemon=True)
        thread.start()

with col_stop:
    if st.button("Stop", disabled=not st.session_state.running):
        st.session_state.running = False
        st.info("Stop requested -- will halt after current step completes.")

# ---------------------------------------------------------------------------
# Live update loop
# ---------------------------------------------------------------------------
if st.session_state.running or st.session_state.latest_frame is not None:
    status_placeholder = st.empty()

    for _ in range(num_steps * 2):
        if not st.session_state.running and st.session_state.latest_frame is None:
            break

        frame = st.session_state.latest_frame
        if frame is not None:
            video_placeholder.image(frame, channels="RGB", use_container_width=True)

        summary = st.session_state.latest_metrics
        if summary is not None:
            with metrics_placeholder.container():
                st.metric("Total Inferences", summary.get("total_inferences", 0))
                st.metric("Throughput",
                          f"{summary.get('aggregate_throughput_fps', 0):.1f} FPS")
                st.metric("Avg Inference",
                          f"{summary.get('avg_inference_ms', 0):.0f} ms")
                st.metric("Environments", summary.get("num_envs", 0))

            per_env = summary.get("per_env", [])
            for i, slot in enumerate(metric_slots):
                if i < len(per_env):
                    e = per_env[i]
                    slot.markdown(f"""
                    <div class="metric-card">
                        <h3>{e.get('control_freq_hz', 0):.1f} Hz</h3>
                        <p>{e.get('model_name', 'N/A')} -- Chip {e.get('env_id', i)}</p>
                        <p>Dist: {e.get('current_distance', 0):.3f}m</p>
                    </div>
                    """, unsafe_allow_html=True)

        if not st.session_state.running:
            status_placeholder.success("Demo complete.")
            break

        time.sleep(0.15)  # ~7 FPS UI refresh

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#636e72; font-size:0.8rem;">'
    'Tenstorrent Robotics Intelligence Suite v0.1 -- Powered by Blackhole AI Accelerators'
    '</p>',
    unsafe_allow_html=True,
)
