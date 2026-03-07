#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Streamlit LLM Fine-tuning Application
Real-time monitoring and control interface for LLM fine-tuning with SLURM job dispatch.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
import re
from datetime import datetime
from pathlib import Path
import yaml
import subprocess
import json

from job_manager import JobManager, JobInfo, JobStatus, PARTITION_DEVICE_MAPPING

device_mesh_shapes = {
    "N150": [1, 1],
    "N300": [1, 2],
    "LoudBox": [1, 8],
    "Galaxy": [1, 32],
    "3-tier": [5, 32],
}

# Set page configuration
st.set_page_config(
    page_title="LLM Fine-tuning Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def init_session_state():
    """Initialize all session state variables."""
    if "training_history" not in st.session_state:
        st.session_state.training_history = []
    if "last_step" not in st.session_state:
        st.session_state.last_step = -1
    if "last_update" not in st.session_state:
        st.session_state.last_update = time.time()
    if "training_process" not in st.session_state:
        st.session_state.training_process = None
    if "training_running" not in st.session_state:
        st.session_state.training_running = False
    if "tt_smi_data" not in st.session_state:
        st.session_state.tt_smi_data = None
    if "tt_smi_error" not in st.session_state:
        st.session_state.tt_smi_error = None
    if "data_collection_paused" not in st.session_state:
        st.session_state.data_collection_paused = False
    if "job_manager" not in st.session_state:
        st.session_state.job_manager = JobManager()
    if "selected_job_id" not in st.session_state:
        st.session_state.selected_job_id = None
    if "execution_mode" not in st.session_state:
        st.session_state.execution_mode = "slurm"
    if "available_partitions" not in st.session_state:
        st.session_state.available_partitions = []


init_session_state()


def parse_output_file(file_path="output.txt"):
    """Parse the output.txt file to extract training metrics.

    Returns:
        dict with 'current' (latest data point) and 'all' (list of all data points)
    """
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return None

        pattern = r"LR:\s*([\d.e+-]+),\s*training_loss:\s*([\d.]+),\s*val_loss:\s*([\d.]+),\s*step:\s*(\d+),\s*epoch:\s*(\d+)"

        all_data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.search(pattern, line)
            if match:
                lr, train_loss, val_loss, step, epoch = match.groups()
                data_point = {
                    "lr": float(lr),
                    "training_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "step": int(step),
                    "epoch": int(epoch),
                    "timestamp": datetime.now(),
                }
                all_data.append(data_point)

        if not all_data:
            return None

        return {"current": all_data[-1], "all": all_data}

    except Exception as e:
        st.error(f"Error parsing output.txt: {e}")

    return None


def parse_validation_file(file_path="validation.txt"):
    """Parse the validation.txt file content."""
    if not os.path.exists(file_path):
        return "No validation data available yet."

    try:
        with open(file_path, "r") as f:
            content = f.read().strip()
        return content if content else "Validation file is empty."
    except Exception as e:
        return f"Error reading validation.txt: {e}"


def create_loss_plot(history_df, eval_every=None, title_suffix=""):
    """Create an interactive plot for training and validation loss."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df["step"],
            y=history_df["training_loss"],
            mode="lines",
            name="Training Loss",
            line=dict(color="#1f77b4", width=2),
            hovertemplate="Step: %{x}<br>Train Loss: %{y:.4f}<extra></extra>",
        )
    )

    if eval_every is not None and eval_every > 0:
        val_mask = ((history_df["step"] + 1) % eval_every == 0) | (
            history_df["step"] == 0
        )
        val_df = history_df[val_mask]
    else:
        val_df = history_df

    fig.add_trace(
        go.Scatter(
            x=val_df["step"],
            y=val_df["val_loss"],
            mode="lines+markers",
            name="Validation Loss",
            line=dict(color="#ff7f0e", width=2),
            marker=dict(size=6),
            hovertemplate="Step: %{x}<br>Val Loss: %{y:.4f}<extra></extra>",
        )
    )

    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Loss")

    fig.update_layout(
        title=f"Loss over Steps{title_suffix}",
        height=500,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_lr_plot(history_df, title_suffix=""):
    """Create an interactive plot for learning rate."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=history_df["step"],
            y=history_df["lr"],
            mode="lines",
            name="Learning Rate",
            line=dict(color="#2ca02c", width=2),
            hovertemplate="Step: %{x}<br>LR: %{y:.2e}<extra></extra>",
        )
    )

    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(
        title_text="Learning Rate", type="log", exponentformat="e", showexponent="all"
    )

    fig.update_layout(
        title=f"Learning Rate over Steps{title_suffix}",
        height=500,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_comparison_plot(jobs_data: dict):
    """Create a comparison plot for multiple jobs."""
    fig = go.Figure()

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    for i, (job_id, data) in enumerate(jobs_data.items()):
        if not data or not data.get("all"):
            continue

        df = pd.DataFrame(data["all"])
        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["training_loss"],
                mode="lines",
                name=f"Job {job_id[:8]} - Train",
                line=dict(color=color, width=2),
                hovertemplate=f"Job {job_id[:8]}<br>Step: %{{x}}<br>Train Loss: %{{y:.4f}}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["val_loss"],
                mode="lines",
                name=f"Job {job_id[:8]} - Val",
                line=dict(color=color, width=2, dash="dash"),
                hovertemplate=f"Job {job_id[:8]}<br>Step: %{{x}}<br>Val Loss: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Loss")

    fig.update_layout(
        title="Job Comparison - Training & Validation Loss",
        height=600,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_training_yaml(
    config_dict,
    output_path=None,
):
    """Create a training YAML configuration file."""
    if output_path is None:
        output_path = f"{os.environ.get('TT_METAL_HOME', '.')}/tt-train/configs/training_overrides.yaml"

    training_config = {
        "batch_size": config_dict["batch_size"],
        "validation_batch_size": config_dict["validation_batch_size"],
        "max_steps": config_dict["max_steps"],
        "gradient_accumulation_steps": config_dict["gradient_accumulation"],
        "eval_every": config_dict["eval_every"],
    }

    scheduler_config = {
        "warmup_steps": config_dict["warmup_steps"],
        "hold_steps": config_dict["hold_steps"],
        "min_lr": config_dict["min_lr"],
        "max_lr": config_dict["max_lr"],
    }

    device_config = {
        "enable_ddp": config_dict["enable_ddp"],
        "mesh_shape": config_dict["mesh_shape"],
    }

    transformer_config = {
        "max_sequence_length": config_dict["max_seq_length"],
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("training_config:\n")
        training_yaml = yaml.dump(
            training_config, default_flow_style=False, sort_keys=False
        )
        for line in training_yaml.strip().split("\n"):
            f.write(f"  {line}\n")

        f.write("\n")

        f.write("transformer_config:\n")
        transformer_yaml = yaml.dump(
            transformer_config, default_flow_style=False, sort_keys=False
        )
        for line in transformer_yaml.strip().split("\n"):
            f.write(f"  {line}\n")

        f.write("\n")

        f.write("scheduler_config:\n")
        scheduler_yaml = yaml.dump(
            scheduler_config, default_flow_style=False, sort_keys=False
        )
        for line in scheduler_yaml.strip().split("\n"):
            f.write(f"  {line}\n")

        f.write("\n")

        f.write("device_config:\n")
        f.write(f"  enable_ddp: {device_config['enable_ddp']}\n")
        mesh_shape_flow = yaml.dump(
            device_config["mesh_shape"], default_flow_style=True
        ).strip()
        f.write(f"  mesh_shape: {mesh_shape_flow}\n")

    return output_path


def start_local_training(config_dict):
    """Start the training process locally (non-SLURM mode)."""
    try:
        yaml_output_dir = config_dict.get(
            "yaml_dir", f"{os.environ.get('TT_METAL_HOME', '.')}/tt-train/configs"
        )
        os.makedirs(yaml_output_dir, exist_ok=True)
        yaml_output_path = os.path.join(yaml_output_dir, "training_overrides.yaml")

        yaml_path = create_training_yaml(config_dict, output_path=yaml_output_path)
        yaml_abs_path = os.path.abspath(yaml_path)
        st.success(f"Created configuration file: {yaml_abs_path}")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "gsm8k_finetune.py")

        if not os.path.exists(script_path):
            return False, f"Training script not found at: {script_path}"

        log_dir = script_dir
        stdout_log = os.path.join(log_dir, "training_stdout.log")
        stderr_log = os.path.join(log_dir, "training_stderr.log")

        with open(stdout_log, "w") as stdout_f, open(stderr_log, "w") as stderr_f:
            process = subprocess.Popen(
                ["python3", script_path],
                stdout=stdout_f,
                stderr=stderr_f,
                cwd=script_dir,
                env=os.environ.copy(),
            )

        st.session_state.training_process = process
        st.session_state.training_running = True
        st.session_state.training_pid = process.pid

        return (
            True,
            f"Training started successfully! PID: {process.pid}\nLogs: {stdout_log}",
        )
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        return False, f"Error starting training: {str(e)}\n{error_details}"


def stop_local_training():
    """Stop the local training process."""
    try:
        if st.session_state.training_process is not None:
            pid = st.session_state.training_process.pid

            st.session_state.training_process.terminate()

            try:
                st.session_state.training_process.wait(timeout=5)
                message = f"Training stopped gracefully (PID: {pid})"
            except subprocess.TimeoutExpired:
                st.session_state.training_process.kill()
                st.session_state.training_process.wait()
                message = f"Training force-stopped (PID: {pid})"

            st.session_state.training_process = None
            st.session_state.training_running = False

            return True, message
        else:
            return False, "No training process to stop"
    except Exception as e:
        st.session_state.training_process = None
        st.session_state.training_running = False
        return False, f"Error stopping training: {str(e)}"


def check_local_training_status():
    """Check if the local training process is still running."""
    if st.session_state.training_process is not None:
        poll = st.session_state.training_process.poll()
        if poll is not None:
            st.session_state.training_running = False
            st.session_state.training_process = None
    return st.session_state.training_running


def get_job_training_data(job_info: JobInfo):
    """Get training data for a specific job."""
    if not job_info:
        return None

    output_file = Path(job_info.output_dir) / "output.txt"
    return parse_output_file(str(output_file))


def render_jobs_table():
    """Render the jobs monitoring table."""
    st.subheader("Active Jobs")

    jobs = st.session_state.job_manager.get_all_jobs()

    if not jobs:
        st.info("No jobs submitted yet. Submit a job using the sidebar.")
        return

    # Build job ID to index mapping for selection handling
    job_id_to_idx = {job.job_id: idx for idx, job in enumerate(jobs)}

    jobs_data = []
    for job in jobs:
        training_data = get_job_training_data(job)
        current_step = "N/A"
        train_loss = "N/A"
        val_loss = "N/A"

        if training_data and training_data.get("current"):
            current = training_data["current"]
            current_step = current.get("step", "N/A")
            train_loss = f"{current.get('training_loss', 0):.4f}"
            val_loss = f"{current.get('val_loss', 0):.4f}"

        status_emoji = {
            "PENDING": "🟡",
            "RUNNING": "🟢",
            "COMPLETED": "✅",
            "FAILED": "❌",
            "CANCELLED": "⚫",
            "UNKNOWN": "❓",
        }.get(job.status, "❓")

        jobs_data.append(
            {
                "Select": job.job_id == st.session_state.selected_job_id,
                "Job ID": job.job_id,
                "Name": job.job_name,
                "Partition": job.partition,
                "Status": f"{status_emoji} {job.status}",
                "Step": current_step,
                "Train Loss": train_loss,
                "Val Loss": val_loss,
                "Submitted": job.submit_time[:19] if job.submit_time else "N/A",
            }
        )

    df = pd.DataFrame(jobs_data)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col2:
        if st.button("Refresh Jobs", use_container_width=True):
            st.session_state.job_manager.update_job_statuses()
            st.rerun()

    with col3:
        if st.session_state.selected_job_id:
            job = st.session_state.job_manager.get_job(st.session_state.selected_job_id)
            if job and job.status in ["PENDING", "RUNNING"]:
                if st.button(
                    "Cancel Selected Job", use_container_width=True, type="secondary"
                ):
                    success, msg = st.session_state.job_manager.cancel_job(
                        st.session_state.selected_job_id
                    )
                    if success:
                        st.success(msg)
                    else:
                        st.error(msg)
                    st.rerun()

    # Use data_editor for interactive checkboxes
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        hide_index=True,
        disabled=[
            "Job ID",
            "Name",
            "Partition",
            "Status",
            "Step",
            "Train Loss",
            "Val Loss",
            "Submitted",
        ],
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", default=False),
            "Job ID": st.column_config.TextColumn("Job ID", width="small"),
            "Name": st.column_config.TextColumn("Name", width="medium"),
            "Partition": st.column_config.TextColumn("Partition", width="small"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Step": st.column_config.TextColumn("Step", width="small"),
            "Train Loss": st.column_config.TextColumn("Train Loss", width="small"),
            "Val Loss": st.column_config.TextColumn("Val Loss", width="small"),
            "Submitted": st.column_config.TextColumn("Submitted", width="medium"),
        },
        key="jobs_table_editor",
    )

    # Check if selection changed in the table
    selected_rows = edited_df[edited_df["Select"] == True]
    if len(selected_rows) > 0:
        # Get the most recently selected job (last one if multiple)
        new_selected_id = selected_rows.iloc[-1]["Job ID"]
        if new_selected_id != st.session_state.selected_job_id:
            st.session_state.selected_job_id = new_selected_id
            st.rerun()
    elif len(selected_rows) == 0 and st.session_state.selected_job_id is not None:
        # All checkboxes unchecked - only clear if user explicitly unchecked
        original_selected = df[df["Select"] == True]
        if len(original_selected) > 0:
            # User unchecked the previously selected job
            st.session_state.selected_job_id = None
            st.rerun()


def render_job_details(job_info: JobInfo, eval_every: int):
    """Render detailed view for a selected job."""
    if not job_info:
        st.info("Select a job from the table above to view details.")
        return

    st.subheader(f"Job Details: {job_info.job_name}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Job ID", job_info.job_id)
    with col2:
        st.metric("Partition", job_info.partition)
    with col3:
        st.metric("Nodes", job_info.nodes)
    with col4:
        status_emoji = {
            "PENDING": "🟡",
            "RUNNING": "🟢",
            "COMPLETED": "✅",
            "FAILED": "❌",
            "CANCELLED": "⚫",
        }.get(job_info.status, "❓")
        st.metric("Status", f"{status_emoji} {job_info.status}")

    training_data = get_job_training_data(job_info)

    if training_data and training_data.get("current"):
        current = training_data["current"]
        max_steps = job_info.config.get("max_steps", 0)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Step", f"{current['step']:,}")
        with col2:
            st.metric("Training Loss", f"{current['training_loss']:.4f}")
        with col3:
            st.metric("Validation Loss", f"{current['val_loss']:.4f}")
        with col4:
            st.metric("Learning Rate", f"{current['lr']:.2e}")

        if max_steps > 0:
            progress = current["step"] / max_steps
            st.progress(min(progress, 1.0))
            st.caption(
                f"Progress: {current['step']:,} / {max_steps:,} steps ({progress*100:.1f}%)"
            )

        history_df = pd.DataFrame(training_data["all"])
        history_df = history_df.drop_duplicates(subset=["step"], keep="last")

        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            fig_loss = create_loss_plot(
                history_df,
                eval_every=eval_every,
                title_suffix=f" - {job_info.job_name}",
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        with plot_col2:
            fig_lr = create_lr_plot(history_df, title_suffix=f" - {job_info.job_name}")
            st.plotly_chart(fig_lr, use_container_width=True)
    else:
        st.info("Waiting for training data...")

    with st.expander("Job Configuration"):
        st.json(job_info.config)


def render_job_comparison():
    """Render job comparison view."""
    st.subheader("Job Comparison")

    jobs = st.session_state.job_manager.get_all_jobs()
    if len(jobs) < 2:
        st.info("Submit at least 2 jobs to enable comparison.")
        return

    job_options = [f"{j.job_id} - {j.job_name}" for j in jobs]
    selected_jobs = st.multiselect(
        "Select jobs to compare",
        job_options,
        default=job_options[: min(2, len(job_options))],
    )

    if len(selected_jobs) < 2:
        st.warning("Select at least 2 jobs to compare.")
        return

    jobs_data = {}
    for selection in selected_jobs:
        job_id = selection.split(" - ")[0]
        job = st.session_state.job_manager.get_job(job_id)
        if job:
            data = get_job_training_data(job)
            if data:
                jobs_data[job_id] = data

    if jobs_data:
        fig = create_comparison_plot(jobs_data)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Comparison Table")
        comparison_data = []
        for job_id, data in jobs_data.items():
            job = st.session_state.job_manager.get_job(job_id)
            if data and data.get("all"):
                df = pd.DataFrame(data["all"])
                comparison_data.append(
                    {
                        "Job": f"{job_id[:8]} - {job.job_name}",
                        "Final Train Loss": f"{df['training_loss'].iloc[-1]:.4f}",
                        "Best Train Loss": f"{df['training_loss'].min():.4f}",
                        "Final Val Loss": f"{df['val_loss'].iloc[-1]:.4f}",
                        "Best Val Loss": f"{df['val_loss'].min():.4f}",
                        "Steps Completed": df["step"].max(),
                    }
                )

        if comparison_data:
            st.dataframe(
                pd.DataFrame(comparison_data), use_container_width=True, hide_index=True
            )


def main():
    st.title("LLM Fine-tuning Dashboard")
    st.markdown(
        "Real-time monitoring interface for LLM fine-tuning with SLURM job dispatch"
    )

    with st.sidebar:
        st.header("Configuration")

        st.subheader("Execution Mode")
        execution_mode = st.radio(
            "Mode",
            ["SLURM Job", "Local Process"],
            index=0 if st.session_state.execution_mode == "slurm" else 1,
            help="SLURM: Submit jobs to cluster queue. Local: Run directly on this machine.",
        )
        st.session_state.execution_mode = (
            "slurm" if execution_mode == "SLURM Job" else "local"
        )

        if st.session_state.execution_mode == "slurm":
            st.subheader("SLURM Settings")

            if not st.session_state.available_partitions:
                st.session_state.available_partitions = (
                    st.session_state.job_manager.get_available_partitions()
                )

            partitions = st.session_state.available_partitions
            partition_names = [p["name"] for p in partitions]
            partition_descriptions = {
                p["name"]: f"{p['name']} - {p.get('description', 'Unknown')}"
                for p in partitions
            }

            selected_partition_display = st.selectbox(
                "Partition",
                [partition_descriptions[n] for n in partition_names],
                index=0,
                help="SLURM partition to submit the job to",
            )
            selected_partition = selected_partition_display.split(" - ")[0]

            partition_info = next(
                (p for p in partitions if p["name"] == selected_partition), {}
            )
            max_nodes = partition_info.get("max_nodes", 4)

            # Check if this is a non-lb partition (Galaxy)
            is_lb_partition = "lb" in selected_partition.lower()

            if is_lb_partition:
                num_nodes = st.number_input(
                    "Number of Nodes",
                    min_value=1,
                    max_value=max_nodes,
                    value=1,
                    step=1,
                    help=f"Number of nodes to request (max: {max_nodes})",
                )
            else:
                num_nodes = 1  # Galaxy jobs run on single node

            job_name = st.text_input(
                "Job Name",
                value=f"gsm8k_{datetime.now().strftime('%m%d_%H%M')}",
                help="Name for the SLURM job",
            )

            # Mesh shape is determined by partition type
            if is_lb_partition:
                mesh_shape = [8, 1]
            else:
                mesh_shape = [32, 1]
        else:
            st.subheader("Device Settings")
            devices_options = ["N150", "N300", "LoudBox", "Galaxy", "3-tier"]
            selected_devices = st.selectbox("Devices", devices_options, index=2)
            mesh_shape = device_mesh_shapes[selected_devices]
            selected_partition = None
            num_nodes = 1
            job_name = None

        st.subheader("Model Settings")
        # Map display names to model_config paths
        model_config_mapping = {
            "TinyLlama 1.1B": '"model_configs/tinyllama.yaml"',
            "GPT-2": '"model_configs/gpt2s.yaml"',
        }
        model_options = list(model_config_mapping.keys())
        selected_model_display = st.selectbox("Base Model", model_options, index=0)
        selected_model_config = model_config_mapping[selected_model_display]

        st.subheader("Dataset Settings")
        dataset_options = ["gsm8k", "math_qa", "aqua_rat", "svamp", "mawps"]
        selected_dataset = st.selectbox("Dataset", dataset_options, index=0)

        st.subheader("Hyperparameters")

        col_lr1, col_lr2 = st.columns(2)
        with col_lr1:
            min_lr = st.number_input(
                "Min Learning Rate",
                min_value=1e-7,
                max_value=1e-2,
                value=3e-5,
                format="%.2e",
                help="Minimum learning rate (decay target)",
            )
        with col_lr2:
            max_lr = st.number_input(
                "Max Learning Rate",
                min_value=1e-6,
                max_value=1e-2,
                value=1e-4,
                format="%.2e",
                help="Maximum learning rate (peak value)",
            )

        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=64,
                value=64,
                step=1,
                help="Training batch size per device",
            )

            max_steps = st.number_input(
                "Max Steps", min_value=10, max_value=100000, value=60, step=100
            )

        with col2:
            warmup_steps = st.number_input(
                "Warmup Steps", min_value=0, max_value=1000, value=20, step=10
            )

            hold_steps = st.number_input(
                "Hold Steps", min_value=0, max_value=10000, value=40, step=10
            )

        eval_every = st.number_input(
            "Eval Every", min_value=10, max_value=1000, value=20, step=10
        )
        validation_batch_size = st.number_input(
            "Validation Batch Size",
            min_value=1,
            max_value=32,
            value=4,
            step=1,
            help="Validation batch size per device",
        )

        gradient_accumulation = st.number_input(
            "Gradient Accumulation Steps", min_value=1, max_value=128, value=8, step=1
        )

        max_seq_length = st.number_input(
            "Max Sequence Length", min_value=128, max_value=4096, value=512, step=128
        )

        effective_batch_size = (
            batch_size * gradient_accumulation * mesh_shape[0] * mesh_shape[1]
        )
        st.markdown(f"Effective batch size: **{effective_batch_size}**")
        st.markdown(
            f"Effective num. tokens per batch: **{effective_batch_size * max_seq_length}**"
        )

        st.divider()

        st.subheader(
            "Submit Job"
            if st.session_state.execution_mode == "slurm"
            else "Training Control"
        )

        config = {
            "min_lr": min_lr,
            "max_lr": max_lr,
            "batch_size": batch_size,
            "validation_batch_size": validation_batch_size,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "hold_steps": hold_steps,
            "eval_every": eval_every,
            "gradient_accumulation": gradient_accumulation,
            "max_seq_length": max_seq_length,
            "enable_ddp": mesh_shape != [1, 1],
            "mesh_shape": mesh_shape,
            "model": selected_model_display,
            "model_config": selected_model_config,
            "dataset": selected_dataset,
        }

        if st.session_state.execution_mode == "slurm":
            if st.button("Submit SLURM Job", use_container_width=True, type="primary"):
                success, message, job_info = st.session_state.job_manager.submit_job(
                    config=config,
                    partition=selected_partition,
                    nodes=num_nodes,
                    job_name=job_name,
                )
                if success:
                    st.success(message)
                    st.session_state.selected_job_id = job_info.job_id
                    st.rerun()
                else:
                    st.error(message)
        else:
            is_running = check_local_training_status()

            if is_running:
                st.success("Training is running")
            else:
                st.info("Training is not running")

            col_start, col_stop = st.columns(2)

            with col_start:
                if st.button(
                    "Start Training", use_container_width=True, disabled=is_running
                ):
                    config[
                        "yaml_dir"
                    ] = f"{os.environ.get('TT_METAL_HOME', '.')}/tt-train/configs"
                    success, message = start_local_training(config)
                    if success:
                        st.session_state.data_collection_paused = False
                        st.success(message)
                        st.session_state.last_update = 0
                        st.rerun()
                    else:
                        st.error(message)

            with col_stop:
                if st.button(
                    "Stop Training", use_container_width=True, disabled=not is_running
                ):
                    success, message = stop_local_training()
                    if success:
                        st.warning(message)
                        st.rerun()
                    else:
                        st.error(message)

        st.divider()

        with st.expander("Configuration Summary"):
            st.json(config)

        st.divider()
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider(
            "Refresh Interval (seconds)", min_value=1, max_value=10, value=5, step=1
        )

        if st.button("Refresh Now", use_container_width=True):
            st.session_state.last_update = 0
            st.rerun()

    if st.session_state.execution_mode == "slurm":
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["Jobs Overview", "Job Details", "Comparison", "Logs", "About"]
        )

        with tab1:
            render_jobs_table()

        with tab2:
            if st.session_state.selected_job_id:
                job = st.session_state.job_manager.get_job(
                    st.session_state.selected_job_id
                )
                render_job_details(job, eval_every)
            else:
                st.info("Select a job from the Jobs Overview tab to view details.")

        with tab3:
            render_job_comparison()

        with tab4:
            st.header("Job Logs")

            if st.session_state.selected_job_id:
                job = st.session_state.job_manager.get_job(
                    st.session_state.selected_job_id
                )
                if job:
                    job_dir = Path(job.output_dir)

                    slurm_out_files = list(job_dir.glob("slurm_*.out"))
                    slurm_err_files = list(job_dir.glob("slurm_*.err"))

                    if slurm_out_files:
                        st.subheader("SLURM Output")
                        for out_file in sorted(slurm_out_files):
                            with open(out_file, "r") as f:
                                content = f.read()
                            if content:
                                lines = content.split("\n")
                                last_lines = lines[-100:] if len(lines) > 100 else lines
                                st.text_area(
                                    f"Output ({out_file.name})",
                                    "\n".join(last_lines),
                                    height=300,
                                    disabled=True,
                                )

                    if slurm_err_files:
                        st.subheader("SLURM Errors")
                        for err_file in sorted(slurm_err_files):
                            with open(err_file, "r") as f:
                                content = f.read()
                            if content.strip():
                                st.text_area(
                                    f"Errors ({err_file.name})",
                                    content,
                                    height=200,
                                    disabled=True,
                                )
                            else:
                                st.success(f"No errors in {err_file.name}")

                    if not slurm_out_files and not slurm_err_files:
                        st.info("No log files found yet. Job may be pending.")
            else:
                st.info("Select a job to view its logs.")

        with tab5:
            render_about_section()

    else:
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Training Progress", "Validation Output", "Training Logs", "About"]
        )

        with tab1:
            render_local_training_progress(
                eval_every, max_steps, batch_size, max_seq_length, gradient_accumulation
            )

        with tab2:
            render_validation_output("validation.txt")

        with tab3:
            render_local_training_logs()

        with tab4:
            render_about_section()

    if auto_refresh:
        current_time = time.time()
        if current_time - st.session_state.last_update >= refresh_interval:
            st.session_state.last_update = current_time
            time.sleep(refresh_interval)
            st.rerun()


def render_local_training_progress(
    eval_every, max_steps, batch_size, max_seq_length, gradient_accumulation
):
    """Render training progress for local execution mode."""
    if st.session_state.data_collection_paused:
        st.warning(
            "Data collection is paused. Click 'Resume Data Collection' to start monitoring again."
        )
        if st.button("Resume Data Collection", type="primary"):
            st.session_state.data_collection_paused = False
            st.rerun()

    current_data = None
    output_file = "output.txt"

    if not st.session_state.data_collection_paused:
        parsed_data = parse_output_file(output_file)

        if parsed_data:
            current_data = parsed_data["current"]
            all_data = parsed_data["all"]

            if len(st.session_state.training_history) == 0:
                st.session_state.training_history = all_data
                st.session_state.last_step = current_data["step"]
            else:
                for data_point in all_data:
                    if data_point["step"] > st.session_state.last_step:
                        st.session_state.training_history.append(data_point)
                        st.session_state.last_step = data_point["step"]

    col1, col2, col3, col4 = st.columns(4)

    if current_data:
        metrics = [
            (
                "Current Step",
                f"{current_data['step']:,}",
                f"Epoch {current_data['epoch']}",
            ),
            ("Training Loss", f"{current_data['training_loss']:.4f}", None),
            ("Validation Loss", f"{current_data['val_loss']:.4f}", None),
            ("Learning Rate", f"{current_data['lr']:.2e}", None),
        ]

        for col, (label, value, delta) in zip([col1, col2, col3, col4], metrics):
            with col:
                if delta:
                    st.metric(label, value, delta=delta)
                else:
                    st.metric(label, value)

        if max_steps > 0:
            progress = current_data["step"] / max_steps
            st.progress(min(progress, 1.0))
            st.caption(
                f"Progress: {current_data['step']:,} / {max_steps:,} steps ({progress*100:.1f}%)"
            )
    else:
        for col, label in zip(
            [col1, col2, col3, col4],
            ["Current Step", "Training Loss", "Validation Loss", "Learning Rate"],
        ):
            with col:
                st.metric(label, "N/A")

        st.info(
            "Waiting for training data... Make sure `output.txt` is being generated."
        )

    st.divider()

    if st.session_state.training_history:
        history_df = pd.DataFrame(st.session_state.training_history)
        history_df = history_df.drop_duplicates(subset=["step"], keep="last")

        st.subheader("Training Statistics")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Best Training Loss",
                f"{history_df['training_loss'].min():.4f}",
                delta=f"Step {history_df.loc[history_df['training_loss'].idxmin(), 'step']}",
            )

        with col2:
            st.metric(
                "Best Validation Loss",
                f"{history_df['val_loss'].min():.4f}",
                delta=f"Step {history_df.loc[history_df['val_loss'].idxmin(), 'step']}",
            )

        with col3:
            st.metric("Total Steps", f"{len(history_df):,}")

        with col4:
            if len(history_df) > 1:
                time_diff = (
                    history_df["timestamp"].iloc[-1] - history_df["timestamp"].iloc[0]
                ).total_seconds()
                step_diff = history_df["step"].iloc[-1] - history_df["step"].iloc[0]

                if time_diff > 0 and step_diff > 0:
                    tokens_per_step = (
                        batch_size * max_seq_length * gradient_accumulation
                    )
                    total_tokens = step_diff * tokens_per_step
                    tokens_per_sec = total_tokens / time_diff

                    if tokens_per_sec >= 1000:
                        st.metric("Rate", f"{tokens_per_sec/1000:.2f}k tokens/s")
                    else:
                        st.metric("Rate", f"{tokens_per_sec:.2f} tokens/s")
                else:
                    st.metric("Rate", "N/A")
            else:
                st.metric("Rate", "N/A")

        st.divider()

        plot_col1, plot_col2 = st.columns(2)

        with plot_col1:
            fig_loss = create_loss_plot(history_df, eval_every=eval_every)
            st.plotly_chart(fig_loss, use_container_width=True)

        with plot_col2:
            fig_lr = create_lr_plot(history_df)
            st.plotly_chart(fig_lr, use_container_width=True)
    else:
        st.info("Training plots will appear here once data is available.")


def render_validation_output(validation_file):
    """Render validation output tab."""
    st.header("Latest Validation Output")

    if st.session_state.data_collection_paused:
        st.warning("Data collection is paused.")
        st.info("No validation data available (data collection paused).")
    else:
        validation_content = parse_validation_file(validation_file)

        if (
            validation_content
            and validation_content != "No validation data available yet."
        ):
            st.text_area(
                "Validation Results", validation_content, height=500, disabled=True
            )

            st.download_button(
                label="Download Validation Output",
                data=validation_content,
                file_name=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )
        else:
            st.info("Waiting for validation data...")


def render_local_training_logs():
    """Render training logs for local execution mode."""
    st.header("Training Logs")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    stdout_log = os.path.join(script_dir, "training_stdout.log")
    stderr_log = os.path.join(script_dir, "training_stderr.log")

    if os.path.exists(stdout_log):
        st.subheader("Standard Output")
        try:
            with open(stdout_log, "r") as f:
                stdout_content = f.read()
                if stdout_content:
                    lines = stdout_content.split("\n")
                    last_lines = lines[-100:] if len(lines) > 100 else lines
                    st.text_area(
                        "Training Output (last 100 lines)",
                        "\n".join(last_lines),
                        height=400,
                        disabled=True,
                    )

                    st.download_button(
                        label="Download Full Output Log",
                        data=stdout_content,
                        file_name=f"training_stdout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                        mime="text/plain",
                    )
                else:
                    st.info("Output log is empty")
        except Exception as e:
            st.error(f"Error reading output log: {e}")
    else:
        st.info("No output log file found. Start training to generate logs.")

    st.divider()

    if os.path.exists(stderr_log):
        st.subheader("Error Output")
        try:
            with open(stderr_log, "r") as f:
                stderr_content = f.read()
                if stderr_content:
                    st.text_area("Error Log", stderr_content, height=200, disabled=True)

                    st.download_button(
                        label="Download Error Log",
                        data=stderr_content,
                        file_name=f"training_stderr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
                        mime="text/plain",
                    )
                else:
                    st.success("No errors reported")
        except Exception as e:
            st.error(f"Error reading error log: {e}")

    st.divider()

    render_tt_smi_section()


def render_tt_smi_section():
    """Render TT-SMI system status section."""
    st.subheader("TT-SMI System Status")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption("Shows Tenstorrent board info and telemetry")
    with col2:
        refresh_smi = st.button("Refresh TT-SMI", use_container_width=True)

    if refresh_smi:
        try:
            result = subprocess.run(
                ["tt-smi", "-s"], capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                smi_output = result.stdout
                if smi_output:
                    try:
                        smi_data = json.loads(smi_output)
                        smi_data = smi_data["device_info"][0]
                        filtered_data = {}
                        if "board_info" in smi_data:
                            filtered_data["board_info"] = smi_data["board_info"]
                        if "telemetry" in smi_data:
                            filtered_data["telemetry"] = smi_data["telemetry"]

                        if filtered_data:
                            st.session_state.tt_smi_data = filtered_data
                            st.session_state.tt_smi_error = None
                        else:
                            st.session_state.tt_smi_data = None
                            st.session_state.tt_smi_error = (
                                "No board_info or telemetry data found"
                            )
                    except json.JSONDecodeError as e:
                        st.session_state.tt_smi_data = None
                        st.session_state.tt_smi_error = (
                            f"Failed to parse tt-smi output: {e}"
                        )
                else:
                    st.session_state.tt_smi_data = None
                    st.session_state.tt_smi_error = "tt-smi produced no output"
            else:
                st.session_state.tt_smi_data = None
                st.session_state.tt_smi_error = (
                    f"tt-smi failed with return code {result.returncode}"
                )
        except subprocess.TimeoutExpired:
            st.session_state.tt_smi_data = None
            st.session_state.tt_smi_error = "tt-smi command timed out"
        except FileNotFoundError:
            st.session_state.tt_smi_data = None
            st.session_state.tt_smi_error = "tt-smi command not found"
        except Exception as e:
            st.session_state.tt_smi_data = None
            st.session_state.tt_smi_error = f"Error running tt-smi: {str(e)}"

    if st.session_state.tt_smi_data:
        formatted_json = json.dumps(st.session_state.tt_smi_data, indent=2)
        st.code(formatted_json, language="json")
    elif st.session_state.tt_smi_error:
        st.warning(st.session_state.tt_smi_error)
    else:
        st.info("Click 'Refresh TT-SMI' to view system status")


def render_about_section():
    """Render the About section."""
    st.header("About This Dashboard")

    st.markdown(
        """
    ### LLM Fine-tuning Dashboard

    This Streamlit application provides real-time monitoring for LLM fine-tuning experiments
    with support for SLURM job dispatch and multi-job monitoring.

    #### Features:
    - **SLURM Job Dispatch**: Submit training jobs to cluster partitions
    - **Multi-Job Monitoring**: Track and compare multiple training jobs
    - **Real-time Monitoring**: Live updates of training and validation loss
    - **Interactive Plots**: Zoom, pan, and hover over data points
    - **Job Comparison**: Compare training curves across different jobs
    - **Local Execution**: Option to run training locally without SLURM

    #### Execution Modes:

    **SLURM Mode:**
    - Submit jobs to cluster queue
    - Select partition and node count
    - Monitor multiple jobs simultaneously
    - Compare job performance

    **Local Mode:**
    - Run training directly on current machine
    - Real-time output monitoring
    - Suitable for development and testing

    #### File Format:

    **output.txt** should contain lines in the following format:
    ```
    LR: 3e-3, training_loss: 0.97, val_loss: 1.01, step: 120, epoch: 1
    ```

    ---

    **Version**: 2.0.0
    **Created**: 2025
    """
    )


if __name__ == "__main__":
    main()
