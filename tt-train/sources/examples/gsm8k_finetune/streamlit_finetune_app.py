#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Streamlit LLM Fine-tuning Application
Real-time monitoring and control interface for LLM fine-tuning.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
import os
import re
from datetime import datetime
import yaml
import subprocess
import json

device_mesh_shapes = {
    "N150": [1, 1],
    "N300": [1, 2],
    "LoudBox": [1, 8],
    "Galaxy": [1, 32],
    "3-tier": [5, 32],  # Example shape for 5-Galaxy system
}

# Set page configuration
st.set_page_config(
    page_title="LLM Fine-tuning Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
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

        # Parse the format: "LR: 3e-3, training_loss: 0.97, val_loss: 1.01, step: 120, epoch: 1"
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

        return {
            "current": all_data[-1],
            "all": all_data,
        }  # Most recent data point  # All data points

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


def create_loss_plot(history_df, eval_every=None):
    """Create an interactive plot for training and validation loss."""
    fig = go.Figure()

    # Training loss
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

    # Validation loss - filter to only show at eval_every intervals
    if eval_every is not None and eval_every > 0:
        # Filter to only show validation loss at eval_every intervals
        # This includes step 0 and steps that are multiples of eval_every
        val_mask = ((history_df["step"] + 1) % eval_every == 0) | (
            history_df["step"] == 0
        )
        val_df = history_df[val_mask]
    else:
        # If eval_every not provided, show all validation points
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

    # Update layout
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(title_text="Loss")

    fig.update_layout(
        title="Loss over Steps",
        height=500,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_lr_plot(history_df):
    """Create an interactive plot for learning rate."""
    fig = go.Figure()

    # Learning rate
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

    # Update layout
    fig.update_xaxes(title_text="Step")
    fig.update_yaxes(
        title_text="Learning Rate", type="log", exponentformat="e", showexponent="all"
    )

    fig.update_layout(
        title="Learning Rate over Steps",
        height=500,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_training_yaml(
    config_dict,
    output_path="/home/ubuntu/tt-metal/tt-train/configs/training_overrides.yaml",
):
    """Create a training YAML configuration file."""
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

    # Write YAML with blank lines between top-level sections
    with open(output_path, "w") as f:
        f.write("training_config:\n")
        training_yaml = yaml.dump(
            training_config, default_flow_style=False, sort_keys=False
        )
        # Indent each line
        for line in training_yaml.strip().split("\n"):
            f.write(f"  {line}\n")

        f.write("\n")  # Blank line between sections

        f.write("transformer_config:\n")
        transformer_yaml = yaml.dump(
            transformer_config, default_flow_style=False, sort_keys=False
        )
        # Indent each line
        for line in transformer_yaml.strip().split("\n"):
            f.write(f"  {line}\n")

        f.write("\n")  # Blank line between sections

        f.write("scheduler_config:\n")
        scheduler_yaml = yaml.dump(
            scheduler_config, default_flow_style=False, sort_keys=False
        )
        # Indent each line
        for line in scheduler_yaml.strip().split("\n"):
            f.write(f"  {line}\n")

        f.write("\n")  # Blank line between sections

        f.write("device_config:\n")
        # Handle mesh_shape as flow sequence, others as block style
        f.write(f"  enable_ddp: {device_config['enable_ddp']}\n")
        mesh_shape_flow = yaml.dump(
            device_config["mesh_shape"], default_flow_style=True
        ).strip()
        f.write(f"  mesh_shape: {mesh_shape_flow}\n")

    return output_path


def start_training(config_dict):
    """Start the training process."""
    try:
        # Get the YAML output directory from config, or use default
        yaml_output_dir = config_dict.get(
            "yaml_dir", "/home/ubuntu/tt-metal/tt-train/configs"
        )

        # Ensure the directory exists
        os.makedirs(yaml_output_dir, exist_ok=True)

        # Construct the full path with the fixed filename
        yaml_output_path = os.path.join(yaml_output_dir, "training_overrides.yaml")

        # Create the YAML config file
        yaml_path = create_training_yaml(config_dict, output_path=yaml_output_path)
        yaml_abs_path = os.path.abspath(yaml_path)
        st.success(f"Created configuration file: {yaml_abs_path}")

        # Get the directory containing gsm8k_finetune.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, "gsm8k_finetune.py")

        # Check if script exists
        if not os.path.exists(script_path):
            return False, f"Training script not found at: {script_path}"

        # Create log files for stdout and stderr
        log_dir = script_dir
        stdout_log = os.path.join(log_dir, "training_stdout.log")
        stderr_log = os.path.join(log_dir, "training_stderr.log")

        # Start the training process
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


def stop_training():
    """Stop the training process."""
    try:
        if st.session_state.training_process is not None:
            pid = st.session_state.training_process.pid

            # Send SIGTERM to gracefully stop the process
            st.session_state.training_process.terminate()

            # Wait for a few seconds for graceful shutdown
            try:
                st.session_state.training_process.wait(timeout=5)
                message = f"Training stopped gracefully (PID: {pid})"
            except subprocess.TimeoutExpired:
                # If it doesn't stop, force kill
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


def check_training_status():
    """Check if the training process is still running."""
    if st.session_state.training_process is not None:
        poll = st.session_state.training_process.poll()
        if poll is not None:
            # Process has ended
            st.session_state.training_running = False
            st.session_state.training_process = None
    return st.session_state.training_running


def main():
    # Header
    st.title("LLM Fine-tuning Dashboard")
    st.markdown("Real-time monitoring interface for LLM fine-tuning on GSM8K dataset")

    # Sidebar - Configuration
    with st.sidebar:
        st.header("Configuration")

        # Model selection
        st.subheader("Model Settings")
        model_options = [
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            "meta-llama/Llama-2-7b-hf",
            "gpt2",
            "gpt2-medium",
            "gpt2-large",
            "microsoft/phi-2",
        ]
        selected_model = st.selectbox("Base Model", model_options, index=0)

        # Device selection
        st.subheader("Device Settings")
        devices_options = [
            "N150",  # [1,1]
            "N300",  # [1,2]
            "LoudBox",  # [1, 8]
            "Galaxy",  # [1, 32]
            "3-tier",  # [???]
        ]
        selected_devices = st.selectbox("Devices", devices_options, index=2)
        # Dataset selection
        st.subheader("Dataset Settings")
        dataset_options = ["gsm8k", "math_qa", "aqua_rat", "svamp", "mawps"]
        selected_dataset = st.selectbox("Dataset", dataset_options, index=0)

        # Hyperparameters
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
            batch_size
            * gradient_accumulation
            * device_mesh_shapes[selected_devices][0]
            * device_mesh_shapes[selected_devices][1]
        )
        st.markdown(f"Effective batch size: **{effective_batch_size}**")
        st.markdown(
            f"Effective num. tokens per batch: **{effective_batch_size * max_seq_length}**"
        )

        st.divider()

        # Training Control
        st.subheader("Training Control")

        # YAML output directory configuration
        yaml_output_dir = st.text_input(
            "Override YAML Directory",
            value=f"{os.environ['TT_METAL_HOME']}/tt-train/configs/",
            help="Directory where training_overrides.yaml will be saved",
        )

        # Check current training status
        is_running = check_training_status()

        if is_running:
            st.success("Training is running")
        else:
            st.info("Training is not running")

        col_start, col_stop = st.columns(2)

        with col_start:
            if st.button(
                "Start Training", use_container_width=True, disabled=is_running
            ):
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
                    "yaml_dir": yaml_output_dir,
                    "enable_ddp": True if selected_devices != "N150" else False,
                    "mesh_shape": device_mesh_shapes[selected_devices],
                }
                success, message = start_training(config)
                if success:
                    st.session_state.data_collection_paused = (
                        False  # Resume data collection when training starts
                    )
                    st.success(message)
                    st.session_state.last_update = 0  # Force update
                    st.rerun()
                else:
                    st.error(message)

        with col_stop:
            if st.button(
                "Stop Training", use_container_width=True, disabled=not is_running
            ):
                success, message = stop_training()
                if success:
                    st.warning(message)
                    st.rerun()
                else:
                    st.error(message)

        # Show YAML file location if it exists
        if os.path.exists("training_overrides.yaml"):
            st.caption(f"Config: training_overrides.yaml")

        # Show training process info if running
        if is_running and hasattr(st.session_state, "training_pid"):
            st.caption(f"Process ID: {st.session_state.training_pid}")

            # Show log file location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            stdout_log = os.path.join(script_dir, "training_stdout.log")
            if os.path.exists(stdout_log):
                st.caption(f"Log: training_stdout.log")

        st.divider()

        # File paths
        st.subheader("File Paths")
        output_file = st.text_input(
            "Training Output File",
            value="output.txt",
            help="File containing training metrics",
        )

        validation_file = st.text_input(
            "Validation Output File",
            value="validation.txt",
            help="File containing validation results",
        )

        st.divider()

        # Display configuration summary
        with st.expander("Configuration Summary"):
            st.json(
                {
                    "model": selected_model,
                    "dataset": selected_dataset,
                    "min_lr": min_lr,
                    "max_lr": max_lr,
                    "batch_size": batch_size,
                    "max_steps": max_steps,
                    "warmup_steps": warmup_steps,
                    "hold_steps": hold_steps,
                    "eval_every": eval_every,
                    "gradient_accumulation": gradient_accumulation,
                    "max_seq_length": max_seq_length,
                }
            )

        # Refresh controls
        st.divider()
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider(
            "Refresh Interval (seconds)", min_value=1, max_value=10, value=5, step=1
        )

        if st.button("Refresh Now", use_container_width=True):
            st.session_state.last_update = 0  # Force update
            st.rerun()

        if st.button("Reset Dashboard", use_container_width=True, type="secondary"):
            # Clear all training data
            st.session_state.training_history = []
            st.session_state.last_step = -1
            st.session_state.last_update = time.time()
            st.session_state.tt_smi_data = None
            st.session_state.tt_smi_error = None
            st.session_state.data_collection_paused = True

            # Clear output.txt and validation.txt files
            try:
                # Clear output.txt
                if os.path.exists(output_file):
                    with open(output_file, "w") as f:
                        f.write("")

                # Clear validation.txt
                if os.path.exists(validation_file):
                    with open(validation_file, "w") as f:
                        f.write("")
            except Exception as e:
                st.error(f"Error clearing files: {e}")

            # Immediately rerun to refresh the display
            st.rerun()

        st.divider()

        # Download section
        st.subheader("Downloads")
        if st.session_state.training_history:
            history_df = pd.DataFrame(st.session_state.training_history)
            history_df = history_df.drop_duplicates(subset=["step"], keep="last")
            csv_data = history_df.to_csv(index=False)
            st.download_button(
                label="Download Training History (CSV)",
                data=csv_data,
                file_name=f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("No training data to download yet")

        # Debug section
        with st.expander("Debug Info"):
            st.text(f"Output file exists: {os.path.exists(output_file)}")
            st.text(f"Output file path: {os.path.abspath(output_file)}")
            st.text(f"Last step tracked: {st.session_state.last_step}")
            st.text(f"History length: {len(st.session_state.training_history)}")

            # Test parsing
            test_data = parse_output_file(output_file)
            if test_data:
                st.success("File parsed successfully!")
                st.json(
                    {
                        "current_step": test_data["current"]["step"],
                        "total_data_points": len(test_data["all"]),
                        "current_data": test_data["current"],
                    }
                )
            else:
                st.error("Failed to parse file")
                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        st.text("File contents:")
                        st.code(f.read())

    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Training Progress", "Validation Output", "Training Logs", "About"]
    )

    with tab1:
        # Show data collection status and control
        if st.session_state.data_collection_paused:
            st.warning(
                "Data collection is paused. Click 'Resume Data Collection' to start monitoring again."
            )
            if st.button("Resume Data Collection", type="primary"):
                st.session_state.data_collection_paused = False
                st.rerun()

        # Parse current training data and update history (only if not paused)
        current_data = None
        if not st.session_state.data_collection_paused:
            parsed_data = parse_output_file(output_file)

            if parsed_data:
                current_data = parsed_data["current"]
                all_data = parsed_data["all"]

                # If we have no history yet, load all data from file
                if len(st.session_state.training_history) == 0:
                    st.session_state.training_history = all_data
                    st.session_state.last_step = current_data["step"]
                else:
                    # Only append new steps that we don't have yet
                    for data_point in all_data:
                        if data_point["step"] > st.session_state.last_step:
                            st.session_state.training_history.append(data_point)
                            st.session_state.last_step = data_point["step"]

        # Current status metrics
        col1, col2, col3, col4 = st.columns(4)

        if current_data:
            # Display current metrics
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

            # Show training progress
            if max_steps > 0:
                progress = current_data["step"] / max_steps
                st.progress(min(progress, 1.0))
                st.caption(
                    f"Progress: {current_data['step']:,} / {max_steps:,} steps ({progress*100:.1f}%)"
                )
        else:
            metrics = [
                ("Current Step", "N/A"),
                ("Training Loss", "N/A"),
                ("Validation Loss", "N/A"),
                ("Learning Rate", "N/A"),
            ]

            for col, (label, value) in zip([col1, col2, col3, col4], metrics):
                with col:
                    st.metric(label, value)

            st.info(
                "Waiting for training data... Make sure `output.txt` is being generated."
            )

        st.divider()

        # Plot training history
        if st.session_state.training_history:
            history_df = pd.DataFrame(st.session_state.training_history)

            # Remove duplicates based on step
            history_df = history_df.drop_duplicates(subset=["step"], keep="last")

            # Show statistics
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
                # Calculate tokens per second
                if len(history_df) > 1:
                    # Get time difference between first and last step
                    time_diff = (
                        history_df["timestamp"].iloc[-1]
                        - history_df["timestamp"].iloc[0]
                    ).total_seconds()
                    step_diff = history_df["step"].iloc[-1] - history_df["step"].iloc[0]

                    if time_diff > 0 and step_diff > 0:
                        # Calculate tokens per second
                        # tokens_per_step = batch_size * max_seq_length * gradient_accumulation
                        tokens_per_step = (
                            batch_size * max_seq_length * gradient_accumulation
                        )
                        total_tokens = step_diff * tokens_per_step
                        tokens_per_sec = total_tokens / time_diff

                        # Format with appropriate units
                        if tokens_per_sec >= 1000:
                            st.metric("Rate", f"{tokens_per_sec/1000:.2f}k tokens/s")
                        else:
                            st.metric("Rate", f"{tokens_per_sec:.2f} tokens/s")
                    else:
                        st.metric("Rate", "N/A")
                else:
                    st.metric("Rate", "N/A")

            st.divider()

            # Create and display plots side by side
            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                fig_loss = create_loss_plot(history_df, eval_every=eval_every)
                st.plotly_chart(fig_loss, use_container_width=True)

            with plot_col2:
                fig_lr = create_lr_plot(history_df)
                st.plotly_chart(fig_lr, use_container_width=True)
        else:
            st.info("Training plots will appear here once data is available.")

    with tab2:
        st.header("Latest Validation Output")

        # Show data collection status if paused
        if st.session_state.data_collection_paused:
            st.warning(
                "Data collection is paused. Click 'Resume Data Collection' in the Training Progress tab to start monitoring again."
            )
            st.info("No validation data available (data collection paused).")
        else:
            # Parse and display validation content
            validation_content = parse_validation_file(validation_file)

            if (
                validation_content
                and validation_content != "No validation data available yet."
            ):
                st.text_area(
                    "Validation Results", validation_content, height=500, disabled=True
                )

                # Download validation output
                st.download_button(
                    label="Download Validation Output",
                    data=validation_content,
                    file_name=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )
            else:
                st.info(
                    "Waiting for validation data... Make sure `validation.txt` is being generated."
                )

    with tab3:
        st.header("Training Logs")

        script_dir = os.path.dirname(os.path.abspath(__file__))
        stdout_log = os.path.join(script_dir, "training_stdout.log")
        stderr_log = os.path.join(script_dir, "training_stderr.log")

        # Display stdout log
        if os.path.exists(stdout_log):
            st.subheader("Standard Output")
            try:
                with open(stdout_log, "r") as f:
                    stdout_content = f.read()
                    if stdout_content:
                        # Show last 100 lines
                        lines = stdout_content.split("\n")
                        last_lines = lines[-100:] if len(lines) > 100 else lines
                        st.text_area(
                            "Training Output (last 100 lines)",
                            "\n".join(last_lines),
                            height=400,
                            disabled=True,
                        )

                        # Download button
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

        # Display stderr log
        if os.path.exists(stderr_log):
            st.subheader("Error Output")
            try:
                with open(stderr_log, "r") as f:
                    stderr_content = f.read()
                    if stderr_content:
                        st.text_area(
                            "Error Log", stderr_content, height=200, disabled=True
                        )

                        # Download button
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

        # Display TT-SMI output
        st.subheader("TT-SMI System Status")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("Shows Tenstorrent board info and telemetry")
        with col2:
            refresh_smi = st.button("Refresh TT-SMI", use_container_width=True)

        # Only run tt-smi when button is clicked
        if refresh_smi:
            try:
                # Run tt-smi -s command
                result = subprocess.run(
                    ["tt-smi", "-s"], capture_output=True, text=True, timeout=10
                )

                if result.returncode == 0:
                    smi_output = result.stdout
                    if smi_output:
                        try:
                            # Parse JSON output
                            smi_data = json.loads(smi_output)
                            smi_data = smi_data["device_info"][0]
                            # Extract board_info and telemetry
                            filtered_data = {}
                            if "board_info" in smi_data:
                                filtered_data["board_info"] = smi_data["board_info"]
                            if "telemetry" in smi_data:
                                filtered_data["telemetry"] = smi_data["telemetry"]

                            if filtered_data:
                                # Store in session state
                                st.session_state.tt_smi_data = filtered_data
                                st.session_state.tt_smi_error = None
                            else:
                                st.session_state.tt_smi_data = None
                                st.session_state.tt_smi_error = "No board_info or telemetry data found in tt-smi output"
                        except json.JSONDecodeError as e:
                            st.session_state.tt_smi_data = None
                            st.session_state.tt_smi_error = f"Failed to parse tt-smi output as JSON: {e}\n\nRaw Output:\n{smi_output}"
                    else:
                        st.session_state.tt_smi_data = None
                        st.session_state.tt_smi_error = "tt-smi produced no output"

                    # Store stderr if there are warnings
                    if result.stderr:
                        if st.session_state.tt_smi_error:
                            st.session_state.tt_smi_error += (
                                f"\n\nWarnings:\n{result.stderr}"
                            )
                        else:
                            st.session_state.tt_smi_error = (
                                f"Warnings:\n{result.stderr}"
                            )
                else:
                    st.session_state.tt_smi_data = None
                    st.session_state.tt_smi_error = (
                        f"tt-smi command failed with return code {result.returncode}"
                    )
                    if result.stderr:
                        st.session_state.tt_smi_error += (
                            f"\n\nError Output:\n{result.stderr}"
                        )
            except subprocess.TimeoutExpired:
                st.session_state.tt_smi_data = None
                st.session_state.tt_smi_error = (
                    "tt-smi command timed out after 10 seconds"
                )
            except FileNotFoundError:
                st.session_state.tt_smi_data = None
                st.session_state.tt_smi_error = "tt-smi command not found. Make sure Tenstorrent tools are installed and in PATH."
            except Exception as e:
                st.session_state.tt_smi_data = None
                st.session_state.tt_smi_error = f"Error running tt-smi: {str(e)}"

        # Display the current TT-SMI data from session state
        if st.session_state.tt_smi_data:
            formatted_json = json.dumps(st.session_state.tt_smi_data, indent=2)
            st.code(formatted_json, language="json")
        elif st.session_state.tt_smi_error:
            if (
                "Failed to parse" in st.session_state.tt_smi_error
                or "Raw Output" in st.session_state.tt_smi_error
            ):
                parts = st.session_state.tt_smi_error.split("Raw Output:")
                st.error(parts[0])
                if len(parts) > 1:
                    st.text_area(
                        "Raw Output", parts[1].strip(), height=300, disabled=True
                    )
            elif (
                "Warnings:" in st.session_state.tt_smi_error
                or "Error Output:" in st.session_state.tt_smi_error
            ):
                parts = st.session_state.tt_smi_error.split("\n\n")
                st.error(parts[0])
                if len(parts) > 1:
                    with st.expander("Details"):
                        st.text("\n\n".join(parts[1:]))
            else:
                if "not found" in st.session_state.tt_smi_error:
                    st.warning(st.session_state.tt_smi_error)
                elif (
                    "No board_info" in st.session_state.tt_smi_error
                    or "no output" in st.session_state.tt_smi_error
                ):
                    st.info(st.session_state.tt_smi_error)
                else:
                    st.error(st.session_state.tt_smi_error)
        else:
            st.info("Click 'Refresh TT-SMI' to view system status")

    with tab4:
        st.header("About This Dashboard")

        st.markdown(
            """
        ### LLM Fine-tuning Dashboard

        This Streamlit application provides real-time monitoring for LLM fine-tuning experiments.

        #### Features:
        - **Real-time Monitoring**: Live updates of training and validation loss
        - **Interactive Plots**: Zoom, pan, and hover over data points for detailed information
        - **Hyperparameter Configuration**: Easy-to-use interface for setting training parameters
        - **Validation Output**: Display of latest validation results
        - **Data Export**: Download training history and validation outputs
        - **No Auto-Scroll**: Updates happen in place without jumping to the top of the page

        #### File Format:

        **output.txt** should contain lines in the following format:
        ```
        LR: 3e-3, training_loss: 0.97, val_loss: 1.01, step: 120, epoch: 1
        ```

        **validation.txt** can contain any text content (e.g., model outputs, metrics, examples).

        #### Usage:
        1. Configure your training parameters in the sidebar
        2. Start your training script (it should write to `output.txt` and `validation.txt`)
        3. Monitor progress in real-time on this dashboard
        4. Download results for further analysis

        #### Tips:
        - Enable auto-refresh for continuous monitoring
        - Adjust refresh interval based on your training speed
        - Use the configuration summary to verify your settings
        - Download training history regularly for backup

        ---

        **Version**: 1.1.0
        **Created**: 2025
        """
        )

    # Auto-refresh logic using st.rerun() with time check
    if auto_refresh:
        current_time = time.time()
        if current_time - st.session_state.last_update >= refresh_interval:
            st.session_state.last_update = current_time
            time.sleep(refresh_interval)
            st.rerun()


if __name__ == "__main__":
    main()
