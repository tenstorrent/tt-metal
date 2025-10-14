#!/usr/bin/env python3
"""
Streamlit LLM Fine-tuning Application
Real-time monitoring and control interface for LLM fine-tuning.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import re
from datetime import datetime
from pathlib import Path
import yaml
import subprocess
import signal

# Set page configuration
st.set_page_config(page_title="LLM Fine-tuning Dashboard", layout="wide", initial_sidebar_state="expanded")

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


def parse_output_file(file_path="output.txt"):
    """Parse the output.txt file to extract training metrics."""
    if not os.path.exists(file_path):
        return None

    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        # Get the last non-empty line with data
        if not lines:
            return None

        # Find the last non-empty line
        last_line = None
        for line in reversed(lines):
            if line.strip():
                last_line = line.strip()
                break

        if not last_line:
            return None

        # Parse the format: "LR: 3e-3, training_loss: 0.97, val_loss: 1.01, step: 120, epoch: 1"
        pattern = (
            r"LR:\s*([\d.e+-]+),\s*training_loss:\s*([\d.]+),\s*val_loss:\s*([\d.]+),\s*step:\s*(\d+),\s*epoch:\s*(\d+)"
        )
        match = re.search(pattern, last_line)

        if match:
            lr, train_loss, val_loss, step, epoch = match.groups()
            return {
                "lr": float(lr),
                "training_loss": float(train_loss),
                "val_loss": float(val_loss),
                "step": int(step),
                "epoch": int(epoch),
                "timestamp": datetime.now(),
            }
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


def create_loss_plot(history_df):
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

    # Validation loss
    fig.add_trace(
        go.Scatter(
            x=history_df["step"],
            y=history_df["val_loss"],
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
    fig.update_yaxes(title_text="Learning Rate", type="log", exponentformat="e", showexponent="all")

    fig.update_layout(
        title="Learning Rate over Steps",
        height=500,
        showlegend=True,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=50, r=50, t=80, b=50),
    )

    return fig


def create_training_yaml(config_dict, output_path="training_overrides.yaml"):
    """Create a training YAML configuration file."""
    training_config = {
        "batch_size": config_dict["batch_size"],
        "max_steps": config_dict["max_steps"],
        "learning_rate": config_dict["learning_rate"],
        "gradient_accumulation_steps": config_dict["gradient_accumulation"],
        "eval_every": config_dict["eval_every"],
        "transformer_config": {"max_sequence_length": config_dict["max_seq_length"]},
    }

    scheduler_config = {"warmup_steps": config_dict["warmup_steps"]}

    # Write YAML with blank lines between top-level sections
    with open(output_path, "w") as f:
        f.write("training_config:\n")
        training_yaml = yaml.dump(training_config, default_flow_style=False, sort_keys=False)
        # Indent each line
        for line in training_yaml.strip().split("\n"):
            f.write(f"  {line}\n")

        f.write("\n")  # Blank line between sections

        f.write("scheduler_config:\n")
        scheduler_yaml = yaml.dump(scheduler_config, default_flow_style=False, sort_keys=False)
        # Indent each line
        for line in scheduler_yaml.strip().split("\n"):
            f.write(f"  {line}\n")

    return output_path


def start_training(config_dict):
    """Start the training process."""
    try:
        # Create the YAML config file
        yaml_path = create_training_yaml(config_dict)
        st.success(f"Created configuration file: {yaml_path}")

        # Start the training process
        # Note: This assumes gsm8k_finetune.py can be run with the config
        # Adjust the command based on your actual training script
        cmd = ["python", "gsm8k_finetune.py", "--config", yaml_path]

        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.path.dirname(os.path.abspath(__file__))
        )

        st.session_state.training_process = process
        st.session_state.training_running = True

        return True, "Training started successfully!"
    except Exception as e:
        return False, f"Error starting training: {str(e)}"


def stop_training():
    """Stop the training process."""
    try:
        if st.session_state.training_process is not None:
            # Send SIGTERM to gracefully stop the process
            st.session_state.training_process.terminate()

            # Wait for a few seconds for graceful shutdown
            try:
                st.session_state.training_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # If it doesn't stop, force kill
                st.session_state.training_process.kill()
                st.session_state.training_process.wait()

            st.session_state.training_process = None
            st.session_state.training_running = False

            return True, "Training stopped successfully!"
        else:
            return False, "No training process to stop"
    except Exception as e:
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

        # Dataset selection
        st.subheader("Dataset Settings")
        dataset_options = ["gsm8k", "math_qa", "aqua_rat", "svamp", "mawps"]
        selected_dataset = st.selectbox("Dataset", dataset_options, index=0)

        # Hyperparameters
        st.subheader("Hyperparameters")

        learning_rate = st.number_input(
            "Learning Rate",
            min_value=1e-6,
            max_value=1e-2,
            value=3e-4,
            format="%.2e",
            help="Initial learning rate for training",
        )

        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=4, step=1)

            max_steps = st.number_input("Max Steps", min_value=100, max_value=100000, value=1000, step=100)

        with col2:
            warmup_steps = st.number_input("Warmup Steps", min_value=0, max_value=1000, value=20, step=10)

            eval_every = st.number_input("Eval Every", min_value=10, max_value=1000, value=100, step=10)

        gradient_accumulation = st.number_input(
            "Gradient Accumulation Steps", min_value=1, max_value=32, value=1, step=1
        )

        max_seq_length = st.number_input("Max Sequence Length", min_value=128, max_value=4096, value=512, step=128)

        st.divider()

        # Training Control
        st.subheader("Training Control")

        # Check current training status
        is_running = check_training_status()

        if is_running:
            st.success("Training is running")
        else:
            st.info("Training is not running")

        col_start, col_stop = st.columns(2)

        with col_start:
            if st.button("Start Training", use_container_width=True, disabled=is_running):
                config = {
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "max_steps": max_steps,
                    "warmup_steps": warmup_steps,
                    "eval_every": eval_every,
                    "gradient_accumulation": gradient_accumulation,
                    "max_seq_length": max_seq_length,
                }
                success, message = start_training(config)
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        with col_stop:
            if st.button("Stop Training", use_container_width=True, disabled=not is_running):
                success, message = stop_training()
                if success:
                    st.warning(message)
                    st.rerun()
                else:
                    st.error(message)

        # Show YAML file location if it exists
        if os.path.exists("training_overrides.yaml"):
            st.caption(f"Config: training_overrides.yaml")

        st.divider()

        # File paths
        st.subheader("File Paths")
        output_file = st.text_input("Training Output File", value="output.txt", help="File containing training metrics")

        validation_file = st.text_input(
            "Validation Output File", value="validation.txt", help="File containing validation results"
        )

        st.divider()

        # Display configuration summary
        with st.expander("Configuration Summary"):
            st.json(
                {
                    "model": selected_model,
                    "dataset": selected_dataset,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "max_steps": max_steps,
                    "warmup_steps": warmup_steps,
                    "eval_every": eval_every,
                    "gradient_accumulation": gradient_accumulation,
                    "max_seq_length": max_seq_length,
                }
            )

        # Refresh controls
        st.divider()
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (seconds)", min_value=1, max_value=10, value=2, step=1)

        if st.button("Refresh Now", use_container_width=True):
            st.session_state.last_update = 0  # Force update
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
                st.json(test_data)
            else:
                st.error("Failed to parse file")
                if os.path.exists(output_file):
                    with open(output_file, "r") as f:
                        st.text("File contents:")
                        st.code(f.read())

    # Main content area
    tab1, tab2, tab3 = st.tabs(["Training Progress", "Validation Output", "About"])

    with tab1:
        # Parse current training data and update history
        current_data = parse_output_file(output_file)

        if current_data:
            # Update history if this is a new step OR if we haven't recorded anything yet
            if current_data["step"] >= st.session_state.last_step:
                # Only append if it's actually a new step or first entry
                if current_data["step"] > st.session_state.last_step or len(st.session_state.training_history) == 0:
                    st.session_state.training_history.append(current_data)
                st.session_state.last_step = current_data["step"]

        # Current status metrics
        col1, col2, col3, col4 = st.columns(4)

        if current_data:
            # Display current metrics
            with col1:
                st.metric("Current Step", f"{current_data['step']:,}", delta=f"Epoch {current_data['epoch']}")

            with col2:
                st.metric("Training Loss", f"{current_data['training_loss']:.4f}")

            with col3:
                st.metric("Validation Loss", f"{current_data['val_loss']:.4f}")

            with col4:
                st.metric("Learning Rate", f"{current_data['lr']:.2e}")

            # Show training progress
            if max_steps > 0:
                progress = current_data["step"] / max_steps
                st.progress(min(progress, 1.0))
                st.caption(f"Progress: {current_data['step']:,} / {max_steps:,} steps ({progress*100:.1f}%)")
        else:
            with col1:
                st.metric("Current Step", "N/A")
            with col2:
                st.metric("Training Loss", "N/A")
            with col3:
                st.metric("Validation Loss", "N/A")
            with col4:
                st.metric("Learning Rate", "N/A")

            st.info("Waiting for training data... Make sure `output.txt` is being generated.")

        st.divider()

        # Plot training history
        if st.session_state.training_history:
            history_df = pd.DataFrame(st.session_state.training_history)

            # Remove duplicates based on step
            history_df = history_df.drop_duplicates(subset=["step"], keep="last")

            # Show statistics
            st.subheader("Training Statistics")

            col1, col2, col3 = st.columns(3)

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

            st.divider()

            # Create and display plots side by side
            plot_col1, plot_col2 = st.columns(2)

            with plot_col1:
                fig_loss = create_loss_plot(history_df)
                st.plotly_chart(fig_loss, use_container_width=True)

            with plot_col2:
                fig_lr = create_lr_plot(history_df)
                st.plotly_chart(fig_lr, use_container_width=True)
        else:
            st.info("Training plots will appear here once data is available.")

    with tab2:
        st.header("Latest Validation Output")

        # Parse and display validation content
        validation_content = parse_validation_file(validation_file)

        if validation_content and validation_content != "No validation data available yet.":
            st.text_area("Validation Results", validation_content, height=500, disabled=True)

            # Download validation output
            st.download_button(
                label="Download Validation Output",
                data=validation_content,
                file_name=f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )
        else:
            st.info("Waiting for validation data... Make sure `validation.txt` is being generated.")

    with tab3:
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
