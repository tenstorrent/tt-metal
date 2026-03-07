#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0
"""
Telemetry Analysis and Plotting Script

This script parses training logs and tt-smi telemetry data, matches them by timestamp,
and generates plots showing the correlation between training steps and device metrics.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


class TrainingLog:
    """Represents a training step with timestamp."""

    def __init__(
        self,
        step: int,
        loss: float,
        timestamp: float,
        step_time: Optional[float] = None,
    ):
        self.step = step
        self.loss = loss
        self.timestamp = timestamp
        self.step_time = step_time  # Step time in milliseconds


class TelemetryLog:
    """Represents a tt-smi telemetry snapshot."""

    def __init__(self, timestamp: str, telemetry: Dict[str, str]):
        self.timestamp = self._parse_timestamp(timestamp)
        self.telemetry = telemetry

    @staticmethod
    def _parse_timestamp(timestamp_str: str) -> float:
        """Parse ISO timestamp to Unix timestamp."""
        from datetime import datetime

        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.timestamp()


def parse_training_logs(log_file: Path) -> List[TrainingLog]:
    """
    Parse training logs to extract step number, loss, timestamp, and step time.

    Expected format:
        Step: 1, Loss: 10.5, Timestamp: 1737372273.483914
        Full step time 402.063 ms, cache entries: 79
    """
    training_logs = []

    # Pattern to match: Step: N, Loss: X.XX, Timestamp: XXXXX.XXXXXX
    step_pattern = re.compile(
        r"Step:\s*(\d+),\s*Loss:\s*([\d.]+),\s*Timestamp:\s*([\d.]+)"
    )
    # Pattern to match: Full step time X.XX ms, cache entries: Y (with optional timestamp)
    step_time_pattern = re.compile(r"Full step time ([\d.]+) ms")

    current_step = None
    current_loss = None
    current_timestamp = None

    with open(log_file, "r") as f:
        for line in f:
            # Try to match step line
            step_match = step_pattern.search(line)
            if step_match:
                # If we have a previous step without step time, add it
                if current_step is not None:
                    training_logs.append(
                        TrainingLog(current_step, current_loss, current_timestamp, None)
                    )

                current_step = int(step_match.group(1))
                current_loss = float(step_match.group(2))
                current_timestamp = float(step_match.group(3))

            # Try to match step time line
            time_match = step_time_pattern.search(line)
            if time_match and current_step is not None:
                step_time = float(time_match.group(1))
                training_logs.append(
                    TrainingLog(
                        current_step, current_loss, current_timestamp, step_time
                    )
                )
                current_step = None  # Reset to avoid duplicate entries

    # Add last step if it doesn't have step time
    if current_step is not None:
        training_logs.append(
            TrainingLog(current_step, current_loss, current_timestamp, None)
        )

    if not training_logs:
        print(f"Warning: No training step logs found in {log_file}", file=sys.stderr)
    else:
        print(f"Parsed {len(training_logs)} training steps from {log_file}")
        steps_with_time = sum(1 for log in training_logs if log.step_time is not None)
        print(f"  - {steps_with_time} steps have step time data")

    return training_logs


def parse_telemetry_logs(log_file: Path) -> List[TelemetryLog]:
    """
    Parse tt-smi JSON logs to extract telemetry data with timestamps.

    Expected structure:
        {
            "time": "2026-01-20T11:04:33.483914",
            "device_info": [
                {
                    "telemetry": {
                        "voltage": "0.92",
                        "current": " 27.0",
                        "power": " 25.0",
                        "aiclk": "1000",
                        "asic_temperature": "45.6",
                        "heartbeat": "1559"
                    }
                }
            ]
        }
    """
    telemetry_logs = []

    with open(log_file, "r") as f:
        content = f.read()

    # Split by entries (separated by blank lines or closing braces)
    # We need to find individual JSON objects
    json_objects = []
    current_obj = ""
    brace_count = 0

    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            if brace_count == 0 and current_obj:
                json_objects.append(current_obj)
                current_obj = ""
            continue

        current_obj += line + "\n"
        brace_count += line.count("{") - line.count("}")

        if brace_count == 0 and current_obj.strip():
            json_objects.append(current_obj)
            current_obj = ""

    # Add any remaining object
    if current_obj.strip():
        json_objects.append(current_obj)

    for json_str in json_objects:
        try:
            data = json.loads(json_str)
            if "time" in data and "device_info" in data:
                timestamp = data["time"]
                telemetry = data["device_info"][0].get("telemetry", {})
                telemetry_logs.append(TelemetryLog(timestamp, telemetry))
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            # Skip malformed entries
            continue

    if not telemetry_logs:
        print(f"Warning: No telemetry data found in {log_file}", file=sys.stderr)
    else:
        print(f"Parsed {len(telemetry_logs)} telemetry snapshots from {log_file}")

    return telemetry_logs


def match_logs(
    training_logs: List[TrainingLog], telemetry_logs: List[TelemetryLog]
) -> Tuple[List[TrainingLog], List[TelemetryLog]]:
    """
    Match training logs to telemetry logs by timestamp.

    For each training step, find the nearest telemetry snapshot.
    Discard matches where the time difference exceeds the average step time.
    """
    if not training_logs or not telemetry_logs:
        return [], []

    # Calculate average step time
    if len(training_logs) > 1:
        time_diffs = [
            training_logs[i + 1].timestamp - training_logs[i].timestamp
            for i in range(len(training_logs) - 1)
        ]
        avg_step_time = np.mean(time_diffs)
    else:
        avg_step_time = float("inf")  # No threshold if only one step

    print(f"Average step time: {avg_step_time:.3f} seconds")

    matched_training = []
    matched_telemetry = []

    # Convert telemetry to array for efficient searching
    telemetry_timestamps = np.array([t.timestamp for t in telemetry_logs])

    for train_log in training_logs:
        # Find nearest telemetry timestamp
        time_diffs = np.abs(telemetry_timestamps - train_log.timestamp)
        nearest_idx = np.argmin(time_diffs)
        min_diff = time_diffs[nearest_idx]

        # Only match if within threshold
        if min_diff <= avg_step_time:
            matched_training.append(train_log)
            matched_telemetry.append(telemetry_logs[nearest_idx])

    print(f"Matched {len(matched_training)} out of {len(training_logs)} training steps")
    print(f"Discarded {len(training_logs) - len(matched_training)} unmatched steps")

    return matched_training, matched_telemetry


def plot_telemetry(
    training_logs: List[TrainingLog],
    telemetry_logs: List[TelemetryLog],
    output_file: Path,
    all_training_logs: Optional[List[TrainingLog]] = None,
):
    """
    Create plots showing telemetry metrics vs training step.

    Creates a single figure with subplots for:
    - Loss vs Step
    - Step Time vs Step
    - Each telemetry metric vs Step

    Args:
        training_logs: Training logs (matched with telemetry if available)
        telemetry_logs: Telemetry logs (matched with training if available)
        output_file: Path to save the plot
        all_training_logs: All training logs (for loss/step time plots). If None, uses training_logs.
    """
    if not training_logs:
        print("Error: No training data to plot", file=sys.stderr)
        return

    # Use all training logs for loss/step time plots if provided, otherwise use matched logs
    logs_for_metrics = all_training_logs if all_training_logs else training_logs

    # Extract training data
    all_steps = [log.step for log in logs_for_metrics]
    all_losses = [log.loss for log in logs_for_metrics]
    # Filter out first two steps from step time (compilation outliers)
    all_step_times = [
        log.step_time
        for log in logs_for_metrics
        if log.step_time is not None and log.step > 2
    ]
    steps_with_time = [
        log.step
        for log in logs_for_metrics
        if log.step_time is not None and log.step > 2
    ]

    # Define telemetry fields to plot
    telemetry_fields = [
        ("voltage", "Voltage (V)"),
        ("current", "Current (A)"),
        ("power", "Power (W)"),
        ("aiclk", "AI Clock (MHz)"),
        ("asic_temperature", "ASIC Temperature (°C)"),
        ("heartbeat", "Heartbeat"),
    ]

    # Extract telemetry values (only for matched logs)
    telemetry_data = {}
    if telemetry_logs:
        for field, label in telemetry_fields:
            values = []
            for telem in telemetry_logs:
                value_str = telem.telemetry.get(field, "0").strip()
                try:
                    values.append(float(value_str))
                except ValueError:
                    values.append(np.nan)
            telemetry_data[field] = (values, label)

    # Calculate layout: 2 columns, rows for loss + step_time + telemetry
    n_telemetry = len(telemetry_fields) if telemetry_logs else 0
    n_plots = 2 + n_telemetry  # Loss, Step Time, and telemetry plots
    n_rows = (n_plots + 1) // 2  # Round up

    # Create subplots
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    fig.suptitle(
        "Training Metrics and Device Telemetry", fontsize=16, fontweight="bold"
    )

    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_plots == 1 else list(axes)

    plot_idx = 0

    # Plot 1: Loss vs Step
    ax = axes_flat[plot_idx]
    ax.plot(
        all_steps,
        all_losses,
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=4,
        color="blue",
    )
    ax.set_xlabel("Training Step", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    ax.set_title("Loss vs Step", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plot_idx += 1

    # Plot 2: Step Time vs Step
    ax = axes_flat[plot_idx]
    if all_step_times:
        ax.plot(
            steps_with_time,
            all_step_times,
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=4,
            color="green",
        )
        ax.set_xlabel("Training Step", fontsize=10)
        ax.set_ylabel("Step Time (ms)", fontsize=10)
        ax.set_title("Step Time vs Step", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No step time data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title("Step Time vs Step", fontsize=11, fontweight="bold")
    plot_idx += 1

    # Plot telemetry metrics
    if telemetry_logs:
        matched_steps = [log.step for log in training_logs]
        for field, (values, label) in telemetry_data.items():
            if plot_idx >= len(axes_flat):
                break
            ax = axes_flat[plot_idx]
            ax.plot(
                matched_steps,
                values,
                marker="o",
                linestyle="-",
                linewidth=1.5,
                markersize=4,
            )
            ax.set_xlabel("Training Step", fontsize=10)
            ax.set_ylabel(label, fontsize=10)
            ax.set_title(label, fontsize=11, fontweight="bold")
            ax.grid(True, alpha=0.3)

            # Format y-axis to show appropriate precision
            if not all(np.isnan(values)):
                ax.ticklabel_format(useOffset=False)
            plot_idx += 1

    # Hide unused subplots
    for idx in range(plot_idx, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()

    # Save plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_file}")

    # Print statistics
    print("\nTraining Statistics:")
    print("-" * 60)
    if all_losses:
        print(
            f"{'Loss':25s} - Min: {min(all_losses):8.4f}, "
            f"Max: {max(all_losses):8.4f}, "
            f"Mean: {np.mean(all_losses):8.4f}"
        )
    if all_step_times:
        print(
            f"{'Step Time (ms)':25s} - Min: {min(all_step_times):8.2f}, "
            f"Max: {max(all_step_times):8.2f}, "
            f"Mean: {np.mean(all_step_times):8.2f}"
        )

    if telemetry_logs:
        print("\nTelemetry Statistics:")
        print("-" * 60)
        for field, (values, label) in telemetry_data.items():
            clean_values = [v for v in values if not np.isnan(v)]
            if clean_values:
                print(
                    f"{label:25s} - Min: {min(clean_values):8.2f}, "
                    f"Max: {max(clean_values):8.2f}, "
                    f"Mean: {np.mean(clean_values):8.2f}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Parse training and telemetry logs, match by timestamp, and generate plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s --training-log training.log --telemetry-log telemetry.log --output plots.png
        """,
    )

    parser.add_argument(
        "--training-log", type=Path, required=True, help="Path to the training log file"
    )

    parser.add_argument(
        "--telemetry-log",
        type=Path,
        required=True,
        help="Path to the telemetry log file (from poll_telemetry.sh)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to save the output plot (PNG file)",
    )

    args = parser.parse_args()

    # Validate input files
    if not args.training_log.exists():
        print(
            f"Error: Training log file not found: {args.training_log}", file=sys.stderr
        )
        sys.exit(1)

    if not args.telemetry_log.exists():
        print(
            f"Error: Telemetry log file not found: {args.telemetry_log}",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Parsing logs...")
    print("=" * 60)

    # Parse logs
    training_logs = parse_training_logs(args.training_log)
    telemetry_logs = parse_telemetry_logs(args.telemetry_log)

    if not training_logs:
        print("Error: No training data found", file=sys.stderr)
        sys.exit(1)

    # Match logs if telemetry data is available
    matched_training = training_logs
    matched_telemetry = []

    if telemetry_logs:
        print("\nMatching logs by timestamp...")
        print("=" * 60)
        matched_training, matched_telemetry = match_logs(training_logs, telemetry_logs)

        if not matched_training:
            print(
                "Warning: No matching telemetry data found, plotting training metrics only",
                file=sys.stderr,
            )
            matched_training = training_logs
            matched_telemetry = []
    else:
        print("\nNo telemetry data found, plotting training metrics only")
        print("=" * 60)

    print("\nGenerating plots...")
    print("=" * 60)

    # Generate plots (use all training logs for metrics, matched logs for telemetry)
    plot_telemetry(
        matched_training,
        matched_telemetry,
        args.output,
        all_training_logs=training_logs,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
