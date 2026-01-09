# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger


class TensorDataParser:
    """
    Parser for tensor log data collected during model execution.
    Converts raw tensor metadata logs into structured analysis formats.
    """

    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)
        self.raw_data: List[Dict[str, Any]] = []
        self.tensor_data: List[Dict[str, Any]] = []
        self.operation_data: List[Dict[str, Any]] = []

    def load_data(self) -> None:
        """Load tensor log data from JSONL file."""
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_file}")

        with open(self.log_file, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    self.raw_data.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")

        logger.info(f"Loaded {len(self.raw_data)} log entries from {self.log_file}")

    def parse_tensor_data(self) -> None:
        """Parse tensor entries from raw data."""
        for entry in self.raw_data:
            # Skip operation start/end entries
            if entry.get("event_type") in ["operation_start", "operation_end"]:
                self.operation_data.append(entry)
                continue

            # Process tensor entries
            if "tensor_type" in entry and "step_name" in entry:
                parsed_entry = self._parse_tensor_entry(entry)
                if parsed_entry:
                    self.tensor_data.append(parsed_entry)

        logger.info(f"Parsed {len(self.tensor_data)} tensor entries and {len(self.operation_data)} operation entries")

    def _parse_tensor_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single tensor entry."""
        try:
            parsed = {
                "step_name": entry.get("step_name", "unknown"),
                "tensor_name": entry.get("tensor_name", "unknown"),
                "tensor_type": entry.get("tensor_type", "unknown"),
                "timestamp": entry.get("timestamp", "0"),
            }

            # Add additional info
            if "row_idx" in entry:
                parsed["row_idx"] = entry["row_idx"]

            # Parse tensor-specific fields
            if entry["tensor_type"] == "ttnn":
                parsed.update(self._parse_ttnn_fields(entry))
            elif entry["tensor_type"] == "torch":
                parsed.update(self._parse_torch_fields(entry))

            return parsed

        except Exception as e:
            logger.warning(f"Failed to parse tensor entry: {e}")
            return None

    def _parse_ttnn_fields(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Parse TTNN-specific tensor fields."""
        fields = {}

        # Basic tensor properties
        for field in ["shape", "dtype", "layout", "rank", "storage_type"]:
            if field in entry:
                fields[field] = entry[field]

        # Memory configuration
        memory_config = entry.get("memory_config", {})
        if memory_config:
            fields["memory_layout"] = memory_config.get("memory_layout", "unknown")
            fields["buffer_type"] = memory_config.get("buffer_type", "unknown")

        # Shard specification
        shard_spec = entry.get("shard_spec")
        if shard_spec and shard_spec is not None:
            fields["shard_shape"] = shard_spec.get("shape")
            fields["shard_orientation"] = shard_spec.get("orientation")
            fields["shard_halo"] = shard_spec.get("halo")
        else:
            fields["shard_shape"] = None
            fields["shard_orientation"] = None
            fields["shard_halo"] = None

        # Device information
        fields["device_mesh_shape"] = entry.get("device_mesh_shape")
        fields["device_type"] = entry.get("device_type", "unknown")

        return fields

    def _parse_torch_fields(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Parse torch-specific tensor fields."""
        fields = {}

        # Basic tensor properties
        for field in [
            "shape",
            "dtype",
            "device",
            "rank",
            "requires_grad",
            "is_contiguous",
            "stride",
            "element_count",
            "memory_bytes",
        ]:
            if field in entry:
                fields[field] = entry[field]

        return fields

    def create_summary_table(self) -> pd.DataFrame:
        """Create a summary table of all tensor operations."""
        if not self.tensor_data:
            return pd.DataFrame()

        df = pd.DataFrame(self.tensor_data)

        # Add computed fields
        if "shape" in df.columns:
            df["shape_str"] = df["shape"].apply(lambda x: str(x) if x else "unknown")
            df["num_elements"] = df["shape"].apply(
                lambda x: 1 if not x else pd.eval("*".join(map(str, x))) if isinstance(x, list) else 1
            )

        return df

    def analyze_step_flow(self) -> Dict[str, Any]:
        """Analyze the flow of operations through the decoder block."""
        if not self.tensor_data:
            return {}

        df = pd.DataFrame(self.tensor_data)

        analysis = {
            "total_steps": len(df["step_name"].unique()),
            "step_sequence": df["step_name"].unique().tolist(),
            "tensor_counts_by_step": df["step_name"].value_counts().to_dict(),
        }

        # Analyze tensor shapes by step
        shape_by_step = {}
        for step in df["step_name"].unique():
            step_data = df[df["step_name"] == step]
            shapes = step_data["shape_str"].unique() if "shape_str" in step_data else []
            shape_by_step[step] = shapes.tolist() if hasattr(shapes, "tolist") else list(shapes)

        analysis["shapes_by_step"] = shape_by_step

        # Analyze memory configurations
        if "memory_layout" in df.columns:
            memory_configs = (
                df.groupby("step_name")["memory_layout"]
                .apply(lambda x: x.unique().tolist() if hasattr(x.unique(), "tolist") else list(x.unique()))
                .to_dict()
            )
            analysis["memory_configs_by_step"] = memory_configs

        return analysis

    def generate_report(self, output_file: Optional[Path] = None) -> str:
        """Generate a comprehensive analysis report."""
        if not self.tensor_data:
            return "No tensor data to analyze."

        df = self.create_summary_table()
        analysis = self.analyze_step_flow()

        report = []
        report.append("# Tensor Flow Analysis Report")
        report.append("")

        report.append(f"## Summary")
        report.append(f"- Total tensor operations: {len(df)}")
        report.append(f"- Unique steps: {analysis['total_steps']}")
        report.append(f"- Log file: {self.log_file}")
        report.append("")

        report.append("## Step Sequence")
        for i, step in enumerate(analysis["step_sequence"], 1):
            count = analysis["tensor_counts_by_step"][step]
            report.append(f"{i}. {step} ({count} tensors)")
        report.append("")

        report.append("## Tensor Shapes by Step")
        for step, shapes in analysis["shapes_by_step"].items():
            report.append(f"### {step}")
            for shape in shapes:
                report.append(f"  - {shape}")
        report.append("")

        if "memory_configs_by_step" in analysis:
            report.append("## Memory Configurations by Step")
            for step, configs in analysis["memory_configs_by_step"].items():
                report.append(f"### {step}")
                for config in configs:
                    report.append(f"  - {config}")
            report.append("")

        # Add detailed tensor table
        report.append("## Detailed Tensor Information")
        if len(df) > 0:
            # Select key columns for the report
            key_columns = ["step_name", "tensor_name", "tensor_type", "shape_str"]
            if "dtype" in df.columns:
                key_columns.append("dtype")
            if "memory_layout" in df.columns:
                key_columns.append("memory_layout")
            if "row_idx" in df.columns:
                key_columns.append("row_idx")

            available_columns = [col for col in key_columns if col in df.columns]
            summary_df = df[available_columns].head(50)  # Limit to first 50 rows for readability

            report.append("```")
            report.append(summary_df.to_string(index=False))
            report.append("```")

        report_text = "\n".join(report)

        if output_file:
            output_file = Path(output_file)
            output_file.write_text(report_text)
            logger.info(f"Report written to {output_file}")

        return report_text

    def save_csv(self, output_file: Path) -> None:
        """Save tensor data as CSV."""
        if not self.tensor_data:
            logger.warning("No tensor data to save")
            return

        df = self.create_summary_table()
        df.to_csv(output_file, index=False)
        logger.info(f"Tensor data saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Parse tensor log data")
    parser.add_argument("log_file", type=Path, help="Path to tensor log file")
    parser.add_argument("--output-csv", type=Path, help="Output CSV file")
    parser.add_argument("--output-report", type=Path, help="Output report file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="DEBUG")

    # Parse the data
    parser = TensorDataParser(args.log_file)
    parser.load_data()
    parser.parse_tensor_data()

    # Generate outputs
    if args.output_csv:
        parser.save_csv(args.output_csv)

    if args.output_report:
        parser.generate_report(args.output_report)
    else:
        # Print to stdout
        report = parser.generate_report()
        print(report)


if __name__ == "__main__":
    main()
