#!/usr/bin/env python3

"""
Script to convert LoRA weights log file to CSV format.
"""

import re
import csv
import sys
import os


def parse_log_to_csv(log_file_path, csv_file_path):
    """
    Parse LoRA log file and convert to CSV.

    Args:
        log_file_path: Path to the input log file
        csv_file_path: Path to the output CSV file

    Returns:
        int: Number of entries processed
    """
    print(f"Converting {log_file_path} to {csv_file_path}")
    print("=" * 50)

    # CSV column headers
    headers = [
        "Module_Path",
        "Weight_Name",
        "Shape",
        "Data_Type",
        "Device",
        "Location",
        "Size_MB",
        "Elements",
        "Bytes_Per_Element",
        "Address",
        "Is_Shared",
        "Shared_With",
        "Host_Creation_ms",
        "Host_To_Device_ms",
        "Total_Time_ms",
        "Description",
    ]

    entries_processed = 0

    try:
        with open(log_file_path, "r") as log_file, open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)

            # Write headers
            writer.writerow(headers)

            for line_num, line in enumerate(log_file, 1):
                # Skip non-weight entries
                if "| Address:" not in line:
                    continue

                # Extract information using regex
                entry = {}

                # Module path
                module_match = re.search(r"INFO \| ([^|]+) \|", line)
                entry["Module_Path"] = module_match.group(1).strip() if module_match else ""

                # Weight name
                weight_match = re.search(r"INFO \| [^|]+ \| ([^|]+) \|", line)
                entry["Weight_Name"] = weight_match.group(1).strip() if weight_match else ""

                # Shape
                shape_match = re.search(r"Shape: ([^|]+) \|", line)
                entry["Shape"] = shape_match.group(1).strip() if shape_match else ""

                # Data type
                dtype_match = re.search(r"Dtype: ([^|]+) \|", line)
                entry["Data_Type"] = dtype_match.group(1).strip() if dtype_match else ""

                # Device
                device_match = re.search(r"Device: ([^|]+) \|", line)
                entry["Device"] = device_match.group(1).strip() if device_match else ""

                # Location
                location_match = re.search(r"Location: ([^|]+) \|", line)
                entry["Location"] = location_match.group(1).strip() if location_match else ""

                # Size in MB
                size_match = re.search(r"Size: ([0-9.]+) MB", line)
                entry["Size_MB"] = float(size_match.group(1)) if size_match else 0.0

                # Elements
                elements_match = re.search(r"Elements: ([0-9,]+) \|", line)
                if elements_match:
                    entry["Elements"] = int(elements_match.group(1).replace(",", ""))
                else:
                    entry["Elements"] = 0

                # Bytes per element
                bytes_match = re.search(r"Bytes/elem: ([0-9]+) \|", line)
                entry["Bytes_Per_Element"] = int(bytes_match.group(1)) if bytes_match else 0

                # Address
                addr_match = re.search(r"Address: (0x[0-9a-fA-F]+)", line)
                entry["Address"] = addr_match.group(1) if addr_match else ""

                # Check for shared weights
                entry["Is_Shared"] = "[SHARED" in line

                if entry["Is_Shared"]:
                    shared_match = re.search(r"\[SHARED with ([^\]]+)\]", line)
                    entry["Shared_With"] = shared_match.group(1) if shared_match else ""
                else:
                    entry["Shared_With"] = ""

                # Extract timing information
                host_creation_match = re.search(r"Host_Creation: ([0-9.]+) ms", line)
                entry["Host_Creation_ms"] = float(host_creation_match.group(1)) if host_creation_match else None

                host_to_device_match = re.search(r"Host_To_Device: ([0-9.]+) ms", line)
                entry["Host_To_Device_ms"] = float(host_to_device_match.group(1)) if host_to_device_match else None

                total_time_match = re.search(r"Total_Time: ([0-9.]+) ms", line)
                entry["Total_Time_ms"] = float(total_time_match.group(1)) if total_time_match else None

                # Description (everything after timing or address, whichever comes last)
                # Try to match description after timing info first
                desc_match = re.search(r"Total_Time: [0-9.]+ ms \| (.+)$", line)
                if not desc_match:
                    # Fallback to old pattern for entries without timing
                    desc_match = re.search(r"Address: [^|]+ \| (.+)$", line)
                entry["Description"] = desc_match.group(1).strip() if desc_match else ""

                # Write to CSV
                row = [
                    entry["Module_Path"],
                    entry["Weight_Name"],
                    entry["Shape"],
                    entry["Data_Type"],
                    entry["Device"],
                    entry["Location"],
                    entry["Size_MB"],
                    entry["Elements"],
                    entry["Bytes_Per_Element"],
                    entry["Address"],
                    entry["Is_Shared"],
                    entry["Shared_With"],
                    entry["Host_Creation_ms"],
                    entry["Host_To_Device_ms"],
                    entry["Total_Time_ms"],
                    entry["Description"],
                ]

                writer.writerow(row)
                entries_processed += 1

                if entries_processed % 100 == 0:
                    print(f"Processed {entries_processed} entries...")

    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file_path}")
        return 0
    except Exception as e:
        print(f"Error processing file: {e}")
        return 0

    return entries_processed


def create_summary_csv(log_file_path, summary_csv_path):
    """
    Create a summary CSV with aggregated statistics.
    """
    print(f"\nCreating summary CSV: {summary_csv_path}")

    # Component breakdown
    component_stats = {}
    dtype_stats = {}

    try:
        with open(log_file_path, "r") as f:
            for line in f:
                if "| Address:" not in line:
                    continue

                # Extract component, data type, and size
                module_match = re.search(r"INFO \| ([^|]+) \|", line)
                dtype_match = re.search(r"Dtype: ([^|]+) \|", line)
                size_match = re.search(r"Size: ([0-9.]+) MB", line)

                if module_match and size_match:
                    module = module_match.group(1).strip()
                    size_mb = float(size_match.group(1))

                    # Extract component (first part of module path)
                    component = module.split(".")[0] if "." in module else module

                    if component not in component_stats:
                        component_stats[component] = {"count": 0, "total_mb": 0.0}

                    component_stats[component]["count"] += 1
                    component_stats[component]["total_mb"] += size_mb

                if dtype_match and size_match:
                    dtype = dtype_match.group(1).strip()
                    size_mb = float(size_match.group(1))

                    if dtype not in dtype_stats:
                        dtype_stats[dtype] = {"count": 0, "total_mb": 0.0}

                    dtype_stats[dtype]["count"] += 1
                    dtype_stats[dtype]["total_mb"] += size_mb

        # Write component summary CSV
        with open(summary_csv_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Component breakdown
            writer.writerow(["=== COMPONENT BREAKDOWN ==="])
            writer.writerow(["Component", "Weight_Count", "Total_MB", "Percentage", "Avg_MB_Per_Weight"])

            total_mb = sum(stats["total_mb"] for stats in component_stats.values())

            for component, stats in sorted(component_stats.items(), key=lambda x: x[1]["total_mb"], reverse=True):
                percentage = (stats["total_mb"] / total_mb) * 100 if total_mb > 0 else 0
                avg_mb = stats["total_mb"] / stats["count"] if stats["count"] > 0 else 0
                writer.writerow(
                    [component, stats["count"], f"{stats['total_mb']:.2f}", f"{percentage:.1f}%", f"{avg_mb:.2f}"]
                )

            writer.writerow([])  # Empty row

            # Data type breakdown
            writer.writerow(["=== DATA TYPE BREAKDOWN ==="])
            writer.writerow(["Data_Type", "Weight_Count", "Total_MB", "Percentage"])

            for dtype, stats in sorted(dtype_stats.items(), key=lambda x: x[1]["total_mb"], reverse=True):
                percentage = (stats["total_mb"] / total_mb) * 100 if total_mb > 0 else 0
                writer.writerow([dtype, stats["count"], f"{stats['total_mb']:.2f}", f"{percentage:.1f}%"])

            writer.writerow([])  # Empty row

            # Overall summary
            writer.writerow(["=== OVERALL SUMMARY ==="])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total_Weights", sum(stats["count"] for stats in component_stats.values())])
            writer.writerow(["Total_Memory_MB", f"{total_mb:.2f}"])
            writer.writerow(["Total_Memory_GB", f"{total_mb/1024:.2f}"])
            writer.writerow(
                ["Average_Weight_Size_MB", f"{total_mb/sum(stats['count'] for stats in component_stats.values()):.2f}"]
            )

    except Exception as e:
        print(f"Error creating summary CSV: {e}")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        # Default files
        log_file = "lora_weights_logs/lora_weights_20260211_091436.log"
        csv_file = "lora_weights_analysis.csv"
        summary_file = "lora_weights_summary.csv"
        print(f"Using default files:")
        print(f"  Log: {log_file}")
        print(f"  CSV: {csv_file}")
        print(f"  Summary: {summary_file}")
    else:
        log_file = sys.argv[1]
        base_name = os.path.splitext(os.path.basename(log_file))[0]
        csv_file = f"{base_name}.csv"
        summary_file = f"{base_name}_summary.csv"

    if not os.path.exists(log_file):
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    print("LoRA Log to CSV Converter")
    print("=" * 30)
    print()

    # Convert log to detailed CSV
    entries = parse_log_to_csv(log_file, csv_file)

    if entries > 0:
        print(f"\n✅ Successfully processed {entries} entries")
        print(f"📄 Detailed CSV created: {csv_file}")

        # Create summary CSV
        create_summary_csv(log_file, summary_file)
        print(f"📊 Summary CSV created: {summary_file}")

        # Show file sizes
        csv_size = os.path.getsize(csv_file) / 1024
        summary_size = os.path.getsize(summary_file) / 1024

        print(f"\nFile sizes:")
        print(f"  {csv_file}: {csv_size:.1f} KB")
        print(f"  {summary_file}: {summary_size:.1f} KB")

        print(f"\n🎯 CSV files ready for analysis in Excel/Google Sheets!")
    else:
        print("❌ No entries processed")


if __name__ == "__main__":
    main()
