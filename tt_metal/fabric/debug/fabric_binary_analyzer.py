#!/usr/bin/env python3
"""
Fabric Erisc Router Binary Analyzer

A utility script to analyze the binary sizes for fabric_erisc_router kernels.
Automatically discovers fabric_erisc_router ELF binaries in the tt-metal cache
and reports text and data section sizes using readelf, with detailed statistics.

Usage: python3 fabric_binary_analyzer.py [options]
"""

import argparse
import re
import subprocess
import sys
import statistics
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import List, Optional, Tuple

# Data structures
BinaryInfo = namedtuple(
    "BinaryInfo",
    [
        "path",
        "kernel_name",
        "kernel_hash",
        "build_hash",
        "text_size",
        "data_size",
        "total_size",
    ],
)


class FabricBinaryAnalyzer:
    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize the analyzer with cache directory."""
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "tt-metal-cache"
        self.cache_dir = Path(cache_dir)
        self.binaries: List[BinaryInfo] = []

    def discover_binaries(self) -> None:
        """Discover fabric_erisc_router ELF binaries in the cache directory."""
        if not self.cache_dir.exists():
            print(f"Cache directory not found: {self.cache_dir}")
            print("Make sure fabric kernels have been compiled at least once.")
            return

        # Look specifically for fabric_erisc_router binaries
        erisc_pattern = "**/fabric_erisc_router/**/*.elf"
        elf_files = list(self.cache_dir.glob(erisc_pattern))

        if not elf_files:
            print(f"No fabric_erisc_router ELF binaries found in {self.cache_dir}")
            print("Make sure fabric router kernels have been compiled.")
            return

        print(f"Found {len(elf_files)} fabric_erisc_router ELF binaries, analyzing...")

        for elf_path in elf_files:
            try:
                binary_info = self._analyze_binary(elf_path)
                if binary_info:
                    self.binaries.append(binary_info)
            except Exception as e:
                print(f"Warning: Error analyzing {elf_path}: {e}", file=sys.stderr)

    def _analyze_binary(self, elf_path: Path) -> Optional[BinaryInfo]:
        """Analyze a single ELF binary and extract size information."""
        # Parse the path to extract metadata
        # Expected structure: cache/.../kernels/fabric_erisc_router/kernel_hash/erisc/erisc.elf
        path_parts = elf_path.parts

        try:
            # Find the kernels directory index
            kernels_idx = None
            for i, part in enumerate(path_parts):
                if part == "kernels":
                    kernels_idx = i
                    break

            if kernels_idx is None or kernels_idx + 3 >= len(path_parts):
                return None

            # Extract path components - be more robust about build hash extraction
            # Find build hash by looking for the directory before kernels that's not tt-metal-cache
            build_hash = "unknown"
            for i in range(kernels_idx - 1, -1, -1):
                if path_parts[i] != "tt-metal-cache" and len(path_parts[i]) > 5:
                    build_hash = path_parts[i]
                    break

            kernel_name = path_parts[kernels_idx + 1]
            kernel_hash = path_parts[kernels_idx + 2]

            # Get size information using readelf
            text_size, data_size = self._get_section_sizes(elf_path)
            total_size = text_size + data_size

            return BinaryInfo(
                path=str(elf_path),
                kernel_name=kernel_name,
                kernel_hash=kernel_hash,
                build_hash=build_hash,
                text_size=text_size,
                data_size=data_size,
                total_size=total_size,
            )

        except Exception as e:
            print(f"Warning: Error parsing path {elf_path}: {e}", file=sys.stderr)
            return None

    def _get_section_sizes(self, elf_path: Path) -> Tuple[int, int]:
        """Extract text and data section sizes using readelf."""
        try:
            # Use readelf -S to get section headers
            result = subprocess.run(["readelf", "-S", str(elf_path)], capture_output=True, text=True, check=True)

            text_size = 0
            data_size = 0

            # Parse the section header table
            for line in result.stdout.split("\n"):
                # Look for .text, .data, and .bss sections
                # Format: [Nr] Name Type Addr Off Size ES Flg Lk Inf Al
                if re.search(r"\s+\.text\s+", line):
                    # Extract size (6th column in hex)
                    match = re.search(r"\s+\.text\s+\w+\s+\w+\s+\w+\s+(\w+)\s+\w+", line)
                    if match:
                        text_size = int(match.group(1), 16)

                elif re.search(r"\s+\.data\s+", line):
                    match = re.search(r"\s+\.data\s+\w+\s+\w+\s+\w+\s+(\w+)\s+\w+", line)
                    if match:
                        data_size += int(match.group(1), 16)

                elif re.search(r"\s+\.bss\s+", line):
                    match = re.search(r"\s+\.bss\s+\w+\s+\w+\s+\w+\s+(\w+)\s+\w+", line)
                    if match:
                        data_size += int(match.group(1), 16)

            return text_size, data_size

        except subprocess.CalledProcessError as e:
            print(f"Error running readelf on {elf_path}: {e}")
            return 0, 0
        except Exception as e:
            print(f"Error parsing readelf output for {elf_path}: {e}")
            return 0, 0

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

    def _calculate_stats(self, values: List[int]) -> dict:
        """Calculate statistical metrics for a list of values."""
        if not values:
            return {}

        sorted_values = sorted(values)
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": sorted_values[int(0.95 * len(sorted_values))] if len(sorted_values) > 1 else sorted_values[0],
        }

    def print_summary_stats(self) -> None:
        """Print summary statistics for fabric_erisc_router binaries."""
        if not self.binaries:
            print("No fabric_erisc_router binaries found to analyze.")
            return

        print(f"\n{'='*100}")
        print(f"FABRIC ERISC ROUTER BINARY SIZE ANALYSIS")
        print(f"{'='*100}")
        print(f"Total fabric_erisc_router binaries analyzed: {len(self.binaries)}")

        # Extract size data for statistics
        text_sizes = [b.text_size for b in self.binaries]
        data_sizes = [b.data_size for b in self.binaries]
        total_sizes = [b.total_size for b in self.binaries]

        # Calculate statistics
        text_stats = self._calculate_stats(text_sizes)
        data_stats = self._calculate_stats(data_sizes)
        total_stats = self._calculate_stats(total_sizes)

        # Detailed statistics table with sections as rows, stats as columns
        print(f"\n{'SIZE STATISTICS:':<50}")
        print(f"{'='*100}")
        print(f"{'Section':<12} {'Minimum':<12} {'Maximum':<12} {'Mean':<12} {'Median':<12} {'95th %ile':<12}")
        print(f"{'-'*100}")

        # Text section row
        if text_stats:
            print(
                f"{'Text Size':<12} {self._format_size(text_stats['min']):<12} "
                f"{self._format_size(text_stats['max']):<12} {self._format_size(int(text_stats['mean'])):<12} "
                f"{self._format_size(text_stats['median']):<12} {self._format_size(text_stats['p95']):<12}"
            )

        # Data section row
        if data_stats:
            print(
                f"{'Data Size':<12} {self._format_size(data_stats['min']):<12} "
                f"{self._format_size(data_stats['max']):<12} {self._format_size(int(data_stats['mean'])):<12} "
                f"{self._format_size(data_stats['median']):<12} {self._format_size(data_stats['p95']):<12}"
            )

        # Total section row
        if total_stats:
            print(
                f"{'Total Size':<12} {self._format_size(total_stats['min']):<12} "
                f"{self._format_size(total_stats['max']):<12} {self._format_size(int(total_stats['mean'])):<12} "
                f"{self._format_size(total_stats['median']):<12} {self._format_size(total_stats['p95']):<12}"
            )

        # Group by build hash only
        print(f"\n{'BINARIES BY BUILD:':<50}")
        print(f"{'='*100}")

        by_build = defaultdict(list)
        for binary in self.binaries:
            by_build[binary.build_hash].append(binary)

        for build_hash, binaries in sorted(by_build.items()):
            count = len(binaries)
            avg_total = sum(b.total_size for b in binaries) / count
            unique_configs = len(set(b.kernel_hash for b in binaries))
            print(
                f"Build Hash: {build_hash} ({count} binaries, {unique_configs} unique configs, avg: {self._format_size(int(avg_total))})"
            )

    def print_detailed_report(self) -> None:
        """Print detailed report of all fabric_erisc_router binaries."""
        if not self.binaries:
            return

        print(f"\n{'='*100}")
        print(f"DETAILED FABRIC_ERISC_ROUTER BINARY REPORT")
        print(f"{'='*100}")

        # Sort by total size descending, then by kernel hash
        sorted_binaries = sorted(self.binaries, key=lambda x: (-x.total_size, x.kernel_hash))

        print(f"{'Kernel Hash':<20} {'Text Size':<12} {'Data Size':<12} {'Total Size':<12} {'Build Hash':<12}")
        print(f"{'-'*90}")

        for binary in sorted_binaries:
            print(
                f"{binary.kernel_hash[:18]+'...' if len(binary.kernel_hash) > 18 else binary.kernel_hash:<20} "
                f"{self._format_size(binary.text_size):<12} {self._format_size(binary.data_size):<12} "
                f"{self._format_size(binary.total_size):<12} "
                f"{binary.build_hash[:10]+'...':<12}"
            )


def main():
    parser = argparse.ArgumentParser(description="Analyze fabric_erisc_router binary sizes with detailed statistics")
    parser.add_argument(
        "--cache-dir", type=Path, help="Path to tt-metal-cache directory (default: ~/.cache/tt-metal-cache)"
    )
    parser.add_argument(
        "--detailed", action="store_true", help="Show detailed report of all fabric_erisc_router binaries"
    )

    args = parser.parse_args()

    analyzer = FabricBinaryAnalyzer(args.cache_dir)
    analyzer.discover_binaries()

    if analyzer.binaries:
        analyzer.print_summary_stats()

        if args.detailed:
            analyzer.print_detailed_report()
    else:
        print("No fabric_erisc_router binaries found. Make sure to compile fabric router kernels first.")
        sys.exit(1)


if __name__ == "__main__":
    main()
