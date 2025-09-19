# Fabric Debug Tools

This directory contains debugging and analysis tools for fabric components.

## fabric_binary_analyzer.py

A specialized utility script to analyze the binary sizes of `fabric_erisc_router` kernels compiled in the tt-metal cache.

### Features

- **Fabric-Focused Analysis**: Specifically analyzes `fabric_erisc_router` binaries only
- **Detailed Statistics**: Provides min, max, mean, median, and 95th percentile statistics
- **Size Analysis**: Uses `readelf` to extract text and data section sizes from load segments
- **Build Analysis**: Shows unique configurations per build hash
- **Clean Output**: No warnings for firmware files or other kernel types

### Usage

```bash
# Basic analysis
python3 tt_metal/fabric/debug/fabric_binary_analyzer.py

# Detailed report
python3 tt_metal/fabric/debug/fabric_binary_analyzer.py --detailed

# Custom cache directory
python3 tt_metal/fabric/debug/fabric_binary_analyzer.py --cache-dir /path/to/cache

# Show help
python3 tt_metal/fabric/debug/fabric_binary_analyzer.py --help
```

### Example Output

```
====================================================================================================
FABRIC ERISC ROUTER BINARY SIZE ANALYSIS
====================================================================================================
Total fabric_erisc_router binaries analyzed: 100

SIZE STATISTICS:
====================================================================================================
Section      Minimum      Maximum      Mean         Median       95th %ile
----------------------------------------------------------------------------------------------------
Text Size    7.4 KB       10.3 KB      8.8 KB       9.0 KB       10.0 KB
Data Size    68 B         152 B        104 B        88.0 B       152 B
Total Size   7.4 KB       10.3 KB      8.9 KB       9.1 KB       10.1 KB

BINARIES BY BUILD:
====================================================================================================
Build Hash: 1a6f17ff97 (100 binaries, 100 unique configs, avg: 8.9 KB)
```

### Requirements

- Python 3.6+
- `readelf` utility (part of binutils)
- Compiled fabric kernels in cache directory

### Notes

- The tool specifically analyzes `fabric_erisc_router` ELF binaries in `~/.cache/tt-metal-cache/`
- Text section contains executable code (RISC-V instructions)
- Data section includes both `.data` and `.bss` sections combined
- Statistics are organized with sections as rows and statistical measures as columns
- Statistics include min, max, mean, median, and 95th percentile for comprehensive analysis
- Path parsing is generic and works with any cache subdirectory structure
- Each unique kernel hash represents a different router configuration
