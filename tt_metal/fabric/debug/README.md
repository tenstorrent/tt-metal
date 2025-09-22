# Fabric Debug Tools

## fabric_binary_analyzer.py

A utility to analyze binary sizes of `fabric_erisc_router` kernels from the tt-metal cache.

### Features

- **Focused Analysis**: Analyzes only `fabric_erisc_router` binaries
- **Comprehensive Statistics**: Min, max, mean, median, and 95th percentile for text/data sections
- **ELF Analysis**: Extracts section sizes using `readelf` from load segments
- **Build Tracking**: Groups binaries by build hash with configuration counts
- **Clean Interface**: Simple command-line interface with clear output

### Usage

```bash
# Basic analysis with statistics summary
python3 tt_metal/fabric/debug/fabric_binary_analyzer.py

# Include detailed per-binary breakdown
python3 tt_metal/fabric/debug/fabric_binary_analyzer.py --detailed

# Use custom cache directory
python3 tt_metal/fabric/debug/fabric_binary_analyzer.py --cache-dir /path/to/cache
```

### Example Output

```
Found 100 fabric_erisc_router ELF binaries, analyzing...

FABRIC ERISC ROUTER BINARY SIZE ANALYSIS
Total fabric_erisc_router binaries analyzed: 100

SIZE STATISTICS:
Section      Minimum    Maximum    Mean       Median     95th %ile
Text Size    7.4 KB     10.3 KB    8.8 KB     9.0 KB     10.0 KB
Data Size    68 B       152 B      104 B      88.0 B     152 B
Total Size   7.4 KB     10.3 KB    8.9 KB     9.1 KB     10.1 KB

BINARIES BY BUILD:
Build Hash: 1a6f17ff97 (100 binaries, 100 unique configs, avg: 8.9 KB)
```

### Requirements

- Python 3.6+
- `readelf` utility (from binutils package)
- Compiled fabric router kernels in cache directory

### Technical Details

- Analyzes `fabric_erisc_router` ELF binaries in `~/.cache/tt-metal-cache/`
- Text section: executable RISC-V instructions
- Data section: combined `.data` and `.bss` sections
- Statistics: min/max/mean/median/95th percentile across all binaries
- Each kernel hash represents a unique router configuration
