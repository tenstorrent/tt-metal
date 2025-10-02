# Mandelbrot Set on Tenstorrent Mesh Device

This example demonstrates how to compute the Mandelbrot set on a Tenstorrent mesh device using both C++ (TT-Metalium) and Python (TTNN) implementations.

## Overview

The Mandelbrot set is a fractal defined by the iterative formula:
- z₀ = 0
- z_{n+1} = z_n² + c

Where c is a complex number representing a point in the complex plane. Points that don't diverge (|z| ≤ 2) after a maximum number of iterations are considered part of the Mandelbrot set.

## Files

### Main Implementations
- `mandelbrot_mesh.cpp` - C++ implementation using fixed-point kernel (recommended)
- `mandelbrot_mesh_simple.cpp` - C++ implementation with detailed parallelization comments
- `python_mandelbrot_mesh.py` - Python implementation using TTNN mesh operations

### Compute Kernels
- `kernels/compute/mandelbrot_fixed.cpp` - Fixed-point kernel (avoids floating-point issues)
- `kernels/compute/mandelbrot_simple.cpp` - Simple kernel with detailed parallelization comments
- `kernels/compute/mandelbrot_compute.cpp` - Original kernel with memcpy fixes

### Support Files
- `kernels/dataflow/mandelbrot_writer.cpp` - Data movement kernel for writing results
- `CMakeLists.txt` - Build configuration for all versions
- `build_and_run.sh` - Automated build and run script

## Features

- **Distributed Computing**: Utilizes 2x4 mesh device configuration for parallel computation
- **Mesh Device Programming**: Demonstrates proper use of TT-Metalium mesh operations
- **Image Output**: Generates colorful PPM/PNG images of the Mandelbrot set
- **Configurable Parameters**: Adjustable resolution, iteration count, and zoom level

## Usage

### Python Version (Recommended)

```bash
cd /home/tt-metal-apv/tt_metal/programming_examples/alexp_examples/mandelbrot_mesh
python3 python_mandelbrot_mesh.py
```

### C++ Version

#### Option 1: Using Build Script (Recommended)
```bash
# Build and run fixed-point version (recommended)
./build_and_run.sh

# Build and run simple version with detailed comments
./build_and_run.sh simple

# Build and run both versions
./build_and_run.sh both

# Clean build and run
./build_and_run.sh --clean both

# Show all options
./build_and_run.sh --help
```

#### Option 2: Manual Build
```bash
# Build both versions
mkdir build && cd build
cmake ..
make mandelbrot_mesh mandelbrot_mesh_simple

# Run fixed-point version
./mandelbrot_mesh

# Run simple version
./mandelbrot_mesh_simple
```

## Parameters

You can modify these parameters in the code:

- **Resolution**: `width` and `height` (default: 512x512)
- **Iterations**: `max_iterations` (default: 100)
- **View Window**: `x_min`, `x_max`, `y_min`, `y_max` (default: classic Mandelbrot view)

## Output

The program generates image files showing the Mandelbrot set:
- `mandelbrot_ttnn_mesh.png` - TTNN mesh device result
- `mandelbrot_cpu_reference.png` - CPU reference (fallback)
- `mandelbrot_set.ppm` - C++ version output

## Kernel Versions

### 1. **Fixed-Point Kernel** (`mandelbrot_fixed.cpp`) - **Recommended**
- Uses integer arithmetic to avoid floating-point conversion issues
- Implements 16.16 fixed-point format for coordinates
- Most robust and compilation-safe version
- Used by: `mandelbrot_mesh` executable

### 2. **Simple Kernel** (`mandelbrot_simple.cpp`) - **Educational**
- Includes extensive comments explaining parallelization strategy
- Shows device workload distribution clearly
- Great for understanding the mesh programming model
- Used by: `mandelbrot_mesh_simple` executable

### 3. **Original Kernel** (`mandelbrot_compute.cpp`) - **Reference**
- Uses memcpy for safe float conversion
- Demonstrates standard floating-point approach
- Available as reference implementation

## Mesh Device Architecture

This example uses a 2x4 mesh configuration:
- 8 devices total arranged in 2 rows and 4 columns
- Each device computes a portion of the image
- Results are gathered and combined into the final image

## Color Scheme

The generated images use a gradient color scheme:
- **Black**: Points in the Mandelbrot set (no escape)
- **Red**: Points that escape quickly
- **Yellow**: Points that escape moderately
- **White**: Points that escape slowly

## Technical Details

- Uses `bfloat16` precision for efficient computation
- Tile-based processing (32x32 elements per tile)
- Distributed memory management across mesh devices
- Optimized data movement kernels for performance

## Dependencies

- TT-Metalium framework
- TTNN Python bindings
- PIL (Python Imaging Library)
- NumPy
- PyTorch
- Matplotlib

## Troubleshooting

If you encounter device initialization issues:
1. Ensure TT_METAL_DPRINT_CORES environment variable is set
2. Check that mesh devices are available and properly configured
3. The Python version includes CPU fallback for testing

## Example Output

```
Mandelbrot Set Computation on Tenstorrent Mesh Device
============================================================
Image dimensions: 512 x 512
Max iterations: 100

System mesh topology:
[Mesh visualization output]

Computing Mandelbrot set on Tenstorrent mesh device...
Iteration 0/100
Iteration 10/100
...
Iteration 90/100
Gathering results from mesh...
Mandelbrot set saved to mandelbrot_ttnn_mesh.png
✓ TTNN mesh computation completed successfully!

Mandelbrot set computation finished!
Check the generated PNG images to see the results.
```
