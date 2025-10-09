#!/bin/bash

# TTNN Visualizer + Tracy Profiling Script
# Usage: ./run_tracy_with_visualizer.sh

set -e

echo "=================================="
echo "Tracy + TTNN Visualizer Profiling"
echo "=================================="
echo ""

# Set up environment
export TT_METAL_HOME=/workspace/tt-metal-apv
export TTNN_CONFIG_PATH=/workspace/tt-metal-apv/ttnn_visualizer_config.json
export PYTHONPATH=/workspace/tt-metal-apv:$PYTHONPATH
export LD_LIBRARY_PATH=/workspace/tt-metal-apv/build_Release_tracy/lib:$LD_LIBRARY_PATH

# Add Tracy build ttnn to Python path (needs to be first so it's found before the source)
export PYTHONPATH=/workspace/tt-metal-apv/build_Release_tracy:$PYTHONPATH

# Get the report name from the config
REPORT_NAME=$(python3 -c "import json; print(json.load(open('$TTNN_CONFIG_PATH'))['report_name'])")
echo "Report name: $REPORT_NAME"
echo ""

# Run the pytest with Tracy profiler
echo "Running pytest with Tracy profiler..."
echo "Command: python -m tracy -p -r -v -m 'pytest models/tt_transformers/demo/simple_text_demo.py -k \"performance and batch-1\"'"
echo ""

python -m tracy -p -r -v -m 'pytest models/tt_transformers/demo/simple_text_demo.py -k "performance and batch-1"'

echo ""
echo "=================================="
echo "Profiling Complete!"
echo "=================================="
echo ""
echo "Generated reports:"
echo "  Memory Report: generated/profiler/$REPORT_NAME/"
echo "  Performance Report: generated/profiler/$REPORT_NAME/"
echo ""
echo "To visualize:"
echo "  1. Start ttnn-visualizer:"
echo "     cd /workspace/ttnn-visualizer"
echo "     source myenv/bin/activate"
echo "     ttnn-visualizer"
echo ""
echo "  2. Open browser to: http://localhost:8000"
echo ""
echo "  3. Upload the report folder:"
echo "     /workspace/tt-metal-apv/generated/profiler/$REPORT_NAME/"
echo ""
