#!/bin/bash

echo "=== Current Directory ==="
echo "Current directory: $(pwd)"
echo ""

echo "=== Environment Variables from Docker Command ==="
echo "TT_METAL_HOME: $TT_METAL_HOME"
echo "PYTHONPATH: $PYTHONPATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "ARCH_NAME: $ARCH_NAME"
echo "LOGURU_LEVEL: $LOGURU_LEVEL"
echo ""

echo "=== All Environment Variables (filtered) ==="
echo "Showing TT_METAL, PYTHONPATH, LD_LIBRARY_PATH, ARCH_NAME, and LOGURU_LEVEL:"
env | grep -E "(TT_METAL_HOME|PYTHONPATH|LD_LIBRARY_PATH|ARCH_NAME|LOGURU_LEVEL)" | sort
