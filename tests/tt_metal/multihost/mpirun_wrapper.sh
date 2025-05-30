#!/bin/bash
# Purpose: Wrapper script to find and execute the appropriate mpirun command
# Usage: ./mpirun_wrapper.sh [mpirun arguments]

# Find mpirun-ulfm executable
if command -v mpirun-ulfm &> /dev/null; then
    MPIRUN="mpirun-ulfm"
elif [ -x "/usr/local/bin/mpirun-ulfm" ]; then
    MPIRUN="/usr/local/bin/mpirun-ulfm"
else
    # Fall back to mpirun if ULFM not found
    echo "Warning: mpirun-ulfm not found in PATH or /usr/local/bin, falling back to mpirun" >&2
    MPIRUN="mpirun"
fi

echo "Using MPI command: $MPIRUN" >&2

# Execute with all passed arguments
exec $MPIRUN "$@"
