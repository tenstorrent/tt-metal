# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
#
# source me: environment for the L2CPU component in this tt-metal tree.
#   TT_METAL_HOME          defaults to the repository that contains this script
#   PYTHONPATH             ttnn, tt-metal tools and tools/l2cpu/host prepended
#   L2CPU_PY               python interpreter of the tt-metal build (default: $TT_METAL_HOME/python_env/bin/python
#                          if present, else python3); exported as PY
# Chip selection stays environment-driven: set TT_VISIBLE_DEVICES (and, for one chip of a multi-chip board,
# TT_MESH_GRAPH_DESC_PATH) yourself before sourcing; nothing is forced here.
_l2cpu_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TT_METAL_HOME="${TT_METAL_HOME:-$(cd "$_l2cpu_here/../../.." && pwd)}"
export PYTHONPATH="$TT_METAL_HOME/ttnn:$TT_METAL_HOME:$TT_METAL_HOME/tools:$TT_METAL_HOME/tools/l2cpu/host${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${L2CPU_PY:-}" ]; then
  export PY="$L2CPU_PY"
elif [ -x "$TT_METAL_HOME/python_env/bin/python" ]; then
  export PY="$TT_METAL_HOME/python_env/bin/python"
else
  export PY=python3
fi
unset _l2cpu_here
