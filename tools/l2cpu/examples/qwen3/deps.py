# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
#
# SPDX-License-Identifier: Apache-2.0
"""Single place where this example binds to the other l2cpu components. Importing it puts on sys.path:
    tools/l2cpu/host  -> package `l2cpu` (bring-up component: hw, bringup, ctl, monitor)
    tools/l2cpu       -> package `tensix` (Tensix <-> L2CPU link: ops, kernels/l2cpu_link.h)
    tools/l2cpu/sampling/lib -> module `x280s_ref` (host build of the sampling library: the bit-exact reference)
Also: default locations of caches and outputs (environment variables, see README.md)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
L2CPU = os.path.abspath(os.path.join(HERE, "..", ".."))  # tools/l2cpu
for p in (os.path.join(L2CPU, "sampling", "lib"), L2CPU, os.path.join(L2CPU, "host")):
    if p not in sys.path:
        sys.path.insert(0, p)


def weights_cache(model):
    """TT_CACHE_PATH for a model: $TT_CACHE_PATH if set, else $L2CPU_QWEN3_CACHE/<model basename>
    (default ~/.cache/tt-l2cpu-qwen3/<model basename>)."""
    if os.environ.get("TT_CACHE_PATH"):
        return os.environ["TT_CACHE_PATH"]
    root = os.environ.get("L2CPU_QWEN3_CACHE", os.path.expanduser("~/.cache/tt-l2cpu-qwen3"))
    return os.path.join(root, model.split("/")[-1])


def out_dir(sub=""):
    """Result/log directory: $L2CPU_QWEN3_OUT (default ./l2cpu_qwen3_out)."""
    d = os.path.join(os.environ.get("L2CPU_QWEN3_OUT", "l2cpu_qwen3_out"), sub)
    os.makedirs(d, exist_ok=True)
    return d
