# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Native ternary (2-bit) execution for Ternary-Bonsai-2-27B on Blackhole.

Host-side harness (:mod:`gguf_ternary`, :mod:`hadamard`, :mod:`bfp2`,
:mod:`cpu_ref`) plus the device engine (:mod:`ternary_matmul`,
:mod:`hadamard_op`, :mod:`linear`).  Nothing imports ttnn at module scope, so
the harness modules stay usable in a bare numpy environment.
"""
