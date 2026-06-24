# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Model-agnostic prefill runner.

A generic chunked-prefill driver (standalone file input or H2D-socket request mode) that depends
only on `ttnn` + a small `PrefillModelAdapter` seam. Each model (deepseek_v3_d_p, kimi_k2_6, future
models) registers an adapter; the runner core never imports a model package.

See `adapter.py` for the seam and `registry.py` for how variants are resolved.
"""
