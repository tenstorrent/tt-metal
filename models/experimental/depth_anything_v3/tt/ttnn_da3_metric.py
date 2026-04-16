# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ttnn implementation of Depth Anything V3 metric branch.

Iteration 0: skeleton only — does not implement the model yet. The benchmark
harness detects the absence of `run` and falls back to the torch CPU reference
so iteration 0 produces a parseable baseline. Subsequent iterations replace
this stub with real on-chip code piece by piece (backbone first, then DPT)."""
