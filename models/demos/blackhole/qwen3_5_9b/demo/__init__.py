# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Runnable text-generation demo for the Qwen3.5 hybrid model.

  * demo.py         — the entry point (run ``python .../demo/demo.py``). Opens a mesh,
                      loads a checkpoint, tokenizes a prompt, greedily generates, and
                      detokenizes. Supports --mode eager / trace / both.
  * trace_runner.py — TracedRunner: captures prefill and the per-token decode step as
                      ttnn traces and replays them with a single device dispatch (the
                      --trace path), which is what hides host dispatch latency on decode.

These drive tt/model.py; they contain no model logic of their own.
"""
