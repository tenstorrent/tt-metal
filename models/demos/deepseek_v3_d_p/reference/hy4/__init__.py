# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview (HF ``hy_v4``) host reference and golden-trace tooling for prefill bring-up.

* ``config``      -- ``Hy4RefConfig`` read from the checkpoint's ``config.json`` (+ a tiny random config).
* ``paths``       -- shared checkpoint / golden-trace locations (env-overridable); the prefill adapter
                     consumes ``HY4_PREFILL_TRACE_DEFAULT`` as its ``prefill_trace_default``.
* ``checkpoint``  -- lazy safetensors reader for the BF16 and the official MXFP8 checkpoints.
* ``model``       -- streaming fp32 reference forward (one layer resident at a time, query-blocked
                     sparse attention) that captures what the device writes per layer.
* ``golden``      -- golden-trace writer in the prefill producer's layout, key-frame transforms,
                     top-k agreement.
* ``generate_golden`` / ``compare`` -- CLIs (``python -m models.demos.deepseek_v3_d_p.reference.hy4.<name>``).

Nothing here imports ttnn: it runs on a host without devices.
"""
