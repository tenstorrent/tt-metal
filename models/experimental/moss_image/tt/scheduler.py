# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""MOSS CFG and Euler arithmetic, kept FP32 on the accelerator."""


def guided_euler_step(sample, positive, negative, guidance_scale: float, dt: float):
    import ttnn as t

    sample = t.typecast(sample, t.float32)
    positive = t.typecast(positive, t.float32)
    negative = t.typecast(negative, t.float32)
    velocity = t.add(negative, t.multiply(t.subtract(positive, negative), guidance_scale))
    return t.add(sample, t.multiply(velocity, dt)), velocity
