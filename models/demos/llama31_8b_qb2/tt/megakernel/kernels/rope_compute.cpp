// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
// Native arithmetic/rounding and HiFi4/FP32 configuration. Reader publishes
// its scalar last, after loading input and both trig rows, as the native
// compute body borrows input/cos/sin storage and produces their CB counters.
#include "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/compute/rotary_embedding_hf_sharded.cpp"
