// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#ifndef QB2_ENTRY
#define QB2_ENTRY kernel_main
#else
#define kernel_main QB2_ENTRY
#endif
// Native arithmetic/rounding and HiFi4/FP32 configuration. Reader publishes
// its scalar last, after loading input and both trig rows, as the native
// compute body borrows input/cos/sin storage and produces their CB counters.
#include "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/compute/rotary_embedding_hf_sharded.cpp"

#undef kernel_main
