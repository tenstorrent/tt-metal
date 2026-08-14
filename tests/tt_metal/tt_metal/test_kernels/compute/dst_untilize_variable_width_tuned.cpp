// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Hand-tuned flavor of dst_untilize_variable_width: BH DEST remap configured
// once, every subsequent pack_untilize_dest_init passes configure_remap=false.
// The best a careful author can do with the public LLK 1.0 API.

#define VW_SKIP_REMAP 1
#include "dst_untilize_variable_width.cpp"
