# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn

LLAMA_INPUT_SHAPES = [(2304, 1536), (1536, 2304), (2304, 3840), (2304, 3840), (3840, 2304)]

LLAMA_INPUT_DTYPES = [ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat4_b, ttnn.bfloat8_b]

QWEN_INPUT_SHAPES = [(1536, 1536), (1536, 1536), (1536, 3840), (1536, 3840), (3840, 1536)]

QWEN_INPUT_DTYPES = [ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat4_b, ttnn.bfloat8_b]
