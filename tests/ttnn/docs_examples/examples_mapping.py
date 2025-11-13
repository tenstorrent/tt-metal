# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from . import test_data_movement_examples as data_movement

FUNCTION_TO_EXAMPLES_MAPPING_DICT = {
    # Core
    # Tensor Creation
    # Matrix Multiplication
    # Pointwise Unary
    # Pointwise Binary
    # Pointwise Ternary
    # Losses
    # Reduction
    # Data movement
    "ttnn.concat": data_movement.test_concat,
    "ttnn.nonzero": data_movement.test_nonzero,
    "ttnn.pad": data_movement.test_pad,
    "ttnn.permute": data_movement.test_permute,
    "ttnn.reshape": data_movement.test_reshape,
    "ttnn.repeat": data_movement.test_repeat,
    "ttnn.repeat_interleave": data_movement.test_repeat_interleave,
    "ttnn.slice": data_movement.test_slice,
    "ttnn.tilize": data_movement.test_tilize,
    "ttnn.tilize_with_val_padding": data_movement.test_tilize_with_val_padding,
    "ttnn.fill_rm": data_movement.test_fill_rm,
    "ttnn.fill_ones_rm": data_movement.test_fill_ones_rm,
    "ttnn.untilize": data_movement.test_untilize,
    "ttnn.untilize_with_unpadding": data_movement.test_untilize_with_unpadding,
    "ttnn.indexed_fill": data_movement.test_indexed_fill,
    "ttnn.gather": data_movement.test_gather,
    "ttnn.sort": data_movement.test_sort,
    # Normalization
    # Normalization Program Configs
    # Moreh Operations
    # Transformers
    # CCL
    # Embedding
    # Convolution
    # Pooling
    # Vision
    # KV Cache
}
