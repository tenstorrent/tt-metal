# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Minimal stand-in for the ttnn module, shaped to produce the same HTML
that docs.tenstorrent.com emits for ttnn.remainder."""


def remainder(
    input_tensor_a, input_tensor_b, *, memory_config=None, dtype=None, output_tensor=None, activations=None, queue_id=0
):
    r"""Performs remainder on :attr:`input_tensor_a` and :attr:`input_tensor_b`.

    Args:
        input_tensor_a (ttnn.Tensor): the input tensor.
        input_tensor_b (ttnn.Tensor or Number): the input tensor.

    Keyword Args:
        memory_config (ttnn.MemoryConfig, optional): memory configuration for the operation. Defaults to `None`.
        dtype (ttnn.DataType, optional): data type for the output tensor. Defaults to `None`.
        output_tensor (ttnn.Tensor, optional): preallocated output tensor. Defaults to `None`.
        activations (List[str], optional): list of activation functions to apply to the output tensor. Defaults to `None`.
        queue_id (int, optional): command queue id. Defaults to `0`.

    Returns:
        ttnn.Tensor: the output tensor.

    Note:
        Supported dtypes and layouts:

        .. list-table::
           :header-rows: 1

           * - Dtypes
             - Layouts
           * - BFLOAT16, BFLOAT8_B, BFLOAT4_B, FLOAT32, INT32, UINT32
             - TILE, ROW_MAJOR

        If the input tensor is ROW_MAJOR layout, it will be internally converted to TILE layout.

        Operands must have the same dtype.

    .. admonition:: Example

        .. code-block:: python

            # Create two tensors for remainder operation
            tensor1 = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT)
            tensor2 = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT)
            output = ttnn.remainder(tensor1, tensor2)
    """


def add(input_tensor_a, input_tensor_b):
    """Adds two tensors together elementwise.

    Args:
        input_tensor_a (ttnn.Tensor): the input tensor.
        input_tensor_b (ttnn.Tensor): the input tensor.

    Returns:
        ttnn.Tensor: the output tensor.
    """


def subtract(input_tensor_a, input_tensor_b):
    """Subtracts one tensor from another elementwise.

    Returns:
        ttnn.Tensor: the output tensor.
    """
