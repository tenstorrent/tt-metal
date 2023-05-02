import torch
from typing import List, Tuple, Union, Optional
from .conversion_wrapper import convert_tt_tensors_wrapper
from .. import tensor as ttl_tensor


@convert_tt_tensors_wrapper
def full(size: List[int], fill_value: float) -> ttl_tensor.Tensor:
    """
    Creates a ``tt_lib.tensor.Tensor`` of size ``size`` filled with ``fill_value``.
    """
    return torch.full(size, fill_value)


@convert_tt_tensors_wrapper
def reshape(
    input: ttl_tensor.Tensor, N: int, C: int, H: int, W: int
) -> ttl_tensor.Tensor:
    """
    Returns a new ``tt_lib.tensor.Tensor`` with the same data and number of elements as ``input``, but with the specified shape.
    """
    return torch.reshape(input, (N, C, H, W))


@convert_tt_tensors_wrapper
def chunk(
    input: ttl_tensor.Tensor, chunks: int, dim: int = 0
) -> List[ttl_tensor.Tensor]:
    """
    Attempts to split a ``tt_lib.tensor.Tensor`` into the specified number of chunks. Each chunk is a new copy of part of the input tensor.
    """
    return torch.chunk(input, chunks, dim)


@convert_tt_tensors_wrapper
def conv2d(
    input: ttl_tensor.Tensor,
    weight: ttl_tensor.Tensor,
    bias: Optional[ttl_tensor.Tensor],
    stride: Union[int, Tuple] = 1,
    padding: Union[int, str, Tuple] = 0,
    dilation: Union[int, Tuple] = 1,
    groups: int = 1,
) -> ttl_tensor.Tensor:
    """
    Applies a 2D convolution over an input image composed of several input planes.
    """
    return torch.nn.functional.conv2d(
        input, weight, bias, stride, padding, dilation, groups
    )


@convert_tt_tensors_wrapper
def group_norm(
    input: ttl_tensor.Tensor,
    num_groups: int,
    weight: Optional[ttl_tensor.Tensor] = None,
    bias: Optional[ttl_tensor.Tensor] = None,
    eps: float = 1e-05,
) -> ttl_tensor.Tensor:
    """
    Applies Group Normalization for last certain number of dimensions.
    """
    return torch.nn.functional.group_norm(input, num_groups, weight, bias, eps)


@convert_tt_tensors_wrapper
def layer_norm(
    input: ttl_tensor.Tensor,
    normalized_shape: Union[int, List[int]],
    weight: Optional[ttl_tensor.Tensor] = None,
    bias: Optional[ttl_tensor.Tensor] = None,
    eps: float = 1e-05,
) -> ttl_tensor.Tensor:
    """
    Applies Layer Normalization for last certain number of dimensions.
    """
    return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)


@convert_tt_tensors_wrapper
def pad(
    input: ttl_tensor.Tensor,
    pad: Tuple[int],
    mode: str = "constant",
    value: Optional[int] = None,
) -> ttl_tensor.Tensor:
    """
    Pads tensor.
    """
    return torch.nn.functional.pad(input, pad, mode, value)


@convert_tt_tensors_wrapper
def repeat_interleave(
    input: ttl_tensor.Tensor,
    repeats: Union[ttl_tensor.Tensor, int],
    dim: Optional[int] = None,
    *,
    output_size: Optional[int] = None
) -> ttl_tensor.Tensor:
    """
    Repeat elements of a tensor.
    """
    return torch.repeat_interleave(input, repeats, dim, output_size=output_size)


@convert_tt_tensors_wrapper
def concat(tensors: List[ttl_tensor.Tensor], dim: int = 0) -> ttl_tensor.Tensor:
    """
    Concatenates the given sequence of ``seq`` tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty.
    """
    return torch.concat(tensors, dim)


@convert_tt_tensors_wrapper
def sigmoid(input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    """
    Computes the logitistic sigmoid on the elements of ``input``.
    """
    return torch.sigmoid(input)


@convert_tt_tensors_wrapper
def silu(input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
    r"""
    Applies the Sigmoid Linear Unit (SiLU) function, element-wise.
    The SiLU function is also known as the swish function.

    .. math::
        \text{silu}(x) = x * \sigma(x), \text{where } \sigma(x) \text{ is the logistic sigmoid.}
    """
    return torch.nn.functional.silu(input)


@convert_tt_tensors_wrapper
def softmax(input: ttl_tensor.Tensor, dim: Optional[int] = None) -> ttl_tensor.Tensor:
    r"""
    Applies a softmax function.

    Softmax is defined as:

    :math:`\text{Softmax}(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    """
    return torch.nn.functional.softmax(input, dim)


class Conv2d(torch.nn.Module):
    r"""
    Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)

    """

    @convert_tt_tensors_wrapper
    def __init__(
        self,
        weights: ttl_tensor.Tensor,
        biases: ttl_tensor.Tensor,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = 1,
        padding: Union[int, str, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.pt_fallback = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.pt_fallback.weight = torch.nn.Parameter(weights)
        self.pt_fallback.bias = torch.nn.Parameter(biases)

    @convert_tt_tensors_wrapper
    def forward(self, input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
        return self.pt_fallback(input)


class GroupNorm(torch.nn.Module):
    r"""
    Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    """

    @convert_tt_tensors_wrapper
    def __init__(
        self,
        weights: ttl_tensor.Tensor,
        biases: ttl_tensor.Tensor,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
    ):
        super().__init__()
        self.pt_fallback = torch.nn.GroupNorm(num_groups, num_channels, eps, affine)
        self.pt_fallback.weight = torch.nn.Parameter(weights)
        self.pt_fallback.bias = torch.nn.Parameter(biases)

    @convert_tt_tensors_wrapper
    def forward(self, input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
        return self.pt_fallback(input)


class LayerNorm(torch.nn.Module):
    r"""
    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    """

    @convert_tt_tensors_wrapper
    def __init__(
        self,
        weights: ttl_tensor.Tensor,
        biases: ttl_tensor.Tensor,
        normalized_shape: Union[int, List[int]],
        eps: float = 1e-05,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.pt_fallback = torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
        self.pt_fallback.weight = torch.nn.Parameter(weights)
        self.pt_fallback.bias = torch.nn.Parameter(biases)

    @convert_tt_tensors_wrapper
    def forward(self, input: ttl_tensor.Tensor) -> ttl_tensor.Tensor:
        return self.pt_fallback(input)
