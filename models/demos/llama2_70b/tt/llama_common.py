import torch
from typing import Tuple
import tt_lib
from models.utility_functions import tt2torch_tensor, torch2tt_tensor


def rms_decomp(x, norm_weight, eps):
    squared = tt_lib.tensor.pow(x, 2)
    # mean_squared = tt_lib.tensor.mean(squared, )
    sum_squared = tt_lib.tensor.reduce(squared, tt_lib.tensor.ReduceOpMath.SUM, tt_lib.tensor.ReduceOpDim.W, scaler=1.0)
    # Tensor is 1,1,32,1+31 now
    mean_squared = tt_lib.tensor.div_unary(sum_squared, x.shape()[-1])
    mean_squared_eps = tt_lib.tensor.add_unary(mean_squared, eps)
    rms = tt_lib.tensor.pow(mean_squared_eps, 0.5)
    rms_recip = tt_lib.tensor.recip(rms)
    normed_x = tt_lib.tensor.bcast(x, rms_recip, math_op=tt_lib.tensor.BcastOpMath.MUL, dim=tt_lib.tensor.BcastOpDim.W)
    norm_out = tt_lib.tensor.mul(normed_x, norm_weight)
    return norm_out


def tt_all_reduce(tensors):
    """
    reduction on a list of tensors
    """
    base_tensor = tensors[0]
    for tensor in tensors[1:]:
        base_tensor = tt_lib.tensor.add(base_tensor, tensor)
    dev = base_tensor.device()
    # Emulate replication on all chips
    res_pt = tt2torch_tensor(base_tensor)
    res = [torch2tt_tensor(res_pt.clone(), dev) for _ in range(len(tensors))]
    return res


def generate_rot_emb(dhead, end, batch):
    cos, sin = tt_precompute_freqs(dhead, end)
    rot_mat = freqs_to_rotation_matrix(cos, sin)
    return rot_mat


def precompute_freqs(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for sine and cosine values with given dimensions.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors containing cosine and sine values.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def freqs_to_rotation_matrix(cos_freqs, sin_freqs):
    """
    Transform cos/sin frequencies to a rotation matrix.
    """
    emb_size, emb_dim = cos_freqs.shape
    dhead = emb_dim * 2
    rot_emb_matrix = torch.zeros(emb_size, dhead, dhead)
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(0, dhead, 2)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(1, dhead, 2)] = cos_freqs.clone()
    rot_emb_matrix[..., torch.arange(0, dhead, 2), torch.arange(1, dhead, 2)] = -sin_freqs.clone()
    rot_emb_matrix[..., torch.arange(1, dhead, 2), torch.arange(0, dhead, 2)] = sin_freqs.clone()

    rot_emb_matrix = rot_emb_matrix.transpose(-1, -2)  # Necessary for correct rotation when applied as (x @ R)
    return rot_emb_matrix


def gather_rotary_emb(rot_emb_matrix, position_ids):
    """
    Gather the rotary embeddings for a given position_ids
    """
    batch_size, seqlen = position_ids.shape
    emb_size, _, dhead = rot_emb_matrix.shape
    position_ids = position_ids.view(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, dhead, dhead)
    rot_emb = rot_emb_matrix.gather(0, position_ids).view(batch_size, seqlen, dhead, dhead)
    return rot_emb


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rotation_mat: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given cosine and sine frequency tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        rotation_mat (torch.Tensor): Precomputed rotation matrix.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    xq_out = xq @ rotation_mat
    xk_out = xk @ rotation_mat
    return xq_out, xk_out
