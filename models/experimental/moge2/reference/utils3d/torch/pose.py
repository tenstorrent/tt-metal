import torch
from torch import Tensor
from typing import *

from .transforms import transform_points, make_affine_matrix
from .utils import matrix_trace, vector_outer


__all__ = ['kabasch', 'umeyama', 'affine_umeyama', 'solve_pose', 'segment_solve_pose', 'solve_poses_sequential', 'segment_solve_poses_sequential', 'pose_graph_optimization']


import torch


class Kabsch(torch.autograd.Function):
    "Customized backward function for Kabsch (SVD) for rotation matrix from covariance matrix."
    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, cov: Tensor, eps: float = 1e-12):
        U, S, Vh = torch.linalg.svd(cov)        
        d = torch.where(torch.linalg.det(U) * torch.linalg.det(Vh) >= 0, 1., -1.)
        s = torch.ones(cov.shape[:-2] + (3,), dtype=cov.dtype, device=cov.device)
        s[..., -1] = d
        R = U @ (s[..., :, None] * Vh)
        ctx.save_for_backward(S, Vh, R, d)
        ctx.eps = eps
        return R

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_R: Tensor):
        S, Vh, R, d = ctx.saved_tensors
        eps = ctx.eps
        
        V = Vh.transpose(-1, -2)
        M = R.transpose(-1, -2) @ grad_R
        
        M_skew_times_2 = M - M.transpose(-1, -2)
        
        V_M_skew_V = Vh @ M_skew_times_2 @ V
        
        S_mod = S.clone()
        S_mod[..., -1] *= d
        D = S_mod[..., :, None] + S_mod[..., None, :]
        D = D + eps
        
        Omega_hat = V_M_skew_V / D
        Omega = V @ Omega_hat @ Vh
        
        grad_cov = R @ Omega
        
        return grad_cov, None


def _kabasch_classic(cov: Tensor, eps: float = 1e-12):
    """Reference implementation. Would encounter NaN gradients when singular values are too close.
    """
    U, _, Vh = torch.linalg.svd(cov)
    det = torch.sign(torch.linalg.det(U) * torch.linalg.det(Vh))
    ones = torch.ones_like(det)
    R = U @ (torch.stack([ones, ones, det], dim=-1)[..., :, None] * Vh)
    return R


def kabasch(cov: Tensor, eps: float = 1e-12):
    """Backward gradients friendly Kabasch method (compute rotation from input covarience matrix).
    """
    return Kabsch.apply(cov, eps)


def umeyama(cov_yx: Tensor, cov_xx: Optional[Tensor] = None, cov_yy: Optional[Tensor] = None, mean_x: Optional[Tensor] = None, mean_y: Optional[Tensor] = None, eps: float = 1e-12) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Umeyama method to solve for scale `s`, rotation `R` and translation `t` such that `y_i ~= s R x_i + t`.

    Parameters
    ----
    - `cov_yx`: (..., 3, 3) covariance matrix between y and x points.
    - `cov_xx`: (..., 3, 3) covariance matrix of x points. If None, no scaling is solved.
    - `cov_yy`: (..., 3, 3) covariance matrix of y points. If None, no scaling is solved.
    - `mean_x`: (..., 3) mean of x points. If None, no translation is solved.
    - `mean_y`: (..., 3) mean of y points. If None, no translation is solved.

    Specifically, based on provided inputs:
    
    - To solve the rotation `R`, `cov_yx` must be given.
    - To solve the scale `s`, at least one of `cov_xx` and `cov_yy` must be given.
        - (Recommended) If both `cov_xx` and `cov_yy` are given, the scale will be solved by minimizing a symmetric cost:
            `||s R X + t - Y||_F^2 / ||Y||_F^2 + ||s R^T (Y - t)  - X||_F^2 / ||X||_F^2`
        - If only `cov_xx` is given, the scale will be solved by minimizing forward cost
            `||s R X  + t - Y||_F^2`
        - If only `cov_yy` is given, the scale will be solved by minimizing inverse cost 
            `||s R^T (Y - t)  - X||_F^2`
    - To solve the translation `t`, provide `mean_x` and `mean_y`.

    Returns
    ----
    - `s`: (...) scale factor. None if both cov_xx and cov_yy are None. 
    - `R`: (..., 3, 3) rotation matrix.
    - `t`: (..., 3) translation vector. None if mean_x or mean_y is None.
    """
    R = kabasch(cov_yx)
    if cov_xx is not None and cov_yy is None:
        s = matrix_trace(cov_yx @ R.swapaxes(-2, -1), dim1=-2, dim2=-1) / matrix_trace(cov_xx, dim1=-2, dim2=-1).clamp_min(eps)
    if cov_xx is None and cov_yy is not None:
        s = matrix_trace(cov_yy, dim1=-2, dim2=-1) / matrix_trace(cov_yx @ R.swapaxes(-2, -1), dim1=-2, dim2=-1).clamp_min(eps)
    elif cov_xx is not None and cov_yy is not None:
        x_fnorm = matrix_trace(cov_xx, dim1=-2, dim2=-1).clamp_min(eps)
        y_fnorm = matrix_trace(cov_yy, dim1=-2, dim2=-1).clamp_min(eps)
        s = torch.sqrt(y_fnorm / x_fnorm)
    else:
        s = None
    if mean_x is not None and mean_y is not None:
        if s is not None:
            t = mean_y - transform_points(mean_x, s[..., None, None] * R)
        else:
            t = mean_y - transform_points(mean_x, R)
    else:
        t = None
    return s, R, t


@torch.no_grad()
def affine_umeyama(cov_yx: Tensor, cov_xx: Tensor, cov_yy: Tensor, mean_x: Tensor, mean_y: Tensor, lam: float = 1e-2, niter: int = 8, eps: float = 1e-12) -> Tuple[Tensor, Tensor]:
    """
    Extended Procrustes analysis to solve for affine transformation `A` and translation `t` such that `y_i ~= A x_i + t`.

    NOTE: This function is indifferentiable due to the iterative solving process.

    Parameters
    ----
    - `cov_yx`: (..., 3, 3) covariance matrix between y and x points.
    - `cov_xx`: (..., 3, 3) covariance matrix of x points.
    - `cov_yy`: (..., 3, 3) covariance matrix of y points.
    - `mean_x`: (..., 3) mean of x points.
    - `mean_y`: (..., 3) mean of y points.
    - `lam`: rigidity regularization weight.
    - `gamma`: symmetricity regularization annealing factor.
    - `niter`: number of iterations for solving.

    Returns
    ----
    - `A`: (..., 3, 3) affine transformation matrix.
    - `t`: (..., 3) translation vector.
    """
    dtype = mean_x.dtype
    R = kabasch(cov_yx)
    tr_xx = matrix_trace(cov_xx, dim1=-2, dim2=-1).clamp_min(eps)
    tr_yy = matrix_trace(cov_yy, dim1=-2, dim2=-1).clamp_min(eps)

    cov_yx, cov_xy = cov_yx / tr_xx[..., None, None], cov_yx.swapaxes(-2, -1) / tr_yy[..., None, None]
    cov_xx, cov_yy = cov_xx / tr_xx[..., None, None], cov_yy / tr_yy[..., None, None]
    
    A, B = torch.zeros_like(R), torch.zeros_like(R)
    I = torch.eye(cov_yx.shape[-1], dtype=dtype, device=cov_yx.device)
    
    def _step(A, B, R, cov_yx, cov_xy, cov_xx, cov_yy, lam, gamma):
        A = (cov_yx + lam * R + gamma * B.swapaxes(-2, -1)) @ torch.linalg.inv(cov_xx + lam * I + gamma * (B @ B.swapaxes(-2, -1)))
        B = (cov_xy + lam * R.swapaxes(-2, -1) + gamma * A.swapaxes(-2, -1)) @ torch.linalg.inv(cov_yy + lam * I + gamma * (A @ A.swapaxes(-2, -1)))
        err = torch.square(A @ B - I).mean(axis=(-2, -1))
        return A, B, err
    
    not_converged = torch.argwhere(torch.ones(R.shape[:-2], dtype=torch.bool, device=R.device))
    for i in range(niter):
        gamma_i = 1.2 ** i - 1
        non_converged_indices = tuple(not_converged.T)
        A[non_converged_indices], B[non_converged_indices], err = _step(*(x[non_converged_indices] for x in (A, B, R, cov_yx, cov_xy, cov_xx, cov_yy)), lam, gamma_i)
        not_converged = not_converged[err >= 1e-6]
        if len(not_converged) == 0:
            break
    t = mean_y - transform_points(mean_x, A)
    return A, t


def solve_pose(
    p: Tensor, 
    q: Tensor, 
    w: Optional[Tensor] = None, 
    *,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2, 
    niter: int = 5,
    eps: float = 1e-12
) -> Tensor:
    """Solve for the pose (transformation from p to q) given weighted point correspondences.
    
    Parameters
    ----
    - `p`: (..., N, 3) source points
    - `q`: (..., N, 3) target points
    - `w`: optional (..., N) weights for each point correspondence. If None, uniform weights are used.
    - `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
        - For 'rigid', only rotation and translation are allowed.
        - For 'similar', uniform scaling, rotation and translation are allowed.
        - For 'affine', full affine transformation is allowed. Using least squares.
    - `lam`: regularization weight for 'affine' mode.
    - `niter`: number of iterations for 'affine' mode.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `pose`: (..., 4, 4) transformations matrix from p to q.
    """
    if w is None:
        w = torch.ones(p.shape[:-1], dtype=p.dtype, device=p.device)
    w_sum = torch.sum(w, dim=-1).clamp_min(eps)
    p_mean = torch.sum(p * w[..., None], dim=-2) / w_sum[..., None]
    q_mean = torch.sum(q * w[..., None], dim=-2) / w_sum[..., None]
    p = p - p_mean[..., None, :]
    q = q - q_mean[..., None, :]
    pw = p * w[..., None]
    qw = q * w[..., None]
    cov_qp = torch.sum(vector_outer(qw, p), dim=-3) / w_sum[..., None, None]
    if mode == 'similar' or mode == 'affine':
        cov_pp = torch.sum(vector_outer(pw, p), dim=-3) / w_sum[..., None, None]
        cov_qq = torch.sum(vector_outer(qw, q), dim=-3) / w_sum[..., None, None]
    
    if mode == 'rigid':
        _, R, t = umeyama(cov_qp, mean_x=p_mean, mean_y=q_mean, eps=eps)
        pose = make_affine_matrix(R, t)
    elif mode == 'similar':
        s, R, t = umeyama(cov_qp, cov_xx=cov_pp, cov_yy=cov_qq, mean_x=p_mean, mean_y=q_mean, eps=eps)
        pose = make_affine_matrix(s * R, t)
    elif mode == 'affine':
        A, t = affine_umeyama(cov_qp, cov_pp, cov_qq, p_mean, q_mean, lam=lam, niter=niter, eps=eps)
        pose = make_affine_matrix(A, t)
    
    return pose


def segment_solve_pose(
    p: Tensor, 
    q: Tensor, 
    w: Optional[Tensor] = None, 
    *,
    offsets: Tensor, 
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2, 
    niter: int = 5,
    eps: float = 1e-12
) -> Tensor:
    """Solve for the pose (transformation from p to q: q ≈ pose @ p) given weighted point correspondences.

    NOTE: Affine mode is solved by iterative method and may be indifferentiable. Use with `torch.no_grad()` if you don't need gradients.
    
    Parameters
    ----
    - `p`: (N, 3) source points
    - `q`: (N, 3) target points
    - `w`: (N,) weights for each point correspondence
    - `offsets`: (S + 1,) segment offsets. Points in each segment belong to the same rigid / affine body.
    - `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
        - For 'rigid', only rotation and translation are allowed.
        - For 'similar', uniform scaling, rotation and translation are allowed.
        - For 'affine', full affine transformation is allowed. Using least squares.
    - `lam`: regularization weight for 'affine' mode.
    - `niter`: number of iterations for 'affine' mode.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `pose`: (S, 4, 4) transformations matrix from p to q.
    """
    if w is None:
        w = torch.ones(p.shape[:-1], device=p.device, dtype=p.dtype)

    lengths = torch.diff(offsets)
    w_sum = torch.segment_reduce(w, 'sum', offsets=offsets, axis=0).clamp_min(eps)
    p_mean = torch.segment_reduce(p * w[..., None], 'sum', offsets=offsets, axis=0) / w_sum[:, None]
    q_mean = torch.segment_reduce(q * w[..., None], 'sum', offsets=offsets, axis=0) / w_sum[:, None]
    p = p - torch.repeat_interleave(p_mean, lengths, dim=0)
    q = q - torch.repeat_interleave(q_mean, lengths, dim=0)
    pw = p * w[..., None]
    qw = q * w[..., None]
    cov_qp = torch.segment_reduce(vector_outer(qw, p), 'sum', offsets=offsets, axis=0) / w_sum[:, None, None]
    if mode == 'similar' or mode == 'affine':
        cov_pp = torch.segment_reduce(vector_outer(pw, p), 'sum', offsets=offsets, axis=0) / w_sum[:, None, None]
        cov_qq = torch.segment_reduce(vector_outer(qw, q), 'sum', offsets=offsets, axis=0) / w_sum[:, None, None]

    if mode == 'rigid':
        _, R, t = umeyama(cov_qp, mean_x=p_mean, mean_y=q_mean, eps=eps)
        pose = make_affine_matrix(R, t)
    elif mode == 'similar':
        s, R, t = umeyama(cov_qp, cov_xx=cov_pp, cov_yy=cov_qq, mean_x=p_mean, mean_y=q_mean, eps=eps)
        pose = make_affine_matrix(s * R, t)
    elif mode == 'affine':
        A, t = affine_umeyama(cov_qp, cov_pp, cov_qq, p_mean, q_mean, lam=lam, niter=niter, eps=eps)
        pose = make_affine_matrix(A, t)
    
    return pose


def solve_poses_sequential(
    trajectories: Tensor,
    weights: Optional[Tensor] = None,
    *,
    accum: Optional[Tuple[Tensor, ...]] = None,
    min_valid_size: int = 3,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2,
    niter: int = 8,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor, Tensor], Tensor, Tensor, Tuple[Tensor, ...]]:
    """
    Given trajectories of points over time, sequentially solve for the poses (transformations from canonical to each frame) of each body at each frame.

    Parameters
    ----
    - `trajectories`: (T, ..., N, 3) posed points. T is number of frames. `...` is optional batch dimensions. N is number of points per group.
    - `weights`: (T, ..., N) quardratic error term weights for each point at each frame
    - `accum`: accumulated statistics from previous calls. If None, start fresh.
    - `min_valid_size`: minimum number of valid points in each frame to consider the segment / group valid.
    - `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
        - For 'rigid', only rotation and translation are allowed.
        - For 'similar', uniform scaling, rotation and translation are allowed.
        - For 'affine', full affine transformation is allowed. Using least squares.
    - `lam`: rigidity regularization weight for 'affine' mode.
    - `niter`: number of iterations for 'affine' mode.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `poses`: (T, ..., 4, 4) transformations from canonical to each frame.
    - `valid`: (T, ...) boolean mask indicating valid segments
    - `stats`: canonical statistics of each group `(mu, cov, tot_w, nnz)`.
    - `canonical_points`: (..., N, 3) canonical points.
    - `err`: (..., N) per-point RMS error over all time.
    - `accum`: per-point accumulated statistics. Pass it to the next call for incremental solving.
    """
    dtype = trajectories.dtype
    device = trajectories.device
    num_frames = trajectories.shape[0]
    num_points = trajectories.shape[-2]
    batch_shape = trajectories.shape[1:-2]

    if weights is None:
        weights = torch.ones((num_frames, *batch_shape, num_points), dtype=dtype, device=device)

    poses = torch.zeros((num_frames, *batch_shape, 4, 4), dtype=dtype, device=device)

    if accum is not None:
        accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz = [a.clone() for a in accum]
    else:
        accum_sqrtw = torch.zeros((*batch_shape, num_points), dtype=dtype, device=device)
        accum_sqrtwx = torch.zeros((*batch_shape, num_points, 3), dtype=dtype, device=device)
        accum_sqrtwxx = torch.zeros((*batch_shape, num_points, 3, 3), dtype=dtype, device=device)
        accum_w = torch.zeros((*batch_shape, num_points), dtype=dtype, device=device)
        accum_wx = torch.zeros((*batch_shape, num_points, 3), dtype=dtype, device=device)
        accum_wxx = torch.zeros((*batch_shape, num_points, 3, 3), dtype=dtype, device=device)
        accum_nnz = torch.zeros((*batch_shape, num_points), dtype=dtype, device=device)

    for i in range(num_frames):
        mean_sqrtwx = accum_sqrtwx / accum_sqrtw.clamp_min(eps)[..., None]
        wi, yi = weights[i], trajectories[i]
        sqrtwi = torch.sqrt(wi)
        w = sqrtwi * accum_sqrtw
        sum_w = torch.sum(w, dim=-1).clamp_min(eps)
        center_x = torch.sum(sqrtwi[..., None] * accum_sqrtwx, dim=-2) / sum_w[..., None]
        center_y = torch.sum(w[..., None] * yi, dim=-2) / sum_w[..., None]
        xc = mean_sqrtwx - center_x[..., None, :]
        yc = yi - center_y[..., None, :]
        cov_yx = torch.einsum('...i,...ij,...ik->...jk', w, yc, xc) / sum_w[..., None, None]
        if mode == 'affine' or mode == 'similar':
            cov_xx = (torch.einsum('...i,...ij,...ik->...jk', w, xc, xc) + torch.einsum('...i,...ijk->...jk', sqrtwi, accum_sqrtwxx)) / sum_w[..., None, None]
            cov_yy = torch.einsum('...i,...ij,...ik->...jk', w, yc, yc) / sum_w[..., None, None]

        if mode == 'rigid':
            _, R, t = umeyama(cov_yx, mean_x=center_x, mean_y=center_y, eps=eps)
            poses[i] = make_affine_matrix(R, t)
        elif mode == 'similar':
            s, R, t = umeyama(cov_yx, cov_xx=cov_xx, mean_x=center_x, mean_y=center_y, eps=eps)
            poses[i] = make_affine_matrix(s * R, t)
        elif mode == 'affine':
            A, t = affine_umeyama(cov_yx, cov_xx, cov_yy, center_x, center_y, lam=lam, niter=niter, eps=eps)
            poses[i] = make_affine_matrix(A, t)

        xi = transform_points(yi, torch.linalg.inv(poses[i])[..., None, :, :])

        # Update accum
        old_mean_sqrtwx, old_accum_sqrtw = mean_sqrtwx.clone(), accum_sqrtw.clone()
        accum_sqrtw = accum_sqrtw + sqrtwi
        accum_sqrtwx = accum_sqrtwx + sqrtwi[..., None] * xi
        mean_sqrtwx = accum_sqrtwx / accum_sqrtw.clamp_min(eps)[..., None]
        accum_sqrtwxx = accum_sqrtwxx + old_accum_sqrtw[..., None, None] * vector_outer(mean_sqrtwx - old_mean_sqrtwx) + sqrtwi[..., None, None] * vector_outer(xi - mean_sqrtwx)

        mean_wx = accum_wx / accum_w.clamp_min(eps)[..., None]
        old_mean_wx, old_accum_w = mean_wx.clone(), accum_w.clone()
        accum_w = accum_w + wi
        accum_wx = accum_wx + wi[..., None] * xi
        mean_wx = accum_wx / accum_w.clamp_min(eps)[..., None]
        accum_wxx = accum_wxx + old_accum_w[..., None, None] * vector_outer(mean_wx - old_mean_wx) + wi[..., None, None] * vector_outer(xi - mean_wx)
        accum_nnz = accum_nnz + (wi > 0).to(dtype)

    tot_w = torch.sum(accum_w, dim=-1)
    mu = torch.sum(accum_wx, dim=-2) / tot_w.clamp_min(eps)[..., None]
    mean_wx = accum_wx / accum_w.clamp_min(eps)[..., None]
    sigma = torch.sum(accum_wxx + accum_w[..., None, None] * vector_outer(mu[..., None, :] - mean_wx), dim=-3) / tot_w.clamp_min(eps)[..., None, None]
    nnz = torch.sum(accum_nnz, dim=-1)
    valid = torch.sum(weights > 0, dim=-1) >= min_valid_size
    err = torch.sqrt(matrix_trace(accum_wxx, dim1=-2, dim2=-1) / accum_nnz.clamp_min(eps))

    return poses, valid, (mu, sigma, tot_w, nnz), mean_wx, err, (accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz)


def segment_solve_poses_sequential(
    trajectories: Tensor,
    weights: Optional[Tensor] = None,
    offsets: Tensor = None,
    *,
    accum: Optional[Tuple[Tensor, ...]] = None,
    min_valid_size: int = 3,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2,
    niter: int = 8,
    eps: float = 1e-12,
) -> Tuple[Tensor, Tensor, Tuple[Tensor, Tensor, Tensor, Tensor], Tensor, Tensor, Tuple[Tensor, ...]]:
    """
    Segment array mode for `solve_poses_sequential`.

    Parameters
    ----
    - `trajectories`: (T, N, 3) posed points.
    - `weights`: (T, N) quardratic error term weights for each point at each frame
    - `offsets`: (S + 1,) segment offsets. Points in each segment belong to the same rigid / affine body.
    - `accum`: accumulated statistics from previous calls. If None, start fresh.
    - `min_valid_size`: minimum number of valid points in each frame to consider the segment / group valid.
    - `mode`: mode of transformation to apply. Can be 'rigid', 'similar', or 'affine'.
    - `lam`: rigidity regularization weight for 'affine' mode.
    - `niter`: number of iterations for 'affine' mode.
    - `eps`: small value to prevent division by zero.

    Returns
    ----
    - `poses`: (T, S, 4, 4) transformations from canonical to each frame.
    - `valid`: (T, S) boolean mask indicating valid segments.
    - `stats`: canonical statistics `(mu, cov, tot_w, nnz)`.
    - `canonical_points`: (N, 3) canonical points.
    - `err`: (N,) per-point RMS error over all time.
    - `accum`: per-point accumulated statistics for incremental solving.
    """
    dtype = trajectories.dtype
    device = trajectories.device
    num_frames = trajectories.shape[0]
    num_points = trajectories.shape[1]

    if weights is None:
        weights = torch.ones((num_frames, num_points), dtype=dtype, device=device)

    num_segments = offsets.shape[0] - 1
    lengths = torch.diff(offsets)
    poses = torch.zeros((num_frames, num_segments, 4, 4), dtype=dtype, device=device)

    if accum is not None:
        accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz = [a.clone() for a in accum]
    else:
        accum_sqrtw = torch.zeros((num_points,), dtype=dtype, device=device)
        accum_sqrtwx = torch.zeros((num_points, 3), dtype=dtype, device=device)
        accum_sqrtwxx = torch.zeros((num_points, 3, 3), dtype=dtype, device=device)
        accum_w = torch.zeros((num_points,), dtype=dtype, device=device)
        accum_wx = torch.zeros((num_points, 3), dtype=dtype, device=device)
        accum_wxx = torch.zeros((num_points, 3, 3), dtype=dtype, device=device)
        accum_nnz = torch.zeros((num_points,), dtype=dtype, device=device)

    for i in range(num_frames):
        mean_sqrtwx = accum_sqrtwx / accum_sqrtw.clamp_min(eps)[:, None]
        wi, yi = weights[i], trajectories[i]
        sqrtwi = torch.sqrt(wi)
        w = sqrtwi * accum_sqrtw
        sum_w = torch.segment_reduce(w, 'sum', offsets=offsets, axis=0).clamp_min(eps)
        center_x = torch.segment_reduce(sqrtwi[:, None] * accum_sqrtwx, 'sum', offsets=offsets, axis=0) / sum_w[:, None]
        center_y = torch.segment_reduce(w[:, None] * yi, 'sum', offsets=offsets, axis=0) / sum_w[:, None]
        center_x_broadcast = torch.repeat_interleave(center_x, lengths, dim=0)
        center_y_broadcast = torch.repeat_interleave(center_y, lengths, dim=0)
        xc = mean_sqrtwx - center_x_broadcast
        yc = yi - center_y_broadcast
        cov_yx = torch.segment_reduce(w[:, None, None] * vector_outer(yc, xc), 'sum', offsets=offsets, axis=0) / sum_w[:, None, None]
        if mode == 'affine' or mode == 'similar':
            cov_xx = torch.segment_reduce(sqrtwi[:, None, None] * accum_sqrtwxx + w[:, None, None] * vector_outer(xc), 'sum', offsets=offsets, axis=0) / sum_w[:, None, None]
            cov_yy = torch.segment_reduce(w[:, None, None] * vector_outer(yc), 'sum', offsets=offsets, axis=0) / sum_w[:, None, None]

        if mode == 'rigid':
            _, R, t = umeyama(cov_yx, mean_x=center_x, mean_y=center_y, eps=eps)
            poses[i] = make_affine_matrix(R, t)
        elif mode == 'similar':
            s, R, t = umeyama(cov_yx, cov_xx=cov_xx, mean_x=center_x, mean_y=center_y, eps=eps)
            poses[i] = make_affine_matrix(s * R, t)
        elif mode == 'affine':
            A, t = affine_umeyama(cov_yx, cov_xx, cov_yy, center_x, center_y, lam=lam, niter=niter, eps=eps)
            poses[i] = make_affine_matrix(A, t)

        xi = transform_points(yi, torch.repeat_interleave(torch.linalg.inv(poses[i]), lengths, dim=0))

        # Update accum
        old_mean_sqrtwx, old_accum_sqrtw = mean_sqrtwx.clone(), accum_sqrtw.clone()
        accum_sqrtw = accum_sqrtw + sqrtwi
        accum_sqrtwx = accum_sqrtwx + sqrtwi[:, None] * xi
        mean_sqrtwx = accum_sqrtwx / accum_sqrtw.clamp_min(eps)[:, None]
        accum_sqrtwxx = accum_sqrtwxx + old_accum_sqrtw[:, None, None] * vector_outer(mean_sqrtwx - old_mean_sqrtwx) + sqrtwi[:, None, None] * vector_outer(xi - mean_sqrtwx)

        mean_wx = accum_wx / accum_w.clamp_min(eps)[:, None]
        old_mean_wx, old_accum_w = mean_wx.clone(), accum_w.clone()
        accum_w = accum_w + wi
        accum_wx = accum_wx + wi[:, None] * xi
        mean_wx = accum_wx / accum_w.clamp_min(eps)[:, None]
        accum_wxx = accum_wxx + old_accum_w[:, None, None] * vector_outer(mean_wx - old_mean_wx) + wi[:, None, None] * vector_outer(xi - mean_wx)
        accum_nnz = accum_nnz + (wi > 0).to(dtype)

    tot_w = torch.segment_reduce(accum_w, 'sum', offsets=offsets, axis=0)
    mu = torch.segment_reduce(accum_wx, 'sum', offsets=offsets, axis=0) / tot_w.clamp_min(eps)[:, None]
    mean_wx = accum_wx / accum_w.clamp_min(eps)[:, None]
    mu_broadcast = torch.repeat_interleave(mu, lengths, dim=0)
    sigma = torch.segment_reduce(accum_wxx + accum_w[:, None, None] * vector_outer(mu_broadcast - mean_wx), 'sum', offsets=offsets, axis=0) / tot_w.clamp_min(eps)[:, None, None]
    nnz = torch.segment_reduce(accum_nnz, 'sum', offsets=offsets, axis=0)
    # `torch.segment_reduce` requires axis == offsets.ndim - 1, so reduce along dim 0 by transposing.
    valid = torch.segment_reduce((weights > 0).to(dtype).transpose(0, 1).contiguous(), 'sum', offsets=offsets, axis=0).transpose(0, 1) >= min_valid_size
    err = torch.sqrt(matrix_trace(accum_wxx, dim1=-2, dim2=-1) / accum_nnz.clamp_min(eps))

    return poses, valid, (mu, sigma, tot_w, nnz), mean_wx, err, (accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz)


def _pose_graph_optimization_construct_laplacian(edge: Tensor, num_nodes: int, R: Tensor, w: Tensor) -> Tensor:
    # For each edge i->j, contributes:
    # - `-w_ij * R_ij^T` to block (i, j)
    # - `-w_ij * R_ij` to block (j, i),
    # - `w_ij * I` to diagonal block (i, i) and `w_ij * I` to diagonal block (j, j).
    w_agg = torch.zeros(num_nodes, device=R.device, dtype=R.dtype).index_add(0, edge.reshape(-1), w.repeat_interleave(2))
    diag_elements = w_agg.repeat_interleave(3)   # (N * 3,)
    edge_elements = (-w[:, None, None] * R.mT).reshape(-1)   # (E * 3 * 3,)
    laplacian_data = torch.cat([diag_elements, edge_elements.reshape(-1), edge_elements.reshape(-1)], dim=0)

    local3 = torch.arange(3, device=R.device)
    local3x3 = torch.stack(torch.meshgrid(local3, local3, indexing='ij'), dim=-1)   # to get the local 3x3 block coordinates for each edge
    diag_coords = (torch.arange(num_nodes, device=R.device)[:, None].expand(num_nodes, 3) * 3 + local3[None, :]).reshape(-1, 1).expand(-1, 2)   # (N * 3, 2)
    edge_coords = (edge[:, None, None, :] * 3 + local3x3[None, :, :, :]).reshape(-1, 2)                       # (E * 3 * 3, 2)
    laplacian_coords = torch.cat([diag_coords, edge_coords, edge_coords.flip(-1)], dim=0)               # (2, N * 3 + 2 * E * 3 * 3)

    laplacian = torch.sparse_coo_tensor(laplacian_coords.T, laplacian_data, size=(num_nodes * 3, num_nodes * 3), check_invariants=False)
    return laplacian


def _pose_graph_optimization_eigen_decomposition(laplacian: Tensor) -> Tensor:
    _, eigenvectors = torch.lobpcg(laplacian, k=3, largest=False, tol=1e-5)
    R_global = eigenvectors.reshape(-1, 3, 3)
    R_global = torch.cat([
        R_global[:, :, :2],
        torch.sign(torch.linalg.det(R_global))[:, None, None] * R_global[:, :, 2:3]
    ], dim=-1)
    R_global = kabasch(R_global)    # Ensure SO(3)

    return R_global


def _pose_graph_optimization_procrustes_iteration(R_global: Tensor, edges: Tensor, R_rel: Tensor, w: Tensor, niter: int = 10) -> Tensor:
    DAMP = 0.4
    edges_flat, edges_flat_swap = edges.reshape(-1), edges.flip(1).reshape(-1)

    w_R = w[:, None, None] * R_rel
    w_R_dual = torch.stack([w_R, w_R.mT], dim=1).reshape(-1, 3, 3) 
    w_agg = torch.zeros(R_global.shape[0], device=R_global.device, dtype=R_global.dtype).index_add(0, edges_flat, w.repeat_interleave(2))
    for _ in range(niter):
        M = (DAMP * w_agg[..., None, None] * R_global).index_add(0, edges_flat_swap, w_R_dual @ R_global.index_select(0, edges_flat))
        R_global = kabasch(M)
    return R_global


def pose_graph_optimization(num_nodes: int, edges: Tensor, poses: Tensor, w: Tensor | None = None, niter: int = 10) -> tuple[Tensor, Tensor, Tensor]:
    """Pose graph optimization to solve for global poses given relative poses (must be rigid transformations).

    Parameters
    ----
    - `num_nodes`: number of nodes `N` in the pose graph.
    - `edges`: (E, 2) edge list of the pose graph. Each edge is represented by a pair of node indices `i -> j`.
    - `poses`: (E, 4, 4) relative poses of transformation from node `i` to node `j` for each edge. Must be rigid transformations.
    - `w`: (E,) optional weights for each edge.
    - `niter`: number of Procrustes iterations to refine global poses. If 0, only the initial solution by Laplacian SVD is returned.

    Returns
    ----
    - `poses_global`: (N, 4, 4) global poses (world-to-camera, canonical-to-observation, global-to-node, etc.) for each node.

        `poses_relative[i->j] ≈ poses_global[j] @ poses_global[i].inv()`
    """
    if w is None:
        w = torch.ones(edges.shape[0], device=poses.device, dtype=poses.dtype)

    if poses.shape[-1] == 3:
        # Rotation only
        R_relative, t_relative = poses[:, :3, :3], None
    else:
        # Rigid transformation
        R_relative, t_relative = poses[:, :3, :3], poses[:, :3, 3]

    # Solve initial global rotations by laplacian eigen-decomposition.
    laplacian = _pose_graph_optimization_construct_laplacian(edges, num_nodes, R_relative, w)
    R_global = _pose_graph_optimization_eigen_decomposition(laplacian)
    
    # Refine global rotations by Procrustes iterations.
    if niter > 0:
        R_global = _pose_graph_optimization_procrustes_iteration(R_global, edges, R_relative, w, niter=niter)

    # Solve global translations 
    if t_relative is not None:
        w_t = w[:, None] * t_relative
        b = torch.zeros((num_nodes, 3), device=R_relative.device, dtype=R_relative.dtype).index_add(0, edges[:, 1], w_t).index_add(0, edges[:, 0], -(R_relative.mT @ w_t[:, :, None]).squeeze(-1)).reshape(-1)
        # NOTE: currently we have to use dense solver for translations since PyTorch doesn't support sparse linear solver well.
        t_global = torch.linalg.lstsq(laplacian.to_dense(), b).solution.reshape(num_nodes, 3)    
    else:
        t_global = None

    if t_global is not None:
        poses_global = make_affine_matrix(R_global, t_global)
    else:
        poses_global = R_global
    return poses_global

