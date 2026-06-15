import numpy as np
from numpy import ndarray
from typing import *
from numbers import Number
from ..helpers import no_warnings

from .transforms import make_affine_matrix, transform_points
from .utils import safe_inv, vector_outer


__all__ = [
    'kabasch',
    'umeyama',
    'affine_umeyama',
    'solve_pose',
    'segment_solve_pose',
    'solve_poses_sequential',
    'segment_solve_poses_sequential',
]


def kabasch(cov: ndarray) -> ndarray:
    U, _, Vh = np.linalg.svd(cov)
    Vh[..., 2, :] *= np.sign(np.linalg.det(U @ Vh))[..., None]
    R = U @ Vh
    return R


def umeyama(cov_yx: ndarray, cov_xx: Optional[ndarray] = None, cov_yy: Optional[ndarray] = None, mean_x: Optional[ndarray] = None, mean_y: Optional[ndarray] = None) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Procrustes analysis to solve for scale `s`, rotation `R` and translation `t` such that `y_i ~= s R x_i + t`.

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
    dtype = cov_yx.dtype
    R = kabasch(cov_yx)
    if cov_xx is not None and cov_yy is None:
        s = np.trace(cov_yx @ R.swapaxes(-2, -1), axis1=-2, axis2=-1) / np.maximum(np.trace(cov_xx, axis1=-2, axis2=-1), np.finfo(dtype).tiny)
    if cov_xx is None and cov_yy is not None:
        s = np.trace(cov_yy, axis1=-2, axis2=-1) / np.maximum(np.trace(cov_yx @ R.swapaxes(-2, -1), axis1=-2, axis2=-1), np.finfo(dtype).tiny)
    elif cov_xx is not None and cov_yy is not None:
        x_fnorm = np.maximum(np.trace(cov_xx, axis1=-2, axis2=-1), np.finfo(dtype).tiny)
        y_fnorm = np.maximum(np.trace(cov_yy, axis1=-2, axis2=-1), np.finfo(dtype).tiny)
        s = np.sqrt(y_fnorm / x_fnorm)
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


def affine_umeyama(cov_yx: ndarray, cov_xx: ndarray, cov_yy: ndarray, mean_x: ndarray, mean_y: ndarray, lam: float = 1e-2, niter: int = 8) -> Tuple[ndarray, ndarray]:
    """
    Extended Procrustes analysis to solve for affine transformation `A` and translation `t` such that `y_i ~= A x_i + t`.

    Parameters
    ----
    - `cov_yx`: (..., 3, 3) covariance matrix between y
    - `cov_xx`: (..., 3, 3) covariance matrix of x points.
    - `cov_yy`: (..., 3, 3) covariance matrix of y
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
    dtype = cov_yx.dtype
    R = kabasch(cov_yx)
    tr_xx = np.maximum(np.trace(cov_xx, axis1=-2, axis2=-1), np.finfo(dtype).tiny)
    tr_yy = np.maximum(np.trace(cov_yy, axis1=-2, axis2=-1), np.finfo(dtype).tiny)
    
    cov_yx, cov_xy = cov_yx / tr_xx[..., None, None], cov_yx.swapaxes(-2, -1) / tr_yy[..., None, None]
    cov_xx, cov_yy = cov_xx / tr_xx[..., None, None], cov_yy / tr_yy[..., None, None]
    
    A, B = np.zeros_like(R), np.zeros_like(R)
    I = np.eye(cov_yx.shape[-1], dtype=dtype)
    
    def _step(A, B, R, cov_yx, cov_xy, cov_xx, cov_yy, lam, gamma):
        A = (cov_yx + lam * R + gamma * B.swapaxes(-2, -1)) @ safe_inv(cov_xx + lam * I + gamma * (B @ B.swapaxes(-2, -1)))
        B = (cov_xy + lam * R.swapaxes(-2, -1) + gamma * A.swapaxes(-2, -1)) @ safe_inv(cov_yy + lam * I + gamma * (A @ A.swapaxes(-2, -1)))
        err = np.square(A @ B - I).mean(axis=(-2, -1))
        return A, B, err
    
    not_converged = np.argwhere(np.ones(R.shape[:-2], dtype=bool))
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
    p: np.ndarray, 
    q: np.ndarray, 
    w: Optional[np.ndarray] = None, 
    *,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2, 
    niter: int = 5
) -> np.ndarray:
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
    
    Returns
    ----
    - `pose`: (..., 4, 4) transformations matrix from p to q.
    """
    if w is None:
        w = np.ones(p.shape[:-1], dtype=p.dtype)
    w_sum = np.maximum(np.sum(w, axis=-1), np.finfo(p.dtype).tiny)
    p_mean = np.sum(w[..., None] * p, axis=-2) / w_sum[..., None]
    q_mean = np.sum(w[..., None] * q, axis=-2) / w_sum[..., None]
    p = p - p_mean[..., None, :]
    q = q - q_mean[..., None, :]
    pw = p * w[..., None]
    qw = q * w[..., None]
    cov_qp = np.sum(vector_outer(qw, p), axis=-3) / w_sum[..., None, None]
    if mode == 'similar' or mode == 'affine':
        cov_pp = np.sum(vector_outer(pw, p), axis=-3) / w_sum[..., None, None]
        cov_qq = np.sum(vector_outer(qw, q), axis=-3) / w_sum[..., None, None]
    
    if mode == 'rigid':
        _, R, t = umeyama(cov_qp, mean_x=p_mean, mean_y=q_mean)
        pose = make_affine_matrix(R, t)
    elif mode == 'similar':
        s, R, t = umeyama(cov_qp, cov_xx=cov_pp, cov_yy=cov_qq, mean_x=p_mean, mean_y=q_mean)
        pose = make_affine_matrix(s * R, t)
    elif mode == 'affine':
        A, t = affine_umeyama(cov_qp, cov_pp, cov_qq, p_mean, q_mean, lam=lam, niter=niter)
        pose = make_affine_matrix(A, t)
    
    return pose


def segment_solve_pose(
    p: np.ndarray, 
    q: np.ndarray, 
    w: Optional[np.ndarray] = None, 
    *,
    offsets: np.ndarray, 
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2, 
    niter: int = 5
) -> np.ndarray:
    """Solve for the pose (transformation from p to q) given weighted point correspondences.
    
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
    
    Returns
    ----
    - `pose`: (S, 4, 4) transformations matrix from p to q.
    """
    if w is None:
        w = np.ones(p.shape[:-1], dtype=p.dtype)

    lengths = np.diff(offsets)
    w_sum = np.maximum(np.add.reduceat(w, offsets[:-1], axis=0), np.finfo(p.dtype).tiny)
    p_mean = np.add.reduceat(w[..., None] * p, offsets[:-1], axis=0) / w_sum[:, None]
    q_mean = np.add.reduceat(w[..., None] * q, offsets[:-1], axis=0) / w_sum[:, None]
    p = p - np.repeat(p_mean, lengths, axis=0)
    q = q - np.repeat(q_mean, lengths, axis=0)
    pw = p * w[..., None]
    qw = q * w[..., None]
    cov_qp = np.add.reduceat(vector_outer(qw, p), offsets[:-1], axis=0) / w_sum[:, None, None]
    if mode == 'similar' or mode == 'affine':
        cov_pp = np.add.reduceat(vector_outer(pw, p), offsets[:-1], axis=0) / w_sum[:, None, None]    
        cov_qq = np.add.reduceat(vector_outer(qw, q), offsets[:-1], axis=0) / w_sum[:, None, None]
    
    if mode == 'rigid':
        _, R, t = umeyama(cov_qp, mean_x=p_mean, mean_y=q_mean)
        pose = make_affine_matrix(R, t)
    elif mode == 'similar':
        s, R, t = umeyama(cov_qp, cov_xx=cov_pp, cov_yy=cov_qq, mean_x=p_mean, mean_y=q_mean)
        pose = make_affine_matrix(s * R, t)
    elif mode == 'affine':
        A, t = affine_umeyama(cov_qp, cov_pp, cov_qq, p_mean, q_mean, lam=lam, niter=niter)
        pose = make_affine_matrix(A, t)
    
    return pose


def solve_poses_sequential(
    trajectories: ndarray,
    weights: Optional[ndarray] = None,
    *,
    accum: Optional[Tuple[ndarray, ...]] = None,
    min_valid_size: int = 3,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2,
    niter: int = 8
) -> Tuple[ndarray, Tuple[ndarray, ...], Tuple[ndarray, ndarray, ndarray, ndarray]]:
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

    Returns
    ----
    - `poses`: (T, ..., 4, 4) transformations from canonical to each frame.
    - `valid`: (T, ...) boolean mask indicating valid segments
    - `stats`: canonical statistics of each group,
        It is a tuple of:
        - `mu`: (..., 3) weighted mean of points
        - `cov`: (..., 3, 3) weighted covariance of points
        - `tot_w`: (...,) total weight of points
        - `nnz`: (...,) number of non-zero weight points
    - `canonical_points`: (..., N, 3) canonical points.
    - `err`: (..., N,) per-point RMS error over all time := sqrt(sum_over_time(per_point_weights * per_point_squared_error) / per_point_nnz)
        Use this to filter outliers as needed.
    - `accum`: per point accumulated statistics. Just pass it to the next call for incremental solving.
        It is a tuple of:
        - `accum_sqrtw`: (..., N,) sum of sqrt(weights)
        - `accum_sqrtwx`: (..., N, 3) sum of sqrt(weights) * x
        - `accum_sqrtwxx`: (...N, 3, 3) sum of sqrt(weights) * outer(x - mean_sqrtwx, x - mean_sqrtwx)
        - `accum_w`: (..., N,) sum of weights
        - `accum_wx`: (..., N, 3) sum of weights * x
        - `accum_wxx`: (..., N, 3, 3) sum of weights * outer(x - mean_wx, x - mean_wx)
        - `accum_nnz`: (..., N,) number of non-zero weight accumulations

    Example
    ----
    ```
    accum = None
    poses, valid = [], []
    for new_trajectories_chunk in data_stream:
        # new_trajectories_chunk: (T_chunk, N, 3)
        poses_chunk, valid_chunk, stats, canonical_points, err, accum = solve_poses(
            new_trajectories_chunk,
            accum=accum,
        )
        poses.append(poses_chunk)
        valid.append(valid_chunk)
        # `stats`, `canonical_points` and `err` are returned and updated every chunk.
    poses = np.concatenate(poses, axis=0)   # (T_all, 4, 4), poses over all frames
    valid = np.concatenate(valid, axis=0)   # (T_all,), poses' validity over all frames
    """
    dtype = trajectories.dtype
    num_frames = trajectories.shape[0]
    num_points = trajectories.shape[-2]
    batch_shape = trajectories.shape[1:-2]

    if weights is None:
        weights = np.ones((num_frames, *batch_shape, num_points), dtype=dtype)

    poses = np.zeros((num_frames, *batch_shape, 4, 4), dtype=dtype)
        
    if accum is not None:
        accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz = [a.copy() for a in accum]
    else:
        accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz = \
            np.zeros((*batch_shape, num_points,), dtype=dtype), \
            np.zeros((*batch_shape, num_points, 3), dtype=dtype), \
            np.zeros((*batch_shape, num_points, 3, 3), dtype=dtype), \
            np.zeros((*batch_shape, num_points,), dtype=dtype), \
            np.zeros((*batch_shape, num_points, 3), dtype=dtype), \
            np.zeros((*batch_shape, num_points, 3, 3), dtype=dtype), \
            np.zeros((*batch_shape, num_points,), dtype=dtype)
    
    for i in range(num_frames):
        # Compute weighted statistics
        mean_sqrtwx = accum_sqrtwx / np.maximum(accum_sqrtw, np.finfo(trajectories.dtype).tiny)[..., None]
        wi, yi = weights[i], trajectories[i]
        sqrtwi = np.sqrt(wi)
        w = sqrtwi * accum_sqrtw
        sum_w = np.sum(w, axis=-1) + np.finfo(dtype).tiny
        center_x = np.sum(sqrtwi[..., None] * accum_sqrtwx, axis=-2) / sum_w[..., None]
        center_y = np.sum(w[..., None] * yi, axis=-2) / sum_w[..., None]
        xc = mean_sqrtwx - center_x[..., None, :]
        yc = yi - center_y[..., None, :]
        cov_yx = np.einsum('...i,...ij,...ik->...jk', w, yc, xc) / sum_w[..., None, None]
        if mode == 'affine' or mode == 'similar':
            cov_xx = (np.einsum('...i,...ij,...ik->...jk', w, xc, xc) + np.einsum('...i,...ijk->...jk', sqrtwi, accum_sqrtwxx)) / sum_w[..., None, None]
            cov_yy = np.einsum('...i,...ij,...ik->...jk', w, yc, yc) / sum_w[..., None, None]
        
        # Solve for pose
        if mode == 'rigid':
            _, R, t = umeyama(cov_yx, mean_x=center_x, mean_y=center_y)
            poses[i] = make_affine_matrix(R, t)
        elif mode == 'similar':
            s, R, t = umeyama(cov_yx, cov_xx=cov_xx, mean_x=center_x, mean_y=center_y, niter=niter)
            poses[i] = make_affine_matrix(s * R, t)
        elif mode == 'affine':
            A, t = affine_umeyama(cov_yx, cov_xx, cov_yy, center_x, center_y, lam=lam, niter=niter)
            poses[i] = make_affine_matrix(A, t)

        xi = transform_points(yi, safe_inv(poses[i])[..., None, :, :])

        # Update accum
        old_mean_sqrtwx, old_accum_sqrtw = mean_sqrtwx.copy(), accum_sqrtw.copy()
        accum_sqrtw += sqrtwi
        accum_sqrtwx += sqrtwi[..., None] * xi
        mean_sqrtwx = accum_sqrtwx / np.maximum(accum_sqrtw, np.finfo(dtype).tiny)[..., None]
        accum_sqrtwxx += old_accum_sqrtw[..., None, None] * vector_outer(mean_sqrtwx - old_mean_sqrtwx) + sqrtwi[..., None, None] * vector_outer(xi - mean_sqrtwx)

        mean_wx = accum_wx / np.maximum(accum_w, np.finfo(dtype).tiny)[..., None]
        old_mean_wx, old_accum_w = mean_wx.copy(), accum_w.copy()
        accum_w += wi
        accum_wx += wi[..., None] * xi
        mean_wx = accum_wx / np.maximum(accum_w, np.finfo(dtype).tiny)[..., None]
        accum_wxx += old_accum_w[..., None, None] * vector_outer(mean_wx - old_mean_wx) + wi[..., None, None] * vector_outer(xi - mean_wx)
        accum_nnz += wi > 0

    tot_w = np.sum(accum_w, axis=-1)
    mu = np.sum(accum_wx, axis=-2) / np.maximum(tot_w, np.finfo(dtype).tiny)[..., None]
    mean_wx = accum_wx / np.maximum(accum_w, np.finfo(dtype).tiny)[..., None]
    sigma = np.sum(accum_wxx + accum_w[..., None, None] * vector_outer(mu[..., None, :] - mean_wx), axis=-3) / np.maximum(tot_w, np.finfo(dtype).tiny)[..., None, None]
    nnz = np.sum(accum_nnz, axis=-1)
    valid = np.sum(weights > 0, axis=-1) >= min_valid_size
    err = np.sqrt(np.trace(accum_wxx, axis1=-2, axis2=-1) / np.maximum(accum_nnz, np.finfo(dtype).tiny))

    return poses, valid, (mu, sigma, tot_w, nnz), mean_wx, err, (accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz)


def segment_solve_poses_sequential(
    trajectories: ndarray,
    weights: Optional[ndarray] = None,
    offsets: ndarray = None,
    *,
    accum: Optional[Tuple[ndarray, ...]] = None,
    min_valid_size: int = 3,
    mode: Literal['rigid', 'similar', 'affine'] = 'rigid',
    lam: float = 1e-2,
    niter: int = 8
) -> Tuple[ndarray, Tuple[ndarray, ...], Tuple[ndarray, ndarray, ndarray, ndarray]]:
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
        - For 'rigid', only rotation and translation are allowed.
        - For 'similar', uniform scaling, rotation and translation are allowed. 
        - For 'affine', full affine transformation is allowed. Using least squares.
    - `lam`: rigidity regularization weight for 'affine' mode.
    - `niter`: number of iterations for 'affine' mode.

    Returns
    ----
    - `poses`: (T, S, 4, 4) transformations from canonical to each frame.
    - `valid`: (T, S) boolean mask indicating valid segments
    - `stats`: canonical statistics of each group,
        It is a tuple of:
        - `mu`: (S, 3) weighted mean of points
        - `cov`: (S, 3, 3) weighted covariance of points
        - `tot_w`: (S,) total weight of points
        - `nnz`: (S,) number of non-zero weight points
    - `canonical_points`: (N, 3) canonical points.
    - `err`: (N,) per-point RMS error over all time := sqrt(sum_over_time(per_point_weights * per_point_squared_error) / per_point_nnz)
        Use this to filter outliers as needed.
    - `accum`: per point accumulated statistics. Just pass it to the next call for incremental solving.
        It is a tuple of:
        - `accum_sqrtw`: (N,) sum of sqrt(weights)
        - `accum_sqrtwx`: (N, 3) sum of sqrt(weights) * x
        - `accum_sqrtwxx`: (N, 3, 3) sum of sqrt(weights) * outer(x - mean_sqrtwx, x - mean_sqrtwx)
        - `accum_w`: (N,) sum of weights
        - `accum_wx`: (N, 3) sum of weights * x
        - `accum_wxx`: (N, 3, 3) sum of weights * outer(x - mean_wx, x - mean_wx)
        - `accum_nnz`: (N,) number of non-zero weight accumulations
    """
    dtype = trajectories.dtype
    num_frames = trajectories.shape[0]
    num_points = trajectories.shape[1]

    if weights is None:
        weights = np.ones((num_frames, num_points), dtype=dtype)

    num_segments = len(offsets) - 1
    lengths = np.diff(offsets)
    poses = np.zeros((num_frames, num_segments, 4, 4), dtype=dtype)
        
    if accum is not None:
        accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz = [a.copy() for a in accum]
    else:
        accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz = \
            np.zeros((num_points,), dtype=dtype), \
            np.zeros((num_points, 3), dtype=dtype), \
            np.zeros((num_points, 3, 3), dtype=dtype), \
            np.zeros((num_points,), dtype=dtype), \
            np.zeros((num_points, 3), dtype=dtype), \
            np.zeros((num_points, 3, 3), dtype=dtype), \
            np.zeros((num_points,), dtype=dtype)
    
    for i in range(num_frames):
        # Compute weighted statistics
        mean_sqrtwx = accum_sqrtwx / np.maximum(accum_sqrtw, np.finfo(trajectories.dtype).tiny)[..., None]
        wi, yi = weights[i], trajectories[i]
        sqrtwi = np.sqrt(wi)
        w = sqrtwi * accum_sqrtw
        sum_w = np.add.reduceat(w, offsets[:-1], axis=0) + np.finfo(dtype).tiny
        center_x = np.add.reduceat(sqrtwi[:, None] * accum_sqrtwx, offsets[:-1], axis=0) / sum_w[:, None]
        center_y = np.add.reduceat(w[:, None] * yi, offsets[:-1], axis=0) / sum_w[:, None]
        center_x_broadcast = np.repeat(center_x, lengths, axis=0)
        center_y_broadcast = np.repeat(center_y, lengths, axis=0)
        xc = mean_sqrtwx - center_x_broadcast
        yc = yi - center_y_broadcast
        cov_yx = np.add.reduceat(w[:, None, None] * vector_outer(yc, xc), offsets[:-1], axis=0) / sum_w[:, None, None]
        if mode == 'affine' or mode == 'similar':
            cov_xx = np.add.reduceat(sqrtwi[:, None, None] * accum_sqrtwxx + w[:, None, None] * vector_outer(xc), offsets[:-1], axis=0) / sum_w[:, None, None]
            cov_yy = np.add.reduceat(w[:, None, None] * vector_outer(yc), offsets[:-1], axis=0) / sum_w[:, None, None]
        
        # Solve for pose
        if mode == 'rigid':
            _, R, t = umeyama(cov_yx, mean_x=center_x, mean_y=center_y)
            poses[i] = make_affine_matrix(R, t)
        elif mode == 'similar':
            s, R, t = umeyama(cov_yx, cov_xx=cov_xx, mean_x=center_x, mean_y=center_y, niter=niter)
            poses[i] = make_affine_matrix(s * R, t)
        elif mode == 'affine':
            A, t = affine_umeyama(cov_yx, cov_xx, cov_yy, center_x, center_y, lam=lam, niter=niter)
            poses[i] = make_affine_matrix(A, t)

        xi = transform_points(yi, np.repeat(safe_inv(poses[i]), lengths, axis=0))

        # Update accum
        old_mean_sqrtwx, old_accum_sqrtw = mean_sqrtwx.copy(), accum_sqrtw.copy()
        accum_sqrtw += sqrtwi
        accum_sqrtwx += sqrtwi[..., None] * xi
        mean_sqrtwx = accum_sqrtwx / np.maximum(accum_sqrtw, np.finfo(dtype).tiny)[..., None]
        accum_sqrtwxx += old_accum_sqrtw[..., None, None] * vector_outer(mean_sqrtwx - old_mean_sqrtwx) + sqrtwi[..., None, None] * vector_outer(xi - mean_sqrtwx)

        mean_wx = accum_wx / np.maximum(accum_w, np.finfo(dtype).tiny)[..., None]
        old_mean_wx, old_accum_w = mean_wx.copy(), accum_w.copy()
        accum_w += wi
        accum_wx += wi[..., None] * xi
        mean_wx = accum_wx / np.maximum(accum_w, np.finfo(dtype).tiny)[..., None]
        accum_wxx += old_accum_w[..., None, None] * vector_outer(mean_wx - old_mean_wx) + wi[..., None, None] * vector_outer(xi - mean_wx)
        accum_nnz += wi > 0

    tot_w = np.add.reduceat(accum_w, offsets[:-1], axis=0)
    mu = np.add.reduceat(accum_wx, offsets[:-1], axis=0) / np.maximum(tot_w, np.finfo(dtype).tiny)[:, None]
    mean_wx = accum_wx / np.maximum(accum_w, np.finfo(dtype).tiny)[:, None]
    mu_broadcast = np.repeat(mu, lengths, axis=0)
    sigma = np.add.reduceat(accum_wxx + accum_w[:, None, None] * vector_outer(mu_broadcast - mean_wx), offsets[:-1], axis=0) / np.maximum(tot_w, np.finfo(dtype).tiny)[:, None, None]
    nnz = np.add.reduceat(accum_nnz, offsets[:-1], axis=0)
    valid = np.add.reduceat(weights > 0, offsets[:-1], axis=1) >= min_valid_size
    err = np.sqrt(np.trace(accum_wxx, axis1=-2, axis2=-1) / np.maximum(accum_nnz, np.finfo(dtype).tiny))

    return poses, valid, (mu, sigma, tot_w, nnz), mean_wx, err, (accum_sqrtw, accum_sqrtwx, accum_sqrtwxx, accum_w, accum_wx, accum_wxx, accum_nnz)