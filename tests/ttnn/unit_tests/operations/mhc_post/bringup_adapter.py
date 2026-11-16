# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapter: the bring-up op (ttnn.bringup.mhc_post, branch dnijemcevic/model_bringup) behind the
eval's mhc_post signature and registry declarations, so the golden suite and perf_feed_measure.py
can run it unchanged. Used only by GALAXY_VALIDATION.md, Part C: copy it to
ttnn/ttnn/operations/mhc_post/__init__.py in a model_bringup worktree. Not an eval candidate.
The bring-up op feeds only the normalized y (EXCLUSIONS) and needs hc_norm_width (filled from C)."""
import ttnn
from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue


def default_compute_kernel_config():
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
    )


def _tag_alignment(inputs, axes):
    return "tile_aligned" if inputs[1][-2] % 32 == 0 else "h_non_aligned"


def _tag_feed_split(inputs, axes):
    return "block_aligned" if inputs[0][-1] % 128 == 0 else "not_block_aligned"


INPUT_TAGGERS = {"alignment": _tag_alignment, "feed_split": _tag_feed_split}
SUPPORTED = {
    "dtype": [ttnn.float32, ttnn.bfloat16],
    "sublayer_dtype": [ttnn.float32, ttnn.bfloat16],
    "layout": [ttnn.TILE_LAYOUT],
    "fp32_dest_acc_en": [True],
    "mode": ["plain", "stats", "collapse"],
    "coef_source": ["separate", "packed"],
    "alignment": ["tile_aligned", "h_non_aligned"],
    "norm": ["off", "on"],
    "feed": ["off", "on"],
    "feed_split": ["block_aligned", "not_block_aligned"],
}
EXCLUSIONS = [{"feed": "on", "norm": "off"}]  # the fork feeds the normalized y only


def _same(a, b):
    if a is b:
        return True
    try:
        return a.buffer_address() == b.buffer_address()
    except Exception:
        return False


def validate(
    input_tensor,
    residual,
    post,
    comb,
    *,
    compute_kernel_config=None,
    stats_weight=None,
    collapse_pre=None,
    norm_weight=None,
    feed_output=None,
    **_,
):
    cfg = compute_kernel_config or default_compute_kernel_config()
    mode = "collapse" if collapse_pre is not None else ("stats" if stats_weight is not None else "plain")
    axes = {
        "dtype": residual.dtype,
        "sublayer_dtype": input_tensor.dtype,
        "layout": residual.layout,
        "fp32_dest_acc_en": bool(cfg.fp32_dest_acc_en),
        "mode": mode,
        "coef_source": "packed" if _same(post, comb) else "separate",
        "norm": "on" if norm_weight is not None else "off",
        "feed": "on" if feed_output is not None else "off",
    }
    ins = (list(input_tensor.shape), list(residual.shape), list(post.shape), list(comb.shape))
    for k, f in INPUT_TAGGERS.items():
        axes[k] = f(ins, axes)
    for k, allowed in SUPPORTED.items():
        if axes[k] not in allowed:
            raise UnsupportedAxisValue(f"mhc_post: {k}={axes[k]!r}")
    for exc in EXCLUSIONS:
        if all(axes.get(k) == v for k, v in exc.items()):
            raise ExcludedCell(f"mhc_post: {exc}")
    return axes


def mhc_post(
    input_tensor,
    residual,
    post,
    comb,
    *,
    compute_kernel_config=None,
    post_col=0,
    comb_col=0,
    stats_weight=None,
    collapse_pre=None,
    hc_scale=None,
    hc_base=None,
    hc_norm_width=None,
    hc_norm_eps=1e-6,
    hc_eps=1e-6,
    hc_sinkhorn_iters=20,
    output_tensor=None,
    stats_output=None,
    hc_output=None,
    y_output=None,
    max_grid=None,
    norm_weight=None,
    norm_eps=1e-6,
    feed_output=None,
    feed_semaphores=None,
    feed_num_links=1,
    feed_cluster_axis=1,
):
    validate(
        input_tensor,
        residual,
        post,
        comb,
        compute_kernel_config=compute_kernel_config,
        stats_weight=stats_weight,
        collapse_pre=collapse_pre,
        norm_weight=norm_weight,
        feed_output=feed_output,
    )
    if hc_norm_width is None:
        hc_norm_width = float(residual.shape[-1])
    return ttnn.bringup.mhc_post(
        input_tensor,
        residual,
        post,
        comb,
        compute_kernel_config=compute_kernel_config or default_compute_kernel_config(),
        stats_weight=stats_weight,
        post_col=post_col,
        comb_col=comb_col,
        output_tensor=output_tensor,
        stats_output=stats_output,
        max_grid=max_grid,
        collapse_pre=collapse_pre,
        hc_scale=hc_scale,
        hc_base=hc_base,
        hc_norm_width=hc_norm_width,
        hc_norm_eps=hc_norm_eps,
        hc_eps=hc_eps,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        hc_output=hc_output,
        y_output=y_output,
        norm_weight=norm_weight,
        norm_eps=norm_eps,
        feed_output=feed_output,
        feed_semaphores=feed_semaphores,
        feed_num_links=feed_num_links,
        feed_cluster_axis=feed_cluster_axis,
    )
