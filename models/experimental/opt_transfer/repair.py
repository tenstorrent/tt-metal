from typing import Callable, Optional

from models.experimental.opt_transfer.schema import Diagnosis


def localize_culprit(applied: list, pcc_with: Callable, threshold: float) -> Optional[str]:
    """Return the single fusion whose removal restores PCC>=threshold, else None."""
    if pcc_with(set()) >= threshold:
        return None
    for f in applied:
        if pcc_with({f}) >= threshold:
            return f
    return None


def build_diagnosis(
    node,
    per_block_pcc,
    tf_pcc,
    free_run_divergence_frac,
    config_tried,
    drift_min_frac=0.9,
) -> Diagnosis:
    if (
        tf_pcc is not None
        and tf_pcc >= 0.99
        and free_run_divergence_frac is not None
        and free_run_divergence_frac < drift_min_frac
    ):
        return Diagnosis(
            node=node,
            axis="long_decode_drift",
            measured=free_run_divergence_frac,
            config_tried=config_tried,
        )
    return Diagnosis(
        node=node,
        axis="per_block_pcc",
        measured=per_block_pcc,
        config_tried=config_tried,
    )
