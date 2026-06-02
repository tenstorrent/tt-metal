from typing import Callable, Optional


def localize_culprit(applied: list, pcc_with: Callable, threshold: float) -> Optional[str]:
    """Return the single fusion whose removal restores PCC>=threshold, else None."""
    if pcc_with(set()) >= threshold:
        return None
    for f in applied:
        if pcc_with({f}) >= threshold:
            return f
    return None
