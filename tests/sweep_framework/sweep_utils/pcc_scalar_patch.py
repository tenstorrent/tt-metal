# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Sweep-local fix: decide a SINGLE-ELEMENT comparison by relative tolerance, not by PCC.

Pearson correlation is undefined for a one-element result -- zero variance makes the
denominator 0, so it comes out NaN. `comp_pcc` then falls back to `torch.allclose` with a fixed
rtol=1e-5 that has no relationship to the `pcc` the caller asked for. A caller requesting
pcc=0.999 wants ~0.1% agreement; 1e-5 demands 0.001%, tighter than the op's own arithmetic
delivers, so a correct result is reported as PCC exactly 0.0.

Observed: model_traced `max` with no `dim` (global reduce to a scalar) returned 3.334717 against
a torch golden of 3.334923 -- 6.2e-5 relative error, better than bfloat16 precision and well
inside pcc=0.999. allclose's 1e-4 + 1e-5*|b| = 1.33e-4 budget rejected the 2.06e-4 difference
and returned float(False) == 0.0. Repro: repro_pcc_scalar_fallback.py.

Deliberately implemented HERE rather than in models/common/utility_functions.py: comp_pcc has
~739 callers across tests/ and models/, and only 6 pass rtol explicitly, so editing it changes
the verdict rule for hundreds of unrelated model tests to fix a handful of sweep vectors. This
patch is installed only when the sweep framework is imported, so its scope is the sweeps.

Scoped to numel == 1, where correlation is MATHEMATICALLY undefined rather than merely
degenerate. A constant tensor with many elements keeps the strict path: many samples agreeing is
real evidence, and loosening there could hide a regression. Note also that `comp_pcc`'s
"one tensor is all zero" branch stays untouched -- an all-zeros result where the golden is not
zero is a genuine failure, not a tolerance question.

NOTE: this makes a scalar comparison tolerance-based, which means it accepts the op's own
precision error. For `ttnn.max` on float32 that error is real and arguably a bug -- the returned
value is not present in the input at all (see issue #51889). This patch stops the misleading
"PCC 0.0" report; it does not certify the op. If #51889 is confirmed a defect, these vectors
should fail again, with a message about selection semantics rather than a fabricated 0.0.
"""

import torch

from loguru import logger

# Ceiling on the derived tolerance. A caller asking pcc=0.5 wants little correlation, but 50%
# relative error on a lone value is not a meaningful check.
MAX_SCALAR_RTOL = 1e-2

_installed = False
_orig_comp_pcc = None


def scalar_rtol(pcc, rtol):
    """Relative tolerance for a single-element comparison, derived from the requested pcc.

    (1 - pcc) is the natural scalar analogue of a correlation threshold. max() with the caller's
    rtol means this can only LOOSEN, never tighten, so nothing that passes today starts failing.
    """
    try:
        derived = min(1.0 - float(pcc), MAX_SCALAR_RTOL)
    except (TypeError, ValueError):
        return rtol
    return max(rtol, derived) if derived > 0 else rtol


def comp_pcc_scalar_aware(golden, calculated, pcc=0.99, rtol=1e-05, atol=1e-04):
    """comp_pcc, but a one-element result is judged by tolerance instead of correlation.

    Anything with more than one element is delegated to the original untouched.
    """
    try:
        g = torch.as_tensor(golden)
        c = torch.as_tensor(calculated)
    except Exception:
        return _orig_comp_pcc(golden, calculated, pcc, rtol, atol)

    if g.numel() != 1 or c.numel() != 1:
        return _orig_comp_pcc(golden, calculated, pcc, rtol, atol)

    gf = g.flatten().float()
    cf = c.flatten().float()

    # Preserve the original's NaN semantics rather than letting allclose decide them.
    if torch.isnan(gf).all() and torch.isnan(cf).all():
        return True, 1.0
    if torch.isnan(gf).any() != torch.isnan(cf).any():
        return False, 0.0
    if torch.equal(gf, cf):
        return True, 1.0

    tol = scalar_rtol(pcc, rtol)
    passed = bool(torch.allclose(gf, cf, rtol=tol, atol=atol))
    if not passed:
        logger.warning(
            f"single-element comparison failed: golden={gf.item()} actual={cf.item()} "
            f"(rel err {abs(gf.item() - cf.item()) / max(abs(gf.item()), 1e-30):.2e}, "
            f"rtol={tol:.1e} derived from pcc={pcc})"
        )
    return passed, float(passed)


def install():
    """Point the sweeps' comparison path at comp_pcc_scalar_aware. Idempotent.

    Patches `comp_pcc` in tests.ttnn.utils_for_testing, which is where the sweep modules'
    `check_with_pcc` resolves it from at call time -- so the sweeps pick this up without any
    edit to models/common/utility_functions.py or to the 109 sweep modules.
    """
    global _installed, _orig_comp_pcc
    if _installed:
        return
    try:
        import tests.ttnn.utils_for_testing as uft

        _orig_comp_pcc = uft.comp_pcc
        uft.comp_pcc = comp_pcc_scalar_aware
        _installed = True
    except Exception:
        # Never let a diagnostic patch break a run; the strict behaviour is simply retained.
        logger.warning("could not install the single-element PCC comparison patch", exc_info=True)
