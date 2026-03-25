from models.common.utility_functions import comp_pcc
from tests.ttnn.utils_for_testing import assert_allclose, assert_relative_frobenius


def assert_numeric_metrics(
    expected,
    actual,
    rtol=None,
    atol=None,
    frobenius_threshold=None,
    pcc_threshold=None,
    check_allclose=True,
    check_frobenius=True,
    check_pcc=True,
):
    if check_allclose:
        allclose_kwargs = {}
        if rtol is not None:
            allclose_kwargs["rtol"] = rtol
        if atol is not None:
            allclose_kwargs["atol"] = atol
        assert_allclose(expected, actual, **allclose_kwargs)

    if check_frobenius:
        frobenius_kwargs = {}
        if frobenius_threshold is not None:
            frobenius_kwargs["threshold"] = frobenius_threshold
        assert_relative_frobenius(expected, actual, **frobenius_kwargs)

    if check_pcc:
        threshold = 0.98 if pcc_threshold is None else pcc_threshold
        passing_pcc, pcc_message = comp_pcc(expected, actual, threshold)
        assert passing_pcc, pcc_message
