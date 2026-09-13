from agent.hang_probe import classify_trial

_S = "TE_LOOP_DONE"


def test_pass_requires_sentinel_and_clean_exit():
    assert classify_trial("ran\nTE_LOOP_DONE\n", 0, _S) == "pass"


def test_killed_run_is_never_pass_even_with_sentinel():
    assert classify_trial("ran\nTE_LOOP_DONE\n", 124, _S) == "hang"


def test_timeout_and_sigkill_are_hang():
    assert classify_trial("partial\n", 124, _S) == "hang"
    assert classify_trial("partial\n", -9, _S) == "hang"


def test_clean_exit_without_sentinel_is_fail_not_pass():
    assert classify_trial("collected 1 item\nSKIPPED\n", 0, _S) == "fail"


def test_error_exit_is_fail():
    assert classify_trial("Traceback\nRuntimeError\n", 1, _S) == "fail"
