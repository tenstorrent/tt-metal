from runner_failure_common import matching_signature_labels


def test_setup_runner_failure_signature_matches_start_hook_exit() -> None:
    log_text = """
2026-08-11T19:44:53.4057888Z A job started hook has been configured by the self-hosted runner administrator
2026-08-11T19:44:53.4201322Z ##[group]Run '/opt/tt_metal_infra/scripts/ci/wormhole_b0/reset.sh'
2026-08-11T19:45:05.5053575Z Setting up MLPerf mount...
2026-08-11T19:45:06.0872135Z ##[error]Process completed with exit code 1.
2026-08-11T19:45:06.1670277Z A job completed hook has been configured by the self-hosted runner administrator
"""

    assert "Set up runner failure" in matching_signature_labels(log_text)


def test_setup_runner_failure_signature_ignores_later_step_exit() -> None:
    log_text = """
2026-08-11T19:44:53.4057888Z A job started hook has been configured by the self-hosted runner administrator
2026-08-11T19:45:05.5052769Z tt-smi completed successfully.
2026-08-11T19:45:06.1670277Z A job completed hook has been configured by the self-hosted runner administrator
2026-08-11T19:45:10.0000000Z ##[group]Run pytest tests
2026-08-11T19:45:11.0000000Z ##[error]Process completed with exit code 1.
"""

    assert "Set up runner failure" not in matching_signature_labels(log_text)
