import pytest


@pytest.fixture(scope="session")
def workflow_run_gh_environment():
    yield {
        "github_event_name": "workflow_run",
    }
