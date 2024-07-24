from infra.data_collection.github.utils import get_github_runner_environment
from infra.data_collection.cicd import create_cicd_json_for_data_analysis

if __name__ == "__main__":
    github_runner_environment = get_github_runner_environment()
    github_pipeline_json_filename = "workflow.json"
    github_jobs_json_filename = "workflow_jobs.json"

    create_cicd_json_for_data_analysis(
        github_runner_environment,
        github_pipeline_json_filename,
        github_jobs_json_filename,
    )
