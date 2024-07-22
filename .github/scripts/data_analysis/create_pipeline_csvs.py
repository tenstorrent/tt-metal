from infra.data_collection.github.utils import create_csvs_for_data_analysis, get_github_runner_environment

if __name__ == "__main__":
    github_runner_environment = get_github_runner_environment()
    github_pipeline_json_filename = "workflow.json"
    github_jobs_json_filename = "workflow_jobs.json"

    create_csvs_for_data_analysis(
        github_runner_environment,
        github_pipeline_json_filename,
        github_jobs_json_filename,
    )
