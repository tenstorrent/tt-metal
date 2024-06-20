from infra.data_collection.github.utils import create_csvs_for_data_analysis

if __name__ == "__main__":
    github_context_json_filename = "github_context.json"
    github_pipeline_json_filename = "workflow.json"
    github_jobs_json_filename = "workflow_jobs.json"

    create_csvs_for_data_analysis(
        github_context_json_filename,
        github_pipeline_json_filename,
        github_jobs_json_filename,
    )
