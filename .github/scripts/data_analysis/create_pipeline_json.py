import pathlib

from infra.data_collection.github.utils import get_github_runner_environment
from infra.data_collection.cicd import create_cicd_json_for_data_analysis, get_cicd_json_filename

if __name__ == "__main__":
    github_runner_environment = get_github_runner_environment()
    github_pipeline_json_filename = "workflow.json"
    github_jobs_json_filename = "workflow_jobs.json"

    workflow_outputs_dir = pathlib.Path("generated/cicd").resolve()
    assert workflow_outputs_dir.is_dir()
    assert workflow_outputs_dir.exists()

    pipeline = create_cicd_json_for_data_analysis(
        workflow_outputs_dir,
        github_runner_environment,
        github_pipeline_json_filename,
        github_jobs_json_filename,
    )

    cicd_json_filename = get_cicd_json_filename(pipeline)

    with open(cicd_json_filename, "w") as f:
        f.write(pipeline.model_dump_json())

    github_pipeline_json = pipeline.github_pipeline_id

    cicd_json_copy_filename = f"pipelinecopy_{github_pipeline_id}.json"
    with open(cicd_json_copy_filename, "w") as f:
        f.write(pipeline.model_dump_json())
