"""
Script to convert job failure cluster data from slack-output-analysis action
to pydantic model and save as JSON for upload to database.

This script:
1. Reads incremental_error_report.json from workspace root (output by slack-output-analysis action)
2. Transforms data to ensure all required fields are populated (handles None values)
3. Creates JobFailureCluster pydantic model instances
4. Serializes to JSON and saves to generated/job_failure_cluster/
"""
import json
import pathlib
from datetime import datetime, timezone
from loguru import logger

from infra.data_collection.pydantic_models import JobFailureCluster


def ensure_required_fields(cluster_data: dict) -> dict:
    """
    Ensure all required fields are populated, using fallbacks for None values.

    Timestamp fields can now be None (Optional[datetime] in pydantic model),
    but we still need to ensure other required fields like job names are populated.
    """
    # Make a copy to avoid modifying the original
    data = cluster_data.copy()

    # Ensure job_name, centroid_job_name, oldest_job_name are never None
    if data.get("job_name") is None:
        job_id = data.get("github_job_id", "unknown")
        data["job_name"] = f"Job-{job_id}"
        logger.warning(f"job_name is None for job {job_id}, using fallback")

    if data.get("centroid_job_name") is None:
        centroid_id = data.get("centroid_job_id", "unknown")
        data["centroid_job_name"] = f"Job-{centroid_id}"
        logger.warning(f"centroid_job_name is None for centroid {centroid_id}, using fallback")

    if data.get("oldest_job_name") is None:
        oldest_id = data.get("oldest_job_id", "unknown")
        data["oldest_job_name"] = f"Job-{oldest_id}"
        logger.warning(f"oldest_job_name is None for oldest {oldest_id}, using fallback")

    # Ensure error messages are never None (use empty string as fallback)
    if data.get("centroid_job_error_message") is None:
        data["centroid_job_error_message"] = data.get("job_error_message", "")
        logger.warning(f"centroid_job_error_message is None, using job_error_message as fallback")

    if data.get("oldest_job_error_message") is None:
        data["oldest_job_error_message"] = data.get("job_error_message", "")
        logger.warning(f"oldest_job_error_message is None, using job_error_message as fallback")

    return data


def find_job_failure_cluster_data():
    """
    Find job failure cluster data file.

    The slack-output-analysis action outputs incremental_error_report.json in the root
    of the workspace.
    """
    data_file = pathlib.Path("incremental_error_report.json")

    if not data_file.exists():
        raise FileNotFoundError(
            f"Could not find incremental_error_report.json in workspace root ({data_file.absolute()}). "
            "Make sure the slack-output-analysis action has run successfully."
        )

    logger.info(f"Found job failure cluster data at: {data_file}")
    return str(data_file)


def create_job_failure_cluster_json():
    """
    Main function to create JSON files from job failure cluster data.
    """
    # Find the input data file
    input_data_path = find_job_failure_cluster_data()

    # Read JSON data
    with open(input_data_path, "r") as f:
        raw_data = json.load(f)

    # Handle both single object and array of objects
    if isinstance(raw_data, list):
        clusters_data = raw_data
    elif isinstance(raw_data, dict):
        # If it's a dict, check if it has a key containing the array
        # Common patterns: "clusters", "data", "results", "failures"
        for key in ["clusters", "data", "results", "failures", "job_failure_clusters"]:
            if key in raw_data and isinstance(raw_data[key], list):
                clusters_data = raw_data[key]
                break
        else:
            # Single object, wrap in list
            clusters_data = [raw_data]
    else:
        raise ValueError(f"Unexpected data format: {type(raw_data)}")

    # Ensure output directory exists
    output_dir = pathlib.Path("generated/job_failure_cluster")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each cluster record
    created_files = []
    for idx, cluster_data in enumerate(clusters_data):
        try:
            # Ensure all required fields are populated (handle None values)
            cluster_data = ensure_required_fields(cluster_data)

            # Create pydantic model instance (validates data)
            cluster = JobFailureCluster(**cluster_data)

            # Generate output filename
            github_job_id = cluster_data.get("github_job_id", f"unknown_{idx}")
            output_filename = output_dir / f"job_failure_cluster_{github_job_id}.json"

            # Serialize to JSON and save
            json_data = cluster.model_dump_json(indent=2)
            with open(output_filename, "w") as f:
                f.write(json_data)

            created_files.append(output_filename)
            logger.info(f"Created JSON file: {output_filename}")

        except Exception as e:
            logger.error(f"Failed to process cluster record {idx}: {e}")
            logger.error(f"Problematic data: {json.dumps(cluster_data, indent=2)}")
            raise

    # Also create a combined file with all records (useful for batch upload)
    if len(clusters_data) > 1:
        combined_file = output_dir / "job_failure_cluster_batch.json"
        clusters_list = []
        for cluster_data in clusters_data:
            # Ensure all required fields are populated
            cluster_data = ensure_required_fields(cluster_data)
            cluster = JobFailureCluster(**cluster_data)
            clusters_list.append(json.loads(cluster.model_dump_json()))

        with open(combined_file, "w") as f:
            json.dump(clusters_list, f, indent=2)
        created_files.append(combined_file)
        logger.info(f"Created combined batch file: {combined_file}")

    logger.info(f"Successfully created {len(created_files)} JSON file(s)")
    return created_files


if __name__ == "__main__":
    create_job_failure_cluster_json()
