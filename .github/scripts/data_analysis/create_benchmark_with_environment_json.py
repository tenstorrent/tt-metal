from infra.data_collection.github.utils import (
    get_github_partial_benchmark_json_filenames,
    create_json_with_github_benchmark_environment,
)

if __name__ == "__main__":
    github_partial_benchmark_json_filenames = get_github_partial_benchmark_json_filenames()

    for json_filename in github_partial_benchmark_json_filenames:
        create_json_with_github_benchmark_environment(json_filename)
