from infra.data_collection.github.utils import (
    get_github_partial_benchmark_data_filenames,
    create_json_with_github_benchmark_environment,
)

if __name__ == "__main__":
    github_partial_benchmark_data_filenames = get_github_partial_benchmark_data_filenames()

    for benchmark_data_filename in github_partial_benchmark_data_filenames:
        create_json_with_github_benchmark_environment(benchmark_data_filename)
