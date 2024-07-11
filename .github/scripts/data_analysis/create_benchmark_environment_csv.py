from infra.data_collection.github.utils import (
    get_github_benchmark_environment_csv_filename,
    create_csv_for_github_benchmark_environment,
)

if __name__ == "__main__":
    github_benchmark_environment_csv_filename = get_github_benchmark_environment_csv_filename()
    create_csv_for_github_benchmark_environment(github_benchmark_environment_csv_filename)
