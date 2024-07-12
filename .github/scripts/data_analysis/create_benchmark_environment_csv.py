from infra.data_collection.github.utils import (
    get_github_benchmark_environment_csv_filenames,
    create_csv_for_github_benchmark_environment,
)

if __name__ == "__main__":
    github_benchmark_environment_csv_filenames = get_github_benchmark_environment_csv_filenames()

    for csv_filename in github_benchmark_environment_csv_filenames:
        create_csv_for_github_benchmark_environment(csv_filename)
