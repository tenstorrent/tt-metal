import csv
import click


def get_column_from_csv(file_path, column_name):
    with open(file_path, mode="r") as file:
        csv_reader = csv.DictReader(file)
        column_data = [row[column_name] for row in csv_reader]
    return column_data


def convert_data_to_numbers(data):
    return [int(x) for x in data if x]


def sum_list(numbers):
    return sum(numbers)


def convert_ns_to_seconds(nanoseconds):
    return nanoseconds / 1_000_000_000


def convert_to_samples_per_sec(seconds, samples):
    return samples / seconds


def extract_elements_by_param(data, row_name, param):
    return {key: [value[i] for i in range(len(value)) if data[row_name][i] == param] for key, value in data.items()}


# def store_data_to_csv(data, file_path):
#     with open(file_path, mode='w', newline='') as file:
#         # print(data)
#         # print(data.keys())
#         writer = csv.DictWriter(file, fieldnames=data.keys())
#         writer.writeheader()
#         writer.writerows([dict(zip(data, t)) for t in zip(*data.values())])
#     print(f"Data stored to {file_path}", file_path)


# function for calculation of samples per second for measured kernel time
def calculate_dev_kernel_duration(file_path, column_name, batch=16):
    column_data = get_column_from_csv(file_path, column_name)
    column_data = convert_data_to_numbers(column_data)
    s = sum_list(column_data)
    seconds = convert_ns_to_seconds(s)
    samples_per_second = convert_to_samples_per_sec(seconds, batch)
    return samples_per_second


def get_multiple_columns_from_csv(file_path, column_names):
    with open(file_path, mode="r") as file:
        csv_reader = csv.DictReader(file)
        columns_data = {column: [] for column in column_names}
        for row in csv_reader:
            for column in column_names:
                columns_data[column].append(row[column])
    return columns_data


def calculate_max_sum(data, input_columns):
    ret_val = int(0)
    for i in range(len(data[input_columns[0]])):
        ret_val += max(int(data[column][i]) for column in input_columns)
    return ret_val


@click.command()
@click.option("-f", "--filepath", type=str, required=True, help="Path to the CSV file")
@click.option("-b", "--batch", type=int, default=16, help="Batch size")
@click.option("-ee", "--e2e_ratio", type=float, default=0.85, help="E2E ratio")
def main(filepath, batch, e2e_ratio):
    path = filepath
    samples_per_second = calculate_dev_kernel_duration(path, "DEVICE KERNEL DURATION [ns]", batch)
    print(f"Kernel device duration {samples_per_second} [samples/s]")

    columns = ["OP CODE", "DEVICE KERNEL DURATION [ns]", "PM IDEAL [ns]", "PM COMPUTE [ns]", "PM BANDWIDTH [ns]"]
    data_with_estimations = get_multiple_columns_from_csv(path, columns)

    matmul_data = extract_elements_by_param(data_with_estimations, "OP CODE", "Matmul")
    conv_data = extract_elements_by_param(data_with_estimations, "OP CODE", "OptimizedConvNew")

    input_columns = ["PM IDEAL [ns]", "PM COMPUTE [ns]", "PM BANDWIDTH [ns]"]
    matmul_ideal_execution_time = calculate_max_sum(matmul_data, input_columns)
    print(f"Matmul ideal time: {matmul_ideal_execution_time} [ns]. ")
    conv_ideal_execution_time = calculate_max_sum(conv_data, input_columns)
    print(f"Conv ideal time: {conv_ideal_execution_time} [ns].")

    cs_model = (
        (batch / (convert_ns_to_seconds(matmul_ideal_execution_time / 0.4 + conv_ideal_execution_time / 0.3)))
        * e2e_ratio
        / 2
    )
    cc_model = cs_model / 2
    print(f"CS model: {cs_model:.2f} [samples/s]")
    print(f"CC model: {cc_model:.2f} [samples/s]")


if __name__ == "__main__":
    main()
