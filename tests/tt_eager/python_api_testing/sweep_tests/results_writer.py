# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import xlsxwriter
import csv
import ast
import os
import datetime
from pathlib import Path
from loguru import logger


def write_csv_to_xlsx(args):
    input_csv_file = Path(args.input_csv_file)
    assert input_csv_file.exists()

    if args.output_xlsx_file:
        output_xlsx_file = Path(args.output_xlsx_file)
    else:
        output_xlsx_file = input_csv_file.with_suffix("")
        output_xlsx_file = output_xlsx_file.with_suffix(".xlsx")

    if not args.no_timestamp:
        time = os.path.getmtime(input_csv_file)
        dt_prefix = datetime.datetime.fromtimestamp(time, tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M")
        output_xlsx_file = output_xlsx_file.with_name(f"{dt_prefix}_{output_xlsx_file.name}")

    logger.info(f"Writing {input_csv_file} to {output_xlsx_file}.")

    workbook = xlsxwriter.Workbook(output_xlsx_file)
    worksheet = workbook.add_worksheet()

    # Formatters for writing to cell
    PERCENT_FORMAT = workbook.add_format({"num_format": "0.00%"})

    with open(input_csv_file, "r") as f:
        csv_reader = csv.reader(f)

        # Keep track of maximum number of shapes and dims
        csv_rows = []
        shape_max, dim_max = 0, 0
        for row_idx, row in enumerate(csv_reader):
            csv_rows.append(row)

            if row_idx > 0:
                # Shapes column
                shape_col = row[1]
                shapes = ast.literal_eval(shape_col)  # Assume shape is a list of lists
                shape_count = len(shapes)
                dim_count = len(shapes[0])
                shape_max = max(shape_max, shape_count)
                dim_max = max(dim_max, dim_count)

        for row_idx, row in enumerate(csv_rows):
            col_idx = 0

            if row_idx == 0:
                for col in row:
                    # Shapes column
                    if col_idx == 1:
                        worksheet.write(row_idx, col_idx, col)
                        col_idx += 1

                        for shape_idx in range(shape_max):
                            for dim_idx in range(dim_max):
                                worksheet.write(
                                    row_idx,
                                    col_idx,
                                    f"shape{shape_idx+1}_dim{dim_idx+1}",
                                )
                                col_idx += 1
                    else:
                        worksheet.write(row_idx, col_idx, col)
                        col_idx += 1

                # HACK for matmul perf
                if args.add_matmul_perf:
                    worksheet.write(row_idx, col_idx, "math_util (%)")
                    col_idx += 1

            else:
                for col in row:
                    # Shapes column
                    if col_idx == 1:
                        worksheet.write(row_idx, col_idx, col)
                        col_idx += 1

                        # Write individual dims of each shape as separate columns
                        shapes = ast.literal_eval(col)  # Assume shape is a list of lists
                        for shape_count, shape in enumerate(shapes):
                            for dim in shape:
                                worksheet.write(row_idx, col_idx, dim)
                                col_idx += 1
                        # Skip columns if it has less shapes than max
                        col_idx += (shape_max - shape_count - 1) * (dim_max)

                    else:
                        worksheet.write(row_idx, col_idx, int(col) if col.isdigit() else col)
                        col_idx += 1

                # HACK for matmul perf
                if args.add_matmul_perf:
                    M, K, N = shapes[0][2], shapes[0][3], shapes[1][3]
                    num_cores_used = int(row[8])
                    kernel_runtime = int(row[12])

                    total_ops = 2 * M * K * N
                    ideal_compute_cycles = total_ops / (512 * num_cores_used)  # HiFi4: 512 OPs/CC/Tensix core
                    total_cycles = kernel_runtime * 1.2  # ns * 1.2GHz clock
                    math_util = ideal_compute_cycles / total_cycles
                    worksheet.write(row_idx, col_idx, math_util, PERCENT_FORMAT)
                    col_idx += 1

    workbook.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick script to convert csv to xlsx")
    parser.add_argument(
        "-i",
        "--input-csv-file",
        help="Input csv file",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output-xlsx-file",
        help="Output xlsx file",
    )
    parser.add_argument(
        "-nt",
        "--no-timestamp",
        help="Don't add timestamp to output file name",
        action="store_true",
    )
    parser.add_argument(
        "--add-matmul-perf",
        help="Add math utilization for matmul to last column",
        action="store_true",
    )

    args = parser.parse_args()

    write_csv_to_xlsx(args)
