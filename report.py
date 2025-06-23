import sys
import pandas as pd
from async_perf_csv import perf_report
from tabulate import tabulate

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <csv_file_path>", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]

    try:
        # Generate the report and convert it to a DataFrame
        average_df = perf_report(csv_path)

        # Print the DataFrame in a pretty table format
        print("Min - Avg - Max by Common Runs:")
        print(tabulate(average_df, headers="keys", tablefmt="pretty"))

    except Exception as e:
        print(f"Error in performance report generation: {e}", file=sys.stderr)
        sys.exit(1)
