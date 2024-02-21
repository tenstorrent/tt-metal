import argparse
import pandas as pd
from tabulate import tabulate

LOG_DEBUG = False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Performance result analyzer tool.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--debug', action='store_true', help='Print debug information')
    parser.add_argument('--output', type=str, default='performance_analysis.csv', help='Output CSV file name')
    parser.add_argument('--top_k_ops', type=int, default=20, help='Number of top longest operations to list')
    return parser.parse_args()

def calculate_data_size(row):
    total_size = 0
    data_type_sizes = {'BFLOAT16': 2, 'BFLOAT8_B': 1}
    
    for i in range(8):  # For each input from 0 to 7
        w = x = y = z = 0
        try:
            w = float(row.get(f'INPUT_{i}_W', 0) or 0)
            x = float(row.get(f'INPUT_{i}_X', 0) or 0)
            y = float(row.get(f'INPUT_{i}_Y', 0) or 0)
            z = float(row.get(f'INPUT_{i}_Z', 0) or 0)
        except ValueError as e:
            if LOG_DEBUG:
                print(f"Warning: Non-numeric data encountered in row {row.name} for input {i}: {e}. \
                  Treating as zeros. w={w}, x={x}, y={y}, z={z}.")
            continue  # Skip to the next input if there's a conversion error
        
        data_type = str(row.get(f'INPUT_{i}_DATA TYPE', 'BFLOAT16')).strip()
        
        
        # Calculate size and add to total
        size = w * x * y * z * data_type_sizes.get(data_type, 2)  # Default to 2B if unknown data type
        total_size += size
    
    return total_size

def calculate_total_duration(df, column_name):
    numeric_durations = pd.to_numeric(df[column_name], errors='coerce')
    total_duration_ns = numeric_durations.sum()
    # Convert nanoseconds to milliseconds
    return total_duration_ns / 1e6

def find_top_k_longest_ops(df, k, column_name):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    top_k = df.nlargest(k, column_name)[['GLOBAL CALL COUNT', 'OP CODE', column_name]]
    # Convert nanoseconds to milliseconds for display
    top_k[column_name[:-5] + ' (ms)'] = top_k[column_name] / 1e6
    top_k['GLOBAL CALL COUNT'] = top_k['GLOBAL CALL COUNT'].astype(int)
    top_k.drop(columns=[column_name], inplace=True)
    return top_k

def main():
    args = parse_arguments()
    global LOG_DEBUG
    LOG_DEBUG = args.debug
    
    # Load CSV
    df = pd.read_csv(args.csv_file)
    
    # Perform calculations
    total_size_bytes = df.apply(calculate_data_size, axis=1).sum()
    total_device_duration_ms = calculate_total_duration(df, 'DEVICE FW DURATION [ns]')
    total_host_duration_ms = calculate_total_duration(df, 'HOST DURATION [ns]')
    top_k_longest_device_ops = find_top_k_longest_ops(df, args.top_k_ops, 'DEVICE FW DURATION [ns]')
        
    # Convert to gigabytes
    total_size_gb = total_size_bytes / (1024**3)
    
    # Compile results
    results = {
        'Total Input Data (GB)': [total_size_gb],
        'Total Device Duration (ms)': [total_device_duration_ms],
        'Total Host Duration (ms)': [total_host_duration_ms]
    }
    results_df = pd.DataFrame(results)
    
    # Append top k longest operations to the results dataframe (for CSV output)
    all_results_df = pd.concat([results_df, top_k_longest_device_ops], axis=1)
    
    # Save results to CSV
    all_results_df.to_csv(args.output, index=False)
    
    # Pretty print the results table
    print(tabulate(results_df, headers='keys', tablefmt='pretty'))
    print(tabulate(top_k_longest_device_ops, headers='keys', tablefmt='pretty'))


if __name__ == "__main__":
    main()
