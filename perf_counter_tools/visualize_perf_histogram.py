#!/usr/bin/env python3
"""
Script to create a histogram of performance counter utilization distribution.
Shows the distribution of utilization values across all cores.
Utilization = (value / ref_cnt) * 100
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys


def load_and_filter_data(csv_file, runtime_id, counter_type):
    """Load CSV and filter by runtime_id and counter_type."""
    try:
        # Read CSV, handling potential whitespace in column names
        df = pd.read_csv(csv_file, skipinitialspace=True)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Filter by runtime_id and counter_type
        filtered_df = df[
            (df['runtime id'] == runtime_id) & 
            (df['counter type'] == counter_type)
        ].copy()
        
        if filtered_df.empty:
            print(f"Error: No data found for runtime_id={runtime_id} and counter_type={counter_type}")
            return None
        
        # Calculate utilization = (value / ref cnt) * 100 as percentage
        filtered_df['utilization'] = (filtered_df['value'] / filtered_df['ref cnt']) * 100
        
        return filtered_df
        
    except FileNotFoundError:
        print(f"Error: File not found: {csv_file}")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None


def create_histogram(df, runtime_id, counter_type, output_file=None, bins=20):
    """Create a histogram of utilization distribution."""
    
    utilization = df['utilization'].values
    
    # Calculate statistics
    mean_util = utilization.mean()
    median_util = np.median(utilization)
    std_util = utilization.std()
    min_util = utilization.min()
    max_util = utilization.max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create histogram
    n, bins_edges, patches = ax.hist(
        utilization,
        bins=bins,
        edgecolor='black',
        linewidth=1.2,
        alpha=0.7,
        color='steelblue'
    )
    
    # Color bars based on value
    cm = plt.cm.RdYlGn
    bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    
    # Normalize for colormap
    col = (bin_centers - min_util) / (max_util - min_util)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    # Add vertical lines for mean and median
    ax.axvline(mean_util, color='red', linestyle='--', linewidth=2.5, 
               label=f'Mean: {mean_util:.2f}%', alpha=0.8)
    ax.axvline(median_util, color='orange', linestyle='--', linewidth=2.5, 
               label=f'Median: {median_util:.2f}%', alpha=0.8)
    
    # Labels and title
    ax.set_xlabel('Utilization (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Number of Cores', fontsize=13, fontweight='bold')
    
    title = f'Core Utilization Distribution\n'
    title += f'Runtime ID: {runtime_id} | Counter Type: {counter_type}'
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    
    # Add legend
    ax.legend(fontsize=11, loc='best', framealpha=0.9)
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Statistics:\n'
    stats_text += f'Cores: {len(utilization)}\n'
    stats_text += f'Min: {min_util:.2f}%\n'
    stats_text += f'Max: {max_util:.2f}%\n'
    stats_text += f'Mean: {mean_util:.2f}%\n'
    stats_text += f'Median: {median_util:.2f}%\n'
    stats_text += f'Std Dev: {std_util:.2f}%'
    
    # Place text box in upper right
    props = dict(boxstyle='round,pad=0.8', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5)
    ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Histogram saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def create_detailed_stats_plot(df, runtime_id, counter_type, output_file=None):
    """Create a more detailed plot with histogram and box plot."""
    
    utilization = df['utilization'].values
    
    # Calculate statistics
    mean_util = utilization.mean()
    median_util = np.median(utilization)
    q1 = np.percentile(utilization, 25)
    q3 = np.percentile(utilization, 75)
    std_util = utilization.std()
    min_util = utilization.min()
    max_util = utilization.max()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1, 0.3], hspace=0.3)
    
    # Top subplot: Histogram
    ax1 = fig.add_subplot(gs[0])
    
    n, bins_edges, patches = ax1.hist(
        utilization,
        bins=25,
        edgecolor='black',
        linewidth=1.2,
        alpha=0.7,
        color='steelblue'
    )
    
    # Color bars
    cm = plt.cm.RdYlGn
    bin_centers = 0.5 * (bins_edges[:-1] + bins_edges[1:])
    col = (bin_centers - min_util) / (max_util - min_util)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    # Add vertical lines
    ax1.axvline(mean_util, color='red', linestyle='--', linewidth=2.5, 
                label=f'Mean: {mean_util:.2f}%', alpha=0.8)
    ax1.axvline(median_util, color='orange', linestyle='--', linewidth=2.5, 
                label=f'Median: {median_util:.2f}%', alpha=0.8)
    ax1.axvline(q1, color='green', linestyle=':', linewidth=2, 
                label=f'Q1: {q1:.2f}%', alpha=0.7)
    ax1.axvline(q3, color='green', linestyle=':', linewidth=2, 
                label=f'Q3: {q3:.2f}%', alpha=0.7)
    
    ax1.set_ylabel('Number of Cores', fontsize=13, fontweight='bold')
    title = f'Core Utilization Distribution Analysis\n'
    title += f'Runtime ID: {runtime_id} | Counter Type: {counter_type}'
    ax1.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax1.legend(fontsize=10, loc='best', framealpha=0.9, ncol=2)
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Middle subplot: Box plot
    ax2 = fig.add_subplot(gs[1])
    
    bp = ax2.boxplot(
        utilization,
        vert=False,
        widths=0.6,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        boxprops=dict(facecolor='lightblue', alpha=0.7, edgecolor='black', linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        medianprops=dict(color='red', linewidth=2.5),
        meanprops=dict(color='darkred', linestyle='--', linewidth=2.5)
    )
    
    ax2.set_xlabel('Utilization (%)', fontsize=13, fontweight='bold')
    ax2.set_yticks([])
    ax2.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    ax2.set_title('Box Plot', fontsize=12, fontweight='bold', pad=10)
    
    # Bottom subplot: Statistics table
    ax3 = fig.add_subplot(gs[2])
    ax3.axis('off')
    
    stats_data = [
        ['Cores', 'Min (%)', 'Q1 (%)', 'Median (%)', 'Mean (%)', 'Q3 (%)', 'Max (%)', 'Std Dev (%)'],
        [f'{len(utilization)}', f'{min_util:.2f}', f'{q1:.2f}', f'{median_util:.2f}', 
         f'{mean_util:.2f}', f'{q3:.2f}', f'{max_util:.2f}', f'{std_util:.2f}']
    ]
    
    table = ax3.table(
        cellText=stats_data,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(stats_data[0])):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')
    
    # Style data row
    for i in range(len(stats_data[0])):
        cell = table[(1, i)]
        cell.set_facecolor('#E8F5E9')
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Detailed statistics plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create histogram of performance counter utilization distribution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple histogram
  %(prog)s -f perf_counters.csv -r 1024 -c SFPU_COUNTER
  
  # Save to file with custom bins
  %(prog)s -f perf_counters.csv -r 1024 -c SFPU_COUNTER -o histogram.png -b 30
  
  # Detailed statistics plot
  %(prog)s -f perf_counters.csv -r 1024 -c SFPU_COUNTER -o detailed.png --detailed
        """
    )
    
    parser.add_argument(
        '-f', '--file',
        required=True,
        help='Path to the CSV file containing performance counter data'
    )
    
    parser.add_argument(
        '-r', '--runtime-id',
        type=int,
        required=True,
        help='Runtime ID to filter (e.g., 1024)'
    )
    
    parser.add_argument(
        '-c', '--counter-type',
        required=True,
        help='Counter type to filter (e.g., SFPU_COUNTER)'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file path for the histogram (e.g., histogram.png). If not provided, displays interactively.'
    )
    
    parser.add_argument(
        '-b', '--bins',
        type=int,
        default=20,
        help='Number of bins for the histogram (default: 20)'
    )
    
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Create a detailed plot with histogram, box plot, and statistics table'
    )
    
    args = parser.parse_args()
    
    # Load and filter data
    print(f"Loading data from: {args.file}")
    print(f"Filtering by runtime_id={args.runtime_id}, counter_type={args.counter_type}")
    
    df = load_and_filter_data(args.file, args.runtime_id, args.counter_type)
    
    if df is None:
        sys.exit(1)
    
    print(f"Found {len(df)} cores with matching criteria")
    
    # Create visualization
    if args.detailed:
        create_detailed_stats_plot(df, args.runtime_id, args.counter_type, args.output)
    else:
        create_histogram(df, args.runtime_id, args.counter_type, args.output, args.bins)
    
    print("Done!")


if __name__ == '__main__':
    main()

