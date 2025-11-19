#!/usr/bin/env python3
"""
Script to visualize performance counter data as a grid.
Each core is shown as a square with coordinates and utilization metric.
Utilization = value / ref_cnt
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
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


def create_grid_visualization(df, runtime_id, counter_type, output_file=None):
    """Create a grid visualization with each core as a square."""
    
    # Get the range of coordinates
    x_coords = sorted(df['core_x'].unique())
    y_coords = sorted(df['core_y'].unique())
    
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Create a grid for utilization values
    grid_width = x_max - x_min + 1
    grid_height = y_max - y_min + 1
    
    # Initialize grid with NaN (for missing cores)
    utilization_grid = np.full((grid_height, grid_width), np.nan)
    
    # Fill in the utilization values
    for _, row in df.iterrows():
        x_idx = int(row['core_x']) - x_min
        y_idx = int(row['core_y']) - y_min
        utilization_grid[y_idx, x_idx] = row['utilization']
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max(12, grid_width * 1.2), max(10, grid_height * 1.2)))
    
    # Get min/max utilization for color normalization
    valid_utils = df['utilization'].values
    util_min, util_max = valid_utils.min(), valid_utils.max()
    norm = Normalize(vmin=util_min, vmax=util_max)
    
    # Use a colormap
    cmap = plt.cm.RdYlGn  # Red (low) to Yellow to Green (high)
    
    # Draw each core as a square
    for _, row in df.iterrows():
        x = int(row['core_x']) - x_min
        y = int(row['core_y']) - y_min
        util = row['utilization']
        
        # Get color based on utilization
        color = cmap(norm(util))
        
        # Draw square
        square = mpatches.Rectangle(
            (x, y), 1, 1,
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
        ax.add_patch(square)
        
        # Add text: coordinates and utilization
        coord_text = f"({int(row['core_x'])}, {int(row['core_y'])})"
        util_text = f"{util:.2f}%"
        
        # Add coordinate text at top of square
        ax.text(
            x + 0.5, y + 0.7,
            coord_text,
            ha='center', va='center',
            fontsize=9,
            fontweight='bold',
            color='black'
        )
        
        # Add utilization text at bottom of square
        ax.text(
            x + 0.5, y + 0.3,
            util_text,
            ha='center', va='center',
            fontsize=10,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    # Set axis properties
    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)
    ax.set_aspect('equal')
    
    # Set x-axis ticks to show actual core_x values
    ax.set_xticks(np.arange(grid_width) + 0.5)
    ax.set_xticklabels([x_min + i for i in range(grid_width)])
    ax.set_xlabel('Core X', fontsize=12, fontweight='bold')
    
    # Set y-axis ticks to show actual core_y values (flip for natural orientation)
    ax.set_yticks(np.arange(grid_height) + 0.5)
    ax.set_yticklabels([y_min + i for i in range(grid_height)])
    ax.set_ylabel('Core Y', fontsize=12, fontweight='bold')
    
    # Invert y-axis so (0,0) is at top-left
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Add title
    title = f'Core Utilization Grid\n'
    title += f'Runtime ID: {runtime_id} | Counter Type: {counter_type}\n'
    title += f'Utilization (%) = (value / ref_cnt) × 100'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Utilization (%)', fontsize=12, fontweight='bold')
    
    # Add statistics text
    stats_text = f'Min: {util_min:.2f}% | Max: {util_max:.2f}% | Mean: {valid_utils.mean():.2f}% | Std: {valid_utils.std():.2f}%'
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11, 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize performance counter data as a grid',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -f perf_counters.csv -r 1024 -c SFPU_COUNTER
  %(prog)s -f perf_counters.csv -r 1024 -c SFPU_COUNTER -o output.png
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
        help='Output file path for the visualization (e.g., output.png). If not provided, displays interactively.'
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
    create_grid_visualization(df, args.runtime_id, args.counter_type, args.output)
    
    print("Done!")


if __name__ == '__main__':
    main()

