#!/usr/bin/env python3
"""
Compare max_ab vs max_abc strategy CSV results row by row.
"""

import pandas as pd
import sys
from pathlib import Path


def load_csv(path):
    """Load CSV and filter out empty/error rows."""
    df = pd.read_csv(path)
    # Remove empty rows
    df = df.dropna(subset=["op_type"])
    return df


def create_config_key(row):
    """Create a unique key for matching configurations."""
    return (
        row["op_type"],
        row["a_shape"],
        row["a_sharding"],
        row["a_cores"],
        row["b_shape"],
        row["b_sharding"],
        row["b_cores"],
        row["c_sharding"],
    )


def main():
    results_dir = Path("/workspace/tests/ttnn/benchmarks/binary_ng/results")

    # Load both CSVs
    max_ab_path = results_dir / "example_multiple_ops_max_ab_20251113_013450.csv"
    max_abc_path = results_dir / "example_multiple_ops_max_abc_20251113_062946.csv"

    print("=" * 80)
    print("COMPARING max_ab vs max_abc GRID SELECTION STRATEGIES")
    print("=" * 80)
    print()

    df_ab = load_csv(max_ab_path)
    df_abc = load_csv(max_abc_path)

    print(f"max_ab:  {len(df_ab)} configurations")
    print(f"max_abc: {len(df_abc)} configurations")
    print()

    # Check for errors in each
    errors_ab = df_ab[df_ab["error"].notna() & (df_ab["error"] != "")]
    errors_abc = df_abc[df_abc["error"].notna() & (df_abc["error"] != "")]

    print(f"Errors in max_ab:  {len(errors_ab)}")
    print(f"Errors in max_abc: {len(errors_abc)}")
    print()

    if len(errors_abc) > 0:
        print("ERROR DETAILS in max_abc:")
        print("-" * 80)
        for idx, row in errors_abc.iterrows():
            error_msg = str(row["error"])
            # Extract the key part of the error
            if "TT_FATAL" in error_msg:
                key_error = error_msg.split("TT_FATAL")[1].split("\\n")[0][:100]
            else:
                key_error = error_msg[:100]
            print(
                f"Row {idx}: {row['a_shape']} ({row['a_sharding']},{row['a_cores']}) + "
                f"{row['b_shape']} ({row['b_sharding']},{row['b_cores']}) → "
                f"{row['c_shape']} ({row['c_sharding']},{row['c_cores']})"
            )
            print(f"  Error: {key_error}")
            print()

    # Create dictionaries keyed by configuration
    ab_dict = {}
    abc_dict = {}

    for _, row in df_ab.iterrows():
        if pd.notna(row["kernel_time_us"]) and row["kernel_time_us"] > 0:
            key = create_config_key(row)
            ab_dict[key] = row

    for _, row in df_abc.iterrows():
        if pd.notna(row["kernel_time_us"]) and row["kernel_time_us"] > 0:
            key = create_config_key(row)
            abc_dict[key] = row

    # Find matching configurations
    common_keys = set(ab_dict.keys()) & set(abc_dict.keys())
    only_ab = set(ab_dict.keys()) - set(abc_dict.keys())
    only_abc = set(abc_dict.keys()) - set(ab_dict.keys())

    print("=" * 80)
    print("CONFIGURATION OVERLAP")
    print("=" * 80)
    print(f"Common configurations: {len(common_keys)}")
    print(f"Only in max_ab: {len(only_ab)}")
    print(f"Only in max_abc: {len(only_abc)}")
    print()

    # Analyze differences for common configurations
    differences = []

    for key in common_keys:
        ab_row = ab_dict[key]
        abc_row = abc_dict[key]

        ab_time = ab_row["kernel_time_us"]
        abc_time = abc_row["kernel_time_us"]

        diff_us = abc_time - ab_time
        pct_diff = (diff_us / ab_time) * 100 if ab_time > 0 else 0

        ab_cores = ab_row["compute_cores"]
        abc_cores = abc_row["compute_cores"]
        c_cores = ab_row["c_cores"]

        differences.append(
            {
                "config": key,
                "ab_time": ab_time,
                "abc_time": abc_time,
                "diff_us": diff_us,
                "pct_diff": pct_diff,
                "ab_cores": ab_cores,
                "abc_cores": abc_cores,
                "c_cores": c_cores,
                "c_sharding": ab_row["c_sharding"],
                "a_cores": ab_row["a_cores"],
                "b_cores": ab_row["b_cores"],
            }
        )

    # Convert to DataFrame for analysis
    diff_df = pd.DataFrame(differences)

    print("=" * 80)
    print("OVERALL PERFORMANCE COMPARISON")
    print("=" * 80)
    print(f"Total comparisons: {len(diff_df)}")
    print()
    print(f"Mean difference: {diff_df['diff_us'].mean():.3f} μs ({diff_df['pct_diff'].mean():.2f}%)")
    print(f"Median difference: {diff_df['diff_us'].median():.3f} μs ({diff_df['pct_diff'].median():.2f}%)")
    print(f"Std dev: {diff_df['diff_us'].std():.3f} μs")
    print()
    print(
        f"max_ab faster: {(diff_df['diff_us'] < 0).sum()} cases ({(diff_df['diff_us'] < 0).sum() / len(diff_df) * 100:.1f}%)"
    )
    print(
        f"max_abc faster: {(diff_df['diff_us'] > 0).sum()} cases ({(diff_df['diff_us'] > 0).sum() / len(diff_df) * 100:.1f}%)"
    )
    print(f"Identical: {(diff_df['diff_us'] == 0).sum()} cases")
    print()

    # Analyze by whether compute cores differ
    diff_df["cores_differ"] = diff_df["ab_cores"] != diff_df["abc_cores"]

    print("=" * 80)
    print("IMPACT OF COMPUTE CORE DIFFERENCES")
    print("=" * 80)

    same_cores = diff_df[~diff_df["cores_differ"]]
    diff_cores = diff_df[diff_df["cores_differ"]]

    print(f"Same compute cores: {len(same_cores)} cases")
    if len(same_cores) > 0:
        print(f"  Mean diff: {same_cores['diff_us'].mean():.3f} μs ({same_cores['pct_diff'].mean():.2f}%)")
        print(f"  Median diff: {same_cores['diff_us'].median():.3f} μs ({same_cores['pct_diff'].median():.2f}%)")
    print()

    print(f"Different compute cores: {len(diff_cores)} cases")
    if len(diff_cores) > 0:
        print(f"  Mean diff: {diff_cores['diff_us'].mean():.3f} μs ({diff_cores['pct_diff'].mean():.2f}%)")
        print(f"  Median diff: {diff_cores['diff_us'].median():.3f} μs ({diff_cores['pct_diff'].median():.2f}%)")
        print()

        # Show examples where cores differ
        print("  Examples where compute cores differ:")
        for _, row in diff_cores.head(10).iterrows():
            config = row["config"]
            print(
                f"    {config[1]} ({config[2]},{row['a_cores']}) + {config[4]} ({config[5]},{row['b_cores']}) → c={config[7]},{row['c_cores']}"
            )
            print(f"      max_ab: {row['ab_cores']} cores, {row['ab_time']:.2f} μs")
            print(f"      max_abc: {row['abc_cores']} cores, {row['abc_time']:.2f} μs")
            print(f"      Diff: {row['diff_us']:+.2f} μs ({row['pct_diff']:+.1f}%)")
    print()

    # Analyze by output sharding type
    print("=" * 80)
    print("PERFORMANCE BY OUTPUT SHARDING TYPE")
    print("=" * 80)

    for c_sharding in ["height", "width", "block", "interleaved"]:
        subset = diff_df[diff_df["c_sharding"] == c_sharding]
        if len(subset) > 0:
            print(f"\n{c_sharding.upper()} sharded output: {len(subset)} cases")
            print(f"  Mean diff: {subset['diff_us'].mean():.3f} μs ({subset['pct_diff'].mean():.2f}%)")
            print(f"  max_ab faster: {(subset['diff_us'] < 0).sum()} cases")
            print(f"  max_abc faster: {(subset['diff_us'] > 0).sum()} cases")

            # Check if cores differ for this sharding type
            cores_same = subset[~subset["cores_differ"]]
            cores_diff = subset[subset["cores_differ"]]
            if len(cores_diff) > 0:
                print(f"  → When cores differ ({len(cores_diff)} cases): {cores_diff['pct_diff'].mean():+.2f}% avg")
            if len(cores_same) > 0:
                print(f"  → When cores same ({len(cores_same)} cases): {cores_same['pct_diff'].mean():+.2f}% avg")

    print()

    # Top 10 improvements and regressions
    print("=" * 80)
    print("TOP 10 CASES WHERE max_abc IS FASTER")
    print("=" * 80)
    top_abc = diff_df.nsmallest(10, "diff_us")
    for idx, row in top_abc.iterrows():
        config = row["config"]
        print(
            f"{config[1]} ({config[2]},{row['a_cores']}) + {config[4]} ({config[5]},{row['b_cores']}) → "
            f"c={config[7]},{row['c_cores']}"
        )
        print(f"  max_ab: {row['ab_cores']} cores, {row['ab_time']:.2f} μs")
        print(f"  max_abc: {row['abc_cores']} cores, {row['abc_time']:.2f} μs")
        print(f"  Improvement: {-row['diff_us']:.2f} μs ({-row['pct_diff']:.1f}%)")
        print()

    print("=" * 80)
    print("TOP 10 CASES WHERE max_ab IS FASTER")
    print("=" * 80)
    top_ab = diff_df.nlargest(10, "diff_us")
    for idx, row in top_ab.iterrows():
        config = row["config"]
        print(
            f"{config[1]} ({config[2]},{row['a_cores']}) + {config[4]} ({config[5]},{row['b_cores']}) → "
            f"c={config[7]},{row['c_cores']}"
        )
        print(f"  max_ab: {row['ab_cores']} cores, {row['ab_time']:.2f} μs")
        print(f"  max_abc: {row['abc_cores']} cores, {row['abc_time']:.2f} μs")
        print(f"  Improvement: {row['diff_us']:.2f} μs ({row['pct_diff']:.1f}%)")
        print()

    # Key insights
    print("=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()

    # Pattern 1: When max_abc uses more cores (c's cores)
    abc_uses_more = diff_df[diff_df["abc_cores"] > diff_df["ab_cores"]]
    if len(abc_uses_more) > 0:
        print(f"1. When max_abc uses MORE cores ({len(abc_uses_more)} cases):")
        print(f"   Average performance: {abc_uses_more['pct_diff'].mean():+.2f}%")
        if abc_uses_more["pct_diff"].mean() > 0:
            print(f"   → max_abc is SLOWER despite using more cores")
        else:
            print(f"   → max_abc is FASTER with more cores")
        print()

    # Pattern 2: When max_abc uses fewer cores
    abc_uses_less = diff_df[diff_df["abc_cores"] < diff_df["ab_cores"]]
    if len(abc_uses_less) > 0:
        print(f"2. When max_abc uses FEWER cores ({len(abc_uses_less)} cases):")
        print(f"   Average performance: {abc_uses_less['pct_diff'].mean():+.2f}%")
        print()

    # Pattern 3: Interleaved output
    interleaved_out = diff_df[diff_df["c_sharding"] == "interleaved"]
    if len(interleaved_out) > 0:
        print(f"3. Interleaved output ({len(interleaved_out)} cases):")
        print(f"   Average performance: {interleaved_out['pct_diff'].mean():+.2f}%")
        cores_differ_interleaved = interleaved_out[interleaved_out["cores_differ"]]
        if len(cores_differ_interleaved) > 0:
            print(f"   When cores differ: {cores_differ_interleaved['pct_diff'].mean():+.2f}%")
            print(f"   → max_ab ignores C (interleaved), uses max(A,B)")
            print(f"   → max_abc might choose C's cores (full grid) or max(A,B)")
        print()

    # Pattern 4: Sharded output where max_abc should match C
    for c_type in ["height", "width", "block"]:
        sharded_out = diff_df[diff_df["c_sharding"] == c_type]
        if len(sharded_out) > 0:
            cores_match_c = sharded_out[sharded_out["abc_cores"] == sharded_out["c_cores"]]
            cores_not_match_c = sharded_out[sharded_out["abc_cores"] != sharded_out["c_cores"]]

            if len(cores_match_c) > 0:
                print(f"4. {c_type.upper()} output where max_abc uses C's cores ({len(cores_match_c)} cases):")
                print(f"   Average performance: {cores_match_c['pct_diff'].mean():+.2f}%")
                print()


if __name__ == "__main__":
    main()
