#!/usr/bin/env python3
"""
Dry-run version of the chunk size sweep script for testing.
Tests only a few combinations to verify the script works correctly.
"""

import os
import sys

# Import the main sweeper class
from sweep_chunk_sizes import ChunkSizeSweeper, TEST_FILE

# Override chunk sizes for dry run
DRY_RUN_CHUNK_SIZES = [32, 64]


class DryRunSweeper(ChunkSizeSweeper):
    """Dry-run version with limited test cases."""

    def run_sweep(self):
        """Execute a limited parameter sweep for testing."""
        self.log("="*60)
        self.log("DRY RUN MODE - Testing limited configurations")
        self.log(f"Testing chunk sizes: {DRY_RUN_CHUNK_SIZES}")
        self.log(f"Test file: {TEST_FILE}")
        self.log("="*60 + "\n")

        try:
            # Backup original file
            self.backup_test_file()

            # Test only a few combinations
            test_combinations = [
                (32, 32),
                (32, 64),
                (64, 32),
            ]

            for q_chunk, k_chunk in test_combinations:
                self.log(f"\nTesting q_chunk_size={q_chunk}, k_chunk_size={k_chunk}")

                # Modify test parameters
                if not self.modify_test_parameters(q_chunk, k_chunk):
                    self.log("  Failed to modify test file, skipping")
                    continue

                # Verify the modification
                with open(TEST_FILE, 'r') as f:
                    lines = f.readlines()
                    modified_line = lines[614].strip()  # 0-indexed
                    self.log(f"  Modified line: {modified_line}")

                # Run test with tracy
                success, csv_path = self.run_tracy_test()

                if not success:
                    self.results.append({
                        'q_chunk_size': q_chunk,
                        'k_chunk_size': k_chunk,
                        'max_time_ns': None,
                        'min_time_ns': None,
                        'status': 'FAILED/OOM'
                    })
                    self.log(f"  FAILED")
                else:
                    # Parse results
                    max_time, min_time = self.parse_tracy_csv(csv_path)

                    if max_time is None:
                        self.results.append({
                            'q_chunk_size': q_chunk,
                            'k_chunk_size': k_chunk,
                            'max_time_ns': None,
                            'min_time_ns': None,
                            'status': 'PARSE_ERROR'
                        })
                        self.log("  Failed to parse results")
                    else:
                        self.results.append({
                            'q_chunk_size': q_chunk,
                            'k_chunk_size': k_chunk,
                            'max_time_ns': max_time,
                            'min_time_ns': min_time,
                            'status': 'SUCCESS'
                        })
                        self.log(f"  ✓ MAX: {max_time/1e9:.3f}s, MIN: {min_time/1e9:.3f}s")

                # Save progress
                self.save_results()

            # Final summary
            self.save_results()
            self.log("\n" + "="*60)
            self.log("DRY RUN COMPLETE")
            self.log(f"Tested {len(test_combinations)} combinations")
            self.log(f"Results file: {self.results_file}")
            self.log(f"Log file: {self.log_file}")
            self.log("="*60)

        except KeyboardInterrupt:
            self.log("\n\nDry run interrupted by user")
            self.save_results()

        except Exception as e:
            self.log(f"\n\nUnexpected error: {e}")
            import traceback
            self.log(traceback.format_exc())

        finally:
            # Always restore original file
            self.restore_test_file()
            self.log("\nTest file restored to original state")


def main():
    """Main entry point for dry run."""
    if not os.path.exists(TEST_FILE):
        print(f"Error: Test file not found: {TEST_FILE}")
        print("Please run this script from /tt-metal directory")
        sys.exit(1)

    print("Starting dry run with limited test cases...")
    print("This will test only 3 combinations: (32,32), (32,64), (64,32)")
    print()

    sweeper = DryRunSweeper()
    sweeper.run_sweep()


if __name__ == "__main__":
    main()
