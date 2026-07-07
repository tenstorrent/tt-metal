#!/usr/bin/env python3

import subprocess
import re
import json
import csv
import time
import statistics
from datetime import datetime
from typing import List, Dict, Optional
import os
import glob

# Import GitHubPerformanceUploader if available
try:
    from push_to_github import GitHubPerformanceUploader
    GITHUB_AVAILABLE = True
except ImportError as e:
    GITHUB_AVAILABLE = False
    GITHUB_IMPORT_ERROR = str(e)

class PerfMeasurement:
    def __init__(self, rerun_mode=False, auto_upload=False):
        self.results = []
        self.failed_tests = []
        self.start_time = datetime.now()
        self.rerun_mode = rerun_mode
        self.auto_upload = auto_upload
        self.today_date = self.start_time.strftime("%Y%m%d")
        
        # For dynamic ETA calculation
        self.test_completion_times = []
        self.current_test_start_time = None
        
        # Track partial files for cleanup
        self.partial_files = []
        
        # Load existing results from today only if in rerun mode
        if self.rerun_mode:
            self.load_existing_results()
        
    def get_git_commit_id(self) -> str:
        """Get the current git commit ID."""
        try:
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "unknown"
        except Exception as e:
            print(f"⚠️ Warning: Could not get git commit ID: {e}")
            return "unknown"

    def load_existing_results(self):
        """Load existing results from today to avoid re-running successful tests."""
        pattern = f"eltwise_perf_results_{self.today_date}_*.json"
        existing_files = glob.glob(pattern)
        
        if not existing_files:
            print(f"📅 No existing results found for today ({self.today_date})")
            return
            
        # Get the most recent file from today
        latest_file = max(existing_files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
                
            self.results = data.get('results', [])
            self.failed_tests = data.get('metadata', {}).get('failed_test_names', [])
            
            successful_tests = [r['test_name'] for r in self.results]
            
            print(f"📂 Loaded existing results from: {latest_file}")
            print(f"✅ Found {len(successful_tests)} successful tests from today")
            print(f"❌ Found {len(self.failed_tests)} failed tests from today")
            
            if successful_tests:
                print(f"🔄 Already completed: {', '.join(successful_tests[:5])}" + 
                      (f" and {len(successful_tests)-5} more..." if len(successful_tests) > 5 else ""))
                      
        except Exception as e:
            print(f"⚠️ Error loading existing results: {e}")
            self.results = []
            self.failed_tests = []
    
    def get_tests_to_run(self) -> List[str]:
        """Get list of tests that need to be run based on mode and existing results."""
        all_tests = self.get_all_test_names()
        
        if not all_tests:
            return []
        
        # If not in rerun mode, run all tests (original behavior)
        if not self.rerun_mode:
            print(f"🚀 Standard mode: Running all {len(all_tests)} tests")
            return all_tests
            
        # Rerun mode: Run tests that haven't been successful yet today (failed + not run)
        already_successful = {result['test_name'] for result in self.results}
        tests_to_run = [test for test in all_tests if test not in already_successful]
        
        print(f"📊 Rerun mode: Running {len(tests_to_run)} tests (skipping {len(already_successful)} successful)")
        
        if not tests_to_run:
            print("✅ All tests already completed successfully today!")
                
        return tests_to_run

    def start_test_timing(self):
        """Start timing for current test."""
        self.current_test_start_time = datetime.now()
    
    def end_test_timing(self):
        """End timing for current test and record duration."""
        if self.current_test_start_time:
            test_duration = (datetime.now() - self.current_test_start_time).total_seconds()
            self.test_completion_times.append(test_duration)
            self.current_test_start_time = None
            return test_duration
        return None
    
    def calculate_dynamic_eta(self, completed_tests: int, total_tests: int) -> str:
        """Calculate dynamic ETA based on average test completion time."""
        if not self.test_completion_times or completed_tests >= total_tests:
            return "calculating..."
        
        avg_time_per_test = statistics.mean(self.test_completion_times)
        remaining_tests = total_tests - completed_tests
        estimated_remaining_seconds = remaining_tests * avg_time_per_test
        
        return self.format_duration(estimated_remaining_seconds)
    
    def format_duration(self, seconds: float) -> str:
        """Format duration in seconds to human readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes:.0f}m {secs:.0f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours:.0f}h {minutes:.0f}m"

    def get_all_test_names(self) -> List[str]:
        """Extract all test function names from the test file."""
        try:
            cmd = ["python", "-m", "pytest", "test_eltwise_operations.py", "--collect-only", "-q"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Tests to exclude (known failing tests)
            excluded_tests = {
                'test_complex_tensor',
                'test_real', 
                'test_imag',
                'test_frac_bw'
            }
            
            test_names = []
            for line in result.stdout.split('\n'):
                if '<Function test_' in line:
                    # Extract test name from '<Function test_abs>'
                    match = re.search(r'<Function (test_\w+)>', line)
                    if match:
                        test_name = match.group(1)
                        if test_name not in excluded_tests:
                            test_names.append(test_name)
                        else:
                            print(f"⚠️ Excluding known failing test: {test_name}")
            
            print(f"Found {len(test_names)} total tests available (excluded {len(excluded_tests)} known failing tests)")
            return test_names
            
        except Exception as e:
            print(f"Error getting test names: {e}")
            return []
    
    def extract_kernel_duration(self, output: str) -> Optional[float]:
        """Extract kernel duration from ttperf output."""
        try:
            # Look for pattern: "⏱️ DEVICE KERNEL DURATION [ns] total: 24987.00 ns"
            pattern = r'DEVICE KERNEL DURATION \[ns\] total:\s+([\d.]+)\s+ns'
            match = re.search(pattern, output)
            
            if match:
                return float(match.group(1))
            else:
                print("Could not find kernel duration in output")
                return None
        except Exception as e:
            print(f"Error extracting kernel duration: {e}")
            return None
    
    def run_single_perf_test(self, test_name: str, run_number: int) -> Optional[float]:
        """Run a single performance test and extract kernel duration."""
        try:
            cmd = ["ttperf", f"test_eltwise_operations.py::TestEltwiseOperations::{test_name}"]
            print(f"  Run {run_number}: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                duration = self.extract_kernel_duration(result.stdout)
                if duration is not None:
                    print(f"    ✅ Duration: {duration} ns")
                    return duration
                else:
                    print(f"    ❌ Could not extract duration from output")
            else:
                print(f"    ❌ Test failed (exit code {result.returncode})")
                stderr_lines = result.stderr.strip().splitlines()
                # Show last meaningful lines from stderr
                relevant = [l for l in stderr_lines if l.strip()][-10:]
                for line in relevant:
                    print(f"       {line}")
                if not relevant and result.stdout:
                    stdout_lines = result.stdout.strip().splitlines()
                    for line in stdout_lines[-5:]:
                        print(f"       {line}")
                
        except subprocess.TimeoutExpired:
            print(f"    ⏰ Test {test_name} run {run_number} timed out")
        except Exception as e:
            print(f"    ❌ Error running test: {e}")
        
        return None
    
    def run_perf_measurement_for_test(self, test_name: str) -> Optional[Dict]:
        """Run performance measurement 3 times for a single test and calculate average."""
        print(f"\n📊 Measuring {test_name}...")
        
        self.start_test_timing()
        
        durations = []
        for run_num in range(1, 4):  # 3 runs
            duration = self.run_single_perf_test(test_name, run_num)
            if duration is not None:
                durations.append(duration)
            time.sleep(1)  # Brief pause between runs
        
        test_completion_time = self.end_test_timing()
        
        if durations:
            avg_duration = statistics.mean(durations)
            std_deviation = statistics.stdev(durations) if len(durations) > 1 else 0
            
            result = {
                'test_name': test_name,
                'operation_name': test_name.replace('test_', ''),
                'runs': durations,
                'successful_runs': len(durations),
                'average_duration_ns': avg_duration,
                'std_deviation_ns': std_deviation,
                'min_duration_ns': min(durations),
                'max_duration_ns': max(durations),
                'timestamp': datetime.now().isoformat()
            }
            
            completion_msg = f"  ✅ Average: {avg_duration:.2f} ns (±{std_deviation:.2f}) from {len(durations)} runs"
            if test_completion_time:
                completion_msg += f" | Completed in {self.format_duration(test_completion_time)}"
            print(completion_msg)
            
            # Remove from failed tests if it was there and now succeeded
            if test_name in self.failed_tests:
                self.failed_tests.remove(test_name)
                print(f"  🎉 Test {test_name} now passed! Removed from failed list.")
                
            return result
        else:
            print(f"  ❌ All runs failed for {test_name}")
            if test_name not in self.failed_tests:
                self.failed_tests.append(test_name)
            return None
    
    def save_results(self, final=False):
        """Save results to JSON and CSV files."""
        suffix = "final" if final else f"partial_{len(self.results)}"
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        json_filename = f"eltwise_perf_results_{timestamp}_{suffix}.json"
        
        with open(json_filename, 'w') as f:
            json.dump({
                'metadata': {
                    'measurement_date': self.start_time.isoformat(),
                    'total_tests': len(self.results) + len(self.failed_tests),
                    'successful_tests': len(self.results),
                    'failed_tests': len(self.failed_tests),
                    'failed_test_names': self.failed_tests,
                    'rerun_mode': self.rerun_mode,
                    'git_commit_id': self.get_git_commit_id()
                },
                'results': self.results
            }, f, indent=2)
        
        # Save CSV for database upload
        csv_filename = f"eltwise_perf_results_{timestamp}_{suffix}.csv"
        with open(csv_filename, 'w', newline='') as f:
            if self.results:
                fieldnames = [
                    'test_name', 'operation_name', 'average_duration_ns', 
                    'std_deviation_ns', 'min_duration_ns', 'max_duration_ns',
                    'successful_runs', 'timestamp'
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    # Exclude 'runs' field for CSV as it's an array
                    csv_row = {k: v for k, v in result.items() if k != 'runs'}
                    writer.writerow(csv_row)
        
        # Track partial files for cleanup
        if not final:
            self.partial_files.extend([json_filename, csv_filename])
        
        print(f"\n📂 Results saved to:")
        print(f"  📄 JSON: {json_filename}")
        print(f"  📊 CSV: {csv_filename}")
        
        # Clean up partial files if this is the final save
        if final and self.partial_files:
            self.cleanup_partial_files()
        
        return json_filename, csv_filename
    
    def cleanup_partial_files(self):
        """Remove partial files after final save is completed."""
        cleaned_count = 0
        for partial_file in self.partial_files:
            try:
                if os.path.exists(partial_file):
                    os.remove(partial_file)
                    cleaned_count += 1
            except Exception as e:
                print(f"⚠️ Warning: Could not remove partial file {partial_file}: {e}")
        
        if cleaned_count > 0:
            print(f"🧹 Cleaned up {cleaned_count} partial files")
        
        self.partial_files = []

    def run_all_measurements(self):
        """Run performance measurements for tests based on selected mode."""
        tests_to_run = self.get_tests_to_run()
        
        if not tests_to_run:
            # Still save current state even if no tests to run
            if self.results and self.rerun_mode:
                json_file, csv_file = self.save_results(final=True)
                
                # Upload to GitHub if requested, even when no new tests run
                if self.auto_upload:
                    print("📤 No new tests to run, but uploading existing results...")
                    upload_success = self.upload_to_github(json_file)
                    if upload_success:
                        print("📤 Automatic upload completed successfully!")
                    else:
                        print("⚠️ Automatic upload failed, but results are saved locally")
            elif not self.rerun_mode:
                print("❌ No tests found!")
            return
        
        print(f"🚀 Starting performance measurement for {len(tests_to_run)} tests")
        print(f"📅 Start time: {self.start_time}")
        print(f"🔧 Git commit: {self.get_git_commit_id()}")
        print(f"⏱️ Estimated time: ~{len(tests_to_run) * 2} minutes (initial estimate)")
        
        for i, test_name in enumerate(tests_to_run, 1):
            # Calculate dynamic ETA
            eta = self.calculate_dynamic_eta(i - 1, len(tests_to_run))
            progress_pct = i / len(tests_to_run) * 100
            
            if i == 1:
                print(f"\n🔄 Progress: {i}/{len(tests_to_run)} ({progress_pct:.1f}%) | ETA: {eta}")
            else:
                avg_time = statistics.mean(self.test_completion_times) if self.test_completion_times else 0
                print(f"\n🔄 Progress: {i}/{len(tests_to_run)} ({progress_pct:.1f}%) | ETA: {eta} | Avg: {self.format_duration(avg_time)}/test")
            
            result = self.run_perf_measurement_for_test(test_name)
            if result:
                # Check if this test already exists in results (in case of rerun)
                existing_idx = next((idx for idx, r in enumerate(self.results) 
                                   if r['test_name'] == test_name), None)
                if existing_idx is not None:
                    self.results[existing_idx] = result
                    print(f"  🔄 Updated existing result for {test_name}")
                else:
                    self.results.append(result)
            
            # Save intermediate results every 10 tests
            if i % 10 == 0:
                self.save_results()
                print(f"💾 Intermediate save completed at test {i}")
        
        # Final save
        json_file, csv_file = self.save_results(final=True)
        
        # Upload to GitHub if requested
        if self.auto_upload:
            upload_success = self.upload_to_github(json_file)
            if upload_success:
                print("📤 Automatic upload completed successfully!")
            else:
                print("⚠️ Automatic upload failed, but results are saved locally")
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        print(f"\n✅ Performance measurement completed!")
        print(f"⏱️ Total time: {duration}")
        if self.test_completion_times:
            avg_per_test = statistics.mean(self.test_completion_times)
            print(f"📊 Average time per test: {self.format_duration(avg_per_test)}")
        print(f"📈 Total successful tests: {len(self.results)}")
        print(f"❌ Total failed tests: {len(self.failed_tests)}")
        
        if self.failed_tests:
            print(f"🔍 Failed tests: {', '.join(self.failed_tests)}")
            if not self.rerun_mode:
                print(f"💡 To rerun failed/remaining tests, use: python {__file__} --rerun")
    
    def upload_to_github(self, json_file_path: str):
        """Upload results to GitHub repository."""
        print(f"\n📤 Attempting to upload results to GitHub...")
        
        if not GITHUB_AVAILABLE:
            print("❌ Upload failed: GitHubPerformanceUploader not available")
            if 'GITHUB_IMPORT_ERROR' in globals():
                print(f"   Import error: {GITHUB_IMPORT_ERROR}")
            print("💡 Falling back to manual upload using push_to_github.py script")
            return self.manual_upload_fallback(json_file_path)
        
        # Check for required repository URL
        repo_url = "git@github.com:Aswincloud/ttnn-performance-dashboard.git"
        
        if not repo_url:
            print("❌ Upload failed: Missing GitHub repository URL")
            return self.manual_upload_fallback(json_file_path)
        
        print(f"🔗 Repository: {repo_url}")
        print(f"📄 File: {json_file_path}")
        
        try:
            uploader = GitHubPerformanceUploader(repo_url)
            success = uploader.upload_results(json_file_path)
            
            if success:
                print("🎉 Successfully uploaded results to GitHub!")
                return True
            else:
                print("❌ GitHubPerformanceUploader failed")
                return self.manual_upload_fallback(json_file_path)
                
        except Exception as e:
            print(f"❌ Upload error: {e}")
            return self.manual_upload_fallback(json_file_path)
    
    def manual_upload_fallback(self, json_file_path: str):
        """Fallback to manual upload using push_to_github.py script."""
        print("🔄 Attempting manual upload using push_to_github.py script...")
        
        try:
            # Try to run the push_to_github.py script directly
            cmd = ["python3", "push_to_github.py", json_file_path]
            print(f"🚀 Running: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("🎉 Manual upload successful!")
                print(result.stdout)
                return True
            else:
                print(f"❌ Manual upload failed: {result.stderr}")
                print("💡 You can manually upload later with:")
                print(f"   python3 push_to_github.py {json_file_path}")
                return False
                
        except Exception as e:
            print(f"❌ Manual upload error: {e}")
            print("💡 You can manually upload later with:")
            print(f"   python3 push_to_github.py {json_file_path}")
            return False

def main():
    """Main function to run performance measurements."""
    import argparse
    
    parser = argparse.ArgumentParser(description='TTNN Eltwise Operations Performance Measurement')
    parser.add_argument('--rerun', action='store_true', 
                       help='Skip tests that already passed today and run only missing/failed tests')
    parser.add_argument('--upload', action='store_true',
                       help='Automatically upload results to the database after completion')

    args = parser.parse_args()
    
    print("🎯 TTNN Eltwise Operations Performance Measurement")
    print("=" * 50)
    
    if args.rerun:
        print("📊 Mode: Smart rerun (skipping today's successful tests)")
    else:
        print("🚀 Mode: Standard run (all tests)")
    
    if args.upload:
        if GITHUB_AVAILABLE:
            print("📤 Auto-upload: Enabled (will upload to GitHub)")
        else:
            print("⚠️ Auto-upload: Disabled (push_to_github.py not found)")

    perf = PerfMeasurement(rerun_mode=args.rerun, auto_upload=args.upload)
    perf.run_all_measurements()

if __name__ == "__main__":
    main() 