#!/usr/bin/env python3

import subprocess
import os
import shutil
from pathlib import Path

def run_command(cmd):
    """Run a command and return the output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        return None
    return result.stdout.strip()

def main():
    print("🚀 Regenerating binary files for AOT compilation tests...")
    
    # Step 1: Clear old cache to ensure fresh generation
    cache_dir = Path.home() / ".cache" / "tt-metal-cache"
    if cache_dir.exists():
        print(f"📁 Clearing old cache: {cache_dir}")
        shutil.rmtree(cache_dir)
    
    # Step 2: Run the original test to generate fresh binaries
    print("⚙️  Running kernel to generate fresh binaries...")
    cmd = "./build_Release/test/tt_metal/unit_tests_api --gtest_filter='*TestCreateKernelFromBinary*' --gtest_also_run_disabled_tests"
    
    # We expect this to fail, but it should generate the binaries in the process
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print("✅ Kernel execution completed (expected to fail due to missing binaries)")
    
    # Step 3: Find the generated cache directory
    print("🔍 Looking for generated binaries in cache...")
    cache_dirs = list(cache_dir.glob("*/"))
    if not cache_dirs:
        print("❌ No cache directories found! Something went wrong.")
        return False
    
    latest_cache = max(cache_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📂 Found cache directory: {latest_cache}")
    
    # Step 4: Find kernel binaries in the cache
    kernel_dirs = list(latest_cache.glob("**/kernels/simple_add/*/"))
    if not kernel_dirs:
        print("❌ No simple_add kernel directories found in cache!")
        return False
    
    for kernel_dir in kernel_dirs:
        print(f"🔍 Found kernel directory: {kernel_dir}")
        hash_value = kernel_dir.name
        print(f"📊 New hash: {hash_value}")
        
        # Step 5: Determine device architecture
        # Check if this is wormhole or blackhole based on the cache path
        device_arch = None
        if "wormhole" in str(kernel_dir).lower():
            device_arch = "wormhole"
        elif "blackhole" in str(kernel_dir).lower():
            device_arch = "blackhole"
        else:
            # Try to detect from parent directories or use wormhole as default
            device_arch = "wormhole"  # Default fallback
        
        print(f"🎯 Detected device architecture: {device_arch}")
        
        # Step 6: Create target directory structure
        target_base = Path("tests/tt_metal/tt_metal/api/simple_add_binaries")
        target_dir = target_base / device_arch / "kernels" / "simple_add" / hash_value
        
        print(f"📁 Creating target directory: {target_dir}")
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 7: Copy all processor binaries
        for processor_dir in kernel_dir.iterdir():
            if processor_dir.is_dir():
                print(f"📋 Copying {processor_dir.name} binaries...")
                target_proc_dir = target_dir / processor_dir.name
                if target_proc_dir.exists():
                    shutil.rmtree(target_proc_dir)
                shutil.copytree(processor_dir, target_proc_dir)
                print(f"✅ Copied {processor_dir} -> {target_proc_dir}")
    
    print("🎉 Binary regeneration completed successfully!")
    print("🧪 You can now run the tests:")
    print("   ./build_Release/test/tt_metal/unit_tests_api --gtest_filter='*TestCreateKernelFromBinary*'")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
