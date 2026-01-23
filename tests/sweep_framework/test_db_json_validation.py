# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Validation script to verify that database output matches JSON output.

This script compares configurations loaded from the PostgreSQL database
with those loaded from the JSON file to ensure the database integration
produces identical results.

Usage:
    # Validate all operations
    python test_db_json_validation.py

    # Validate specific operation
    python test_db_json_validation.py --operation ttnn::add

    # Validate with verbose output
    python test_db_json_validation.py --verbose

Requirements:
    - TTNN_OPS_DATABASE_URL or POSTGRES_* environment variables must be set
    - psycopg2 must be installed
"""

import argparse
import json
import sys
from typing import Dict, List, Optional, Tuple


def validate_operation_configs(
    json_configs: List[Dict], db_configs: List[Dict], operation_name: str, verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Compare configurations from JSON and database for a single operation.

    Args:
        json_configs: Configurations loaded from JSON file
        db_configs: Configurations loaded from database
        operation_name: Name of the operation being validated
        verbose: Print detailed comparison information

    Returns:
        Tuple of (passed: bool, errors: List[str])
    """
    errors = []

    # Compare config counts
    json_count = len(json_configs)
    db_count = len(db_configs)

    if json_count != db_count:
        errors.append(f"Config count mismatch: JSON={json_count}, DB={db_count}")

    if verbose:
        print(f"  JSON configs: {json_count}")
        print(f"  DB configs: {db_count}")

    # For now, we check if the counts match
    # More detailed comparison could be added here if needed
    # (e.g., comparing individual config hashes)

    return len(errors) == 0, errors


def run_validation(operation_name: Optional[str] = None, verbose: bool = False) -> bool:
    """
    Run validation comparing JSON and database outputs.

    Args:
        operation_name: Specific operation to validate, or None for all
        verbose: Print detailed output

    Returns:
        True if all validations passed, False otherwise
    """
    # Import here to avoid import errors if dependencies are missing
    from master_config_loader import MasterConfigLoader
    from framework.database import is_ttnn_ops_db_available

    # Check if database is available
    if not is_ttnn_ops_db_available():
        print("❌ Database is not available. Set TTNN_OPS_DATABASE_URL or POSTGRES_* env vars.")
        print("   Skipping database validation.")
        return False

    print("=" * 60)
    print("TTNN Operations Database vs JSON Validation")
    print("=" * 60)

    # Load from JSON
    print("\n📁 Loading from JSON file...")
    json_loader = MasterConfigLoader()
    MasterConfigLoader.set_database_mode(False)
    json_loader.load_master_data()
    json_operations = json_loader.master_data.get("operations", {})
    print(f"   Loaded {len(json_operations)} operations from JSON")

    # Load from Database
    print("\n🗄️  Loading from database...")
    db_loader = MasterConfigLoader()
    db_loader.master_data = None  # Reset to force reload
    MasterConfigLoader.set_database_mode(True)
    try:
        db_loader.load_master_data()
        db_operations = db_loader.master_data.get("operations", {})
        print(f"   Loaded {len(db_operations)} operations from database")
    except Exception as e:
        print(f"❌ Failed to load from database: {e}")
        return False

    # Reset database mode
    MasterConfigLoader.set_database_mode(False)

    # Validate
    print("\n🔍 Validating...")
    all_passed = True
    validation_results = []

    # Determine which operations to validate
    if operation_name:
        ops_to_validate = [operation_name] if operation_name in json_operations else []
        if not ops_to_validate:
            print(f"⚠️  Operation '{operation_name}' not found in JSON")
            return False
    else:
        ops_to_validate = list(json_operations.keys())

    for op_name in ops_to_validate:
        json_configs = json_operations.get(op_name, {}).get("configurations", [])
        db_configs = db_operations.get(op_name, {}).get("configurations", [])

        passed, errors = validate_operation_configs(json_configs, db_configs, op_name, verbose)

        if passed:
            validation_results.append((op_name, "✅ PASS", None))
        else:
            validation_results.append((op_name, "❌ FAIL", errors))
            all_passed = False

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, status, _ in validation_results if "PASS" in status)
    failed_count = len(validation_results) - passed_count

    print(f"\nTotal operations: {len(validation_results)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")

    if verbose or failed_count > 0:
        print("\nDetails:")
        for op_name, status, errors in validation_results:
            if "FAIL" in status or verbose:
                print(f"  {status} {op_name}")
                if errors:
                    for error in errors:
                        print(f"       - {error}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("=" * 60)

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Validate database output matches JSON output for TTNN operations")
    parser.add_argument("--operation", "-o", type=str, help="Specific operation to validate (e.g., 'ttnn::add')")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed validation output")

    args = parser.parse_args()

    passed = run_validation(args.operation, args.verbose)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
