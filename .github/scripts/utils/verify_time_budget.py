import yaml
import json
import sys
import os
from collections import defaultdict


def verify_timeouts(tests_file, time_budget_file):
    """
    Verifies that the SUM of all test timeouts for each SKU in tests_file
    is within the total time budget defined in time_budget_file.
    """
    # These could be passed in from the workflow
    TEAM = "ops"
    WORKFLOW = "sanity"

    print(f"Loading time budgets from: {time_budget_file}")
    with open(time_budget_file, "r") as f:
        budgets = yaml.safe_load(f)

    print(f"Loading tests from: {tests_file}")
    with open(tests_file, "r") as f:
        tests = yaml.safe_load(f)

    errors_found = False

    # Navigate to the relevant part of the config
    try:
        workflow_budgets = budgets[TEAM][WORKFLOW]
    except KeyError:
        print(f"Error: Could not find time budgets for team '{TEAM}' and workflow '{WORKFLOW}' in {time_budget_file}.")
        sys.exit(1)

    # --- Part 1: Validate test file format and sum timeouts per SKU ---
    print("\n--- Summing Test Timeouts per SKU ---")

    sku_timeout_totals = defaultdict(int)

    for test in tests:
        test_name = test.get("name", "Unnamed Test")

        # Validate that mandatory keys exist
        if "timeout" not in test:
            print(f"  [ERROR] Validation FAILED! Test '{test_name}' is missing the mandatory 'timeout' key.")
            errors_found = True
            continue
        if "sku" not in test:
            print(f"  [ERROR] Validation FAILED! Test '{test_name}' is missing the mandatory 'sku' key.")
            errors_found = True
            continue

        test_timeout = test["timeout"]
        test_sku = test["sku"]

        # Add this test's timeout to the total for its SKU
        sku_timeout_totals[test_sku] += test_timeout
        print(f"  Test '{test_name}' (SKU: {test_sku}) adds {test_timeout} min.")

    # If there were any validation errors (missing keys), fail now.
    if errors_found:
        print(f"\nMissing keys in {tests_file}. Please fix the entries above.")
        sys.exit(1)

    # --- Part 2: Verify Total Time Budget for each SKU ---
    print("\n--- Verifying Total Time Budgets ---")

    for sku, total_time_requested in sku_timeout_totals.items():
        try:
            # Directly access the total budget for the SKU
            total_time_budget = workflow_budgets["skus"][sku]["time_budget"]

            print(
                f"Checking total for SKU '{sku}': Requested = {total_time_requested} min, Budget = {total_time_budget} min"
            )

            if total_time_requested > total_time_budget:
                print(
                    f"  [ERROR] Total Time Budget FAILED for SKU '{sku}'! The sum of all test timeouts ({total_time_requested} min) exceeds the allocated budget of {total_time_budget} min."
                )
                errors_found = True
            else:
                print(f"  [OK] Total time for SKU '{sku}' is within the budget.")

        except (KeyError, TypeError):
            print(
                f"  [ERROR] Configuration FAILED! Could not find a 'time_budget' key for SKU '{sku}' in {time_budget_file}."
            )
            errors_found = True
            continue

    # --- Final Verdict ---
    if errors_found:
        print("\nVerification failed.")
        sys.exit(1)
    else:
        print("\nVerification successful.")
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify_time_budget.py <path_to_tests.yaml> <path_to_machine_time_allocation.yaml>")
        sys.exit(1)

    verify_timeouts(sys.argv[1], sys.argv[2])
