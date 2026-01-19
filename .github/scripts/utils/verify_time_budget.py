import yaml
import sys
import os
from collections import defaultdict


def verify_timeouts(tests_file, time_budget_file, workflow_name):
    """
    Verifies that the SUM of all test timeouts for each (Team, SKU) pair in tests_file
    is within the total time budget defined in time_budget_file for the given workflow.
    """
    print(f"Loading time budgets from: {time_budget_file}")
    with open(time_budget_file, "r") as f:
        budgets = yaml.safe_load(f)

    print(f"Loading tests from: {tests_file}")
    with open(tests_file, "r") as f:
        tests = yaml.safe_load(f)

    errors_found = False

    # --- Part 1: Validate test file format and sum timeouts per (Team, SKU) pair ---
    print("\n--- Summing Test Timeouts per Team and SKU ---")

    # The key will now be a tuple: (team, sku)
    # e.g., {('ops', 'N300'): 20, ('models', 'N300'): 15}
    budget_totals = defaultdict(int)

    for test in tests:
        test_name = test.get("name", "Unnamed Test")

        # Validate that all mandatory keys exist for this test
        required_keys = ["timeout", "sku", "team"]
        missing_keys = [key for key in required_keys if key not in test]
        if missing_keys:
            print(
                f"  [ERROR] Validation FAILED! Test '{test_name}' is missing mandatory keys: {', '.join(missing_keys)}."
            )
            errors_found = True
            continue  # Skip this invalid test

        test_timeout = test["timeout"]
        test_sku = test["sku"]
        test_team = test["team"]

        # Use a tuple (team, sku) as the key for summation
        budget_key = (test_team, test_sku)
        budget_totals[budget_key] += test_timeout
        print(f"  Test '{test_name}' (Team: {test_team}, SKU: {test_sku}) adds {test_timeout} min.")

    if errors_found:
        print(f"\nMissing keys in {tests_file}. Please fix the entries above.")
        sys.exit(1)

    # --- Part 2: Verify Total Time Budget for each (Team, SKU) pair ---
    print("\n--- Verifying Total Time Budgets ---")

    for (team, sku), total_time_requested in budget_totals.items():
        try:
            # Navigate the budgets config using team, workflow, and sku
            total_time_budget = budgets[team][workflow_name][sku]

            print(
                f"Checking total for Team '{team}', SKU '{sku}': Requested = {total_time_requested} min, Budget = {total_time_budget} min"
            )

            if total_time_requested > total_time_budget:
                print(
                    f"  [ERROR] Total Time Budget FAILED for Team '{team}', SKU '{sku}'! "
                    f"The sum of test timeouts ({total_time_requested} min) exceeds the allocated budget of {total_time_budget} min."
                )
                errors_found = True
            else:
                print(f"  [OK] Total time for Team '{team}', SKU '{sku}' is within the budget.")

        except (KeyError, TypeError):
            print(
                f"  [ERROR] Configuration FAILED! Could not find a 'time_budget' for Team '{team}', Workflow '{workflow_name}', SKU '{sku}' in {time_budget_file}."
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
    if len(sys.argv) != 4:
        print("Usage: python verify_time_budget.py <path_to_tests.yaml> <path_to_time_budget.yaml> <workflow_name>")
        sys.exit(1)

    verify_timeouts(sys.argv[1], sys.argv[2], sys.argv[3])
